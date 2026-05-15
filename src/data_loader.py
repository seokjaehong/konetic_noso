"""
데이터 수집 및 로딩 모듈
- koenergy.kr 제공 XLS/CSV 데이터 로딩 (every/ 폴더)
- 공공데이터포털 API (B551893) 시간별 화력발전 실적 수집
- Open-Meteo API (Japan MSM) 기상 데이터 수집
- 에어코리아 API 대기 데이터 수집
"""

import os
import pandas as pd
import numpy as np
import requests
import requests_cache
from retry_requests import retry
import openmeteo_requests
from pathlib import Path
from datetime import datetime, date
from urllib.parse import unquote
import time

# ── 경로 설정 ──────────────────────────────────────────────────
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── API 키 로딩 ─────────────────────────────────────────────────
def _get_api_key() -> str:
    """환경변수 또는 .env 파일에서 공공데이터포털 API 키 로딩"""
    key = os.environ.get("PUBLIC_DATA_API_KEY", "")
    if not key:
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("PUBLIC_DATA_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    break
    return unquote(key)  # URL 인코딩된 키 디코딩


# ══════════════════════════════════════════════════════════════
# 1. koenergy.kr 데이터 로딩
#    (수동 다운로드 후 data/raw/ 에 저장된 파일을 읽어들임)
# ══════════════════════════════════════════════════════════════

def load_emission_data(filepath: str | None = None) -> pd.DataFrame:
    """
    대기오염물질 배출실적 로딩 (SOx, NOx, 먼지)
    koenergy.kr → 환경/안전 → 대기오염물질 배출실적 에서 다운로드
    기대 컬럼: 날짜, SOx(kg), NOx(kg), 먼지(kg)
    """
    path = Path(filepath) if filepath else RAW_DIR / "emission.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    # 날짜 파싱 (YYYY-MM-DD 또는 YYYY.MM.DD 등)
    df["date"] = pd.to_datetime(df.iloc[:, 0], infer_datetime_format=True)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_generation_data(filepath: str | None = None) -> pd.DataFrame:
    """
    발전 실적 데이터 로딩 (발전량, 열효율, 이용률)
    koenergy.kr → 경영정보 → 발전실적 에서 다운로드
    기대 컬럼: 날짜, 발전량(MWh), 열효율(%), 이용률(%), 용량(MW)
    """
    path = Path(filepath) if filepath else RAW_DIR / "generation.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["date"] = pd.to_datetime(df.iloc[:, 0], infer_datetime_format=True)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_fuel_data(filepath: str | None = None) -> pd.DataFrame:
    """
    연료 소비 실적 로딩 (유연탄, LNG 사용량)
    koenergy.kr → 경영정보 → 연료사용실적 에서 다운로드
    기대 컬럼: 날짜, 유연탄(ton), LNG(ton)
    """
    path = Path(filepath) if filepath else RAW_DIR / "fuel.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["date"] = pd.to_datetime(df.iloc[:, 0], infer_datetime_format=True)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_renewable_data(filepath: str | None = None) -> pd.DataFrame:
    """
    신재생 발전 실적 로딩 (태양광, 풍력 - 시간 단위)
    기대 컬럼: 날짜시간, 태양광(MWh), 풍력(MWh)
    시간→일 단위 집계 후 반환
    """
    path = Path(filepath) if filepath else RAW_DIR / "renewable.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["datetime"] = pd.to_datetime(df.iloc[:, 0], infer_datetime_format=True)
    df["date"] = df["datetime"].dt.normalize()

    # 시간 → 일 단위 집계
    # 발전량은 합산, 비율/효율 등은 평균
    agg_cols = {col: "sum" for col in df.select_dtypes("number").columns}
    df_daily = df.groupby("date").agg(agg_cols).reset_index()
    df_daily.columns.name = None
    return df_daily


def load_ambient_air_data(filepath: str | None = None) -> pd.DataFrame:
    """
    발전소 주변 대기농도 로딩 (NO2, PM10 등)
    기대 컬럼: 날짜, NO2(ppm), PM10(㎍/m³), SO2(ppm)
    """
    path = Path(filepath) if filepath else RAW_DIR / "ambient_air.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["date"] = pd.to_datetime(df.iloc[:, 0], infer_datetime_format=True)
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════
# 2. 공공데이터포털 API — 시간별 화력발전 실적 (B551893)
#    https://apis.data.go.kr/B551893/fire-power-by-hour/list
#    필수 파라미터: startD, endD (YYYYMMDD)
# ══════════════════════════════════════════════════════════════

# 발전소 코드 매핑 (API 응답의 ippt 코드)
PLANT_CODES = {
    "영동":   ["8450", "8452"],
    "삼천포": ["85C3", "85C4", "85D5", "85D6"],
    "여수":   ["8491", "8492"],
    "영흥":   ["8641", "8642", "8643", "8644", "8645", "8646"],  # 전 6호기
    "분당":   ["8731", "8732", "8733", "8734", "8735", "8736", "8737", "8738", "873A", "873B"],
}


def _fetch_thermal_one_day(key: str, date_str: str, target_codes: set) -> list:
    """단일 날짜의 화력발전 실적 수집 (내부 함수)
    공식 파라미터: serviceKey, page, size, startD, endD
    """
    base_url = "https://apis.data.go.kr/B551893/fire-power-by-hour/list"
    params = {
        "serviceKey": key,
        "startD": date_str,
        "endD": date_str,
        "page": 1,
        "size": 50,      # 하루 최대 유닛 수: 영동2+삼천포12+여수4+영흥2 = 20개
        "dataType": "JSON",
    }
    resp = requests.get(base_url, params=params, timeout=30)
    if resp.status_code != 200:
        return []
    body = resp.json().get("reponse", {})
    if body.get("header", {}).get("resultCode", "") != "00":
        return []
    content = body.get("body", {}).get("content", []) or []
    return [item for item in content if item.get("ippt", "") in target_codes]


def fetch_thermal_generation(
    start_date: str,
    end_date: str,
    plant_names: list[str] | None = None,
    api_key: str | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    """
    공공데이터포털 API로 시간별 화력발전 실적 수집 후 일 단위 집계 반환

    ※ API 특성상 날짜별로 개별 호출함 (startD=endD=해당일)

    Parameters
    ----------
    start_date  : "YYYY-MM-DD" 또는 "YYYYMMDD"
    end_date    : "YYYY-MM-DD" 또는 "YYYYMMDD"
    plant_names : 수집할 발전소명 목록 (None이면 전체)
    api_key     : 공공데이터포털 인증키 (None이면 .env에서 로딩)
    cache       : True면 이미 저장된 CSV 파일 재사용

    Returns
    -------
    pd.DataFrame: 일 단위 집계 (발전소별 총발전량, 시간대별 발전량)
    """
    key = api_key or _get_api_key()
    if not key:
        raise ValueError("PUBLIC_DATA_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

    start_d = pd.Timestamp(start_date).strftime("%Y%m%d")
    end_d   = pd.Timestamp(end_date).strftime("%Y%m%d")

    # 캐시 확인
    save_path = RAW_DIR / f"thermal_gen_{start_d}_{end_d}.csv"
    if cache and save_path.exists():
        print(f"[CACHE] 화력발전 실적 로딩: {save_path}")
        df = pd.read_csv(save_path, parse_dates=["date"])
        return df

    # 필터링할 발전소 코드
    target_codes: set[str] = set()
    if plant_names:
        for name in plant_names:
            target_codes.update(PLANT_CODES.get(name, []))
    else:
        for codes in PLANT_CODES.values():
            target_codes.update(codes)

    # 발전소명 역매핑
    code_to_name = {c: name for name, codes in PLANT_CODES.items() for c in codes}

    # 날짜별 개별 호출 (API 특성상 하루씩만 반환)
    date_range = pd.date_range(start=start_d, end=end_d, freq="D")
    all_records = []
    print(f"[API] 화력발전 실적 수집: {start_d} ~ {end_d} ({len(date_range)}일)")

    for i, dt in enumerate(date_range):
        ds = dt.strftime("%Y%m%d")
        records = _fetch_thermal_one_day(key, ds, target_codes)
        all_records.extend(records)
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(date_range)}일 완료...")
        time.sleep(0.1)  # API 요청 간격

    if not all_records:
        print("[WARN] 화력발전 데이터 없음")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["dgenYmd"])
    df["plant_name"] = df["ippt"].map(code_to_name)

    # 시간별 발전량 컬럼 수치 변환
    hour_cols = [f"qhorGen{i:02d}" for i in range(1, 25)]
    for col in hour_cols + ["qsum", "qavg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 일 단위 집계 (같은 발전소 여러 호기 합산)
    agg_dict: dict = {"qsum": "sum", "qavg": "mean"}
    for col in hour_cols:
        if col in df.columns:
            agg_dict[col] = "sum"

    df_daily = (
        df.groupby(["date", "plant_name"])
        .agg({k: v for k, v in agg_dict.items() if k in df.columns})
        .reset_index()
    )
    df_daily.rename(columns={"qsum": "daily_gen_kwh", "qavg": "avg_gen_kwh"}, inplace=True)
    df_daily["daily_gen_mwh"] = df_daily["daily_gen_kwh"] / 1000

    df_daily.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 화력발전 실적 저장: {save_path}  ({len(df_daily)} rows)")
    return df_daily


def fetch_thermal_generation_chunked(
    start_date: str,
    end_date: str,
    plant_names: list[str] | None = None,
    chunk_months: int = 3,
) -> pd.DataFrame:
    """
    장기간 데이터 수집 시 분기별로 나눠서 저장 (중간 저장으로 재시도 가능)

    Parameters
    ----------
    chunk_months : 한 번에 수집할 개월 수 (기본 3개월)
    """
    date_ranges = pd.date_range(start=start_date, end=end_date, freq=f"{chunk_months}MS")
    end_ts = pd.Timestamp(end_date)

    all_dfs = []
    for i, chunk_start in enumerate(date_ranges):
        chunk_end = min(chunk_start + pd.DateOffset(months=chunk_months) - pd.Timedelta(days=1), end_ts)
        print(f"\n[청크 {i+1}/{len(date_ranges)}] {chunk_start.date()} ~ {chunk_end.date()}")
        try:
            df = fetch_thermal_generation(
                start_date=chunk_start.strftime("%Y%m%d"),
                end_date=chunk_end.strftime("%Y%m%d"),
                plant_names=plant_names,
                cache=True,
            )
            if not df.empty:
                all_dfs.append(df)
        except Exception as e:
            print(f"  [WARN] 수집 실패: {e}")

    if not all_dfs:
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True).drop_duplicates(subset=["date", "plant_name"])
    result = result.sort_values(["date", "plant_name"]).reset_index(drop=True)

    # 전체 기간 통합 파일 저장
    sd = pd.Timestamp(start_date).strftime("%Y%m%d")
    ed = pd.Timestamp(end_date).strftime("%Y%m%d")
    full_path = RAW_DIR / f"thermal_gen_{sd}_{ed}_full.csv"
    result.to_csv(full_path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] 전체 기간 저장: {full_path}  ({len(result)} rows)")
    return result


# ══════════════════════════════════════════════════════════════
# 3. Open-Meteo API — Japan MSM 고해상도 기상 데이터
#    무료 / API 키 불필요 / 5km 격자 고해상도
# ══════════════════════════════════════════════════════════════

# 한국남동발전 주요 발전소 좌표 (영동, 삼천포, 여수, 분당, 신보령)
PLANT_COORDS = {
    "영동":   {"lat": 37.1756, "lon": 129.3610},
    "삼천포": {"lat": 35.0019, "lon": 128.0644},
    "여수":   {"lat": 34.7604, "lon": 127.7442},
    "분당":   {"lat": 37.3595, "lon": 127.1054},
    "신보령": {"lat": 36.3567, "lon": 126.5631},
    "영흥":   {"lat": 37.2457, "lon": 126.4656},
}


def fetch_weather_openmeteo(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    plant_name: str = "plant",
) -> pd.DataFrame:
    """
    Open-Meteo API에서 Japan MSM 모델 기반 기상 데이터 수집

    Parameters
    ----------
    lat, lon    : 발전소 위도/경도
    start_date  : "YYYY-MM-DD"
    end_date    : "YYYY-MM-DD"
    plant_name  : 저장 파일명 구분자

    Returns
    -------
    pd.DataFrame: 일 단위 기상 데이터
    """
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    om = openmeteo_requests.Client(session=retry_session)

    # Japan MSM: 5km 고해상도, 한반도 커버
    # 일 단위 집계 변수
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "relative_humidity_2m_max",
            "relative_humidity_2m_min",
            "relative_humidity_2m_mean",
            "wind_speed_10m_max",
            "wind_speed_10m_mean",
            "wind_direction_10m_dominant",
            "precipitation_sum",
            "surface_pressure_mean",
            "shortwave_radiation_sum",   # 태양광 발전 상관 분석용
        ],
        "models": "jma_seamless",        # Japan MSM 모델
        "timezone": "Asia/Seoul",
    }

    responses = om.weather_api("https://historical-forecast-api.open-meteo.com/v1/forecast", params=params)
    r = responses[0]
    daily = r.Daily()

    df = pd.DataFrame({
        "date":         pd.date_range(
            start=pd.Timestamp(daily.Time(), unit="s", tz="Asia/Seoul"),
            end=pd.Timestamp(daily.TimeEnd(), unit="s", tz="Asia/Seoul"),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left",
        ).normalize(),
        "temp_max":     daily.Variables(0).ValuesAsNumpy(),
        "temp_min":     daily.Variables(1).ValuesAsNumpy(),
        "temp_mean":    daily.Variables(2).ValuesAsNumpy(),
        "humidity_max": daily.Variables(3).ValuesAsNumpy(),
        "humidity_min": daily.Variables(4).ValuesAsNumpy(),
        "humidity_mean":daily.Variables(5).ValuesAsNumpy(),
        "wind_speed_max":    daily.Variables(6).ValuesAsNumpy(),
        "wind_speed_mean":   daily.Variables(7).ValuesAsNumpy(),
        "wind_dir_dominant": daily.Variables(8).ValuesAsNumpy(),
        "precipitation":     daily.Variables(9).ValuesAsNumpy(),
        "pressure_mean":     daily.Variables(10).ValuesAsNumpy(),
        "solar_radiation":   daily.Variables(11).ValuesAsNumpy(),
    })

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()

    # 풍향 sin/cos 인코딩 (순환 변수 처리)
    wind_rad = np.deg2rad(df["wind_dir_dominant"])
    df["wind_sin"] = np.sin(wind_rad)
    df["wind_cos"] = np.cos(wind_rad)

    # 대기정체 지수: 풍속 낮음 + 습도 높음 → 오염 집중 위험
    df["stagnation_idx"] = df["humidity_mean"] / (df["wind_speed_mean"].clip(lower=0.1))

    save_path = RAW_DIR / f"weather_{plant_name}_{start_date}_{end_date}.csv"
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 기상 데이터 저장: {save_path}")
    return df


def fetch_all_plants_weather(start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
    """모든 발전소 기상 데이터 수집"""
    result = {}
    for name, coords in PLANT_COORDS.items():
        print(f"  → {name} 발전소 기상 데이터 수집 중...")
        df = fetch_weather_openmeteo(
            lat=coords["lat"],
            lon=coords["lon"],
            start_date=start_date,
            end_date=end_date,
            plant_name=name,
        )
        result[name] = df
        time.sleep(0.5)   # API 요청 간격
    return result


# ══════════════════════════════════════════════════════════════
# 3. 에어코리아 API — 주변 배경 대기 농도
#    공공데이터포털 API 키 필요: https://www.data.go.kr
# ══════════════════════════════════════════════════════════════

def fetch_airkorea(
    station_name: str,
    start_date: str,
    end_date: str,
    api_key: str,
) -> pd.DataFrame:
    """
    에어코리아 시간별 측정 데이터 수집 후 일 단위 평균 반환

    Parameters
    ----------
    station_name : 측정소명 (예: "삼천포")
    api_key      : 공공데이터포털 인증키 (URL 인코딩된 키)
    """
    base_url = "https://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMsrstnAcctoRltmMesureDnsty"
    records = []
    page = 1

    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    current = start_dt

    while current <= end_dt:
        params = {
            "serviceKey": api_key,
            "returnType": "json",
            "numOfRows": 24,
            "pageNo": page,
            "stationName": station_name,
            "dataTerm": "DAILY",
            "ver": "1.0",
        }
        resp = requests.get(base_url, params=params, timeout=30)
        if resp.status_code != 200:
            print(f"[WARN] 에어코리아 API 오류: {resp.status_code}")
            break

        items = resp.json().get("response", {}).get("body", {}).get("items", [])
        if not items:
            break

        records.extend(items)
        current += pd.Timedelta(days=1)
        page += 1
        time.sleep(0.3)

    if not records:
        print("[WARN] 에어코리아 데이터 없음")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # 측정 항목: SO2, CO, O3, NO2, PM10, PM25
    numeric_cols = ["so2Value", "coValue", "o3Value", "no2Value", "pm10Value", "pm25Value"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "dataTime" in df.columns:
        df["date"] = pd.to_datetime(df["dataTime"]).dt.normalize()
        df = df.groupby("date")[numeric_cols].mean().reset_index()

    save_path = RAW_DIR / f"airkorea_{station_name}.csv"
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 에어코리아 데이터 저장: {save_path}")
    return df


# ══════════════════════════════════════════════════════════════
# 4. 통합 데이터셋 병합
# ══════════════════════════════════════════════════════════════

def build_master_dataset(
    emission_df: pd.DataFrame,
    generation_df: pd.DataFrame,
    fuel_df: pd.DataFrame,
    renewable_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    ambient_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    모든 데이터를 날짜 기준으로 병합하여 마스터 데이터셋 생성

    리샘플링 논거:
    - 신재생 시간 단위 → 일 합산 (발전량 성격이므로 합산이 적절)
    - 월 단위 데이터가 있을 경우 일 단위 보간 필요 (별도 처리)
    """
    # 기준: emission 날짜 범위
    base = emission_df[["date"]].copy()

    dfs = [base, generation_df, fuel_df, renewable_df, weather_df]
    if ambient_df is not None:
        dfs.append(ambient_df)

    master = base.copy()
    for df in dfs[1:]:
        if "date" in df.columns:
            master = master.merge(df, on="date", how="left", suffixes=("", "_dup"))
            # 중복 컬럼 제거
            dup_cols = [c for c in master.columns if c.endswith("_dup")]
            master.drop(columns=dup_cols, inplace=True)

    master = master.sort_values("date").reset_index(drop=True)

    # 연료 믹스 비율 파생 변수
    if "유연탄" in master.columns and "LNG" in master.columns:
        total_fuel = master["유연탄"] + master["LNG"]
        master["coal_ratio"] = master["유연탄"] / total_fuel.clip(lower=1e-6)
        master["lng_ratio"] = master["LNG"] / total_fuel.clip(lower=1e-6)

    # 계절 더미
    master["month"] = master["date"].dt.month
    master["season"] = master["month"].map({
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "autumn", 10: "autumn", 11: "autumn",
    })

    save_path = PROCESSED_DIR / "master_dataset.csv"
    master.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 마스터 데이터셋 저장: {save_path}  ({len(master)} rows)")
    return master


if __name__ == "__main__":
    # 사용 예시
    print("=== 기상 데이터 수집 (Open-Meteo Japan MSM) ===")
    # 삼천포 발전소 기준 3년치 수집
    weather = fetch_weather_openmeteo(
        lat=PLANT_COORDS["삼천포"]["lat"],
        lon=PLANT_COORDS["삼천포"]["lon"],
        start_date="2022-01-01",
        end_date="2024-12-31",
        plant_name="삼천포",
    )
    print(weather.head())
    print(f"\n컬럼: {weather.columns.tolist()}")
