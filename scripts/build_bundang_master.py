"""
분당화력 마스터 데이터셋 빌드 스크립트
- 배출 (NOx, kg/day): 농도×유량×86400/1e6
- 발전량 (MWh/day): 화력 xlsx 집계
- 기상: 기상정보 xls 분당 데이터
- 이용률, 열효율: 발전실적 xls 월별 → 일별 보간
- 연료: 연료소비 xls 월별 LNG → 일별 보간
결과: data/processed/bundang_master.csv
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
import glob
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

BUNDANG_CAPACITY_MW = 900  # CG1~8 (77.76×8=622) + CS1(185) + CS2(115) ≈ 900MW
PLANT = "분당"

# ─────────────────────────────────────────────────
# 1. 배출 데이터 (일별 NOx kg/day)
# ─────────────────────────────────────────────────
print("=== 1. 배출 데이터 로딩 ===")
df_e = pd.read_excel(RAW_DIR / "한국남동발전_대기오염물질배출농도(일평균).xls", header=None)
df_e.columns = df_e.iloc[0]
df_e = df_e.iloc[1:].reset_index(drop=True)

df_e_bd = df_e[df_e["사업소"] == PLANT].copy()
df_e_bd["date"] = pd.to_datetime(df_e_bd["일자"].astype(str), format="%Y%m%d", errors="coerce")
df_e_bd = df_e_bd.dropna(subset=["date"])

for col in ["SOX", "NOX", "먼지", "유량"]:
    df_e_bd[col] = pd.to_numeric(df_e_bd[col], errors="coerce")

# 질량 배출량 (kg/day per 호기) = 농도(mg/Sm³) × 유량(Sm³/min) × 1440 min/day / 1e6
# 유량 단위: Sm³/min (원천 데이터 기준)
# 분당 SOx ≈ 0 (LNG), 먼지도 미미 → NOx 중심
df_e_bd["NOx_kg"] = df_e_bd["NOX"] * df_e_bd["유량"] * 1440 / 1e6
df_e_bd["SOx_kg"] = df_e_bd["SOX"].fillna(0) * df_e_bd["유량"] * 1440 / 1e6
df_e_bd["dust_kg"] = df_e_bd["먼지"].fillna(0) * df_e_bd["유량"] * 1440 / 1e6

em_daily = df_e_bd.groupby("date").agg(
    NOx=("NOx_kg", "sum"),
    SOx=("SOx_kg", "sum"),
    dust=("dust_kg", "sum"),
    NOx_conc=("NOX", "mean"),
    n_units=("호기", "count"),
).reset_index()
em_daily.rename(columns={"dust": "먼지"}, inplace=True)

print(f"  배출 일수: {len(em_daily)}, 기간: {em_daily['date'].min().date()} ~ {em_daily['date'].max().date()}")
print(f"  NOx 평균: {em_daily['NOx'].mean():.0f} kg/day")

# ─────────────────────────────────────────────────
# 2. 발전량 (일별 MWh) — 화력 xlsx 집계
# ─────────────────────────────────────────────────
print("\n=== 2. 발전량 로딩 ===")
xlsx_files = sorted(glob.glob(str(RAW_DIR / "every" / "화력_*.xlsx")))
# 최신 중복 파일 제거 (화력_202604.xlsx, 화력_202604_after_search.xlsx)
xlsx_files = [f for f in xlsx_files if "after_search" not in f]

dfs_gen = []
for f in xlsx_files:
    df = pd.read_excel(f)
    dfs_gen.append(df)

df_gen_raw = pd.concat(dfs_gen, ignore_index=True)
df_gen_raw["date"] = pd.to_datetime(df_gen_raw["일자"]).dt.normalize()

# 총량(KW) 컬럼명과 달리 실제 단위는 KWh/day (24시간 발전량 합산값)
# → MWh/day 변환: / 1000
df_gen_raw["daily_mwh_unit"] = df_gen_raw["총량(KW)"] / 1000

gen_daily = df_gen_raw.groupby("date").agg(
    api_gen_mwh=("daily_mwh_unit", "sum"),
).reset_index()

print(f"  발전량 일수: {len(gen_daily)}, 기간: {gen_daily['date'].min().date()} ~ {gen_daily['date'].max().date()}")
print(f"  api_gen_mwh 평균: {gen_daily['api_gen_mwh'].mean():.0f} MWh/day")

# 이용률 계산 (실발전량 / 설비용량)
gen_daily["utilization"] = (gen_daily["api_gen_mwh"] / (BUNDANG_CAPACITY_MW * 24) * 100).clip(0, 100)

# ─────────────────────────────────────────────────
# 3. 기상 데이터
# ─────────────────────────────────────────────────
print("\n=== 3. 기상 데이터 로딩 ===")
df_w = pd.read_excel(RAW_DIR / "한국남동발전_기상정보(일평균).xls", header=None)
df_w.columns = df_w.iloc[0]
df_w = df_w.iloc[1:].reset_index(drop=True)

df_w_bd = df_w[df_w["사업소"] == PLANT].copy()
df_w_bd["date"] = pd.to_datetime(df_w_bd["일자"].astype(str), format="%Y%m%d", errors="coerce")
df_w_bd = df_w_bd.dropna(subset=["date"])

for col in ["온도", "습도", "풍향", "풍속", "강수량", "기압"]:
    df_w_bd[col] = pd.to_numeric(df_w_bd[col], errors="coerce")

weather = df_w_bd.groupby("date").agg(
    temp_mean=("온도", "mean"),
    humidity_mean=("습도", "mean"),
    wind_dir=("풍향", "mean"),
    wind_speed_mean=("풍속", "mean"),
    precipitation=("강수량", "mean"),
    pressure_mean=("기압", "mean"),
).reset_index()

# 파생 변수
wind_rad = np.deg2rad(weather["wind_dir"].fillna(0))
weather["wind_sin"] = np.sin(wind_rad)
weather["wind_cos"] = np.cos(wind_rad)
weather["stagnation_idx"] = weather["humidity_mean"] / weather["wind_speed_mean"].clip(lower=0.1)

# Open-Meteo 기상 (분당 좌표) 데이터가 있으면 병합
openmeteo_path = RAW_DIR / "weather_분당_2020-07-01_2026-04-30.csv"
if openmeteo_path.exists():
    df_om = pd.read_csv(openmeteo_path, parse_dates=["date"])
    om_cols = ["date", "solar_radiation", "wind_speed_max"]
    om_available = [c for c in om_cols if c in df_om.columns]
    weather = weather.merge(df_om[om_available], on="date", how="left")
    print(f"  Open-Meteo 병합: {om_cols}")
else:
    weather["solar_radiation"] = np.nan
    weather["wind_speed_max"] = weather["wind_speed_mean"] * 1.5  # 근사값

print(f"  기상 일수: {len(weather)}, 기간: {weather['date'].min().date()} ~ {weather['date'].max().date()}")

# ─────────────────────────────────────────────────
# 4. 열효율 (발전실적 월별 → 일별 보간)
# ─────────────────────────────────────────────────
print("\n=== 4. 열효율/발전실적 로딩 ===")
df_perf = pd.read_excel(RAW_DIR / "한국남동발전_발전실적.xls")
df_bd_perf = df_perf[df_perf["사업소"] == "분당"].copy()
df_bd_perf["ym"] = pd.to_datetime(df_bd_perf["일자"].astype(str), format="%Y%m")

# CG 호기만 열효율 있음 (CS는 증기터빈, 열효율=0으로 표기됨 → 제외)
cg_mask = df_bd_perf["호기"].str.startswith("CG")
df_cg = df_bd_perf[cg_mask].copy()
df_cg["열효율(%)"] = pd.to_numeric(df_cg["열효율(%)"], errors="coerce")
df_cg = df_cg[df_cg["열효율(%)"] > 0]  # 0인 것 제외

monthly_eff = df_cg.groupby("ym")["열효율(%)"].mean().reset_index()
monthly_eff.rename(columns={"ym": "date"}, inplace=True)
# 분당 발전실적 월별 이용률 (전체 호기 합산)
monthly_util_perf = df_bd_perf.groupby("ym")["이용률(%)"].mean().reset_index()
monthly_util_perf.rename(columns={"ym": "date"}, inplace=True)

print(f"  열효율 월별 데이터: {len(monthly_eff)}건")
print(f"  열효율 평균: {monthly_eff['열효율(%)'].mean():.1f}%")

# ─────────────────────────────────────────────────
# 5. 연료 소비 (월별 LNG ton → 일별 보간)
# ─────────────────────────────────────────────────
print("\n=== 5. 연료 소비 로딩 ===")
df_fuel = pd.read_excel(RAW_DIR / "한국남동발전_연료소비실적.xls")
df_bd_fuel = df_fuel[df_fuel["사업소"] == "분당화력"].copy()
df_bd_fuel["date"] = pd.to_datetime(df_bd_fuel["일자"].astype(str), format="%Y%m")
df_bd_fuel["LNG"] = pd.to_numeric(df_bd_fuel["LNG"], errors="coerce")
monthly_fuel = df_bd_fuel.groupby("date")["LNG"].sum().reset_index()
print(f"  LNG 연료 월별: {len(monthly_fuel)}건, 평균 {monthly_fuel['LNG'].mean():.0f} ton/month")

# ─────────────────────────────────────────────────
# 6. 월별 → 일별 보간 헬퍼
# ─────────────────────────────────────────────────
def monthly_to_daily(monthly_df: pd.DataFrame, value_col: str, date_col: str = "date") -> pd.Series:
    """월별 값을 일별로 균등 배분 후 일별 시리즈 반환 (인덱스=date)"""
    result = {}
    for _, row in monthly_df.iterrows():
        month_start = row[date_col]
        days_in_month = pd.Timestamp(month_start).days_in_month
        daily_val = row[value_col] / days_in_month
        for d in range(days_in_month):
            day = month_start + pd.Timedelta(days=d)
            result[day] = daily_val
    return pd.Series(result)

# 열효율 일별 (월평균 유지)
eff_monthly_avg = monthly_eff.set_index("date")["열효율(%)"]
daily_eff = eff_monthly_avg.reindex(
    pd.date_range(eff_monthly_avg.index.min(),
                  eff_monthly_avg.index.max() + pd.offsets.MonthEnd(), freq="D"),
    method="ffill"
).reset_index()
daily_eff.columns = ["date", "heat_efficiency"]

# LNG 일별
lng_daily_series = monthly_to_daily(monthly_fuel, "LNG")
lng_daily = lng_daily_series.reset_index()
lng_daily.columns = ["date", "LNG"]

# ─────────────────────────────────────────────────
# 7. 통합 마스터 빌드
# ─────────────────────────────────────────────────
print("\n=== 6. 마스터 데이터셋 빌드 ===")

# 기준: 발전량 데이터 날짜 범위 (배출 + 기상 공통 구간)
date_range = pd.date_range(
    max(em_daily["date"].min(), gen_daily["date"].min(), weather["date"].min()),
    min(em_daily["date"].max(), gen_daily["date"].max(), weather["date"].max()),
    freq="D"
)
master = pd.DataFrame({"date": date_range})

# 순차 병합
master = master.merge(em_daily, on="date", how="left")
master = master.merge(gen_daily[["date", "api_gen_mwh", "utilization"]], on="date", how="left")
master = master.merge(weather, on="date", how="left")
master = master.merge(daily_eff, on="date", how="left")
master = master.merge(lng_daily, on="date", how="left")

# 파생 변수
master["gen_mwh_combined"] = master["api_gen_mwh"]
master["coal_ratio"] = 0.0
master["lng_ratio"] = 1.0
master["유연탄"] = 0.0
master["solar_mwh"] = 0.0
master["wind_mwh"] = 0.0
master["renewable_mwh"] = 0.0
master["renewable_ratio"] = 0.0
master["plant_type"] = "bundang_cc"   # LNG 복합화력
master["is_coal"] = 0                  # 석탄 여부 (삼천포=1, 분당=0)

# 시간 특성
master["month"] = master["date"].dt.month
master["dayofweek"] = master["date"].dt.dayofweek
master["year"] = master["date"].dt.year
master["season"] = master["month"].map({
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
})
master["seasonal_mgmt"] = master["month"].isin([12, 1, 2, 3]).astype(int)

# Lag 특성
master = master.sort_values("date").reset_index(drop=True)
for target in ["SOx", "NOx", "먼지"]:
    if target in master.columns:
        master[f"{target}_lag1"] = master[target].shift(1)
        master[f"{target}_lag7"] = master[target].shift(7)

# 결측 현황
print(f"  마스터 shape: {master.shape}")
print(f"  기간: {master['date'].min().date()} ~ {master['date'].max().date()}")
print(f"  결측치 비율 (주요 컬럼):")
key_cols = ["NOx", "api_gen_mwh", "utilization", "heat_efficiency", "LNG", "temp_mean"]
for col in key_cols:
    if col in master.columns:
        pct = master[col].isna().mean() * 100
        print(f"    {col}: {pct:.1f}%")

save_path = PROCESSED_DIR / "bundang_master.csv"
master.to_csv(save_path, index=False, encoding="utf-8-sig")
print(f"\n[OK] 분당 마스터 저장: {save_path}  ({len(master)} rows × {len(master.columns)} cols)")
