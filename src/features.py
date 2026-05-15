"""
특성 엔지니어링 모듈
- 입력 특성 생성 및 타겟 변수 정의
- 리샘플링 논거 문서화

[삼천포 발전소 특성 노트]
삼천포는 순수 유연탄 발전소로 LNG 소비가 없어 coal_ratio가 항상 1.0.
따라서 최적화 변수는 '연료 믹스' 대신 '발전량(이용률)'을 사용.
  - 대기정체 조건에서 이용률 감발 → 배출량 감축
  - 최적화 변수: gen_mwh_combined (API 실측값 우선, 없으면 월별 보간값)
"""

import pandas as pd
import numpy as np


# ── 타겟 변수 ──────────────────────────────────────────────────
TARGET_COLS = ["SOx", "NOx", "먼지"]   # 배출량 (kg/일)

# ── 특성 그룹 정의 ─────────────────────────────────────────────
WEATHER_FEATURES = [
    "temp_mean",
    "humidity_mean",
    "wind_speed_mean", "wind_speed_max",
    "wind_sin", "wind_cos",          # 풍향 순환 인코딩
    "precipitation",
    "pressure_mean",
    "solar_radiation",
    "stagnation_idx",                # 대기정체 지수 = 습도/풍속
]

GENERATION_FEATURES = [
    "gen_mwh_combined",   # 일 발전량 MWh (API 실측 우선, 없으면 월별 보간)
    "utilization",        # 이용률 % — 최적화 핵심 변수
    "heat_efficiency",    # 열효율 %
]

FUEL_FEATURES = [
    "유연탄",   # ton/일 (월→일 보간) — 발전 강도 proxy
]

RENEWABLE_FEATURES = [
    "solar_mwh",   # 삼천포 태양광 MWh
    "wind_mwh",    # 삼천포 풍력 MWh
    "renewable_ratio",
]

TIME_FEATURES = [
    "month",
    "dayofweek",
    "season_spring", "season_summer", "season_autumn",
    "seasonal_mgmt",   # 계절관리제 더미 (12~3월, 출력 제한 기간)
    "year_trend",      # (date - 2020-01-01).days / 365 — 설비개선 장기 트렌드 흡수
]

LAG_FEATURES = [
    "SOx_lag1", "NOx_lag1", "먼지_lag1",   # 전일 배출량
    "SOx_lag7", "NOx_lag7", "먼지_lag7",   # 7일 전 배출량 (주간 패턴)
    "sox_factor_ma30",   # SOx 실효 배출계수 30일 이동평균 (연료 황 함량 proxy)
]

ALL_FEATURES = (
    WEATHER_FEATURES
    + GENERATION_FEATURES
    + FUEL_FEATURES
    + RENEWABLE_FEATURES
    + TIME_FEATURES
    + LAG_FEATURES
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    마스터 데이터셋에서 모델 입력 특성 생성

    리샘플링 논거 (보고서 Ⅱ장 기재):
    - 신재생 발전량: 시간 단위 → 일 합산 (발전량은 누적 성격이므로 합산 적절)
    - 연료/발전 월 단위 → 일 단위 균등 배분 (기저부하 특성, 일별 편차 작음)
    - 기상: 일 평균/최대값 (koenergy 자체 센서 + Open-Meteo 보완)
    - gen_mwh_combined: API 실측(2022~) 우선, 이전 구간은 월별 보간값 사용
    """
    data = df.copy()

    # ── 발전량 통합 컬럼 생성 (API 실측 우선) ─────────────────
    if "api_gen_mwh" in data.columns and "gen_mwh" in data.columns:
        # NaN만 gen_mwh로 보완 (0은 LNG peaker 실제 정지일이므로 유지)
        data["gen_mwh_combined"] = data["api_gen_mwh"].fillna(data["gen_mwh"])
    elif "api_gen_mwh" in data.columns:
        data["gen_mwh_combined"] = data["api_gen_mwh"]
    elif "gen_mwh" in data.columns:
        data["gen_mwh_combined"] = data["gen_mwh"]

    # ── 시간 특성 ────────────────────────────────────────────
    data["month"] = data["date"].dt.month
    data["dayofweek"] = data["date"].dt.dayofweek
    # 장기 트렌드 (환경설비 개선 등 구조적 변화 흡수)
    data["year_trend"] = (data["date"] - pd.Timestamp("2020-01-01")).dt.days / 365.0

    # 계절 더미 (winter 기준)
    season_dummies = pd.get_dummies(data["season"], prefix="season", drop_first=False)
    for col in ["season_spring", "season_summer", "season_autumn"]:
        if col not in season_dummies.columns:
            season_dummies[col] = 0
    data = pd.concat([data, season_dummies[["season_spring", "season_summer", "season_autumn"]]], axis=1)

    # ── 계절관리제 더미 (12~3월) ──────────────────────────────
    # 근거: 미세먼지 계절관리제(2019년 도입) — 12~3월 석탄발전 상한제 시행
    # 출력 상한 80% 제한 → 이용률·배출량 패턴이 계절과 별도로 변화
    data["seasonal_mgmt"] = data["month"].isin([12, 1, 2, 3]).astype(int)

    # ── 대기정체 지수 (없으면 생성) ───────────────────────────
    if "stagnation_idx" not in data.columns:
        data["stagnation_idx"] = data["humidity_mean"] / data["wind_speed_mean"].clip(lower=0.1)

    # ── 풍향 인코딩 (없으면 생성) ─────────────────────────────
    if "wind_sin" not in data.columns and "wind_dir" in data.columns:
        wind_rad = np.deg2rad(data["wind_dir"])
        data["wind_sin"] = np.sin(wind_rad)
        data["wind_cos"] = np.cos(wind_rad)

    # ── 신재생 발전 비율 ──────────────────────────────────────
    if "renewable_ratio" not in data.columns:
        renewable_total = data.get("solar_mwh", pd.Series(0, index=data.index)).fillna(0) + \
                          data.get("wind_mwh", pd.Series(0, index=data.index)).fillna(0)
        base_gen = data.get("gen_mwh_combined", pd.Series(1, index=data.index)).clip(lower=1)
        data["renewable_ratio"] = renewable_total / (base_gen + renewable_total)
    # every/ 데이터 없던 기간(삼천포 2020-2021 등) NaN → 신재생 0으로 처리
    data["renewable_ratio"] = data["renewable_ratio"].fillna(0)

    # ── Lag 특성 (전일/7일 배출량) ───────────────────────────
    # 근거: 설비 연속 운전 패턴(warming-up), 보수 사이클, 규제 대응 연속성
    # 주의: lag 사용 시 첫 7행 결측 발생 → dropna 필요
    data = data.sort_values("date").reset_index(drop=True)
    for target in TARGET_COLS:
        if target in data.columns:
            data[f"{target}_lag1"] = data[target].shift(1)
            data[f"{target}_lag7"] = data[target].shift(7)

    # ── 연료 품질 Proxy (SOx 실효 배출계수 30일 이동평균) ───────
    # 근거: 연료도입실적에 황 함량(S%) 컬럼 없음 → 역산 proxy 사용
    #   sox_factor = SOx_kg / 유연탄_ton → 높을수록 고황탄 도입 추정
    #   30일 이동평균(shift=1)으로 lag 처리하여 타겟 누수 방지
    if "SOx" in data.columns and "유연탄" in data.columns:
        data["sox_factor"] = data["SOx"] / data["유연탄"].clip(lower=1)
        data["sox_factor_ma30"] = (
            data["sox_factor"].shift(1).rolling(window=30, min_periods=7).mean()
        )

    # ── 배출계수 파생변수 (분석/보고용, 모델 피처 미포함 — 타겟 역산이므로 누수 위험) ──
    # SOx_per_mwh, NOx_per_mwh, dust_per_mwh: kg/MWh — 설비 효율 지표
    # gen_per_coal: MWh/ton — 단위연료당 발전량 (연료 효율 지표)
    if "gen_mwh_combined" in data.columns:
        gen_base = data["gen_mwh_combined"].clip(lower=1)
        for t in TARGET_COLS:
            if t in data.columns:
                col_name = t.replace("먼지", "dust") + "_per_mwh"
                data[col_name] = (data[t] / gen_base).clip(upper=100)
    if "유연탄" in data.columns and "gen_mwh_combined" in data.columns:
        data["gen_per_coal"] = (data["gen_mwh_combined"] / data["유연탄"].clip(lower=0.1)).clip(upper=1000)

    # ── 열효율 구간 (분석용) ────────────────────────────────────
    if "heat_efficiency" in data.columns:
        data["heat_efficiency_bin"] = pd.cut(
            data["heat_efficiency"],
            bins=[0, 25, 32, 100],
            labels=["저효율(<25%)", "중효율(25~32%)", "고효율(>32%)"],
        )

    # ── 기상 구간 (분석용) ─────────────────────────────────────
    if "temp_mean" in data.columns:
        data["temp_bin"] = pd.cut(
            data["temp_mean"],
            bins=[-50, 5, 20, 60],
            labels=["저온(<5°C)", "적온(5~20°C)", "고온(>20°C)"],
        )
    if "wind_speed_mean" in data.columns:
        data["wind_bin"] = pd.cut(
            data["wind_speed_mean"],
            bins=[0, 2, 5, 100],
            labels=["약풍(<2m/s)", "중풍(2~5m/s)", "강풍(>5m/s)"],
        )

    # ── 존재하는 특성만 선택 ──────────────────────────────────
    available = [c for c in ALL_FEATURES if c in data.columns]
    analysis_cols = [c for c in [
        "SOx_per_mwh", "NOx_per_mwh", "dust_per_mwh",
        "gen_per_coal", "heat_efficiency_bin", "temp_bin", "wind_bin",
    ] if c in data.columns]
    return data[["date"] + available + [t for t in TARGET_COLS if t in data.columns] + analysis_cols]


ANALYSIS_ONLY_COLS = [
    "SOx_per_mwh", "NOx_per_mwh", "dust_per_mwh",
    "gen_per_coal", "heat_efficiency_bin", "temp_bin", "wind_bin",
]

def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """특성(X)과 타겟(y) 분리
    분석용 파생변수(ANALYSIS_ONLY_COLS)는 모델 피처에서 제외:
    - 배출계수(SOx/NOx/먼지 per MWh)는 타겟을 역산한 값 → 피처 누수
    - 구간 변수(Categorical dtype)는 mean() 호출 시 오류
    """
    exclude = set(["date"] + TARGET_COLS + ANALYSIS_ONLY_COLS)
    feature_cols = [c for c in df.columns if c not in exclude]
    target_cols = [c for c in TARGET_COLS if c in df.columns]
    X = df[feature_cols].copy()
    y = df[target_cols].copy()
    return X, y


def train_test_split_temporal(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    시계열 특성을 보존한 Train/Validation/Test 분할
    (랜덤 분할 금지 - 미래 데이터 누수 방지)
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    print(f"Train: {train['date'].min().date()} ~ {train['date'].max().date()} ({len(train)}일)")
    print(f"Val  : {val['date'].min().date()} ~ {val['date'].max().date()} ({len(val)}일)")
    print(f"Test : {test['date'].min().date()} ~ {test['date'].max().date()} ({len(test)}일)")
    return train, val, test
