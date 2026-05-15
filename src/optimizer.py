"""
발전량(이용률) 최적화 모듈 — 삼천포 유연탄 발전소
- 삼천포는 순수 유연탄 발전소 → 연료 믹스 최적화 불가
- 최적화 변수: 이용률(utilization) / 발전량(gen_mwh)
  → 대기정체 조건에서 발전량을 줄여 배출량 감축
  → 감발로 인한 전력 손실은 LNG 계통(분당화력 등)으로 보완하는 시나리오
- 시나리오별 배출 감축 효과 분석
- B/C 분석 (비용 대비 편익)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

from pathlib import Path
from sklearn.multioutput import MultiOutputRegressor

REPORT_DIR = Path(__file__).parent.parent / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── 발전소 물리적 제약 (현실성 확보) ──────────────────────────
CONSTRAINTS = {
    "utilization_min": 0.25,      # 최소 이용률 25% (계통 안정 하한)
    "utilization_max": 0.95,      # 최대 이용률 95%
    "utilization_current_avg": 0.527,  # 현재 평균 이용률 (52.7%)
    "min_generation_mwh": 8_000,  # 최소 발전량 MWh/일 (삼천포 기준)
    "max_generation_mwh": 40_000, # 최대 발전량 MWh/일
    "rated_capacity_mw": 3_240,   # 삼천포 설비용량 3,240MW (1~6호기 합산)
}

# ── 경제성 파라미터 (B/C 분석용) ──────────────────────────────
ECONOMICS = {
    # 2024년 기준 국내 연료 단가 (원/ton)
    "coal_price_per_ton": 150_000,    # 유연탄 약 150,000원/ton
    "lng_price_per_ton": 900_000,     # LNG 약 900,000원/ton (대체 연료)
    # 삼천포 감발 시 LNG 발전 대체 비용 (MWh당)
    "coal_gen_cost_per_mwh": 65_000,  # 유연탄 발전 원가 ~65,000원/MWh
    "lng_gen_cost_per_mwh": 120_000,  # LNG 발전 원가 ~120,000원/MWh
    # 대기오염 사회적 비용 (원/kg) — 환경부 대기배출부과금 고시 기준 (2018년 이후)
    "sox_social_cost": 500,           # SOx: 500원/kg
    "nox_social_cost": 2_130,         # NOx: 2,130원/kg
    "dust_social_cost": 770,          # 먼지(PM): 770원/kg
}


# ══════════════════════════════════════════════════════════════
# 1. 이용률 그리드 서치 최적화
# ══════════════════════════════════════════════════════════════

def optimize_utilization_grid(
    model: MultiOutputRegressor,
    weather_row: pd.Series,
    base_gen_mwh: float,
    base_utilization: float,
    feature_template: pd.DataFrame,
    target_names: list[str],
    step: float = 0.05,
    plant_config: dict | None = None,
) -> pd.DataFrame:
    """
    이용률을 grid search하여 배출량 최소화 운전 조건 탐색

    삼천포는 순수 유연탄 발전소이므로 연료 믹스 대신
    발전량(이용률) 조정을 통한 배출 감축 최적화를 수행.

    Parameters
    ----------
    weather_row      : 해당 날의 기상 데이터 (1행)
    base_gen_mwh     : 현재 평균 발전량 (MWh/일)
    base_utilization : 현재 평균 이용률 (0~1)
    feature_template : 전체 특성 컬럼 구조
    step             : 이용률 탐색 간격 (기본 5%)

    Returns
    -------
    pd.DataFrame: 각 이용률별 예측 배출량 및 비용
    """
    cfg = plant_config or {}
    util_min = cfg.get("utilization_min", CONSTRAINTS["utilization_min"])
    util_max = cfg.get("utilization_max", CONSTRAINTS["utilization_max"])
    rated_mw = cfg.get("rated_capacity_mw", CONSTRAINTS["rated_capacity_mw"])
    coal_gen_cost = cfg.get("coal_gen_cost_per_mwh", ECONOMICS["coal_gen_cost_per_mwh"])
    lng_gen_cost  = cfg.get("lng_gen_cost_per_mwh",  ECONOMICS["lng_gen_cost_per_mwh"])
    sox_cost  = cfg.get("sox_social_cost",  ECONOMICS["sox_social_cost"])
    nox_cost  = cfg.get("nox_social_cost",  ECONOMICS["nox_social_cost"])
    dust_cost = cfg.get("dust_social_cost", ECONOMICS["dust_social_cost"])

    # 계절관리제(12~3월) 이용률 상한 적용
    # weather_row에 seasonal_mgmt=1인 경우 seasonal_mgmt_max로 util_max 제한
    seasonal_mgmt_max = cfg.get("seasonal_mgmt_max", None)
    if seasonal_mgmt_max is not None:
        is_seasonal_mgmt = False
        if isinstance(weather_row, pd.Series):
            is_seasonal_mgmt = bool(weather_row.get("seasonal_mgmt", 0))
        if is_seasonal_mgmt:
            util_max = min(util_max, seasonal_mgmt_max)

    util_range = np.arange(util_min, util_max + step, step)
    results = []

    for util in util_range:
        # gen_mwh: 현재 기준값에서 이용률 비율로 비례 스케일링
        # (rated_capacity × util × 24 방식은 이용률 기준 설비용량이 다를 때 오류 발생)
        gen_mwh = base_gen_mwh * (util / base_utilization) if base_utilization > 0 else rated_mw * util * 24

        # 사업소별 열효율 회귀식 적용 (실데이터 기반)
        # 삼천포: efficiency = 0.3111×util(%) + 14.62  (r=+0.72, 이용률↑→효율↑)
        # 영흥  : efficiency = -0.2589×util(%) + 48.75 (r=-0.29, 이용률↑→효율↓, 기저부하)
        # 분당  : efficiency = 0.0416×util(%) + 26.97  (r=+0.54, LNG 복합)
        eff_slope     = cfg.get("efficiency_slope",     0.3111)
        eff_intercept = cfg.get("efficiency_intercept", 14.62)
        eff_min       = cfg.get("efficiency_min",       0.20)
        eff_max       = cfg.get("efficiency_max",       0.45)
        estimated_efficiency = max(eff_min, min(eff_max,
            (eff_slope * util * 100 + eff_intercept) / 100))
        coal_ton = gen_mwh / 3.6 * (1 / estimated_efficiency) / 1000  # 동적 열효율 적용

        # 특성 벡터 구성
        row = feature_template.copy()
        for col in weather_row.index:
            if col in row.columns:
                row[col] = weather_row[col]

        if "gen_mwh_combined" in row.columns:
            row["gen_mwh_combined"] = gen_mwh
        if "utilization" in row.columns:
            row["utilization"] = util * 100  # % 단위
        if "유연탄" in row.columns:
            row["유연탄"] = coal_ton

        pred = model.predict(row)[0]
        emission_dict = dict(zip(target_names, pred))
        # 음수 예측 방지
        emission_dict = {k: max(v, 0) for k, v in emission_dict.items()}

        # 사회적 비용
        social_cost = (
            emission_dict.get("SOx", 0) * sox_cost +
            emission_dict.get("NOx", 0) * nox_cost +
            emission_dict.get("먼지", 0) * dust_cost
        )

        # 발전 원가 (감발 시 대체 비용 포함)
        # 대체 비용 = 감발량 × (LNG 원가 - 석탄 원가) [증분 비용만 계상]
        # 석탄 원가(65,000원/MWh)는 절감되므로 순 추가 비용은 차액만
        gen_cost = gen_mwh * coal_gen_cost
        reduction_mwh = max(base_gen_mwh - gen_mwh, 0)
        replacement_cost = reduction_mwh * max(lng_gen_cost - coal_gen_cost, 0)

        results.append({
            "utilization": round(util, 2),
            "gen_mwh": round(gen_mwh, 0),
            "heat_efficiency_pct": round(estimated_efficiency * 100, 1),
            **emission_dict,
            "social_cost_krw": social_cost,
            "gen_cost_krw": gen_cost,
            "replacement_cost_krw": replacement_cost,
            "total_cost_krw": social_cost + gen_cost + replacement_cost,
        })

    return pd.DataFrame(results)


def find_optimal(grid_df: pd.DataFrame, objective: str = "emission") -> dict:
    """
    최적 이용률 선택

    Parameters
    ----------
    objective : "emission" (배출 최소화) | "cost" (총비용 최소화)
    """
    available_targets = [t for t in ["SOx", "NOx", "먼지"] if t in grid_df.columns]

    if objective == "emission":
        grid_df["total_emission"] = grid_df[available_targets].sum(axis=1)
        best_idx = grid_df["total_emission"].idxmin()
    else:
        best_idx = grid_df["total_cost_krw"].idxmin()

    best = grid_df.loc[best_idx].to_dict()
    print(f"\n[최적 이용률] objective={objective}")
    print(f"  이용률: {best['utilization']*100:.0f}%  |  발전량: {best['gen_mwh']:.0f} MWh/일")
    for t in available_targets:
        print(f"  {t}: {best[t]:.1f} kg/일")
    return best


# ══════════════════════════════════════════════════════════════
# 2. 시나리오 분석
# ══════════════════════════════════════════════════════════════

SCENARIOS = {
    "A_summer_stagnation": {
        "desc": "여름 고온·고습·저풍속 (대기정체일)",
        # temp 32→29.0: 실데이터 max=29.3, 해안입지 특성상 내륙보다 낮음
        # wind 0.8→1.0: 데이터 min=0.74 경계값 회피, stagnation_idx=88 충분
        "overrides": {
            "temp_mean": 29.0, "humidity_mean": 88.0,
            "wind_speed_mean": 1.0, "precipitation": 0.0,
            # stagnation_idx = 88/1.0 = 88 (데이터 85th percentile, 강한 정체)
        },
    },
    "B_winter_cold": {
        "desc": "겨울 저온·건조·강풍 (난방수요 피크)",
        # 물리적으로 타당. 계절관리제(12~3월) 80% 이용률 상한 적용 대상
        "overrides": {
            "temp_mean": -5.0, "humidity_mean": 45.0,
            "wind_speed_mean": 7.0, "precipitation": 0.0,
            "seasonal_mgmt": 1,   # 12~3월 계절관리제 시행기간
        },
    },
    "C_solar_max": {
        "desc": "태양광 최대 발전일 (맑은 여름날, 석탄 감발 여력)",
        # precipitation=0.0 추가: 맑은 날 강수 물리적 모순 해소
        # feature_overrides: 실발전량 연동 (solar_radiation=25 비례 추정)
        "overrides": {
            "temp_mean": 28.0, "humidity_mean": 60.0,
            "wind_speed_mean": 3.0, "solar_radiation": 25.0,
            "precipitation": 0.0,          # 맑은 날 → 강수 없음
        },
        "feature_overrides": {
            "solar_mwh": 75.0,             # solar_radiation=25 비례 추정 (데이터 90th=75.7)
            "wind_mwh": 0.3,               # 약풍(3m/s) → 풍력 미미
        },
    },
    "D_wind_max": {
        "desc": "풍력 최대 발전일 (저기압·전선 통과, 강풍·흐림)",
        # solar_radiation=8.0 추가: 강풍=저기압권=흐림, 일사량 감소
        # wind_speed_max=18.0: 강풍일 최대풍속 정합
        # feature_overrides: wind_mwh 현실화 (max=12.7 기준 추정)
        "overrides": {
            "temp_mean": 15.0, "humidity_mean": 65.0,
            "wind_speed_mean": 9.0, "wind_speed_max": 18.0,
            "solar_radiation": 8.0,        # 저기압·흐림 → 일사량 감소
            "precipitation": 5.0,
        },
        "feature_overrides": {
            "solar_mwh": 15.0,             # 흐린 날 태양광 저조
            "wind_mwh": 10.0,              # wind_speed=9m/s 강풍 실발전 추정
        },
    },
    "E_spring_dust": {
        "desc": "봄 황사 잔류·약풍 (초미세먼지 주의보 지속)",
        # wind 1.5→2.5: 황사 유입 후 잔류 정체. 완전 무풍보다 약간 바람 있어야 황사 도달
        # solar_radiation=18.0: 황사 산란 효과, 직달 일사 소폭 감소
        "overrides": {
            "temp_mean": 13.0, "humidity_mean": 72.0,
            "wind_speed_mean": 2.5,        # 1.5→2.5 (황사 잔류 정체 현실적 풍속)
            "solar_radiation": 18.0,       # 황사 산란 효과
            "precipitation": 0.0,
            # stagnation_idx = 72/2.5 = 28.8 (데이터 50~75th percentile, 중간 정체)
        },
    },
    "F_winter_stagnation": {
        "desc": "겨울 대기정체 (계절관리제 시행기간, 12~3월)",
        # solar_radiation=6.0: 겨울 단일(短日), 흐린 대기정체일 일사량
        # 계절관리제 적용 → 이용률 80% 상한 정책 맥락
        "overrides": {
            "temp_mean": 2.0, "humidity_mean": 83.0,
            "wind_speed_mean": 1.0, "precipitation": 0.0,
            "solar_radiation": 6.0,        # 겨울 흐린 대기정체일 일사량
            "seasonal_mgmt": 1,            # 12~3월 계절관리제 → 이용률 80% 상한
            # stagnation_idx = 83/1.0 = 83 (상위 10% 극단, seasonal_mgmt=1)
        },
        "feature_overrides": {
            "solar_mwh": 10.0,             # 겨울 저일사 → 태양광 저조
        },
    },
    "G_renewable_peak_spring": {
        "desc": "봄·가을 이행기 신재생 우세 (태양광+풍력 합계 최대, 석탄 감발 압력)",
        # 강풍+최대일사 물리 모순 해소: 봄·가을 중강풍 + 중상위 일사 조합
        # wind_speed=5.5(중강풍): 풍력 발전 가능하되 저기압권 흐림 아님
        # solar_radiation=20.0: 맑은 봄날 중상위 일사 (강풍 아닌 날)
        "overrides": {
            "temp_mean": 16.0, "humidity_mean": 58.0,
            "wind_speed_mean": 5.5, "wind_speed_max": 10.0,
            "solar_radiation": 20.0,
            "precipitation": 0.0,
        },
        "feature_overrides": {
            "solar_mwh": 58.0,             # solar_radiation=20 비례 추정
            "wind_mwh": 4.0,               # wind_speed=5.5 중간 발전 추정
            # 신재생 합계 ~62 MWh (현실적 최대 조합)
        },
    },
    "H_summer_peak": {
        "desc": "폭염·냉방수요 피크 (전력 계통 최대 급전 요구)",
        # A(여름정체)와 차별화: H는 폭염+높은 이용률 강제 맥락
        # wind 1.5→1.2로 소폭 강화, solar_radiation 추가
        "overrides": {
            "temp_mean": 29.0, "humidity_mean": 87.0,
            "wind_speed_mean": 1.2,        # A(1.0)와 소폭 차별화
            "solar_radiation": 22.0,       # 폭염 맑은 날 높은 일사
            "precipitation": 0.0,
            # 운영 맥락: 냉방수요 피크 → 전력거래소 최대 급전 요구 상황
        },
        "feature_overrides": {
            "solar_mwh": 64.0,             # solar_radiation=22 비례 추정
        },
    },
}


def run_scenario_analysis(
    model: MultiOutputRegressor,
    base_weather: pd.Series,
    base_gen_mwh: float,
    base_utilization: float,
    feature_template: pd.DataFrame,
    target_names: list[str],
    plant_config: dict | None = None,
    report_dir: Path | None = None,
) -> pd.DataFrame:
    """
    시나리오별 현재 vs 최적 이용률 운전 배출량 비교

    SCENARIOS 딕셔너리의 두 가지 override 지원:
    - overrides: 기상 변수 (base_weather에 적용)
    - feature_overrides: 신재생 실발전량 등 feature_template에 직접 적용
      예) solar_mwh, wind_mwh — 기상 변수와 연동해 현실성 확보

    Returns
    -------
    pd.DataFrame: 시나리오별 현재/최적 배출량 및 감축률
    """
    scenario_results = []

    for sc_key, sc_info in SCENARIOS.items():
        sc_weather = base_weather.copy()
        for k, v in sc_info["overrides"].items():
            sc_weather[k] = v  # base_weather에 없는 키(seasonal_mgmt 등)도 추가 허용

        # 대기정체 지수 재계산
        hum = sc_weather.get("humidity_mean", 60) if hasattr(sc_weather, "get") else sc_weather["humidity_mean"]
        ws  = sc_weather.get("wind_speed_mean", 1) if hasattr(sc_weather, "get") else sc_weather["wind_speed_mean"]
        sc_weather["stagnation_idx"] = hum / max(ws, 0.1)

        # feature_overrides 적용 (solar_mwh, wind_mwh 등 신재생 실발전량 연동)
        sc_feature_template = feature_template.copy()
        for k, v in sc_info.get("feature_overrides", {}).items():
            if k in sc_feature_template.columns:
                sc_feature_template[k] = v

        # 이용률 그리드 서치
        grid = optimize_utilization_grid(
            model, sc_weather, base_gen_mwh, base_utilization,
            sc_feature_template, target_names, plant_config=plant_config,
        )

        # 현재 이용률 기준 배출량
        current_row = grid.iloc[(grid["utilization"] - base_utilization).abs().argsort()[:1]]
        optimal = find_optimal(grid, objective="emission")
        current = current_row.iloc[0].to_dict()

        row = {"scenario": sc_key, "description": sc_info["desc"]}
        for t in target_names:
            row[f"{t}_current"] = current.get(t, np.nan)
            row[f"{t}_optimal"] = optimal.get(t, np.nan)
            if current.get(t, 0) > 0:
                row[f"{t}_reduction_pct"] = (current[t] - optimal[t]) / current[t] * 100
            else:
                row[f"{t}_reduction_pct"] = 0.0

        row["current_utilization"] = base_utilization
        row["optimal_utilization"] = optimal["utilization"]
        row["current_gen_mwh"] = current.get("gen_mwh", base_gen_mwh)
        row["optimal_gen_mwh"] = optimal.get("gen_mwh", 0)
        row["social_cost_saving_krw"] = (
            current.get("social_cost_krw", 0) - optimal.get("social_cost_krw", 0)
        )
        row["replacement_cost_krw"] = optimal.get("replacement_cost_krw", 0)
        scenario_results.append(row)

    result_df = pd.DataFrame(scenario_results)
    out_dir = report_dir or REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_dir / "scenario_results.csv", index=False, encoding="utf-8-sig")
    print(f"\n[OK] 시나리오 분석 결과 저장: {out_dir / 'scenario_results.csv'}")
    return result_df


# ══════════════════════════════════════════════════════════════
# 3. B/C 분석 (보고서 Ⅴ장)
# ══════════════════════════════════════════════════════════════

def calc_bc_analysis(
    scenario_df: pd.DataFrame,
    annual_operation_days: int = 330,
    report_dir: Path | None = None,
) -> pd.DataFrame:
    """
    연간 기준 비용 대비 편익 분석

    편익: 대기오염 사회적 비용 절감 (연간)
    비용: 감발로 인한 LNG 대체 발전 추가 비용 (연간)
    """
    bc_rows = []
    for _, row in scenario_df.iterrows():
        annual_benefit = row.get("social_cost_saving_krw", 0) * annual_operation_days
        annual_cost = row.get("replacement_cost_krw", 0) * annual_operation_days
        bc_ratio = annual_benefit / max(annual_cost, 1)

        # 감발량 (MWh/일)
        gen_reduction = row.get("current_gen_mwh", 0) - row.get("optimal_gen_mwh", 0)
        util_reduction_pct = (row.get("current_utilization", 0) - row.get("optimal_utilization", 0)) * 100

        bc_rows.append({
            "scenario": row["scenario"],
            "description": row["description"],
            "이용률_현재(%)": round(row.get("current_utilization", 0) * 100, 1),
            "이용률_최적(%)": round(row.get("optimal_utilization", 0) * 100, 1),
            "감발량_MWh일": round(gen_reduction, 0),
            "annual_benefit_억원": round(annual_benefit / 1e8, 2),
            "annual_replacement_cost_억원": round(annual_cost / 1e8, 2),
            "BC_ratio": round(bc_ratio, 2),
            "verdict": "경제적" if bc_ratio >= 1.0 else "환경 편익 우선 고려",
        })

    bc_df = pd.DataFrame(bc_rows)
    out_dir = report_dir or REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    bc_df.to_csv(out_dir / "bc_analysis.csv", index=False, encoding="utf-8-sig")
    print("\n[B/C 분석 결과]")
    print(bc_df[["scenario", "이용률_현재(%)", "이용률_최적(%)", "annual_benefit_억원", "BC_ratio", "verdict"]].to_string(index=False))
    return bc_df
