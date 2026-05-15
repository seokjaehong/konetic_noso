"""
시스템 레벨 연료 믹스 최적화 모듈 (Step 4)

목적: 삼천포(석탄) + 영흥(석탄) + 분당(LNG) + 영동(석탄/바이오) + 여수(석탄)
      5사업소 총 배출 사회적 비용 최소화 under 수요 충족 제약

결정 변수:
  - gen_삼천포 (MWh/일): 이용률 50~84% (실데이터 p5 기반)
  - gen_영흥   (MWh/일): 이용률 65~94% (실데이터 기반)
  - gen_영동   (MWh/일): 고정 (현재 평균, 소규모 사업소)
  - gen_여수   (MWh/일): 고정 (현재 평균, 소규모 사업소)
  - gen_분당   (MWh/일): demand - gen_삼 - gen_영 - gen_영동 - gen_여수 (역산)

목적함수:
  min: ΣSOx×500 + ΣNOx×2130 + Σ먼지×770  (원/kg, 환경부 대기배출부과금)
  단, 분당은 NOx만

제약:
  - demand_total = gen_삼+gen_영+gen_분+gen_영동+gen_여수 (수요 충족)
  - 0 ≤ gen_분 ≤ gen_분_max (분당 용량 제한)
  - 계절관리제: 삼천포·영흥 12~3월 이용률 상한 80%
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

REPORTS_BASE  = Path(__file__).parent.parent / 'reports'
PROCESSED_DIR = Path(__file__).parent.parent / 'data' / 'processed'

# ── 사업소별 설비 및 운영 제약 ────────────────────────────────
PLANT_CONSTRAINTS = {
    '삼천포': {
        'base_gen_mwh': 28_427,        # 현재 평균 발전량 MWh/일
        'base_utilization': 0.700,     # 현재 평균 이용률 (가중평균 수정 후)
        'util_min': 0.50,              # 실데이터 p5=54.7%
        'util_max': 0.84,              # 실데이터 p95=84.1%
        'seasonal_mgmt_max': 0.80,     # 계절관리제 이용률 상한
        'gen_cost_per_mwh': 65_000,    # 유연탄 발전 원가 (원/MWh)
        'eff_slope': 0.3111,           # 열효율 회귀: 이용률↑→효율↑ (r=+0.72)
        'eff_intercept': 14.62,
        'eff_min': 0.20, 'eff_max': 0.45,
    },
    '영흥': {
        'base_gen_mwh': 67_585,
        'base_utilization': 0.828,
        'util_min': 0.65,              # 실데이터 p1=59.5% → 65%
        'util_max': 0.94,              # 실데이터 max=94.1%
        'seasonal_mgmt_max': 0.80,
        'gen_cost_per_mwh': 65_000,
        'eff_slope': -0.2589,          # 열효율 회귀: 이용률↑→효율↓ 기저부하 특성 (r=-0.29)
        'eff_intercept': 48.75,
        'eff_min': 0.20, 'eff_max': 0.40,
    },
    '분당': {
        'base_gen_mwh': 6_693,
        'base_utilization': 0.310,
        'gen_max_mwh': 890 * 24,       # 설비용량 890MW 기준 MWh/일
        'gen_cost_per_mwh': 120_000,   # LNG 발전 원가 (원/MWh)
        'eff_slope': 0.0416,           # 열효율 회귀: LNG 복합 (r=+0.54)
        'eff_intercept': 26.97,
        'eff_min': 0.25, 'eff_max': 0.50,
    },
    '영동': {
        'base_gen_mwh': 4_526,         # 현재 평균 발전량 MWh/일
        'base_utilization': 0.688,     # 현재 평균 이용률 68.8%
        'util_min': 0.45,              # 실데이터 p5=43.5%
        'util_max': 0.90,              # 실데이터 p95=89.1%
        'seasonal_mgmt_max': 0.80,
        'gen_max_mwh': 400 * 24,       # 설비용량 400MW
        'gen_cost_per_mwh': 70_000,    # 석탄/바이오매스 혼소 원가
        'eff_slope': 0.31,
        'eff_intercept': 14.29,        # at util=68.8% → eff=35.6%
        'eff_min': 0.25, 'eff_max': 0.45,
    },
    '여수': {
        'base_gen_mwh': 8_867,         # 현재 평균 발전량 MWh/일
        'base_utilization': 0.656,     # 현재 평균 이용률 65.6%
        'util_min': 0.45,              # 실데이터 p5=47.4%
        'util_max': 0.84,              # 실데이터 max=84.7%
        'seasonal_mgmt_max': 0.80,
        'gen_max_mwh': 680 * 24,       # 설비용량 680MW
        'gen_cost_per_mwh': 65_000,    # 유연탄 발전 원가
        'eff_slope': 0.31,
        'eff_intercept': 15.62,        # at util=65.6% → eff=36.0%
        'eff_min': 0.25, 'eff_max': 0.45,
    },
}

SOCIAL_COSTS = {
    'SOx':  500,    # 원/kg — 환경부 대기배출부과금 (2018년 이후 고시 기준)
    'NOx':  2_130,  # 원/kg — 환경부 대기배출부과금
    '먼지':  770,   # 원/kg — 환경부 대기배출부과금
}


def predict_emissions(model, feature_template: pd.DataFrame,
                      gen_mwh: float, base_gen_mwh: float, base_util: float,
                      plant_name: str, is_bundang: bool = False) -> dict:
    """
    단일 발전량 조건에서 배출량 예측.
    utilization은 gen_mwh / base_gen_mwh 비율로 스케일링.
    """
    row = feature_template.copy()
    if 'gen_mwh_combined' in row.columns:
        row['gen_mwh_combined'] = gen_mwh
    if 'utilization' in row.columns:
        scaled_util = base_util * (gen_mwh / max(base_gen_mwh, 1))
        row['utilization'] = scaled_util * 100  # %

    # 유연탄 소비 추정 (석탄만) — 사업소별 열효율 회귀식 적용
    if '유연탄' in row.columns and not is_bundang:
        cfg = PLANT_CONSTRAINTS.get(plant_name, {})
        eff_s = cfg.get('eff_slope', 0.3111)
        eff_i = cfg.get('eff_intercept', 14.62)
        eff_min = cfg.get('eff_min', 0.20)
        eff_max = cfg.get('eff_max', 0.45)
        eff = max(eff_min, min(eff_max, (eff_s * scaled_util * 100 + eff_i) / 100))
        row['유연탄'] = gen_mwh / 3.6 * (1 / eff) / 1000

    pred = model.predict(row)[0]
    pred = np.maximum(pred, 0)

    if is_bundang:
        return {'SOx': 0.0, 'NOx': float(pred[0]), '먼지': 0.0}
    else:
        target_names = ['SOx', 'NOx', '먼지']
        return {t: float(pred[i]) for i, t in enumerate(target_names) if i < len(pred)}


def calc_social_cost(emissions: dict) -> float:
    """사회적 비용 합산 (원/일)"""
    return sum(emissions.get(t, 0) * SOCIAL_COSTS[t] for t in SOCIAL_COSTS)


def _apply_overrides(ft: pd.DataFrame, overrides: dict, is_seasonal_mgmt: bool) -> pd.DataFrame:
    """기상 오버라이드 및 계절관리제 플래그 적용"""
    row = ft.copy()
    for k, v in overrides.items():
        if k in row.columns:
            row[k] = v
    if 'stagnation_idx' in row.columns:
        hum = row['humidity_mean'].values[0] if 'humidity_mean' in row.columns else 60
        ws  = max(row['wind_speed_mean'].values[0] if 'wind_speed_mean' in row.columns else 1, 0.1)
        row['stagnation_idx'] = hum / ws
    if 'seasonal_mgmt' in row.columns:
        row['seasonal_mgmt'] = int(is_seasonal_mgmt)
    return row


def run_fuel_mix_grid_search(
    models: dict,
    feature_templates: dict,
    demand_mwh: float,
    weather_overrides: dict,
    is_seasonal_mgmt: bool = False,
    sam_steps: int = 10,
    yh_steps: int = 8,
    yd_steps: int = 5,
    ys_steps: int = 5,
) -> pd.DataFrame:
    """
    삼천포 × 영흥 × 영동 × 여수 이용률 4D 그리드 서치.
    분당 발전량은 demand - gen_삼 - gen_영 - gen_영동 - gen_여수 로 역산.

    Parameters
    ----------
    models : {plant: model}
    feature_templates : {plant: pd.DataFrame}
    demand_mwh : 총 수요 MWh/일 (5사업소 합산)
    weather_overrides : 기상 오버라이드 dict
    is_seasonal_mgmt : 계절관리제 여부 (True이면 이용률 상한 80%)
    sam_steps, yh_steps, yd_steps, ys_steps : 각 사업소 이용률 그리드 단계 수
    """
    sam_cfg = PLANT_CONSTRAINTS['삼천포']
    yh_cfg  = PLANT_CONSTRAINTS['영흥']
    yd_cfg  = PLANT_CONSTRAINTS['영동']
    ys_cfg  = PLANT_CONSTRAINTS['여수']
    bd_cfg  = PLANT_CONSTRAINTS['분당']

    def _util_max(cfg):
        return min(cfg['util_max'], cfg['seasonal_mgmt_max']) if is_seasonal_mgmt else cfg['util_max']

    sam_utils = np.linspace(sam_cfg['util_min'], _util_max(sam_cfg), sam_steps)
    yh_utils  = np.linspace(yh_cfg['util_min'],  _util_max(yh_cfg),  yh_steps)
    yd_utils  = np.linspace(yd_cfg['util_min'],  _util_max(yd_cfg),  yd_steps)  if '영동' in models else [yd_cfg['base_utilization']]
    ys_utils  = np.linspace(ys_cfg['util_min'],  _util_max(ys_cfg),  ys_steps)  if '여수' in models else [ys_cfg['base_utilization']]

    records = []

    for sam_util in sam_utils:
        gen_sam = sam_cfg['base_gen_mwh'] * (sam_util / sam_cfg['base_utilization'])
        sam_row = _apply_overrides(feature_templates['삼천포'], weather_overrides, is_seasonal_mgmt)

        for yh_util in yh_utils:
            gen_yh = yh_cfg['base_gen_mwh'] * (yh_util / yh_cfg['base_utilization'])
            yh_row = _apply_overrides(feature_templates['영흥'], weather_overrides, is_seasonal_mgmt)

            for yd_util in yd_utils:
                gen_yd = yd_cfg['base_gen_mwh'] * (yd_util / yd_cfg['base_utilization'])
                yd_row = _apply_overrides(feature_templates.get('영동', feature_templates['삼천포']),
                                          weather_overrides, is_seasonal_mgmt)

                for ys_util in ys_utils:
                    gen_ys = ys_cfg['base_gen_mwh'] * (ys_util / ys_cfg['base_utilization'])

                    gen_bd = demand_mwh - gen_sam - gen_yh - gen_yd - gen_ys

                    # 분당 용량 제약
                    if gen_bd < 0 or gen_bd > bd_cfg['gen_max_mwh']:
                        continue

                    ys_row = _apply_overrides(feature_templates.get('여수', feature_templates['삼천포']),
                                              weather_overrides, is_seasonal_mgmt)
                    bd_row = _apply_overrides(feature_templates['분당'], weather_overrides, is_seasonal_mgmt)

                    em_sam = predict_emissions(
                        models['삼천포'], sam_row,
                        gen_sam, sam_cfg['base_gen_mwh'], sam_cfg['base_utilization'], '삼천포',
                    )
                    em_yh = predict_emissions(
                        models['영흥'], yh_row,
                        gen_yh, yh_cfg['base_gen_mwh'], yh_cfg['base_utilization'], '영흥',
                    )
                    em_yd = predict_emissions(
                        models['영동'], yd_row,
                        gen_yd, yd_cfg['base_gen_mwh'], yd_cfg['base_utilization'], '영동',
                    ) if '영동' in models else {'SOx': 0.0, 'NOx': 0.0, '먼지': 0.0}
                    em_ys = predict_emissions(
                        models['여수'], ys_row,
                        gen_ys, ys_cfg['base_gen_mwh'], ys_cfg['base_utilization'], '여수',
                    ) if '여수' in models else {'SOx': 0.0, 'NOx': 0.0, '먼지': 0.0}
                    em_bd = predict_emissions(
                        models['분당'], bd_row,
                        gen_bd, bd_cfg['base_gen_mwh'], bd_cfg['base_utilization'], '분당',
                        is_bundang=True,
                    )

                    total_em = {
                        t: em_sam.get(t, 0) + em_yh.get(t, 0) + em_yd.get(t, 0)
                           + em_ys.get(t, 0) + em_bd.get(t, 0)
                        for t in ['SOx', 'NOx', '먼지']
                    }
                    total_cost = calc_social_cost(total_em)

                    gen_cost = (gen_sam * sam_cfg['gen_cost_per_mwh'] +
                                gen_yh  * yh_cfg['gen_cost_per_mwh'] +
                                gen_yd  * yd_cfg['gen_cost_per_mwh'] +
                                gen_ys  * ys_cfg['gen_cost_per_mwh'] +
                                gen_bd  * bd_cfg['gen_cost_per_mwh'])

                    records.append({
                        'sam_util':  round(sam_util, 3),
                        'yh_util':   round(yh_util, 3),
                        'yd_util':   round(yd_util, 3),
                        'ys_util':   round(ys_util, 3),
                        'gen_sam':   round(gen_sam, 0),
                        'gen_yh':    round(gen_yh, 0),
                        'gen_yd':    round(gen_yd, 0),
                        'gen_ys':    round(gen_ys, 0),
                        'gen_bd':    round(gen_bd, 0),
                        'SOx_total':       round(total_em['SOx'], 0),
                        'NOx_total':       round(total_em['NOx'], 0),
                        'dust_total':      round(total_em['먼지'], 0),
                        'social_cost_krw': round(total_cost, 0),
                        'gen_cost_krw':    round(gen_cost, 0),
                        'total_cost_krw':  round(total_cost + gen_cost, 0),
                    })

    if not records:
        print(f'  [경고] 유효한 조합 없음 (demand={demand_mwh:,.0f} MWh)')
        return pd.DataFrame()

    df = pd.DataFrame(records)
    return df.sort_values('social_cost_krw').reset_index(drop=True)
