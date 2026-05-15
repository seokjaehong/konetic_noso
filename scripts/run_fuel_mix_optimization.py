"""
시스템 레벨 연료 믹스 최적화 실행 (Step 4) — 5사업소 4D 그리드

결정 변수: 삼천포(10) × 영흥(8) × 영동(5) × 여수(5) = 2,000 조합/시나리오
역산:      분당 = 총수요 - 삼천포 - 영흥 - 영동 - 여수

4개 시나리오:
  S0: 정상 (중위 기상)
  SA: 여름 대기정체
  SE: 봄 황사
  SF: 겨울 계절관리제

출력:
  reports/fuel_mix_01_2d_heatmap.png
  reports/fuel_mix_02_scenario_comparison.png
  reports/fuel_mix_03_baseline_vs_optimal.png
  reports/fuel_mix_optimization_results.csv
  reports/fuel_mix_scenario_summary.csv
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent / 'src'))
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from features import build_features, get_X_y
from fuel_mix_optimizer import (
    PLANT_CONSTRAINTS, SOCIAL_COSTS,
    run_fuel_mix_grid_search, predict_emissions, calc_social_cost,
    _apply_overrides,
)

# 4D 그리드 단계 수
GRID_STEPS = {'sam': 10, 'yh': 8, 'yd': 5, 'ys': 5}  # 총 2,000 조합/시나리오

matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

PROCESSED_DIR = Path(__file__).parent.parent / 'data' / 'processed'
REPORTS_BASE  = Path(__file__).parent.parent / 'reports'

DROP_BUNDANG = [
    'precipitation', 'pressure_mean', 'solar_radiation',
    'solar_mwh', 'wind_mwh', 'renewable_ratio', 'sox_factor_ma30',
    'SOx_lag1', 'SOx_lag7', '먼지_lag1', '먼지_lag7',
    '유연탄', 'coal_ratio',
]

# 사업소별 설정 (is_bundang, target)
PLANT_CONFIGS = {
    '삼천포': {'is_bundang': False, 'targets': ['SOx', 'NOx', '먼지']},
    '영흥':   {'is_bundang': False, 'targets': ['SOx', 'NOx', '먼지']},
    '분당':   {'is_bundang': True,  'targets': ['NOx']},
    '영동':   {'is_bundang': False, 'targets': ['SOx', 'NOx', '먼지']},
    '여수':   {'is_bundang': False, 'targets': ['SOx', 'NOx', '먼지']},
}

# ── 계절별 실수요 계산 (5사업소 gen_mwh 실데이터 합산) ──────────
def _calc_seasonal_demand() -> dict:
    """마스터 CSV 실데이터 기반 계절별 일평균 총 수요 MWh/일"""
    total = None
    for plant in ['삼천포', '영흥', '분당', '영동', '여수']:
        path = PROCESSED_DIR / f'master_{plant}.csv'
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=['date'])
        df['month'] = df['date'].dt.month
        df['season_key'] = df['month'].map({
            7: 'summer', 8: 'summer',           # 여름 (냉방 피크)
            5: 'spring', 6: 'spring',            # 봄 황사 시기
            12:'winter', 1:'winter',
            2:'winter',  3:'winter',             # 겨울 계절관리제
            4:'normal',  9:'normal',
            10:'normal', 11:'normal',            # 정상 (봄·가을)
        })
        gen = df.groupby('season_key')['gen_mwh'].mean()
        total = gen if total is None else total.add(gen, fill_value=0)
    return total.round(0).to_dict() if total is not None else {}

_SEASON_DEMAND = _calc_seasonal_demand()
_AVG_DEMAND = sum(_SEASON_DEMAND.values()) / len(_SEASON_DEMAND) if _SEASON_DEMAND else 116_098

# ── 4개 시나리오 정의 (계절별 실수요 반영) ───────────────────
FUEL_MIX_SCENARIOS = {
    'S0_normal': {
        'desc': '정상 (봄·가을 평균)',
        'overrides': {},
        'is_seasonal_mgmt': False,
        'demand_key': 'normal',
    },
    'SA_summer_stagnation': {
        'desc': '여름 대기정체 (냉방 피크)',
        'overrides': {
            'temp_mean': 29.0, 'humidity_mean': 88.0,
            'wind_speed_mean': 1.0, 'precipitation': 0.0,
        },
        'is_seasonal_mgmt': False,
        'demand_key': 'summer',
    },
    'SE_spring_dust': {
        'desc': '봄 황사 (수요 저점)',
        'overrides': {
            'temp_mean': 13.0, 'humidity_mean': 72.0,
            'wind_speed_mean': 2.5, 'precipitation': 0.0,
        },
        'is_seasonal_mgmt': False,
        'demand_key': 'spring',
    },
    'SF_winter_stagnation': {
        'desc': '겨울 계절관리제 (난방 수요)',
        'overrides': {
            'temp_mean': 2.0, 'humidity_mean': 83.0,
            'wind_speed_mean': 1.0, 'precipitation': 0.0,
        },
        'is_seasonal_mgmt': True,
        'demand_key': 'winter',
    },
}


def load_plant(plant: str, is_bundang: bool, targets: list):
    model = joblib.load(REPORTS_BASE / plant / f'emission_model_{plant}.pkl')
    master = pd.read_csv(PROCESSED_DIR / f'master_{plant}.csv', parse_dates=['date'])
    feat_df = build_features(master)
    feat_df = feat_df.dropna(subset=targets)
    X, _ = get_X_y(feat_df)
    if is_bundang:
        drop = [c for c in DROP_BUNDANG if c in X.columns]
        X = X.drop(columns=drop)
    ft = X.mean().to_frame().T.reset_index(drop=True)
    return model, ft


# ── 모델/특성 템플릿 로딩 ─────────────────────────────────────
print('=== 모델 로딩 ===')
models = {}
feature_templates = {}
for plant, cfg in PLANT_CONFIGS.items():
    model_path = REPORTS_BASE / plant / f'emission_model_{plant}.pkl'
    if not model_path.exists():
        print(f'  [SKIP] {plant} 모델 없음')
        continue
    models[plant], feature_templates[plant] = load_plant(
        plant, cfg['is_bundang'], cfg['targets']
    )
    print(f'  [OK] {plant}  ({feature_templates[plant].shape[1]}개 특성)')

required = ['삼천포', '영흥', '분당']
missing = [p for p in required if p not in models]
if missing:
    print(f'[ERROR] 필수 모델 없음: {missing}')
    exit(1)

# ── 총 수요: 5사업소 기준 평균 합산 ─────────────────────────
demand_base = sum(
    PLANT_CONSTRAINTS[p]['base_gen_mwh']
    for p in ['삼천포', '영흥', '분당', '영동', '여수']
    if p in PLANT_CONSTRAINTS
)
print(f'\n기준 수요: {demand_base:,.0f} MWh/일 (5사업소 평균 합산)')
print(f'  삼천포:{PLANT_CONSTRAINTS["삼천포"]["base_gen_mwh"]:,} '
      f'영흥:{PLANT_CONSTRAINTS["영흥"]["base_gen_mwh"]:,} '
      f'분당:{PLANT_CONSTRAINTS["분당"]["base_gen_mwh"]:,} '
      f'영동:{PLANT_CONSTRAINTS["영동"]["base_gen_mwh"]:,} '
      f'여수:{PLANT_CONSTRAINTS["여수"]["base_gen_mwh"]:,}')

print(f'  그리드: 삼천포×영흥×영동×여수 = {GRID_STEPS["sam"]}×{GRID_STEPS["yh"]}×{GRID_STEPS["yd"]}×{GRID_STEPS["ys"]} = {GRID_STEPS["sam"]*GRID_STEPS["yh"]*GRID_STEPS["yd"]*GRID_STEPS["ys"]:,}조합/시나리오')


# ── 현재 베이스라인 배출량 계산 ───────────────────────────────
def calc_baseline(weather_overrides: dict = {}, is_sm: bool = False) -> dict:
    """현재 평균 운전 기준 5사업소 배출량"""
    em_total = {'SOx': 0.0, 'NOx': 0.0, '먼지': 0.0}
    for plant, cfg in PLANT_CONFIGS.items():
        if plant not in models:
            continue
        pc = PLANT_CONSTRAINTS[plant]
        row = _apply_overrides(feature_templates[plant], weather_overrides, is_sm)
        em = predict_emissions(
            models[plant], row,
            pc['base_gen_mwh'], pc['base_gen_mwh'],
            pc['base_utilization'], plant,
            is_bundang=cfg['is_bundang'],
        )
        for t in em_total:
            em_total[t] += em.get(t, 0)
    return em_total


# ── 시나리오별 최적화 실행 ────────────────────────────────────
print('\n=== 시나리오별 연료 믹스 최적화 (5사업소) ===')
all_results = []
scenario_summaries = []

for sc_key, sc_info in FUEL_MIX_SCENARIOS.items():
    print(f'\n[{sc_key}] {sc_info["desc"]}')
    overrides = sc_info['overrides']
    is_sm = sc_info['is_seasonal_mgmt']

    # 계절별 실수요 적용 (없으면 5사업소 평균 합산으로 fallback)
    demand_mwh = _SEASON_DEMAND.get(sc_info['demand_key'], demand_base)
    print(f'  수요: {demand_mwh:,.0f} MWh/일 (계절={sc_info["demand_key"]})')

    grid_df = run_fuel_mix_grid_search(
        models=models,
        feature_templates=feature_templates,
        demand_mwh=demand_mwh,
        weather_overrides=overrides,
        is_seasonal_mgmt=is_sm,
        sam_steps=GRID_STEPS['sam'],
        yh_steps=GRID_STEPS['yh'],
        yd_steps=GRID_STEPS['yd'],
        ys_steps=GRID_STEPS['ys'],
    )

    if grid_df.empty:
        print(f'  [SKIP] 유효 조합 없음')
        continue

    best = grid_df.iloc[0]
    gen_bd_best = best['gen_bd']
    print(f'  유효 조합: {len(grid_df):,}개')
    print(f'  최적: 삼천포 {best["sam_util"]*100:.0f}% / 영흥 {best["yh_util"]*100:.0f}%'
          f' / 영동 {best["yd_util"]*100:.0f}% / 여수 {best["ys_util"]*100:.0f}%'
          f' / 분당 {gen_bd_best:,.0f} MWh')
    print(f'  총 NOx: {best["NOx_total"]/1000:.1f} ton/일 / 사회적비용: {best["social_cost_krw"]/1e8:.2f} 억원')

    baseline_em = calc_baseline(overrides, is_sm)
    baseline_cost = calc_social_cost(baseline_em)
    saving = baseline_cost - best['social_cost_krw']
    saving_pct = saving / max(baseline_cost, 1) * 100
    print(f'  절감: {saving/1e8:.2f} 억원/일 ({saving_pct:.1f}%)')

    grid_df.insert(0, 'scenario', sc_key)
    all_results.append(grid_df)

    bd_util = gen_bd_best / PLANT_CONSTRAINTS['분당']['gen_max_mwh']
    scenario_summaries.append({
        'scenario': sc_key,
        'description': sc_info['desc'],
        'demand_mwh': demand_mwh,
        'n_combinations': len(grid_df),
        'optimal_sam_util_pct':  round(best['sam_util'] * 100, 1),
        'optimal_yh_util_pct':   round(best['yh_util']  * 100, 1),
        'optimal_yd_util_pct':   round(best['yd_util']  * 100, 1),
        'optimal_ys_util_pct':   round(best['ys_util']  * 100, 1),
        'optimal_bd_util_pct':   round(bd_util * 100, 1),
        'optimal_bd_mwh':        round(gen_bd_best, 0),
        'baseline_NOx_ton':      round(baseline_em['NOx'] / 1000, 2),
        'optimal_NOx_ton':       round(best['NOx_total'] / 1000, 2),
        'baseline_SOx_ton':      round(baseline_em['SOx'] / 1000, 2),
        'optimal_SOx_ton':       round(best['SOx_total'] / 1000, 2),
        'baseline_dust_ton':     round(baseline_em['먼지'] / 1000, 2),
        'optimal_dust_ton':      round(best['dust_total'] / 1000, 2),
        'baseline_social_cost_억원': round(baseline_cost / 1e8, 3),
        'optimal_social_cost_억원':  round(best['social_cost_krw'] / 1e8, 3),
        'saving_억원_daily':    round(saving / 1e8, 3),
        'saving_pct':           round(saving_pct, 1),
    })

summary_df = pd.DataFrame(scenario_summaries)
print('\n=== 시나리오별 최적 결과 요약 ===')
cols_show = ['scenario', 'description',
             'optimal_sam_util_pct', 'optimal_yh_util_pct',
             'optimal_yd_util_pct',  'optimal_ys_util_pct',
             'optimal_bd_util_pct',
             'baseline_NOx_ton', 'optimal_NOx_ton', 'saving_pct']
print(summary_df[cols_show].to_string(index=False))

# ── 결과 CSV 저장 ─────────────────────────────────────────────
if all_results:
    combined_grid = pd.concat(all_results, ignore_index=True)
    combined_grid.to_csv(REPORTS_BASE / 'fuel_mix_optimization_results.csv',
                         index=False, encoding='utf-8-sig')
    print(f'\n[OK] 전체 그리드 결과: reports/fuel_mix_optimization_results.csv')

summary_df.to_csv(REPORTS_BASE / 'fuel_mix_scenario_summary.csv',
                  index=False, encoding='utf-8-sig')
print(f'[OK] 시나리오 요약: reports/fuel_mix_scenario_summary.csv')

if not all_results or summary_df.empty:
    print('[경고] 결과 없음 — 시각화 건너뜀')
    exit(0)

# ═══════════════════════════════════════════════════════════════
# 시각화
# ═══════════════════════════════════════════════════════════════
print('\n=== 시각화 생성 ===')

# ── Fig 1: S0 시나리오 2D 히트맵 (삼천포×영흥 이용률) ─────────
s0_grid = combined_grid[combined_grid['scenario'] == 'S0_normal'].copy()
if not s0_grid.empty:
    pivot = s0_grid.pivot_table(
        index='yh_util', columns='sam_util',
        values='social_cost_krw', aggfunc='mean'
    )
    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(9, 7))
        c = ax.contourf(
            pivot.columns * 100, pivot.index * 100,
            pivot.values / 1e8, levels=15, cmap='RdYlGn_r'
        )
        ax.contour(
            pivot.columns * 100, pivot.index * 100,
            pivot.values / 1e8, levels=8, colors='white', alpha=0.4, linewidths=0.7
        )
        cb = fig.colorbar(c, ax=ax)
        cb.set_label('사회적 비용 (억원/일)', fontsize=11)

        best_row = s0_grid.iloc[0]
        ax.scatter(best_row['sam_util'] * 100, best_row['yh_util'] * 100,
                   s=200, c='gold', marker='*', zorder=5,
                   edgecolors='black', linewidths=1, label='최적 조합')
        ax.scatter(
            PLANT_CONSTRAINTS['삼천포']['base_utilization'] * 100,
            PLANT_CONSTRAINTS['영흥']['base_utilization'] * 100,
            s=150, c='blue', marker='D', zorder=5,
            edgecolors='white', linewidths=1, label='현재 운전')

        ax.set_xlabel('삼천포 이용률 (%)', fontsize=12)
        ax.set_ylabel('영흥 이용률 (%)', fontsize=12)
        ax.set_title('시스템 레벨 사회적 비용 등고선\n'
                     '(삼천포×영흥 이용률 격자, 분당=수요역산, 영동·여수 고정, 정상 기상)',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(REPORTS_BASE / 'fuel_mix_01_2d_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print('[저장] fuel_mix_01_2d_heatmap.png')

# ── Fig 2: 4개 시나리오 최적 발전 믹스 + NOx 비교 ────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

x = np.arange(len(summary_df))
labels = [r['description'] for _, r in summary_df.iterrows()]

# 왼쪽: 최적 이용률 비교 (5사업소)
ax = axes[0]
plant_util_info = [
    ('삼천포', 'optimal_sam_util_pct', '#E74C3C'),
    ('영흥',   'optimal_yh_util_pct',  '#3498DB'),
    ('영동',   'optimal_yd_util_pct',  '#F39C12'),
    ('여수',   'optimal_ys_util_pct',  '#9B59B6'),
    ('분당',   'optimal_bd_util_pct',  '#2ECC71'),
]
n_plants = len(plant_util_info)
w = 0.15
offsets = np.linspace(-(n_plants - 1) / 2 * w, (n_plants - 1) / 2 * w, n_plants)
for (plant, col, color), offset in zip(plant_util_info, offsets):
    if col not in summary_df.columns:
        continue
    bars = ax.bar(x + offset, summary_df[col], w,
                  label=f'{plant}', color=color, alpha=0.85)
    for bar, val in zip(bars, summary_df[col]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=7)

# 현재 이용률 점선
for plant, col, color in plant_util_info[:4]:
    base_util = PLANT_CONSTRAINTS[plant]['base_utilization'] * 100
    ax.axhline(base_util, color=color, ls='--', lw=0.9, alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=12, ha='right', fontsize=9)
ax.set_ylabel('최적 이용률 (%)')
ax.set_title('시나리오별 최적 이용률 (5사업소)\n(점선=현재 평균)', fontweight='bold')
ax.legend(fontsize=8, ncol=3)
ax.set_ylim(0, 130)
ax.grid(axis='y', alpha=0.3)

# 오른쪽: 현재 vs 최적 NOx 비교
ax2 = axes[1]
ax2.bar(x - 0.2, summary_df['baseline_NOx_ton'], 0.35,
        label='현재 NOx', color='#E74C3C', alpha=0.75)
ax2.bar(x + 0.2, summary_df['optimal_NOx_ton'], 0.35,
        label='최적 NOx', color='#2ECC71', alpha=0.75)
for i, row in summary_df.iterrows():
    pct = (row['baseline_NOx_ton'] - row['optimal_NOx_ton']) / max(row['baseline_NOx_ton'], 1) * 100
    if abs(pct) > 0.1:
        ax2.annotate(f'{pct:.1f}%↓',
                     xy=(i, row['optimal_NOx_ton']),
                     xytext=(0, 5), textcoords='offset points',
                     ha='center', fontsize=8, color='green', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=12, ha='right', fontsize=9)
ax2.set_ylabel('총 NOx (ton/일)')
ax2.set_title('시나리오별 현재 vs 최적 NOx 배출\n(5사업소 합산)', fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('5사업소 시스템 레벨 연료 믹스 최적화 결과 (4D 그리드)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(REPORTS_BASE / 'fuel_mix_02_scenario_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('[저장] fuel_mix_02_scenario_comparison.png')

# ── Fig 3: 시나리오별 사회적 비용 절감 효과 ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

ax = axes[0]
bars = ax.bar(x, summary_df['saving_억원_daily'],
              color=['#2ECC71' if v > 0 else '#E74C3C' for v in summary_df['saving_억원_daily']],
              alpha=0.85, edgecolor='white')
for bar, val in zip(bars, summary_df['saving_억원_daily']):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.002 if val >= 0 else -0.01),
            f'{val:.3f}\n억원/일', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.axhline(y=0, color='black', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=12, ha='right', fontsize=10)
ax.set_ylabel('사회적 비용 절감 (억원/일)')
ax.set_title('시나리오별 사회적 비용 절감\n(현재 운전 대비, 양수=절감)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 연간 환산
ax2 = axes[1]
annual = summary_df['saving_억원_daily'] * 365
bars2 = ax2.bar(x, annual,
                color=['#2ECC71' if v > 0 else '#E74C3C' for v in annual],
                alpha=0.85, edgecolor='white')
for bar, val in zip(bars2, annual):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + (0.5 if val >= 0 else -2),
             f'{val:.1f}\n억원/년', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax2.axhline(y=0, color='black', lw=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=12, ha='right', fontsize=10)
ax2.set_ylabel('사회적 비용 절감 (억원/년)')
ax2.set_title('시나리오별 연간 절감 환산\n(일 절감 × 365)', fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('5사업소 시스템 레벨 연료 믹스 최적화 — 사회적 비용 절감 효과',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(REPORTS_BASE / 'fuel_mix_03_baseline_vs_optimal.png', dpi=150, bbox_inches='tight')
plt.close()
print('[저장] fuel_mix_03_baseline_vs_optimal.png')

# ── Fig 4: 최적 발전 믹스 구성 (스택형) ─────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

plant_cols = {
    '삼천포': ('gen_sam', '#E74C3C'),
    '영흥':   ('gen_yh',  '#3498DB'),
    '영동':   ('gen_yd',  '#F39C12'),
    '여수':   ('gen_ys',  '#9B59B6'),
    '분당':   ('gen_bd',  '#2ECC71'),
}

sc_keys = summary_df['scenario'].tolist()
sc_labels = summary_df['description'].tolist()

bottoms = np.zeros(len(sc_keys))
for plant, (col, color) in plant_cols.items():
    vals = []
    for sk in sc_keys:
        row = combined_grid[combined_grid['scenario'] == sk]
        if row.empty or col not in row.columns:
            vals.append(0)
        else:
            vals.append(row.iloc[0][col] / 1000)  # MWh → GWh
    vals = np.array(vals)
    ax.bar(x, vals, bottom=bottoms, label=plant, color=color, alpha=0.85, edgecolor='white')
    for xi, (v, b) in enumerate(zip(vals, bottoms)):
        if v > 3:
            ax.text(xi, b + v / 2, f'{v:.0f}', ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')
    bottoms += vals

ax.set_xticks(x)
ax.set_xticklabels(sc_labels, rotation=12, ha='right', fontsize=10)
ax.set_ylabel('최적 발전량 (GWh/일)')
ax.set_title('시나리오별 최적 발전 믹스 구성\n(5사업소 합산 = 총 수요 [계절별 실수요 반영])', fontweight='bold')
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)
max_demand = max(_SEASON_DEMAND.values()) if _SEASON_DEMAND else demand_base
ax.set_ylim(0, max_demand / 1000 * 1.12)
# 시나리오별 수요 점선
for xi, sk in enumerate(sc_keys):
    d_key = FUEL_MIX_SCENARIOS[sk]['demand_key']
    d_mwh = _SEASON_DEMAND.get(d_key, demand_base)
    ax.plot([xi - 0.4, xi + 0.4], [d_mwh / 1000, d_mwh / 1000],
            color='black', ls='--', lw=1.2)
    ax.text(xi, d_mwh / 1000 + 0.5, f'{d_mwh/1000:.0f}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(REPORTS_BASE / 'fuel_mix_04_mix_composition.png', dpi=150, bbox_inches='tight')
plt.close()
print('[저장] fuel_mix_04_mix_composition.png')

print('\n' + '='*65)
print('  5사업소 연료 믹스 최적화 완료')
print('='*65)
