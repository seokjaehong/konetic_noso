"""
3개 사업소 이용률 최적화 + 시나리오 분석 + B/C 분석
- 삼천포 (유연탄, SOx/NOx/먼지)
- 영흥   (유연탄, SOx/NOx/먼지)
- 분당   (LNG복합, NOx만)
"""

import sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from features import build_features, get_X_y, TARGET_COLS
from optimizer import (
    optimize_utilization_grid, find_optimal,
    run_scenario_analysis, calc_bc_analysis,
    SCENARIOS, ECONOMICS,
)

matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

PROCESSED_DIR = _ROOT / 'data' / 'processed'
REPORTS_BASE  = _ROOT / 'reports'

# ── 사업소별 최적화 설정 ─────────────────────────────────────
PLANT_OPT_CONFIGS = {
    '삼천포': {
        'rated_capacity_mw': 3_240,     # 삼천포 1~6호기 합산
        'coal_gen_cost_per_mwh': 65_000,
        'lng_gen_cost_per_mwh': 120_000,
        'sox_social_cost': 500,    # 환경부 대기배출부과금 원/kg
        'nox_social_cost': 2_130,
        'dust_social_cost': 770,
        'efficiency_slope': 0.3111,      # 실측 회귀: 이용률↑→효율↑ (r=+0.72)
        'efficiency_intercept': 14.62,
        'efficiency_min': 0.20,
        'efficiency_max': 0.45,
        'utilization_min': 0.50,           # 실데이터 최솟값 41%, 5th pct 54.7% → 보수적 하한 50%
        'utilization_max': 0.84,           # 실데이터 99th percentile 기준
        'active_targets': ['SOx', 'NOx', '먼지'],
        'drop_features': [],
        'gen_mwh_src': 'gen_mwh',       # 발전량 기준 컬럼 (월별 보간)
        'plant_label': '삼천포 유연탄',
    },
    '영흥': {
        'rated_capacity_mw': 5_080,     # 영흥 1~6호기 합산
        'coal_gen_cost_per_mwh': 65_000,
        'lng_gen_cost_per_mwh': 120_000,
        'sox_social_cost': 500,    # 환경부 대기배출부과금 원/kg
        'nox_social_cost': 2_130,
        'dust_social_cost': 770,
        'efficiency_slope': -0.2589,     # 실측 회귀: 이용률↑→효율↓ 기저부하 특성 (r=-0.29)
        'efficiency_intercept': 48.75,
        'efficiency_min': 0.20,
        'efficiency_max': 0.40,
        'utilization_min': 0.65,        # 실데이터 1st percentile 59.5% → 65% (계통 안정 하한)
        'utilization_max': 0.94,        # 실데이터 99th percentile 기준
        'seasonal_mgmt_max': 0.80,      # 계절관리제(12~3월) 이용률 80% 상한
        'active_targets': ['SOx', 'NOx', '먼지'],
        'drop_features': [],
        'gen_mwh_src': 'gen_mwh',
        'plant_label': '영흥 유연탄',
    },
    '영동': {
        'rated_capacity_mw': 400,       # 영동 1~2호기 합산 (200MW×2)
        'coal_gen_cost_per_mwh': 70_000,    # 국내탄+바이오매스 혼소 원가
        'lng_gen_cost_per_mwh': 120_000,
        'sox_social_cost': 500,
        'nox_social_cost': 2_130,
        'dust_social_cost': 770,
        'efficiency_slope': 0.10,
        'efficiency_intercept': 25.0,
        'efficiency_min': 0.20,
        'efficiency_max': 0.45,
        'utilization_min': 0.45,        # 실데이터 p5=43.5% → 45%
        'utilization_max': 0.90,        # 실데이터 max=92.2%
        'active_targets': ['SOx', 'NOx', '먼지'],
        'drop_features': [],
        'gen_mwh_src': 'gen_mwh',
        'plant_label': '영동에코 바이오매스',
    },
    '여수': {
        'rated_capacity_mw': 680,       # 여수 1~2호기 합산 (340MW×2)
        'coal_gen_cost_per_mwh': 68_000,    # 석탄+중유 혼소 원가
        'lng_gen_cost_per_mwh': 120_000,
        'sox_social_cost': 500,
        'nox_social_cost': 2_130,
        'dust_social_cost': 770,
        'efficiency_slope': 0.15,
        'efficiency_intercept': 22.0,
        'efficiency_min': 0.20,
        'efficiency_max': 0.45,
        'utilization_min': 0.45,        # 실데이터 min=41.1%, p5=47.4% → 45%
        'utilization_max': 0.84,        # 실데이터 max=84.7%
        'active_targets': ['SOx', 'NOx', '먼지'],
        'drop_features': [],
        'gen_mwh_src': 'gen_mwh',
        'plant_label': '여수 화력',
    },
    '분당': {
        'rated_capacity_mw': 890,       # 분당화력 설비용량 890MW
        'coal_gen_cost_per_mwh': 120_000,   # LNG 발전 원가
        'lng_gen_cost_per_mwh': 150_000,    # 대체 LNG 첨두 비용
        'sox_social_cost': 500,    # 환경부 대기배출부과금 원/kg
        'nox_social_cost': 2_130,
        'dust_social_cost': 770,
        'efficiency_slope': 0.0416,      # 실측 회귀: LNG 복합 (r=+0.54)
        'efficiency_intercept': 26.97,
        'efficiency_min': 0.25,
        'efficiency_max': 0.50,
        'utilization_min': 0.20,
        'utilization_max': 0.95,
        'active_targets': ['NOx'],
        'drop_features': [
            'precipitation', 'pressure_mean', 'solar_radiation',
            'solar_mwh', 'wind_mwh', 'renewable_ratio', 'sox_factor_ma30',
            'SOx_lag1', 'SOx_lag7', '먼지_lag1', '먼지_lag7',
            '유연탄', 'coal_ratio',
        ],
        'gen_mwh_src': 'api_gen_mwh',   # 분당은 API 실측 완전
        'plant_label': '분당 LNG복합',
    },
}

# ── 공통 기상 컬럼 목록 ──────────────────────────────────────
WEATHER_COLS = [
    'temp_mean', 'humidity_mean', 'wind_speed_mean', 'wind_speed_max',
    'wind_sin', 'wind_cos', 'precipitation', 'pressure_mean',
    'solar_radiation', 'stagnation_idx',
]


def run_plant_optimization(plant: str, cfg: dict):
    print(f'\n{"="*65}')
    print(f'  {cfg["plant_label"]} 최적화 분석')
    print(f'{"="*65}')

    report_dir = REPORTS_BASE / plant
    report_dir.mkdir(parents=True, exist_ok=True)

    # ── 모델 로딩 ─────────────────────────────────────────────
    model_path = report_dir / f'emission_model_{plant}.pkl'
    if not model_path.exists():
        print(f'[SKIP] 모델 없음: {model_path}')
        return None

    model = joblib.load(model_path)
    print(f'[OK] 모델 로딩: {model_path.name}')

    # ── 데이터 로딩 및 특성 생성 ──────────────────────────────
    master = pd.read_csv(PROCESSED_DIR / f'master_{plant}.csv', parse_dates=['date'])
    feature_df = build_features(master)

    active_targets = cfg['active_targets']
    feature_df = feature_df.dropna(subset=active_targets)

    X_full, y_full = get_X_y(feature_df)

    # 분당 제외 피처 제거
    drop = [c for c in cfg['drop_features'] if c in X_full.columns]
    X_full = X_full.drop(columns=drop)

    # 활성 타겟만 유지
    y_full = y_full[[t for t in active_targets if t in y_full.columns]]

    feature_template = X_full.mean().to_frame().T.reset_index(drop=True)
    print(f'  특성 수: {X_full.shape[1]}, 활성 타겟: {active_targets}')

    # ── 기준 운전 조건 ────────────────────────────────────────
    gen_src = cfg['gen_mwh_src']
    if gen_src in master.columns and master[gen_src].notna().any():
        avg_gen_mwh = master[gen_src].mean()
    else:
        avg_gen_mwh = master['gen_mwh'].mean()
    avg_utilization = master['utilization'].mean() / 100

    print(f'  현재 평균 발전량: {avg_gen_mwh:,.0f} MWh/일')
    print(f'  현재 평균 이용률: {avg_utilization*100:.1f}%')

    # 기상 평균 (특성 템플릿 기반)
    base_weather = X_full[[c for c in WEATHER_COLS if c in X_full.columns]].mean()

    # ── 이용률 그리드 서치 ────────────────────────────────────
    print('\n[1] 이용률 그리드 서치...')
    grid_df = optimize_utilization_grid(
        model=model,
        weather_row=base_weather,
        base_gen_mwh=avg_gen_mwh,
        base_utilization=avg_utilization,
        feature_template=feature_template,
        target_names=active_targets,
        step=0.05,
        plant_config=cfg,
    )

    # 그리드 서치 시각화
    fig, axes = plt.subplots(1, len(active_targets), figsize=(6 * len(active_targets), 5))
    if len(active_targets) == 1:
        axes = [axes]

    colors = ['#E74C3C', '#E67E22', '#8E44AD']
    for ax, col, color in zip(axes, active_targets, colors):
        ax.plot(grid_df['utilization'] * 100, grid_df[col] / 1000,
                'o-', color=color, lw=2, ms=5)
        min_idx = grid_df[col].idxmin()
        opt_util = grid_df.loc[min_idx, 'utilization'] * 100
        ax.axvline(x=opt_util, color='red', ls='--', lw=1.5, label=f'최적: {opt_util:.0f}%')
        ax.axvline(x=avg_utilization * 100, color='gray', ls=':', lw=1.5,
                   label=f'현재: {avg_utilization*100:.0f}%')
        ax.set_xlabel('이용률 (%)')
        ax.set_ylabel(f'{col} 배출량 (ton/일)')
        ax.set_title(f'{col} 배출 최적화', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle(f'{cfg["plant_label"]} 이용률 vs 배출량 (연평균 기상 조건)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(report_dir / 'opt_01_grid_search.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [저장] opt_01_grid_search.png')

    # ── 시나리오 분석 ─────────────────────────────────────────
    print('\n[2] 시나리오 분석...')
    scenario_df = run_scenario_analysis(
        model=model,
        base_weather=base_weather,
        base_gen_mwh=avg_gen_mwh,
        base_utilization=avg_utilization,
        feature_template=feature_template,
        target_names=active_targets,
        plant_config=cfg,
        report_dir=report_dir,
    )

    # 시나리오 시각화
    reduction_cols = [f'{t}_reduction_pct' for t in active_targets
                      if f'{t}_reduction_pct' in scenario_df.columns]
    if reduction_cols:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        ax = axes[0]
        x = np.arange(len(scenario_df))
        w = 0.8 / len(reduction_cols)
        bar_colors = ['#E74C3C', '#E67E22', '#8E44AD']
        for i, (rcol, color) in enumerate(zip(reduction_cols, bar_colors)):
            t_name = rcol.replace('_reduction_pct', '')
            bars = ax.bar(x + i*w - (len(reduction_cols)-1)*w/2,
                          scenario_df[rcol], w, label=t_name, color=color, alpha=0.8)
            for bar, val in zip(bars, scenario_df[rcol]):
                if abs(val) > 0.5:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                            f'{val:.1f}%', ha='center', va='bottom', fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(scenario_df['description'], rotation=12, ha='right', fontsize=8)
        ax.set_ylabel('배출량 감축률 (%)')
        ax.set_title('시나리오별 배출 감축 효과', fontweight='bold')
        ax.legend()
        ax.axhline(y=0, color='black', lw=0.8)
        ax.grid(axis='y', alpha=0.3)

        ax2 = axes[1]
        x2 = np.arange(len(scenario_df))
        ax2.bar(x2 - 0.2, scenario_df['current_utilization'] * 100, 0.4,
                label='현재 이용률', color='#3498DB', alpha=0.8)
        ax2.bar(x2 + 0.2, scenario_df['optimal_utilization'] * 100, 0.4,
                label='최적 이용률', color='#2ECC71', alpha=0.8)
        ax2.set_xticks(x2)
        ax2.set_xticklabels(scenario_df['description'], rotation=12, ha='right', fontsize=8)
        ax2.set_ylabel('이용률 (%)')
        ax2.set_title('시나리오별 현재 vs 최적 이용률', fontweight='bold')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        plt.suptitle(f'{cfg["plant_label"]} 시나리오별 최적 운전 분석',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(report_dir / 'opt_02_scenario_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  [저장] opt_02_scenario_analysis.png')

    # ── B/C 분석 ──────────────────────────────────────────────
    print('\n[3] B/C 분석...')
    bc_df = calc_bc_analysis(scenario_df, annual_operation_days=330, report_dir=report_dir)

    # B/C 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(bc_df))

    ax = axes[0]
    ax.bar(x - 0.2, bc_df['annual_benefit_억원'], 0.4,
           label='연간 사회적 편익', color='#2ECC71', alpha=0.85)
    ax.bar(x + 0.2, bc_df['annual_replacement_cost_억원'], 0.4,
           label='연간 대체 비용', color='#E74C3C', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(bc_df['description'], rotation=12, ha='right', fontsize=8)
    ax.set_ylabel('억원/년')
    ax.set_title('시나리오별 편익 vs 비용 (연간)', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    ax2 = axes[1]
    bar_colors2 = ['#2ECC71' if v >= 1.0 else '#E74C3C' for v in bc_df['BC_ratio']]
    bars = ax2.bar(x, bc_df['BC_ratio'], color=bar_colors2, alpha=0.85)
    ax2.axhline(y=1.0, color='black', ls='--', lw=1.5, label='B/C = 1.0 기준선')
    for bar, val in zip(bars, bc_df['BC_ratio']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(bc_df['description'], rotation=12, ha='right', fontsize=8)
    ax2.set_ylabel('B/C 비율')
    ax2.set_title('시나리오별 B/C 비율', fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle(f'{cfg["plant_label"]} B/C 분석', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(report_dir / 'opt_03_bc_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [저장] opt_03_bc_analysis.png')

    # ── 정책 권고 요약 출력 ───────────────────────────────────
    print(f'\n{"─"*65}')
    print(f'  {cfg["plant_label"]} 정책 권고 요약')
    print(f'{"─"*65}')

    STAGNATION_THRESHOLD = 34.5
    stagnation_vals = {
        'A_summer_stagnation':   88.0 / 1.0,
        'B_winter_cold':         45.0 / 7.0,
        'C_solar_max':           60.0 / 3.0,
        'D_wind_max':            65.0 / 9.0,
        'E_spring_dust':         72.0 / 2.5,
        'F_winter_stagnation':   83.0 / 1.0,
        'G_renewable_peak_spring': 58.0 / 5.5,
        'H_summer_peak':         87.0 / 1.2,
    }

    print(f'\n{"시나리오":<30} {"정체지수":>8} {"위험":>6} {"최적이용률":>10} {"NOx감축":>8} {"B/C":>6}')
    print('─' * 75)

    for _, row in scenario_df.iterrows():
        sc = row['scenario']
        stag = stagnation_vals.get(sc, 0)
        is_stag = stag > STAGNATION_THRESHOLD
        bc_row = bc_df[bc_df['scenario'] == sc]
        bc_val = bc_row['BC_ratio'].values[0] if not bc_row.empty else 0
        nox_red = row.get('NOx_reduction_pct', 0)

        print(f'{row["description"][:28]:<30} {stag:>8.1f} {"★" if is_stag else "○":>6}'
              f' {row["optimal_utilization"]*100:>9.0f}%'
              f' {nox_red:>7.1f}%'
              f' {bc_val:>6.2f}')

    print()
    return bc_df


def main():
    all_results = []

    for plant, cfg in PLANT_OPT_CONFIGS.items():
        bc_df = run_plant_optimization(plant, cfg)
        if bc_df is not None:
            bc_df.insert(0, 'plant', plant)
            all_results.append(bc_df)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(REPORTS_BASE / 'optimization_summary_all.csv',
                        index=False, encoding='utf-8-sig')
        print(f'\n[OK] 전체 최적화 결과 저장: reports/optimization_summary_all.csv')

    print('\n' + '='*65)
    print('  3사업소 최적화 분석 완료')
    print('='*65)


if __name__ == '__main__':
    main()
