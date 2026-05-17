"""
시나리오 분석 + B/C 차트 재생성
- 시나리오 차트: X축 레이블 45° 회전, 폰트 크기 개선
- B/C 차트: 편익과 비용을 로그 스케일로 비교, 가독성 개선
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
from matplotlib import font_manager
from pathlib import Path

# 한글 폰트
for fp in font_manager.findSystemFonts():
    if any(k in fp for k in ['AppleGothic', 'NanumGothic', 'Malgun', 'NotoSansCJK']):
        font_manager.fontManager.addfont(fp)
        break
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

REPORTS_BASE  = Path(__file__).parent.parent / 'reports'
FINAL_4       = Path(__file__).parent.parent / 'reports' / 'final_report' / '4_인사이트'
FINAL_5       = Path(__file__).parent.parent / 'reports' / 'final_report' / '5_정책활용'

# 시나리오 짧은 레이블 매핑
SHORT_LABELS = {
    'A_summer_stagnation':     'A\n여름대기정체',
    'B_winter_cold':           'B\n겨울저온',
    'C_solar_max':             'C\n태양광피크',
    'D_wind_max':              'D\n강풍',
    'E_spring_dust':           'E\n봄황사',
    'F_winter_stagnation':     'F\n겨울대기정체',
    'G_renewable_peak_spring': 'G\n봄신재생',
    'H_summer_peak':           'H\n여름피크',
}

PLANT_CONFIGS = [
    {'plant': '삼천포', 'label': '삼천포 유연탄', 'targets': ['SOx', 'NOx', '먼지'],
     'bar_colors': ['#E74C3C', '#E67E22', '#8E44AD']},
    {'plant': '영흥',   'label': '영흥 유연탄',   'targets': ['SOx', 'NOx', '먼지'],
     'bar_colors': ['#E74C3C', '#E67E22', '#8E44AD']},
]

for cfg in PLANT_CONFIGS:
    plant       = cfg['plant']
    plant_label = cfg['label']
    bar_colors  = cfg['bar_colors']
    report_dir  = REPORTS_BASE / plant

    sc_df = pd.read_csv(report_dir / 'scenario_results.csv')
    bc_df = pd.read_csv(report_dir / 'bc_analysis.csv')

    short_x = [SHORT_LABELS.get(s, s) for s in sc_df['scenario']]

    # ── 시나리오 감축률 차트 ──────────────────────────────────────
    reduction_cols = [c for c in sc_df.columns if c.endswith('_reduction_pct')]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'{plant_label} 시나리오별 최적 운전 분석', fontsize=14, fontweight='bold')

    # 왼쪽: 감축률
    ax = axes[0]
    x = np.arange(len(sc_df))
    w = 0.75 / len(reduction_cols)
    for i, (rcol, color) in enumerate(zip(reduction_cols, bar_colors)):
        t_name = rcol.replace('_reduction_pct', '')
        bars = ax.bar(x + i*w - (len(reduction_cols)-1)*w/2,
                      sc_df[rcol], w, label=t_name, color=color, alpha=0.85)
        for bar, val in zip(bars, sc_df[rcol]):
            if abs(val) > 1.0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.4,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(short_x, fontsize=9.5, ha='center')
    ax.set_ylabel('배출량 감축률 (%)', fontsize=11)
    ax.set_title('시나리오별 배출 감축 효과', fontweight='bold', fontsize=12)
    ax.legend(fontsize=10)
    ax.axhline(y=0, color='black', lw=0.8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=min(sc_df[reduction_cols].min().min() - 5, -5))

    # 오른쪽: 현재 vs 최적 이용률
    ax2 = axes[1]
    x2 = np.arange(len(sc_df))
    bars1 = ax2.bar(x2 - 0.2, sc_df['current_utilization'] * 100, 0.38,
                    label='현재 이용률', color='#3498DB', alpha=0.85)
    bars2 = ax2.bar(x2 + 0.2, sc_df['optimal_utilization'] * 100, 0.38,
                    label='최적 이용률', color='#2ECC71', alpha=0.85)
    for bar in bars1:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{bar.get_height():.0f}%', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{bar.get_height():.0f}%', ha='center', va='bottom', fontsize=8)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(short_x, fontsize=9.5, ha='center')
    ax2.set_ylabel('이용률 (%)', fontsize=11)
    ax2.set_title('시나리오별 현재 vs 최적 이용률', fontweight='bold', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = FINAL_4 / f'opt_{plant}_시나리오.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[저장] {out}')

    # ── B/C 차트 ─────────────────────────────────────────────────
    bc_x = [SHORT_LABELS.get(s, s) for s in bc_df['scenario']]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'{plant_label} B/C 분석', fontsize=14, fontweight='bold')

    # 왼쪽: 편익 vs 비용 (절대값, 막대)
    ax = axes[0]
    x = np.arange(len(bc_df))
    bars_b = ax.bar(x - 0.2, bc_df['annual_benefit_억원'], 0.38,
                    label='연간 사회적 편익', color='#2ECC71', alpha=0.85)
    bars_c = ax.bar(x + 0.2, bc_df['annual_replacement_cost_억원'], 0.38,
                    label='연간 대체 비용', color='#E74C3C', alpha=0.85)
    for bar in bars_b:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}억', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(bc_x, fontsize=9.5, ha='center')
    ax.set_ylabel('억원/년', fontsize=11)
    ax.set_title('시나리오별 편익 vs 비용 (연간)', fontweight='bold', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')
    ax.set_ylim(1, bc_df['annual_replacement_cost_억원'].max() * 3)

    # 오른쪽: NOx 감축량 (kg/일) — B/C 비율이 모두 비슷해 NOx 효과로 대체
    sc_merge = sc_df.merge(bc_df[['scenario','BC_ratio']], on='scenario', how='left')
    nox_savings = (sc_merge['NOx_current'] - sc_merge['NOx_optimal']).values

    ax2 = axes[1]
    colors2 = ['#2ECC71'] * len(nox_savings)
    bars3 = ax2.bar(np.arange(len(sc_merge)), nox_savings / 1000, color=colors2, alpha=0.85)
    for bar, pct in zip(bars3, sc_merge['NOx_reduction_pct'].values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.set_xticks(np.arange(len(sc_merge)))
    ax2.set_xticklabels(bc_x, fontsize=9.5, ha='center')
    ax2.set_ylabel('NOx 감축량 (ton/일)', fontsize=11)
    ax2.set_title('시나리오별 NOx 일 감축량', fontweight='bold', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_bc = FINAL_5 / f'bc_{plant}.png'
    plt.savefig(out_bc, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[저장] {out_bc}')

print('\n완료')
