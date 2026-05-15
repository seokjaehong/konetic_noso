"""
발주자 요구 형식 최종 시나리오 표 생성 (현행 vs 최적 비교)

출력:
  reports/scenario_final_table.csv   — 기계 처리용
  reports/scenario_final_table.md    — 보고서 삽입용 Markdown
  reports/scenario_final_summary.png — 현행 vs 최적 시각화

표 구성:
  [현행 운영]  비계절관리제(4~11월) 실적 평균
  [현행 운영]  계절관리제(12~3월)   실적 평균
  [최적 권고]  비계절관리제 최적 배분 (시나리오 A/B/C 공통)
  [최적 권고]  계절관리제 최적 배분  (시나리오 D)
"""

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fuel_mix_optimizer import PLANT_CONSTRAINTS, SOCIAL_COSTS

matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

REPORTS_BASE  = _ROOT / 'reports'
PROCESSED_DIR = _ROOT / 'data' / 'processed'

SAM = PLANT_CONSTRAINTS['삼천포']
YH  = PLANT_CONSTRAINTS['영흥']
BD  = PLANT_CONSTRAINTS['분당']
DEMAND_BASE = SAM['base_gen_mwh'] + YH['base_gen_mwh'] + BD['base_gen_mwh']


def estimate_efficiency(util_pct: float, cfg: dict) -> float:
    s  = cfg.get('eff_slope', 0.3111)
    i  = cfg.get('eff_intercept', 14.62)
    lo = cfg.get('eff_min', 0.20) * 100
    hi = cfg.get('eff_max', 0.45) * 100
    return max(lo, min(hi, s * util_pct + i))


def social_cost_억(sox_kg, nox_kg, dust_kg):
    return (sox_kg  * SOCIAL_COSTS['SOx'] +
            nox_kg  * SOCIAL_COSTS['NOx'] +
            dust_kg * SOCIAL_COSTS['먼지']) / 1e8


def make_row(label, desc, category,
             sam_util, yh_util, bd_util,
             gen_sam, gen_yh, gen_bd,
             sox_kg, nox_kg, dust_kg,
             eff_sam_pct, eff_yh_pct,
             baseline_sc_억=None):

    gen_total = gen_sam + gen_yh + gen_bd
    coal_pct  = (gen_sam + gen_yh) / max(gen_total, 1) * 100
    lng_pct   = gen_bd / max(gen_total, 1) * 100

    # 가중 열효율
    coal_gen = gen_sam + gen_yh
    eff_coal = (gen_sam * eff_sam_pct + gen_yh * eff_yh_pct) / max(coal_gen, 1)
    eff_bd   = estimate_efficiency(bd_util, BD)
    eff_total = (gen_sam * eff_sam_pct + gen_yh * eff_yh_pct + gen_bd * eff_bd) / max(gen_total, 1)

    sc_억 = social_cost_억(sox_kg, nox_kg, dust_kg)
    saving = (baseline_sc_억 - sc_억) if baseline_sc_억 is not None else 0.0

    return {
        '구분':            category,
        '시나리오':        label,
        '설명':            desc,
        '삼천포 이용률(%)': round(sam_util, 1),
        '영흥 이용률(%)':   round(yh_util, 1),
        '분당 이용률(%)':   round(bd_util, 1),
        '총발전량(MWh)':    round(gen_total, 0),
        '석탄비중(%)':      round(coal_pct, 1),
        'LNG비중(%)':       round(lng_pct, 1),
        '예상_SOx(ton/일)': round(sox_kg / 1000, 2),
        '예상_NOx(ton/일)': round(nox_kg / 1000, 2),
        '예상_먼지(ton/일)': round(dust_kg / 1000, 2),
        '삼천포열효율(%)':  round(eff_sam_pct, 1),
        '영흥열효율(%)':    round(eff_yh_pct, 1),
        '가중열효율(%)':    round(eff_total, 1),
        '부과금(억원/일)':  round(sc_억, 3),
        '절감(억원/일)':    round(saving, 3),
    }


# ══════════════════════════════════════════════════════════════
# 1. 현행 운영 실적 — master CSV에서 직접 계산
# ══════════════════════════════════════════════════════════════
sam_m = pd.read_csv(PROCESSED_DIR / 'master_삼천포.csv', parse_dates=['date'])
yh_m  = pd.read_csv(PROCESSED_DIR / 'master_영흥.csv',   parse_dates=['date'])
bd_m  = pd.read_csv(PROCESSED_DIR / 'master_분당.csv',    parse_dates=['date'])

for df in [sam_m, yh_m, bd_m]:
    df['seasonal_mgmt'] = df['date'].dt.month.isin([12, 1, 2, 3]).astype(int)

def actual_stats(df, sm_flag, gen_col='api_gen_mwh'):
    sub = df[df['seasonal_mgmt'] == sm_flag].copy()
    gen_src = gen_col if gen_col in sub.columns else 'gen_mwh'
    return {
        'util':  sub['utilization'].mean(),
        'gen':   sub[gen_src].mean() if gen_src in sub.columns else sub['gen_mwh'].mean(),
        'NOx':   sub['NOx'].mean()  if 'NOx'  in sub.columns else 0,
        'SOx':   sub['SOx'].mean()  if 'SOx'  in sub.columns else 0,
        'dust':  sub['먼지'].mean() if '먼지' in sub.columns else 0,
        'eff':   sub['heat_efficiency'].mean() if 'heat_efficiency' in sub.columns else 0,
    }

rows = []

for sm_flag, sm_label, sm_desc in [
    (0, '현행 — 비계절관리제', '4~11월 실적 평균 (2020~2026)'),
    (1, '현행 — 계절관리제',  '12~3월 실적 평균 (2020~2026)'),
]:
    s = actual_stats(sam_m, sm_flag)
    y = actual_stats(yh_m,  sm_flag)
    b = actual_stats(bd_m,  sm_flag, gen_col='api_gen_mwh')

    # 3사업소 합산 일 배출량
    sox_total  = s['SOx']  + y['SOx']  + b['SOx']
    nox_total  = s['NOx']  + y['NOx']  + b['NOx']
    dust_total = s['dust'] + y['dust'] + b['dust']

    row = make_row(
        label=sm_label, desc=sm_desc, category='현행',
        sam_util=s['util'], yh_util=y['util'], bd_util=b['util'],
        gen_sam=s['gen'],   gen_yh=y['gen'],   gen_bd=b['gen'],
        sox_kg=sox_total, nox_kg=nox_total, dust_kg=dust_total,
        eff_sam_pct=s['eff'], eff_yh_pct=y['eff'],
        baseline_sc_억=None,
    )
    rows.append(row)

# 현행 비계절관리제 기준 사회적 비용 (절감 기준값)
baseline_non_sm_sc = rows[0]['부과금(억원/일)']
baseline_sm_sc     = rows[1]['부과금(억원/일)']
# 절감 계산을 위해 현행 행에도 절감=0 명시
rows[0]['절감(억원/일)'] = 0.0
rows[1]['절감(억원/일)'] = 0.0


# ══════════════════════════════════════════════════════════════
# 2. 최적 권고 시나리오 — fuel_mix 그리드 결과에서
# ══════════════════════════════════════════════════════════════
grid_path = REPORTS_BASE / 'fuel_mix_optimization_results.csv'
if not grid_path.exists():
    print('[ERROR] fuel_mix_optimization_results.csv 없음')
    exit(1)

grid_all = pd.read_csv(grid_path)
sc_sum   = pd.read_csv(REPORTS_BASE / 'fuel_mix_scenario_summary.csv')

# 비계절관리제 대표: S0 최적 (= SA, SE와 동일 최적 조합)
for sc_key, label, desc, category, baseline_sc in [
    ('S0_normal',           '최적 — 비계절관리제', '삼천포 36%↓ + 영흥 94%↑ (정상·여름·봄 공통 최적)', '최적 권고', baseline_non_sm_sc),
    ('SF_winter_stagnation','최적 — 계절관리제',   '삼천포 35%↓ + 영흥 80% + 분당 LNG 확대',            '최적 권고', baseline_sm_sc),
]:
    sc_grid = grid_all[grid_all['scenario'] == sc_key]
    if sc_grid.empty:
        continue
    best = sc_grid.sort_values('social_cost_krw').iloc[0]

    sam_util = float(best['sam_util']) * 100
    yh_util  = float(best['yh_util'])  * 100
    gen_sam  = float(best['gen_sam'])
    gen_yh   = float(best['gen_yh'])
    gen_bd   = float(best['gen_bd'])
    bd_util  = gen_bd / (BD['gen_max_mwh']) * 100

    row = make_row(
        label=label, desc=desc, category=category,
        sam_util=sam_util, yh_util=yh_util, bd_util=bd_util,
        gen_sam=gen_sam, gen_yh=gen_yh, gen_bd=gen_bd,
        sox_kg=float(best['SOx_total']),
        nox_kg=float(best['NOx_total']),
        dust_kg=float(best['dust_total']),
        eff_sam_pct=estimate_efficiency(sam_util, SAM),
        eff_yh_pct=estimate_efficiency(yh_util, YH),
        baseline_sc_억=baseline_sc,
    )
    rows.append(row)

df_table = pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# 3. 출력
# ══════════════════════════════════════════════════════════════
df_table.to_csv(REPORTS_BASE / 'scenario_final_table.csv', index=False, encoding='utf-8-sig')
print('[OK] scenario_final_table.csv')

# ── Markdown ─────────────────────────────────────────────────
md = ['\n### 현행 운영 vs 최적 권고 시나리오 비교\n']
md.append('| 구분 | 시나리오 | 삼천포% | 영흥% | 분당% | 총발전량 | 석탄비중 | LNG비중 | NOx(ton/일) | SOx(ton/일) | 먼지(ton/일) | 가중열효율% | 부과금(억원/일) | 절감(억원/일) |')
md.append('|------|----------|:------:|:----:|:----:|:-------:|:-------:|:------:|:-----------:|:-----------:|:-----------:|:-----------:|:--------------:|:------------:|')
for _, r in df_table.iterrows():
    cat_mark = '🔵' if r['구분'] == '현행' else '🟢'
    md.append(
        f'| {cat_mark} **{r["구분"]}** | {r["시나리오"]} '
        f'| {r["삼천포 이용률(%)"]:.1f}% | {r["영흥 이용률(%)"]:.1f}% | {r["분당 이용률(%)"]:.1f}% '
        f'| {r["총발전량(MWh)"]:,.0f} | {r["석탄비중(%)"]:.1f}% | {r["LNG비중(%)"]:.1f}% '
        f'| **{r["예상_NOx(ton/일)"]:.2f}** | {r["예상_SOx(ton/일)"]:.2f} | {r["예상_먼지(ton/일)"]:.2f} '
        f'| {r["가중열효율(%)"]:.1f}% | {r["부과금(억원/일)"]:.3f} | **{r["절감(억원/일)"]:.3f}** |'
    )

with open(REPORTS_BASE / 'scenario_final_table.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(md))
print('[OK] scenario_final_table.md')

# ── 시각화 ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('현행 운영 vs 최적 권고: 비계절관리제(4~11월) · 계절관리제(12~3월)', fontsize=14, fontweight='bold')

labels = [r['시나리오'] for _, r in df_table.iterrows()]
short  = ['현행\n비계절', '현행\n계절', '최적\n비계절', '최적\n계절']
x = np.arange(4)
colors = ['#95A5A6', '#7F8C8D', '#27AE60', '#2980B9']

def bar_plot(ax, values, title, ylabel, fmt='{:.2f}'):
    bars = ax.bar(x, values, color=colors, alpha=0.85, edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                fmt.format(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=10)
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_ylabel(ylabel); ax.grid(axis='y', alpha=0.3)

nox_vals  = df_table['예상_NOx(ton/일)'].tolist()
sox_vals  = df_table['예상_SOx(ton/일)'].tolist()
dust_vals = df_table['예상_먼지(ton/일)'].tolist()
eff_vals  = df_table['가중열효율(%)'].tolist()
coal_vals = df_table['석탄비중(%)'].tolist()
lng_vals  = df_table['LNG비중(%)'].tolist()

bar_plot(axes[0,0], nox_vals,  'NOx 배출량 (ton/일)',    'ton/일')
bar_plot(axes[0,1], sox_vals,  'SOx 배출량 (ton/일)',    'ton/일')
bar_plot(axes[0,2], dust_vals, '먼지 배출량 (ton/일)',   'ton/일')
bar_plot(axes[1,0], eff_vals,  '가중 열효율 (%)',        '%', fmt='{:.1f}%')

# 연료 믹스 누적 막대
ax = axes[1,1]
ax.bar(x, coal_vals, color='#E74C3C', alpha=0.8, label='석탄(%)', edgecolor='white')
ax.bar(x, lng_vals,  bottom=coal_vals, color='#3498DB', alpha=0.8, label='LNG(%)', edgecolor='white')
for i, (c, l) in enumerate(zip(coal_vals, lng_vals)):
    ax.text(i, c/2, f'{c:.1f}%', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    ax.text(i, c + l/2, f'{l:.1f}%', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(short, fontsize=10)
ax.set_title('연료 믹스 (석탄 vs LNG)', fontweight='bold', fontsize=11)
ax.set_ylabel('발전 비중 (%)'); ax.legend(); ax.grid(axis='y', alpha=0.3)

# 절감액
ax = axes[1,2]
saving_vals = df_table['절감(억원/일)'].tolist()
save_colors = ['#BDC3C7', '#BDC3C7', '#27AE60', '#2980B9']
bars = ax.bar(x, saving_vals, color=save_colors, alpha=0.85, edgecolor='white', linewidth=1.2)
for bar, val in zip(bars, saving_vals):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.3f}억', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#27AE60')
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(short, fontsize=10)
ax.set_title('부과금 기준 절감액 (억원/일)', fontweight='bold', fontsize=11)
ax.set_ylabel('억원/일'); ax.grid(axis='y', alpha=0.3)

# 범례 추가
from matplotlib.patches import Patch
legend_els = [Patch(facecolor=colors[0], label='현행 비계절관리제'),
              Patch(facecolor=colors[1], label='현행 계절관리제'),
              Patch(facecolor=colors[2], label='최적 비계절관리제'),
              Patch(facecolor=colors[3], label='최적 계절관리제')]
fig.legend(handles=legend_els, loc='lower center', ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(REPORTS_BASE / 'scenario_final_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print('[OK] scenario_final_summary.png')

# ── 콘솔 요약 ────────────────────────────────────────────────
print()
print('='*95)
print('  현행 운영 vs 최적 권고 — 핵심 비교')
print('='*95)
cols = ['구분','시나리오','삼천포 이용률(%)','영흥 이용률(%)','분당 이용률(%)',
        '석탄비중(%)','LNG비중(%)','예상_NOx(ton/일)','예상_SOx(ton/일)','가중열효율(%)','절감(억원/일)']
print(df_table[cols].to_string(index=False))
print('='*95)
print()

# ── 개선 효과 요약 ────────────────────────────────────────────
non_sm_current = df_table[df_table['시나리오'] == '현행 — 비계절관리제'].iloc[0]
non_sm_opt     = df_table[df_table['시나리오'] == '최적 — 비계절관리제'].iloc[0]
sm_current     = df_table[df_table['시나리오'] == '현행 — 계절관리제'].iloc[0]
sm_opt         = df_table[df_table['시나리오'] == '최적 — 계절관리제'].iloc[0]

print('[ 비계절관리제 (4~11월) 개선 효과 ]')
print(f'  NOx:  {non_sm_current["예상_NOx(ton/일)"]:.2f} → {non_sm_opt["예상_NOx(ton/일)"]:.2f} ton/일'
      f'  ({(non_sm_opt["예상_NOx(ton/일)"]/non_sm_current["예상_NOx(ton/일)"]-1)*100:.1f}%)')
print(f'  SOx:  {non_sm_current["예상_SOx(ton/일)"]:.2f} → {non_sm_opt["예상_SOx(ton/일)"]:.2f} ton/일'
      f'  ({(non_sm_opt["예상_SOx(ton/일)"]/non_sm_current["예상_SOx(ton/일)"]-1)*100:.1f}%)')
print(f'  열효율: {non_sm_current["가중열효율(%)"]:.1f}% → {non_sm_opt["가중열효율(%)"]:.1f}%'
      f'  ({non_sm_opt["가중열효율(%)"]-non_sm_current["가중열효율(%)"]:.1f}%p)')
print(f'  절감:  0.000 → {non_sm_opt["절감(억원/일)"]:.3f} 억원/일')
print()
print('[ 계절관리제 (12~3월) 개선 효과 ]')
print(f'  NOx:  {sm_current["예상_NOx(ton/일)"]:.2f} → {sm_opt["예상_NOx(ton/일)"]:.2f} ton/일'
      f'  ({(sm_opt["예상_NOx(ton/일)"]/sm_current["예상_NOx(ton/일)"]-1)*100:.1f}%)')
print(f'  SOx:  {sm_current["예상_SOx(ton/일)"]:.2f} → {sm_opt["예상_SOx(ton/일)"]:.2f} ton/일'
      f'  ({(sm_opt["예상_SOx(ton/일)"]/sm_current["예상_SOx(ton/일)"]-1)*100:.1f}%)')
print(f'  열효율: {sm_current["가중열효율(%)"]:.1f}% → {sm_opt["가중열효율(%)"]:.1f}%'
      f'  ({sm_opt["가중열효율(%)"]-sm_current["가중열효율(%)"]:.1f}%p)')
print(f'  절감:  0.000 → {sm_opt["절감(억원/일)"]:.3f} 억원/일')
