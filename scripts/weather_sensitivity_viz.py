"""
기상 조건별 배출량 예측 감도 분석 시각화

목적: 기상 변수(기온·풍속·대기정체) 변화에 따른 배출량 예측 변동 정량화
     → 보고서 "기상 조건별 배출 예측" 근거 자료

출력:
  reports/{plant}/weather_01_sensitivity_curves.png  — 3변수 감도 곡선 + 시나리오 포인트
  reports/{plant}/weather_02_2d_nox_heatmap.png      — 기온×풍속 NOx 예측 히트맵
  reports/weather_sensitivity_summary.csv            — 변수별 NOx 예측 변동 폭 (정량)
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

matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

PROCESSED_DIR = Path(__file__).parent.parent / 'data' / 'processed'
REPORTS_BASE  = Path(__file__).parent.parent / 'reports'

DROP_BUNDANG = [
    'precipitation', 'pressure_mean', 'solar_radiation',
    'solar_mwh', 'wind_mwh', 'renewable_ratio', 'sox_factor_ma30',
    'SOx_lag1', 'SOx_lag7', '먼지_lag1', '먼지_lag7', '유연탄', 'coal_ratio',
]

PLANT_CONFIGS = {
    '삼천포': {'is_bundang': False, 'targets': ['SOx', 'NOx', '먼지']},
    '영흥':   {'is_bundang': False, 'targets': ['SOx', 'NOx', '먼지']},
    '분당':   {'is_bundang': True,  'targets': ['NOx']},
    '영동':   {'is_bundang': False, 'targets': ['SOx', 'NOx', '먼지']},
    '여수':   {'is_bundang': False, 'targets': ['SOx', 'NOx', '먼지']},
}

# 기상 변수 스위프 범위 (한반도 기후 실측 범위 기준)
SWEEP_VARS = {
    'temp_mean': {
        'range': np.linspace(-12, 33, 46),
        'label': '일평균 기온 (°C)',
        'unit': '°C',
        'recompute_stag': False,
    },
    'wind_speed_mean': {
        'range': np.linspace(0.3, 10.0, 46),
        'label': '일평균 풍속 (m/s)',
        'unit': 'm/s',
        'recompute_stag': True,   # 풍속 변화 → stagnation_idx 연동 재계산
    },
    'stagnation_idx': {
        'range': np.linspace(5, 150, 46),
        'label': '대기정체 지수 (습도/풍속)',
        'unit': '',
        'recompute_stag': False,
    },
}

# 4개 시나리오 포인트 오버레이
SCENARIOS = {
    'S0 정상':        {'temp_mean': None,  'wind_speed_mean': None, 'stagnation_idx': None},
    'SA 여름대기정체': {'temp_mean': 29.0,  'wind_speed_mean': 1.0,  'stagnation_idx': 88.0 / 1.0},
    'SE 봄황사':      {'temp_mean': 13.0,  'wind_speed_mean': 2.5,  'stagnation_idx': 72.0 / 2.5},
    'SF 겨울계절관리제': {'temp_mean': 2.0, 'wind_speed_mean': 1.0,  'stagnation_idx': 83.0 / 1.0},
}
SC_COLORS = ['#2980B9', '#E74C3C', '#F39C12', '#27AE60']

TARGET_COLORS = {'SOx': '#E74C3C', 'NOx': '#3498DB', '먼지': '#8E44AD'}
TARGET_LABELS = {'SOx': 'SOx (kg/일)', 'NOx': 'NOx (kg/일)', '먼지': '먼지 (kg/일)'}


def load_plant(plant, is_bundang, targets):
    model_path = REPORTS_BASE / plant / f'emission_model_{plant}.pkl'
    if not model_path.exists():
        return None, None, None
    model = joblib.load(model_path)
    master = pd.read_csv(PROCESSED_DIR / f'master_{plant}.csv', parse_dates=['date'])
    feat_df = build_features(master).dropna(subset=targets)
    X, _ = get_X_y(feat_df)
    if is_bundang:
        X = X.drop(columns=[c for c in DROP_BUNDANG if c in X.columns])
    base_row = X.mean().to_frame().T.reset_index(drop=True)
    return model, base_row


def predict_row(model, row, is_bundang, targets):
    pred = np.maximum(model.predict(row)[0], 0)
    if is_bundang:
        return {'NOx': float(pred[0] if hasattr(pred, '__len__') else pred)}
    return {t: float(pred[i]) for i, t in enumerate(targets) if i < len(pred)}


def sweep(model, base_row, var_name, vcfg, is_bundang, targets):
    results = {t: [] for t in targets}
    for val in vcfg['range']:
        row = base_row.copy()
        if var_name in row.columns:
            row[var_name] = val
        if vcfg['recompute_stag'] and 'stagnation_idx' in row.columns:
            hum = float(row['humidity_mean'].values[0]) if 'humidity_mean' in row.columns else 60.0
            row['stagnation_idx'] = hum / max(val, 0.1)
        em = predict_row(model, row, is_bundang, targets)
        for t in targets:
            results[t].append(em.get(t, 0.0))
    return results


# ── 사업소별 처리 ─────────────────────────────────────────────
print('=== 기상 감도 분석 시작 ===')
summary_rows = []

for plant, pcfg in PLANT_CONFIGS.items():
    is_bd   = pcfg['is_bundang']
    targets = pcfg['targets']
    report_dir = REPORTS_BASE / plant

    result = load_plant(plant, is_bd, targets)
    if result[0] is None:
        print(f'  [SKIP] {plant} 모델 없음')
        continue
    model, base_row = result
    print(f'\n[{plant}] 감도 분석...')

    # 시나리오 포인트 사전 계산
    sc_preds = {}
    for sc_name, sc_vals in SCENARIOS.items():
        row = base_row.copy()
        for k, v in sc_vals.items():
            if v is not None and k in row.columns:
                row[k] = v
        if 'stagnation_idx' in row.columns:
            hum = float(row['humidity_mean'].values[0]) if 'humidity_mean' in row.columns else 60.0
            ws  = max(float(row['wind_speed_mean'].values[0]), 0.1) if 'wind_speed_mean' in row.columns else 1.0
            row['stagnation_idx'] = hum / ws
        sc_preds[sc_name] = predict_row(model, row, is_bd, targets)

    # ── Fig 1: 감도 곡선 ──────────────────────────────────────
    n_var = len(SWEEP_VARS)
    n_tgt = len(targets)
    fig, axes = plt.subplots(n_var, n_tgt, figsize=(5 * n_tgt, 4 * n_var), squeeze=False)

    for row_i, (var_name, vcfg) in enumerate(SWEEP_VARS.items()):
        sweep_res = sweep(model, base_row, var_name, vcfg, is_bd, targets)
        base_val  = float(base_row[var_name].values[0]) if var_name in base_row.columns else None

        for col_i, tgt in enumerate(targets):
            ax = axes[row_i][col_i]
            vals = sweep_res[tgt]
            ax.plot(vcfg['range'], vals, color=TARGET_COLORS[tgt], lw=2)

            if base_val is not None:
                ax.axvline(base_val, color='gray', ls='--', lw=1.2,
                           label=f'현재 평균 {base_val:.1f}{vcfg["unit"]}')

            for (sc_name, sc_vals), sc_color in zip(SCENARIOS.items(), SC_COLORS):
                sc_x = sc_vals.get(var_name)
                if sc_x is None:
                    sc_x = base_val
                if sc_x is not None and vcfg['range'][0] <= sc_x <= vcfg['range'][-1]:
                    sc_y = np.interp(sc_x, vcfg['range'], vals)
                    ax.scatter(sc_x, sc_y, s=65, color=sc_color, zorder=5,
                               edgecolors='white', linewidths=0.8)
                    ax.annotate(sc_name.split(' ')[0],
                                xy=(sc_x, sc_y), xytext=(4, 4),
                                textcoords='offset points', fontsize=7, color=sc_color)

            delta = (max(vals) - min(vals)) / max(abs(np.mean(vals)), 1) * 100
            ax.text(0.97, 0.05, f'변동 {delta:.1f}%',
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
                    color=TARGET_COLORS[tgt],
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

            ax.set_xlabel(vcfg['label'], fontsize=10)
            ax.set_ylabel(TARGET_LABELS.get(tgt, tgt), fontsize=10)
            if row_i == 0:
                ax.set_title(tgt, fontsize=11, fontweight='bold', color=TARGET_COLORS[tgt])
            if base_val is not None and col_i == 0:
                ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

            summary_rows.append({
                'plant': plant,
                'variable': var_name,
                'target': tgt,
                'base_val': round(base_val, 2) if base_val is not None else None,
                'min_pred': round(min(vals), 1),
                'max_pred': round(max(vals), 1),
                'delta_kg': round(max(vals) - min(vals), 1),
                'delta_pct': round(delta, 1),
            })

    fig.suptitle(f'{plant} — 기상 조건별 배출량 예측 감도 분석\n'
                 f'(나머지 변수 = 학습데이터 평균 고정 | 점 = 시나리오 위치)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = report_dir / 'weather_01_sensitivity_curves.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [저장] {out.name}')

    # ── Fig 2: 기온 × 풍속 → NOx 2D 히트맵 ──────────────────
    temp_grid = np.linspace(-10, 33, 25)
    wind_grid = np.linspace(0.3, 10.0, 25)
    nox_map   = np.zeros((len(wind_grid), len(temp_grid)))

    for ti, temp in enumerate(temp_grid):
        for wi, wind in enumerate(wind_grid):
            row = base_row.copy()
            if 'temp_mean'       in row.columns: row['temp_mean'] = temp
            if 'wind_speed_mean' in row.columns: row['wind_speed_mean'] = wind
            if 'stagnation_idx'  in row.columns:
                hum = float(row['humidity_mean'].values[0]) if 'humidity_mean' in row.columns else 60.0
                row['stagnation_idx'] = hum / max(wind, 0.1)
            nox_map[wi, ti] = predict_row(model, row, is_bd, targets).get('NOx', 0.0)

    fig, ax = plt.subplots(figsize=(9, 6))
    c  = ax.contourf(temp_grid, wind_grid, nox_map, levels=15, cmap='RdYlGn_r')
    ax.contour(temp_grid, wind_grid, nox_map, levels=8,
               colors='white', alpha=0.4, linewidths=0.6)
    cb = fig.colorbar(c, ax=ax)
    cb.set_label('NOx 예측 (kg/일)', fontsize=11)

    if 'temp_mean' in base_row.columns and 'wind_speed_mean' in base_row.columns:
        bx = float(base_row['temp_mean'].values[0])
        by = float(base_row['wind_speed_mean'].values[0])
        ax.scatter(bx, by, s=150, c='blue', marker='D', zorder=5,
                   edgecolors='white', lw=1, label=f'현재 평균 ({bx:.1f}°C, {by:.1f} m/s)')

    for (sc_name, sc_vals), sc_color in zip(SCENARIOS.items(), SC_COLORS):
        sx = sc_vals.get('temp_mean')
        sy = sc_vals.get('wind_speed_mean')
        if sx is None: sx = float(base_row['temp_mean'].values[0]) if 'temp_mean' in base_row.columns else 15
        if sy is None: sy = float(base_row['wind_speed_mean'].values[0]) if 'wind_speed_mean' in base_row.columns else 3
        if temp_grid[0] <= sx <= temp_grid[-1] and wind_grid[0] <= sy <= wind_grid[-1]:
            ax.scatter(sx, sy, s=120, c=sc_color, marker='*', zorder=6,
                       edgecolors='black', lw=0.5, label=sc_name)

    ax.set_xlabel('일평균 기온 (°C)', fontsize=12)
    ax.set_ylabel('일평균 풍속 (m/s)', fontsize=12)
    ax.set_title(f'{plant} — 기온 × 풍속 조합별 NOx 예측\n'
                 f'(우하단 고온+저풍속 = 대기정체 위험지대)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.2)
    plt.tight_layout()
    out2 = report_dir / 'weather_02_2d_nox_heatmap.png'
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [저장] {out2.name}')

# ── 요약 CSV 저장 ─────────────────────────────────────────────
summary_df = pd.DataFrame(summary_rows)
out_csv = REPORTS_BASE / 'weather_sensitivity_summary.csv'
summary_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f'\n[OK] 요약 저장: {out_csv}')

# 보고서 기재용 정량 수치
print('\n=== NOx 기상 감도 정량 요약 (보고서 기재용) ===')
nox_df = summary_df[summary_df['target'] == 'NOx'][
    ['plant', 'variable', 'base_val', 'min_pred', 'max_pred', 'delta_kg', 'delta_pct']
].reset_index(drop=True)
print(nox_df.to_string(index=False))

print('\n=== 완료 ===')
