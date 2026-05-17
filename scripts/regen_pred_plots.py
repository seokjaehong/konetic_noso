"""
예측 vs 실측 산점도 재생성 스크립트
현재 PKL 모델 기준으로 최신 R² 값을 반영한 차트 생성
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent / 'src'))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import joblib
from pathlib import Path
from sklearn.metrics import r2_score

from features import build_features, get_X_y, train_test_split_temporal

# 한글 폰트
for fp in font_manager.findSystemFonts():
    if any(k in fp for k in ['AppleGothic', 'NanumGothic', 'Malgun', 'NotoSansCJK']):
        font_manager.fontManager.addfont(fp)
        break
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

PROCESSED_DIR = Path(__file__).parent.parent / 'data' / 'processed'
REPORTS_BASE  = Path(__file__).parent.parent / 'reports'
FINAL_REPORT  = Path(__file__).parent.parent / 'reports' / 'final_report' / '3_모델링'

BUNDANG_DROP_COLS = [
    'precipitation', 'pressure_mean', 'solar_radiation',
    'solar_mwh', 'wind_mwh', 'renewable_ratio',
    'sox_factor_ma30',
    'SOx_lag1', 'SOx_lag7', '먼지_lag1', '먼지_lag7',
    '유연탄', 'coal_ratio',
]

CONFIGS = [
    {'plant': '삼천포', 'is_bundang': False, 'target_cols': ['SOx', 'NOx', '먼지'],
     'colors': ['#E74C3C', '#E67E22', '#9B59B6']},
    {'plant': '영흥',   'is_bundang': False, 'target_cols': ['SOx', 'NOx', '먼지'],
     'colors': ['#E74C3C', '#E67E22', '#9B59B6']},
    {'plant': '분당',   'is_bundang': True,  'target_cols': ['NOx'],
     'colors': ['#E74C3C']},
]

for cfg in CONFIGS:
    plant       = cfg['plant']
    is_bundang  = cfg['is_bundang']
    target_cols = cfg['target_cols']
    colors      = cfg['colors']

    print(f'\n--- {plant} ---')

    master = pd.read_csv(PROCESSED_DIR / f'master_{plant}.csv', parse_dates=['date'])
    fd = build_features(master).dropna(subset=target_cols).reset_index(drop=True)
    _, _, test_df = train_test_split_temporal(fd)

    X_te, y_te = get_X_y(test_df)
    if is_bundang:
        drop = [c for c in BUNDANG_DROP_COLS if c in X_te.columns]
        X_te = X_te.drop(columns=drop)
    y_te = y_te[[c for c in target_cols if c in y_te.columns]]

    mask = (~X_te.isnull().any(axis=1)) & (~y_te.isnull().any(axis=1))
    X_te, y_te = X_te[mask], y_te[mask]

    model = joblib.load(REPORTS_BASE / plant / f'emission_model_{plant}.pkl')
    y_pred_arr = model.predict(X_te)
    if y_pred_arr.ndim == 1:
        y_pred_arr = y_pred_arr.reshape(-1, 1)

    ncols = len(target_cols)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    if ncols == 1:
        axes = [axes]
    fig.suptitle(f'{plant} 예측 vs 실측', fontsize=14, fontweight='bold')

    for i, (col, color) in enumerate(zip(target_cols, colors)):
        ax = axes[i]
        actual = y_te[col].values
        pred   = y_pred_arr[:, i]

        r2 = r2_score(actual, pred)
        print(f'  {col} R²={r2:.4f}')

        # 단위: kg/일 → ton/일 로 변환 (보고서 가독성)
        actual_t = actual / 1000
        pred_t   = pred   / 1000

        lim_min = min(actual_t.min(), pred_t.min()) * 0.95
        lim_max = max(actual_t.max(), pred_t.max()) * 1.05

        ax.scatter(actual_t, pred_t, alpha=0.5, s=20, color=color)
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=1)
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_xlabel('실측 (ton/일)', fontsize=11)
        ax.set_ylabel('예측 (ton/일)', fontsize=11)
        ax.set_title(f'{col} R²={r2:.3f}', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    out_path = FINAL_REPORT / f'model_{plant}_예측vs실측.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [저장] {out_path}')

print('\n완료')
