"""
3사업소 배출량 예측 모델 학습 스크립트

삼천포/영흥: MultiOutput XGBoost (SOx, NOx, 먼지)
분당: SingleOutput XGBoost (NOx만)
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent / 'src'))

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from features import build_features, get_X_y, train_test_split_temporal
from model import train_model, evaluate_model, explain_shap

PROCESSED_DIR = Path(__file__).parent.parent / 'data' / 'processed'
REPORTS_BASE  = Path(__file__).parent.parent / 'reports'

# 분당 제외 특성 (LNG 발전소)
BUNDANG_DROP_COLS = [
    'precipitation', 'pressure_mean', 'solar_radiation',
    'solar_mwh', 'wind_mwh', 'renewable_ratio',
    'sox_factor_ma30',
    'SOx_lag1', 'SOx_lag7', '먼지_lag1', '먼지_lag7',
    '유연탄', 'coal_ratio',
]

CONFIGS = [
    {'plant': '삼천포', 'is_bundang': False, 'target_cols': ['SOx', 'NOx', '먼지']},
    {'plant': '영흥',   'is_bundang': False, 'target_cols': ['SOx', 'NOx', '먼지']},
    {'plant': '분당',   'is_bundang': True,  'target_cols': ['NOx']},
    # 영동·여수: 극단 이상치(SOx 왜도31, 여수 NOx max=51402)가 학습을 망가뜨림
    # → 학습 데이터 내 99th pct 윈저라이징으로 이상치 억제
    {'plant': '영동',   'is_bundang': False, 'target_cols': ['SOx', 'NOx', '먼지'], 'winsorize_pct': 0.99},
    {'plant': '여수',   'is_bundang': False, 'target_cols': ['SOx', 'NOx', '먼지'], 'winsorize_pct': 0.99},
]

all_metrics = []

for cfg in CONFIGS:
    plant       = cfg['plant']
    is_bundang  = cfg['is_bundang']
    target_cols = cfg['target_cols']

    print(f'\n{"="*65}')
    print(f'  {plant} 모델 학습')
    print(f'{"="*65}')

    report_dir = REPORTS_BASE / plant
    report_dir.mkdir(parents=True, exist_ok=True)

    # 데이터 로딩
    master = pd.read_csv(PROCESSED_DIR / f'master_{plant}.csv', parse_dates=['date'])
    feature_df = build_features(master)
    feature_df = feature_df.dropna(subset=target_cols)

    X_full, y_full = get_X_y(feature_df)

    # 분당 불필요 특성 제거
    if is_bundang:
        drop = [c for c in BUNDANG_DROP_COLS if c in X_full.columns]
        X_full = X_full.drop(columns=drop)
        y_full = y_full[target_cols]

    # 타겟 컬럼 필터
    y_full = y_full[[c for c in target_cols if c in y_full.columns]]

    print(f'  특성 수: {X_full.shape[1]}, 샘플 수: {len(X_full)}')
    print(f'  타겟: {y_full.columns.tolist()}')
    print(f'  타겟 평균: {y_full.mean().round(1).to_dict()}')

    # 시계열 분할
    feature_df_clean = feature_df.dropna(subset=target_cols).reset_index(drop=True)
    train_df, val_df, test_df = train_test_split_temporal(feature_df_clean)

    def get_Xy(df):
        X, y = get_X_y(df)
        if is_bundang:
            drop = [c for c in BUNDANG_DROP_COLS if c in X.columns]
            X = X.drop(columns=drop)
        y = y[[c for c in target_cols if c in y.columns]]
        return X, y

    X_tr, y_tr = get_Xy(train_df)
    X_val, y_val = get_Xy(val_df)
    X_te, y_te = get_Xy(test_df)

    # NaN 제거 (lag 피처 초기 결측)
    mask_tr  = (~X_tr.isnull().any(axis=1)) & (~y_tr.isnull().any(axis=1))
    mask_val = (~X_val.isnull().any(axis=1)) & (~y_val.isnull().any(axis=1))
    mask_te  = (~X_te.isnull().any(axis=1)) & (~y_te.isnull().any(axis=1))
    X_tr, y_tr   = X_tr[mask_tr],   y_tr[mask_tr]
    X_val, y_val = X_val[mask_val], y_val[mask_val]
    X_te, y_te   = X_te[mask_te],   y_te[mask_te]

    print(f'  학습/검증/테스트: {len(X_tr)}/{len(X_val)}/{len(X_te)}')

    # 이상치 윈저라이징 (학습셋만 적용, 검증·테스트는 원본 유지)
    if 'winsorize_pct' in cfg:
        pct = cfg['winsorize_pct']
        caps = {}
        for col in y_tr.columns:
            cap = y_tr[col].quantile(pct)
            caps[col] = cap
            before = y_tr[col].max()
            y_tr[col] = y_tr[col].clip(upper=cap)
            print(f'  [{col}] 윈저라이징 {pct*100:.0f}%: max {before:.0f} → {cap:.0f}')

    # 모델 학습
    model = train_model(X_tr, y_tr, X_val, y_val)

    # 평가
    print(f'\n  [검증셋 성능]')
    metrics_val  = evaluate_model(model, X_val, y_val, 'Val')
    print(f'\n  [테스트셋 성능]')
    metrics_test = evaluate_model(model, X_te, y_te, 'Test')

    for _, row in metrics_test.iterrows():
        all_metrics.append({
            'plant': plant, 'split': 'test',
            'target': row['target'],
            'RMSE': round(row['RMSE'], 2),
            'MAE':  round(row['MAE'],  2),
            'R2':   round(row['R2'],   4),
        })

    # 모델 저장
    model_path = report_dir / f'emission_model_{plant}.pkl'
    joblib.dump(model, model_path)
    print(f'\n  [저장] {model_path}')

    # SHAP (전체 테스트셋 기준)
    print(f'\n  SHAP 계산...')
    X_shap = X_te.iloc[:min(300, len(X_te))]
    try:
        explain_shap(model, X_shap, list(X_shap.columns), list(y_te.columns), save_dir=report_dir)
    except Exception as e:
        print(f'  [WARN] SHAP 실패: {e}')

# 최종 성능 요약
print(f'\n{"="*65}')
print('  전사업소 모델 성능 요약 (테스트셋)')
print(f'{"="*65}')
metrics_df = pd.DataFrame(all_metrics)
print(metrics_df.to_string(index=False))

out_path = REPORTS_BASE / 'model_performance_summary.csv'
metrics_df.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f'\n[OK] 성능 요약 저장: {out_path}')
