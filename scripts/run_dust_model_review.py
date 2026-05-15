"""
먼지 모델 재검토 (Step 2)
M1~M5 모델 비교로 R² > 0 달성 시도

M1: MultiOutput 기준선 (기존 모델)
M2: 단독 XGBoost, 기존 피처
M3: M2 + 먼지 전용 피처 (rolling7, year_trend)
M4: M2 + 이상치 윈저라이징 (99th pct)
M5: M3 + log(먼지+1) 변환
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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from features import build_features, get_X_y, TARGET_COLS
from optimizer import optimize_utilization_grid

matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

PROCESSED_DIR = Path(__file__).parent.parent / 'data' / 'processed'
REPORTS_BASE  = Path(__file__).parent.parent / 'reports'

XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    random_state=42,
    n_jobs=-1,
)


def temporal_split(df, train_r=0.8, val_r=0.1):
    n = len(df)
    t1 = int(n * train_r)
    t2 = int(n * (train_r + val_r))
    return df.iloc[:t1], df.iloc[t1:t2], df.iloc[t2:]


def winsorize(series: pd.Series, pct=0.99) -> pd.Series:
    upper = series.quantile(pct)
    return series.clip(upper=upper)


def evaluate(y_true, y_pred, label=''):
    r2  = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"  {label:30s} R²={r2:+.4f}  MAE={mae:>12,.0f}  RMSE={rmse:>12,.0f}")
    return r2, mae, rmse


def run_dust_review(plant: str):
    print(f'\n{"="*65}')
    print(f'  {plant} 먼지 모델 재검토')
    print(f'{"="*65}')

    report_dir = REPORTS_BASE / plant
    report_dir.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(PROCESSED_DIR / f'master_{plant}.csv', parse_dates=['date'])
    feature_df = build_features(master)
    feature_df = feature_df.dropna(subset=['먼지'])
    feature_df = feature_df.sort_values('date').reset_index(drop=True)

    train_df, val_df, test_df = temporal_split(feature_df)

    # 피처/타겟 분리 (기존 ALL_FEATURES 기준)
    X_all, y_all = get_X_y(feature_df)
    X_tr  = X_all.iloc[:len(train_df)]
    X_val = X_all.iloc[len(train_df):len(train_df)+len(val_df)]
    X_te  = X_all.iloc[len(train_df)+len(val_df):]
    y_tr  = y_all.iloc[:len(train_df)]
    y_val = y_all.iloc[len(train_df):len(train_df)+len(val_df)]
    y_te  = y_all.iloc[len(train_df)+len(val_df):]

    dust_tr  = y_tr['먼지']
    dust_val = y_val['먼지']
    dust_te  = y_te['먼지']

    results = []

    # ── M1: 기존 MultiOutput 기준선 ──────────────────────────────
    print('\n[M1] 기존 MultiOutput 기준선')
    existing_model = joblib.load(report_dir / f'emission_model_{plant}.pkl')
    pred_all = existing_model.predict(X_te)
    target_names_all = [t for t in TARGET_COLS if t in y_te.columns]
    dust_idx = target_names_all.index('먼지') if '먼지' in target_names_all else None
    if dust_idx is not None:
        r2, mae, rmse = evaluate(dust_te, pred_all[:, dust_idx], 'M1 MultiOutput baseline')
        results.append({'model': 'M1_baseline', 'R2': r2, 'MAE': mae, 'RMSE': rmse, 'transform': 'none'})

    # ── M2: 단독 XGBoost, 기존 피처 ──────────────────────────────
    print('\n[M2] 단독 XGBoost, 기존 피처')
    m2 = XGBRegressor(**XGB_PARAMS)
    m2.fit(X_tr, dust_tr)
    pred_m2 = m2.predict(X_te)
    pred_m2 = np.maximum(pred_m2, 0)
    r2, mae, rmse = evaluate(dust_te, pred_m2, 'M2 XGB standalone')
    results.append({'model': 'M2_standalone', 'R2': r2, 'MAE': mae, 'RMSE': rmse, 'transform': 'none'})

    # ── M3: M2 + 먼지 전용 피처 ──────────────────────────────────
    print('\n[M3] M2 + 먼지 전용 피처 (rolling7, year_trend, n_units_change)')
    def add_dust_features(df_feat: pd.DataFrame, feat_df_with_target: pd.DataFrame) -> pd.DataFrame:
        df = df_feat.copy()
        dust_series = feat_df_with_target['먼지']
        df['dust_rolling7'] = dust_series.shift(1).rolling(7, min_periods=3).mean().values
        df['dust_rolling30'] = dust_series.shift(1).rolling(30, min_periods=7).mean().values
        df['year_trend'] = (feat_df_with_target['date'].dt.year - 2022).values
        return df.fillna(df.mean())

    X_tr_m3  = add_dust_features(X_tr,  train_df)
    X_val_m3 = add_dust_features(X_val, val_df)
    X_te_m3  = add_dust_features(X_te,  test_df)

    m3 = XGBRegressor(**XGB_PARAMS)
    m3.fit(X_tr_m3, dust_tr)
    pred_m3 = np.maximum(m3.predict(X_te_m3), 0)
    r2, mae, rmse = evaluate(dust_te, pred_m3, 'M3 XGB + dust features')
    results.append({'model': 'M3_dust_feats', 'R2': r2, 'MAE': mae, 'RMSE': rmse, 'transform': 'none'})

    # ── M4: M2 + 이상치 윈저라이징 ───────────────────────────────
    print('\n[M4] M2 + 이상치 윈저라이징 (99th pct)')
    p99 = dust_tr.quantile(0.99)
    dust_tr_w  = winsorize(dust_tr)
    dust_val_w = winsorize(dust_val, pct=1.0).clip(upper=p99)  # test set는 winsorize 안 함
    print(f'  99th pct 상한: {p99:,.0f} kg/day')
    print(f'  윈저라이징 후 train std 감소: {dust_tr.std():.0f} → {dust_tr_w.std():.0f}')
    m4 = XGBRegressor(**XGB_PARAMS)
    m4.fit(X_tr, dust_tr_w)
    pred_m4 = np.maximum(m4.predict(X_te), 0)
    r2, mae, rmse = evaluate(dust_te, pred_m4, 'M4 XGB + Winsorize 99th')
    results.append({'model': 'M4_winsorize', 'R2': r2, 'MAE': mae, 'RMSE': rmse, 'transform': 'winsorize_99'})

    # ── M5: M3 + log(먼지+1) 변환 ────────────────────────────────
    print('\n[M5] M3 + log(먼지+1) 변환')
    log_dust_tr  = np.log1p(dust_tr)
    m5 = XGBRegressor(**XGB_PARAMS)
    m5.fit(X_tr_m3, log_dust_tr)
    pred_m5_log = m5.predict(X_te_m3)
    pred_m5 = np.expm1(pred_m5_log)
    pred_m5 = np.maximum(pred_m5, 0)
    r2, mae, rmse = evaluate(dust_te, pred_m5, 'M5 XGB + dust feats + log')
    results.append({'model': 'M5_log_transform', 'R2': r2, 'MAE': mae, 'RMSE': rmse, 'transform': 'log1p'})

    # ── M4W: 윈저라이징 + dust 전용 피처 조합 (보너스) ───────────
    print('\n[M4W] 윈저라이징 + dust 전용 피처')
    m4w = XGBRegressor(**XGB_PARAMS)
    m4w.fit(X_tr_m3, dust_tr_w)
    pred_m4w = np.maximum(m4w.predict(X_te_m3), 0)
    r2, mae, rmse = evaluate(dust_te, pred_m4w, 'M4W XGB + winsorize + feats')
    results.append({'model': 'M4W_winsorize_feats', 'R2': r2, 'MAE': mae, 'RMSE': rmse, 'transform': 'winsorize_99'})

    # ── 결과 요약 ─────────────────────────────────────────────────
    result_df = pd.DataFrame(results)
    result_df.insert(0, 'plant', plant)
    print(f'\n{"─"*65}')
    print(f'  {plant} 먼지 모델 비교 요약 (test set)')
    print(f'{"─"*65}')
    print(result_df[['model','R2','MAE','RMSE']].to_string(index=False))

    best_model_name = result_df.loc[result_df['R2'].idxmax(), 'model']
    best_r2 = result_df['R2'].max()
    print(f'\n  최선 모델: {best_model_name} (R²={best_r2:.4f})')

    # ── 최선 모델 예측 vs 실제 시각화 ─────────────────────────────
    pred_best = {
        'M1_baseline': pred_all[:, dust_idx] if dust_idx is not None else None,
        'M2_standalone': pred_m2,
        'M3_dust_feats': pred_m3,
        'M4_winsorize': pred_m4,
        'M5_log_transform': pred_m5,
        'M4W_winsorize_feats': pred_m4w,
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, (name, pred) in zip(axes.flatten(), pred_best.items()):
        if pred is None:
            continue
        r2_p = r2_score(dust_te, pred)
        # clip for visualization (10th~99th pct)
        clip_high = dust_te.quantile(0.95)
        mask = dust_te <= clip_high
        ax.scatter(dust_te[mask], pred[mask], alpha=0.4, s=12, color='steelblue')
        lim = max(dust_te[mask].max(), pred[mask].max())
        ax.plot([0, lim], [0, lim], 'r--', lw=1.2)
        ax.set_xlabel('실제 먼지 (kg/day)')
        ax.set_ylabel('예측 먼지 (kg/day)')
        ax.set_title(f'{name}\nR²={r2_p:.4f}', fontweight='bold')
        ax.grid(alpha=0.3)

    plt.suptitle(f'{plant} 먼지 모델 비교 (예측 vs 실제, 상위 5% 제외)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(report_dir / 'dust_model_pred_vs_actual.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [저장] dust_model_pred_vs_actual.png')

    # ── 아웃라이어 분석 ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(np.log1p(dust_te), bins=40, color='steelblue', alpha=0.7, edgecolor='white')
    axes[0].set_xlabel('log(먼지+1)')
    axes[0].set_ylabel('빈도')
    axes[0].set_title(f'{plant} 먼지 분포 (log 스케일)', fontweight='bold')
    axes[0].grid(alpha=0.3)

    ts_idx = test_df['date'].values
    axes[1].plot(ts_idx, dust_te.values, 'o', ms=3, alpha=0.5, label='실제', color='gray')
    axes[1].plot(ts_idx, pred_m5, 'r-', alpha=0.7, lw=1.2, label='M5 예측')
    axes[1].plot(ts_idx, pred_m4w, 'b-', alpha=0.7, lw=1.2, label='M4W 예측')
    axes[1].set_xlabel('날짜'); axes[1].set_ylabel('먼지 (kg/day)')
    axes[1].set_title(f'{plant} 먼지 시계열 예측 비교', fontweight='bold')
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    plt.suptitle(f'{plant} 먼지 모델 분석', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(report_dir / 'dust_model_outlier_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [저장] dust_model_outlier_analysis.png')

    return result_df


# ── 전체 실행 ─────────────────────────────────────────────────
all_results = []
for plant in ['삼천포', '영흥']:
    res = run_dust_review(plant)
    all_results.append(res)

combined = pd.concat(all_results, ignore_index=True)
out_path = REPORTS_BASE / 'dust_model_comparison.csv'
combined.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f'\n[OK] 먼지 모델 비교 저장: {out_path}')

print('\n' + '='*65)
print('  먼지 모델 재검토 완료')
print('='*65)
print(combined[['plant','model','R2','MAE']].to_string(index=False))
