"""
배출량 예측 모델 모듈
- Multi-output XGBoost (SOx, NOx, 먼지 동시 예측)
- SHAP 기반 인과 해석
- 성능 평가 (RMSE, MAE, R²)
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "AppleGothic"   # macOS 한글
matplotlib.rcParams["axes.unicode_minus"] = False

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "reports"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# 1. 모델 학습
# ══════════════════════════════════════════════════════════════

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    params: dict | None = None,
) -> MultiOutputRegressor:
    """
    XGBoost Multi-output 회귀 모델 학습

    알고리즘 선정 사유 (보고서 Ⅲ장 기재):
    - XGBoost: 표형 데이터에서 최고 성능, 비선형 상호작용 포착
    - SHAP 기반 변수 중요도 해석 가능 → 실무 활용성 ↑
    - Early stopping으로 과적합 방지
    """
    default_params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)

    base_model = XGBRegressor(
        **default_params,
        eval_metric="rmse",
        early_stopping_rounds=30,
    )
    model = MultiOutputRegressor(base_model)

    # MultiOutputRegressor는 fit_params 전달 방식이 다름
    # 각 타겟별 학습
    estimators = []
    for i, target in enumerate(y_train.columns):
        print(f"  [{i+1}/{len(y_train.columns)}] {target} 모델 학습 중...")
        est = XGBRegressor(**default_params, eval_metric="rmse", early_stopping_rounds=30)
        est.fit(
            X_train, y_train[target],
            eval_set=[(X_val, y_val[target])],
            verbose=False,
        )
        estimators.append(est)
        print(f"    Best iteration: {est.best_iteration}")

    model.estimators_ = estimators
    model.n_outputs_ = len(y_train.columns)
    return model


# ══════════════════════════════════════════════════════════════
# 2. 성능 평가
# ══════════════════════════════════════════════════════════════

def evaluate_model(
    model: MultiOutputRegressor,
    X: pd.DataFrame,
    y: pd.DataFrame,
    split_name: str = "Test",
) -> pd.DataFrame:
    """RMSE, MAE, R² 평가"""
    y_pred = pd.DataFrame(
        model.predict(X),
        columns=y.columns,
        index=y.index,
    )
    results = []
    for col in y.columns:
        rmse = np.sqrt(mean_squared_error(y[col], y_pred[col]))
        mae = mean_absolute_error(y[col], y_pred[col])
        r2 = r2_score(y[col], y_pred[col])
        results.append({"target": col, "RMSE": rmse, "MAE": mae, "R2": r2})
        print(f"  {split_name} | {col}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════
# 3. SHAP 기반 해석 (인과 추론 보조)
# ══════════════════════════════════════════════════════════════

def explain_shap(
    model: MultiOutputRegressor,
    X: pd.DataFrame,
    feature_names: list[str],
    target_names: list[str],
    save_dir: Path = MODEL_DIR,
) -> dict[str, np.ndarray]:
    """
    SHAP 변수 중요도 계산 및 시각화

    인과 추론 활용 (보고서 Ⅲ장):
    - SHAP value: "이 변수가 얼마나 배출량을 증가/감소시키는가"
    - 단순 상관관계가 아닌 모델 기반 기여도 → 인과적 해석 근거
    """
    shap_values = {}
    for i, (estimator, target) in enumerate(zip(model.estimators_, target_names)):
        print(f"  SHAP 계산: {target}")
        explainer = shap.TreeExplainer(estimator)
        sv = explainer.shap_values(X)
        shap_values[target] = sv

        # Summary plot
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            sv, X,
            feature_names=feature_names,
            show=False,
            plot_size=None,
        )
        plt.title(f"{target} 배출량 — 변수 중요도 (SHAP)")
        plt.tight_layout()
        fig.savefig(save_dir / f"shap_{target}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    저장: shap_{target}.png")

    # 통합 변수 중요도 (절댓값 평균)
    importance_df = pd.DataFrame(index=feature_names)
    for target, sv in shap_values.items():
        importance_df[target] = np.abs(sv).mean(axis=0)
    importance_df["total"] = importance_df.sum(axis=1)
    importance_df = importance_df.sort_values("total", ascending=False)

    print("\n상위 10개 중요 변수:")
    print(importance_df.head(10).to_string())
    return shap_values


# ══════════════════════════════════════════════════════════════
# 4. 예측 (단일 시점)
# ══════════════════════════════════════════════════════════════

def predict_emissions(
    model: MultiOutputRegressor,
    X: pd.DataFrame,
    target_names: list[str],
) -> pd.DataFrame:
    """배출량 예측 반환 (최적화 모듈에서 호출)"""
    pred = model.predict(X)
    return pd.DataFrame(pred, columns=target_names, index=X.index)
