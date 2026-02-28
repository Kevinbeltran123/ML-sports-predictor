"""Train XGBoost + CatBoost totals regression models from dataset.sqlite.

Predice Score = home_score + away_score (total de puntos, tipico 200-240 NBA).

Uso:
    PYTHONPATH=. python scripts/train_totals_models.py
    PYTHONPATH=. python scripts/train_totals_models.py --dataset dataset_2012-26
"""

import argparse
import json
import sqlite3

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.core.calibration.conformal_regression import ConformalRegressor
from src.config import DATASET_DB, NBA_UO_MODELS_DIR, DROP_COLUMNS_ML, get_logger

logger = get_logger(__name__)

DEFAULT_DATASET = "dataset_2012-26"
TARGET = "Score"
DATE_COL = "Date"

# Columns to drop for totals regression.
# Keep Score in the drop list — target is extracted separately in df_to_xy()
# before features are built, so it must NOT appear as a feature.
DROP_COLUMNS_TOTALS = list(DROP_COLUMNS_ML)

# --- Hyperparameters ---

XGB_PARAMS = {
    "max_depth": 8, "eta": 0.08, "subsample": 0.90,
    "colsample_bytree": 0.85, "colsample_bylevel": 0.75,
    "colsample_bynode": 0.85, "min_child_weight": 20,
    "gamma": 6.0, "lambda": 2.5, "alpha": 0.5,
    "objective": "reg:squarederror",
    "tree_method": "hist", "seed": 42,
    "eval_metric": ["rmse"],
}
XGB_NB = 800

CATBOOST_PARAMS = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": 42,
    "task_type": "CPU",
    "depth": 7,
    "iterations": 1500,
    "learning_rate": 0.008,
    "l2_leaf_reg": 1.5,
    "random_strength": 0.01,
    "bagging_temperature": 2.5,
    "rsm": 0.70,
    "border_count": 254,
}

W_XGB = 0.60
W_CAT = 0.40


def load_and_prepare(dataset_name):
    with sqlite3.connect(DATASET_DB) as con:
        df = pd.read_sql_query(f'SELECT * FROM "{dataset_name}"', con)

    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)

    if TARGET not in df.columns:
        raise ValueError(f"Column '{TARGET}' not found in dataset '{dataset_name}'.")

    df = df.dropna(subset=[TARGET])
    # Filter obvious outliers (Score should be 150-300 for NBA)
    df = df[(df[TARGET] >= 120) & (df[TARGET] <= 350)]
    return df


def df_to_xy(df):
    y = df[TARGET].astype(float).to_numpy()
    X_df = df.drop(columns=DROP_COLUMNS_TOTALS, errors="ignore")
    feature_cols = list(X_df.columns)
    X = X_df.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0).to_numpy(dtype=float)
    return X, y, feature_cols


def compute_ou_accuracy(y_total, y_pred_total, ou_lines):
    """O/U accuracy: does predicted total correctly predict over/under?"""
    actual_over = y_total > ou_lines
    predicted_over = y_pred_total > ou_lines
    correct = actual_over == predicted_over
    # Exclude pushes
    push = y_total == ou_lines
    valid = ~push
    if valid.sum() == 0:
        return 0.0
    return float(correct[valid].mean())


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost + CatBoost Totals Regression")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  TRAIN XGBoost + CatBoost — Totals Regression (O/U)")
    print(f"{'='*65}")

    df = load_and_prepare(args.dataset)

    # Temporal split
    test_dt = pd.to_datetime("2025-10-01")
    train_df = df[df[DATE_COL] < test_dt].copy()
    test_df = df[df[DATE_COL] >= test_dt].copy()

    if len(test_df) == 0:
        print("  WARNING: Test set vacio. Usando 80/20 split.")
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

    X_train, y_train, feature_cols = df_to_xy(train_df)
    X_test, y_test, _ = df_to_xy(test_df)

    # Extract O/U lines for accuracy
    ou_lines_test = test_df["OU"].values if "OU" in test_df.columns else np.full(len(y_test), y_test.mean())

    print(f"  Train: {len(y_train)} | Test: {len(y_test)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Score stats — train: μ={y_train.mean():.1f} σ={y_train.std():.1f}")
    print(f"  Score stats — test:  μ={y_test.mean():.1f} σ={y_test.std():.1f}")

    # --- 1. Train XGBoost ---
    print(f"\n  Training XGBoost Regression (nb={XGB_NB})...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    booster = xgb.train(
        XGB_PARAMS, dtrain,
        num_boost_round=XGB_NB,
        evals=[(dtest, "test")],
        early_stopping_rounds=60,
        verbose_eval=False,
    )
    p_xgb = booster.predict(dtest)
    xgb_mae = mean_absolute_error(y_test, p_xgb)
    xgb_rmse = float(np.sqrt(mean_squared_error(y_test, p_xgb)))
    print(f"  XGBoost — MAE: {xgb_mae:.2f}, RMSE: {xgb_rmse:.2f}")

    # --- 2. Train CatBoost ---
    print(f"\n  Training CatBoost Regression (it={CATBOOST_PARAMS['iterations']})...")
    cat_model = CatBoostRegressor(**CATBOOST_PARAMS)
    cat_model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=60,
        verbose=0,
    )
    p_cat = cat_model.predict(X_test)
    cat_mae = mean_absolute_error(y_test, p_cat)
    cat_rmse = float(np.sqrt(mean_squared_error(y_test, p_cat)))
    print(f"  CatBoost — MAE: {cat_mae:.2f}, RMSE: {cat_rmse:.2f}")

    # --- 3. Ensemble ---
    p_ensemble = W_XGB * p_xgb + W_CAT * p_cat
    ens_mae = mean_absolute_error(y_test, p_ensemble)
    ens_rmse = float(np.sqrt(mean_squared_error(y_test, p_ensemble)))
    ou_acc = compute_ou_accuracy(y_test, p_ensemble, ou_lines_test)
    print(f"\n  Ensemble (XGB {W_XGB:.0%} + Cat {W_CAT:.0%})")
    print(f"    MAE: {ens_mae:.2f}, RMSE: {ens_rmse:.2f}")
    print(f"    O/U Accuracy: {ou_acc:.1%}")

    # Residual sigma
    residuals = y_test - p_ensemble
    residual_sigma = float(np.std(residuals))
    print(f"    Residual σ: {residual_sigma:.1f} pts")

    # --- 4. Save models ---
    NBA_UO_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  Saving models to {NBA_UO_MODELS_DIR}...")

    # XGBoost
    xgb_rmse_str = f"{xgb_rmse:.1f}"
    xgb_name = (
        f"XGBoost_{xgb_rmse_str}rmse_TOTAL_"
        f"md{XGB_PARAMS['max_depth']}_eta{str(XGB_PARAMS['eta']).replace('.','p')}_"
        f"nb{XGB_NB}"
    )
    xgb_path = NBA_UO_MODELS_DIR / f"{xgb_name}.json"
    booster.save_model(str(xgb_path))
    print(f"    XGBoost: {xgb_path.name}")

    # CatBoost
    cat_rmse_str = f"{cat_rmse:.1f}"
    cat_name = (
        f"CatBoost_{cat_rmse_str}rmse_TOTAL_"
        f"d{CATBOOST_PARAMS['depth']}_lr{str(CATBOOST_PARAMS['learning_rate']).replace('.','p')}_"
        f"it{CATBOOST_PARAMS['iterations']}"
    )
    cat_path = NBA_UO_MODELS_DIR / f"{cat_name}.pkl"
    joblib.dump(cat_model, cat_path)
    print(f"    CatBoost: {cat_path.name}")

    # --- 5. Conformal regression ---
    print("\n  Fitting conformal regression...")
    cal_split = int(len(X_train) * 0.8)
    X_cal, y_cal = X_train[cal_split:], y_train[cal_split:]
    dcal = xgb.DMatrix(X_cal)
    p_xgb_cal = booster.predict(dcal)
    p_cat_cal = cat_model.predict(X_cal)
    p_ens_cal = W_XGB * p_xgb_cal + W_CAT * p_cat_cal

    conformal = ConformalRegressor(alpha=0.10)
    conformal.fit(p_ens_cal, y_cal)
    conf_path = NBA_UO_MODELS_DIR / "totals_conformal.pkl"
    joblib.dump(conformal, conf_path)
    print(f"    Conformal: {conformal}")

    # --- 6. Metadata ---
    meta = {
        "dataset": args.dataset,
        "train_size": len(y_train),
        "test_size": len(y_test),
        "n_features": len(feature_cols),
        "xgb_rmse": xgb_rmse,
        "xgb_mae": float(xgb_mae),
        "cat_rmse": cat_rmse,
        "cat_mae": float(cat_mae),
        "ensemble_rmse": ens_rmse,
        "ensemble_mae": float(ens_mae),
        "ou_accuracy": ou_acc,
        "residual_sigma": residual_sigma,
        "weights": {"xgb": W_XGB, "cat": W_CAT},
        "conformal_summary": conformal.summary(),
    }
    meta_path = NBA_UO_MODELS_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  DONE — Ensemble: MAE={ens_mae:.2f}, RMSE={ens_rmse:.2f}, O/U={ou_acc:.1%}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
