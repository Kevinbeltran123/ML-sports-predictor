"""Train XGBoost + CatBoost margin regression models from dataset.sqlite.

Camino B: predice Residual = Margin + MARKET_SPREAD (desviacion del spread).
Residual > 0 → home covers.  Residual < 0 → away covers.

Mejoras sobre Camino A (raw margin):
  - Target centrado en 0 (residual μ≈0), distribucion mas compacta
  - Huber loss (delta=10) — blowouts penalizados linealmente
  - Time-decay sample weights (λ=0.9985 per day)
  - Walk-forward CV para evaluacion robusta

Uso:
    PYTHONPATH=. python scripts/train_margin_models.py
    PYTHONPATH=. python scripts/train_margin_models.py --dataset dataset_2012-26
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
from sklearn.model_selection import TimeSeriesSplit

from src.core.calibration.conformal_regression import ConformalRegressor
from src.config import DATASET_DB, NBA_MARGIN_MODELS_DIR, DROP_COLUMNS_ML, get_logger

logger = get_logger(__name__)

DEFAULT_DATASET = "dataset_2012-26"
TARGET = "Residual"  # was "Margin" — now predicts spread residual
DATE_COL = "Date"

# Columns to drop for margin regression.
# MARKET_SPREAD stays in DROP list — we do NOT want spread as a feature.
# Residual is computed from Margin + MARKET_SPREAD before dropping.
# "Residual" must also be dropped since it's the target (not in original DROP_COLUMNS_ML).
DROP_COLUMNS_MARGIN = list(DROP_COLUMNS_ML) + ["Residual"]

# Time-decay: λ^(days_ago). 0.9985 ≈ 0.95 at 1 season, 0.58 at 5 seasons.
DECAY_LAMBDA = 0.9985

# --- Hyperparameters (Huber loss, adapted from moneyline) ---

XGB_PARAMS = {
    "max_depth": 9, "eta": 0.10, "subsample": 0.90,
    "colsample_bytree": 0.85, "colsample_bylevel": 0.75,
    "colsample_bynode": 0.85, "min_child_weight": 15,
    "gamma": 6.0, "lambda": 2.0, "alpha": 0.5,
    "objective": "reg:pseudohubererror",
    "huber_slope": 10.0,
    "tree_method": "hist", "seed": 42,
    "eval_metric": ["rmse"],
}
XGB_NB = 800

CATBOOST_PARAMS = {
    "loss_function": "Huber:delta=10.0",
    "eval_metric": "RMSE",
    "random_seed": 42,
    "task_type": "CPU",
    "depth": 8,
    "iterations": 1500,
    "learning_rate": 0.008,
    "l2_leaf_reg": 1.0,
    "random_strength": 0.01,
    "bagging_temperature": 2.5,
    "rsm": 0.70,
    "border_count": 254,
}

W_XGB = 0.60
W_CAT = 0.40

# Sigma buckets by |spread| magnitude (calibrated post-training)
SIGMA_BUCKET_EDGES = [2.0, 5.0, 8.0, 12.0, float("inf")]


def load_and_prepare(dataset_name):
    with sqlite3.connect(DATASET_DB) as con:
        df = pd.read_sql_query(f'SELECT * FROM "{dataset_name}"', con)

    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)

    # Require Margin column (raw margin = home_score - away_score)
    if "Margin" not in df.columns:
        raise ValueError(
            "Column 'Margin' not found in dataset. "
            "Margin must be pre-computed as home_score - away_score."
        )

    # Require MARKET_SPREAD to compute residual
    if "MARKET_SPREAD" not in df.columns:
        raise ValueError(
            "Column 'MARKET_SPREAD' not found in dataset. "
            "Cannot compute residual without market spread."
        )

    # Compute Residual = Margin + MARKET_SPREAD
    # Home covers when Margin > -MARKET_SPREAD, i.e. Margin + MARKET_SPREAD > 0
    df["Residual"] = df["Margin"] + df["MARKET_SPREAD"]

    # Drop rows where spread is NaN (can't compute residual)
    df = df.dropna(subset=[TARGET, "MARKET_SPREAD"])
    return df


def df_to_xy(df):
    y = df[TARGET].astype(float).to_numpy()
    X_df = df.drop(columns=DROP_COLUMNS_MARGIN, errors="ignore")
    feature_cols = list(X_df.columns)
    X = X_df.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0).to_numpy(dtype=float)
    return X, y, feature_cols


def compute_sigma_buckets(y_true, y_pred, spreads):
    """Compute residual sigma per spread magnitude bucket."""
    residuals = y_true - y_pred
    abs_spreads = np.abs(spreads)
    buckets = {}
    prev_edge = 0.0
    for edge in SIGMA_BUCKET_EDGES:
        mask = (abs_spreads > prev_edge) & (abs_spreads <= edge)
        if mask.sum() > 10:
            sigma = float(np.std(residuals[mask]))
        else:
            sigma = float(np.std(residuals))
        label = f"{prev_edge}-{edge}" if edge != float('inf') else f"{prev_edge}+"
        buckets[label] = {"edge": edge, "sigma": round(sigma, 2), "n": int(mask.sum())}
        prev_edge = edge
    buckets["global"] = {"sigma": round(float(np.std(residuals)), 2), "n": len(residuals)}
    return buckets


def compute_ats_accuracy(y_residual, y_pred_residual):
    """ATS accuracy: does the predicted residual correctly predict cover?

    Residual = Margin + MARKET_SPREAD. Home covers when residual > 0.
    No need for spread input — it's already baked into the residual.
    """
    actual_cover = y_residual > 0
    predicted_cover = y_pred_residual > 0
    correct = actual_cover == predicted_cover
    # Exclude pushes (residual == 0)
    push = y_residual == 0
    valid = ~push
    if valid.sum() == 0:
        return 0.0
    return float(correct[valid].mean())


def _compute_sample_weights(dates):
    """Time-decay weights: recent games matter more."""
    days_ago = (dates.max() - dates).dt.days
    return DECAY_LAMBDA ** days_ago.values


def _walk_forward_cv(X, y, sample_weights, n_splits=5):
    """Walk-forward CV on training data (reporting only, doesn't affect final model)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        sw_tr = sample_weights[tr_idx]

        dtrain_cv = xgb.DMatrix(X_tr, label=y_tr, weight=sw_tr)
        dval_cv = xgb.DMatrix(X_val, label=y_val)
        bst = xgb.train(XGB_PARAMS, dtrain_cv, num_boost_round=XGB_NB,
                         evals=[(dval_cv, "val")], early_stopping_rounds=60,
                         verbose_eval=False)
        p_val = bst.predict(dval_cv)
        fold_rmse = float(np.sqrt(mean_squared_error(y_val, p_val)))
        rmses.append(fold_rmse)
        print(f"    Fold {fold+1}: RMSE={fold_rmse:.2f} (n={len(val_idx)})")

    mean_rmse = float(np.mean(rmses))
    print(f"    Walk-forward CV mean RMSE: {mean_rmse:.2f} ± {float(np.std(rmses)):.2f}")
    return mean_rmse


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost + CatBoost Residual Regression (Camino B)")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  TRAIN XGBoost + CatBoost — Spread Residual (Camino B)")
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

    # Extract spreads for sigma buckets (grouping variable)
    spreads_test = test_df["MARKET_SPREAD"].values if "MARKET_SPREAD" in test_df.columns else np.zeros(len(y_test))

    # Time-decay sample weights
    train_weights = _compute_sample_weights(train_df[DATE_COL])

    print(f"  Train: {len(y_train)} | Test: {len(y_test)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Residual stats — train: μ={y_train.mean():.2f} σ={y_train.std():.1f}")
    print(f"  Residual stats — test:  μ={y_test.mean():.2f} σ={y_test.std():.1f}")
    print(f"  Time-decay: λ={DECAY_LAMBDA}, oldest weight={train_weights.min():.3f}")

    # --- 0. Walk-forward CV (reporting only) ---
    print(f"\n  Walk-forward CV (5 folds, XGBoost only)...")
    wf_rmse = _walk_forward_cv(X_train, y_train, train_weights)

    # --- 1. Train XGBoost ---
    print(f"\n  Training XGBoost Regression (nb={XGB_NB}, Huber δ=10)...")
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=train_weights)
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
    print(f"\n  Training CatBoost Regression (it={CATBOOST_PARAMS['iterations']}, Huber δ=10)...")
    cat_model = CatBoostRegressor(**CATBOOST_PARAMS)
    cat_model.fit(
        X_train, y_train,
        sample_weight=train_weights,
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
    ats_acc = compute_ats_accuracy(y_test, p_ensemble)
    print(f"\n  Ensemble (XGB {W_XGB:.0%} + Cat {W_CAT:.0%})")
    print(f"    MAE: {ens_mae:.2f}, RMSE: {ens_rmse:.2f}")
    print(f"    ATS Accuracy: {ats_acc:.1%}")

    # --- 4. Sigma buckets ---
    sigma_buckets = compute_sigma_buckets(y_test, p_ensemble, spreads_test)
    print(f"\n  Sigma by spread bucket:")
    for label, info in sigma_buckets.items():
        if label != "global":
            print(f"    |spread| {label}: σ={info['sigma']:.1f} (n={info['n']})")
    print(f"    Global: σ={sigma_buckets['global']['sigma']:.1f}")

    # --- 5. Save models ---
    NBA_MARGIN_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  Saving models to {NBA_MARGIN_MODELS_DIR}...")

    # XGBoost
    xgb_rmse_str = f"{xgb_rmse:.1f}"
    xgb_name = (
        f"XGBoost_{xgb_rmse_str}rmse_MARGIN_"
        f"md{XGB_PARAMS['max_depth']}_eta{str(XGB_PARAMS['eta']).replace('.','p')}_"
        f"sub{str(XGB_PARAMS['subsample']).replace('.','p')}_"
        f"nb{XGB_NB}"
    )
    xgb_path = NBA_MARGIN_MODELS_DIR / f"{xgb_name}.json"
    booster.save_model(str(xgb_path))
    print(f"    XGBoost: {xgb_path.name}")

    # CatBoost
    cat_rmse_str = f"{cat_rmse:.1f}"
    cat_name = (
        f"CatBoost_{cat_rmse_str}rmse_MARGIN_"
        f"d{CATBOOST_PARAMS['depth']}_lr{str(CATBOOST_PARAMS['learning_rate']).replace('.','p')}_"
        f"it{CATBOOST_PARAMS['iterations']}"
    )
    cat_path = NBA_MARGIN_MODELS_DIR / f"{cat_name}.pkl"
    joblib.dump(cat_model, cat_path)
    print(f"    CatBoost: {cat_path.name}")

    # --- 6. Conformal regression ---
    print("\n  Fitting conformal regression...")
    cal_split = int(len(X_train) * 0.8)
    X_cal, y_cal = X_train[cal_split:], y_train[cal_split:]
    dcal = xgb.DMatrix(X_cal)
    p_xgb_cal = booster.predict(dcal)
    p_cat_cal = cat_model.predict(X_cal)
    p_ens_cal = W_XGB * p_xgb_cal + W_CAT * p_cat_cal

    conformal = ConformalRegressor(alpha=0.10)
    conformal.fit(p_ens_cal, y_cal)
    conf_path = NBA_MARGIN_MODELS_DIR / "margin_conformal.pkl"
    joblib.dump(conformal, conf_path)
    print(f"    Conformal: {conformal}")

    # --- 7. Sigma buckets + metadata ---
    meta = {
        "target": "residual",
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
        "ats_accuracy": ats_acc,
        "walk_forward_rmse": wf_rmse,
        "decay_lambda": DECAY_LAMBDA,
        "huber_delta": 10.0,
        "weights": {"xgb": W_XGB, "cat": W_CAT},
        "sigma_buckets": sigma_buckets,
        "conformal_summary": conformal.summary(),
    }
    meta_path = NBA_MARGIN_MODELS_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  DONE — Ensemble: MAE={ens_mae:.2f}, RMSE={ens_rmse:.2f}, ATS={ats_acc:.1%}")
    print(f"  Walk-forward RMSE: {wf_rmse:.2f}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
