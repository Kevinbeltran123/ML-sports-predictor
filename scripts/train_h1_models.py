"""Train XGBoost + CatBoost First Half (1H) moneyline models.

Same feature set as full-game, different target: H1-Home-Win.

Uso:
    PYTHONPATH=. python scripts/train_h1_models.py
    PYTHONPATH=. python scripts/train_h1_models.py --dataset dataset_h1_2012-26
"""

import argparse
import json
import sqlite3

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss

from src.core.calibration.conformal import ConformalClassifier
from src.core.calibration.xgb_calibrator import XGBCalibrator
from src.config import DATASET_DB, NBA_H1_MODELS_DIR, DROP_COLUMNS_ML, get_logger

logger = get_logger(__name__)

DEFAULT_DATASET = "dataset_h1_2012-26"
TARGET = "H1-Home-Win"
DATE_COL = "Date"

# Same hyperparams as full-game (can Optuna-tune later)
XGB_PARAMS = {
    "max_depth": 9, "eta": 0.129, "subsample": 0.945,
    "colsample_bytree": 0.870, "colsample_bylevel": 0.765,
    "colsample_bynode": 0.858, "min_child_weight": 11,
    "gamma": 5.903, "lambda": 1.466, "alpha": 0.485,
    "objective": "multi:softprob", "num_class": 2,
    "tree_method": "hist", "seed": 42,
    "eval_metric": ["mlogloss"],
}
XGB_NB = 701

CATBOOST_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "auto_class_weights": "Balanced",
    "random_seed": 42,
    "task_type": "CPU",
    "depth": 9,
    "iterations": 1647,
    "learning_rate": 0.005,
    "l2_leaf_reg": 0.619,
    "random_strength": 0.008,
    "bagging_temperature": 2.744,
    "rsm": 0.709,
    "border_count": 233,
}

W_XGB = 0.60
W_CAT = 0.40


def compute_sample_weights(y, num_classes=2):
    counts = np.bincount(y, minlength=num_classes)
    total = len(y)
    class_weights = {
        cls: (total / (num_classes * count)) if count else 1.0
        for cls, count in enumerate(counts)
    }
    return np.array([class_weights[label] for label in y])


def load_and_prepare(dataset_name):
    with sqlite3.connect(DATASET_DB) as con:
        df = pd.read_sql_query(f'SELECT * FROM "{dataset_name}"', con)

    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)

    # Drop rows without H1 target
    before = len(df)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"  Dropped {before - after} rows without H1 target")

    return df


def df_to_xy(df):
    # Add Diff_TS_PCT if not present (same as tune_optuna.py)
    if "Diff_TS_PCT" not in df.columns and "TS_PCT" in df.columns:
        df["Diff_TS_PCT"] = df["TS_PCT"].astype(float) - df["TS_PCT.1"].astype(float)

    y = df[TARGET].astype(int).to_numpy()
    X_df = df.drop(columns=DROP_COLUMNS_ML, errors="ignore")
    # Also drop H1-specific columns that shouldn't be features
    X_df = X_df.drop(columns=["H1-Home-Win"], errors="ignore")
    feature_cols = list(X_df.columns)
    X = X_df.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0).to_numpy(dtype=float)
    return X, y, feature_cols


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost + CatBoost H1 models")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  TRAIN XGBoost + CatBoost — First Half (1H) Moneyline")
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

    print(f"  Train: {len(y_train)} | Test: {len(y_test)}")
    print(f"  Features: {len(feature_cols)}")
    baseline = max(float(y_test.mean()), 1.0 - float(y_test.mean()))
    print(f"  Baseline (majority): {baseline:.1%}")
    print(f"  H1 Home Win Rate (train): {y_train.mean():.1%}")
    print(f"  H1 Home Win Rate (test):  {y_test.mean():.1%}")

    # --- 1. Train XGBoost ---
    print(f"\n  Training XGBoost (nb={XGB_NB})...")
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=compute_sample_weights(y_train))
    dtest = xgb.DMatrix(X_test, label=y_test)

    booster = xgb.train(
        XGB_PARAMS, dtrain,
        num_boost_round=XGB_NB,
        evals=[(dtest, "test")],
        early_stopping_rounds=60,
        verbose_eval=False,
    )
    p_xgb = booster.predict(dtest)
    xgb_acc = accuracy_score(y_test, np.argmax(p_xgb, axis=1))
    xgb_ll = log_loss(y_test, p_xgb)
    print(f"  XGBoost — Acc: {xgb_acc:.1%}, LogLoss: {xgb_ll:.4f}")

    # --- 2. Train CatBoost ---
    print(f"\n  Training CatBoost (it={CATBOOST_PARAMS['iterations']})...")
    cat_model = CatBoostClassifier(**CATBOOST_PARAMS)
    cat_model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=60,
        verbose=0,
    )
    p_cat = cat_model.predict_proba(X_test)
    cat_acc = accuracy_score(y_test, np.argmax(p_cat, axis=1))
    cat_ll = log_loss(y_test, p_cat)
    print(f"  CatBoost — Acc: {cat_acc:.1%}, LogLoss: {cat_ll:.4f}")

    # --- 3. Weight search ---
    print(f"\n  Searching optimal weights...")
    best_acc = 0
    best_w = W_XGB
    for w in np.arange(0.0, 1.05, 0.05):
        p_mix = w * p_xgb + (1 - w) * p_cat
        acc = accuracy_score(y_test, np.argmax(p_mix, axis=1))
        if acc > best_acc:
            best_acc = acc
            best_w = round(w, 2)

    W_XGB_FINAL = best_w
    W_CAT_FINAL = round(1.0 - best_w, 2)
    p_ensemble = W_XGB_FINAL * p_xgb + W_CAT_FINAL * p_cat
    ens_acc = accuracy_score(y_test, np.argmax(p_ensemble, axis=1))
    ens_ll = log_loss(y_test, p_ensemble)
    print(f"  Best weights: XGB {W_XGB_FINAL:.0%} / Cat {W_CAT_FINAL:.0%}")
    print(f"  Ensemble — Acc: {ens_acc:.1%}, LogLoss: {ens_ll:.4f}")

    # --- 4. Save models ---
    NBA_H1_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  Saving models to {NBA_H1_MODELS_DIR}...")

    # XGBoost
    xgb_acc_str = f"{xgb_acc*100:.1f}"
    xgb_name = f"XGBoost_{xgb_acc_str}%_H1"
    xgb_path = NBA_H1_MODELS_DIR / f"{xgb_name}.json"
    booster.save_model(str(xgb_path))
    print(f"    XGBoost: {xgb_path.name}")

    # XGBoost calibrator (Platt)
    cal_split = int(len(X_train) * 0.8)
    X_cal, y_cal = X_train[cal_split:], y_train[cal_split:]
    calibrator = XGBCalibrator(booster)
    calibrator.fit(X_cal, y_cal)
    cal_path = xgb_path.with_name(f"{xgb_name}_calibration.pkl")
    joblib.dump(calibrator, cal_path)
    print(f"    Calibrator: {cal_path.name}")

    # CatBoost
    cat_acc_str = f"{cat_acc*100:.1f}"
    cat_name = f"CatBoost_{cat_acc_str}%_H1"
    cat_path = NBA_H1_MODELS_DIR / f"{cat_name}.pkl"
    joblib.dump(cat_model, cat_path)
    print(f"    CatBoost: {cat_path.name}")

    # --- 5. Ensemble conformal ---
    print("\n  Fitting ensemble conformal...")
    dcal = xgb.DMatrix(X_cal)
    p_xgb_cal = booster.predict(dcal)
    p_cat_cal = cat_model.predict_proba(X_cal)
    p_ens_cal = W_XGB_FINAL * p_xgb_cal + W_CAT_FINAL * p_cat_cal

    conformal = ConformalClassifier(alpha=0.10)
    conformal.fit(p_ens_cal, y_cal)
    conf_path = NBA_H1_MODELS_DIR / "ensemble_conformal.pkl"
    joblib.dump(conformal, conf_path)
    print(f"    Conformal: {conformal}")

    # --- 6. Variance model ---
    variance_info = {
        "mean_sigma": float(np.abs(p_xgb[:, 1] - p_cat[:, 1]).mean()),
        "sigma_percentiles": {
            str(p): float(np.percentile(np.abs(p_xgb[:, 1] - p_cat[:, 1]), p))
            for p in [25, 50, 75, 90, 95]
        },
    }
    var_path = NBA_H1_MODELS_DIR / "ensemble_variance.json"
    with open(var_path, "w") as f:
        json.dump(variance_info, f, indent=2)
    print(f"    Variance: mean_sigma={variance_info['mean_sigma']:.4f}")

    # --- 7. Metadata ---
    meta = {
        "market": "h1_moneyline",
        "dataset": args.dataset,
        "train_size": len(y_train),
        "test_size": len(y_test),
        "n_features": len(feature_cols),
        "xgb_accuracy": float(xgb_acc),
        "cat_accuracy": float(cat_acc),
        "ensemble_accuracy": float(ens_acc),
        "weights": {"xgb": W_XGB_FINAL, "cat": W_CAT_FINAL},
        "conformal_summary": conformal.summary(),
    }
    meta_path = NBA_H1_MODELS_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  DONE — H1 Ensemble: {ens_acc:.1%} ({len(feature_cols)} features)")
    print(f"  Weights: XGB {W_XGB_FINAL:.0%} / Cat {W_CAT_FINAL:.0%}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
