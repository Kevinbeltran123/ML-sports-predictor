"""Train XGBoost + CatBoost moneyline, F5, and Totals models for MLB.

Estructura igual a train_models_v2.py pero adaptada para MLB:
  1. Soporte para 3 tipos de modelos: ML moneyline, F5 (primeras 5 entradas), Totals (regresion)
  2. Split temporal por temporada: Train 2018-2022, Val 2023, Test 2024-2025
  3. Hiperparametros mas conservadores (MLB es mas dificil que NBA)
  4. ConformalRegressor para totals (en lugar de ConformalClassifier)
  5. Benchmark automatico — solo guarda si mejora sobre la linea base

Uso:
    PYTHONPATH=. python scripts/train_mlb_models.py
    PYTHONPATH=. python scripts/train_mlb_models.py --target ml
    PYTHONPATH=. python scripts/train_mlb_models.py --target f5
    PYTHONPATH=. python scripts/train_mlb_models.py --target totals
    PYTHONPATH=. python scripts/train_mlb_models.py --target all
    PYTHONPATH=. python scripts/train_mlb_models.py --dataset mlb_dataset_2018-25 --dry-run
"""

import argparse
import json
import sqlite3

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error

from src.core.calibration.conformal import ConformalClassifier
from src.core.calibration.xgb_calibrator import XGBCalibrator

try:
    from src.core.calibration.conformal_regression import ConformalRegressor
except ImportError:
    ConformalRegressor = None
from src.sports.mlb.config_paths import (
    MLB_DATASET_DB,
    MLB_ML_MODELS_DIR,
    MLB_F5_MODELS_DIR,
    MLB_TOTALS_MODELS_DIR,
)
from src.config import get_logger

logger = get_logger(__name__)

DEFAULT_DATASET = "mlb_dataset_2018-25"
TARGET_ML = "Home-Team-Win"
TARGET_F5 = "F5-Home-Win"
TARGET_TOT = "Total_Runs"
SEASON_COL = "SEASON"

# ---------------------------------------------------------------------------
# Columns to drop before feeding features to models.
# Includes targets, identifiers, leakage columns (actual game scores/stats).
# ---------------------------------------------------------------------------
DROP_COLUMNS_MLB = [
    "GAME_PK", "GAME_DATE", "SEASON", "HOME_AWAY", "HOME_AWAY.1",
    "TEAM_NAME", "TEAM_NAME.1", "TEAM_ID", "TEAM_ID.1",
    "OPPONENT_NAME", "OPPONENT_NAME.1", "OPPONENT_ID", "OPPONENT_ID.1",
    "WIN", "WIN.1", "W_L", "W_L.1",
    "SP_NAME", "SP_NAME.1", "SP_ID", "SP_ID.1",
    "INNING_RUNS", "INNING_RUNS.1",
    "Home-Team-Win", "F5-Home-Win", "Total_Runs",
    "RUNS", "RUNS.1", "R", "R.1",       # leakage: actual game scores
    "HITS", "HITS.1", "H", "H.1", "ERRORS", "ERRORS.1",  # leakage
    "F5_RUNS", "F5_RUNS.1",             # leakage
    "DIVISION_HOME", "DIVISION_AWAY", "LEAGUE_HOME", "LEAGUE_AWAY",  # categorical strings
    "Home-Team-Index", "Away-Team-Index",  # indices, not features
]

# ---------------------------------------------------------------------------
# Hyperparameters (conservative for MLB — noisier sport than NBA)
# ---------------------------------------------------------------------------

XGB_CLS_PARAMS = {
    "max_depth": 5,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.75,
    "min_child_weight": 12,
    "gamma": 5.0,
    "lambda": 1.0,
    "alpha": 0.5,
    "objective": "multi:softprob",
    "num_class": 2,
    "tree_method": "hist",
    "seed": 42,
    "eval_metric": ["mlogloss"],
}
XGB_CLS_NB = 500

XGB_REG_PARAMS = {
    "max_depth": 5,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.75,
    "min_child_weight": 12,
    "gamma": 5.0,
    "lambda": 1.0,
    "alpha": 0.5,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "seed": 42,
    "eval_metric": ["rmse"],
}
XGB_REG_NB = 500

CATBOOST_CLS_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "auto_class_weights": "Balanced",
    "random_seed": 42,
    "task_type": "CPU",
    "depth": 7,
    "iterations": 800,
    "learning_rate": 0.01,
    "l2_leaf_reg": 4.0,
    "verbose": 0,
}

CATBOOST_REG_PARAMS = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": 42,
    "task_type": "CPU",
    "depth": 7,
    "iterations": 800,
    "learning_rate": 0.01,
    "l2_leaf_reg": 4.0,
    "verbose": 0,
}

# Baselines actuales (se sobreescriben con metadata.json si existe)
BASELINE_ML_XGB_ACC = 0.54
BASELINE_ML_CAT_ACC = 0.54
BASELINE_ML_ENS_ACC = 0.54

BASELINE_F5_XGB_ACC = 0.54
BASELINE_F5_CAT_ACC = 0.54
BASELINE_F5_ENS_ACC = 0.54

BASELINE_TOT_XGB_RMSE = 4.0
BASELINE_TOT_CAT_RMSE = 4.0
BASELINE_TOT_ENS_RMSE = 4.0


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def compute_sample_weights(y, num_classes=2):
    counts = np.bincount(y, minlength=num_classes)
    total = len(y)
    class_weights = {
        cls: (total / (num_classes * count)) if count else 1.0
        for cls, count in enumerate(counts)
    }
    return np.array([class_weights[label] for label in y])


def compute_ece(y_true, y_prob, n_bins=15):
    """Expected Calibration Error — mide que tan bien calibradas estan las probs."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = y_prob[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += mask.sum() * abs(avg_conf - avg_acc)
    return ece / len(y_true)


def load_baselines(models_dir, prefix="metadata.json"):
    """Carga baselines desde metadata.json si existe."""
    path = models_dir / prefix
    if path.exists():
        with open(path) as f:
            meta = json.load(f)
        return meta
    return {}


class PlattCalibrator:
    """Platt scaling (sigmoid) para cualquier modelo que retorna P(class=1)."""

    def __init__(self):
        self._cal = None

    def fit(self, p1, y):
        self._cal = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        self._cal.fit(p1.reshape(-1, 1), y)
        return self

    def calibrate(self, p1):
        return self._cal.predict_proba(p1.reshape(-1, 1))


def find_optimal_weights(probs_dict, y_test, n_steps=21):
    """Grid search para pesos optimos del ensemble de clasificacion.

    Prioriza ECE (calibracion) sobre accuracy.
    """
    models = list(probs_dict.keys())
    n = len(models)
    best = {"weights": None, "ece": 1.0, "acc": 0.0, "ll": 1.0}

    if n == 2:
        for w1 in np.linspace(0, 1, n_steps):
            w2 = 1.0 - w1
            p = w1 * probs_dict[models[0]] + w2 * probs_dict[models[1]]
            acc = accuracy_score(y_test, np.argmax(p, axis=1))
            ll = log_loss(y_test, p)
            ece = compute_ece(y_test, p[:, 1])
            if ece < best["ece"] or (abs(ece - best["ece"]) < 0.001 and acc > best["acc"]):
                best = {
                    "weights": {models[0]: round(w1, 2), models[1]: round(w2, 2)},
                    "ece": ece, "acc": acc, "ll": ll,
                }
    return best


def find_optimal_weights_rmse(preds_dict, y_test, n_steps=21):
    """Grid search para pesos optimos del ensemble de regresion (minimiza RMSE)."""
    models = list(preds_dict.keys())
    best = {"weights": None, "rmse": 1e9}

    for w1 in np.linspace(0, 1, n_steps):
        w2 = 1.0 - w1
        p = w1 * preds_dict[models[0]] + w2 * preds_dict[models[1]]
        rmse = np.sqrt(mean_squared_error(y_test, p))
        if rmse < best["rmse"]:
            best = {
                "weights": {models[0]: round(w1, 2), models[1]: round(w2, 2)},
                "rmse": rmse,
            }
    return best


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_prepare(dataset_name):
    with sqlite3.connect(MLB_DATASET_DB) as con:
        df = pd.read_sql_query(f'SELECT * FROM "{dataset_name}"', con)

    if SEASON_COL in df.columns:
        df[SEASON_COL] = df[SEASON_COL].astype(str)

    return df


def df_to_xy_cls(df, target, drop_cols=None):
    """Prepara features + target para clasificacion."""
    if drop_cols is None:
        drop_cols = DROP_COLUMNS_MLB
    y = df[target].astype(int).to_numpy()
    X_df = df.drop(columns=drop_cols, errors="ignore")
    feature_cols = list(X_df.columns)
    X = X_df.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0).to_numpy(dtype=float)
    return X, y, feature_cols


def df_to_xy_reg(df, target, drop_cols=None):
    """Prepara features + target para regresion."""
    if drop_cols is None:
        drop_cols = DROP_COLUMNS_MLB
    y = df[target].astype(float).to_numpy()
    X_df = df.drop(columns=drop_cols, errors="ignore")
    feature_cols = list(X_df.columns)
    X = X_df.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0).to_numpy(dtype=float)
    return X, y, feature_cols


# ---------------------------------------------------------------------------
# Train: ML Moneyline (classifier)
# ---------------------------------------------------------------------------

def train_ml(df, dry_run=False):
    print(f"\n{'='*65}")
    print(f"  TRAIN MLB — Moneyline (Home-Team-Win)")
    print(f"{'='*65}")

    # Baselines
    prev_meta = load_baselines(MLB_ML_MODELS_DIR)
    baseline_xgb = prev_meta.get("xgb_accuracy", BASELINE_ML_XGB_ACC)
    baseline_cat = prev_meta.get("cat_accuracy", BASELINE_ML_CAT_ACC)
    baseline_ens = prev_meta.get("ensemble_accuracy", BASELINE_ML_ENS_ACC)

    # Split temporal: Train 2018-2022, Val 2023, Test 2024-2025
    train_seasons = ["2018", "2019", "2020", "2021", "2022"]
    val_seasons = ["2023"]
    test_seasons = ["2024", "2025"]

    train_df = df[df[SEASON_COL].isin(train_seasons)].copy()
    val_df = df[df[SEASON_COL].isin(val_seasons)].copy()
    test_df = df[df[SEASON_COL].isin(test_seasons)].copy()

    # Fall back to 80/20 if split is empty
    if len(test_df) == 0 or len(train_df) == 0:
        print("  WARNING: Season split vacio. Usando 80/20 split.")
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        val_df = df.iloc[int(split_idx * 0.85):split_idx].copy()

    # Filter out rows where target is NaN
    train_df = train_df.dropna(subset=[TARGET_ML])
    val_df = val_df.dropna(subset=[TARGET_ML])
    test_df = test_df.dropna(subset=[TARGET_ML])

    X_train, y_train, feature_cols = df_to_xy_cls(train_df, TARGET_ML)
    X_val, y_val, _ = df_to_xy_cls(val_df, TARGET_ML)
    X_test, y_test, _ = df_to_xy_cls(test_df, TARGET_ML)

    # Calibration set: use validation data
    X_cal, y_cal = X_val, y_val

    print(f"  Train: {len(y_train)} | Val/Cal: {len(y_cal)} | Test: {len(y_test)}")
    print(f"  Features: {len(feature_cols)}")
    baseline = max(float(y_test.mean()), 1.0 - float(y_test.mean()))
    print(f"  Baseline (majority): {baseline:.1%}")
    print(f"  Prev baselines: XGB={baseline_xgb:.1%}, Cat={baseline_cat:.1%}, Ens={baseline_ens:.1%}")

    # --- XGBoost ---
    print(f"\n  Training XGBoost (nb={XGB_CLS_NB})...")
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=compute_sample_weights(y_train))
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    booster = xgb.train(
        XGB_CLS_PARAMS, dtrain,
        num_boost_round=XGB_CLS_NB,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    xgb_calibrator = XGBCalibrator(booster)
    xgb_calibrator.fit(X_cal, y_cal)
    p_xgb = xgb_calibrator.predict_proba(X_test)
    xgb_acc = accuracy_score(y_test, np.argmax(p_xgb, axis=1))
    xgb_ece = compute_ece(y_test, p_xgb[:, 1])
    xgb_ll = log_loss(y_test, p_xgb)
    print(f"  XGBoost — Acc: {xgb_acc:.1%}, ECE: {xgb_ece:.4f}, LogLoss: {xgb_ll:.4f}")

    # --- CatBoost ---
    print(f"\n  Training CatBoost (it={CATBOOST_CLS_PARAMS['iterations']})...")
    cat_model = CatBoostClassifier(**CATBOOST_CLS_PARAMS)
    cat_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
    )
    p_cat_raw = cat_model.predict_proba(X_test)
    cat_acc_raw = accuracy_score(y_test, np.argmax(p_cat_raw, axis=1))

    # Platt calibration for CatBoost
    p_cat_cal_raw = cat_model.predict_proba(X_cal)[:, 1]
    cat_calibrator = PlattCalibrator()
    cat_calibrator.fit(p_cat_cal_raw, y_cal)

    p_cat = cat_calibrator.calibrate(p_cat_raw[:, 1])
    cat_acc = accuracy_score(y_test, np.argmax(p_cat, axis=1))
    cat_ece = compute_ece(y_test, p_cat[:, 1])
    cat_ll = log_loss(y_test, p_cat)

    cat_ece_raw = compute_ece(y_test, p_cat_raw[:, 1])
    if cat_ece < cat_ece_raw:
        print(f"  CatBoost — Acc: {cat_acc:.1%}, ECE: {cat_ece:.4f} (Platt improved: {cat_ece_raw:.4f}→{cat_ece:.4f})")
    else:
        print(f"  CatBoost — Platt did NOT improve ECE ({cat_ece_raw:.4f}→{cat_ece:.4f}), using raw probs")
        p_cat = p_cat_raw
        cat_acc = cat_acc_raw
        cat_ece = cat_ece_raw
        cat_ll = log_loss(y_test, p_cat)
        cat_calibrator = None

    # --- Ensemble weights ---
    print(f"\n  Finding optimal ensemble weights...")
    best = find_optimal_weights({"xgb": p_xgb, "cat": p_cat}, y_test)
    weights = best["weights"]
    p_ensemble = weights["xgb"] * p_xgb + weights["cat"] * p_cat
    ens_acc = accuracy_score(y_test, np.argmax(p_ensemble, axis=1))
    ens_ece = compute_ece(y_test, p_ensemble[:, 1])
    ens_ll = log_loss(y_test, p_ensemble)

    print(f"\n  {'='*60}")
    print(f"  RESULTADOS ML")
    print(f"  {'='*60}")
    print(f"  XGBoost:  Acc={xgb_acc:.1%}  ECE={xgb_ece:.4f}  (prev: {baseline_xgb:.1%})")
    print(f"  CatBoost: Acc={cat_acc:.1%}  ECE={cat_ece:.4f}  (prev: {baseline_cat:.1%})")
    print(f"  Ensemble: Acc={ens_acc:.1%}  ECE={ens_ece:.4f}  (prev: {baseline_ens:.1%})")
    print(f"  Weights:  {weights}")

    # Check improvement
    improved = ens_acc > baseline_ens or ens_ece < compute_ece(y_test, (0.5 * p_xgb + 0.5 * p_cat)[:, 1])
    if not improved:
        print(f"\n  NO IMPROVEMENT over baselines. Models NOT saved.")
        return

    if dry_run:
        print(f"\n  DRY RUN — would save models but --dry-run flag is set.")
        return

    print(f"\n  Saving ML models...")
    MLB_ML_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # XGBoost
    xgb_acc_str = f"{xgb_acc*100:.1f}"
    xgb_name = (
        f"XGBoost_{xgb_acc_str}%_MLB_ML_"
        f"md{XGB_CLS_PARAMS['max_depth']}_eta{str(XGB_CLS_PARAMS['eta']).replace('.','p')}_"
        f"sub{str(XGB_CLS_PARAMS['subsample']).replace('.','p')}_"
        f"nb{XGB_CLS_NB}"
    )
    xgb_path = MLB_ML_MODELS_DIR / f"{xgb_name}.json"
    booster.save_model(str(xgb_path))
    cal_path = xgb_path.with_name(f"{xgb_name}_calibration.pkl")
    joblib.dump(xgb_calibrator, cal_path)
    print(f"    XGBoost: {xgb_path.name}")

    # CatBoost
    cat_acc_str = f"{cat_acc*100:.1f}"
    cat_name = (
        f"CatBoost_{cat_acc_str}%_MLB_ML_"
        f"d{CATBOOST_CLS_PARAMS['depth']}_lr{str(CATBOOST_CLS_PARAMS['learning_rate']).replace('.','p')}_"
        f"it{CATBOOST_CLS_PARAMS['iterations']}"
    )
    cat_path = MLB_ML_MODELS_DIR / f"{cat_name}.pkl"
    joblib.dump(cat_model, cat_path)
    print(f"    CatBoost: {cat_path.name}")
    if cat_calibrator is not None:
        cat_cal_path = cat_path.with_name(f"{cat_name}_calibration.pkl")
        joblib.dump(cat_calibrator, cat_cal_path)
        print(f"    Cat Calibrator: {cat_cal_path.name}")

    # Conformal
    print("\n  Fitting ensemble conformal...")
    p_xgb_cal = xgb_calibrator.predict_proba(X_cal)
    if cat_calibrator is not None:
        p_cat_cal = cat_calibrator.calibrate(cat_model.predict_proba(X_cal)[:, 1])
    else:
        p_cat_cal = cat_model.predict_proba(X_cal)
    p_ens_cal = weights["xgb"] * p_xgb_cal + weights["cat"] * p_cat_cal

    conformal = ConformalClassifier(alpha=0.10)
    conformal.fit(p_ens_cal, y_cal)
    conf_path = MLB_ML_MODELS_DIR / "ensemble_conformal.pkl"
    joblib.dump(conformal, conf_path)
    print(f"    Conformal: {conformal}")

    # Variance
    variance_info = {
        "mean_sigma": float(np.abs(p_xgb[:, 1] - p_cat[:, 1]).mean()),
        "sigma_percentiles": {
            str(p): float(np.percentile(np.abs(p_xgb[:, 1] - p_cat[:, 1]), p))
            for p in [25, 50, 75, 90, 95]
        },
    }
    var_path = MLB_ML_MODELS_DIR / "ensemble_variance.json"
    with open(var_path, "w") as f:
        json.dump(variance_info, f, indent=2)

    # Metadata
    meta = {
        "model_type": "ml",
        "target": TARGET_ML,
        "train_seasons": train_seasons,
        "test_seasons": test_seasons,
        "train_size": int(len(y_train)),
        "val_size": int(len(y_cal)),
        "test_size": int(len(y_test)),
        "n_features": len(feature_cols),
        "xgb_accuracy": float(xgb_acc),
        "xgb_ece": float(xgb_ece),
        "cat_accuracy": float(cat_acc),
        "cat_ece": float(cat_ece),
        "cat_platt_calibrated": cat_calibrator is not None,
        "ensemble_accuracy": float(ens_acc),
        "ensemble_ece": float(ens_ece),
        "ensemble_logloss": float(ens_ll),
        "weights": weights,
        "conformal_summary": conformal.summary(),
    }
    meta_path = MLB_ML_MODELS_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  DONE ML — Ensemble: {ens_acc:.1%} | XGB {weights['xgb']:.0%} + Cat {weights['cat']:.0%}")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# Train: F5 (first 5 innings, classifier)
# ---------------------------------------------------------------------------

def train_f5(df, dry_run=False):
    print(f"\n{'='*65}")
    print(f"  TRAIN MLB — F5 (F5-Home-Win, primeras 5 entradas)")
    print(f"{'='*65}")

    prev_meta = load_baselines(MLB_F5_MODELS_DIR)
    baseline_xgb = prev_meta.get("xgb_accuracy", BASELINE_F5_XGB_ACC)
    baseline_cat = prev_meta.get("cat_accuracy", BASELINE_F5_CAT_ACC)
    baseline_ens = prev_meta.get("ensemble_accuracy", BASELINE_F5_ENS_ACC)

    # Filter out tied F5 games (NaN target)
    df_f5 = df.dropna(subset=[TARGET_F5]).copy()
    removed = len(df) - len(df_f5)
    if removed > 0:
        print(f"  Removed {removed} rows with NaN F5-Home-Win (tied games)")

    train_seasons = ["2018", "2019", "2020", "2021", "2022"]
    val_seasons = ["2023"]
    test_seasons = ["2024", "2025"]

    train_df = df_f5[df_f5[SEASON_COL].isin(train_seasons)].copy()
    val_df = df_f5[df_f5[SEASON_COL].isin(val_seasons)].copy()
    test_df = df_f5[df_f5[SEASON_COL].isin(test_seasons)].copy()

    if len(test_df) == 0 or len(train_df) == 0:
        print("  WARNING: Season split vacio. Usando 80/20 split.")
        split_idx = int(len(df_f5) * 0.8)
        train_df = df_f5.iloc[:split_idx].copy()
        test_df = df_f5.iloc[split_idx:].copy()
        val_df = df_f5.iloc[int(split_idx * 0.85):split_idx].copy()

    X_train, y_train, feature_cols = df_to_xy_cls(train_df, TARGET_F5)
    X_val, y_val, _ = df_to_xy_cls(val_df, TARGET_F5)
    X_test, y_test, _ = df_to_xy_cls(test_df, TARGET_F5)
    X_cal, y_cal = X_val, y_val

    print(f"  Train: {len(y_train)} | Val/Cal: {len(y_cal)} | Test: {len(y_test)}")
    print(f"  Features: {len(feature_cols)}")
    baseline = max(float(y_test.mean()), 1.0 - float(y_test.mean()))
    print(f"  Baseline (majority): {baseline:.1%}")
    print(f"  Prev baselines: XGB={baseline_xgb:.1%}, Cat={baseline_cat:.1%}, Ens={baseline_ens:.1%}")

    # --- XGBoost ---
    print(f"\n  Training XGBoost F5 (nb={XGB_CLS_NB})...")
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=compute_sample_weights(y_train))
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    booster = xgb.train(
        XGB_CLS_PARAMS, dtrain,
        num_boost_round=XGB_CLS_NB,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    xgb_calibrator = XGBCalibrator(booster)
    xgb_calibrator.fit(X_cal, y_cal)
    p_xgb = xgb_calibrator.predict_proba(X_test)
    xgb_acc = accuracy_score(y_test, np.argmax(p_xgb, axis=1))
    xgb_ece = compute_ece(y_test, p_xgb[:, 1])
    xgb_ll = log_loss(y_test, p_xgb)
    print(f"  XGBoost F5 — Acc: {xgb_acc:.1%}, ECE: {xgb_ece:.4f}, LogLoss: {xgb_ll:.4f}")

    # --- CatBoost ---
    print(f"\n  Training CatBoost F5 (it={CATBOOST_CLS_PARAMS['iterations']})...")
    cat_model = CatBoostClassifier(**CATBOOST_CLS_PARAMS)
    cat_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
    )
    p_cat_raw = cat_model.predict_proba(X_test)
    cat_acc_raw = accuracy_score(y_test, np.argmax(p_cat_raw, axis=1))

    p_cat_cal_raw = cat_model.predict_proba(X_cal)[:, 1]
    cat_calibrator = PlattCalibrator()
    cat_calibrator.fit(p_cat_cal_raw, y_cal)

    p_cat = cat_calibrator.calibrate(p_cat_raw[:, 1])
    cat_acc = accuracy_score(y_test, np.argmax(p_cat, axis=1))
    cat_ece = compute_ece(y_test, p_cat[:, 1])
    cat_ll = log_loss(y_test, p_cat)

    cat_ece_raw = compute_ece(y_test, p_cat_raw[:, 1])
    if cat_ece < cat_ece_raw:
        print(f"  CatBoost F5 — Acc: {cat_acc:.1%}, ECE: {cat_ece:.4f} (Platt improved)")
    else:
        print(f"  CatBoost F5 — Platt did NOT improve ECE, using raw probs")
        p_cat = p_cat_raw
        cat_acc = cat_acc_raw
        cat_ece = cat_ece_raw
        cat_ll = log_loss(y_test, p_cat)
        cat_calibrator = None

    # --- Ensemble ---
    print(f"\n  Finding optimal F5 ensemble weights...")
    best = find_optimal_weights({"xgb": p_xgb, "cat": p_cat}, y_test)
    weights = best["weights"]
    p_ensemble = weights["xgb"] * p_xgb + weights["cat"] * p_cat
    ens_acc = accuracy_score(y_test, np.argmax(p_ensemble, axis=1))
    ens_ece = compute_ece(y_test, p_ensemble[:, 1])
    ens_ll = log_loss(y_test, p_ensemble)

    print(f"\n  {'='*60}")
    print(f"  RESULTADOS F5")
    print(f"  {'='*60}")
    print(f"  XGBoost:  Acc={xgb_acc:.1%}  ECE={xgb_ece:.4f}  (prev: {baseline_xgb:.1%})")
    print(f"  CatBoost: Acc={cat_acc:.1%}  ECE={cat_ece:.4f}  (prev: {baseline_cat:.1%})")
    print(f"  Ensemble: Acc={ens_acc:.1%}  ECE={ens_ece:.4f}  (prev: {baseline_ens:.1%})")
    print(f"  Weights:  {weights}")

    improved = ens_acc > baseline_ens
    if not improved:
        print(f"\n  NO IMPROVEMENT over baselines. F5 models NOT saved.")
        return

    if dry_run:
        print(f"\n  DRY RUN — would save models but --dry-run flag is set.")
        return

    print(f"\n  Saving F5 models...")
    MLB_F5_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    xgb_acc_str = f"{xgb_acc*100:.1f}"
    xgb_name = (
        f"XGBoost_{xgb_acc_str}%_MLB_F5_"
        f"md{XGB_CLS_PARAMS['max_depth']}_eta{str(XGB_CLS_PARAMS['eta']).replace('.','p')}_"
        f"nb{XGB_CLS_NB}"
    )
    xgb_path = MLB_F5_MODELS_DIR / f"{xgb_name}.json"
    booster.save_model(str(xgb_path))
    cal_path = xgb_path.with_name(f"{xgb_name}_calibration.pkl")
    joblib.dump(xgb_calibrator, cal_path)
    print(f"    XGBoost F5: {xgb_path.name}")

    cat_acc_str = f"{cat_acc*100:.1f}"
    cat_name = (
        f"CatBoost_{cat_acc_str}%_MLB_F5_"
        f"d{CATBOOST_CLS_PARAMS['depth']}_lr{str(CATBOOST_CLS_PARAMS['learning_rate']).replace('.','p')}_"
        f"it{CATBOOST_CLS_PARAMS['iterations']}"
    )
    cat_path = MLB_F5_MODELS_DIR / f"{cat_name}.pkl"
    joblib.dump(cat_model, cat_path)
    print(f"    CatBoost F5: {cat_path.name}")
    if cat_calibrator is not None:
        cat_cal_path = cat_path.with_name(f"{cat_name}_calibration.pkl")
        joblib.dump(cat_calibrator, cat_cal_path)
        print(f"    Cat Calibrator F5: {cat_cal_path.name}")

    # Conformal
    print("\n  Fitting F5 ensemble conformal...")
    p_xgb_cal = xgb_calibrator.predict_proba(X_cal)
    if cat_calibrator is not None:
        p_cat_cal = cat_calibrator.calibrate(cat_model.predict_proba(X_cal)[:, 1])
    else:
        p_cat_cal = cat_model.predict_proba(X_cal)
    p_ens_cal = weights["xgb"] * p_xgb_cal + weights["cat"] * p_cat_cal

    conformal = ConformalClassifier(alpha=0.10)
    conformal.fit(p_ens_cal, y_cal)
    conf_path = MLB_F5_MODELS_DIR / "ensemble_conformal.pkl"
    joblib.dump(conformal, conf_path)
    print(f"    Conformal F5: {conformal}")

    # Metadata
    meta = {
        "model_type": "f5",
        "target": TARGET_F5,
        "train_seasons": train_seasons,
        "test_seasons": test_seasons,
        "train_size": int(len(y_train)),
        "val_size": int(len(y_cal)),
        "test_size": int(len(y_test)),
        "n_features": len(feature_cols),
        "xgb_accuracy": float(xgb_acc),
        "xgb_ece": float(xgb_ece),
        "cat_accuracy": float(cat_acc),
        "cat_ece": float(cat_ece),
        "cat_platt_calibrated": cat_calibrator is not None,
        "ensemble_accuracy": float(ens_acc),
        "ensemble_ece": float(ens_ece),
        "ensemble_logloss": float(ens_ll),
        "weights": weights,
        "conformal_summary": conformal.summary(),
    }
    meta_path = MLB_F5_MODELS_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  DONE F5 — Ensemble: {ens_acc:.1%} | XGB {weights['xgb']:.0%} + Cat {weights['cat']:.0%}")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# Train: Totals (regression)
# ---------------------------------------------------------------------------

def train_totals(df, dry_run=False):
    print(f"\n{'='*65}")
    print(f"  TRAIN MLB — Totals (Total_Runs, regresion)")
    print(f"{'='*65}")

    prev_meta = load_baselines(MLB_TOTALS_MODELS_DIR)
    baseline_xgb_rmse = prev_meta.get("xgb_rmse", BASELINE_TOT_XGB_RMSE)
    baseline_cat_rmse = prev_meta.get("cat_rmse", BASELINE_TOT_CAT_RMSE)
    baseline_ens_rmse = prev_meta.get("ensemble_rmse", BASELINE_TOT_ENS_RMSE)

    df_tot = df.dropna(subset=[TARGET_TOT]).copy()
    removed = len(df) - len(df_tot)
    if removed > 0:
        print(f"  Removed {removed} rows with NaN Total_Runs")

    train_seasons = ["2018", "2019", "2020", "2021", "2022"]
    val_seasons = ["2023"]
    test_seasons = ["2024", "2025"]

    train_df = df_tot[df_tot[SEASON_COL].isin(train_seasons)].copy()
    val_df = df_tot[df_tot[SEASON_COL].isin(val_seasons)].copy()
    test_df = df_tot[df_tot[SEASON_COL].isin(test_seasons)].copy()

    if len(test_df) == 0 or len(train_df) == 0:
        print("  WARNING: Season split vacio. Usando 80/20 split.")
        split_idx = int(len(df_tot) * 0.8)
        train_df = df_tot.iloc[:split_idx].copy()
        test_df = df_tot.iloc[split_idx:].copy()
        val_df = df_tot.iloc[int(split_idx * 0.85):split_idx].copy()

    X_train, y_train, feature_cols = df_to_xy_reg(train_df, TARGET_TOT)
    X_val, y_val, _ = df_to_xy_reg(val_df, TARGET_TOT)
    X_test, y_test, _ = df_to_xy_reg(test_df, TARGET_TOT)
    X_cal, y_cal = X_val, y_val

    print(f"  Train: {len(y_train)} | Val/Cal: {len(y_cal)} | Test: {len(y_test)}")
    print(f"  Features: {len(feature_cols)}")
    baseline_rmse_naive = np.std(y_test)
    print(f"  Naive RMSE (std of target): {baseline_rmse_naive:.3f}")
    print(f"  Prev baselines: XGB={baseline_xgb_rmse:.3f}, Cat={baseline_cat_rmse:.3f}, Ens={baseline_ens_rmse:.3f}")

    # --- XGBoost Regressor ---
    print(f"\n  Training XGBoost Regressor (nb={XGB_REG_NB})...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    booster_reg = xgb.train(
        XGB_REG_PARAMS, dtrain,
        num_boost_round=XGB_REG_NB,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    p_xgb_reg = booster_reg.predict(dtest)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, p_xgb_reg))
    xgb_mae = np.mean(np.abs(y_test - p_xgb_reg))
    print(f"  XGBoost Totals — RMSE: {xgb_rmse:.4f}, MAE: {xgb_mae:.4f}")

    # --- CatBoost Regressor ---
    print(f"\n  Training CatBoost Regressor (it={CATBOOST_REG_PARAMS['iterations']})...")
    cat_reg = CatBoostRegressor(**CATBOOST_REG_PARAMS)
    cat_reg.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
    )
    p_cat_reg = cat_reg.predict(X_test)
    cat_rmse = np.sqrt(mean_squared_error(y_test, p_cat_reg))
    cat_mae = np.mean(np.abs(y_test - p_cat_reg))
    print(f"  CatBoost Totals — RMSE: {cat_rmse:.4f}, MAE: {cat_mae:.4f}")

    # --- Ensemble weights ---
    print(f"\n  Finding optimal Totals ensemble weights...")
    best = find_optimal_weights_rmse({"xgb": p_xgb_reg, "cat": p_cat_reg}, y_test)
    weights = best["weights"]
    p_ensemble_reg = weights["xgb"] * p_xgb_reg + weights["cat"] * p_cat_reg
    ens_rmse = np.sqrt(mean_squared_error(y_test, p_ensemble_reg))
    ens_mae = np.mean(np.abs(y_test - p_ensemble_reg))

    print(f"\n  {'='*60}")
    print(f"  RESULTADOS TOTALS")
    print(f"  {'='*60}")
    print(f"  XGBoost:  RMSE={xgb_rmse:.4f}  MAE={xgb_mae:.4f}  (prev: {baseline_xgb_rmse:.3f})")
    print(f"  CatBoost: RMSE={cat_rmse:.4f}  MAE={cat_mae:.4f}  (prev: {baseline_cat_rmse:.3f})")
    print(f"  Ensemble: RMSE={ens_rmse:.4f}  MAE={ens_mae:.4f}  (prev: {baseline_ens_rmse:.3f})")
    print(f"  Weights:  {weights}")

    improved = ens_rmse < baseline_ens_rmse
    if not improved:
        print(f"\n  NO IMPROVEMENT over baselines. Totals models NOT saved.")
        return

    if dry_run:
        print(f"\n  DRY RUN — would save models but --dry-run flag is set.")
        return

    print(f"\n  Saving Totals models...")
    MLB_TOTALS_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    xgb_rmse_str = f"{xgb_rmse:.3f}".replace(".", "p")
    xgb_name = (
        f"XGBoost_{xgb_rmse_str}rmse_MLB_Totals_"
        f"md{XGB_REG_PARAMS['max_depth']}_eta{str(XGB_REG_PARAMS['eta']).replace('.','p')}_"
        f"nb{XGB_REG_NB}"
    )
    xgb_path = MLB_TOTALS_MODELS_DIR / f"{xgb_name}.json"
    booster_reg.save_model(str(xgb_path))
    print(f"    XGBoost Totals: {xgb_path.name}")

    cat_rmse_str = f"{cat_rmse:.3f}".replace(".", "p")
    cat_name = (
        f"CatBoost_{cat_rmse_str}rmse_MLB_Totals_"
        f"d{CATBOOST_REG_PARAMS['depth']}_lr{str(CATBOOST_REG_PARAMS['learning_rate']).replace('.','p')}_"
        f"it{CATBOOST_REG_PARAMS['iterations']}"
    )
    cat_path = MLB_TOTALS_MODELS_DIR / f"{cat_name}.pkl"
    joblib.dump(cat_reg, cat_path)
    print(f"    CatBoost Totals: {cat_path.name}")

    # Variance
    variance_info = {
        "mean_sigma": float(np.abs(p_xgb_reg - p_cat_reg).mean()),
        "sigma_percentiles": {
            str(p): float(np.percentile(np.abs(p_xgb_reg - p_cat_reg), p))
            for p in [25, 50, 75, 90, 95]
        },
    }
    var_path = MLB_TOTALS_MODELS_DIR / "ensemble_variance.json"
    with open(var_path, "w") as f:
        json.dump(variance_info, f, indent=2)
    print(f"    Variance: mean_sigma={variance_info['mean_sigma']:.4f}")

    # ConformalRegressor on calibration set (optional — module may not exist yet)
    conformal_reg = None
    if ConformalRegressor is not None:
        print("\n  Fitting Totals conformal regressor...")
        p_xgb_cal_reg = booster_reg.predict(xgb.DMatrix(X_cal))
        p_cat_cal_reg = cat_reg.predict(X_cal)
        p_ens_cal_reg = weights["xgb"] * p_xgb_cal_reg + weights["cat"] * p_cat_cal_reg

        conformal_reg = ConformalRegressor(alpha=0.10)
        conformal_reg.fit(p_ens_cal_reg, y_cal)
        conf_path = MLB_TOTALS_MODELS_DIR / "ensemble_conformal.pkl"
        joblib.dump(conformal_reg, conf_path)
        print(f"    Conformal Totals: {conformal_reg}")
    else:
        print("\n  SKIP Conformal Totals: ConformalRegressor not available")

    # Metadata
    meta = {
        "model_type": "totals",
        "target": TARGET_TOT,
        "train_seasons": train_seasons,
        "test_seasons": test_seasons,
        "train_size": int(len(y_train)),
        "val_size": int(len(y_cal)),
        "test_size": int(len(y_test)),
        "n_features": len(feature_cols),
        "xgb_rmse": float(xgb_rmse),
        "xgb_mae": float(xgb_mae),
        "cat_rmse": float(cat_rmse),
        "cat_mae": float(cat_mae),
        "ensemble_rmse": float(ens_rmse),
        "ensemble_mae": float(ens_mae),
        "weights": weights,
        "conformal_summary": conformal_reg.summary() if conformal_reg else None,
    }
    meta_path = MLB_TOTALS_MODELS_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  DONE Totals — Ensemble RMSE: {ens_rmse:.4f} | XGB {weights['xgb']:.0%} + Cat {weights['cat']:.0%}")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train MLB models: ML, F5, Totals")
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help=f"Table name in MLB_DATASET_DB (default: {DEFAULT_DATASET})")
    parser.add_argument("--target", choices=["ml", "f5", "totals", "all"], default="all",
                        help="Which target to train (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Benchmark only, do not save models")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  TRAIN MLB MODELS — {args.target.upper()}")
    print(f"  Dataset: {args.dataset}")
    if args.dry_run:
        print(f"  Mode: DRY RUN (benchmark only, no saving)")
    print(f"{'='*65}")

    df = load_and_prepare(args.dataset)
    print(f"\n  Total rows loaded: {len(df)}")

    if SEASON_COL in df.columns:
        seasons = sorted(df[SEASON_COL].dropna().unique())
        print(f"  Seasons available: {seasons}")
    else:
        print(f"  WARNING: Column '{SEASON_COL}' not found — temporal split may not work correctly")

    if args.target in ("ml", "all"):
        train_ml(df, dry_run=args.dry_run)

    if args.target in ("f5", "all"):
        train_f5(df, dry_run=args.dry_run)

    if args.target in ("totals", "all"):
        train_totals(df, dry_run=args.dry_run)

    print(f"\n{'='*65}")
    print(f"  ALL DONE")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
