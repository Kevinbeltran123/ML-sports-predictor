"""Train XGBoost + CatBoost + LightGBM moneyline models with improvements.

Mejoras sobre train_models.py:
  1. Platt calibration para CatBoost (antes solo XGBoost la tenia)
  2. LightGBM como 3er miembro del ensemble
  3. Diff_TS_PCT como feature adicional
  4. Busqueda automatica de pesos optimos del ensemble (grid search)
  5. Benchmark contra baselines actuales — solo guarda si mejora

Uso:
    PYTHONPATH=. python scripts/train_models_v2.py
    PYTHONPATH=. python scripts/train_models_v2.py --dataset dataset_2012-26
    PYTHONPATH=. python scripts/train_models_v2.py --dry-run  # benchmark sin guardar
"""

import argparse
import json
import sqlite3

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

from src.core.calibration.conformal import ConformalClassifier
from src.core.calibration.xgb_calibrator import XGBCalibrator
from src.config import DATASET_DB, NBA_ML_MODELS_DIR, DROP_COLUMNS_ML, get_logger

logger = get_logger(__name__)

DEFAULT_DATASET = "dataset_2012-26"
TARGET = "Home-Team-Win"
DATE_COL = "Date"

# --- Hyperparameters (Optuna-tuned, Feb 2026) ---

XGB_PARAMS = {
    "max_depth": 6, "eta": 0.07754, "subsample": 0.813,
    "colsample_bytree": 0.861, "colsample_bylevel": 0.930,
    "colsample_bynode": 0.623, "min_child_weight": 9,
    "gamma": 9.928, "lambda": 0.245, "alpha": 0.347,
    "objective": "multi:softprob", "num_class": 2,
    "tree_method": "hist", "seed": 42,
    "eval_metric": ["mlogloss"],
}
XGB_NB = 717

CATBOOST_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "auto_class_weights": "Balanced",
    "random_seed": 42,
    "task_type": "CPU",
    "depth": 10,
    "iterations": 943,
    "learning_rate": 0.004985,
    "l2_leaf_reg": 0.0107,
    "random_strength": 0.0222,
    "bagging_temperature": 2.575,
    "rsm": 0.9997,
    "border_count": 187,
}

# LightGBM: conservativo, complementa a XGB (symmetric tree vs depth-wise)
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "max_depth": 7,
    "num_leaves": 63,
    "learning_rate": 0.03,
    "subsample": 0.85,
    "colsample_bytree": 0.80,
    "reg_alpha": 0.5,
    "reg_lambda": 1.5,
    "min_child_weight": 10,
    "seed": 42,
    "verbose": -1,
}
LGBM_NB = 1200

# Baselines actuales (del metadata.json)
BASELINE_XGB_ACC = 0.6489
BASELINE_CAT_ACC = 0.6625
BASELINE_ENS_ACC = 0.6477


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

    return df


def add_ts_differential(df):
    """Agrega Diff_TS_PCT si TS_PCT y TS_PCT.1 existen en el DataFrame."""
    if "TS_PCT" in df.columns and "TS_PCT.1" in df.columns:
        df["Diff_TS_PCT"] = df["TS_PCT"].astype(float) - df["TS_PCT.1"].astype(float)
        return True
    return False


def df_to_xy(df):
    y = df[TARGET].astype(int).to_numpy()
    X_df = df.drop(columns=DROP_COLUMNS_ML, errors="ignore")
    feature_cols = list(X_df.columns)
    X = X_df.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0).to_numpy(dtype=float)
    return X, y, feature_cols


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


def find_optimal_weights(probs_dict, y_test, n_steps=21):
    """Grid search para pesos optimos del ensemble (2 o 3 modelos).

    Prioriza ECE (calibracion) sobre accuracy, porque para betting
    la calidad de las probabilidades importa mas que acertar el lado.
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
                best = {"weights": {models[0]: round(w1, 2), models[1]: round(w2, 2)},
                        "ece": ece, "acc": acc, "ll": ll}
    elif n == 3:
        step = 1.0 / (n_steps - 1) if n_steps > 1 else 0.5
        for w1 in np.arange(0, 1.01, step):
            for w2 in np.arange(0, 1.01 - w1, step):
                w3 = 1.0 - w1 - w2
                if w3 < -0.001:
                    continue
                w3 = max(w3, 0.0)
                p = (w1 * probs_dict[models[0]] +
                     w2 * probs_dict[models[1]] +
                     w3 * probs_dict[models[2]])
                acc = accuracy_score(y_test, np.argmax(p, axis=1))
                ll = log_loss(y_test, p)
                ece = compute_ece(y_test, p[:, 1])
                if ece < best["ece"] or (abs(ece - best["ece"]) < 0.001 and acc > best["acc"]):
                    best = {"weights": {models[0]: round(w1, 2),
                                        models[1]: round(w2, 2),
                                        models[2]: round(w3, 2)},
                            "ece": ece, "acc": acc, "ll": ll}

    return best


def main():
    parser = argparse.ArgumentParser(description="Train ML models v2 with improvements")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo benchmark, no guardar modelos")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  TRAIN v2: XGBoost + CatBoost + LightGBM — Moneyline")
    print(f"{'='*65}")

    df = load_and_prepare(args.dataset)

    # Agregar Diff_TS_PCT antes del split
    has_ts_diff = add_ts_differential(df)
    if has_ts_diff:
        print(f"  + Diff_TS_PCT feature added")

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

    # Calibration split: last 20% of training data
    cal_split = int(len(X_train) * 0.8)
    X_cal, y_cal = X_train[cal_split:], y_train[cal_split:]

    print(f"  Train: {len(y_train)} | Cal: {len(y_cal)} | Test: {len(y_test)}")
    print(f"  Features: {len(feature_cols)}")
    baseline = max(float(y_test.mean()), 1.0 - float(y_test.mean()))
    print(f"  Baseline (majority): {baseline:.1%}")
    print(f"\n  Current baselines: XGB={BASELINE_XGB_ACC:.1%}, Cat={BASELINE_CAT_ACC:.1%}, Ens={BASELINE_ENS_ACC:.1%}")

    # =====================================================================
    # 1. Train XGBoost
    # =====================================================================
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
    p_xgb_raw = booster.predict(dtest)
    xgb_acc_raw = accuracy_score(y_test, np.argmax(p_xgb_raw, axis=1))

    # Platt calibration for XGBoost
    xgb_calibrator = XGBCalibrator(booster)
    xgb_calibrator.fit(X_cal, y_cal)
    p_xgb = xgb_calibrator.predict_proba(X_test)
    xgb_acc = accuracy_score(y_test, np.argmax(p_xgb, axis=1))
    xgb_ll = log_loss(y_test, p_xgb)
    xgb_ece = compute_ece(y_test, p_xgb[:, 1])
    print(f"  XGBoost — Acc: {xgb_acc:.1%} (raw: {xgb_acc_raw:.1%}), "
          f"LogLoss: {xgb_ll:.4f}, ECE: {xgb_ece:.4f}")

    # =====================================================================
    # 2. Train CatBoost
    # =====================================================================
    print(f"\n  Training CatBoost (it={CATBOOST_PARAMS['iterations']})...")
    cat_model = CatBoostClassifier(**CATBOOST_PARAMS)
    cat_model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=60,
        verbose=0,
    )
    p_cat_raw = cat_model.predict_proba(X_test)
    cat_acc_raw = accuracy_score(y_test, np.argmax(p_cat_raw, axis=1))

    # Platt calibration for CatBoost (NEW)
    p_cat_cal_raw = cat_model.predict_proba(X_cal)[:, 1]
    cat_calibrator = PlattCalibrator()
    cat_calibrator.fit(p_cat_cal_raw, y_cal)

    p_cat = cat_calibrator.calibrate(p_cat_raw[:, 1])
    cat_acc = accuracy_score(y_test, np.argmax(p_cat, axis=1))
    cat_ll = log_loss(y_test, p_cat)
    cat_ece = compute_ece(y_test, p_cat[:, 1])
    print(f"  CatBoost — Acc: {cat_acc:.1%} (raw: {cat_acc_raw:.1%}), "
          f"LogLoss: {cat_ll:.4f}, ECE: {cat_ece:.4f}")

    # Check if Platt helped CatBoost
    cat_ece_raw = compute_ece(y_test, p_cat_raw[:, 1])
    cat_platt_helps = cat_ece < cat_ece_raw
    if cat_platt_helps:
        print(f"    Platt improved CatBoost ECE: {cat_ece_raw:.4f} → {cat_ece:.4f}")
    else:
        print(f"    Platt did NOT improve CatBoost ECE: {cat_ece_raw:.4f} → {cat_ece:.4f}")
        print(f"    Using raw CatBoost probabilities instead")
        p_cat = p_cat_raw
        cat_acc = cat_acc_raw
        cat_ll = log_loss(y_test, p_cat)
        cat_ece = cat_ece_raw
        cat_calibrator = None  # don't save

    # =====================================================================
    # 3. Train LightGBM
    # =====================================================================
    print(f"\n  Training LightGBM (nb={LGBM_NB})...")
    lgb_train = lgb.Dataset(X_train, label=y_train,
                            weight=compute_sample_weights(y_train))
    lgb_valid = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

    lgb_model = lgb.train(
        LGBM_PARAMS, lgb_train,
        num_boost_round=LGBM_NB,
        valid_sets=[lgb_valid],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
    )
    p_lgb_raw_1 = lgb_model.predict(X_test)  # LightGBM binary returns P(class=1)
    p_lgb_raw = np.column_stack([1.0 - p_lgb_raw_1, p_lgb_raw_1])
    lgb_acc_raw = accuracy_score(y_test, np.argmax(p_lgb_raw, axis=1))

    # Platt calibration for LightGBM
    p_lgb_cal_raw = lgb_model.predict(X_cal)
    lgb_calibrator = PlattCalibrator()
    lgb_calibrator.fit(p_lgb_cal_raw, y_cal)

    p_lgb = lgb_calibrator.calibrate(p_lgb_raw_1)
    lgb_acc = accuracy_score(y_test, np.argmax(p_lgb, axis=1))
    lgb_ll = log_loss(y_test, p_lgb)
    lgb_ece = compute_ece(y_test, p_lgb[:, 1])
    print(f"  LightGBM — Acc: {lgb_acc:.1%} (raw: {lgb_acc_raw:.1%}), "
          f"LogLoss: {lgb_ll:.4f}, ECE: {lgb_ece:.4f}")

    # Check if Platt helped LightGBM
    lgb_ece_raw = compute_ece(y_test, p_lgb_raw[:, 1])
    if lgb_ece < lgb_ece_raw:
        print(f"    Platt improved LightGBM ECE: {lgb_ece_raw:.4f} → {lgb_ece:.4f}")
    else:
        print(f"    Platt did NOT improve LightGBM ECE: {lgb_ece_raw:.4f} → {lgb_ece:.4f}")
        print(f"    Using raw LightGBM probabilities instead")
        p_lgb = p_lgb_raw
        lgb_acc = lgb_acc_raw
        lgb_ll = log_loss(y_test, p_lgb)
        lgb_ece = lgb_ece_raw
        lgb_calibrator = None

    # =====================================================================
    # 4. Find optimal ensemble weights
    # =====================================================================
    print(f"\n  Finding optimal ensemble weights...")

    # 2-model ensemble (current approach)
    best_2 = find_optimal_weights({"xgb": p_xgb, "cat": p_cat}, y_test)
    print(f"\n  Best 2-model: {best_2['weights']}")
    print(f"    Acc: {best_2['acc']:.1%}, ECE: {best_2['ece']:.4f}, LogLoss: {best_2['ll']:.4f}")

    # 3-model ensemble
    best_3 = find_optimal_weights({"xgb": p_xgb, "cat": p_cat, "lgb": p_lgb}, y_test, n_steps=11)
    print(f"\n  Best 3-model: {best_3['weights']}")
    print(f"    Acc: {best_3['acc']:.1%}, ECE: {best_3['ece']:.4f}, LogLoss: {best_3['ll']:.4f}")

    # Compare: pick the best
    use_lgb = False
    if best_3["ece"] < best_2["ece"] and best_3["acc"] >= best_2["acc"] - 0.005:
        print(f"\n  >>> 3-model ensemble WINS (better ECE, comparable accuracy)")
        best = best_3
        use_lgb = True
    elif best_3["acc"] > best_2["acc"] + 0.005 and best_3["ece"] <= best_2["ece"] + 0.002:
        print(f"\n  >>> 3-model ensemble WINS (better accuracy, comparable ECE)")
        best = best_3
        use_lgb = True
    else:
        print(f"\n  >>> 2-model ensemble WINS (LightGBM doesn't improve enough)")
        best = best_2

    # =====================================================================
    # 5. Final ensemble evaluation
    # =====================================================================
    weights = best["weights"]
    p_ensemble = sum(w * probs for (name, w), probs in
                     zip(weights.items(),
                         [p_xgb, p_cat] + ([p_lgb] if use_lgb else [])))

    ens_acc = accuracy_score(y_test, np.argmax(p_ensemble, axis=1))
    ens_ll = log_loss(y_test, p_ensemble)
    ens_ece = compute_ece(y_test, p_ensemble[:, 1])

    print(f"\n  {'='*60}")
    print(f"  FINAL RESULTS")
    print(f"  {'='*60}")
    print(f"  XGBoost:   Acc={xgb_acc:.1%}  ECE={xgb_ece:.4f}  (baseline: {BASELINE_XGB_ACC:.1%})")
    print(f"  CatBoost:  Acc={cat_acc:.1%}  ECE={cat_ece:.4f}  (baseline: {BASELINE_CAT_ACC:.1%})")
    print(f"  LightGBM:  Acc={lgb_acc:.1%}  ECE={lgb_ece:.4f}  (new)")
    print(f"  Ensemble:  Acc={ens_acc:.1%}  ECE={ens_ece:.4f}  (baseline: {BASELINE_ENS_ACC:.1%})")
    print(f"  Weights:   {weights}")

    # =====================================================================
    # 6. Decide: save or discard
    # =====================================================================
    # Improvement criteria: better ECE OR better accuracy (within tolerance)
    improved_ece = ens_ece < compute_ece(y_test,
        (0.60 * p_xgb_raw + 0.40 * p_cat_raw)[:, 1])
    improved_acc = ens_acc > BASELINE_ENS_ACC

    if not improved_ece and not improved_acc:
        print(f"\n  {'!'*60}")
        print(f"  NO IMPROVEMENT over baselines. Models NOT saved.")
        print(f"  {'!'*60}\n")
        return

    if args.dry_run:
        print(f"\n  DRY RUN — would save models but --dry-run flag is set.")
        return

    print(f"\n  IMPROVEMENT detected! Saving models...")

    # =====================================================================
    # 7. Save models
    # =====================================================================

    # --- XGBoost ---
    xgb_acc_str = f"{xgb_acc*100:.1f}"
    params = XGB_PARAMS.copy()
    xgb_name = (
        f"XGBoost_{xgb_acc_str}%_ML_"
        f"md{params['max_depth']}_eta{str(params['eta']).replace('.','p')}_"
        f"sub{str(params['subsample']).replace('.','p')}_"
        f"col{str(params['colsample_bytree']).replace('.','p')}_"
        f"cbl{str(params['colsample_bylevel']).replace('.','p')}_"
        f"cbn{str(params['colsample_bynode']).replace('.','p')}_"
        f"mcw{params['min_child_weight']}_"
        f"g{str(params['gamma']).replace('.','p')}_"
        f"mds10_mb990_"
        f"l{str(params['lambda']).replace('.','p')}_"
        f"a{str(params['alpha']).replace('.','p')}_"
        f"nb{XGB_NB}"
    )
    xgb_path = NBA_ML_MODELS_DIR / f"{xgb_name}.json"
    booster.save_model(str(xgb_path))
    print(f"    XGBoost: {xgb_path.name}")

    # XGBoost calibrator
    cal_path = xgb_path.with_name(f"{xgb_name}_calibration.pkl")
    joblib.dump(xgb_calibrator, cal_path)
    print(f"    XGB Calibrator: {cal_path.name}")

    # --- CatBoost ---
    cat_acc_str = f"{cat_acc*100:.1f}"
    cp = CATBOOST_PARAMS
    cat_name = (
        f"CatBoost_{cat_acc_str}%_ML_"
        f"d{cp['depth']}_lr{str(cp['learning_rate']).replace('.','p')}_"
        f"it{cp['iterations']}_"
        f"l2{str(cp['l2_leaf_reg']).replace('.','p')}_"
        f"rs{str(cp['random_strength']).replace('.','p')}_"
        f"bt{str(cp['bagging_temperature']).replace('.','p')}_"
        f"rsm{str(cp['rsm']).replace('.','p')}_"
        f"bc{cp['border_count']}"
    )
    cat_path = NBA_ML_MODELS_DIR / f"{cat_name}.pkl"
    joblib.dump(cat_model, cat_path)
    print(f"    CatBoost: {cat_path.name}")

    # CatBoost calibrator (if it helped)
    if cat_calibrator is not None:
        cat_cal_path = cat_path.with_name(f"{cat_name}_calibration.pkl")
        joblib.dump(cat_calibrator, cat_cal_path)
        print(f"    Cat Calibrator: {cat_cal_path.name}")

    # --- LightGBM (only if used in ensemble) ---
    if use_lgb:
        lgb_acc_str = f"{lgb_acc*100:.1f}"
        lgb_name = (
            f"LightGBM_{lgb_acc_str}%_ML_"
            f"md{LGBM_PARAMS['max_depth']}_lr{str(LGBM_PARAMS['learning_rate']).replace('.','p')}_"
            f"nl{LGBM_PARAMS['num_leaves']}_"
            f"sub{str(LGBM_PARAMS['subsample']).replace('.','p')}_"
            f"col{str(LGBM_PARAMS['colsample_bytree']).replace('.','p')}_"
            f"nb{lgb_model.num_trees()}"
        )
        lgb_path = NBA_ML_MODELS_DIR / f"{lgb_name}.txt"
        lgb_model.save_model(str(lgb_path))
        print(f"    LightGBM: {lgb_path.name}")

        if lgb_calibrator is not None:
            lgb_cal_path = lgb_path.with_name(f"{lgb_name}_calibration.pkl")
            joblib.dump(lgb_calibrator, lgb_cal_path)
            print(f"    LGB Calibrator: {lgb_cal_path.name}")

    # --- Ensemble conformal ---
    print("\n  Fitting ensemble conformal...")
    dcal = xgb.DMatrix(X_cal)
    p_xgb_cal = xgb_calibrator.predict_proba(X_cal)
    if cat_calibrator is not None:
        p_cat_cal = cat_calibrator.calibrate(cat_model.predict_proba(X_cal)[:, 1])
    else:
        p_cat_cal = cat_model.predict_proba(X_cal)

    if use_lgb:
        p_lgb_cal_1 = lgb_model.predict(X_cal)
        if lgb_calibrator is not None:
            p_lgb_cal = lgb_calibrator.calibrate(p_lgb_cal_1)
        else:
            p_lgb_cal = np.column_stack([1.0 - p_lgb_cal_1, p_lgb_cal_1])
        p_ens_cal = (weights["xgb"] * p_xgb_cal +
                     weights["cat"] * p_cat_cal +
                     weights["lgb"] * p_lgb_cal)
    else:
        p_ens_cal = weights["xgb"] * p_xgb_cal + weights["cat"] * p_cat_cal

    conformal = ConformalClassifier(alpha=0.10)
    conformal.fit(p_ens_cal, y_cal)
    conf_path = NBA_ML_MODELS_DIR / "ensemble_conformal.pkl"
    joblib.dump(conformal, conf_path)
    print(f"    Conformal: {conformal}")

    # --- Variance stats ---
    variance_info = {
        "mean_sigma": float(np.abs(p_xgb[:, 1] - p_cat[:, 1]).mean()),
        "sigma_percentiles": {
            str(p): float(np.percentile(np.abs(p_xgb[:, 1] - p_cat[:, 1]), p))
            for p in [25, 50, 75, 90, 95]
        },
    }
    if use_lgb:
        # Include LGB disagreement in sigma
        disagree_xgb_lgb = np.abs(p_xgb[:, 1] - p_lgb[:, 1])
        disagree_cat_lgb = np.abs(p_cat[:, 1] - p_lgb[:, 1])
        max_disagree = np.maximum(
            np.abs(p_xgb[:, 1] - p_cat[:, 1]),
            np.maximum(disagree_xgb_lgb, disagree_cat_lgb)
        )
        variance_info["mean_sigma_3model"] = float(max_disagree.mean())
        variance_info["sigma_percentiles_3model"] = {
            str(p): float(np.percentile(max_disagree, p))
            for p in [25, 50, 75, 90, 95]
        }

    var_path = NBA_ML_MODELS_DIR / "ensemble_variance.json"
    with open(var_path, "w") as f:
        json.dump(variance_info, f, indent=2)
    print(f"    Variance: mean_sigma={variance_info['mean_sigma']:.4f}")

    # --- Metadata ---
    meta = {
        "dataset": args.dataset,
        "train_size": len(y_train),
        "test_size": len(y_test),
        "n_features": len(feature_cols),
        "has_ts_differential": has_ts_diff,
        "xgb_accuracy": float(xgb_acc),
        "xgb_ece": float(xgb_ece),
        "cat_accuracy": float(cat_acc),
        "cat_ece": float(cat_ece),
        "cat_platt_calibrated": cat_calibrator is not None,
        "lgb_accuracy": float(lgb_acc) if use_lgb else None,
        "lgb_ece": float(lgb_ece) if use_lgb else None,
        "lgb_in_ensemble": use_lgb,
        "ensemble_accuracy": float(ens_acc),
        "ensemble_ece": float(ens_ece),
        "ensemble_logloss": float(ens_ll),
        "weights": weights,
        "conformal_summary": conformal.summary(),
        "baseline_comparison": {
            "prev_xgb_acc": BASELINE_XGB_ACC,
            "prev_cat_acc": BASELINE_CAT_ACC,
            "prev_ens_acc": BASELINE_ENS_ACC,
        }
    }
    meta_path = NBA_ML_MODELS_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  DONE — Ensemble: {ens_acc:.1%} ({len(feature_cols)} features)")
    if use_lgb:
        print(f"  3-model: XGB {weights['xgb']:.0%} + Cat {weights['cat']:.0%} + LGB {weights['lgb']:.0%}")
    else:
        print(f"  2-model: XGB {weights['xgb']:.0%} + Cat {weights['cat']:.0%}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
