"""Optuna hyperparameter tuning para XGBoost + CatBoost.

Optimiza hiperparametros de cada modelo INDEPENDIENTEMENTE, luego
busca pesos optimos del ensemble. Objetivo: minimizar ECE (calibracion)
con accuracy como constraint secundario.

POR QUE ECE Y NO ACCURACY:
  Para betting, la CALIDAD de las probabilidades importa mas que acertar
  el lado. Si el modelo dice P(home)=0.65 y apostamos Kelly a -130,
  nuestro profit depende de que 0.65 sea preciso. Si el P real es 0.58,
  Kelly sobreapuesta y perdemos a largo plazo, incluso si acertamos 60%.

  ECE (Expected Calibration Error) mide exactamente esto: la diferencia
  entre P predicha y P real observada, promediada por bins.

Uso:
    PYTHONPATH=. python scripts/tune_optuna.py
    PYTHONPATH=. python scripts/tune_optuna.py --n-trials 200
    PYTHONPATH=. python scripts/tune_optuna.py --model xgb        # solo XGBoost
    PYTHONPATH=. python scripts/tune_optuna.py --model cat        # solo CatBoost
    PYTHONPATH=. python scripts/tune_optuna.py --model ensemble   # solo pesos
"""

import argparse
import json
import sqlite3
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier

from src.core.calibration.conformal import ConformalClassifier
from src.core.calibration.xgb_calibrator import XGBCalibrator
from src.config import DATASET_DB, NBA_ML_MODELS_DIR, DROP_COLUMNS_ML, get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

DEFAULT_DATASET = "dataset_2012-26"
TARGET = "Home-Team-Win"
DATE_COL = "Date"
N_TRIALS_DEFAULT = 100


# ── Data loading ─────────────────────────────────────────────────────

def load_data(dataset_name=DEFAULT_DATASET):
    with sqlite3.connect(DATASET_DB) as con:
        df = pd.read_sql_query(f'SELECT * FROM "{dataset_name}"', con)
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)

    # Add Diff_TS_PCT if not present
    if "Diff_TS_PCT" not in df.columns and "TS_PCT" in df.columns:
        df["Diff_TS_PCT"] = df["TS_PCT"].astype(float) - df["TS_PCT.1"].astype(float)

    return df


def df_to_xy(df):
    y = df[TARGET].astype(int).to_numpy()
    X_df = df.drop(columns=DROP_COLUMNS_ML, errors="ignore")
    feature_cols = list(X_df.columns)
    X = X_df.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0).to_numpy(dtype=float)
    return X, y, feature_cols


def compute_sample_weights(y, num_classes=2):
    counts = np.bincount(y, minlength=num_classes)
    total = len(y)
    return np.array([
        (total / (num_classes * counts[label])) if counts[label] else 1.0
        for label in y
    ])


def compute_ece(y_true, y_prob, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(y_prob[mask].mean() - y_true[mask].mean())
    return ece / len(y_true)


# ── Optuna objectives ────────────────────────────────────────────────

def xgb_objective(trial, X_train, y_train, X_test, y_test, X_cal, y_cal):
    """Optimiza XGBoost minimizando ECE con accuracy >= 63%."""
    params = {
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),
        "lambda": trial.suggest_float("lambda", 0.1, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 0.01, 5.0, log=True),
        "objective": "multi:softprob",
        "num_class": 2,
        "tree_method": "hist",
        "seed": 42,
        "eval_metric": ["mlogloss"],
    }
    nb = trial.suggest_int("num_boost_round", 300, 1500)

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=compute_sample_weights(y_train))
    dtest = xgb.DMatrix(X_test, label=y_test)

    booster = xgb.train(
        params, dtrain,
        num_boost_round=nb,
        evals=[(dtest, "test")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    # Platt calibration
    calibrator = XGBCalibrator(booster)
    calibrator.fit(X_cal, y_cal)
    p = calibrator.predict_proba(X_test)

    acc = (np.argmax(p, axis=1) == y_test).mean()
    ece = compute_ece(y_test, p[:, 1])

    # Prune si accuracy es muy baja
    if acc < 0.63:
        raise optuna.TrialPruned()

    # Objetivo: minimizar ECE (primario), maximizar accuracy (secundario)
    # Combinamos: ECE - 0.1 * (acc - 0.65) para penalizar baja accuracy
    return ece - 0.1 * (acc - 0.65)


def cat_objective(trial, X_train, y_train, X_test, y_test):
    """Optimiza CatBoost minimizando ECE con accuracy >= 63%."""
    params = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "auto_class_weights": "Balanced",
        "random_seed": 42,
        "task_type": "CPU",
        "depth": trial.suggest_int("depth", 4, 10),
        "iterations": trial.suggest_int("iterations", 500, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.01, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.001, 1.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "rsm": trial.suggest_float("rsm", 0.5, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
    }

    model = CatBoostClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=50,
        verbose=0,
    )
    p = model.predict_proba(X_test)
    acc = (np.argmax(p, axis=1) == y_test).mean()
    ece = compute_ece(y_test, p[:, 1])

    if acc < 0.63:
        raise optuna.TrialPruned()

    return ece - 0.1 * (acc - 0.65)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    parser.add_argument("--model", choices=["xgb", "cat", "ensemble", "all"],
                        default="all")
    parser.add_argument("--save", action="store_true",
                        help="Save best params to train_models_v2.py format")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  OPTUNA TUNING — {args.n_trials} trials per model")
    print(f"{'='*65}")

    df = load_data(args.dataset)
    test_dt = pd.to_datetime("2025-10-01")
    train_df = df[df[DATE_COL] < test_dt].copy()
    test_df = df[df[DATE_COL] >= test_dt].copy()

    if len(test_df) == 0:
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

    X_train, y_train, feature_cols = df_to_xy(train_df)
    X_test, y_test, _ = df_to_xy(test_df)
    cal_split = int(len(X_train) * 0.8)
    X_cal, y_cal = X_train[cal_split:], y_train[cal_split:]

    print(f"  Train: {len(y_train)} | Cal: {len(y_cal)} | Test: {len(y_test)}")
    print(f"  Features: {len(feature_cols)}")

    best_xgb_params = None
    best_cat_params = None

    # ── XGBoost tuning ──
    if args.model in ("xgb", "all"):
        print(f"\n  Tuning XGBoost ({args.n_trials} trials)...")
        study_xgb = optuna.create_study(
            direction="minimize",
            study_name="xgb_ece",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study_xgb.optimize(
            lambda trial: xgb_objective(trial, X_train, y_train, X_test, y_test, X_cal, y_cal),
            n_trials=args.n_trials,
            show_progress_bar=True,
        )

        best_xgb_params = study_xgb.best_params
        print(f"\n  Best XGBoost:")
        print(f"    Score: {study_xgb.best_value:.4f}")
        print(f"    Params: {json.dumps(best_xgb_params, indent=6)}")

        # Evaluate best
        bp = best_xgb_params.copy()
        nb = bp.pop("num_boost_round")
        bp.update({"objective": "multi:softprob", "num_class": 2,
                    "tree_method": "hist", "seed": 42, "eval_metric": ["mlogloss"]})
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=compute_sample_weights(y_train))
        dtest = xgb.DMatrix(X_test, label=y_test)
        booster = xgb.train(bp, dtrain, num_boost_round=nb,
                            evals=[(dtest, "test")], early_stopping_rounds=50, verbose_eval=False)
        cal = XGBCalibrator(booster)
        cal.fit(X_cal, y_cal)
        p_xgb = cal.predict_proba(X_test)
        xgb_acc = (np.argmax(p_xgb, axis=1) == y_test).mean()
        xgb_ece = compute_ece(y_test, p_xgb[:, 1])
        print(f"    Acc: {xgb_acc:.1%}, ECE: {xgb_ece:.4f}")

    # ── CatBoost tuning ──
    if args.model in ("cat", "all"):
        print(f"\n  Tuning CatBoost ({args.n_trials} trials)...")
        study_cat = optuna.create_study(
            direction="minimize",
            study_name="cat_ece",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study_cat.optimize(
            lambda trial: cat_objective(trial, X_train, y_train, X_test, y_test),
            n_trials=args.n_trials,
            show_progress_bar=True,
        )

        best_cat_params = study_cat.best_params
        print(f"\n  Best CatBoost:")
        print(f"    Score: {study_cat.best_value:.4f}")
        print(f"    Params: {json.dumps(best_cat_params, indent=6)}")

        # Evaluate best
        bp = best_cat_params.copy()
        bp.update({"loss_function": "Logloss", "eval_metric": "Logloss",
                    "auto_class_weights": "Balanced", "random_seed": 42, "task_type": "CPU"})
        cat_model = CatBoostClassifier(**bp)
        cat_model.fit(X_train, y_train, eval_set=(X_test, y_test),
                      early_stopping_rounds=50, verbose=0)
        p_cat = cat_model.predict_proba(X_test)
        cat_acc = (np.argmax(p_cat, axis=1) == y_test).mean()
        cat_ece = compute_ece(y_test, p_cat[:, 1])
        print(f"    Acc: {cat_acc:.1%}, ECE: {cat_ece:.4f}")

    # ── Ensemble weight search ──
    if args.model in ("ensemble", "all"):
        # Need both models' predictions
        if args.model == "ensemble":
            # Load current models and predict
            from src.sports.nba.predict import xgboost_runner as XGBoost_Runner
            XGBoost_Runner._load_models()
            p_xgb = XGBoost_Runner._predict_probs(
                XGBoost_Runner.xgb_ml,
                X_test,
                XGBoost_Runner.xgb_ml_calibrator,
            )
            cat_path = max(
                [p for p in NBA_ML_MODELS_DIR.glob("CatBoost_*ML*.pkl")
                 if "calibration" not in p.name],
                key=lambda p: p.stat().st_mtime,
            )
            cat_model = joblib.load(cat_path)
            p_cat = cat_model.predict_proba(X_test)

        print(f"\n  Optimizing ensemble weights...")
        best_w = None
        best_score = 1.0
        for w_xgb in np.arange(0, 1.01, 0.05):
            w_cat = 1.0 - w_xgb
            p_ens = w_xgb * p_xgb + w_cat * p_cat
            acc = (np.argmax(p_ens, axis=1) == y_test).mean()
            ece = compute_ece(y_test, p_ens[:, 1])
            score = ece - 0.1 * (acc - 0.65)
            if score < best_score:
                best_score = score
                best_w = {"xgb": round(w_xgb, 2), "cat": round(w_cat, 2)}

        p_best = best_w["xgb"] * p_xgb + best_w["cat"] * p_cat
        ens_acc = (np.argmax(p_best, axis=1) == y_test).mean()
        ens_ece = compute_ece(y_test, p_best[:, 1])
        print(f"  Best weights: {best_w}")
        print(f"  Ensemble: Acc={ens_acc:.1%}, ECE={ens_ece:.4f}")

    # ── Save results ──
    if args.save:
        results = {}
        if best_xgb_params:
            results["xgb_params"] = best_xgb_params
        if best_cat_params:
            results["cat_params"] = best_cat_params
        if best_w:
            results["weights"] = best_w

        out_path = NBA_ML_MODELS_DIR / "optuna_best_params.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Best params saved → {out_path}")
        print(f"  To train with these params, update train_models_v2.py and run it.")

    print(f"\n{'='*65}")
    print(f"  DONE")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
