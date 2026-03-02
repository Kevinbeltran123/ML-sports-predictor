"""Optuna hyperparameter tuning para XGBoost + CatBoost margin regression.

Optimiza hiperparametros de cada modelo INDEPENDIENTEMENTE, luego
busca pesos optimos del ensemble. Objetivo: minimizar RMSE con ATS
accuracy como constraint secundario.

Uso:
    PYTHONPATH=. python scripts/tune_margin_optuna.py
    PYTHONPATH=. python scripts/tune_margin_optuna.py --n-trials 200
    PYTHONPATH=. python scripts/tune_margin_optuna.py --model xgb
    PYTHONPATH=. python scripts/tune_margin_optuna.py --model cat
"""

import argparse
import json
import sqlite3
import warnings

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.config import DATASET_DB, NBA_MARGIN_MODELS_DIR, DROP_COLUMNS_MARGIN, get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Use enriched dataset (build_margin_features.py); fall back to base if missing.
DEFAULT_DATASET = "dataset_margin_enriched"
FALLBACK_DATASET = "dataset_2012-26"
TARGET = "Residual"
DATE_COL = "Date"
N_TRIALS_DEFAULT = 100
DECAY_LAMBDA = 0.9985

_DROP_COLUMNS_MARGIN = list(DROP_COLUMNS_MARGIN) + ["Residual"]


# ── Data loading ─────────────────────────────────────────────────────

def load_data(dataset_name=DEFAULT_DATASET):
    with sqlite3.connect(DATASET_DB) as con:
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        if dataset_name not in tables:
            logger.warning("Dataset '%s' not found. Falling back to '%s'. Run build_margin_features.py first.", dataset_name, FALLBACK_DATASET)
            dataset_name = FALLBACK_DATASET
        df = pd.read_sql_query(f'SELECT * FROM "{dataset_name}"', con)
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)

    if "Diff_TS_PCT" not in df.columns and "TS_PCT" in df.columns:
        df["Diff_TS_PCT"] = df["TS_PCT"].astype(float) - df["TS_PCT.1"].astype(float)

    if "Margin" not in df.columns:
        raise ValueError("Column 'Margin' not found in dataset.")
    if "MARKET_SPREAD" not in df.columns:
        raise ValueError("Column 'MARKET_SPREAD' not found in dataset.")

    df["Residual"] = df["Margin"] + df["MARKET_SPREAD"]
    df = df.dropna(subset=[TARGET, "MARKET_SPREAD"])
    return df


def df_to_xy(df):
    y = df[TARGET].astype(float).to_numpy()
    X_df = df.drop(columns=_DROP_COLUMNS_MARGIN, errors="ignore")
    feature_cols = list(X_df.columns)
    X = X_df.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0).to_numpy(dtype=float)
    return X, y, feature_cols


def compute_sample_weights(dates):
    days_ago = (dates.max() - dates).dt.days
    return DECAY_LAMBDA ** days_ago.values


def compute_ats_accuracy(y_residual, y_pred_residual):
    actual_cover = y_residual > 0
    predicted_cover = y_pred_residual > 0
    correct = actual_cover == predicted_cover
    push = y_residual == 0
    valid = ~push
    if valid.sum() == 0:
        return 0.0
    return float(correct[valid].mean())


# ── Optuna objectives ────────────────────────────────────────────────

def xgb_objective(trial, X_train, y_train, X_test, y_test, train_weights):
    """Optimiza XGBoost regression minimizando RMSE con ATS >= 52%."""
    params = {
        # Expanded depth range — enriched dataset (~220 features) benefits from deeper trees
        "max_depth": trial.suggest_int("max_depth", 4, 14),
        "eta": trial.suggest_float("eta", 0.005, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
        "gamma": trial.suggest_float("gamma", 0.0, 20.0),
        "lambda": trial.suggest_float("lambda", 0.1, 15.0, log=True),
        "alpha": trial.suggest_float("alpha", 0.01, 10.0, log=True),
        "objective": "reg:pseudohubererror",
        "huber_slope": trial.suggest_float("huber_slope", 3.0, 25.0),
        "tree_method": "hist",
        "seed": 42,
        "eval_metric": ["rmse"],
    }
    nb = trial.suggest_int("num_boost_round", 300, 2000)

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=train_weights)
    dtest = xgb.DMatrix(X_test, label=y_test)

    booster = xgb.train(
        params, dtrain,
        num_boost_round=nb,
        evals=[(dtest, "test")],
        early_stopping_rounds=60,
        verbose_eval=False,
    )

    p = booster.predict(dtest)
    rmse = float(np.sqrt(mean_squared_error(y_test, p)))
    ats = compute_ats_accuracy(y_test, p)

    if ats < 0.52:
        raise optuna.TrialPruned()

    # Objective: minimize RMSE, reward ATS above 53%
    return rmse - 2.0 * max(0, ats - 0.53)


def cat_objective(trial, X_train, y_train, X_test, y_test, train_weights):
    """Optimiza CatBoost regression minimizando RMSE con ATS >= 52%."""
    params = {
        "loss_function": f"Huber:delta={trial.suggest_float('huber_delta', 5.0, 20.0):.1f}",
        "eval_metric": "RMSE",
        "random_seed": 42,
        "task_type": "CPU",
        "depth": trial.suggest_int("depth", 4, 12),
        "iterations": trial.suggest_int("iterations", 500, 3000),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.01, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.001, 1.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "rsm": trial.suggest_float("rsm", 0.5, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
    }

    model = CatBoostRegressor(**params)
    model.fit(
        X_train, y_train,
        sample_weight=train_weights,
        eval_set=(X_test, y_test),
        early_stopping_rounds=60,
        verbose=0,
    )
    p = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, p)))
    ats = compute_ats_accuracy(y_test, p)

    if ats < 0.52:
        raise optuna.TrialPruned()

    return rmse - 2.0 * max(0, ats - 0.53)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Optuna tuning for margin regression")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    parser.add_argument("--model", choices=["xgb", "cat", "ensemble", "all"],
                        default="all")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  OPTUNA MARGIN TUNING — {args.n_trials} trials per model")
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

    train_weights = compute_sample_weights(train_df[DATE_COL])
    spreads_test = test_df["MARKET_SPREAD"].values if "MARKET_SPREAD" in test_df.columns else np.zeros(len(y_test))

    print(f"  Train: {len(y_train)} | Test: {len(y_test)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Residual stats — train: μ={y_train.mean():.2f} σ={y_train.std():.1f}")
    print(f"  Residual stats — test:  μ={y_test.mean():.2f} σ={y_test.std():.1f}")

    best_xgb_params = None
    best_cat_params = None
    p_xgb = None
    p_cat = None

    # ── XGBoost tuning ──
    if args.model in ("xgb", "all"):
        print(f"\n  Tuning XGBoost Margin ({args.n_trials} trials)...")
        study_xgb = optuna.create_study(
            direction="minimize",
            study_name="xgb_margin",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study_xgb.optimize(
            lambda trial: xgb_objective(trial, X_train, y_train, X_test, y_test, train_weights),
            n_trials=args.n_trials,
            show_progress_bar=True,
        )

        best_xgb_params = study_xgb.best_params
        print(f"\n  Best XGBoost Margin:")
        print(f"    Score: {study_xgb.best_value:.4f}")
        print(f"    Params: {json.dumps(best_xgb_params, indent=6)}")

        # Evaluate best
        bp = best_xgb_params.copy()
        nb = bp.pop("num_boost_round")
        hs = bp.pop("huber_slope")
        bp.update({
            "objective": "reg:pseudohubererror", "huber_slope": hs,
            "tree_method": "hist", "seed": 42, "eval_metric": ["rmse"],
        })
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=train_weights)
        dtest = xgb.DMatrix(X_test, label=y_test)
        booster = xgb.train(bp, dtrain, num_boost_round=nb,
                            evals=[(dtest, "test")], early_stopping_rounds=60, verbose_eval=False)
        p_xgb = booster.predict(dtest)
        xgb_rmse = float(np.sqrt(mean_squared_error(y_test, p_xgb)))
        xgb_mae = mean_absolute_error(y_test, p_xgb)
        xgb_ats = compute_ats_accuracy(y_test, p_xgb)
        print(f"    RMSE: {xgb_rmse:.2f}, MAE: {xgb_mae:.2f}, ATS: {xgb_ats:.1%}")

    # ── CatBoost tuning ──
    if args.model in ("cat", "all"):
        print(f"\n  Tuning CatBoost Margin ({args.n_trials} trials)...")
        study_cat = optuna.create_study(
            direction="minimize",
            study_name="cat_margin",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study_cat.optimize(
            lambda trial: cat_objective(trial, X_train, y_train, X_test, y_test, train_weights),
            n_trials=args.n_trials,
            show_progress_bar=True,
        )

        best_cat_params = study_cat.best_params
        print(f"\n  Best CatBoost Margin:")
        print(f"    Score: {study_cat.best_value:.4f}")
        print(f"    Params: {json.dumps(best_cat_params, indent=6)}")

        # Evaluate best
        bp = best_cat_params.copy()
        hd = bp.pop("huber_delta")
        bp.update({
            "loss_function": f"Huber:delta={hd:.1f}",
            "eval_metric": "RMSE", "random_seed": 42, "task_type": "CPU",
        })
        cat_model = CatBoostRegressor(**bp)
        cat_model.fit(X_train, y_train, sample_weight=train_weights,
                      eval_set=(X_test, y_test), early_stopping_rounds=60, verbose=0)
        p_cat = cat_model.predict(X_test)
        cat_rmse = float(np.sqrt(mean_squared_error(y_test, p_cat)))
        cat_mae = mean_absolute_error(y_test, p_cat)
        cat_ats = compute_ats_accuracy(y_test, p_cat)
        print(f"    RMSE: {cat_rmse:.2f}, MAE: {cat_mae:.2f}, ATS: {cat_ats:.1%}")

    # ── Ensemble weight search ──
    if args.model in ("ensemble", "all") and p_xgb is not None and p_cat is not None:
        print(f"\n  Optimizing ensemble weights...")
        best_w = None
        best_score = 999.0
        for w_xgb_val in np.arange(0.3, 0.81, 0.05):
            w_cat_val = 1.0 - w_xgb_val
            p_ens = w_xgb_val * p_xgb + w_cat_val * p_cat
            rmse = float(np.sqrt(mean_squared_error(y_test, p_ens)))
            ats = compute_ats_accuracy(y_test, p_ens)
            score = rmse - 2.0 * max(0, ats - 0.53)
            if score < best_score:
                best_score = score
                best_w = {"xgb": round(float(w_xgb_val), 2), "cat": round(float(w_cat_val), 2)}

        p_best = best_w["xgb"] * p_xgb + best_w["cat"] * p_cat
        ens_rmse = float(np.sqrt(mean_squared_error(y_test, p_best)))
        ens_mae = mean_absolute_error(y_test, p_best)
        ens_ats = compute_ats_accuracy(y_test, p_best)
        print(f"  Best weights: {best_w}")
        print(f"  Ensemble: RMSE={ens_rmse:.2f}, MAE={ens_mae:.2f}, ATS={ens_ats:.1%}")

    # ── Save results ──
    results = {}
    if best_xgb_params:
        results["xgb_params"] = best_xgb_params
    if best_cat_params:
        results["cat_params"] = best_cat_params
    if args.model in ("ensemble", "all") and best_w:
        results["weights"] = best_w

    NBA_MARGIN_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = NBA_MARGIN_MODELS_DIR / "optuna_best_params.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Best params saved → {out_path}")

    print(f"\n{'='*65}")
    print(f"  DONE — Run 'train_margin_models.py' to retrain with optimized params")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
