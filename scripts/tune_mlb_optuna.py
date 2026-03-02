"""Optuna hyperparameter tuning para modelos MLB (XGBoost + CatBoost).

Optimiza hiperparametros de cada modelo INDEPENDIENTEMENTE por tipo:
  - ML (moneyline): objetivo ECE (calibracion), constraint accuracy >= 53%
  - F5 (primeras 5 entradas): mismo objetivo ECE, constraint accuracy >= 53%
  - Totals (regresion): objetivo RMSE

POR QUE ECE PARA ML/F5:
  MLB es mas ruidoso que NBA (~54% accuracy es el techo real).
  La calidad de las probabilidades (ECE) importa mas que acertar el lado,
  porque Kelly depende directamente de P(home win). Un modelo mal calibrado
  sobreapuesta y pierde a largo plazo, aunque acierte el 54%.

POR QUE RMSE PARA TOTALS:
  Es una tarea de regresion. RMSE mide el error en carreras (unidades reales),
  lo que nos dice directamente que tan cerca esta la prediccion del total real.

DIFERENCIAS vs tune_optuna.py (NBA):
  - 150 trials por modelo (MLB necesita mas exploracion, espacio mas ruidoso)
  - Constraints de accuracy mas bajos (53% en lugar de 63%)
  - Soporte para 3 targets via --target (ml/f5/totals/all)
  - Objective score penaliza accuracy < 55% en ML/F5
  - Totals usa RMSE como objetivo (regresion)

Uso:
    PYTHONPATH=. python scripts/tune_mlb_optuna.py
    PYTHONPATH=. python scripts/tune_mlb_optuna.py --n-trials 200
    PYTHONPATH=. python scripts/tune_mlb_optuna.py --model xgb
    PYTHONPATH=. python scripts/tune_mlb_optuna.py --model cat
    PYTHONPATH=. python scripts/tune_mlb_optuna.py --model all
    PYTHONPATH=. python scripts/tune_mlb_optuna.py --target f5
    PYTHONPATH=. python scripts/tune_mlb_optuna.py --target totals --model xgb
    PYTHONPATH=. python scripts/tune_mlb_optuna.py --target all --n-trials 200 --save
"""

import argparse
import json
import sqlite3
import warnings

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

from src.core.calibration.xgb_calibrator import XGBCalibrator
from src.sports.mlb.config_paths import (
    MLB_DATASET_DB,
    MLB_ML_MODELS_DIR,
    MLB_F5_MODELS_DIR,
    MLB_TOTALS_MODELS_DIR,
)
from src.config import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

DEFAULT_DATASET = "mlb_dataset_2018-25"
TARGET_ML = "Home-Team-Win"
TARGET_F5 = "F5-Home-Win"
TARGET_TOT = "Total_Runs"
SEASON_COL = "SEASON"
N_TRIALS_DEFAULT = 150

# Columnas a excluir de los features (igual que train_mlb_models.py)
DROP_COLUMNS_MLB = [
    "GAME_PK", "GAME_DATE", "SEASON", "HOME_AWAY", "HOME_AWAY.1",
    "TEAM_NAME", "TEAM_NAME.1", "TEAM_ID", "TEAM_ID.1",
    "OPPONENT_NAME", "OPPONENT_NAME.1", "OPPONENT_ID", "OPPONENT_ID.1",
    "WIN", "WIN.1", "W_L", "W_L.1",
    "SP_NAME", "SP_NAME.1", "SP_ID", "SP_ID.1",
    "INNING_RUNS", "INNING_RUNS.1",
    "Home-Team-Win", "F5-Home-Win", "Total_Runs",
    "RUNS", "RUNS.1", "R", "R.1",
    "HITS", "HITS.1", "H", "H.1", "ERRORS", "ERRORS.1",
    "F5_RUNS", "F5_RUNS.1",
    "DIVISION_HOME", "DIVISION_AWAY", "LEAGUE_HOME", "LEAGUE_AWAY",
    "Home-Team-Index", "Away-Team-Index",
]


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def load_data(dataset_name=DEFAULT_DATASET, target=TARGET_ML):
    with sqlite3.connect(MLB_DATASET_DB) as con:
        df = pd.read_sql_query(f'SELECT * FROM "{dataset_name}"', con)

    if SEASON_COL in df.columns:
        df[SEASON_COL] = df[SEASON_COL].astype(str)

    # Remove NaN targets
    df = df.dropna(subset=[target]).copy()
    return df


def df_to_xy_cls(df, target):
    y = df[target].astype(int).to_numpy()
    X_df = df.drop(columns=DROP_COLUMNS_MLB, errors="ignore")
    feature_cols = list(X_df.columns)
    X = X_df.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0).to_numpy(dtype=float)
    return X, y, feature_cols


def df_to_xy_reg(df, target):
    y = df[target].astype(float).to_numpy()
    X_df = df.drop(columns=DROP_COLUMNS_MLB, errors="ignore")
    feature_cols = list(X_df.columns)
    X = X_df.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0).to_numpy(dtype=float)
    return X, y, feature_cols


def temporal_split(df, train_seasons, val_seasons, test_seasons):
    """Divide el dataset por temporada. Retorna (train_df, val_df, test_df)."""
    train_df = df[df[SEASON_COL].isin(train_seasons)].copy()
    val_df = df[df[SEASON_COL].isin(val_seasons)].copy()
    test_df = df[df[SEASON_COL].isin(test_seasons)].copy()

    if len(test_df) == 0 or len(train_df) == 0:
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        val_df = df.iloc[int(split_idx * 0.85):split_idx].copy()

    return train_df, val_df, test_df


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


# ---------------------------------------------------------------------------
# Optuna objectives — Classification (ML + F5)
# ---------------------------------------------------------------------------

def xgb_cls_objective(trial, X_train, y_train, X_test, y_test, X_cal, y_cal):
    """Optimiza XGBoost clasificacion minimizando ECE con accuracy >= 53%."""
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "eta": trial.suggest_float("eta", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 30),
        "gamma": trial.suggest_float("gamma", 1.0, 15.0),
        "lambda": trial.suggest_float("lambda", 0.1, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 0.01, 5.0, log=True),
        "objective": "multi:softprob",
        "num_class": 2,
        "tree_method": "hist",
        "seed": 42,
        "eval_metric": ["mlogloss"],
    }
    nb = trial.suggest_int("num_boost_round", 200, 1000)

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=compute_sample_weights(y_train))
    dtest = xgb.DMatrix(X_test, label=y_test)

    booster = xgb.train(
        params, dtrain,
        num_boost_round=nb,
        evals=[(dtest, "test")],
        early_stopping_rounds=40,
        verbose_eval=False,
    )

    calibrator = XGBCalibrator(booster)
    calibrator.fit(X_cal, y_cal)
    p = calibrator.predict_proba(X_test)

    acc = (np.argmax(p, axis=1) == y_test).mean()
    ece = compute_ece(y_test, p[:, 1])

    # Prune trials con accuracy muy baja (MLB techo ~54%, no exigimos mucho)
    if acc < 0.53:
        raise optuna.TrialPruned()

    # Objetivo: minimizar ECE (primario), penalizar accuracy < 55%
    # La constante 0.55 es la "aspiracion" de accuracy — mas baja que NBA (0.65)
    return ece - 0.1 * (acc - 0.55)


def cat_cls_objective(trial, X_train, y_train, X_test, y_test):
    """Optimiza CatBoost clasificacion minimizando ECE con accuracy >= 53%."""
    params = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "auto_class_weights": "Balanced",
        "random_seed": 42,
        "task_type": "CPU",
        "depth": trial.suggest_int("depth", 4, 9),
        "iterations": trial.suggest_int("iterations", 400, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 15.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.001, 2.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "rsm": trial.suggest_float("rsm", 0.5, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "verbose": 0,
    }

    model = CatBoostClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=40,
    )
    p = model.predict_proba(X_test)
    acc = (np.argmax(p, axis=1) == y_test).mean()
    ece = compute_ece(y_test, p[:, 1])

    if acc < 0.53:
        raise optuna.TrialPruned()

    return ece - 0.1 * (acc - 0.55)


# ---------------------------------------------------------------------------
# Optuna objectives — Regression (Totals)
# ---------------------------------------------------------------------------

def xgb_reg_objective(trial, X_train, y_train, X_test, y_test):
    """Optimiza XGBoost regresion minimizando RMSE."""
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "eta": trial.suggest_float("eta", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 30),
        "gamma": trial.suggest_float("gamma", 0.5, 10.0),
        "lambda": trial.suggest_float("lambda", 0.1, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 0.01, 5.0, log=True),
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "seed": 42,
        "eval_metric": ["rmse"],
    }
    nb = trial.suggest_int("num_boost_round", 200, 1000)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    booster = xgb.train(
        params, dtrain,
        num_boost_round=nb,
        evals=[(dtest, "test")],
        early_stopping_rounds=40,
        verbose_eval=False,
    )
    preds = booster.predict(dtest)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    return rmse


def cat_reg_objective(trial, X_train, y_train, X_val, y_val, X_test, y_test):
    """Optimiza CatBoost regresion minimizando RMSE."""
    params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": 42,
        "task_type": "CPU",
        "depth": trial.suggest_int("depth", 4, 9),
        "iterations": trial.suggest_int("iterations", 400, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 15.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.001, 2.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "rsm": trial.suggest_float("rsm", 0.5, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "verbose": 0,
    }

    model = CatBoostRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=40,
    )
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse


# ---------------------------------------------------------------------------
# Tuning runners
# ---------------------------------------------------------------------------

def run_cls_tuning(model_choice, n_trials, dataset_name, target, label):
    """Ejecuta tuning de clasificacion (ML o F5)."""
    print(f"\n  Loading {label} data from {dataset_name}...")
    df = load_data(dataset_name, target)

    train_df, val_df, test_df = temporal_split(
        df,
        train_seasons=["2018", "2019", "2020", "2021", "2022"],
        val_seasons=["2023"],
        test_seasons=["2024", "2025"],
    )

    X_train, y_train, feature_cols = df_to_xy_cls(train_df, target)
    X_val, y_val, _ = df_to_xy_cls(val_df, target)
    X_test, y_test, _ = df_to_xy_cls(test_df, target)
    X_cal, y_cal = X_val, y_val

    print(f"  Train: {len(y_train)} | Cal: {len(y_cal)} | Test: {len(y_test)}")
    print(f"  Features: {len(feature_cols)}")

    best_xgb_params = None
    best_cat_params = None
    p_xgb = None
    p_cat = None

    # --- XGBoost ---
    if model_choice in ("xgb", "both"):
        print(f"\n  Tuning XGBoost {label} ({n_trials} trials)...")
        study_xgb = optuna.create_study(
            direction="minimize",
            study_name=f"xgb_{label.lower()}_ece",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study_xgb.optimize(
            lambda trial: xgb_cls_objective(trial, X_train, y_train, X_test, y_test, X_cal, y_cal),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        best_xgb_params = study_xgb.best_params
        print(f"\n  Best XGBoost {label}:")
        print(f"    Score: {study_xgb.best_value:.4f}")
        print(f"    Params: {json.dumps(best_xgb_params, indent=6)}")

        # Evaluate best
        bp = best_xgb_params.copy()
        nb = bp.pop("num_boost_round")
        bp.update({
            "objective": "multi:softprob", "num_class": 2,
            "tree_method": "hist", "seed": 42, "eval_metric": ["mlogloss"],
        })
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=compute_sample_weights(y_train))
        dtest = xgb.DMatrix(X_test, label=y_test)
        booster = xgb.train(bp, dtrain, num_boost_round=nb,
                            evals=[(dtest, "test")], early_stopping_rounds=40, verbose_eval=False)
        cal = XGBCalibrator(booster)
        cal.fit(X_cal, y_cal)
        p_xgb = cal.predict_proba(X_test)
        xgb_acc = accuracy_score(y_test, np.argmax(p_xgb, axis=1))
        xgb_ece = compute_ece(y_test, p_xgb[:, 1])
        print(f"    Acc: {xgb_acc:.1%}, ECE: {xgb_ece:.4f}")

    # --- CatBoost ---
    if model_choice in ("cat", "both"):
        print(f"\n  Tuning CatBoost {label} ({n_trials} trials)...")
        study_cat = optuna.create_study(
            direction="minimize",
            study_name=f"cat_{label.lower()}_ece",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study_cat.optimize(
            lambda trial: cat_cls_objective(trial, X_train, y_train, X_test, y_test),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        best_cat_params = study_cat.best_params
        print(f"\n  Best CatBoost {label}:")
        print(f"    Score: {study_cat.best_value:.4f}")
        print(f"    Params: {json.dumps(best_cat_params, indent=6)}")

        # Evaluate best
        bp = best_cat_params.copy()
        bp.update({
            "loss_function": "Logloss", "eval_metric": "Logloss",
            "auto_class_weights": "Balanced", "random_seed": 42, "task_type": "CPU",
            "verbose": 0,
        })
        cat_model = CatBoostClassifier(**bp)
        cat_model.fit(X_train, y_train, eval_set=(X_test, y_test),
                      early_stopping_rounds=40)
        p_cat = cat_model.predict_proba(X_test)
        cat_acc = accuracy_score(y_test, np.argmax(p_cat, axis=1))
        cat_ece = compute_ece(y_test, p_cat[:, 1])
        print(f"    Acc: {cat_acc:.1%}, ECE: {cat_ece:.4f}")

    # --- Ensemble weight search (solo si tenemos ambos) ---
    if p_xgb is not None and p_cat is not None:
        print(f"\n  Optimizing ensemble weights ({label})...")
        best_w = None
        best_score = 1.0
        for w_xgb in np.arange(0, 1.01, 0.05):
            w_cat = 1.0 - w_xgb
            p_ens = w_xgb * p_xgb + w_cat * p_cat
            acc = accuracy_score(y_test, np.argmax(p_ens, axis=1))
            ece = compute_ece(y_test, p_ens[:, 1])
            score = ece - 0.1 * (acc - 0.55)
            if score < best_score:
                best_score = score
                best_w = {"xgb": round(w_xgb, 2), "cat": round(w_cat, 2)}

        p_best = best_w["xgb"] * p_xgb + best_w["cat"] * p_cat
        ens_acc = accuracy_score(y_test, np.argmax(p_best, axis=1))
        ens_ece = compute_ece(y_test, p_best[:, 1])
        print(f"  Best weights ({label}): {best_w}")
        print(f"  Ensemble ({label}): Acc={ens_acc:.1%}, ECE={ens_ece:.4f}")
    else:
        best_w = None

    return best_xgb_params, best_cat_params, best_w


def run_reg_tuning(model_choice, n_trials, dataset_name, target, label):
    """Ejecuta tuning de regresion (Totals)."""
    print(f"\n  Loading {label} data from {dataset_name}...")
    df = load_data(dataset_name, target)

    train_df, val_df, test_df = temporal_split(
        df,
        train_seasons=["2018", "2019", "2020", "2021", "2022"],
        val_seasons=["2023"],
        test_seasons=["2024", "2025"],
    )

    X_train, y_train, feature_cols = df_to_xy_reg(train_df, target)
    X_val, y_val, _ = df_to_xy_reg(val_df, target)
    X_test, y_test, _ = df_to_xy_reg(test_df, target)

    print(f"  Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
    print(f"  Features: {len(feature_cols)}")
    naive_rmse = np.std(y_test)
    print(f"  Naive RMSE (std): {naive_rmse:.4f}")

    best_xgb_params = None
    best_cat_params = None
    p_xgb_reg = None
    p_cat_reg = None

    # --- XGBoost ---
    if model_choice in ("xgb", "both"):
        print(f"\n  Tuning XGBoost {label} ({n_trials} trials)...")
        study_xgb = optuna.create_study(
            direction="minimize",
            study_name=f"xgb_{label.lower()}_rmse",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study_xgb.optimize(
            lambda trial: xgb_reg_objective(trial, X_train, y_train, X_test, y_test),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        best_xgb_params = study_xgb.best_params
        print(f"\n  Best XGBoost {label}:")
        print(f"    RMSE: {study_xgb.best_value:.4f}")
        print(f"    Params: {json.dumps(best_xgb_params, indent=6)}")

        # Evaluate best
        bp = best_xgb_params.copy()
        nb = bp.pop("num_boost_round")
        bp.update({
            "objective": "reg:squarederror",
            "tree_method": "hist", "seed": 42, "eval_metric": ["rmse"],
        })
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        booster = xgb.train(bp, dtrain, num_boost_round=nb,
                            evals=[(dtest, "test")], early_stopping_rounds=40, verbose_eval=False)
        p_xgb_reg = booster.predict(dtest)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, p_xgb_reg))
        xgb_mae = np.mean(np.abs(y_test - p_xgb_reg))
        print(f"    RMSE: {xgb_rmse:.4f}, MAE: {xgb_mae:.4f}")

    # --- CatBoost ---
    if model_choice in ("cat", "both"):
        print(f"\n  Tuning CatBoost {label} ({n_trials} trials)...")
        study_cat = optuna.create_study(
            direction="minimize",
            study_name=f"cat_{label.lower()}_rmse",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study_cat.optimize(
            lambda trial: cat_reg_objective(trial, X_train, y_train, X_val, y_val, X_test, y_test),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        best_cat_params = study_cat.best_params
        print(f"\n  Best CatBoost {label}:")
        print(f"    RMSE: {study_cat.best_value:.4f}")
        print(f"    Params: {json.dumps(best_cat_params, indent=6)}")

        # Evaluate best
        bp = best_cat_params.copy()
        bp.update({
            "loss_function": "RMSE", "eval_metric": "RMSE",
            "random_seed": 42, "task_type": "CPU", "verbose": 0,
        })
        cat_model = CatBoostRegressor(**bp)
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val),
                      early_stopping_rounds=40)
        p_cat_reg = cat_model.predict(X_test)
        cat_rmse = np.sqrt(mean_squared_error(y_test, p_cat_reg))
        cat_mae = np.mean(np.abs(y_test - p_cat_reg))
        print(f"    RMSE: {cat_rmse:.4f}, MAE: {cat_mae:.4f}")

    # --- Ensemble weight search ---
    if p_xgb_reg is not None and p_cat_reg is not None:
        print(f"\n  Optimizing ensemble weights ({label}, RMSE)...")
        best_w = None
        best_rmse = 1e9
        for w_xgb in np.arange(0, 1.01, 0.05):
            w_cat = 1.0 - w_xgb
            p_ens = w_xgb * p_xgb_reg + w_cat * p_cat_reg
            rmse = np.sqrt(mean_squared_error(y_test, p_ens))
            if rmse < best_rmse:
                best_rmse = rmse
                best_w = {"xgb": round(w_xgb, 2), "cat": round(w_cat, 2)}

        print(f"  Best weights ({label}): {best_w}")
        print(f"  Ensemble RMSE ({label}): {best_rmse:.4f}")
    else:
        best_w = None

    return best_xgb_params, best_cat_params, best_w


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Optuna MLB hyperparameter tuning")
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help=f"Table name in MLB_DATASET_DB (default: {DEFAULT_DATASET})")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT,
                        help=f"Number of Optuna trials per model (default: {N_TRIALS_DEFAULT})")
    parser.add_argument("--model", choices=["xgb", "cat", "all"], default="all",
                        help="Which model to tune (default: all)")
    parser.add_argument("--target", choices=["ml", "f5", "totals", "all"], default="all",
                        help="Which target to tune (default: all)")
    parser.add_argument("--save", action="store_true",
                        help="Save best params to models/mlb/<type>/optuna_best_params.json")
    args = parser.parse_args()

    # Map 'all' to 'both' internally for model choice
    model_choice = "both" if args.model == "all" else args.model

    print(f"\n{'='*65}")
    print(f"  OPTUNA MLB TUNING — Target: {args.target.upper()} | Model: {args.model.upper()}")
    print(f"  Trials: {args.n_trials} per model")
    print(f"{'='*65}")

    targets_to_run = []
    if args.target in ("ml", "all"):
        targets_to_run.append(("ml", TARGET_ML, "ML", MLB_ML_MODELS_DIR))
    if args.target in ("f5", "all"):
        targets_to_run.append(("f5", TARGET_F5, "F5", MLB_F5_MODELS_DIR))
    if args.target in ("totals", "all"):
        targets_to_run.append(("totals", TARGET_TOT, "Totals", MLB_TOTALS_MODELS_DIR))

    for target_key, target_col, label, out_dir in targets_to_run:
        if target_key in ("ml", "f5"):
            best_xgb_params, best_cat_params, best_w = run_cls_tuning(
                model_choice=model_choice,
                n_trials=args.n_trials,
                dataset_name=args.dataset,
                target=target_col,
                label=label,
            )
        else:
            best_xgb_params, best_cat_params, best_w = run_reg_tuning(
                model_choice=model_choice,
                n_trials=args.n_trials,
                dataset_name=args.dataset,
                target=target_col,
                label=label,
            )

        results = {}
        if best_xgb_params:
            results["xgb_params"] = best_xgb_params
        if best_cat_params:
            results["cat_params"] = best_cat_params
        if best_w:
            results["weights"] = best_w

        # --- Save results ---
        if args.save and results:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "optuna_best_params.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n  Best params saved -> {out_path}")
            print(f"  To train with these params, update train_mlb_models.py and run it.")

    if not args.save:
        print(f"\n  (Use --save to persist best params to JSON)")

    print(f"\n{'='*65}")
    print(f"  DONE")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
