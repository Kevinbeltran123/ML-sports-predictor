"""Train XGBoost + CatBoost WNBA moneyline models from wnba_dataset.sqlite.

Usage:
    PYTHONPATH=. python scripts/train_models_wnba.py
    PYTHONPATH=. python scripts/train_models_wnba.py --dataset wnba_dataset_2024-25
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
from src.config import get_logger
from src.sports.wnba.config_paths import WNBA_DATASET_DB, WNBA_ML_MODELS_DIR

logger = get_logger(__name__)

DEFAULT_DATASET = "wnba_dataset"
TARGET = "Home-Team-Win"
DATE_COL = "Date"

# WNBA models use shallower trees + more regularization vs NBA (less data, shorter season)
XGB_PARAMS = {
    "max_depth": 7, "eta": 0.10, "subsample": 0.85,
    "colsample_bytree": 0.80, "colsample_bylevel": 0.75,
    "colsample_bynode": 0.80, "min_child_weight": 8,
    "gamma": 4.0, "lambda": 2.0, "alpha": 0.8,
    "objective": "multi:softprob", "num_class": 2,
    "tree_method": "hist", "seed": 42,
    "eval_metric": ["mlogloss"],
}
XGB_NB = 500

CATBOOST_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "auto_class_weights": "Balanced",
    "random_seed": 42,
    "task_type": "CPU",
    "depth": 7,
    "iterations": 1000,
    "learning_rate": 0.01,
    "l2_leaf_reg": 3.0,
    "random_strength": 0.5,
    "bagging_temperature": 1.0,
    "rsm": 0.75,
    "border_count": 128,
}

W_XGB = 0.60
W_CAT = 0.40

# BRef-dependent and NBA-specific columns absent from the WNBA feature set
DROP_COLUMNS_WNBA = [
    # Metadata / targets
    "index", "Score", "Home-Team-Win", "Margin", "TEAM_NAME", "Date",
    "index.1", "TEAM_NAME.1", "Date.1", "OU-Cover", "OU",
    # Redundant stats
    "Net_Rtg", "Net_Rtg.1", "eFG_PCT", "eFG_PCT.1", "ELO_PROB",
    "FG3M", "FG3M.1", "W_RANK", "W_RANK.1", "L_RANK", "L_RANK.1",
    "GP", "GP.1",
    # Ablation drops (shared with NBA)
    "AVAIL_AWAY", "AVAIL_HOME", "Avg_Pace",
    "BLKA", "BLKA.1", "BLK_RANK", "BLK_RANK.1",
    "DIFF_FG_PCT", "FG3A", "FG3A.1", "FG3A_RANK", "FG3A_RANK.1",
    "FGM", "FGM.1", "FG_PCT.1", "FTA", "FTA.1", "FTA_RANK", "FTA_RANK.1",
    "FTM", "FT_PCT", "FT_PCT.1", "FT_Rate.1",
    "MARKET_SPREAD", "OREB", "OREB_RANK.1",
    "PLUS_MINUS", "PLUS_MINUS.1", "PLUS_MINUS_RANK", "PLUS_MINUS_RANK.1",
    "PTS", "PTS.1", "REB", "REB_RANK.1",
    "STL", "STL.1", "TOV", "TOV_PCT.1", "TOV_RANK.1",
    "TS_PCT", "TS_PCT.1", "TZ_CHANGE_HOME",
    "W_PCT", "W_PCT.1", "W_PCT_RANK", "W_PCT_RANK.1",
    "ORB_PCT.1", "SRS_AWAY", "SRS_DIFF",
    # BRef features (not available for WNBA)
    "SC_RA_RATE_HOME", "SC_RA_RATE_AWAY",
    "SC_RA_FG_PCT_HOME", "SC_RA_FG_PCT_AWAY",
    "SC_PAINT_RATE_HOME", "SC_PAINT_RATE_AWAY",
    "SC_MID_RATE_HOME", "SC_MID_RATE_AWAY",
    "SC_CORNER3_RATE_HOME", "SC_CORNER3_RATE_AWAY",
    "SC_AVG_DIST_HOME", "SC_AVG_DIST_AWAY",
    "ZONE_AVG_DIST_HOME", "ZONE_FG3A_RATE_HOME",
    "ZONE_PAINT_FG_PCT_HOME", "ZONE_CLOSE_MID_FG_PCT_HOME",
    "ZONE_MID_FG_PCT_HOME", "ZONE_LONG2_FG_PCT_HOME",
    "ZONE_CORNER3_PCT_HOME", "ZONE_DUNK_RATE_HOME",
    "ZONE_AVG_DIST_AWAY", "ZONE_FG3A_RATE_AWAY",
    "ZONE_PAINT_FG_PCT_AWAY", "ZONE_CLOSE_MID_FG_PCT_AWAY",
    "ZONE_MID_FG_PCT_AWAY", "ZONE_LONG2_FG_PCT_AWAY",
    "ZONE_CORNER3_PCT_AWAY", "ZONE_DUNK_RATE_AWAY",
    "ONOFF_NET_TOP5_HOME", "ONOFF_NET_TOP5_AWAY",
    "ONOFF_SPREAD_HOME", "ONOFF_SPREAD_AWAY",
    "ADV_BPM_TOP5_HOME", "ADV_BPM_TOP5_AWAY",
    "ADV_MAX_BPM_HOME", "ADV_MAX_BPM_AWAY",
    "ADV_USG_CONCENTRATION_HOME", "ADV_USG_CONCENTRATION_AWAY",
    "ADV_TS_TEAM_HOME", "ADV_TS_TEAM_AWAY",
    "LS_Q4_PCT_HOME", "LS_Q4_PCT_AWAY",
    "LS_2H_RATIO_HOME", "LS_2H_RATIO_AWAY",
    "LS_Q1_PCT_HOME", "LS_Q1_PCT_AWAY",
    "LS_SCORING_VAR_HOME", "LS_SCORING_VAR_AWAY",
    "ESPN_LINE_MOVE", "ESPN_TOTAL_MOVE",
    "ESPN_OPEN_ML_PROB", "ESPN_BOOK_DISAGREEMENT",
    "LINEUP_DIVERSITY_HOME", "LINEUP_DIVERSITY_AWAY",
    "LINEUP_STAR_FRAC_HOME", "LINEUP_STAR_FRAC_AWAY",
    "BENCH_PPG_GAP_HOME", "BENCH_PPG_GAP_AWAY",
    "BENCH_DEPTH_HOME", "BENCH_DEPTH_AWAY",
    "REF_CREW_TOTAL_TENDENCY", "REF_CREW_OVER_PCT", "REF_CREW_HOME_WIN_PCT",
    "STAR_MISSING_HOME", "STAR_MISSING_AWAY",
    "N_ROTATION_OUT_HOME", "N_ROTATION_OUT_AWAY",
    "MISSING_BPM_HOME", "MISSING_BPM_AWAY",
    "AVAIL_DIFF",
]


def compute_sample_weights(y, num_classes=2):
    counts = np.bincount(y, minlength=num_classes)
    total = len(y)
    class_weights = {
        cls: (total / (num_classes * count)) if count else 1.0
        for cls, count in enumerate(counts)
    }
    return np.array([class_weights[label] for label in y])


def load_and_prepare(dataset_name):
    with sqlite3.connect(WNBA_DATASET_DB) as con:
        df = pd.read_sql_query(f'SELECT * FROM "{dataset_name}"', con)
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    return df


def df_to_xy(df):
    y = df[TARGET].astype(int).to_numpy()
    X_df = df.drop(columns=DROP_COLUMNS_WNBA, errors="ignore")
    feature_cols = list(X_df.columns)
    X = X_df.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0).to_numpy(dtype=float)
    return X, y, feature_cols


def main():
    parser = argparse.ArgumentParser(description="Train WNBA XGBoost + CatBoost ML models")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  TRAIN XGBoost + CatBoost — WNBA Moneyline")
    print(f"{'='*65}")

    df = load_and_prepare(args.dataset)

    # Temporal split: train < 2025-05-01 (start of 2025 WNBA season), test >= that date
    test_dt = pd.to_datetime("2025-05-01")
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

    # 1. XGBoost
    print(f"\n  Training XGBoost (nb={XGB_NB})...")
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=compute_sample_weights(y_train))
    dtest = xgb.DMatrix(X_test, label=y_test)
    booster = xgb.train(
        XGB_PARAMS, dtrain,
        num_boost_round=XGB_NB,
        evals=[(dtest, "test")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    p_xgb = booster.predict(dtest)
    xgb_acc = accuracy_score(y_test, np.argmax(p_xgb, axis=1))
    xgb_ll = log_loss(y_test, p_xgb)
    print(f"  XGBoost — Acc: {xgb_acc:.1%}, LogLoss: {xgb_ll:.4f}")

    # 2. CatBoost
    print(f"\n  Training CatBoost (it={CATBOOST_PARAMS['iterations']})...")
    cat_model = CatBoostClassifier(**CATBOOST_PARAMS)
    cat_model.fit(X_train, y_train, eval_set=(X_test, y_test),
                  early_stopping_rounds=50, verbose=0)
    p_cat = cat_model.predict_proba(X_test)
    cat_acc = accuracy_score(y_test, np.argmax(p_cat, axis=1))
    cat_ll = log_loss(y_test, p_cat)
    print(f"  CatBoost — Acc: {cat_acc:.1%}, LogLoss: {cat_ll:.4f}")

    # 3. Ensemble
    p_ensemble = W_XGB * p_xgb + W_CAT * p_cat
    ens_acc = accuracy_score(y_test, np.argmax(p_ensemble, axis=1))
    ens_ll = log_loss(y_test, p_ensemble)
    print(f"\n  Ensemble (XGB {W_XGB:.0%} + Cat {W_CAT:.0%}) — Acc: {ens_acc:.1%}, LogLoss: {ens_ll:.4f}")

    # 4. Save models
    WNBA_ML_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  Saving models to {WNBA_ML_MODELS_DIR}...")

    xgb_acc_str = f"{xgb_acc*100:.1f}"
    p = XGB_PARAMS
    xgb_name = (
        f"XGBoost_{xgb_acc_str}%_WNBA_ML_"
        f"md{p['max_depth']}_eta{str(p['eta']).replace('.','p')}_"
        f"sub{str(p['subsample']).replace('.','p')}_"
        f"col{str(p['colsample_bytree']).replace('.','p')}_"
        f"nb{XGB_NB}"
    )
    xgb_path = WNBA_ML_MODELS_DIR / f"{xgb_name}.json"
    booster.save_model(str(xgb_path))
    print(f"    XGBoost: {xgb_path.name}")

    cal_split = int(len(X_train) * 0.8)
    X_cal, y_cal = X_train[cal_split:], y_train[cal_split:]
    calibrator = XGBCalibrator(booster)
    calibrator.fit(X_cal, y_cal)
    cal_path = xgb_path.with_name(f"{xgb_name}_calibration.pkl")
    joblib.dump(calibrator, cal_path)
    print(f"    Calibrator: {cal_path.name}")

    cat_acc_str = f"{cat_acc*100:.1f}"
    cp = CATBOOST_PARAMS
    cat_name = (
        f"CatBoost_{cat_acc_str}%_WNBA_ML_"
        f"d{cp['depth']}_lr{str(cp['learning_rate']).replace('.','p')}_"
        f"it{cp['iterations']}_"
        f"l2{str(cp['l2_leaf_reg']).replace('.','p')}"
    )
    cat_path = WNBA_ML_MODELS_DIR / f"{cat_name}.pkl"
    joblib.dump(cat_model, cat_path)
    print(f"    CatBoost: {cat_path.name}")

    # 5. Ensemble conformal
    print("\n  Fitting ensemble conformal...")
    dcal = xgb.DMatrix(X_cal)
    p_xgb_cal = booster.predict(dcal)
    p_cat_cal = cat_model.predict_proba(X_cal)
    p_ens_cal = W_XGB * p_xgb_cal + W_CAT * p_cat_cal
    conformal = ConformalClassifier(alpha=0.10)
    conformal.fit(p_ens_cal, y_cal)
    conf_path = WNBA_ML_MODELS_DIR / "ensemble_conformal.pkl"
    joblib.dump(conformal, conf_path)
    print(f"    Conformal: {conformal}")

    # 6. Variance info
    variance_info = {
        "mean_sigma": float(np.abs(p_xgb[:, 1] - p_cat[:, 1]).mean()),
        "sigma_percentiles": {
            str(p): float(np.percentile(np.abs(p_xgb[:, 1] - p_cat[:, 1]), p))
            for p in [25, 50, 75, 90, 95]
        },
    }
    var_path = WNBA_ML_MODELS_DIR / "ensemble_variance.json"
    with open(var_path, "w") as f:
        json.dump(variance_info, f, indent=2)
    print(f"    Variance: mean_sigma={variance_info['mean_sigma']:.4f}")

    # 7. Metadata
    meta = {
        "dataset": args.dataset,
        "train_size": len(y_train),
        "test_size": len(y_test),
        "n_features": len(feature_cols),
        "xgb_accuracy": float(xgb_acc),
        "cat_accuracy": float(cat_acc),
        "ensemble_accuracy": float(ens_acc),
        "weights": {"xgb": W_XGB, "cat": W_CAT},
        "conformal_summary": conformal.summary(),
    }
    meta_path = WNBA_ML_MODELS_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  DONE — Ensemble: {ens_acc:.1%} ({len(feature_cols)} features)")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
