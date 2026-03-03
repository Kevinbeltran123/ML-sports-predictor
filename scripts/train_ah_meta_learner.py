"""Entrena AH Meta-Learner: LogisticRegression stacking sobre señales CLF+REG.

Reemplaza el blend lineal P(cover) = w*REG + (1-w)*CLF con un modelo
que aprende pesos no-lineales de ~12 señales disponibles al momento del blend.

**Sin data leakage:** Para cada fold temporal, se re-entrenan los base models
(XGBoost ML + XGBoost Margin) en datos anteriores al fold, generando
predicciones Out-of-Fold (OOF) genuinamente out-of-sample.

Safeguard: solo guarda el modelo si supera el baseline (blend lineal) en
ATS accuracy o Brier score.

Uso:
    PYTHONPATH=. python scripts/train_ah_meta_learner.py
    PYTHONPATH=. python scripts/train_ah_meta_learner.py --dry-run
"""

import argparse
import json
import sqlite3
import time

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.config import (
    DATASET_DB, NBA_MARGIN_MODELS_DIR, NBA_ML_MODELS_DIR,
    DROP_COLUMNS_ML, DROP_COLUMNS_MARGIN, get_logger,
)
from src.core.betting.spread_math import (
    p_cover, p_cover_from_residual, game_sigma_from_interval, sigma_for_line,
    expected_margin,
)
from src.core.betting.ah_meta_learner import EXPECTED_FEATURES

logger = get_logger(__name__)

DATASET_TABLE = "dataset_margin_enriched"

# Walk-forward season boundaries (used for both OOF and meta-learner CV)
FOLD_BOUNDARIES = [
    "2019-07-01",
    "2020-07-01",
    "2021-07-01",
    "2022-07-01",
    "2023-07-01",
]

# Margin interaction columns (in enriched dataset, not in ML training)
MARGIN_ONLY_COLS = [
    "NET_RTG_DIFF", "EFG_NET_HOME", "BPM_GAP", "PACE_NET_FACTOR",
    "AVAIL_BPM_CROSS", "BENCH_DEPTH_NET", "LS_Q4_NET", "ONOFF_NET_GAP",
    "ESPN_MOVE_ABS", "SCORING_VAR_NET",
]

# XGBoost params from Optuna/production
ML_XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 2,
    "max_depth": 6,
    "eta": 0.078,
    "subsample": 0.813,
    "colsample_bytree": 0.861,
    "min_child_weight": 9,
    "gamma": 9.928,
    "lambda": 0.245,
    "alpha": 0.347,
    "eval_metric": "mlogloss",
    "nthread": 4,
    "verbosity": 0,
}
ML_XGB_ROUNDS = 717

MARGIN_XGB_PARAMS = {
    "objective": "reg:squarederror",
    "max_depth": 11,
    "eta": 0.00933,
    "subsample": 0.922,
    "eval_metric": "rmse",
    "nthread": 4,
    "verbosity": 0,
}
MARGIN_XGB_ROUNDS = 1763


def load_dataset():
    """Carga dataset enriquecido con Margin y MARKET_SPREAD."""
    with sqlite3.connect(DATASET_DB) as con:
        df = pd.read_sql(f'SELECT * FROM "{DATASET_TABLE}"', con)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Margin", "MARKET_SPREAD"])
    df["Margin"] = pd.to_numeric(df["Margin"], errors="coerce")
    df["MARKET_SPREAD"] = pd.to_numeric(df["MARKET_SPREAD"], errors="coerce")
    df = df.dropna(subset=["Margin", "MARKET_SPREAD"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def prepare_ml_features(df):
    """Prepara features para XGBoost ML classifier."""
    drop_cols = [c for c in DROP_COLUMNS_ML + MARGIN_ONLY_COLS if c in df.columns]
    frame = df.drop(columns=drop_cols, errors="ignore")
    if "TS_PCT" in df.columns and "TS_PCT.1" in df.columns:
        frame["Diff_TS_PCT"] = df["TS_PCT"] - df["TS_PCT.1"]
    frame = frame.select_dtypes(include=[np.number])
    return frame.fillna(frame.median()).fillna(0).values.astype(float)


def prepare_margin_features(df):
    """Prepara features para XGBoost Margin regressor."""
    _drop = list(DROP_COLUMNS_MARGIN) + ["Residual"]
    drop_cols = [c for c in _drop if c in df.columns]
    frame = df.drop(columns=drop_cols, errors="ignore")
    frame = frame.select_dtypes(include=[np.number])
    return frame.replace([np.inf, -np.inf], np.nan).fillna(frame.median()).fillna(0).values.astype(float)


def train_oof_base_models(df_train, df_test):
    """Entrena XGBoost ML + Margin en df_train, predice en df_test.

    Returns:
        p_home_test: P(home win) array for test games
        sigma_arr_test: disagreement sigma (0.05 default, single model)
        reg_pred_test: margin/residual predictions for test games
    """
    # --- ML XGBoost ---
    ml_train_X = prepare_ml_features(df_train)
    ml_test_X = prepare_ml_features(df_test)
    y_train_ml = pd.to_numeric(df_train["Home-Team-Win"], errors="coerce").fillna(0).values.astype(int)

    dtrain_ml = xgb.DMatrix(ml_train_X, label=y_train_ml)
    dtest_ml = xgb.DMatrix(ml_test_X)
    bst_ml = xgb.train(ML_XGB_PARAMS, dtrain_ml, num_boost_round=ML_XGB_ROUNDS)
    p_xgb = bst_ml.predict(dtest_ml)
    if p_xgb.ndim == 2:
        p_home_test = p_xgb[:, 1]
    else:
        p_home_test = p_xgb
    p_home_test = np.clip(p_home_test, 0.01, 0.99)
    sigma_arr_test = np.full(len(df_test), 0.05)  # single model = no disagreement

    # --- Margin XGBoost (residual target) ---
    mg_train_X = prepare_margin_features(df_train)
    mg_test_X = prepare_margin_features(df_test)
    y_train_res = (
        pd.to_numeric(df_train["Margin"], errors="coerce").fillna(0).values
        + pd.to_numeric(df_train["MARKET_SPREAD"], errors="coerce").fillna(0).values
    )

    dtrain_mg = xgb.DMatrix(mg_train_X, label=y_train_res)
    dtest_mg = xgb.DMatrix(mg_test_X)
    bst_mg = xgb.train(MARGIN_XGB_PARAMS, dtrain_mg, num_boost_round=MARGIN_XGB_ROUNDS)
    reg_pred_test = bst_mg.predict(dtest_mg)

    return p_home_test, sigma_arr_test, reg_pred_test


def compute_meta_features_from_predictions(df, p_home, sigma_arr, reg_pred):
    """Computa el feature vector del meta-learner a partir de predicciones OOF."""
    n = len(df)
    spreads = df["MARKET_SPREAD"].values.astype(float)
    ats_rate_home = df["ATS_RATE_HOME"].values.astype(float) if "ATS_RATE_HOME" in df.columns else np.full(n, 0.5)
    ats_streak_home = df["ATS_STREAK_HOME"].values.astype(float) if "ATS_STREAK_HOME" in df.columns else np.zeros(n)

    meta_X = np.zeros((n, len(EXPECTED_FEATURES)))

    for i in range(n):
        line = spreads[i]
        ah_sigma = sigma_for_line(line)  # bucket sigma (no interval width for OOF)

        clf_p = p_cover(float(p_home[i]), line, sigma=ah_sigma)

        raw_pred = float(reg_pred[i])
        # Residual model: reg_pred = Margin + Spread, > 0 means home covers
        reg_p = p_cover_from_residual(raw_pred, ah_sigma, line=line)
        reg_margin = raw_pred - line

        model_margin = expected_margin(float(p_home[i]))
        divergence = abs(model_margin - (-line))

        meta_X[i] = [
            clf_p,                              # clf_p_cover
            reg_p,                              # reg_p_cover
            abs(line),                          # abs_spread
            ah_sigma,                           # ah_sigma
            divergence,                         # divergence
            float(sigma_arr[i]),                # sigma_i
            0.0,                                # reg_conf_margin (no conformal for OOF)
            abs(clf_p - reg_p),                 # clf_reg_gap
            float(p_home[i]),                   # prob_home
            float(ats_rate_home[i]),            # ats_rate
            float(ats_streak_home[i]),          # ats_streak
            reg_margin,                         # reg_margin
        ]

    return meta_X


def compute_baseline_blend(meta_X, spreads):
    """Computa P(cover) con el blend lineal actual (baseline)."""
    n = len(meta_X)
    blend_buckets = [
        (2.0, 0.65), (5.0, 0.60), (8.0, 0.55),
        (12.0, 0.50), (float("inf"), 0.45),
    ]
    p_blend = np.zeros(n)
    for i in range(n):
        clf_p = meta_X[i, 0]
        reg_p = meta_X[i, 1]
        abs_sp = abs(spreads[i])
        w = 0.60
        for threshold, weight in blend_buckets:
            if abs_sp <= threshold:
                w = weight
                break
        p_blend[i] = w * reg_p + (1 - w) * clf_p
    return p_blend


def ats_accuracy(p_cover_arr, actual_cover):
    """Fracción de veces que P(cover) > 0.5 predice correctamente."""
    preds = (p_cover_arr > 0.5).astype(int)
    return (preds == actual_cover).mean()


def ats_ece(p_cover_arr, actual_cover, n_bins=10):
    """Expected Calibration Error para ATS."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_cover_arr >= lo) & (p_cover_arr < hi)
        if mask.sum() == 0:
            continue
        avg_pred = p_cover_arr[mask].mean()
        avg_actual = actual_cover[mask].mean()
        ece += mask.sum() / len(p_cover_arr) * abs(avg_pred - avg_actual)
    return ece


def main():
    parser = argparse.ArgumentParser(description="Train AH Meta-Learner")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate only, don't save")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  TRAIN AH META-LEARNER (OOF Stacking, No Leakage)")
    print("=" * 60)

    df = load_dataset()
    print(f"  Dataset: {len(df):,} games, {len(df.columns)} columns")

    residuals = df["Margin"].values + df["MARKET_SPREAD"].values
    actual_cover = np.where(residuals > 0, 1.0, np.where(residuals < 0, 0.0, 0.5))
    non_push = actual_cover != 0.5
    print(f"  Pushes: {(~non_push).sum()} ({(~non_push).mean() * 100:.1f}%)")

    dates = df["Date"].values
    spreads = df["MARKET_SPREAD"].values

    # ==============================
    # Generate OOF meta features
    # ==============================
    print("\n" + "-" * 60)
    print("  GENERATING OOF PREDICTIONS (re-training base models per fold)")
    print("-" * 60)

    # OOF arrays: fill in predictions for each game using models trained WITHOUT that game
    oof_meta_X = np.full((len(df), len(EXPECTED_FEATURES)), np.nan)
    oof_baseline = np.full(len(df), np.nan)

    t_total = time.time()

    for fold_idx, boundary in enumerate(FOLD_BOUNDARIES):
        boundary_dt = np.datetime64(boundary)
        train_mask_idx = np.where(dates < boundary_dt)[0]
        if fold_idx < len(FOLD_BOUNDARIES) - 1:
            next_boundary = np.datetime64(FOLD_BOUNDARIES[fold_idx + 1])
            test_mask_idx = np.where((dates >= boundary_dt) & (dates < next_boundary))[0]
        else:
            test_mask_idx = np.where(dates >= boundary_dt)[0]

        n_train, n_test = len(train_mask_idx), len(test_mask_idx)
        if n_train < 500 or n_test < 50:
            print(f"  Fold {fold_idx + 1}: skip (train={n_train}, test={n_test})")
            continue

        t0 = time.time()
        print(f"  Fold {fold_idx + 1}: train={n_train:,}, test={n_test:,}...", end="", flush=True)

        df_train = df.iloc[train_mask_idx].copy()
        df_test = df.iloc[test_mask_idx].copy()

        # Train fresh base models on train data, predict on test data
        p_home_test, sigma_test, reg_pred_test = train_oof_base_models(df_train, df_test)

        # Compute meta features from OOF predictions
        fold_meta_X = compute_meta_features_from_predictions(
            df_test, p_home_test, sigma_test, reg_pred_test
        )
        fold_baseline = compute_baseline_blend(fold_meta_X, df_test["MARKET_SPREAD"].values)

        oof_meta_X[test_mask_idx] = fold_meta_X
        oof_baseline[test_mask_idx] = fold_baseline

        elapsed = time.time() - t0
        print(f" done ({elapsed:.0f}s)")

    total_elapsed = time.time() - t_total
    print(f"\n  Total OOF generation: {total_elapsed:.0f}s")

    # Filter to games with OOF predictions (all folds) and non-pushes
    has_oof = ~np.isnan(oof_meta_X[:, 0])
    valid = has_oof & non_push
    print(f"  Games with OOF predictions: {has_oof.sum():,}")
    print(f"  Valid (non-push): {valid.sum():,}")

    meta_X = oof_meta_X[valid]
    y = actual_cover[valid]
    p_baseline = oof_baseline[valid]
    dates_valid = dates[valid]
    spreads_valid = spreads[valid]

    # Feature stats
    print("\n  OOF Feature statistics:")
    for j, name in enumerate(EXPECTED_FEATURES):
        col = meta_X[:, j]
        print(f"    {name:20s}: μ={col.mean():.4f}  σ={col.std():.4f}  "
              f"min={col.min():.4f}  max={col.max():.4f}")

    # ==============================
    # Walk-forward CV of the meta-learner (on OOF features)
    # ==============================
    print("\n" + "-" * 60)
    print("  WALK-FORWARD CV (meta-learner on OOF features)")
    print("-" * 60)

    wf_results = []
    for fold_idx, boundary in enumerate(FOLD_BOUNDARIES):
        boundary_dt = np.datetime64(boundary)
        train_mask = dates_valid < boundary_dt
        if fold_idx < len(FOLD_BOUNDARIES) - 1:
            next_boundary = np.datetime64(FOLD_BOUNDARIES[fold_idx + 1])
            test_mask = (dates_valid >= boundary_dt) & (dates_valid < next_boundary)
        else:
            test_mask = dates_valid >= boundary_dt

        n_train, n_test = train_mask.sum(), test_mask.sum()
        if n_train < 100 or n_test < 50:
            print(f"  Fold {fold_idx + 1}: skip (train={n_train}, test={n_test})")
            continue

        X_train, y_train = meta_X[train_mask], y[train_mask]
        X_test, y_test = meta_X[test_mask], y[test_mask]

        lr = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")),
        ])
        lr.fit(X_train, y_train)
        p_meta_test = lr.predict_proba(X_test)[:, 1]
        p_base_test = p_baseline[test_mask]

        meta_ats = ats_accuracy(p_meta_test, y_test)
        base_ats = ats_accuracy(p_base_test, y_test)
        meta_ece = ats_ece(p_meta_test, y_test)
        base_ece = ats_ece(p_base_test, y_test)
        meta_brier = brier_score_loss(y_test, p_meta_test)
        base_brier = brier_score_loss(y_test, p_base_test)
        delta_ats = (meta_ats - base_ats) * 100

        wf_results.append({
            "fold": fold_idx + 1, "n_train": n_train, "n_test": n_test,
            "meta_ats": meta_ats, "base_ats": base_ats,
            "meta_ece": meta_ece, "base_ece": base_ece,
            "meta_brier": meta_brier, "base_brier": base_brier,
        })

        marker = "+" if delta_ats > 0 else "-" if delta_ats < 0 else "="
        print(f"  Fold {fold_idx + 1}: train={n_train:,} test={n_test:,}")
        print(f"    ATS: META={meta_ats:.1%} vs BLEND={base_ats:.1%} ({marker}{abs(delta_ats):.1f}pp)")
        print(f"    ECE: META={meta_ece:.4f} vs BLEND={base_ece:.4f}")
        print(f"    Brier: META={meta_brier:.4f} vs BLEND={base_brier:.4f}")

    if not wf_results:
        print("\n  ERROR: No valid folds. Aborting.")
        return

    # Summary
    print("\n" + "-" * 60)
    print("  WALK-FORWARD SUMMARY")
    print("-" * 60)
    avg_meta_ats = np.mean([r["meta_ats"] for r in wf_results])
    avg_base_ats = np.mean([r["base_ats"] for r in wf_results])
    avg_meta_ece = np.mean([r["meta_ece"] for r in wf_results])
    avg_base_ece = np.mean([r["base_ece"] for r in wf_results])
    avg_meta_brier = np.mean([r["meta_brier"] for r in wf_results])
    avg_base_brier = np.mean([r["base_brier"] for r in wf_results])
    total_test = sum(r["n_test"] for r in wf_results)

    print(f"  Total test games: {total_test:,}")
    print(f"  Meta-Learner:  ATS={avg_meta_ats:.1%}  ECE={avg_meta_ece:.4f}  Brier={avg_meta_brier:.4f}")
    print(f"  Linear Blend:  ATS={avg_base_ats:.1%}  ECE={avg_base_ece:.4f}  Brier={avg_base_brier:.4f}")
    print(f"  Delta ATS: {(avg_meta_ats - avg_base_ats) * 100:+.2f}pp")
    print(f"  Delta ECE: {(avg_meta_ece - avg_base_ece) * 100:+.2f}pp")

    beats_baseline = avg_meta_ats > avg_base_ats or avg_meta_brier < avg_base_brier

    # ==============================
    # Train final model on ALL OOF meta features
    # ==============================
    print("\n" + "-" * 60)
    print("  FINAL MODEL (trained on all OOF meta features)")
    print("-" * 60)

    lr_final = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")),
    ])
    lr_final.fit(meta_X, y)

    p_final = lr_final.predict_proba(meta_X)[:, 1]
    final_ats = ats_accuracy(p_final, y)
    final_ece = ats_ece(p_final, y)
    base_final_ats = ats_accuracy(p_baseline, y)
    base_final_ece = ats_ece(p_baseline, y)

    print(f"  In-sample: META ATS={final_ats:.1%}, ECE={final_ece:.4f}")
    print(f"  In-sample: BLEND ATS={base_final_ats:.1%}, ECE={base_final_ece:.4f}")

    lr_step = lr_final.named_steps["lr"]
    print("\n  LogReg coefficients (scaled features):")
    for j, name in enumerate(EXPECTED_FEATURES):
        print(f"    {name:20s}: {lr_step.coef_[0, j]:+.4f}")
    print(f"    {'intercept':20s}: {lr_step.intercept_[0]:+.4f}")

    # Probability distribution
    print("\n  P(cover) distribution (meta-learner):")
    for lo, hi in [(0, 0.40), (0.40, 0.45), (0.45, 0.50), (0.50, 0.55),
                   (0.55, 0.60), (0.60, 1.0)]:
        mask = (p_final >= lo) & (p_final < hi)
        if mask.sum() > 0:
            actual = y[mask].mean()
            print(f"    P∈[{lo:.2f},{hi:.2f}): n={mask.sum():,}  actual={actual:.1%}")

    # ==============================
    # Save
    # ==============================
    if args.dry_run:
        print("\n  DRY RUN — not saving.")
    elif not beats_baseline:
        print(f"\n  SAFEGUARD: Meta-learner did NOT beat baseline in walk-forward.")
        print(f"  Meta ATS={avg_meta_ats:.1%} vs Blend ATS={avg_base_ats:.1%}")
        print(f"  Meta Brier={avg_meta_brier:.4f} vs Blend Brier={avg_base_brier:.4f}")
        print(f"  NOT saving. Keep using linear blend.")
    else:
        out_path = NBA_MARGIN_MODELS_DIR / "ah_meta_learner.pkl"
        artifact = {
            "model": lr_final,
            "feature_names": EXPECTED_FEATURES,
            "walk_forward_ats": avg_meta_ats,
            "walk_forward_ece": avg_meta_ece,
            "baseline_ats": avg_base_ats,
            "baseline_ece": avg_base_ece,
            "oof_training": True,
        }
        joblib.dump(artifact, out_path)
        print(f"\n  SAVED: {out_path}")
        print(f"  Walk-forward ATS: {avg_meta_ats:.1%} (vs {avg_base_ats:.1%} baseline)")

        meta_path = NBA_MARGIN_MODELS_DIR / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            meta["meta_learner"] = True
            meta["meta_learner_ats"] = round(avg_meta_ats, 4)
            meta["meta_learner_oof"] = True
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            print(f"  Updated metadata.json: meta_learner=true (OOF)")

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
