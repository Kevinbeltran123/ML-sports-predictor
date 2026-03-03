"""Auditoría de calibración ATS (Against The Spread).

Backtest sobre dataset.sqlite:
  1. Carga 17K juegos con Margin y MARKET_SPREAD
  2. Usa modelos ML actuales para computar P(home win)
  3. Convierte P(win) → P(cover) con spread_math actual
  4. Agrupa por bins de P(cover) y compara vs actual cover rate
  5. Calcula ATS-ECE (Expected Calibration Error para spreads)
  6. Reporta ATS accuracy por bucket de spread
  7. Opcionalmente entrena Platt recalibrator para P(cover)

Uso:
    PYTHONPATH=. python scripts/audit_ats_calibration.py
    PYTHONPATH=. python scripts/audit_ats_calibration.py --platt     # entrena recalibrator
    PYTHONPATH=. python scripts/audit_ats_calibration.py --game-sigma # usa game-specific sigma
"""

import argparse
import json
import sqlite3

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.config import DATASET_DB, NBA_ML_MODELS_DIR, NBA_MARGIN_MODELS_DIR, DROP_COLUMNS_ML, get_logger
from src.core.betting.spread_math import (
    p_cover, ah_probabilities, game_sigma_from_interval,
    sigma_for_line, _sigma_for_line, _df_for_line,
)

logger = get_logger(__name__)

DEFAULT_DATASET = "dataset_2012-26"
TARGET = "Home-Team-Win"
DATE_COL = "Date"
TEST_DATE = "2025-10-01"

# Calibration bins for P(cover)
PCOVER_BINS = np.arange(0.30, 0.72, 0.02)

# Spread magnitude buckets
SPREAD_BUCKETS = [(2, "0-2"), (5, "2-5"), (8, "5-8"), (12, "8-12"), (99, "12+")]


def load_dataset(dataset_name):
    with sqlite3.connect(DATASET_DB) as con:
        df = pd.read_sql_query(f'SELECT * FROM "{dataset_name}"', con)
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    return df


def add_ts_differential(df):
    if "TS_PCT" in df.columns and "TS_PCT.1" in df.columns:
        df["Diff_TS_PCT"] = df["TS_PCT"].astype(float) - df["TS_PCT.1"].astype(float)


def df_to_x(df):
    X_df = df.drop(columns=DROP_COLUMNS_ML, errors="ignore")
    X = X_df.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0).to_numpy(dtype=float)
    return X


def load_ml_models():
    """Carga modelos ML de produccion para generar P(home win)."""
    import re
    # XGBoost
    xgb_candidates = list(NBA_ML_MODELS_DIR.glob("XGBoost_*ML*.json"))
    if not xgb_candidates:
        raise FileNotFoundError("No XGBoost ML model found")
    acc_re = re.compile(r"_(\d+(?:\.\d+)?)%_")
    best_xgb = max(xgb_candidates, key=lambda p: float(acc_re.search(p.name).group(1)) if acc_re.search(p.name) else 0)
    booster = xgb.Booster()
    booster.load_model(str(best_xgb))
    print(f"  XGBoost loaded: {best_xgb.name}")

    # CatBoost
    cat_candidates = [p for p in NBA_ML_MODELS_DIR.glob("CatBoost_*ML*.pkl")
                      if "conformal" not in p.name.lower() and "calibrat" not in p.name.lower()]
    cat_model = None
    if cat_candidates:
        best_cat = max(cat_candidates, key=lambda p: float(acc_re.search(p.name).group(1)) if acc_re.search(p.name) else 0)
        cat_model = joblib.load(best_cat)
        print(f"  CatBoost loaded: {best_cat.name}")

    # Weights
    meta_path = NBA_ML_MODELS_DIR / "metadata.json"
    w_xgb, w_cat = 0.95, 0.05
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        w = meta.get("weights", {})
        if "xgb" in w:
            w_xgb, w_cat = w["xgb"], w.get("cat", 1 - w["xgb"])

    return booster, cat_model, w_xgb, w_cat


def predict_p_home(booster, cat_model, w_xgb, w_cat, X):
    """Genera P(home win) del ensemble para cada juego."""
    dmat = xgb.DMatrix(X)
    p_xgb = booster.predict(dmat)  # (N, 2)
    p_home_xgb = p_xgb[:, 1]

    if cat_model is not None:
        p_cat = cat_model.predict_proba(X)[:, 1]
        return w_xgb * p_home_xgb + w_cat * p_cat
    return p_home_xgb


def compute_p_cover_array(p_home_arr, spreads, use_game_sigma=False, interval_widths=None):
    """Computa P(home cover) para cada juego."""
    n = len(p_home_arr)
    p_cover_arr = np.zeros(n)
    sigmas_used = np.zeros(n)

    for i in range(n):
        if use_game_sigma and interval_widths is not None:
            sigma = game_sigma_from_interval(float(interval_widths[i]), float(spreads[i]))
        else:
            sigma = None  # usa bucket default
        p_cover_arr[i] = p_cover(float(p_home_arr[i]), float(spreads[i]), sigma=sigma)
        sigmas_used[i] = sigma if sigma else sigma_for_line(float(spreads[i]))

    return p_cover_arr, sigmas_used


def compute_actual_cover(margins, spreads):
    """ATS result: home covers when Margin + MARKET_SPREAD > 0.
    Push (== 0) se cuenta como 0.5."""
    residuals = margins + spreads
    covers = np.where(residuals > 0, 1.0, np.where(residuals == 0, 0.5, 0.0))
    return covers


def calibration_curve(p_cover_arr, actual_covers, bins):
    """Calibration curve: predicted P(cover) bins vs actual cover rate."""
    rows = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (p_cover_arr >= lo) & (p_cover_arr < hi)
        n = mask.sum()
        if n < 10:
            continue
        actual_rate = actual_covers[mask].mean()
        predicted_rate = p_cover_arr[mask].mean()
        rows.append({
            "bin": f"{lo:.2f}-{hi:.2f}",
            "n": int(n),
            "predicted": round(predicted_rate, 4),
            "actual": round(actual_rate, 4),
            "gap": round(actual_rate - predicted_rate, 4),
        })
    return pd.DataFrame(rows)


def compute_ats_ece(p_cover_arr, actual_covers, n_bins=15):
    """Expected Calibration Error para ATS."""
    bins = np.linspace(0.3, 0.7, n_bins + 1)
    ece = 0.0
    total = len(p_cover_arr)
    for i in range(n_bins):
        mask = (p_cover_arr >= bins[i]) & (p_cover_arr < bins[i + 1])
        n = mask.sum()
        if n == 0:
            continue
        avg_pred = p_cover_arr[mask].mean()
        avg_actual = actual_covers[mask].mean()
        ece += (n / total) * abs(avg_actual - avg_pred)
    return ece


def ats_accuracy_by_spread(p_cover_arr, actual_covers, spreads, buckets):
    """ATS accuracy por bucket de spread magnitude."""
    rows = []
    abs_spreads = np.abs(spreads)
    prev = 0
    for edge, label in buckets:
        mask = (abs_spreads > prev) & (abs_spreads <= edge)
        n = mask.sum()
        if n < 10:
            prev = edge
            continue
        # Pick el lado con mayor P(cover) — si P(home_cover) > 0.5, pick home
        picks = (p_cover_arr[mask] > 0.5).astype(float)
        actual = actual_covers[mask]
        # Correct picks: picked home cover AND home covered, OR picked away AND away covered
        correct = np.where(picks == 1, actual, 1 - actual)
        acc = correct.mean()
        rows.append({
            "spread_bucket": label,
            "n": int(n),
            "ats_accuracy": round(acc, 4),
            "avg_p_cover": round(p_cover_arr[mask].mean(), 4),
            "avg_sigma": round(sigma_for_line(edge - 1), 1),
        })
        prev = edge
    return pd.DataFrame(rows)


def train_platt_recalibrator(p_cover_train, actual_train):
    """Entrena Platt recalibrator: LogisticRegression(P(cover)) → P(actual cover)."""
    X = p_cover_train.reshape(-1, 1)
    y = (actual_train > 0.5).astype(int)  # binary cover
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    lr.fit(X, y)
    return lr


def main():
    parser = argparse.ArgumentParser(description="ATS Calibration Audit")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--platt", action="store_true", help="Entrena y guarda Platt recalibrator")
    parser.add_argument("--game-sigma", action="store_true", help="Usa game-specific sigma (Q10/Q90)")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  ATS CALIBRATION AUDIT")
    print(f"{'='*65}")

    # Load dataset
    df = load_dataset(args.dataset)
    add_ts_differential(df)
    print(f"  Dataset: {len(df)} games ({df[DATE_COL].min().date()} to {df[DATE_COL].max().date()})")

    # Check required columns
    if "Margin" not in df.columns or "MARKET_SPREAD" not in df.columns:
        print("  ERROR: Dataset missing Margin or MARKET_SPREAD columns")
        return

    margins = df["Margin"].to_numpy(dtype=float)
    spreads = df["MARKET_SPREAD"].to_numpy(dtype=float)

    # Temporal split
    test_dt = pd.to_datetime(TEST_DATE)
    train_mask = df[DATE_COL] < test_dt
    test_mask = df[DATE_COL] >= test_dt
    print(f"  Train: {train_mask.sum()} | Test: {test_mask.sum()}")

    # Load ML models
    print(f"\n  Loading ML models...")
    booster, cat_model, w_xgb, w_cat = load_ml_models()
    print(f"  Weights: XGB={w_xgb:.2f}, Cat={w_cat:.2f}")

    # Generate P(home win) for all games
    X = df_to_x(df)
    p_home = predict_p_home(booster, cat_model, w_xgb, w_cat, X)

    # Compute P(cover) — classifier path
    interval_widths = None
    if args.game_sigma:
        # Load Q10/Q90 models for game-specific sigma
        from src.sports.nba.predict.margin_runner import predict_margin_interval, _load_margin_models
        from src.config import DROP_COLUMNS_MARGIN
        _load_margin_models()
        X_margin = df.drop(columns=DROP_COLUMNS_MARGIN, errors="ignore")
        X_margin = X_margin.replace([np.inf, -np.inf], np.nan).fillna(X_margin.median(numeric_only=True)).fillna(0).to_numpy(dtype=float)
        result = predict_margin_interval(X_margin)
        if result is not None:
            _, _, interval_widths = result
            print(f"  Game-specific sigma: interval_width mean={interval_widths.mean():.1f}")
        else:
            print(f"  WARNING: Q10/Q90 models not available, using bucket sigma")

    p_cover_arr, sigmas = compute_p_cover_array(p_home, spreads, args.game_sigma, interval_widths)
    actual_covers = compute_actual_cover(margins, spreads)

    # --- Overall stats ---
    print(f"\n{'='*65}")
    print(f"  OVERALL ATS STATS")
    print(f"{'='*65}")
    # ATS accuracy: pick the side with P(cover) further from 0.5
    picks_correct = np.where(p_cover_arr > 0.5, actual_covers, 1 - actual_covers)
    overall_ats = picks_correct.mean()
    print(f"  Overall ATS accuracy: {overall_ats:.1%} ({len(df)} games)")
    print(f"  Test ATS accuracy:    {picks_correct[test_mask].mean():.1%} ({test_mask.sum()} games)")

    # --- Calibration curve (test set) ---
    print(f"\n{'='*65}")
    print(f"  P(COVER) CALIBRATION — TEST SET")
    print(f"{'='*65}")

    bins = np.arange(0.30, 0.72, 0.02)
    cal_df = calibration_curve(p_cover_arr[test_mask], actual_covers[test_mask], bins)
    if len(cal_df) > 0:
        print(f"\n  {'Bin':>12s} {'N':>6s} {'Predicted':>10s} {'Actual':>8s} {'Gap':>8s}")
        print(f"  {'-'*48}")
        for _, row in cal_df.iterrows():
            gap_str = f"{row['gap']:+.4f}"
            marker = " *" if abs(row["gap"]) > 0.05 else ""
            print(f"  {row['bin']:>12s} {row['n']:>6d} {row['predicted']:>10.4f} {row['actual']:>8.4f} {gap_str:>8s}{marker}")

    # ECE
    ece_test = compute_ats_ece(p_cover_arr[test_mask], actual_covers[test_mask])
    ece_all = compute_ats_ece(p_cover_arr, actual_covers)
    print(f"\n  ATS-ECE (test): {ece_test:.4f}")
    print(f"  ATS-ECE (all):  {ece_all:.4f}")

    # --- ATS accuracy by spread bucket ---
    print(f"\n{'='*65}")
    print(f"  ATS ACCURACY BY SPREAD BUCKET — TEST SET")
    print(f"{'='*65}")

    spread_df = ats_accuracy_by_spread(
        p_cover_arr[test_mask], actual_covers[test_mask],
        spreads[test_mask], SPREAD_BUCKETS
    )
    if len(spread_df) > 0:
        print(f"\n  {'Bucket':>10s} {'N':>6s} {'ATS Acc':>8s} {'Avg P':>8s} {'σ':>6s}")
        print(f"  {'-'*42}")
        for _, row in spread_df.iterrows():
            print(f"  {row['spread_bucket']:>10s} {row['n']:>6d} {row['ats_accuracy']:>8.1%} {row['avg_p_cover']:>8.4f} {row['avg_sigma']:>6.1f}")

    # --- Sigma distribution ---
    print(f"\n{'='*65}")
    print(f"  SIGMA DISTRIBUTION")
    print(f"{'='*65}")
    print(f"  Mean: {sigmas.mean():.2f} | Std: {sigmas.std():.2f} | Range: [{sigmas.min():.1f}, {sigmas.max():.1f}]")

    # --- Platt recalibration ---
    if args.platt:
        print(f"\n{'='*65}")
        print(f"  PLATT RECALIBRATION")
        print(f"{'='*65}")

        # Train on calibration portion of training set (last 20%)
        train_indices = np.where(train_mask)[0]
        cal_start = int(len(train_indices) * 0.8)
        cal_indices = train_indices[cal_start:]

        platt = train_platt_recalibrator(p_cover_arr[cal_indices], actual_covers[cal_indices])

        # Evaluate on test set
        p_cover_recal = platt.predict_proba(p_cover_arr[test_mask].reshape(-1, 1))[:, 1]
        ece_recal = compute_ats_ece(p_cover_recal, actual_covers[test_mask])

        print(f"  Platt trained on {len(cal_indices)} calibration games")
        print(f"  ATS-ECE before Platt: {ece_test:.4f}")
        print(f"  ATS-ECE after Platt:  {ece_recal:.4f}")
        print(f"  Improvement: {ece_test - ece_recal:+.4f}")

        # Calibration curve after Platt
        cal_df_recal = calibration_curve(p_cover_recal, actual_covers[test_mask], bins)
        if len(cal_df_recal) > 0:
            print(f"\n  {'Bin':>12s} {'N':>6s} {'Predicted':>10s} {'Actual':>8s} {'Gap':>8s}")
            print(f"  {'-'*48}")
            for _, row in cal_df_recal.iterrows():
                gap_str = f"{row['gap']:+.4f}"
                print(f"  {row['bin']:>12s} {row['n']:>6d} {row['predicted']:>10.4f} {row['actual']:>8.4f} {gap_str:>8s}")

        # Save
        if ece_recal < ece_test:
            out_path = NBA_MARGIN_MODELS_DIR / "ats_platt_calibration.pkl"
            NBA_MARGIN_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(platt, out_path)
            print(f"\n  Platt recalibrator saved to {out_path}")
        else:
            print(f"\n  Platt no mejora — NO guardado")

    print(f"\n{'='*65}")
    print(f"  AUDIT COMPLETO")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
