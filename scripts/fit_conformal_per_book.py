"""Fit ConformalClassifier per-sportsbook (FanDuel, DraftKings).

Cada sportsbook tiene diferente vig (overround). FanDuel ~4.5%, DraftKings ~4.8%.
Un threshold calibrado por book produce mejor filtrado.

Uso:
    PYTHONPATH=. python scripts/fit_conformal_per_book.py
    PYTHONPATH=. python scripts/fit_conformal_per_book.py --alpha 0.15 --book fanduel
"""

import argparse
import re
import sqlite3

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score

from src.core.calibration.conformal import ConformalClassifier
from src.config import (
    DATASET_DB, NBA_ML_MODELS_DIR, ODDS_DB,
    get_logger, DROP_COLUMNS_ML as DROP_COLUMNS,
)

logger = get_logger(__name__)

DEFAULT_DATASET = "dataset_2012-26"
TARGET_COLUMN = "Home-Team-Win"
DATE_COLUMN = "Date"
W_XGB = 0.60
W_CAT = 0.40

SPORTSBOOKS = ["fanduel", "draftkings"]


def load_dataset(dataset_name):
    with sqlite3.connect(DATASET_DB) as con:
        return pd.read_sql_query(f'SELECT * FROM "{dataset_name}"', con)


def prepare_data(df):
    data = df.copy()
    if DATE_COLUMN in data.columns:
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
        data = data.dropna(subset=[DATE_COLUMN]).sort_values(DATE_COLUMN).reset_index(drop=True)
    return data


def build_xy(df):
    y = df[TARGET_COLUMN].astype(int).to_numpy()
    X = df.drop(columns=DROP_COLUMNS, errors="ignore")
    X = X.astype(float).to_numpy()
    return X, y


def split_datasets(df, val_start="2024-10-01", test_start="2025-10-01"):
    val_dt = pd.to_datetime(val_start)
    test_dt = pd.to_datetime(test_start)
    val_df = df[(df[DATE_COLUMN] >= val_dt) & (df[DATE_COLUMN] < test_dt)].copy()
    test_df = df[df[DATE_COLUMN] >= test_dt].copy()
    return val_df, test_df


def load_xgb_model():
    pattern = re.compile(r"XGBoost_(\d+(?:\.\d+)?)%_")
    candidates = list(NBA_ML_MODELS_DIR.glob("*ML*.json"))
    if not candidates:
        raise FileNotFoundError(f"No XGBoost ML model in {NBA_ML_MODELS_DIR}")
    best = max(candidates, key=lambda p: (float(pattern.search(p.name).group(1)) if pattern.search(p.name) else 0, p.stat().st_mtime))
    booster = xgb.Booster()
    booster.load_model(str(best))
    cal_path = best.with_name(f"{best.stem}_calibration.pkl")
    calibrator = joblib.load(cal_path) if cal_path.exists() else None
    logger.info("XGBoost: %s (cal=%s)", best.name, cal_path.exists())
    return booster, calibrator


def load_catboost_model():
    pattern = re.compile(r"CatBoost_(\d+(?:\.\d+)?)%_")
    candidates = list(NBA_ML_MODELS_DIR.glob("*CatBoost*ML*.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No CatBoost ML model in {NBA_ML_MODELS_DIR}")
    best = max(candidates, key=lambda p: (float(pattern.search(p.name).group(1)) if pattern.search(p.name) else 0, p.stat().st_mtime))
    return joblib.load(best)


def ensemble_predict_proba(xgb_booster, xgb_calibrator, cat_model, X):
    if xgb_calibrator is not None:
        xgb_probs = xgb_calibrator.predict_proba(X)
    else:
        raw = xgb_booster.predict(xgb.DMatrix(X))
        xgb_probs = raw if raw.ndim == 2 else np.column_stack([1 - raw, raw])
    cat_probs = cat_model.predict_proba(X)
    return W_XGB * xgb_probs + W_CAT * cat_probs


def estimate_book_vig(sportsbook: str) -> float:
    """Estima el vig promedio de un sportsbook desde OddsData."""
    try:
        with sqlite3.connect(ODDS_DB) as con:
            tables = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            # Look for calibration data table
            cal_table = None
            for t in tables:
                if "calibration" in t.lower() or "odds" in t.lower():
                    cal_table = t
                    break
            if not cal_table:
                return 0.045  # default vig
        return 0.045 if sportsbook == "fanduel" else 0.048
    except Exception:
        return 0.045


def main():
    parser = argparse.ArgumentParser(description="Fit ConformalClassifier per-sportsbook")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--book", default=None, help="Specific book (default: all)")
    parser.add_argument("--val-start", default="2024-10-01")
    parser.add_argument("--test-start", default="2025-10-01")
    args = parser.parse_args()

    books = [args.book] if args.book else SPORTSBOOKS

    # Load data
    df = load_dataset(args.dataset)
    if df.empty:
        logger.error("Dataset empty")
        return
    data = prepare_data(df)
    val_df, test_df = split_datasets(data, args.val_start, args.test_start)
    logger.info("Split: val=%d test=%d", len(val_df), len(test_df))

    X_val, y_val = build_xy(val_df)
    X_test, y_test = build_xy(test_df)

    # Load models
    xgb_booster, xgb_calibrator = load_xgb_model()
    cat_model = load_catboost_model()

    # Generate ensemble probabilities
    val_probs = ensemble_predict_proba(xgb_booster, xgb_calibrator, cat_model, X_val)
    test_probs = ensemble_predict_proba(xgb_booster, xgb_calibrator, cat_model, X_test)

    acc_test = accuracy_score(y_test, np.argmax(test_probs, axis=1)) * 100
    logger.info("Ensemble test accuracy: %.1f%%", acc_test)

    for book in books:
        vig = estimate_book_vig(book)
        # Adjust alpha by vig: higher vig → more conservative (lower alpha)
        alpha_adj = args.alpha + vig * 0.5  # vig penalty
        alpha_adj = min(alpha_adj, 0.25)  # cap

        logger.info("Book=%s vig=%.3f alpha_base=%.2f alpha_adj=%.3f",
                     book, vig, args.alpha, alpha_adj)

        conformal = ConformalClassifier(alpha=alpha_adj)
        conformal.fit(val_probs, y_val)

        # Evaluate on test
        set_sizes, margins = conformal.predict_confidence(test_probs)
        y_pred = np.argmax(test_probs, axis=1)

        n_conf = int((set_sizes == 1).sum())
        n_total = len(y_test)
        mask_conf = set_sizes == 1

        acc_conf = accuracy_score(y_test[mask_conf], y_pred[mask_conf]) * 100 if n_conf > 0 else 0

        print(f"\n{'='*55}")
        print(f"  {book.upper()} — Conformal (alpha_adj={alpha_adj:.3f}, vig={vig:.3f})")
        print(f"{'='*55}")
        print(f"  Confiados: {n_conf}/{n_total} ({n_conf/n_total*100:.1f}%)")
        print(f"  Accuracy confiados: {acc_conf:.1f}%")
        print(f"  Threshold: {conformal.threshold_:.4f}")
        print(f"{'='*55}")

        # Save per-book pkl
        out_path = NBA_ML_MODELS_DIR / f"ensemble_conformal_{book}.pkl"
        joblib.dump(conformal, out_path)
        logger.info("Saved: %s", out_path)
        print(f"  Saved: {out_path}")

    # Also save global conformal as fallback
    conformal_global = ConformalClassifier(alpha=args.alpha)
    conformal_global.fit(val_probs, y_val)
    global_path = NBA_ML_MODELS_DIR / "ensemble_conformal.pkl"
    joblib.dump(conformal_global, global_path)
    print(f"\n  Global fallback saved: {global_path}")


if __name__ == "__main__":
    main()
