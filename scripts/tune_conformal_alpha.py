"""Tune conformal alpha para optimizar el trade-off BET volume vs accuracy.

Alpha mas bajo (0.05) = mas conservador, menos BETs pero mas precisos.
Alpha mas alto (0.30) = mas agresivo, mas BETs pero menos filtrado.

Evalua contra datos reales de produccion (BetsTracking.sqlite) y contra
el set de calibracion del entrenamiento.

Uso:
    PYTHONPATH=. python scripts/tune_conformal_alpha.py
    PYTHONPATH=. python scripts/tune_conformal_alpha.py --apply 0.15  # guardar nuevo alpha
"""

import argparse
import sqlite3

import joblib
import numpy as np

from src.config import BETS_DB, NBA_ML_MODELS_DIR, DATASET_DB, DROP_COLUMNS_ML, get_logger
from src.core.calibration.conformal import ConformalClassifier

logger = get_logger(__name__)

TARGET = "Home-Team-Win"
DATE_COL = "Date"


def evaluate_alpha_on_production(alpha_values):
    """Evalua distintos alphas contra predicciones reales con resultado."""
    if not BETS_DB.exists():
        print("  BetsTracking.sqlite no encontrado")
        return

    with sqlite3.connect(BETS_DB) as con:
        df_rows = con.execute("""
            SELECT prob_home, prob_away, ml_correct,
                   ev_home, ev_away, ml_home_odds, ml_away_odds
            FROM predictions
            WHERE ml_correct IS NOT NULL
        """).fetchall()

    if not df_rows:
        print("  No hay predicciones con resultado")
        return

    probs = np.array([[r[1], r[0]] for r in df_rows])  # [P(away), P(home)]
    y_correct = np.array([r[2] for r in df_rows])
    ev_home = np.array([r[3] for r in df_rows])
    ev_away = np.array([r[4] for r in df_rows])
    odds_home = np.array([r[5] for r in df_rows])
    odds_away = np.array([r[6] for r in df_rows])

    n = len(y_correct)
    print(f"\n{'='*70}")
    print(f"  CONFORMAL ALPHA TUNING — Production Data ({n} predictions)")
    print(f"{'='*70}")
    print(f"  {'Alpha':<8} {'Coverage':<10} {'Threshold':<11} {'BET':<8} "
          f"{'BET%':<8} {'BET Acc':<10} {'P&L($100)':<12} {'ROI':<8}")
    print(f"  {'-'*8} {'-'*10} {'-'*11} {'-'*8} {'-'*8} {'-'*10} {'-'*12} {'-'*8}")

    # Cargar conformal actual para obtener cal_scores
    conf_path = NBA_ML_MODELS_DIR / "ensemble_conformal.pkl"
    current_conf = joblib.load(conf_path)
    cal_scores = current_conf.cal_scores_

    results = []
    for alpha in alpha_values:
        # Recompute threshold para este alpha
        n_cal = len(cal_scores)
        quantile_level = min(1.0, (1 - alpha) * (1 + 1 / n_cal))
        quantile = float(np.quantile(cal_scores, quantile_level))
        threshold = 1.0 - quantile

        # Prediction sets
        pred_sets = probs >= threshold
        set_sizes = pred_sets.sum(axis=1).astype(int)

        # BET = set_size 1
        bet_mask = set_sizes == 1
        n_bet = bet_mask.sum()
        bet_pct = n_bet / n if n > 0 else 0
        bet_acc = y_correct[bet_mask].mean() if n_bet > 0 else 0

        # Coverage
        picks = np.argmax(probs, axis=1)
        correct_class = np.where(y_correct == 1, picks, 1 - picks)
        in_set = pred_sets[np.arange(n), correct_class]
        coverage = in_set.mean()

        # Simulated P&L (flat $100 on BET picks, best EV side)
        pnl = 0.0
        for i in range(n):
            if not bet_mask[i]:
                continue
            # Pick best EV side
            pick_home = probs[i, 1] >= 0.5
            odds = int(odds_home[i]) if pick_home else int(odds_away[i])
            if y_correct[i] == 1:
                pnl += 100 * (odds / 100) if odds > 0 else 100 * (100 / abs(odds))
            else:
                pnl -= 100

        roi = (pnl / (n_bet * 100) * 100) if n_bet > 0 else 0

        marker = " ◄" if alpha == current_conf.alpha else ""
        print(f"  {alpha:<8.2f} {coverage:<10.1%} {threshold:<11.3f} "
              f"{n_bet:<8} {bet_pct:<8.0%} {bet_acc:<10.1%} "
              f"${pnl:<+11.2f} {roi:<+7.1f}%{marker}")

        results.append({
            "alpha": alpha, "coverage": coverage, "threshold": threshold,
            "n_bet": n_bet, "bet_pct": bet_pct, "bet_acc": bet_acc,
            "pnl": pnl, "roi": roi,
        })

    return results


def evaluate_alpha_on_training(alpha_values):
    """Evalua alphas contra el set de test del entrenamiento (mas datos)."""
    import pandas as pd

    with sqlite3.connect(DATASET_DB) as con:
        df = pd.read_sql_query('SELECT * FROM "dataset_2012-26"', con)

    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)

    if "Diff_TS_PCT" not in df.columns and "TS_PCT" in df.columns:
        df["Diff_TS_PCT"] = df["TS_PCT"].astype(float) - df["TS_PCT.1"].astype(float)

    test_dt = pd.to_datetime("2025-10-01")
    train_df = df[df[DATE_COL] < test_dt].copy()
    test_df = df[df[DATE_COL] >= test_dt].copy()

    if len(test_df) == 0:
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

    y_test = test_df[TARGET].astype(int).to_numpy()
    X_test = test_df.drop(columns=DROP_COLUMNS_ML, errors="ignore")
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=float)

    # Load current models and predict
    import xgboost as xgb
    from src.core.calibration.xgb_calibrator import XGBCalibrator

    xgb_path = max(
        [p for p in NBA_ML_MODELS_DIR.glob("XGBoost_*ML*.json")],
        key=lambda p: p.stat().st_mtime,
    )
    cal_path = xgb_path.with_name(xgb_path.stem + "_calibration.pkl")
    calibrator = joblib.load(cal_path)
    probs_xgb = calibrator.predict_proba(X_test)

    cat_path = max(
        [p for p in NBA_ML_MODELS_DIR.glob("CatBoost_*ML*.pkl")
         if "calibration" not in p.name],
        key=lambda p: p.stat().st_mtime,
    )
    cat_model = joblib.load(cat_path)
    probs_cat = cat_model.predict_proba(X_test)

    # Ensemble with current weights
    meta_path = NBA_ML_MODELS_DIR / "metadata.json"
    import json
    with open(meta_path) as f:
        meta = json.load(f)
    w_xgb = meta["weights"]["xgb"]
    w_cat = meta["weights"]["cat"]
    probs = w_xgb * probs_xgb + w_cat * probs_cat

    # Calibration scores from current conformal
    conf_path = NBA_ML_MODELS_DIR / "ensemble_conformal.pkl"
    current_conf = joblib.load(conf_path)
    cal_scores = current_conf.cal_scores_

    n = len(y_test)
    print(f"\n{'='*70}")
    print(f"  CONFORMAL ALPHA TUNING — Test Set ({n} games)")
    print(f"{'='*70}")
    print(f"  {'Alpha':<8} {'Coverage':<10} {'Threshold':<11} {'BET':<8} "
          f"{'BET%':<8} {'BET Acc':<10} {'SKIP Acc':<10}")
    print(f"  {'-'*8} {'-'*10} {'-'*11} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")

    for alpha in alpha_values:
        n_cal = len(cal_scores)
        quantile_level = min(1.0, (1 - alpha) * (1 + 1 / n_cal))
        quantile = float(np.quantile(cal_scores, quantile_level))
        threshold = 1.0 - quantile

        pred_sets = probs >= threshold
        set_sizes = pred_sets.sum(axis=1).astype(int)

        bet_mask = set_sizes == 1
        skip_mask = set_sizes == 2
        n_bet = bet_mask.sum()

        pred_labels = np.argmax(probs, axis=1)
        correct = (pred_labels == y_test)

        bet_acc = correct[bet_mask].mean() if n_bet > 0 else 0
        skip_acc = correct[skip_mask].mean() if skip_mask.sum() > 0 else 0

        correct_class = y_test
        in_set = pred_sets[np.arange(n), correct_class]
        coverage = in_set.mean()

        marker = " ◄" if alpha == current_conf.alpha else ""
        print(f"  {alpha:<8.2f} {coverage:<10.1%} {threshold:<11.3f} "
              f"{n_bet:<8} {n_bet/n:<8.0%} {bet_acc:<10.1%} {skip_acc:<10.1%}{marker}")


def apply_alpha(alpha):
    """Refit conformal con nuevo alpha y guarda."""
    import pandas as pd

    with sqlite3.connect(DATASET_DB) as con:
        df = pd.read_sql_query('SELECT * FROM "dataset_2012-26"', con)

    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)

    if "Diff_TS_PCT" not in df.columns and "TS_PCT" in df.columns:
        df["Diff_TS_PCT"] = df["TS_PCT"].astype(float) - df["TS_PCT.1"].astype(float)

    test_dt = pd.to_datetime("2025-10-01")
    train_df = df[df[DATE_COL] < test_dt].copy()

    y_train = train_df[TARGET].astype(int).to_numpy()
    X_train = train_df.drop(columns=DROP_COLUMNS_ML, errors="ignore")
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=float)

    # Use last 20% of training as calibration
    cal_start = int(len(X_train) * 0.8)
    X_cal = X_train[cal_start:]
    y_cal = y_train[cal_start:]

    # Predict with current models
    import json

    xgb_path = max(
        [p for p in NBA_ML_MODELS_DIR.glob("XGBoost_*ML*.json")],
        key=lambda p: p.stat().st_mtime,
    )
    cal_path = xgb_path.with_name(xgb_path.stem + "_calibration.pkl")
    calibrator = joblib.load(cal_path)
    probs_xgb = calibrator.predict_proba(X_cal)

    cat_path = max(
        [p for p in NBA_ML_MODELS_DIR.glob("CatBoost_*ML*.pkl")
         if "calibration" not in p.name],
        key=lambda p: p.stat().st_mtime,
    )
    cat_model = joblib.load(cat_path)
    probs_cat = cat_model.predict_proba(X_cal)

    meta_path = NBA_ML_MODELS_DIR / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    w_xgb = meta["weights"]["xgb"]
    w_cat = meta["weights"]["cat"]
    probs_cal = w_xgb * probs_xgb + w_cat * probs_cat

    # Fit new conformal
    conf = ConformalClassifier(alpha=alpha)
    conf.fit(probs_cal, y_cal)

    out_path = NBA_ML_MODELS_DIR / "ensemble_conformal.pkl"
    joblib.dump(conf, out_path)

    # Update metadata
    meta["conformal_summary"] = conf.summary()
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Applied alpha={alpha}")
    print(f"  {conf}")
    print(f"  Saved → {out_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Tune conformal alpha")
    parser.add_argument("--apply", type=float, help="Apply specific alpha and save")
    args = parser.parse_args()

    alphas = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]

    if args.apply:
        apply_alpha(args.apply)
        return

    evaluate_alpha_on_training(alphas)
    evaluate_alpha_on_production(alphas)

    print(f"\n  Para aplicar un alpha: PYTHONPATH=. python scripts/tune_conformal_alpha.py --apply 0.15")
    print()


if __name__ == "__main__":
    main()
