"""Conformal Prediction Audit — valida cobertura por sportsbook.

Verifica que el ConformalClassifier sigue cumpliendo su garantia de cobertura
(90% por defecto) usando datos reales de BetsTracking.sqlite.

Si la cobertura cae por debajo del target para un book especifico, genera
un conformal per-book con alpha ajustado al vig de ese sportsbook.

POR QUE IMPORTA:
  El conformal global se calibro con datos del modelo (train/cal split).
  Pero en produccion, las odds de cada book tienen vig diferente:
    - FanDuel: ~4.5% vig → probs implicitas infladas
    - DraftKings: ~4.8% vig → ligeramente mas infladas
  El vig afecta la relacion entre nuestras probs y el mercado.
  Un conformal per-book ajusta el threshold a la realidad de cada book.

Uso:
    PYTHONPATH=. python scripts/audit_conformal.py
    PYTHONPATH=. python scripts/audit_conformal.py --min-games 30
    PYTHONPATH=. python scripts/audit_conformal.py --refit  # genera per-book conformal si falla
"""

import argparse
import sqlite3
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd

from src.config import BETS_DB, NBA_ML_MODELS_DIR, get_logger
from src.core.calibration.conformal import ConformalClassifier

logger = get_logger(__name__)

DEFAULT_MIN_GAMES = 30  # minimo de juegos por book para evaluar


def load_predictions_with_results():
    """Carga predicciones que tienen resultado (ml_correct no nulo)."""
    if not BETS_DB.exists():
        logger.error("BetsTracking.sqlite no encontrado")
        return pd.DataFrame()

    con = sqlite3.connect(BETS_DB)
    df = pd.read_sql_query(
        "SELECT * FROM predictions WHERE ml_correct IS NOT NULL",
        con,
    )
    con.close()

    if df.empty:
        logger.warning("No hay predicciones con resultados. Ejecuta update_results primero.")
    return df


def audit_global_conformal(df, conformal):
    """Evalua cobertura del conformal global en datos reales."""
    print(f"\n{'='*60}")
    print(f"  CONFORMAL AUDIT — Global")
    print(f"{'='*60}")
    print(f"  Total predictions con resultado: {len(df)}")
    print(f"  Conformal threshold: {conformal.threshold_:.4f}")
    print(f"  Target coverage: {1 - conformal.alpha:.0%}")
    print()

    # Reconstruir probs y evaluar
    probs = np.column_stack([df["prob_away"].values, df["prob_home"].values])
    set_sizes, margins = conformal.predict_confidence(probs)

    # Cobertura: de los que tienen set_size=1 o 2, cuantos aciertan?
    y_true = df["ml_correct"].astype(int).values

    # Para conformal, "cobertura" = la clase correcta esta en el prediction set
    pred_sets = conformal.predict_sets(probs)
    # La clase predicha es argmax(probs). Pero para cobertura conformal,
    # necesitamos: clase_correcta in prediction_set
    # ml_correct=1 si nuestro pick fue correcto
    # pick = argmax(probs) = 1 si home, 0 si away
    picks = np.argmax(probs, axis=1)
    # Clase correcta: si ml_correct=1, correcta=pick; si ml_correct=0, correcta=1-pick
    correct_class = np.where(y_true == 1, picks, 1 - picks)
    in_set = pred_sets[np.arange(len(df)), correct_class]
    coverage = in_set.mean()

    print(f"  Empirical coverage (all): {coverage:.1%}")

    # Desglose por set_size
    for ss in [1, 2]:
        mask = set_sizes == ss
        if mask.sum() == 0:
            continue
        n_ss = mask.sum()
        acc_ss = y_true[mask].mean()
        cov_ss = in_set[mask].mean()
        print(f"  set_size={ss}: n={n_ss} ({n_ss/len(df):.0%}), "
              f"accuracy={acc_ss:.1%}, coverage={cov_ss:.1%}")

    # Veredicto
    target = 1 - conformal.alpha
    if coverage >= target - 0.02:
        print(f"\n  PASS — cobertura {coverage:.1%} >= target {target:.0%} (tolerancia -2pp)")
    else:
        print(f"\n  FAIL — cobertura {coverage:.1%} < target {target:.0%}")
        print(f"  Recomendacion: refit conformal con datos recientes")

    return coverage, set_sizes, margins


def audit_per_book(df, conformal, min_games=DEFAULT_MIN_GAMES):
    """Evalua cobertura por sportsbook."""
    print(f"\n{'='*60}")
    print(f"  CONFORMAL AUDIT — Per Sportsbook")
    print(f"{'='*60}")

    if "sportsbook" not in df.columns:
        print("  No hay columna sportsbook en la BD")
        return {}

    results = {}
    target = 1 - conformal.alpha

    for book, group in df.groupby("sportsbook"):
        if len(group) < min_games:
            print(f"  {book}: {len(group)} juegos (< {min_games} min, skip)")
            continue

        probs = np.column_stack([group["prob_away"].values, group["prob_home"].values])
        y_true = group["ml_correct"].astype(int).values
        picks = np.argmax(probs, axis=1)
        correct_class = np.where(y_true == 1, picks, 1 - picks)

        pred_sets = conformal.predict_sets(probs)
        in_set = pred_sets[np.arange(len(group)), correct_class]
        coverage = in_set.mean()

        set_sizes, _ = conformal.predict_confidence(probs)
        n_bet = (set_sizes == 1).sum()
        n_skip = (set_sizes == 2).sum()
        acc_bet = y_true[set_sizes == 1].mean() if n_bet > 0 else 0

        status = "PASS" if coverage >= target - 0.02 else "FAIL"
        print(f"  {book}: coverage={coverage:.1%} [{status}], "
              f"n={len(group)}, BET={n_bet} ({acc_bet:.1%}), SKIP={n_skip}")

        results[book] = {
            "coverage": coverage,
            "n_games": len(group),
            "n_bet": n_bet,
            "n_skip": n_skip,
            "accuracy_bet": acc_bet,
            "status": status,
        }

    return results


def refit_per_book(df, min_games=DEFAULT_MIN_GAMES, alpha=0.10):
    """Genera conformal per-book para books con suficientes datos."""
    print(f"\n{'='*60}")
    print(f"  REFIT — Generating per-book conformal classifiers")
    print(f"{'='*60}")

    if "sportsbook" not in df.columns:
        print("  No hay columna sportsbook")
        return

    for book, group in df.groupby("sportsbook"):
        if len(group) < min_games:
            continue

        probs = np.column_stack([group["prob_away"].values, group["prob_home"].values])
        y_true = group["ml_correct"].astype(int).values
        picks = np.argmax(probs, axis=1)
        correct_class = np.where(y_true == 1, picks, 1 - picks)

        # Refit conformal con datos reales de este book
        conf = ConformalClassifier(alpha=alpha)
        conf.fit(probs, correct_class)

        path = NBA_ML_MODELS_DIR / f"ensemble_conformal_{book}.pkl"
        joblib.dump(conf, path)
        print(f"  {book}: threshold={conf.threshold_:.4f}, "
              f"coverage={conf.coverage_:.1%}, saved → {path.name}")


def print_summary(global_coverage, book_results, target=0.90):
    """Resumen final con recomendaciones."""
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")

    all_pass = True
    if global_coverage < target - 0.02:
        print(f"  GLOBAL: FAIL ({global_coverage:.1%} < {target:.0%})")
        all_pass = False
    else:
        print(f"  GLOBAL: PASS ({global_coverage:.1%})")

    for book, info in book_results.items():
        if info["status"] == "FAIL":
            print(f"  {book}: FAIL (coverage {info['coverage']:.1%})")
            all_pass = False

    if all_pass:
        print(f"\n  All checks PASS. Conformal is working correctly.")
    else:
        print(f"\n  Some checks FAILED. Consider running with --refit")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Audit conformal prediction coverage")
    parser.add_argument("--min-games", type=int, default=DEFAULT_MIN_GAMES,
                        help="Minimum games per book to evaluate")
    parser.add_argument("--refit", action="store_true",
                        help="Generate per-book conformal if coverage fails")
    args = parser.parse_args()

    # Load conformal
    conf_path = NBA_ML_MODELS_DIR / "ensemble_conformal.pkl"
    if not conf_path.exists():
        logger.error("ensemble_conformal.pkl no encontrado")
        return
    conformal = joblib.load(conf_path)

    # Load predictions with results
    df = load_predictions_with_results()
    if df.empty:
        return

    # Audit global
    global_coverage, _, _ = audit_global_conformal(df, conformal)

    # Audit per-book
    book_results = audit_per_book(df, conformal, args.min_games)

    # Summary
    print_summary(global_coverage, book_results)

    # Refit if requested
    if args.refit:
        refit_per_book(df, args.min_games)


if __name__ == "__main__":
    main()
