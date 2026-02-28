"""Actualiza resultados de predicciones y muestra reporte de tracking.

Busca todas las predicciones sin resultado en BetsTracking.sqlite,
descarga scores desde nba_api, y actualiza ml_correct/ou_correct.
Despues imprime un reporte acumulado de P&L y accuracy.

Uso:
    PYTHONPATH=. python scripts/update_tracking.py
    PYTHONPATH=. python scripts/update_tracking.py --report     # solo reporte, no actualizar
    PYTHONPATH=. python scripts/update_tracking.py --date 2026-02-20  # actualizar fecha especifica
    PYTHONPATH=. python scripts/update_tracking.py --backfill   # actualizar TODAS las fechas pendientes
"""

import argparse
import sqlite3
from datetime import datetime

from src.config import BETS_DB, get_logger
from src.core.betting.bet_tracker import BetTracker

logger = get_logger(__name__)


def get_pending_dates():
    """Retorna fechas con predicciones sin resultado."""
    if not BETS_DB.exists():
        return []
    with sqlite3.connect(BETS_DB) as con:
        rows = con.execute("""
            SELECT DISTINCT game_date FROM predictions
            WHERE ml_correct IS NULL
            ORDER BY game_date
        """).fetchall()
    return [r[0] for r in rows]


def get_stats():
    """Calcula estadisticas globales del tracking."""
    if not BETS_DB.exists():
        return None
    with sqlite3.connect(BETS_DB) as con:
        total = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        with_result = con.execute(
            "SELECT COUNT(*) FROM predictions WHERE ml_correct IS NOT NULL"
        ).fetchone()[0]
        pending = total - with_result

        if with_result == 0:
            return {"total": total, "with_result": 0, "pending": pending}

        # Accuracy global
        ml_correct = con.execute(
            "SELECT SUM(ml_correct) FROM predictions WHERE ml_correct IS NOT NULL"
        ).fetchone()[0] or 0

        # Por conformal set_size
        bet_rows = con.execute("""
            SELECT COUNT(*), SUM(ml_correct)
            FROM predictions
            WHERE ml_correct IS NOT NULL AND conformal_set_size = 1
        """).fetchone()
        skip_rows = con.execute("""
            SELECT COUNT(*), SUM(ml_correct)
            FROM predictions
            WHERE ml_correct IS NOT NULL AND conformal_set_size = 2
        """).fetchone()

        # Fechas
        dates = con.execute("""
            SELECT MIN(game_date), MAX(game_date)
            FROM predictions WHERE ml_correct IS NOT NULL
        """).fetchone()

        # Por sportsbook
        books = con.execute("""
            SELECT sportsbook, COUNT(*), SUM(ml_correct),
                   SUM(CASE WHEN conformal_set_size = 1 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN conformal_set_size = 1 AND ml_correct = 1 THEN 1 ELSE 0 END)
            FROM predictions
            WHERE ml_correct IS NOT NULL
            GROUP BY sportsbook
        """).fetchall()

        # Simulated P&L (flat $100 bets on BET picks)
        pnl_rows = con.execute("""
            SELECT prob_home, prob_away, ml_home_odds, ml_away_odds,
                   ml_correct, actual_winner, conformal_set_size
            FROM predictions
            WHERE ml_correct IS NOT NULL AND conformal_set_size = 1
        """).fetchall()

        pnl = 0.0
        n_bets = 0
        for ph, pa, ho, ao, correct, winner, cs in pnl_rows:
            pick_home = ph >= 0.5
            odds = ho if pick_home else ao
            if correct:
                if odds > 0:
                    pnl += 100 * (odds / 100)
                else:
                    pnl += 100 * (100 / abs(odds))
            else:
                pnl -= 100
            n_bets += 1

        return {
            "total": total,
            "with_result": with_result,
            "pending": pending,
            "ml_correct": ml_correct,
            "ml_accuracy": ml_correct / with_result if with_result else 0,
            "bet_n": bet_rows[0] or 0,
            "bet_correct": bet_rows[1] or 0,
            "bet_accuracy": (bet_rows[1] or 0) / bet_rows[0] if bet_rows[0] else 0,
            "skip_n": skip_rows[0] or 0,
            "skip_correct": skip_rows[1] or 0,
            "skip_accuracy": (skip_rows[1] or 0) / skip_rows[0] if skip_rows[0] else 0,
            "date_from": dates[0],
            "date_to": dates[1],
            "books": books,
            "pnl": pnl,
            "n_bets": n_bets,
        }


def print_report(stats):
    """Imprime reporte de tracking."""
    if stats is None:
        print("  No BetsTracking.sqlite found")
        return

    print(f"\n{'='*60}")
    print(f"  PREDICTION TRACKING REPORT")
    print(f"{'='*60}")
    print(f"  Total predictions:  {stats['total']}")
    print(f"  With results:       {stats['with_result']}")
    print(f"  Pending:            {stats['pending']}")

    if stats["with_result"] == 0:
        print(f"\n  No results yet. Run without --report to update.")
        print(f"{'='*60}\n")
        return

    print(f"  Date range:         {stats['date_from']} → {stats['date_to']}")

    print(f"\n  --- Accuracy ---")
    print(f"  Overall:  {stats['ml_correct']}/{stats['with_result']} "
          f"({stats['ml_accuracy']:.1%})")

    if stats["bet_n"] > 0:
        print(f"  BET (cs=1): {stats['bet_correct']}/{stats['bet_n']} "
              f"({stats['bet_accuracy']:.1%})")
    if stats["skip_n"] > 0:
        print(f"  SKIP (cs=2): {stats['skip_correct']}/{stats['skip_n']} "
              f"({stats['skip_accuracy']:.1%})")

    print(f"\n  --- Simulated P&L (flat $100 on BET picks) ---")
    print(f"  Bets placed:  {stats['n_bets']}")
    print(f"  Net P&L:      ${stats['pnl']:+.2f}")
    if stats["n_bets"] > 0:
        roi = stats["pnl"] / (stats["n_bets"] * 100) * 100
        print(f"  ROI:          {roi:+.1f}%")

    if stats["books"]:
        print(f"\n  --- Per Sportsbook ---")
        for book, n, correct, n_bet, bet_correct in stats["books"]:
            acc = correct / n if n else 0
            bet_acc = bet_correct / n_bet if n_bet else 0
            print(f"  {book}: {correct}/{n} ({acc:.1%}) | BET: {bet_correct}/{n_bet} ({bet_acc:.1%})")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Update prediction tracking results")
    parser.add_argument("--report", action="store_true", help="Show report only, don't update")
    parser.add_argument("--date", help="Update specific date (YYYY-MM-DD)")
    parser.add_argument("--backfill", action="store_true", help="Update ALL pending dates")
    args = parser.parse_args()

    tracker = BetTracker()

    if not args.report:
        if args.date:
            dates = [args.date]
        elif args.backfill:
            dates = get_pending_dates()
        else:
            # Default: update dates from last 7 days that have pending results
            today = datetime.now()
            all_pending = get_pending_dates()
            # Filter to recent dates (within 7 days is a reasonable window)
            dates = [d for d in all_pending
                     if (today - datetime.strptime(d, "%Y-%m-%d")).days <= 7]
            if not dates and all_pending:
                print(f"  No recent pending dates. Use --backfill for older dates.")
                print(f"  Pending dates: {', '.join(all_pending[:5])}{'...' if len(all_pending) > 5 else ''}")

        if dates:
            print(f"\n  Updating results for {len(dates)} date(s)...")
            for d in dates:
                tracker.update_results(d)

    # Always show report
    stats = get_stats()
    print_report(stats)


if __name__ == "__main__":
    main()
