"""Live betting CSV tracker — append-only log for post-quarter adjustments.

Persists every live adjustment so we can later analyze:
  - Is beta=0.45 well-calibrated? (when model says 70% post-Q1, does home win ~70%?)
  - ROI of live bets vs pre-game bets
  - Which quarters give the best edge signal

CSV location: data/nba/predictions/live_bets.csv
"""

import csv
from datetime import datetime
from pathlib import Path

from src.config import PREDICTIONS_DIR, get_logger

logger = get_logger(__name__)

LIVE_CSV = PREDICTIONS_DIR / "live_bets.csv"

COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "quarter",
    "home_score",
    "away_score",
    "score_diff",
    "p_pregame",
    "p_adjusted",
    "delta",
    "conf_set_size",
    "method",
    "ml_odds",
    "kelly_pct",
    "result",
    "pnl",
]


def _ensure_csv():
    """Creates CSV with header if it doesn't exist."""
    LIVE_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not LIVE_CSV.exists():
        with open(LIVE_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
        logger.info("Live tracker CSV created: %s", LIVE_CSV)


def log_adjustment(
    home_team: str,
    away_team: str,
    quarter: int,
    home_score: int,
    away_score: int,
    p_pregame: float,
    p_adjusted: float,
    conf_set_size: int = 0,
    method: str = "simple",
    ml_odds: int | None = None,
    kelly_pct: float | None = None,
):
    """Append a live adjustment row to the CSV.

    result and pnl are left empty — filled later by update_results().
    """
    _ensure_csv()

    row = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "home_team": home_team,
        "away_team": away_team,
        "quarter": f"Q{quarter}",
        "home_score": home_score,
        "away_score": away_score,
        "score_diff": home_score - away_score,
        "p_pregame": round(p_pregame, 4),
        "p_adjusted": round(p_adjusted, 4),
        "delta": round(p_adjusted - p_pregame, 4),
        "conf_set_size": conf_set_size,
        "method": method,
        "ml_odds": ml_odds or "",
        "kelly_pct": round(kelly_pct, 2) if kelly_pct else "",
        "result": "",
        "pnl": "",
    }

    with open(LIVE_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writerow(row)

    logger.debug("Live tracker: %s vs %s Q%d logged", home_team, away_team, quarter)


def update_results(date: str = None):
    """Fill in result/pnl for rows matching a date.

    Reads final scores from BetsTracking.sqlite or manual input.
    Call this the day after to close out pending rows.

    Args:
        date: YYYY-MM-DD to update. Defaults to today.
    """
    import pandas as pd

    if not LIVE_CSV.exists():
        logger.warning("No live_bets.csv found")
        return

    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    df = pd.read_csv(LIVE_CSV)
    pending = df[(df["date"] == date) & (df["result"] == "")]

    if pending.empty:
        logger.info("No pending results for %s", date)
        return

    # Try to get results from BetsTracking
    try:
        import sqlite3
        from src.config import BETS_DB

        with sqlite3.connect(BETS_DB) as con:
            results = pd.read_sql(
                "SELECT home_team, away_team, result FROM predictions WHERE date = ?",
                con, params=[date],
            )

        if results.empty:
            logger.info("No results in BetsTracking for %s", date)
            return

        result_map = {}
        for _, r in results.iterrows():
            result_map[(r["home_team"], r["away_team"])] = r["result"]

        updated = 0
        for idx, row in df.iterrows():
            if row["date"] != date or row["result"] != "":
                continue
            key = (row["home_team"], row["away_team"])
            if key in result_map:
                winner = result_map[key]
                df.at[idx, "result"] = winner

                # Compute P&L if odds are available and conf_set_size == 1
                odds = row.get("ml_odds")
                if odds and str(odds).strip() and row.get("conf_set_size") == 1:
                    try:
                        odds = int(float(odds))
                        pick_home = float(row["p_adjusted"]) >= 0.5
                        home_won = winner == row["home_team"]
                        bet_won = (pick_home and home_won) or (not pick_home and not home_won)
                        if bet_won:
                            pnl = 100 * (odds / 100) if odds > 0 else 100 * (100 / abs(odds))
                        else:
                            pnl = -100
                        df.at[idx, "pnl"] = round(pnl, 2)
                    except (ValueError, TypeError):
                        pass

                updated += 1

        if updated:
            df.to_csv(LIVE_CSV, index=False)
            logger.info("Updated %d live bet results for %s", updated, date)

    except Exception as e:
        logger.warning("Could not auto-fill results: %s", e)
