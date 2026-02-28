"""Captura closing/opening lines desde OddsAPI para CLV tracking.

Diseñado para cron:
  Opening (mañana):  0 10 * * * PYTHONPATH=. python scripts/collect_closing_lines.py --type opening
  Closing (pre-tip):  */15 18-23 * * * PYTHONPATH=. python scripts/collect_closing_lines.py --type closing
"""

import argparse
import sqlite3
from datetime import datetime

from src.sports.nba.providers.odds_api import OddsApiProvider
from src.config import HISTORICAL_LINES_DB, get_logger

logger = get_logger(__name__)


def _ensure_db():
    """Crea la tabla si no existe."""
    HISTORICAL_LINES_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(HISTORICAL_LINES_DB) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS team_odds_lines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_date TEXT NOT NULL,
                capture_timestamp TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                sportsbook TEXT NOT NULL,
                line_type TEXT NOT NULL,  -- 'opening' or 'closing'
                ml_home INTEGER,
                ml_away INTEGER,
                spread_home REAL,
                ou_total REAL,
                UNIQUE(game_date, home_team, away_team, sportsbook, line_type)
            )
        """)
        con.commit()


def collect_lines(sportsbook: str = "fanduel", line_type: str = "closing",
                   odds_data: dict | None = None):
    """Captura odds actuales y los guarda como opening o closing.

    Args:
        odds_data: Pre-fetched odds dict to reuse (avoids duplicate API call).
                   If None, fetches from API.
    """
    _ensure_db()

    odds = odds_data
    if odds is None:
        provider = OddsApiProvider(sportsbook=sportsbook)
        odds = provider.get_odds()
    if not odds:
        logger.warning("No odds disponibles")
        return 0

    today = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    saved = 0

    with sqlite3.connect(HISTORICAL_LINES_DB) as con:
        for game_key, game_data in odds.items():
            home_team, away_team = game_key.split(":")
            ml_home = game_data.get(home_team, {}).get("money_line_odds")
            ml_away = game_data.get(away_team, {}).get("money_line_odds")
            spread = game_data.get("spread")
            ou = game_data.get("under_over_odds")

            con.execute("""
                INSERT OR REPLACE INTO team_odds_lines
                (game_date, capture_timestamp, home_team, away_team, sportsbook,
                 line_type, ml_home, ml_away, spread_home, ou_total)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (today, timestamp, home_team, away_team, sportsbook,
                  line_type, ml_home, ml_away, spread, ou))
            saved += 1

        con.commit()

    logger.info("Saved %d %s lines for %s (%s)", saved, line_type, today, sportsbook)
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect opening/closing lines for CLV")
    parser.add_argument("--type", choices=["opening", "closing"], default="closing")
    parser.add_argument("--book", default="fanduel")
    args = parser.parse_args()
    collect_lines(sportsbook=args.book, line_type=args.type)
