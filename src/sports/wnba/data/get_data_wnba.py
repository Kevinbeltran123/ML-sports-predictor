"""Fetch WNBA team stats from stats.nba.com (LeagueID=10).

Saves daily box-score rows to data/training/WNBATeamData.sqlite.
Each date becomes its own SQLite table named YYYY-MM-DD, mirroring
the NBA get_data.py pattern so both feeds can share downstream tooling.

Usage:
    PYTHONPATH=. python src/sports/wnba/data/get_data_wnba.py
    PYTHONPATH=. python src/sports/wnba/data/get_data_wnba.py --backfill
    PYTHONPATH=. python src/sports/wnba/data/get_data_wnba.py --backfill --season 2024-25
"""

import argparse
import random
import sqlite3
import time
from datetime import datetime, timedelta

import pandas as pd
import toml

from src.config import CONFIG_PATH, get_logger
from src.core.tools import get_json_data, to_data_frame
from src.sports.wnba.config_paths import WNBA_TEAMS_DB

logger = get_logger(__name__)

WNBA_LEAGUE_ID = "10"

MIN_DELAY_SECONDS = 1
MAX_DELAY_SECONDS = 3
MAX_RETRIES = 3

# stats.nba.com endpoint — same URL template as NBA, LeagueID injected at call time.
# Format args: month, day, start_year, year, season_key, league_id
WNBA_DATA_URL = (
    "https://stats.nba.com/stats/leaguegamelog?"
    "Counter=1000&DateFrom={0:02d}%2F{1:02d}%2F{3}"
    "&DateTo={0:02d}%2F{1:02d}%2F{3}"
    "&Direction=DESC&LeagueID={5}&PlayerOrTeam=T"
    "&Season={4}&SeasonType=Regular+Season&Sorter=DATE"
)


def load_config() -> dict:
    return toml.load(CONFIG_PATH)


def iter_dates(start_date, end_date):
    date_pointer = start_date
    while date_pointer <= end_date:
        yield date_pointer
        date_pointer += timedelta(days=1)


def select_current_season(config: dict, today):
    """Return (season_key, value, start_date, end_date) for today, or Nones."""
    section = config.get("get-data-wnba", config.get("get-data", {}))
    for season_key, value in section.items():
        start_date = datetime.strptime(value["start_date"], "%Y-%m-%d").date()
        end_date = datetime.strptime(value["end_date"], "%Y-%m-%d").date()
        if start_date <= today <= end_date:
            return season_key, value, start_date, end_date
    return None, None, None, None


def get_table_dates(con: sqlite3.Connection) -> set:
    table_dates = set()
    cursor = con.execute("SELECT name FROM sqlite_master WHERE type='table'")
    for (name,) in cursor.fetchall():
        try:
            table_dates.add(datetime.strptime(name, "%Y-%m-%d").date())
        except ValueError:
            continue
    return table_dates


def fetch_data(url: str, date_pointer, start_year: int, season_key: str) -> pd.DataFrame:
    for attempt in range(1, MAX_RETRIES + 1):
        raw_data = get_json_data(
            url.format(
                date_pointer.month,
                date_pointer.day,
                start_year,
                date_pointer.year,
                season_key,
                WNBA_LEAGUE_ID,
            )
        )
        df = to_data_frame(raw_data)
        if not df.empty:
            return df
        if attempt < MAX_RETRIES:
            delay = MIN_DELAY_SECONDS + random.random() * (MAX_DELAY_SECONDS - MIN_DELAY_SECONDS)
            logger.debug("Empty response on attempt %d/%d, sleeping %.1fs", attempt, MAX_RETRIES, delay)
            time.sleep(delay)
    return pd.DataFrame()


def backfill_season(con, url, season_key, value, existing_dates, today):
    start_date = datetime.strptime(value["start_date"], "%Y-%m-%d").date()
    end_date = datetime.strptime(value["end_date"], "%Y-%m-%d").date()
    fetch_end = min(today - timedelta(days=1), end_date)

    missing_dates = [
        d for d in iter_dates(start_date, fetch_end) if d not in existing_dates
    ]

    if not missing_dates:
        logger.info("No missing dates for WNBA season %s.", season_key)
        return

    logger.info("Backfilling %d dates for WNBA season %s.", len(missing_dates), season_key)
    for date_pointer in missing_dates:
        logger.info("Fetching WNBA data: %s", date_pointer)
        df = fetch_data(url, date_pointer, value["start_year"], season_key)
        if df.empty:
            logger.warning("No WNBA data returned for: %s", date_pointer)
            continue
        table_name = date_pointer.strftime("%Y-%m-%d")
        df["Date"] = table_name
        df.to_sql(table_name, con, if_exists="replace", index=False)
        existing_dates.add(date_pointer)
        time.sleep(random.randint(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS))


def main(config=None, db_path=None, today=None, backfill=False, season=None):
    if db_path is None:
        db_path = WNBA_TEAMS_DB
    if config is None:
        config = load_config()
    if today is None:
        today = datetime.today().date()

    url = WNBA_DATA_URL

    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as con:
        existing_dates = get_table_dates(con)

        if backfill:
            section = config.get("get-data-wnba", config.get("get-data", {}))
            season_items = list(section.items())
            if season:
                season_items = [(k, v) for k, v in season_items if k == season]
                if not season_items:
                    logger.error("WNBA season not found in config: %s", season)
                    return
            for season_key, value in season_items:
                backfill_season(con, url, season_key, value, existing_dates, today)
            return

        season_key, value, start_date, end_date = select_current_season(config, today)
        if not season_key:
            logger.warning("No current WNBA season found for today: %s", today)
            return

        fetch_end = min(today, end_date)
        season_dates = [d for d in existing_dates if start_date <= d <= fetch_end]
        latest_date = max(season_dates) if season_dates else None
        fetch_start = start_date if latest_date is None else latest_date + timedelta(days=1)

        if fetch_start > fetch_end:
            logger.info("WNBA data up to date. Latest: %s", latest_date)
            return

        for date_pointer in iter_dates(fetch_start, fetch_end):
            logger.info("Fetching WNBA data: %s", date_pointer)
            df = fetch_data(url, date_pointer, value["start_year"], season_key)
            if df.empty:
                logger.warning("No WNBA data returned for: %s", date_pointer)
                continue
            table_name = date_pointer.strftime("%Y-%m-%d")
            df["Date"] = table_name
            df.to_sql(table_name, con, if_exists="replace", index=False)
            time.sleep(random.randint(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch WNBA team stats (LeagueID=10).")
    parser.add_argument("--backfill", action="store_true", help="Fetch all missing dates.")
    parser.add_argument("--season", help="Limit backfill to a single season key (e.g. 2024-25).")
    args = parser.parse_args()
    main(backfill=args.backfill, season=args.season)
