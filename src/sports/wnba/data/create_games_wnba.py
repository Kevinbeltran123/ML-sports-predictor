"""Build the WNBA training dataset from WNBATeamData.sqlite.

Feature set v1 — core features only (BRef-dependent features omitted):
  - Base stats, advanced, differential, style
  - Rolling averages, Elo/SRS, SOS
  - Fatigue, travel, conference/division
  - Market features (if odds available)

Skipped (no WNBA BRef data):
  zone_shooting, shot_chart, onoff, line_scores,
  player_advanced, espn_lines, lineup_composition

Usage:
    PYTHONPATH=. python src/sports/wnba/data/create_games_wnba.py
    PYTHONPATH=. python src/sports/wnba/data/create_games_wnba.py --season 2024-25
"""

import argparse
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import get_logger
from src.sports.wnba.config_paths import WNBA_TEAMS_DB, WNBA_DATASET_DB
from src.sports.wnba.config_wnba import (
    WNBA_ELO_K,
    WNBA_ELO_BASE,
    WNBA_HOME_ADVANTAGE,
    WNBA_DIVISION,
    team_index_wnba,
)
from src.core.stats.elo_ratings import (
    add_elo_features_to_frame,
    build_elo_history,
    build_srs_history,
    add_srs_features_to_frame,
)
from src.core.stats.rolling_averages import add_rolling_features_to_frame
from src.sports.nba.features.advanced_stats import add_advanced_features
from src.sports.nba.features.differential_features import add_differential_features
from src.sports.nba.features.style_features import add_style_features
from src.sports.nba.features.fatigue import (
    add_fatigue_to_frame,
    add_travel_to_frame,
    add_extended_fatigue_to_frame,
    add_fatigue_combo_to_frame,
)
from src.sports.nba.features.sos import add_sos_to_frame

logger = get_logger(__name__)

OUTPUT_TABLE_PREFIX = "wnba_dataset"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_team_data(db_path, season_filter=None) -> pd.DataFrame:
    """Load all daily tables from WNBATeamData.sqlite into one DataFrame."""
    frames = []
    with sqlite3.connect(db_path) as con:
        cursor = con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        for tbl in sorted(tables):
            try:
                date = datetime.strptime(tbl, "%Y-%m-%d").date()
            except ValueError:
                continue
            if season_filter:
                # season_filter is a (start_date, end_date) tuple of date objects
                if not (season_filter[0] <= date <= season_filter[1]):
                    continue
            df = pd.read_sql_query(f'SELECT * FROM "{tbl}"', con)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def add_conference_division_wnba(df: pd.DataFrame) -> pd.DataFrame:
    """Add CONFERENCE and DIVISION columns using the WNBA mapping."""
    df = df.copy()
    for suffix in ("", ".1"):
        col = f"TEAM_NAME{suffix}"
        if col in df.columns:
            div_col = f"DIVISION{suffix}"
            conf_col = f"CONFERENCE{suffix}"
            df[div_col] = df[col].map(WNBA_DIVISION).fillna("Unknown")
            df[conf_col] = df[div_col]  # single conference; East/West = division
    return df


def add_same_division_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add SAME_DIVISION flag (1 if both teams in same WNBA division)."""
    df = df.copy()
    if "DIVISION" in df.columns and "DIVISION.1" in df.columns:
        df["SAME_DIVISION"] = (df["DIVISION"] == df["DIVISION.1"]).astype(int)
    return df


def pivot_to_game_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert per-team rows (2 rows per game) into single game rows
    with HOME and AWAY columns suffixed .1 for the away team.

    Assumes stats.nba.com MATCHUP column uses ' vs. ' for home and ' @ ' for away.
    """
    if "MATCHUP" not in df.columns:
        logger.warning("MATCHUP column missing — cannot pivot game rows.")
        return df

    home_mask = df["MATCHUP"].str.contains(" vs\\. ", na=False)
    home_df = df[home_mask].copy()
    away_df = df[~home_mask].copy()

    away_rename = {col: f"{col}.1" for col in away_df.columns if col != "GAME_ID"}
    away_df = away_df.rename(columns=away_rename)

    merged = home_df.merge(away_df, on="GAME_ID", suffixes=("", "_dup"))
    dup_cols = [c for c in merged.columns if c.endswith("_dup")]
    merged.drop(columns=dup_cols, inplace=True)

    if "WL" in merged.columns:
        merged["Home-Team-Win"] = (merged["WL"] == "W").astype(int)

    return merged


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #

def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all v1 feature modules to the pivoted game DataFrame."""
    if df.empty:
        return df

    logger.info("Raw game rows: %d", len(df))

    # Core stat features
    try:
        df = add_advanced_features(df)
    except Exception as exc:
        logger.warning("add_advanced_features skipped: %s", exc)

    try:
        df = add_differential_features(df)
    except Exception as exc:
        logger.warning("add_differential_features skipped: %s", exc)

    try:
        df = add_style_features(df)
    except Exception as exc:
        logger.warning("add_style_features skipped: %s", exc)

    # Rolling averages
    try:
        df = add_rolling_features_to_frame(df)
    except Exception as exc:
        logger.warning("add_rolling_features_to_frame skipped: %s", exc)

    # Elo / SRS
    try:
        elo_history = build_elo_history(df, k=WNBA_ELO_K, base=WNBA_ELO_BASE,
                                        home_adv=WNBA_HOME_ADVANTAGE)
        df = add_elo_features_to_frame(df, elo_history)
    except Exception as exc:
        logger.warning("Elo features skipped: %s", exc)

    try:
        srs_history = build_srs_history(df)
        df = add_srs_features_to_frame(df, srs_history)
    except Exception as exc:
        logger.warning("SRS features skipped: %s", exc)

    # SOS
    try:
        df = add_sos_to_frame(df)
    except Exception as exc:
        logger.warning("add_sos_to_frame skipped: %s", exc)

    # Fatigue / travel
    try:
        df = add_fatigue_to_frame(df)
        df = add_travel_to_frame(df)
        df = add_extended_fatigue_to_frame(df)
        df = add_fatigue_combo_to_frame(df)
    except Exception as exc:
        logger.warning("Fatigue/travel features skipped: %s", exc)

    # Conference / division (WNBA-specific)
    df = add_conference_division_wnba(df)
    df = add_same_division_flag(df)

    logger.info("Dataset built: %d rows, %d columns", len(df), len(df.columns))
    return df


def main(season=None, teams_db=None, output_db=None):
    if teams_db is None:
        teams_db = WNBA_TEAMS_DB
    if output_db is None:
        output_db = WNBA_DATASET_DB

    season_filter = None
    table_name = OUTPUT_TABLE_PREFIX
    if season:
        # season format: "2024-25"
        try:
            start_year = int(season.split("-")[0])
            start_dt = datetime(start_year, 5, 1).date()   # WNBA starts ~May
            end_dt = datetime(start_year, 10, 31).date()   # ends ~October
            season_filter = (start_dt, end_dt)
            table_name = f"{OUTPUT_TABLE_PREFIX}_{season}"
        except (ValueError, IndexError):
            logger.error("Invalid season format: %s (expected e.g. 2024-25)", season)
            return

    logger.info("Loading WNBA team data from %s", teams_db)
    raw = load_team_data(teams_db, season_filter=season_filter)

    if raw.empty:
        logger.error("No data loaded from %s", teams_db)
        return

    logger.info("Pivoting to game rows...")
    games = pivot_to_game_rows(raw)

    if games.empty:
        logger.error("Pivot produced no rows — check MATCHUP column format.")
        return

    logger.info("Building features...")
    dataset = build_dataset(games)

    if dataset.empty:
        logger.error("Feature build produced empty dataset.")
        return

    output_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(output_db) as con:
        dataset.to_sql(table_name, con, if_exists="replace", index=False)

    logger.info("Saved %d rows to %s::%s", len(dataset), output_db, table_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build WNBA training dataset.")
    parser.add_argument("--season", help="Season to process (e.g. 2024-25). Omit for all.")
    args = parser.parse_args()
    main(season=args.season)
