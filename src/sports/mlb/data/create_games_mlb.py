"""Build the MLB training dataset from MLBTeamData.sqlite + MLBPitcherData.sqlite.

Assembles all features into mlb_dataset.sqlite with targets:
  - Home-Team-Win (binary, ML moneyline)
  - F5-Home-Win (binary, first 5 innings)
  - Total_Runs (integer, O/U totals)

Usage:
    PYTHONPATH=. python src/sports/mlb/data/create_games_mlb.py
    PYTHONPATH=. python src/sports/mlb/data/create_games_mlb.py --seasons 2024 2025
"""

import argparse
import json
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import get_logger
from src.sports.mlb.config_paths import (
    MLB_TEAMS_DB, MLB_PITCHER_DB, MLB_PARK_DB,
    MLB_DATASET_DB, MLB_TRAINING_DIR,
)
from src.sports.mlb.config_mlb import (
    MLB_ELO_K, MLB_ELO_BASE, MLB_HOME_ADVANTAGE, MLB_SEASON_CARRY,
    MLB_DIVISION, MLB_LEAGUE, MLB_TEAM_ALIASES,
    team_index_mlb,
)
from src.core.stats.elo_ratings import (
    add_elo_features_to_frame,
    build_elo_history,
    build_srs_history,
    add_srs_features_to_frame,
)

logger = get_logger(__name__)
OUTPUT_TABLE_PREFIX = "mlb_dataset"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _normalize_team(name: str) -> str:
    return MLB_TEAM_ALIASES.get(name, name)


def load_team_data(db_path, seasons: list[int] = None) -> pd.DataFrame:
    """Load game logs from MLBTeamData.sqlite into one DataFrame.

    Tables are named 'season_YYYY' with 2 rows per game (home + away perspective).
    """
    frames = []
    with sqlite3.connect(db_path) as con:
        cursor = con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        for tbl in sorted(tables):
            if not tbl.startswith("season_"):
                continue
            year = int(tbl.split("_")[1])
            if seasons and year not in seasons:
                continue
            df = pd.read_sql_query(f'SELECT * FROM "{tbl}"', con)
            frames.append(df)
            logger.info("Loaded %s: %d rows", tbl, len(df))
    if not frames:
        logger.warning("No data loaded from %s", db_path)
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def pivot_to_game_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 2-rows-per-game to 1-row-per-game (home team perspective).

    Home team columns have no suffix. Away team columns get '.1' suffix.
    This mirrors the NBA dataset format used by all core feature modules.
    """
    home = df[df["HOME_AWAY"] == "home"].copy()
    away = df[df["HOME_AWAY"] == "away"].copy()

    if home.empty or away.empty:
        return pd.DataFrame()

    # Rename away columns with .1 suffix
    away_cols = {}
    skip = {"GAME_PK", "GAME_DATE", "SEASON"}
    for col in away.columns:
        if col not in skip:
            away_cols[col] = f"{col}.1"
    away = away.rename(columns=away_cols)

    # Merge on game_pk
    merged = home.merge(away, on=["GAME_PK", "GAME_DATE", "SEASON"], how="inner")
    logger.info("Pivoted to %d game rows", len(merged))
    return merged


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add target columns for ML, F5, and totals models."""
    df = df.copy()

    # ML target: home team wins
    df["Home-Team-Win"] = df["WIN"].astype(int)

    # Totals target: combined runs
    runs_home = df["RUNS"].fillna(0).astype(int)
    runs_away = df["RUNS.1"].fillna(0).astype(int)
    df["Total_Runs"] = runs_home + runs_away

    # F5 target: home team leads after 5 innings
    def _f5_home_win(row):
        try:
            home_innings = json.loads(row.get("INNING_RUNS", "[]"))
            away_innings = json.loads(row.get("INNING_RUNS.1", "[]"))
            home_f5 = sum(home_innings[:5]) if len(home_innings) >= 5 else None
            away_f5 = sum(away_innings[:5]) if len(away_innings) >= 5 else None
            if home_f5 is None or away_f5 is None:
                return np.nan
            return 1 if home_f5 > away_f5 else (0 if away_f5 > home_f5 else np.nan)
        except (json.JSONDecodeError, TypeError):
            return np.nan

    df["F5-Home-Win"] = df.apply(_f5_home_win, axis=1)
    return df


def add_division_league(df: pd.DataFrame) -> pd.DataFrame:
    """Add division, league, and inter-league flags."""
    df = df.copy()

    df["DIVISION_HOME"] = df["TEAM_NAME"].map(MLB_DIVISION).fillna("Unknown")
    df["DIVISION_AWAY"] = df["TEAM_NAME.1"].map(MLB_DIVISION).fillna("Unknown")
    df["LEAGUE_HOME"] = df["TEAM_NAME"].map(MLB_LEAGUE).fillna("Unknown")
    df["LEAGUE_AWAY"] = df["TEAM_NAME.1"].map(MLB_LEAGUE).fillna("Unknown")

    df["SAME_DIVISION"] = (df["DIVISION_HOME"] == df["DIVISION_AWAY"]).astype(int)
    df["SAME_LEAGUE"] = (df["LEAGUE_HOME"] == df["LEAGUE_AWAY"]).astype(int)
    df["IS_INTERLEAGUE"] = (1 - df["SAME_LEAGUE"]).astype(int)
    df["IS_AL_HOME"] = (df["LEAGUE_HOME"] == "AL").astype(int)

    return df


def add_team_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Add integer team indices for model input."""
    df = df.copy()
    df["Home-Team-Index"] = df["TEAM_NAME"].map(team_index_mlb).fillna(-1).astype(int)
    df["Away-Team-Index"] = df["TEAM_NAME.1"].map(team_index_mlb).fillna(-1).astype(int)
    return df


def add_bubble_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Flag 2020 COVID shortened season."""
    df = df.copy()
    df["IS_BUBBLE_SEASON"] = (df["SEASON"] == 2020).astype(int)
    return df


# --------------------------------------------------------------------------- #
# Main build pipeline
# --------------------------------------------------------------------------- #

def build_dataset(seasons: list[int] = None) -> pd.DataFrame:
    """Build complete MLB training dataset."""
    if not MLB_TEAMS_DB.exists():
        logger.error("MLBTeamData.sqlite not found at %s. Run collect_mlb_data.py first.", MLB_TEAMS_DB)
        return pd.DataFrame()

    # 1. Load raw game logs
    raw = load_team_data(MLB_TEAMS_DB, seasons)
    if raw.empty:
        return pd.DataFrame()

    # Normalize team names
    raw["TEAM_NAME"] = raw["TEAM_NAME"].apply(_normalize_team)
    if "OPPONENT_NAME" in raw.columns:
        raw["OPPONENT_NAME"] = raw["OPPONENT_NAME"].apply(_normalize_team)

    # 2. Pivot to game rows (1 row per game, home perspective)
    df = pivot_to_game_rows(raw)
    if df.empty:
        return pd.DataFrame()

    # Sort by date for temporal features
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    # 3. Add targets
    df = add_targets(df)

    # 4. Add team indices
    df = add_team_indices(df)

    # 5. Add division/league flags
    df = add_division_league(df)

    # 6. Add bubble flag
    df = add_bubble_flag(df)

    # 7. Add Elo features (reused from src/core/stats)
    try:
        elo_history = build_elo_history(
            df, k=MLB_ELO_K, base=MLB_ELO_BASE,
            home_advantage=MLB_HOME_ADVANTAGE,
            season_carry=MLB_SEASON_CARRY,
        )
        df = add_elo_features_to_frame(df, elo_history)
        logger.info("Added Elo features")
    except Exception as e:
        logger.warning("Could not add Elo features: %s", e)

    # 8. Add SRS features
    try:
        srs_history = build_srs_history(df)
        df = add_srs_features_to_frame(df, srs_history)
        logger.info("Added SRS features")
    except Exception as e:
        logger.warning("Could not add SRS features: %s", e)

    # 9. Add pitcher features
    try:
        from src.sports.mlb.features.pitcher_features import (
            build_pitcher_lookup, add_sp_features_to_frame,
        )
        pitcher_lookup = build_pitcher_lookup(MLB_PITCHER_DB, MLB_TEAMS_DB)
        df = add_sp_features_to_frame(df, pitcher_lookup)
        logger.info("Added pitcher features")
    except Exception as e:
        logger.warning("Could not add pitcher features: %s", e)

    # 10. Add batting features
    try:
        from src.sports.mlb.features.batting_features import (
            build_batting_lookup, add_batting_features_to_frame,
        )
        batting_lookup = build_batting_lookup(MLB_TEAMS_DB)
        df = add_batting_features_to_frame(df, batting_lookup)
        logger.info("Added batting features")
    except Exception as e:
        logger.warning("Could not add batting features: %s", e)

    # 11. Add bullpen features
    try:
        from src.sports.mlb.features.bullpen_features import (
            build_bullpen_lookup, add_bullpen_features_to_frame,
        )
        bullpen_lookup = build_bullpen_lookup(MLB_TEAMS_DB)
        df = add_bullpen_features_to_frame(df, bullpen_lookup)
        logger.info("Added bullpen features")
    except Exception as e:
        logger.warning("Could not add bullpen features: %s", e)

    # 12. Add park + weather features
    try:
        from src.sports.mlb.features.park_weather import (
            build_park_lookup, add_park_weather_to_frame,
        )
        park_lookup = build_park_lookup(MLB_PARK_DB)
        df = add_park_weather_to_frame(df, park_lookup)
        logger.info("Added park/weather features")
    except Exception as e:
        logger.warning("Could not add park/weather features: %s", e)

    # 13. Add fatigue/travel features
    try:
        from src.sports.mlb.features.fatigue_travel import (
            build_mlb_schedule, add_mlb_fatigue_to_frame,
        )
        schedule = build_mlb_schedule(MLB_TEAMS_DB)
        df = add_mlb_fatigue_to_frame(df, schedule)
        logger.info("Added fatigue/travel features")
    except Exception as e:
        logger.warning("Could not add fatigue features: %s", e)

    # 14. Add SOS features
    try:
        from src.sports.mlb.features.sos import (
            build_win_pct_lookup, add_sos_to_frame,
        )
        win_pct = build_win_pct_lookup(MLB_TEAMS_DB)
        schedule_for_sos = build_mlb_schedule(MLB_TEAMS_DB) if "schedule" not in dir() else schedule
        df = add_sos_to_frame(df, win_pct, schedule_for_sos)
        logger.info("Added SOS features")
    except Exception as e:
        logger.warning("Could not add SOS features: %s", e)

    # 15. Add differential features
    try:
        from src.sports.mlb.features.differential_features import add_differential_features
        df = add_differential_features(df)
        logger.info("Added differential features")
    except Exception as e:
        logger.warning("Could not add differential features: %s", e)

    # 16. Add odds features (if historical odds available)
    try:
        from src.sports.mlb.features.odds_features import add_odds_features_to_frame
        df = add_odds_features_to_frame(df)
        logger.info("Added odds features")
    except Exception as e:
        logger.warning("Could not add odds features: %s", e)

    # Log final shape
    logger.info("Dataset shape: %s, columns: %d", df.shape, len(df.columns))
    logger.info("Targets: Home-Team-Win mean=%.3f, Total_Runs mean=%.1f",
                df["Home-Team-Win"].mean(), df["Total_Runs"].mean())

    f5_valid = df["F5-Home-Win"].dropna()
    if len(f5_valid) > 0:
        logger.info("F5 target: %d valid rows, mean=%.3f", len(f5_valid), f5_valid.mean())

    return df


def save_dataset(df: pd.DataFrame, table_name: str = None):
    """Save dataset to mlb_dataset.sqlite."""
    MLB_TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    if table_name is None:
        years = sorted(df["SEASON"].unique())
        table_name = f"{OUTPUT_TABLE_PREFIX}_{min(years)}-{str(max(years))[-2:]}"

    with sqlite3.connect(MLB_DATASET_DB) as con:
        df.to_sql(table_name, con, if_exists="replace", index=False)
        logger.info("Saved %d rows to %s:%s", len(df), MLB_DATASET_DB.name, table_name)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Build MLB training dataset")
    parser.add_argument("--seasons", nargs="+", type=int, default=None,
                        help="Specific seasons to include (default: all)")
    parser.add_argument("--table", type=str, default=None,
                        help="Override output table name")
    args = parser.parse_args()

    df = build_dataset(args.seasons)
    if df.empty:
        logger.error("Empty dataset — nothing to save.")
        return

    save_dataset(df, args.table)
    logger.info("Done. Final shape: %s", df.shape)


if __name__ == "__main__":
    main()
