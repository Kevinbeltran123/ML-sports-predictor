"""Strength of Schedule (SOS) features for MLB W/L predictor.

SOS captures the quality of recent and season-long opponents.  A team
beating weak opponents looks very different from one posting the same
record against playoff-caliber teams.

6 features total:
    SOS_HOME_L30      -- avg opponent win% over last 30 games (home team)
    SOS_AWAY_L30      -- avg opponent win% over last 30 games (away team)
    SOS_HOME_SEASON   -- avg opponent win% season-to-date (home team)
    SOS_AWAY_SEASON   -- avg opponent win% season-to-date (away team)
    DIFF_SOS_L30      -- SOS_HOME_L30 - SOS_AWAY_L30
    DIFF_SOS_SEASON   -- SOS_HOME_SEASON - SOS_AWAY_SEASON

Usage (training):
    win_pct = build_win_pct_lookup(MLB_TEAMS_DB)
    schedule = build_mlb_schedule(MLB_TEAMS_DB)   # from fatigue_travel
    df = add_sos_to_frame(df, win_pct, schedule)

Usage (live):
    feats = get_sos_features(win_pct, schedule, team_name, game_date)
"""

import sqlite3
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import get_logger
from src.sports.mlb.config_paths import MLB_TEAMS_DB

logger = get_logger(__name__)

_SOS_WINDOW_L30 = 30       # last 30 opponents
_MIN_GAMES = 5             # min games for meaningful win%
_DEFAULT_SOS = 0.500       # league average fallback


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _parse_date(date_val) -> Optional[datetime.date]:
    """Parse various date formats to datetime.date."""
    if isinstance(date_val, str):
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
            try:
                return datetime.strptime(date_val[:10], fmt).date()
            except ValueError:
                continue
        return None
    if hasattr(date_val, "date"):
        return date_val.date() if callable(getattr(date_val, "date")) else date_val
    return None


def _safe_mean(values: List[float]) -> float:
    """Return mean of values or _DEFAULT_SOS if empty."""
    if not values:
        return _DEFAULT_SOS
    return round(float(np.mean(values)), 4)


# ------------------------------------------------------------------
# build_win_pct_lookup
# ------------------------------------------------------------------

def build_win_pct_lookup(db_path=None) -> Dict:
    """Load team data from MLBTeamData.sqlite and compute rolling win%
    per team per season.

    Processes games chronologically within each season, maintaining a
    running W/L record per team.  Win percentages are computed BEFORE
    each game result is added (no leakage).

    Returns:
        {
            (season, team): {
                "dates": [date1, date2, ...],
                "cum_wins": [w1, w2, ...],
                "cum_losses": [l1, l2, ...],
                "opponents": [(date, opponent), ...],
            }
        }
    """
    if db_path is None:
        db_path = MLB_TEAMS_DB

    try:
        con = sqlite3.connect(str(db_path))
    except Exception as exc:
        logger.warning("Cannot open teams DB %s: %s", db_path, exc)
        return {}

    all_games: List[Tuple] = []

    try:
        cursor = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            if "team" not in table.lower() and "game" not in table.lower():
                continue

            try:
                col_info = con.execute(f'PRAGMA table_info("{table}")').fetchall()
                col_names = [c[1].upper() for c in col_info]
                rows = con.execute(f'SELECT * FROM "{table}"').fetchall()
            except Exception:
                continue

            def _col(name: str) -> Optional[int]:
                try:
                    return col_names.index(name.upper())
                except ValueError:
                    return None

            idx_date = _col("GAME_DATE") or _col("DATE")
            idx_home = _col("HOME_TEAM") or _col("HOME") or _col("TEAM_HOME")
            idx_away = _col("AWAY_TEAM") or _col("AWAY") or _col("TEAM_AWAY")
            idx_home_win = (
                _col("HOME_WIN") or _col("HOME_TEAM_WIN") or _col("WIN")
                or _col("RESULT")
            )
            idx_season = _col("SEASON")

            if idx_date is None or idx_home is None or idx_away is None:
                continue

            for row in rows:
                game_date = _parse_date(row[idx_date])
                if game_date is None:
                    continue

                home = str(row[idx_home]).strip() if row[idx_home] else None
                away = str(row[idx_away]).strip() if row[idx_away] else None
                if not home or not away:
                    continue

                home_win = None
                if idx_home_win is not None:
                    try:
                        hw_val = row[idx_home_win]
                        if isinstance(hw_val, (int, float)):
                            home_win = int(float(hw_val))
                        elif isinstance(hw_val, str):
                            hw_lower = hw_val.lower()
                            if hw_lower in ("1", "true", "win", "w", "home"):
                                home_win = 1
                            elif hw_lower in ("0", "false", "loss", "l", "away"):
                                home_win = 0
                    except (TypeError, ValueError):
                        pass

                season = (
                    str(row[idx_season]).strip()
                    if idx_season is not None
                    else str(game_date.year)
                )
                all_games.append((game_date, season, home, away, home_win))

            if all_games:
                break  # found a suitable table

    finally:
        con.close()

    if not all_games:
        logger.warning("build_win_pct_lookup: no game data found in %s", db_path)
        return {}

    # Sort chronologically
    all_games.sort(key=lambda x: x[0])

    # Build per-team cumulative records keyed by (season, team)
    lookup: Dict[Tuple[str, str], Dict] = {}
    current_season = None
    team_wins: Dict[str, int] = defaultdict(int)
    team_losses: Dict[str, int] = defaultdict(int)
    team_opponents: Dict[str, List[Tuple]] = defaultdict(list)

    for game_date, season, home, away, home_win in all_games:
        # Reset on season boundary
        if season != current_season:
            current_season = season
            team_wins.clear()
            team_losses.clear()
            team_opponents.clear()

        # Snapshot BEFORE updating (no leakage)
        for team in (home, away):
            key = (season, team)
            if key not in lookup:
                lookup[key] = {
                    "dates": [],
                    "cum_wins": [],
                    "cum_losses": [],
                    "opponents": [],
                }

        # Record current cumulative state for both teams
        for team, opp in ((home, away), (away, home)):
            key = (season, team)
            lookup[key]["dates"].append(game_date)
            lookup[key]["cum_wins"].append(team_wins[team])
            lookup[key]["cum_losses"].append(team_losses[team])
            lookup[key]["opponents"].append((game_date, opp))

        # Update W/L after snapshot
        if home_win is not None:
            if home_win == 1:
                team_wins[home] += 1
                team_losses[away] += 1
            else:
                team_losses[home] += 1
                team_wins[away] += 1

    logger.info("SOS win_pct lookup built: %d team-seasons", len(lookup))
    return lookup


def _team_win_pct_at_date(
    lookup: Dict,
    season: str,
    team: str,
    before_date,
) -> Optional[float]:
    """Return cumulative win% for a team up to (not including) before_date.

    Searches the lookup for the last snapshot before before_date.
    Returns None if fewer than _MIN_GAMES played.
    """
    key = (season, team)
    data = lookup.get(key)
    if not data or not data["dates"]:
        return None

    # Find last entry strictly before before_date
    dates = data["dates"]
    idx = None
    for i in range(len(dates) - 1, -1, -1):
        if dates[i] < before_date:
            idx = i
            break

    if idx is None:
        return None

    w = data["cum_wins"][idx]
    l = data["cum_losses"][idx]
    total = w + l
    if total < _MIN_GAMES:
        return None

    return w / total


def _compute_sos_l30(
    lookup: Dict,
    season: str,
    team: str,
    game_date,
) -> float:
    """Avg opponent win% over last 30 games for team before game_date."""
    key = (season, team)
    data = lookup.get(key)
    if not data:
        return _DEFAULT_SOS

    opponents = data["opponents"]
    # Filter opponents before game_date
    prior_opps = [(d, opp) for d, opp in opponents if d < game_date]
    if len(prior_opps) < 3:
        return _DEFAULT_SOS

    recent = prior_opps[-_SOS_WINDOW_L30:]
    w_pcts = []
    for d, opp in recent:
        wp = _team_win_pct_at_date(lookup, season, opp, d)
        if wp is not None:
            w_pcts.append(wp)

    return _safe_mean(w_pcts)


def _compute_sos_season(
    lookup: Dict,
    season: str,
    team: str,
    game_date,
) -> float:
    """Avg opponent win% for ALL opponents faced this season before game_date."""
    key = (season, team)
    data = lookup.get(key)
    if not data:
        return _DEFAULT_SOS

    opponents = data["opponents"]
    prior_opps = [(d, opp) for d, opp in opponents if d < game_date]
    if len(prior_opps) < 3:
        return _DEFAULT_SOS

    w_pcts = []
    for d, opp in prior_opps:
        wp = _team_win_pct_at_date(lookup, season, opp, d)
        if wp is not None:
            w_pcts.append(wp)

    return _safe_mean(w_pcts)


# ------------------------------------------------------------------
# add_sos_to_frame
# ------------------------------------------------------------------

def add_sos_to_frame(
    df: pd.DataFrame,
    win_pct: Dict,
    schedule: Dict,
) -> pd.DataFrame:
    """Add SOS features to training DataFrame.

    Expects columns:
        TEAM_NAME   (home team)
        TEAM_NAME.1 (away team)
        GAME_DATE
        SEASON

    Adds 6 columns:
        SOS_HOME_L30, SOS_AWAY_L30,
        SOS_HOME_SEASON, SOS_AWAY_SEASON,
        DIFF_SOS_L30, DIFF_SOS_SEASON

    Args:
        df:       Training DataFrame.
        win_pct:  From build_win_pct_lookup().
        schedule: From fatigue_travel.build_mlb_schedule() (used as
                  fallback for opponent info, but primary data comes
                  from win_pct lookup).

    Returns:
        DataFrame with 6 new SOS columns appended.
    """
    # Resolve column names
    home_col = next(
        (c for c in ("TEAM_NAME", "HOME_TEAM", "TEAM_HOME", "Home") if c in df.columns),
        None,
    )
    away_col = next(
        (c for c in ("TEAM_NAME.1", "AWAY_TEAM", "TEAM_AWAY", "Away") if c in df.columns),
        None,
    )
    date_col = next(
        (c for c in ("GAME_DATE", "Date", "DATE", "date") if c in df.columns),
        None,
    )
    season_col = next(
        (c for c in ("SEASON", "Season", "YEAR") if c in df.columns),
        None,
    )

    if date_col is None or home_col is None or away_col is None:
        logger.warning(
            "add_sos_to_frame: required columns not found (need %s, %s, %s) "
            "-- returning unchanged df",
            "TEAM_NAME", "TEAM_NAME.1", "GAME_DATE",
        )
        return df

    sos_home_l30 = []
    sos_away_l30 = []
    sos_home_season = []
    sos_away_season = []

    for _, row in df.iterrows():
        game_date = _parse_date(row[date_col])
        home_team = str(row.get(home_col, "") or "").strip()
        away_team = str(row.get(away_col, "") or "").strip()

        # Determine season: prefer SEASON column, else derive from year
        if season_col and pd.notna(row.get(season_col)):
            season = str(row[season_col]).strip()
        elif game_date is not None:
            season = str(game_date.year)
        else:
            season = ""

        if game_date is None:
            sos_home_l30.append(_DEFAULT_SOS)
            sos_away_l30.append(_DEFAULT_SOS)
            sos_home_season.append(_DEFAULT_SOS)
            sos_away_season.append(_DEFAULT_SOS)
            continue

        sos_home_l30.append(_compute_sos_l30(win_pct, season, home_team, game_date))
        sos_away_l30.append(_compute_sos_l30(win_pct, season, away_team, game_date))
        sos_home_season.append(_compute_sos_season(win_pct, season, home_team, game_date))
        sos_away_season.append(_compute_sos_season(win_pct, season, away_team, game_date))

    df = df.copy()
    df["SOS_HOME_L30"] = sos_home_l30
    df["SOS_AWAY_L30"] = sos_away_l30
    df["SOS_HOME_SEASON"] = sos_home_season
    df["SOS_AWAY_SEASON"] = sos_away_season
    df["DIFF_SOS_L30"] = df["SOS_HOME_L30"] - df["SOS_AWAY_L30"]
    df["DIFF_SOS_SEASON"] = df["SOS_HOME_SEASON"] - df["SOS_AWAY_SEASON"]

    logger.info("add_sos_to_frame: added 6 SOS columns for %d rows", len(df))
    return df


# ------------------------------------------------------------------
# get_sos_features (live prediction, single team)
# ------------------------------------------------------------------

def get_sos_features(
    win_pct: Dict,
    schedule: Dict,
    team_name: str,
    game_date,
    season: str = None,
) -> Dict[str, float]:
    """Compute SOS features for a single team for live prediction.

    Returns dict with keys: SOS_L30, SOS_SEASON.
    Callers apply _HOME / _AWAY suffix when merging sides.
    """
    gd = _parse_date(game_date) if not hasattr(game_date, "year") else game_date
    if gd is None:
        return {"SOS_L30": _DEFAULT_SOS, "SOS_SEASON": _DEFAULT_SOS}

    if season is None:
        season = str(gd.year)

    sos_l30 = _compute_sos_l30(win_pct, season, team_name, gd)
    sos_season = _compute_sos_season(win_pct, season, team_name, gd)

    return {
        "SOS_L30": sos_l30,
        "SOS_SEASON": sos_season,
    }
