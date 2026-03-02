"""Starting pitcher features for MLB W/L predictor.

The most important module — starting pitcher is the #1 predictor in MLB.
A dominant SP (low FIP, high K%, low BB%) heavily skews the win probability.

24 features total (12 per side):
    FIP_ROLL5, XFIP_ROLL5, SIERA_ROLL5, K_PCT_ROLL5, BB_PCT_ROLL5, K_BB_ROLL5,
    DAYS_REST, PITCH_LOAD_30D, SP_ERA_SEASON, SP_GB_PCT, SP_HR_9, SP_WHIP

Rolling windows use last 5 starts; fall back to season average when
fewer than 5 starts are available (early season or injury returns).

Usage (training):
    lookup = build_pitcher_lookup(pitcher_db_path)
    df = add_sp_features_to_frame(df, lookup)

Usage (live prediction):
    feats = get_game_sp_features(lookup, game_date, season, home_sp, away_sp)
"""

import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.config import get_logger

logger = get_logger(__name__)

# Sentinel default when no data is available.
# Use league-average values so the model isn't distorted.
_SP_DEFAULTS = {
    "FIP_ROLL5": 4.20,
    "XFIP_ROLL5": 4.20,
    "SIERA_ROLL5": 4.20,
    "K_PCT_ROLL5": 0.220,
    "BB_PCT_ROLL5": 0.080,
    "K_BB_ROLL5": 0.140,
    "DAYS_REST": 5,
    "PITCH_LOAD_30D": 30.0,
    "SP_ERA_SEASON": 4.20,
    "SP_GB_PCT": 0.440,
    "SP_HR_9": 1.20,
    "SP_WHIP": 1.30,
}

_ROLL_N = 5  # number of starts for rolling window


def _parse_date(date_val) -> Optional[datetime.date]:
    """Normalise a date value from SQLite to datetime.date."""
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


def _rolling_mean(values: List[float], n: int) -> float:
    """Mean of last n values; returns mean of all if fewer than n."""
    if not values:
        return float("nan")
    return float(sum(values[-n:]) / len(values[-n:]))


def build_pitcher_lookup(pitcher_db_path: str) -> Dict:
    """Build per-pitcher per-start history from MLBPitcherData.sqlite.

    Returns:
        {pitcher_name: {season_str: [
            {start_date, IP, ERA, FIP, xFIP, SIERA, K_PCT, BB_PCT,
             GB_PCT, HR_9, WHIP}
        ]}}  — starts sorted chronologically within each season list.

    Falls back gracefully if tables are missing or columns differ.
    """
    lookup: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))

    try:
        con = sqlite3.connect(pitcher_db_path)
    except Exception as exc:
        logger.warning("Cannot open pitcher DB %s: %s", pitcher_db_path, exc)
        return lookup

    try:
        cursor = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            # Expected table: pitcher_starts or pitcher_starts_YYYY-YY
            if "pitcher" not in table.lower() and "sp_starts" not in table.lower():
                continue

            try:
                rows = con.execute(
                    f'SELECT * FROM "{table}"'
                ).fetchall()
                if not rows:
                    continue
                col_names = [
                    d[0].upper()
                    for d in con.execute(f'PRAGMA table_info("{table}")').fetchall()
                ]
            except Exception:
                continue

            # Determine column indices (case-insensitive)
            def _col(name: str) -> Optional[int]:
                try:
                    return col_names.index(name.upper())
                except ValueError:
                    return None

            idx_name = _col("PITCHER_NAME") or _col("NAME") or _col("PLAYER_NAME")
            idx_date = _col("GAME_DATE") or _col("DATE") or _col("START_DATE")
            idx_season = _col("SEASON")
            idx_ip = _col("IP")
            idx_era = _col("ERA")
            idx_fip = _col("FIP")
            idx_xfip = _col("XFIP")
            idx_siera = _col("SIERA")
            idx_kpct = _col("K_PCT") or _col("K%")
            idx_bbpct = _col("BB_PCT") or _col("BB%")
            idx_gbpct = _col("GB_PCT") or _col("GB%")
            idx_hr9 = _col("HR_9") or _col("HR/9")
            idx_whip = _col("WHIP")

            if idx_name is None or idx_date is None:
                logger.debug("Table %s missing NAME/DATE columns, skipping", table)
                continue

            for row in rows:
                name = row[idx_name]
                if not isinstance(name, str) or not name.strip():
                    continue
                name = name.strip()

                game_date = _parse_date(row[idx_date])
                if game_date is None:
                    continue

                season = str(row[idx_season]).strip() if idx_season is not None else str(game_date.year)

                def _safe_float(idx):
                    if idx is None:
                        return None
                    try:
                        v = row[idx]
                        return float(v) if v is not None else None
                    except (TypeError, ValueError):
                        return None

                record = {
                    "start_date": game_date,
                    "IP": _safe_float(idx_ip),
                    "ERA": _safe_float(idx_era),
                    "FIP": _safe_float(idx_fip),
                    "xFIP": _safe_float(idx_xfip),
                    "SIERA": _safe_float(idx_siera),
                    "K_PCT": _safe_float(idx_kpct),
                    "BB_PCT": _safe_float(idx_bbpct),
                    "GB_PCT": _safe_float(idx_gbpct),
                    "HR_9": _safe_float(idx_hr9),
                    "WHIP": _safe_float(idx_whip),
                }
                lookup[name][season].append(record)

        # Sort each pitcher-season list chronologically
        for pitcher in lookup:
            for season in lookup[pitcher]:
                lookup[pitcher][season].sort(key=lambda x: x["start_date"])

    finally:
        con.close()

    logger.info("Pitcher lookup built: %d pitchers", len(lookup))
    return lookup


def get_sp_features(
    pitcher_lookup: Dict,
    pitcher_name: str,
    game_date,
    season: str,
) -> Dict[str, float]:
    """Compute rolling 5-start features for a starting pitcher before game_date.

    For each stat, uses the rolling mean of the last 5 starts that occurred
    strictly before game_date in the given season.  If fewer than 5 starts
    are available, falls back to the season average of all prior starts.
    If no prior starts exist at all, returns league-average defaults.

    Returns:
        Dict with keys FIP_ROLL5, XFIP_ROLL5, SIERA_ROLL5, K_PCT_ROLL5,
        BB_PCT_ROLL5, K_BB_ROLL5, DAYS_REST, PITCH_LOAD_30D,
        SP_ERA_SEASON, SP_GB_PCT, SP_HR_9, SP_WHIP
    """
    result = dict(_SP_DEFAULTS)

    if not pitcher_name or not isinstance(pitcher_name, str):
        return result

    gd = _parse_date(game_date) if not hasattr(game_date, "year") else game_date
    if gd is None:
        return result

    # Try to find the pitcher; attempt partial name match as fallback
    season_starts = None
    if pitcher_name in pitcher_lookup:
        season_starts = pitcher_lookup[pitcher_name].get(str(season), [])

    if season_starts is None:
        # Partial match (last name search)
        last = pitcher_name.split()[-1].lower()
        candidates = [
            k for k in pitcher_lookup
            if k.split()[-1].lower() == last
        ]
        if len(candidates) == 1:
            season_starts = pitcher_lookup[candidates[0]].get(str(season), [])
        else:
            logger.debug("Pitcher not found in lookup: '%s'", pitcher_name)
            return result

    # Filter starts strictly before game_date
    prior = [s for s in (season_starts or []) if s["start_date"] < gd]
    if not prior:
        return result

    # Helper: collect non-None values for a key from a list of start dicts
    def _vals(starts, key):
        return [s[key] for s in starts if s[key] is not None]

    def _roll(key, n=_ROLL_N):
        vals = _vals(prior, key)
        if not vals:
            return None
        return _rolling_mean(vals, n)

    def _season_avg(key):
        vals = _vals(prior, key)
        return float(sum(vals) / len(vals)) if vals else None

    # Rolling 5-start metrics
    fip = _roll("FIP")
    xfip = _roll("xFIP")
    siera = _roll("SIERA")
    kpct = _roll("K_PCT")
    bbpct = _roll("BB_PCT")

    if fip is not None:
        result["FIP_ROLL5"] = round(fip, 3)
    if xfip is not None:
        result["XFIP_ROLL5"] = round(xfip, 3)
    if siera is not None:
        result["SIERA_ROLL5"] = round(siera, 3)
    if kpct is not None:
        result["K_PCT_ROLL5"] = round(kpct, 4)
    if bbpct is not None:
        result["BB_PCT_ROLL5"] = round(bbpct, 4)
    if kpct is not None and bbpct is not None:
        result["K_BB_ROLL5"] = round(kpct - bbpct, 4)

    # Season ERA
    era_season = _season_avg("ERA")
    if era_season is not None:
        result["SP_ERA_SEASON"] = round(era_season, 3)

    # Ground ball rate and HR/9 (season average — stable metrics)
    gbpct_season = _season_avg("GB_PCT")
    if gbpct_season is not None:
        result["SP_GB_PCT"] = round(gbpct_season, 4)

    hr9_season = _season_avg("HR_9")
    if hr9_season is not None:
        result["SP_HR_9"] = round(hr9_season, 3)

    whip_season = _season_avg("WHIP")
    if whip_season is not None:
        result["SP_WHIP"] = round(whip_season, 3)

    # Days rest = days between last start and game_date
    last_start_date = prior[-1]["start_date"]
    days_rest = (gd - last_start_date).days
    result["DAYS_REST"] = max(0, days_rest)

    # Pitch load = total IP in last 30 days
    cutoff_30 = gd - timedelta(days=30)
    load_starts = [s for s in prior if s["start_date"] >= cutoff_30]
    ip_vals = [s["IP"] for s in load_starts if s["IP"] is not None]
    result["PITCH_LOAD_30D"] = round(sum(ip_vals), 1) if ip_vals else 0.0

    return result


def get_game_sp_features(
    pitcher_lookup: Dict,
    game_date,
    season: str,
    home_sp: str,
    away_sp: str,
) -> Dict[str, float]:
    """Merge home and away SP features into a single game-level dict.

    Returns dict with all keys suffixed _HOME or _AWAY.
    """
    home_feats = get_sp_features(pitcher_lookup, home_sp, game_date, season)
    away_feats = get_sp_features(pitcher_lookup, away_sp, game_date, season)

    merged = {}
    for key, val in home_feats.items():
        merged[f"{key}_HOME"] = val
    for key, val in away_feats.items():
        merged[f"{key}_AWAY"] = val
    return merged


def add_sp_features_to_frame(
    df: pd.DataFrame,
    pitcher_lookup: Dict,
) -> pd.DataFrame:
    """Add SP rolling features for both home and away pitchers to training df.

    Expects columns SP_NAME_HOME, SP_NAME_AWAY, GAME_DATE (or DATE), SEASON.

    Returns DataFrame with 24 new SP feature columns.
    """
    # Determine required column names
    date_col = next(
        (c for c in ("GAME_DATE", "Date", "DATE", "date") if c in df.columns), None
    )
    season_col = next(
        (c for c in ("SEASON", "Season", "season") if c in df.columns), None
    )
    home_sp_col = next(
        (c for c in ("SP_NAME_HOME", "SP_HOME", "HOME_SP") if c in df.columns), None
    )
    away_sp_col = next(
        (c for c in ("SP_NAME_AWAY", "SP_AWAY", "AWAY_SP") if c in df.columns), None
    )

    if date_col is None:
        logger.warning("add_sp_features_to_frame: no date column found — returning unchanged df")
        return df
    if home_sp_col is None or away_sp_col is None:
        logger.warning(
            "add_sp_features_to_frame: SP_NAME_HOME/AWAY columns not found — "
            "returning unchanged df"
        )
        return df

    records = []
    for _, row in df.iterrows():
        game_date = row[date_col]
        season = str(row[season_col]) if season_col else str(
            _parse_date(game_date).year if _parse_date(game_date) else "unknown"
        )
        home_sp = row.get(home_sp_col, "") or ""
        away_sp = row.get(away_sp_col, "") or ""

        feats = get_game_sp_features(pitcher_lookup, game_date, season, home_sp, away_sp)
        records.append(feats)

    if not records:
        return df

    sp_df = pd.DataFrame(records, index=df.index)
    result = pd.concat([df, sp_df], axis=1)
    logger.info(
        "add_sp_features_to_frame: added %d SP columns for %d rows",
        len(sp_df.columns), len(df),
    )
    return result
