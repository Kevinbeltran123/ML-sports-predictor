"""Park factors and weather features for MLB W/L predictor.

Park factors adjust raw offensive stats for the home ballpark's
tendency to inflate or suppress runs, home runs, hits, etc.
Weather — especially wind direction and speed — significantly affects
scoring at open-air parks (think Wrigley on a 20mph out-to-center day).

12 features total:
    PF_RUNS, PF_HR, PF_H, PF_2B, PF_3B, PF_BB   — park factors
    TEMP_F, WIND_SPEED_MPH, WIND_IN_OUT, IS_DOME,
    HUMIDITY_PCT, PRECIP_PROB                      — weather

For domed parks, weather features default to neutral values.
WIND_IN_OUT is a continuous value in [-1, +1]:
    +1 = perfectly blowing out (favours HR / high scoring)
    -1 = perfectly blowing in (suppresses scoring)
     0 = crosswind or no wind

Usage (training):
    park_lookup = build_park_lookup(park_db_path)
    df = add_park_weather_to_frame(df, park_lookup, weather_lookup)

Usage (live):
    feats = get_park_weather_features(park_lookup, home_team, game_date, weather_data)
"""

import math
import sqlite3
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

from src.config import get_logger
from src.sports.mlb.config_mlb import (
    MLB_DOMED_PARKS,
    MLB_PLATE_ORIENTATION,
    MLB_PARKS,
)

logger = get_logger(__name__)

# Neutral park factor (100 = league average)
_PF_DEFAULT = 100.0

# Neutral weather defaults
_WEATHER_DEFAULTS = {
    "TEMP_F": 72.0,
    "WIND_SPEED_MPH": 5.0,
    "WIND_IN_OUT": 0.0,
    "IS_DOME": 0,
    "HUMIDITY_PCT": 50.0,
    "PRECIP_PROB": 0.0,
}

# Default park factors (league average)
_PF_KEYS = ("PF_RUNS", "PF_HR", "PF_H", "PF_2B", "PF_3B", "PF_BB")
_PF_DEFAULTS = {k: _PF_DEFAULT for k in _PF_KEYS}


def _parse_date(date_val) -> Optional[datetime.date]:
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


def compute_wind_in_out(
    wind_dir_deg: float,
    wind_speed: float,
    plate_orientation: float,
) -> float:
    """Compute wind component blowing in or out of the park.

    The plate orientation (MLB_PLATE_ORIENTATION) is the direction home
    plate faces (i.e., the direction from home plate toward centre field).
    Wind blowing toward home plate from centre = blowing IN (suppresses scoring).
    Wind blowing from home plate toward centre = blowing OUT (promotes scoring).

    Formula:
        - outfield_dir = plate_orientation (degrees from North, clockwise)
        - Angle between wind direction and outfield direction is the key.
        - WIND_IN_OUT = cos(angle) * (wind_speed / 20.0) clamped to [-1, 1]
          where angle = 0 means wind blows directly out.

    Args:
        wind_dir_deg:     Meteorological wind direction (degrees, 0=N, 90=E,
                          180=S, 270=W).  Wind direction = the direction
                          FROM which the wind is blowing.
        wind_speed:       Wind speed in MPH.
        plate_orientation: Degrees the home plate faces from North (clockwise).
                           Same convention as MLB_PLATE_ORIENTATION.

    Returns:
        float in [-1, +1]. Positive = blowing out (hitter-friendly).
    """
    if wind_speed <= 0:
        return 0.0

    # Direction toward which wind is blowing (opposite of "from")
    wind_toward = (wind_dir_deg + 180.0) % 360.0

    # plate_orientation = direction from home plate toward CF
    # If wind blows toward CF (i.e., wind_toward == plate_orientation),
    # the wind is blowing OUT → positive WIND_IN_OUT.
    angle_diff = math.radians(wind_toward - plate_orientation)
    raw = math.cos(angle_diff)  # 1 = blowing out, -1 = blowing in

    # Scale by speed (normalised to 20 mph = "strong wind" reference)
    scaled = raw * min(wind_speed / 20.0, 1.0)
    return round(max(-1.0, min(1.0, scaled)), 4)


def build_park_lookup(park_db_path: str) -> Dict:
    """Load park factors from MLBParkFactors.sqlite.

    Returns:
        {team_name: {season_str: {PF_RUNS, PF_HR, PF_H, PF_2B, PF_3B, PF_BB}}}

    Falls back to static _PF_DEFAULTS if table/data is missing.
    """
    lookup: Dict[str, Dict] = {}

    try:
        con = sqlite3.connect(park_db_path)
    except Exception as exc:
        logger.warning("Cannot open park factors DB %s: %s", park_db_path, exc)
        return lookup

    try:
        cursor = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            if "park" not in table.lower() and "factor" not in table.lower():
                continue

            try:
                col_info = con.execute(f'PRAGMA table_info("{table}")').fetchall()
                col_names = [d[0].upper() for d in col_info]
                rows = con.execute(f'SELECT * FROM "{table}"').fetchall()
            except Exception:
                continue

            def _col(name: str) -> Optional[int]:
                try:
                    return col_names.index(name.upper())
                except ValueError:
                    return None

            idx_team = _col("TEAM") or _col("TEAM_NAME")
            idx_season = _col("SEASON") or _col("YEAR")
            idx_pf_r = _col("PF_RUNS") or _col("PF_R") or _col("RUN_FACTOR")
            idx_pf_hr = _col("PF_HR") or _col("HR_FACTOR")
            idx_pf_h = _col("PF_H") or _col("HIT_FACTOR")
            idx_pf_2b = _col("PF_2B") or _col("DOUBLE_FACTOR")
            idx_pf_3b = _col("PF_3B") or _col("TRIPLE_FACTOR")
            idx_pf_bb = _col("PF_BB") or _col("WALK_FACTOR")

            if idx_team is None:
                continue

            for row in rows:
                team = row[idx_team]
                if not isinstance(team, str):
                    continue
                team = team.strip()
                season = str(row[idx_season]).strip() if idx_season is not None else "all"

                def _val(idx, default=_PF_DEFAULT):
                    if idx is None:
                        return default
                    try:
                        v = row[idx]
                        return float(v) if v is not None else default
                    except (TypeError, ValueError):
                        return default

                pf = {
                    "PF_RUNS": _val(idx_pf_r),
                    "PF_HR": _val(idx_pf_hr),
                    "PF_H": _val(idx_pf_h),
                    "PF_2B": _val(idx_pf_2b),
                    "PF_3B": _val(idx_pf_3b),
                    "PF_BB": _val(idx_pf_bb),
                }
                if team not in lookup:
                    lookup[team] = {}
                lookup[team][season] = pf

    finally:
        con.close()

    logger.info("Park factors lookup built: %d teams", len(lookup))
    return lookup


def _get_park_factors(park_lookup: Dict, home_team: str, season: str) -> Dict[str, float]:
    """Retrieve park factors for home_team in a given season.

    Falls back to 'all' key, then to prior seasons, then to defaults.
    """
    team_data = park_lookup.get(home_team, {})
    if not team_data:
        return dict(_PF_DEFAULTS)

    if season in team_data:
        return dict(team_data[season])

    if "all" in team_data:
        return dict(team_data["all"])

    # Use most recent season as fallback
    seasons_sorted = sorted(team_data.keys(), reverse=True)
    if seasons_sorted:
        return dict(team_data[seasons_sorted[0]])

    return dict(_PF_DEFAULTS)


def get_park_weather_features(
    park_lookup: Dict,
    home_team: str,
    game_date,
    weather_data: Optional[Dict] = None,
    season: Optional[str] = None,
) -> Dict:
    """Compute park + weather features for a game.

    Args:
        park_lookup:  From build_park_lookup().
        home_team:    Home team name (determines park and plate orientation).
        game_date:    Game date (str or datetime.date).
        weather_data: Optional dict with keys:
                        temp_f, wind_speed_mph, wind_dir_deg,
                        humidity_pct, precip_prob
                      Pass None for training (historical games) to use defaults.
        season:       Season string (e.g. "2024"). Inferred from game_date if None.

    Returns:
        Dict with 12 features: PF_RUNS, PF_HR, PF_H, PF_2B, PF_3B, PF_BB,
        TEMP_F, WIND_SPEED_MPH, WIND_IN_OUT, IS_DOME, HUMIDITY_PCT, PRECIP_PROB
    """
    result: Dict = {}

    # Determine season
    if season is None:
        gd = _parse_date(game_date) if not hasattr(game_date, "year") else game_date
        season = str(gd.year) if gd else "unknown"

    # Park factors
    pf = _get_park_factors(park_lookup, home_team, season)
    result.update(pf)

    # IS_DOME
    is_dome = 1 if home_team in MLB_DOMED_PARKS else 0
    result["IS_DOME"] = is_dome

    # Weather: for domed parks use neutral defaults regardless of weather_data
    if is_dome:
        result["TEMP_F"] = 72.0
        result["WIND_SPEED_MPH"] = 0.0
        result["WIND_IN_OUT"] = 0.0
        result["HUMIDITY_PCT"] = 50.0
        result["PRECIP_PROB"] = 0.0
        return result

    # Open-air park
    if weather_data is None:
        result.update(_WEATHER_DEFAULTS)
        return result

    temp = weather_data.get("temp_f", _WEATHER_DEFAULTS["TEMP_F"])
    wind_speed = weather_data.get("wind_speed_mph", _WEATHER_DEFAULTS["WIND_SPEED_MPH"])
    wind_dir = weather_data.get("wind_dir_deg")
    humidity = weather_data.get("humidity_pct", _WEATHER_DEFAULTS["HUMIDITY_PCT"])
    precip = weather_data.get("precip_prob", _WEATHER_DEFAULTS["PRECIP_PROB"])

    result["TEMP_F"] = float(temp)
    result["WIND_SPEED_MPH"] = float(wind_speed)
    result["HUMIDITY_PCT"] = float(humidity)
    result["PRECIP_PROB"] = float(precip)

    # Wind in/out using plate orientation from config
    if wind_dir is not None:
        plate_orient = MLB_PLATE_ORIENTATION.get(home_team, 180.0)
        result["WIND_IN_OUT"] = compute_wind_in_out(
            float(wind_dir), float(wind_speed), plate_orient
        )
    else:
        result["WIND_IN_OUT"] = 0.0

    return result


def add_park_weather_to_frame(
    df: pd.DataFrame,
    park_lookup: Dict,
    weather_lookup: Optional[Dict] = None,
) -> pd.DataFrame:
    """Add park and weather features to training DataFrame.

    Expects HOME_TEAM (or TEAM_HOME/Home/HOME) and a date column.
    weather_lookup: optional {(game_date_str, home_team): weather_data_dict}

    For historical training, weather_lookup is typically None and
    weather features default to neutral values.

    Returns DataFrame with 12 new park/weather columns.
    """
    date_col = next(
        (c for c in ("GAME_DATE", "Date", "DATE", "date") if c in df.columns), None
    )
    home_col = next(
        (c for c in ("HOME_TEAM", "TEAM_HOME", "Home", "HOME") if c in df.columns), None
    )
    season_col = next(
        (c for c in ("SEASON", "Season", "season") if c in df.columns), None
    )

    if date_col is None or home_col is None:
        logger.warning(
            "add_park_weather_to_frame: required columns not found — returning unchanged df"
        )
        return df

    records = []
    for _, row in df.iterrows():
        home_team = row.get(home_col, "") or ""
        game_date = row[date_col]
        season = str(row[season_col]) if season_col else None

        # Look up weather if available
        weather_data = None
        if weather_lookup is not None:
            gd_str = str(game_date)[:10]
            weather_data = weather_lookup.get((gd_str, home_team))

        feats = get_park_weather_features(
            park_lookup, home_team, game_date, weather_data, season
        )
        records.append(feats)

    if not records:
        return df

    pw_df = pd.DataFrame(records, index=df.index)
    result = pd.concat([df, pw_df], axis=1)
    logger.info(
        "add_park_weather_to_frame: added %d park/weather columns for %d rows",
        len(pw_df.columns), len(df),
    )
    return result
