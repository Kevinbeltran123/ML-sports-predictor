"""Differential features for MLB W/L predictor.

For each stat pair (home_col, away_col), computes DIFF = home - away.
Differential features reduce dimensionality and force the model to learn
relative advantage directly, consistent with the NBA implementation.

15 differential pairs covering pitching, batting, bullpen, fatigue, and park.

Usage (training and live):
    df = add_differential_features(df)
    game_diffs = get_game_differentials(home_features, away_features)
"""

from typing import Dict, List, Tuple

import pandas as pd

from src.config import get_logger

logger = get_logger(__name__)

# (home_col, away_col, diff_name)
# Convention: DIFF positive = home team advantage (for most stats).
# Exceptions (negative = home disadvantage):
#   DIFF_FIP, DIFF_XFIP, DIFF_SIERA, DIFF_BP_ERA — lower ERA/FIP is better,
#     so positive DIFF means home pitcher is WORSE. Model learns sign automatically.
MLB_DIFF_PAIRS: List[Tuple[str, str, str]] = [
    # Pitching quality
    ("FIP_ROLL5_HOME",    "FIP_ROLL5_AWAY",    "DIFF_FIP"),
    ("XFIP_ROLL5_HOME",   "XFIP_ROLL5_AWAY",   "DIFF_XFIP"),
    ("SIERA_ROLL5_HOME",  "SIERA_ROLL5_AWAY",   "DIFF_SIERA"),
    ("K_BB_ROLL5_HOME",   "K_BB_ROLL5_AWAY",   "DIFF_K_BB"),
    # Team offense
    ("OPS_ROLL15_HOME",   "OPS_ROLL15_AWAY",   "DIFF_OPS"),
    ("RUN_RATE_HOME",     "RUN_RATE_AWAY",      "DIFF_RUN_RATE"),
    # Bullpen
    ("BP_ERA_7D_HOME",    "BP_ERA_7D_AWAY",     "DIFF_BP_ERA"),
    # Schedule / fatigue
    ("DAYS_REST_HOME",    "DAYS_REST_AWAY",     "DIFF_REST"),
    ("GAMES_IN_7_HOME",   "GAMES_IN_7_AWAY",   "DIFF_GAMES_7"),
    ("TRAVEL_DIST_HOME",  "TRAVEL_DIST_AWAY",  "DIFF_TRAVEL"),
    # Offensive form
    ("MOMENTUM_OPS_HOME", "MOMENTUM_OPS_AWAY", "DIFF_MOMENTUM"),
    # Batting discipline
    ("K_PCT_BAT_HOME",    "K_PCT_BAT_AWAY",    "DIFF_K_PCT_BAT"),
    ("BB_PCT_BAT_HOME",   "BB_PCT_BAT_AWAY",   "DIFF_BB_PCT_BAT"),
    # Power
    ("HR_RATE_HOME",      "HR_RATE_AWAY",       "DIFF_HR_RATE"),
    # Defence
    ("ERRORS_RATE_HOME",  "ERRORS_RATE_AWAY",   "DIFF_ERRORS"),
]


def add_differential_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add MLB differential features (home - away) to a training DataFrame.

    Only pairs where BOTH columns exist are computed; missing pairs are
    silently skipped.  Original columns are preserved.

    Returns:
        DataFrame with up to 15 new DIFF_* columns appended.
    """
    new_cols: Dict[str, pd.Series] = {}
    added: List[str] = []
    skipped: List[str] = []

    for col_home, col_away, diff_name in MLB_DIFF_PAIRS:
        if col_home in df.columns and col_away in df.columns:
            new_cols[diff_name] = (
                df[col_home].astype(float) - df[col_away].astype(float)
            )
            added.append(diff_name)
        else:
            skipped.append(diff_name)

    if new_cols:
        new_df = pd.DataFrame(new_cols, index=df.index)
        df = pd.concat([df, new_df], axis=1)

    if skipped:
        logger.debug(
            "add_differential_features: skipped %d pairs (columns missing): %s",
            len(skipped), skipped,
        )
    logger.info(
        "add_differential_features: added %d differential features", len(added)
    )
    return df


def get_game_differentials(
    home_features: Dict[str, float],
    away_features: Dict[str, float],
) -> Dict[str, float]:
    """Compute differential features for a single live-prediction game.

    Args:
        home_features: Dict of home-team features (keys WITHOUT _HOME suffix).
        away_features: Dict of away-team features (keys WITHOUT _AWAY suffix).

    Returns:
        Dict of {DIFF_*: float} for all available pairs.

    Example usage:
        home_feats = get_sp_features(lookup, home_sp, date, season)
        away_feats = get_sp_features(lookup, away_sp, date, season)
        diffs = get_game_differentials(home_feats, away_feats)
    """
    diffs: Dict[str, float] = {}

    for col_home, col_away, diff_name in MLB_DIFF_PAIRS:
        # Strip _HOME/_AWAY suffix if callers pass full column names
        home_key = col_home.replace("_HOME", "")
        away_key = col_away.replace("_AWAY", "")

        h_val = home_features.get(col_home) or home_features.get(home_key)
        a_val = away_features.get(col_away) or away_features.get(away_key)

        if h_val is not None and a_val is not None:
            try:
                diffs[diff_name] = round(float(h_val) - float(a_val), 5)
            except (TypeError, ValueError):
                pass

    return diffs


def get_differential_columns() -> List[str]:
    """Return list of all differential column names."""
    return [diff_name for _, _, diff_name in MLB_DIFF_PAIRS]
