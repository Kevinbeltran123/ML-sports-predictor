"""Odds-derived features for MLB W/L predictor.

Market-implied probabilities and vig magnitude are among the strongest
predictors in any sports model.  The bookmaker's closing line represents
the aggregate of sharp money and is a near-unbiased estimate of true
win probability.

Features produced:
    VIG_MAGNITUDE       — bookmaker overround (sum of implied probs - 1)
    IMPLIED_HOME        — vig-removed implied probability for home team
    IMPLIED_AWAY        — vig-removed implied probability for away team
    F5_VIG              — first 5 innings vig (optional)
    F5_IMPLIED_HOME     — F5 implied prob for home (optional)
    RUNLINE_IMPLIED_HOME — run-line implied prob for home (optional)

Usage (training):
    df = add_odds_features_to_frame(df)  # expects ML_HOME, ML_AWAY columns

Usage (live):
    feats = get_odds_features(ml_home, ml_away, f5_home, f5_away, rl_home, rl_away)
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.config import get_logger

logger = get_logger(__name__)

# Column name mapping (what the training DataFrame might call these)
_ML_HOME_COLS = ("ML_HOME", "ML_Home", "ml_home", "MONEYLINE_HOME")
_ML_AWAY_COLS = ("ML_AWAY", "ML_Away", "ml_away", "MONEYLINE_AWAY")
_F5_HOME_COLS = ("F5_HOME", "F5_ML_HOME", "f5_home")
_F5_AWAY_COLS = ("F5_AWAY", "F5_ML_AWAY", "f5_away")
_RL_HOME_COLS = ("RL_HOME", "RUNLINE_HOME", "rl_home")


def compute_implied_prob(ml_odds: float) -> float:
    """Convert American odds to raw implied probability.

    Formula:
        Favourite (negative odds): |odds| / (|odds| + 100)
        Underdog  (positive odds): 100 / (odds + 100)

    Returns raw probability BEFORE vig removal.

    Examples:
        -150 → 0.600  (60% implied)
        +130 → 0.435  (43.5% implied)
        -110 → 0.524  (52.4% implied — standard juice)
    """
    ml_odds = float(ml_odds)
    if ml_odds < 0:
        return abs(ml_odds) / (abs(ml_odds) + 100.0)
    return 100.0 / (ml_odds + 100.0)


def compute_vig_magnitude(ml_home: float, ml_away: float) -> float:
    """Compute bookmaker overround (vig) for a game.

    VIG = p_home_raw + p_away_raw - 1.0

    Typical MLB range: 0.04-0.07 (roughly 4-7% margin).
    Values > 0.15 indicate bad data.

    Returns float in [0.0, 0.15+].
    """
    p_home = compute_implied_prob(ml_home)
    p_away = compute_implied_prob(ml_away)
    vig = p_home + p_away - 1.0

    if vig > 0.15:
        logger.debug(
            "VIG_MAGNITUDE high (%.3f) for ML_Home=%.0f ML_Away=%.0f — check data",
            vig, ml_home, ml_away,
        )
    return float(vig)


def _remove_vig(p_home_raw: float, p_away_raw: float) -> tuple:
    """Remove bookmaker vig via normalization.

    Each raw implied prob is divided by their sum so they total 1.0.

    Returns (prob_home_fair, prob_away_fair).
    """
    total = p_home_raw + p_away_raw
    if total <= 0:
        return 0.5, 0.5
    return p_home_raw / total, p_away_raw / total


def get_odds_features(
    ml_home: float,
    ml_away: float,
    f5_home: Optional[float] = None,
    f5_away: Optional[float] = None,
    rl_home: Optional[float] = None,
    rl_away: Optional[float] = None,
) -> Dict[str, float]:
    """Compute all odds-derived features for a single game.

    Args:
        ml_home:  Moneyline odds for home team (American format, e.g. -150 or +130).
        ml_away:  Moneyline odds for away team.
        f5_home:  First-5-innings ML for home (optional).
        f5_away:  First-5-innings ML for away (optional).
        rl_home:  Run-line ML for home (e.g. -110 at -1.5 spread, optional).
        rl_away:  Run-line ML for away (optional, used to compute RUNLINE_IMPLIED_HOME).

    Returns dict with:
        VIG_MAGNITUDE, IMPLIED_HOME, IMPLIED_AWAY,
        F5_VIG (0.0 if missing), F5_IMPLIED_HOME (0.5 if missing),
        RUNLINE_IMPLIED_HOME (0.5 if missing)
    """
    result: Dict[str, float] = {}

    # Main moneyline
    vig = compute_vig_magnitude(ml_home, ml_away)
    result["VIG_MAGNITUDE"] = round(vig, 5)

    p_home_raw = compute_implied_prob(ml_home)
    p_away_raw = compute_implied_prob(ml_away)
    p_home_fair, p_away_fair = _remove_vig(p_home_raw, p_away_raw)

    result["IMPLIED_HOME"] = round(p_home_fair, 5)
    result["IMPLIED_AWAY"] = round(p_away_fair, 5)

    # First 5 innings
    if f5_home is not None and f5_away is not None:
        try:
            f5_vig = compute_vig_magnitude(float(f5_home), float(f5_away))
            result["F5_VIG"] = round(f5_vig, 5)
            p_f5_home_raw = compute_implied_prob(float(f5_home))
            p_f5_away_raw = compute_implied_prob(float(f5_away))
            p_f5_home_fair, _ = _remove_vig(p_f5_home_raw, p_f5_away_raw)
            result["F5_IMPLIED_HOME"] = round(p_f5_home_fair, 5)
        except (TypeError, ValueError):
            result["F5_VIG"] = 0.0
            result["F5_IMPLIED_HOME"] = 0.5
    else:
        result["F5_VIG"] = 0.0
        result["F5_IMPLIED_HOME"] = 0.5

    # Run line
    if rl_home is not None and rl_away is not None:
        try:
            p_rl_home_raw = compute_implied_prob(float(rl_home))
            p_rl_away_raw = compute_implied_prob(float(rl_away))
            p_rl_home_fair, _ = _remove_vig(p_rl_home_raw, p_rl_away_raw)
            result["RUNLINE_IMPLIED_HOME"] = round(p_rl_home_fair, 5)
        except (TypeError, ValueError):
            result["RUNLINE_IMPLIED_HOME"] = 0.5
    elif rl_home is not None:
        # Single-side run line (just compute raw)
        try:
            result["RUNLINE_IMPLIED_HOME"] = round(compute_implied_prob(float(rl_home)), 5)
        except (TypeError, ValueError):
            result["RUNLINE_IMPLIED_HOME"] = 0.5
    else:
        result["RUNLINE_IMPLIED_HOME"] = 0.5

    return result


def add_odds_features_to_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add VIG and implied probability features to training DataFrame.

    Expects ML_HOME and ML_AWAY columns (American odds integers).
    Also looks for F5_HOME/F5_AWAY and RL_HOME/RL_AWAY for optional features.

    Returns DataFrame with new odds feature columns.
    """
    ml_home_col = next((c for c in _ML_HOME_COLS if c in df.columns), None)
    ml_away_col = next((c for c in _ML_AWAY_COLS if c in df.columns), None)

    if ml_home_col is None or ml_away_col is None:
        logger.warning(
            "add_odds_features_to_frame: ML_HOME/ML_AWAY columns not found — "
            "returning unchanged df"
        )
        return df

    f5_home_col = next((c for c in _F5_HOME_COLS if c in df.columns), None)
    f5_away_col = next((c for c in _F5_AWAY_COLS if c in df.columns), None)
    rl_home_col = next((c for c in _RL_HOME_COLS if c in df.columns), None)

    records = []
    for _, row in df.iterrows():
        try:
            ml_home = float(row[ml_home_col])
            ml_away = float(row[ml_away_col])
        except (TypeError, ValueError):
            records.append({
                "VIG_MAGNITUDE": 0.05,
                "IMPLIED_HOME": 0.5,
                "IMPLIED_AWAY": 0.5,
                "F5_VIG": 0.0,
                "F5_IMPLIED_HOME": 0.5,
                "RUNLINE_IMPLIED_HOME": 0.5,
            })
            continue

        f5_home = float(row[f5_home_col]) if f5_home_col else None
        f5_away = float(row[f5_away_col]) if f5_away_col else None
        rl_home = float(row[rl_home_col]) if rl_home_col else None

        feats = get_odds_features(ml_home, ml_away, f5_home, f5_away, rl_home)
        records.append(feats)

    if not records:
        return df

    odds_df = pd.DataFrame(records, index=df.index)
    result = pd.concat([df, odds_df], axis=1)
    logger.info(
        "add_odds_features_to_frame: added %d odds columns for %d rows",
        len(odds_df.columns), len(df),
    )
    return result
