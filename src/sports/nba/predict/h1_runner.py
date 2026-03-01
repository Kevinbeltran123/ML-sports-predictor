"""First Half (1H) Moneyline Runner.

Loads H1 XGBoost + CatBoost models from models/h1moneyline/,
runs ensemble prediction, and prints compact output.

Reuses the same feature matrix X from the full-game pipeline.
"""

import json
import re
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from colorama import Fore, Style

from src.config import NBA_H1_MODELS_DIR, get_logger
from src.core.betting import expected_value as Expected_Value
from src.core.betting import kelly_criterion as kc

logger = get_logger(__name__)

# Module-level caches
_h1_xgb_booster = None
_h1_xgb_calibrator = None
_h1_cat_model = None
_h1_conformal = None
_h1_metadata = None


def _load_h1_models():
    """Load H1 models from models/h1moneyline/. Returns True if loaded."""
    global _h1_xgb_booster, _h1_xgb_calibrator, _h1_cat_model, _h1_conformal, _h1_metadata

    if _h1_xgb_booster is not None:
        return True

    if not NBA_H1_MODELS_DIR.exists():
        logger.warning("H1 models dir not found: %s", NBA_H1_MODELS_DIR)
        return False

    # XGBoost — pick by accuracy in filename, newest mtime
    xgb_files = list(NBA_H1_MODELS_DIR.glob("XGBoost_*H1*.json"))
    if not xgb_files:
        logger.warning("No H1 XGBoost model found")
        return False

    def _score_path(p):
        m = re.search(r"(\d+\.\d+)%", p.name)
        acc = float(m.group(1)) if m else 0.0
        return (acc, p.stat().st_mtime)

    xgb_path = max(xgb_files, key=_score_path)
    _h1_xgb_booster = xgb.Booster()
    _h1_xgb_booster.load_model(str(xgb_path))
    logger.info("H1 XGBoost: %s", xgb_path.name)

    # Calibrator
    cal_path = xgb_path.with_name(xgb_path.stem + "_calibration.pkl")
    if cal_path.exists():
        _h1_xgb_calibrator = joblib.load(cal_path)

    # CatBoost
    cat_files = [p for p in NBA_H1_MODELS_DIR.glob("CatBoost_*H1*.pkl")
                 if "calibration" not in p.name]
    if cat_files:
        cat_path = max(cat_files, key=_score_path)
        _h1_cat_model = joblib.load(cat_path)
        logger.info("H1 CatBoost: %s", cat_path.name)

    # Conformal
    conf_path = NBA_H1_MODELS_DIR / "ensemble_conformal.pkl"
    if conf_path.exists():
        _h1_conformal = joblib.load(conf_path)

    # Metadata (weights)
    meta_path = NBA_H1_MODELS_DIR / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            _h1_metadata = json.load(f)

    return True


def predict_h1(data, games, h1_odds_home=None, h1_odds_away=None, kelly_flag=False):
    """Run H1 ensemble prediction and print output.

    Args:
        data: numpy array of features (same as full-game, shape (n_games, n_features))
        games: list of game names (e.g., ["Denver Nuggets", "Boston Celtics"])
        h1_odds_home: list of H1 ML home odds (american), or None
        h1_odds_away: list of H1 ML away odds (american), or None
        kelly_flag: whether to compute kelly sizing

    Returns:
        list of prediction dicts, or None if models not available
    """
    if not _load_h1_models():
        return None

    n = len(games) // 2  # games is flat [home, away, home, away, ...]

    # Predict with XGBoost
    dmat = xgb.DMatrix(data)
    if _h1_xgb_calibrator is not None:
        p_xgb = _h1_xgb_calibrator.predict_proba(data)
    else:
        p_xgb = _h1_xgb_booster.predict(dmat)

    # Predict with CatBoost
    if _h1_cat_model is not None:
        p_cat = _h1_cat_model.predict_proba(data)
    else:
        p_cat = p_xgb  # fallback to XGBoost only

    # Ensemble weights
    w_xgb = 0.60
    w_cat = 0.40
    if _h1_metadata and "weights" in _h1_metadata:
        w_xgb = _h1_metadata["weights"].get("xgb", 0.60)
        w_cat = _h1_metadata["weights"].get("cat", 0.40)

    p_ensemble = w_xgb * p_xgb + w_cat * p_cat

    # Conformal
    set_sizes = None
    if _h1_conformal is not None:
        set_sizes, _ = _h1_conformal.predict_confidence(p_ensemble)

    # Print output
    print(f"\n{'='*65}")
    print(f"  FIRST HALF (1H) MONEYLINE PREDICTIONS")
    if _h1_metadata:
        acc = _h1_metadata.get("ensemble_accuracy", 0)
        print(f"  Model: Ensemble {acc:.1%} | Weights: XGB {w_xgb:.0%} / Cat {w_cat:.0%}")
    print(f"{'='*65}")

    results = []
    for idx in range(len(p_ensemble)):
        home = games[idx * 2]
        away = games[idx * 2 + 1]
        p_home = float(p_ensemble[idx][1])
        p_away = float(p_ensemble[idx][0])

        # Conformal tag
        cs = int(set_sizes[idx]) if set_sizes is not None else 0
        if cs == 1:
            tag = f"{Fore.GREEN}BET{Style.RESET_ALL}"
        elif cs == 2:
            tag = f"{Fore.YELLOW}SKIP{Style.RESET_ALL}"
        else:
            tag = "---"

        # Pick + odds + EV
        pick_home = p_home >= 0.5
        pick = home.split()[-1] if pick_home else away.split()[-1]
        pick_prob = p_home if pick_home else p_away

        h_odds = h1_odds_home[idx] if h1_odds_home and idx < len(h1_odds_home) else None
        a_odds = h1_odds_away[idx] if h1_odds_away and idx < len(h1_odds_away) else None

        ev_str = ""
        kelly_str = ""
        ev_val = None
        k_val = 0.0
        if h_odds and a_odds:
            odds = int(h_odds) if pick_home else int(a_odds)
            ev_val = float(Expected_Value.expected_value(pick_prob, odds))
            ev_str = f"  EV {ev_val:+.1f}%"

            if kelly_flag and ev_val > 0:
                k_val = float(kc.calculate_eighth_kelly(odds, pick_prob))
                kelly_str = f"  K={k_val:.2f}%"

        odds_display = ""
        if h_odds and a_odds:
            odds_display = f"  [{h_odds:+d}/{a_odds:+d}]"

        print(f"  [{tag}] {home.split()[-1]} vs {away.split()[-1]}: "
              f"1H {pick} {pick_prob:.1%}{odds_display}{ev_str}{kelly_str}")

        # Individual model probs for agreement check
        xgb_home = float(p_xgb[idx][1])
        cat_home = float(p_cat[idx][1])
        xgb_pick_home = xgb_home >= 0.5
        cat_pick_home = cat_home >= 0.5
        h1_models_agree = xgb_pick_home == cat_pick_home

        # Plain tag for Telegram (without ANSI color codes)
        tag_plain = "BET" if cs == 1 else ("SKIP" if cs == 2 else "---")

        results.append({
            "home_team": home,
            "away_team": away,
            "h1_prob_home": p_home,
            "h1_prob_away": p_away,
            "h1_xgb_home": xgb_home,
            "h1_cat_home": cat_home,
            "h1_models_agree": h1_models_agree,
            "h1_conformal_set_size": cs,
            "h1_ml_home_odds": h_odds,
            "h1_ml_away_odds": a_odds,
            "h1_ev": ev_val,
            "h1_kelly": k_val,
            "h1_pick": pick,
            "h1_tag": tag_plain,
        })

    print(f"{'='*65}")
    return results
