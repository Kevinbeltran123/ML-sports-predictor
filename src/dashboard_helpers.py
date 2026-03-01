"""Helper functions for Streamlit dashboard.

Bridges existing prediction pipeline → Streamlit DataFrames.
No logic duplication — reuses ensemble_runner, h1_runner, etc.
"""

import numpy as np
import pandas as pd
import xgboost as xgb


def compute_tag(b):
    """Compute BET/SKIP/TRAP/PASS tag from a game block dict."""
    CONF2_KELLY_THRESHOLD = 0.5
    has_trap = b.get("trap_home") or b.get("trap_away")
    max_kelly = max(b.get("kelly_home", 0), b.get("kelly_away", 0))
    conf_ss = b.get("conf_ss")
    conf_uncertain = conf_ss is not None and conf_ss != 1
    conf_override = conf_uncertain and max_kelly >= CONF2_KELLY_THRESHOLD

    if has_trap:
        return "TRAP"
    elif conf_uncertain and not conf_override:
        return "SKIP"
    elif b.get("max_ev", 0) > 0:
        return "BET"
    else:
        return "PASS"


def blocks_to_dataframe(blocks):
    """Convert _build_game_blocks() output to a DataFrame for Streamlit display."""
    rows = []
    for b in blocks:
        tag = compute_tag(b)
        agree = "Yes" if b.get("xgb_agree") else "No"
        rows.append({
            "Game": f"{b['away']} @ {b['home']}",
            "Pick": b["pick"],
            "Prob%": round(b["pick_prob"] * 100, 1),
            "XGB%": b["xgb_conf"],
            "Cat%": b["cat_conf"],
            "Agree": agree,
            "EV": round(b["max_ev"], 1),
            "Kelly%": round(max(b["kelly_home"], b["kelly_away"]), 2),
            "Sigma": round(b["sigma"], 3),
            "Conformal": b.get("conf_ss", "—"),
            "Tag": tag,
            "AH Side": b.get("ah_side") or "—",
            "AH Line": b.get("ah_line", ""),
            "AH EV": round(b.get("ah_ev", 0), 1),
            "O/U": b.get("ou_label", ""),
            "O/U Line": b.get("ou_line", ""),
            "Pred Total": round(b["predicted_total"], 0) if b.get("predicted_total") else "—",
        })
    return pd.DataFrame(rows)


def get_game_contributions(booster, data_row, feature_names):
    """Get per-game feature contributions using XGBoost built-in pred_contribs.

    Returns list of (feature_name, contribution) sorted by absolute value.
    """
    dmat = xgb.DMatrix(data_row.reshape(1, -1), feature_names=feature_names)
    contribs = booster.predict(dmat, pred_contribs=True)[0]
    # For multi:softprob, contribs shape = (n_classes, n_features+1)
    # Use class 1 (home win) contributions
    if contribs.ndim == 2:
        contribs_home = contribs[1]  # home win class
    else:
        contribs_home = contribs
    feature_contribs = list(zip(feature_names, contribs_home[:-1]))
    feature_contribs.sort(key=lambda x: abs(x[1]), reverse=True)
    return feature_contribs[:10]


def get_global_feature_importance(booster, top_n=15):
    """Get global feature importance from XGBoost booster."""
    importance = booster.get_score(importance_type='gain')
    sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return pd.DataFrame(sorted_feats, columns=["Feature", "Gain"])


H1_PROB_THRESHOLD = 0.60
H1_SIGMA_THRESHOLD = 0.10


def compute_h1_safety(h1_result, pregame_block):
    """Compute H1 Safety Score (0-5) from 5 independent checks.

    Checks:
      1. Conformal set_size == 1 (model confident)
      2. FG + H1 pick same team (full-game agreement)
      3. H1 XGB + Cat agree (sub-model agreement)
      4. H1 pick prob > 60% (minimum threshold)
      5. Pregame sigma < 0.10 (low uncertainty game)

    Returns:
        (score, checks_dict) where checks_dict has bool per check.
    """
    r = h1_result
    pb = pregame_block or {}

    h1_pick_home = r["h1_prob_home"] >= 0.5
    h1_pick = r["home_team"].split()[-1] if h1_pick_home else r["away_team"].split()[-1]
    h1_prob = r["h1_prob_home"] if h1_pick_home else r["h1_prob_away"]

    fg_pick = pb.get("pick", "—")
    fg_pick_short = fg_pick.split()[-1] if fg_pick != "—" else "—"

    checks = {
        "conformal": r.get("h1_conformal_set_size", 0) == 1,
        "fg_agree": fg_pick_short == h1_pick,
        "h1_models_agree": r.get("h1_models_agree", False),
        "prob_threshold": h1_prob >= H1_PROB_THRESHOLD,
        "low_sigma": pb.get("sigma", 1.0) < H1_SIGMA_THRESHOLD,
    }
    score = sum(checks.values())
    return score, checks


def h1_safety_label(score):
    """Map safety score to label."""
    if score >= 5:
        return "STRONG BET"
    elif score >= 4:
        return "BET"
    elif score >= 3:
        return "LEAN"
    else:
        return "SKIP"


def h1_results_to_dataframe(h1_results, pregame_blocks):
    """Build H1 vs full-game comparison DataFrame."""
    if not h1_results:
        return pd.DataFrame()

    pregame_map = {}
    for b in pregame_blocks:
        key = (b["home"], b["away"])
        pregame_map[key] = b

    rows = []
    for r in h1_results:
        home = r["home_team"]
        away = r["away_team"]
        h1_pick_home = r["h1_prob_home"] >= 0.5
        h1_pick = home.split()[-1] if h1_pick_home else away.split()[-1]
        h1_prob = r["h1_prob_home"] if h1_pick_home else r["h1_prob_away"]

        pb = pregame_map.get((home, away), {})
        fg_pick_full = pb.get("pick", "—")
        fg_pick_short = fg_pick_full.split()[-1] if fg_pick_full != "—" else "—"
        fg_prob = pb.get("pick_prob", 0)

        # Safety score
        score, checks = compute_h1_safety(r, pb)
        label = h1_safety_label(score)

        # EV from H1 odds
        h1_ev = "—"
        h_odds = r.get("h1_ml_home_odds")
        a_odds = r.get("h1_ml_away_odds")
        if h_odds and a_odds:
            from src.core.betting import expected_value as EV
            odds = int(h_odds) if h1_pick_home else int(a_odds)
            h1_ev = round(float(EV.expected_value(h1_prob, odds)), 1)

        rows.append({
            "Game": f"{away.split()[-1]} @ {home.split()[-1]}",
            "FG Pick": fg_pick_short,
            "FG Prob%": round(fg_prob * 100, 1) if fg_prob else "—",
            "H1 Pick": h1_pick,
            "H1 Prob%": round(h1_prob * 100, 1),
            "H1 XGB%": round(r.get("h1_xgb_home", 0) * 100, 1) if h1_pick_home else round((1 - r.get("h1_xgb_home", 1)) * 100, 1),
            "H1 Cat%": round(r.get("h1_cat_home", 0) * 100, 1) if h1_pick_home else round((1 - r.get("h1_cat_home", 1)) * 100, 1),
            "H1 EV": h1_ev,
            "Safety": f"{score}/5",
            "Verdict": label,
            # Store checks for expander detail
            "_checks": checks,
            "_sigma": pb.get("sigma", None),
        })
    return pd.DataFrame(rows)
