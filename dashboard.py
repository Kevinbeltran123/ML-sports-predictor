"""NBA W/L Predictor — Streamlit Dashboard.

Displays pregame, H1, and live in-game predictions with interpretability.
Single Odds API call per hour (cached). Live data from free NBA CDN.

Usage:
    PYTHONPATH=. streamlit run dashboard.py
"""

import os
import sys
import math

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="NBA W/L Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports from existing pipeline ──────────────────────────────────────
from src.core.odds_cache import OddsCache
from src.sports.nba.predict import xgboost_runner as XGBoost_Runner
from src.sports.nba.predict import ensemble_runner as Ensemble_Runner
from src.sports.nba.predict.ensemble_runner import (
    _generate_all_predictions,
    _build_game_blocks,
    _load_ensemble_conformal,
    _load_variance_model,
    _predict_sigmas,
)
from src.sports.nba.predict.margin_runner import predict_margins, predict_margin_sigma
from src.dashboard_helpers import (
    blocks_to_dataframe,
    compute_tag,
    get_game_contributions,
    get_global_feature_importance,
    h1_results_to_dataframe,
)

# ── Caching: models loaded once per process ─────────────────────────────

@st.cache_resource
def load_models():
    """Load all ML models once."""
    XGBoost_Runner._load_models()
    Ensemble_Runner._load_catboost()
    _load_ensemble_conformal()
    _load_variance_model()
    return True


@st.cache_data(ttl=3600, show_spinner="Fetching odds (1 API call)...")
def fetch_odds(sportsbook):
    """Single Odds API call, cached 1 hour."""
    cache = OddsCache(sportsbook=sportsbook)
    odds = cache.get("basketball_nba")
    # Get quota info
    try:
        from src.sports.nba.providers.odds_api import OddsApiProvider
        provider = OddsApiProvider(sportsbook=sportsbook, api_key=os.environ.get("ODDS_API_KEY", ""))
        quota = provider.get_quota()
    except Exception:
        quota = {"remaining": "?", "used": "?"}
    return odds, quota


@st.cache_data(ttl=600, show_spinner="Building features...")
def build_prediction_context(_odds_hash, sportsbook):
    """Run full feature pipeline. Returns all data needed for predictions."""
    from datetime import datetime
    from predictor import (
        resolve_games, get_json_data, to_data_frame, load_schedule,
        build_all_lookups, create_todays_games_data,
        DATA_URL,
    )

    odds = st.session_state.get("odds_data")
    if odds is None:
        return None

    games, odds = resolve_games(odds, sportsbook)
    if games is None:
        return None

    stats_json = get_json_data(DATA_URL)
    df = to_data_frame(stats_json)
    schedule_df = load_schedule()
    today = datetime.today()

    lookups = build_all_lookups(games)

    data, todays_games_uo, frame_ml, home_team_odds, away_team_odds, market_info, spread_home_odds, spread_away_odds, _data_margin = create_todays_games_data(
        games, df, odds, schedule_df, today,
        lookups["game_logs"], lookups["elo_ratings"], lookups["split_data"],
        lookups["team_availability"], lookups["team_schedule"],
        travel_schedule=lookups["travel_schedule"], sos_lookup=lookups["sos_lookup"],
        srs_lookup=lookups["srs_lookup"], lineup_lookup=lookups["lineup_lookup"],
        ref_assignments=lookups["ref_assignments"], ref_history=lookups["ref_history"],
        ff_lookup=lookups["ff_lookup"],
    )

    return {
        "data": data,
        "todays_games_uo": todays_games_uo,
        "frame_ml": frame_ml,
        "games": games,
        "home_team_odds": home_team_odds,
        "away_team_odds": away_team_odds,
        "market_info": market_info,
        "spread_home_odds": spread_home_odds,
        "spread_away_odds": spread_away_odds,
        "odds": odds,
    }


# ── Sidebar ─────────────────────────────────────────────────────────────

st.sidebar.title("NBA W/L Predictor")

sportsbook = st.sidebar.selectbox(
    "Sportsbook",
    ["fanduel", "draftkings", "betmgm", "pointsbet", "caesars", "wynn"],
    index=0,
)

if st.sidebar.button("Refresh Odds (uses API credits)"):
    fetch_odds.clear()
    build_prediction_context.clear()
    st.rerun()

# Load models
load_models()

# Fetch odds (cached 1hr)
odds_data, quota = fetch_odds(sportsbook)
st.session_state["odds_data"] = odds_data

# Show quota
remaining = quota.get("remaining", "?")
st.sidebar.metric("API Credits Remaining", remaining)

if odds_data is None:
    st.error("No odds data available. Check your ODDS_API_KEY in .env")
    st.stop()

# Build predictions (cached 10min)
odds_hash = hash(str(sorted(odds_data.keys()))) if odds_data else 0
ctx = build_prediction_context(odds_hash, sportsbook)

if ctx is None:
    st.warning("No games found for today.")
    st.stop()

# ── Generate predictions ────────────────────────────────────────────────

@st.cache_data(ttl=600)
def run_pregame_predictions(_ctx_hash):
    """Run ensemble predictions, return blocks."""
    ml_probs, ou_probs, xgb_ml_probs, cat_ml_probs, predicted_totals = _generate_all_predictions(
        ctx["data"], ctx["todays_games_uo"], ctx["frame_ml"]
    )

    conformal = _load_ensemble_conformal(sportsbook=sportsbook)
    conformal_set_sizes = None
    conformal_margins = None
    if conformal and conformal is not False:
        conformal_set_sizes, conformal_margins = conformal.predict_confidence(ml_probs)

    variance_stats = _load_variance_model()
    sigmas = None
    if variance_stats:
        sigmas = _predict_sigmas(variance_stats, ctx["data"], ml_probs, xgb_ml_probs, cat_ml_probs)

    reg_margins = predict_margins(ctx["data"])
    spreads = ctx["market_info"].get("MARKET_SPREAD", np.zeros(len(ctx["games"])))
    reg_sigmas = predict_margin_sigma(spreads) if reg_margins is not None else None

    blocks = _build_game_blocks(
        ctx["games"], ml_probs, ou_probs, xgb_ml_probs, cat_ml_probs,
        ctx["todays_games_uo"], ctx["home_team_odds"], ctx["away_team_odds"],
        kelly_flag=True, market_info=ctx["market_info"],
        spread_home_odds=ctx["spread_home_odds"],
        spread_away_odds=ctx["spread_away_odds"],
        conformal_set_sizes=conformal_set_sizes,
        conformal_margins=conformal_margins,
        sigmas=sigmas,
        reg_margins=reg_margins,
        reg_sigmas=reg_sigmas,
        predicted_totals=predicted_totals,
    )

    return blocks, ml_probs, xgb_ml_probs, cat_ml_probs


ctx_hash = hash(str(ctx["games"]))
blocks, ml_probs, xgb_ml_probs, cat_ml_probs = run_pregame_predictions(ctx_hash)

# ── Tabs ────────────────────────────────────────────────────────────────

tab_pregame, tab_h1, tab_live = st.tabs(["Pregame Picks", "First Half (H1)", "Live In-Game"])


# ════════════════════════════════════════════════════════════════════════
# TAB 1: PREGAME
# ════════════════════════════════════════════════════════════════════════
with tab_pregame:
    st.header("Pregame Ensemble Predictions")

    # Summary metrics
    n_bet = sum(1 for b in blocks if compute_tag(b) == "BET")
    n_total = len(blocks)
    best_ev = max(b["max_ev"] for b in blocks) if blocks else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Games", n_total)
    col2.metric("BET signals", n_bet)
    col3.metric("Best EV", f"{best_ev:+.1f}%")
    col4.metric("Model", "XGB 95% / Cat 5%")

    # Color-coded table
    df_picks = blocks_to_dataframe(blocks)

    def _color_tag(val):
        colors = {"BET": "#1a7a1a", "SKIP": "#b8860b", "TRAP": "#b8860b", "PASS": "#8b0000"}
        c = colors.get(val, "#333")
        return f"background-color: {c}; color: white; font-weight: bold"

    def _color_ev(val):
        try:
            v = float(val)
            return f"color: {'#2ecc40' if v > 0 else '#ff4136'}"
        except (ValueError, TypeError):
            return ""

    def _color_sigma(val):
        try:
            v = float(val)
            if v < 0.07:
                return "color: #2ecc40"
            elif v < 0.10:
                return "color: #ffdc00"
            return "color: #ff4136"
        except (ValueError, TypeError):
            return ""

    styled = df_picks.style.map(_color_tag, subset=["Tag"]) \
                            .map(_color_ev, subset=["EV", "AH EV"]) \
                            .map(_color_sigma, subset=["Sigma"]) \
                            .format(precision=1)

    st.dataframe(styled, use_container_width=True, hide_index=True, height=min(400, 40 * len(df_picks) + 40))

    # ── Per-game detail expanders ──
    st.subheader("Game Details")

    for rank, b in enumerate(blocks, 1):
        tag = compute_tag(b)
        tag_emoji = {"BET": "🟢", "SKIP": "🟡", "TRAP": "🟡", "PASS": "🔴"}.get(tag, "⚪")

        with st.expander(f"{tag_emoji} #{rank} {b['away']} @ {b['home']} — {tag} | EV {b['max_ev']:+.1f}%"):
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("**Moneyline Prediction**")
                st.markdown(f"- **Pick:** {b['pick']} ({b['pick_prob']*100:.1f}%)")
                agree_icon = "✅" if b["xgb_agree"] else "⚠️"
                st.markdown(f"- **XGBoost:** {b['xgb_conf']}% | **CatBoost:** {b['cat_conf']}% {agree_icon}")
                st.markdown(f"- **EV:** {b['max_ev']:+.1f}% | **Kelly:** {max(b['kelly_home'], b['kelly_away']):.2f}%")
                st.markdown(f"- **Sigma:** {b['sigma']:.3f}")
                if b.get("conf_ss") is not None:
                    st.markdown(f"- **Conformal:** set_size={b['conf_ss']}, margin={b.get('conf_margin', 0):.2f}")

            with c2:
                st.markdown("**Spread & O/U**")
                if b.get("ah_side"):
                    st.markdown(f"- **AH:** {b['ah_side']} ({b['ah_line']}) P={b['ah_p']:.1%} EV={b['ah_ev']:+.1f}%")
                st.markdown(f"- **O/U:** {b['ou_label']} {b['ou_line']} ({b['ou_conf']}%)")
                if b.get("predicted_total"):
                    st.markdown(f"- **Predicted Total:** {b['predicted_total']:.0f}")
                if b.get("reg_margin") is not None:
                    st.markdown(f"- **Margin (clf):** {b['margin']:+.1f} | **(reg):** {b['reg_margin']:+.1f}")
                if b.get("trap_home") or b.get("trap_away"):
                    st.warning("Underdog EV+ trap detected — consider AH favorite instead")

            # Model probability comparison chart
            idx = b["idx"]
            chart_data = pd.DataFrame({
                "Model": ["XGBoost", "CatBoost", "Ensemble", "Market"],
                "P(Home)": [
                    float(xgb_ml_probs[idx][1]) * 100,
                    float(cat_ml_probs[idx][1]) * 100,
                    float(ml_probs[idx][1]) * 100,
                    float(ctx["market_info"].get("MARKET_ML_PROB", np.full(len(blocks), 50))[idx]) * 100,
                ],
            })
            st.bar_chart(chart_data, x="Model", y="P(Home)", horizontal=True, height=180)

            # Per-game feature contributions
            try:
                booster = XGBoost_Runner.xgb_ml
                if booster is not None:
                    feature_names = list(ctx["frame_ml"].columns) if hasattr(ctx["frame_ml"], "columns") else None
                    if feature_names:
                        contribs = get_game_contributions(booster, ctx["data"][idx], feature_names)
                        top_5 = contribs[:5]
                        contrib_df = pd.DataFrame(top_5, columns=["Feature", "Contribution"])
                        st.markdown("**Top 5 Feature Contributions** (toward home win)")
                        st.bar_chart(contrib_df, x="Feature", y="Contribution", horizontal=True, height=200)
            except Exception:
                pass

    # Global feature importance
    with st.expander("Global Feature Importance (XGBoost)"):
        try:
            booster = XGBoost_Runner.xgb_ml
            if booster:
                imp_df = get_global_feature_importance(booster, top_n=20)
                st.bar_chart(imp_df, x="Feature", y="Gain", horizontal=True, height=500)
        except Exception as e:
            st.error(f"Could not load feature importance: {e}")


# ════════════════════════════════════════════════════════════════════════
# TAB 2: FIRST HALF (H1)
# ════════════════════════════════════════════════════════════════════════
with tab_h1:
    st.header("First Half (H1) Moneyline Predictions")

    try:
        from src.sports.nba.predict.h1_runner import predict_h1, _load_h1_models

        if _load_h1_models():
            games_flat = [team for pair in ctx["games"] for team in pair]
            h1_home_odds = []
            h1_away_odds = []
            odds = ctx["odds"]
            if odds:
                for home, away in ctx["games"]:
                    key = f"{home}:{away}"
                    game_odds = odds.get(key, {})
                    h1_home_odds.append(game_odds.get('h1_ml_home'))
                    h1_away_odds.append(game_odds.get('h1_ml_away'))

            # Suppress print output from predict_h1
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                h1_results = predict_h1(
                    ctx["data"], games_flat,
                    h1_odds_home=h1_home_odds or None,
                    h1_odds_away=h1_away_odds or None,
                    kelly_flag=True,
                )

            if h1_results:
                from src.dashboard_helpers import compute_h1_safety, h1_safety_label
                h1_df = h1_results_to_dataframe(h1_results, blocks)

                # Display columns (hide internal _checks, _sigma)
                display_cols = [c for c in h1_df.columns if not c.startswith("_")]

                # Color verdict
                def _color_verdict(val):
                    if val == "STRONG BET":
                        return "background-color: #0d6e0d; color: white; font-weight: bold"
                    elif val == "BET":
                        return "background-color: #1a7a1a; color: white; font-weight: bold"
                    elif val == "LEAN":
                        return "background-color: #7a7a1a; color: white"
                    else:
                        return "background-color: #7a1a1a; color: white"

                styled_h1 = h1_df[display_cols].style.map(_color_verdict, subset=["Verdict"]).format(
                    {"FG Prob%": "{:.1f}", "H1 Prob%": "{:.1f}", "H1 XGB%": "{:.1f}", "H1 Cat%": "{:.1f}"},
                    na_rep="—"
                )
                st.dataframe(styled_h1, use_container_width=True, hide_index=True)

                # Safety checklist per game
                st.subheader("Safety Checklist")
                check_labels = {
                    "conformal": "Conformal set_size = 1",
                    "fg_agree": "FG + H1 mismo equipo",
                    "h1_models_agree": "H1 XGB + Cat acuerdan",
                    "prob_threshold": "H1 Prob > 60%",
                    "low_sigma": "Sigma pregame < 0.10",
                }
                for idx, row in h1_df.iterrows():
                    checks = row["_checks"]
                    score_str = row["Safety"]
                    verdict = row["Verdict"]
                    game = row["Game"]
                    pick = row["H1 Pick"]

                    with st.expander(f"{game} → 1H {pick} | {score_str} **{verdict}**"):
                        for key, label in check_labels.items():
                            passed = checks.get(key, False)
                            icon = "✅" if passed else "❌"
                            st.markdown(f"{icon} {label}")
                        sigma_val = row["_sigma"]
                        if sigma_val is not None:
                            st.caption(f"σ = {sigma_val:.3f}")

                # Summary metrics
                col_s1, col_s2, col_s3 = st.columns(3)
                n_strong = (h1_df["Verdict"] == "STRONG BET").sum()
                n_bet = (h1_df["Verdict"] == "BET").sum()
                n_lean = (h1_df["Verdict"] == "LEAN").sum()
                col_s1.metric("STRONG BET", n_strong)
                col_s2.metric("BET", n_bet)
                col_s3.metric("LEAN", n_lean)

                # ── Manual H1 Odds Input for EV/Kelly ──
                st.subheader("H1 Odds Calculator")
                st.caption("Ingresa las cuotas 1H (decimales) de tu sportsbook para calcular EV y Kelly.")

                from src.core.betting import expected_value as EV_mod
                from src.core.betting import kelly_criterion as kc_mod

                def _decimal_to_american(dec_odds):
                    """Convert decimal odds to american odds."""
                    if dec_odds >= 2.0:
                        return int(round((dec_odds - 1) * 100))
                    else:
                        return int(round(-100 / (dec_odds - 1)))

                # Build pregame map for safety score lookup
                _pregame_map = {}
                for b in blocks:
                    _pregame_map[(b["home"], b["away"])] = b

                h1_calc_rows = []
                for _, r in enumerate(h1_results):
                    home = r["home_team"]
                    away = r["away_team"]
                    h1_pick_home = r["h1_prob_home"] >= 0.5
                    pick_name = home.split()[-1] if h1_pick_home else away.split()[-1]
                    pick_prob = r["h1_prob_home"] if h1_pick_home else r["h1_prob_away"]

                    pb = _pregame_map.get((home, away), {})
                    safety_score, _ = compute_h1_safety(r, pb)
                    safety_label = h1_safety_label(safety_score)

                    game_label = f"{away.split()[-1]} @ {home.split()[-1]}"
                    col_game, col_odds, col_result = st.columns([2, 1, 2])

                    with col_game:
                        st.markdown(f"**{game_label}** → 1H {pick_name} ({pick_prob:.1%}) [{safety_score}/5 {safety_label}]")

                    with col_odds:
                        odds_input = st.number_input(
                            f"1H ML {pick_name}",
                            value=0.00, step=0.01, format="%.2f",
                            key=f"h1_odds_{home}_{away}",
                            help="Cuota decimal (ej: 1.80, 2.10). Deja en 0 para skip.",
                        )

                    with col_result:
                        if odds_input > 1.0:
                            american = _decimal_to_american(odds_input)
                            ev = float(EV_mod.expected_value(pick_prob, american))
                            kelly = float(kc_mod.calculate_eighth_kelly(american, pick_prob))

                            ev_color = "🟢" if ev > 0 else "🔴"
                            if ev <= 0:
                                verdict = "PASS (EV-)"
                            elif safety_score >= 4:
                                verdict = "BET"
                            elif safety_score >= 3:
                                verdict = "LEAN"
                            else:
                                verdict = "SKIP"

                            st.markdown(f"{ev_color} **EV: {ev:+.1f}%** | Kelly: {kelly:.2f}% | **{verdict}**")

                            h1_calc_rows.append({
                                "Game": game_label, "Pick": f"1H {pick_name}",
                                "Prob%": round(pick_prob * 100, 1), "Odds": odds_input,
                                "EV": round(ev, 1), "Kelly%": round(kelly, 2),
                                "Safety": f"{safety_score}/5", "Verdict": verdict,
                            })
                        else:
                            st.markdown("—")

                if h1_calc_rows:
                    st.divider()
                    st.subheader("H1 Betting Summary")
                    summary_df = pd.DataFrame(h1_calc_rows)
                    def _color_summary_verdict(val):
                        if val == "BET":
                            return "background-color: #1a7a1a; color: white; font-weight: bold"
                        elif val == "LEAN":
                            return "background-color: #7a7a1a; color: white"
                        elif "PASS" in str(val):
                            return "background-color: #7a1a1a; color: white"
                        elif val == "SKIP":
                            return "background-color: #5a1a1a; color: white"
                        return ""
                    styled_summary = summary_df.style.map(_color_summary_verdict, subset=["Verdict"]).format(
                        {"Prob%": "{:.1f}", "EV": "{:+.1f}", "Kelly%": "{:.2f}"}, na_rep="—"
                    )
                    st.dataframe(styled_summary, use_container_width=True, hide_index=True)

            else:
                st.warning("H1 predictions returned no results.")
        else:
            st.warning("H1 models not found in models/h1moneyline/. Train with: `PYTHONPATH=. python scripts/train_h1_models.py`")
    except Exception as e:
        st.error(f"H1 prediction error: {e}")


# ════════════════════════════════════════════════════════════════════════
# TAB 3: LIVE IN-GAME
# ════════════════════════════════════════════════════════════════════════
with tab_live:
    st.header("Live In-Game Predictions")

    # Initialize session state for probability history
    if "prob_history" not in st.session_state:
        st.session_state["prob_history"] = {}

    @st.fragment(run_every=30)
    def live_panel():
        from src.sports.nba.features.live_game_state import (
            get_live_scoreboard, get_live_box_score, get_live_play_by_play, format_clock,
        )
        from src.sports.nba.features.live_pbp_tracker import LivePBPTracker
        from src.sports.nba.predict.live_betting import bayesian_q1_adjustment

        scoreboard = get_live_scoreboard()
        if not scoreboard:
            st.info("No live games right now. This panel auto-refreshes every 30 seconds.")
            return

        live_games = [g for g in scoreboard if g["status"] == 2]
        scheduled = [g for g in scoreboard if g["status"] == 1]
        finished = [g for g in scoreboard if g["status"] == 3]

        if scheduled:
            st.caption(f"Scheduled: {len(scheduled)} | Live: {len(live_games)} | Final: {len(finished)}")

        if not live_games:
            st.info("No games currently in progress. Showing today's schedule.")
            for g in scoreboard:
                status_label = {1: "🕐 Scheduled", 2: "🔴 Live", 3: "✅ Final"}.get(g["status"], "?")
                score = f"{g['away_score']}-{g['home_score']}" if g["status"] != 1 else ""
                st.markdown(f"- {g['away_team']} @ {g['home_team']} — {status_label} {score}")
            return

        # Match live games to pregame predictions
        for lg in live_games:
            game_id = lg["game_id"]
            home_name = f"{lg.get('home_team_city', '')} {lg['home_team']}".strip()
            away_name = f"{lg.get('away_team_city', '')} {lg['away_team']}".strip()

            # Find pregame prediction for this game
            pregame_prob = 0.5
            pregame_block = None
            for b in blocks:
                if (lg["home_team"] in b["home"] or b["home"] in home_name or
                    lg.get("home_tricode", "???") in b["home"]):
                    pregame_prob = b["pick_prob"] if b["winner"] == 1 else (1 - b["pick_prob"])
                    pregame_block = b
                    break

            # Get live data
            period = lg["period"]
            score_diff = lg["home_score"] - lg["away_score"]
            clock_str = format_clock(lg.get("clock", ""))

            # Get box score for possessions
            box = get_live_box_score(game_id)
            total_poss = 50 * period  # rough estimate
            if box:
                total_poss = max(box["home"]["possessions"] + box["away"]["possessions"], 1) / 2

            # Bayesian adjustment
            p_adjusted, explanation = bayesian_q1_adjustment(
                pregame_prob, score_diff, total_poss, period=period
            )

            delta = (p_adjusted - pregame_prob) * 100

            # Store history
            if game_id not in st.session_state["prob_history"]:
                st.session_state["prob_history"][game_id] = {"Pregame": pregame_prob * 100}
            st.session_state["prob_history"][game_id][f"Q{period}"] = p_adjusted * 100

            # Try in-game XGBoost cascade
            model_used = "Bayesian"
            try:
                from src.sports.nba.predict.ingame_runner import predict_ingame
                actions = get_live_play_by_play(game_id)

                if "pbp_trackers" not in st.session_state:
                    st.session_state["pbp_trackers"] = {}

                if game_id not in st.session_state["pbp_trackers"]:
                    st.session_state["pbp_trackers"][game_id] = LivePBPTracker(
                        home_tricode=lg.get("home_tricode", ""),
                        away_tricode=lg.get("away_tricode", ""),
                    )

                tracker = st.session_state["pbp_trackers"][game_id]
                if actions:
                    tracker.update(actions)

                pbp_feats = tracker.get_features(period_end=period)

                if box and pbp_feats:
                    result = predict_ingame(
                        p_pregame=pregame_prob,
                        box_home=box["home"],
                        box_away=box["away"],
                        period=period,
                        pbp_features=pbp_feats,
                    )
                    if result and result.get("p_home") is not None:
                        p_adjusted = result["p_home"]
                        model_used = result.get("model_used", "XGBoost")
                        delta = (p_adjusted - pregame_prob) * 100
                        st.session_state["prob_history"][game_id][f"Q{period}"] = p_adjusted * 100
            except Exception as e:
                st.caption(f"⚠️ Cascade error: {e}")

            # ── Render game card ──
            st.divider()
            c1, c2, c3 = st.columns([2, 1, 1])

            with c1:
                lead_indicator = "🟢" if score_diff > 0 else ("🔴" if score_diff < 0 else "⚪")
                st.markdown(f"### {away_name} {lg['away_score']} — {lg['home_score']} {home_name} {lead_indicator}")
                st.caption(f"Q{period} {clock_str} | Model: {model_used}")

            with c2:
                delta_color = "normal" if abs(delta) < 5 else ("off" if delta < -5 else "normal")
                st.metric("P(Home Win)", f"{p_adjusted*100:.1f}%", f"{delta:+.1f}pp", delta_color=delta_color)

            with c3:
                st.metric("Pregame", f"{pregame_prob*100:.1f}%")
                st.metric("Score Diff", f"{score_diff:+d}")

            # PBP features summary
            if "pbp_trackers" in st.session_state and game_id in st.session_state["pbp_trackers"]:
                tracker = st.session_state["pbp_trackers"][game_id]
                feats = tracker.get_features(period_end=period)
                if feats:
                    fc1, fc2, fc3, fc4 = st.columns(4)
                    fc1.metric("Lead Changes", feats.get("PBP_LEAD_CHANGES", 0))
                    fc2.metric("Momentum", f"{feats.get('PBP_MOMENTUM', 0):+.2f}")
                    fc3.metric("Largest Lead (H)", feats.get("PBP_LARGEST_LEAD_HOME", 0))
                    fc4.metric("Largest Lead (A)", feats.get("PBP_LARGEST_LEAD_AWAY", 0))

            # Probability evolution chart
            history = st.session_state["prob_history"].get(game_id, {})
            if len(history) > 1:
                hist_df = pd.DataFrame([history]).T
                hist_df.columns = ["P(Home Win) %"]
                st.line_chart(hist_df, height=200)

    live_panel()


# ── Footer ──────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.caption("NBA W/L Predictor | Streamlit Dashboard")
st.sidebar.caption(f"Games today: {len(blocks)}")
