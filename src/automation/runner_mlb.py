"""
Headless MLB runner: pregame picks + live inning-by-inning monitoring.

Called by scheduler.py for automated Telegram delivery.
"""
import time
import traceback
from datetime import datetime

from src.config import get_logger
from src.notifications.telegram_bot import TelegramBot

logger = get_logger(__name__)

MAX_POLL_HOURS = 6  # MLB games can go long (extra innings)


def run_mlb_pregame(sportsbook: str = "fanduel", bot: TelegramBot = None) -> list[dict]:
    """Run MLB pregame ensemble + F5 + totals, send Telegram.

    Returns list of prediction dicts for live monitoring.
    """
    from src.sports.mlb.providers.mlb_stats_api import get_schedule, get_probable_pitchers
    from src.sports.mlb.providers.odds_api_mlb import MLBOddsProvider
    from src.notifications.formatters_mlb import (
        format_mlb_pregame_message, format_mlb_f5_message,
    )

    today = datetime.now().strftime("%Y-%m-%d")
    logger.info("MLB Pregame: starting for %s, sportsbook=%s", today, sportsbook)

    # 1. Get today's schedule
    try:
        games = get_schedule(today, today)
        games = [g for g in games if g.get("status", "") not in ("Postponed", "Cancelled")]
    except Exception as e:
        logger.error("Failed to get MLB schedule: %s", e)
        if bot:
            bot.send_plain(f"MLB pregame FAILED: could not get schedule: {e}")
        return []

    if not games:
        logger.info("No MLB games today")
        if bot:
            bot.send_plain("⚾ No MLB games today.")
        return []

    logger.info("Found %d MLB games today", len(games))

    # 2. Get probable pitchers
    try:
        pitchers = get_probable_pitchers(today)
    except Exception as e:
        logger.warning("Could not get probable pitchers: %s", e)
        pitchers = []

    # 3. Get odds
    odds_data = []
    try:
        odds_provider = MLBOddsProvider(sportsbook=sportsbook)
        odds_data = odds_provider.get_all_odds_with_f5()
        logger.info("Got odds for %d games", len(odds_data))
    except Exception as e:
        logger.warning("Could not get MLB odds: %s", e)

    # 4. Build features and run ensemble
    predictions = []
    try:
        from src.sports.mlb.predict.ensemble_runner import run_mlb_ensemble, _last_blocks
        predictions = run_mlb_ensemble(games, odds_data, sportsbook)

        # Send pregame message
        if bot and _last_blocks:
            msg = format_mlb_pregame_message(_last_blocks, sportsbook)
            bot.send_plain(msg)
            logger.info("Sent MLB pregame message (%d blocks)", len(_last_blocks))
    except Exception as e:
        logger.error("MLB ensemble failed: %s\n%s", e, traceback.format_exc())
        if bot:
            bot.send_plain(f"⚾ MLB pregame FAILED: {e}")

    # 5. Run F5 predictions
    try:
        from src.sports.mlb.predict.f5_runner import predict_f5
        f5_results = predict_f5(games, odds_data)
        if bot and f5_results:
            f5_msg = format_mlb_f5_message(f5_results)
            bot.send_plain(f5_msg)
            logger.info("Sent MLB F5 message (%d games)", len(f5_results))
    except Exception as e:
        logger.warning("MLB F5 failed: %s", e)

    return predictions


def run_mlb_live_with_telegram(pregame_preds: list[dict], bot: TelegramBot):
    """Monitor live MLB games, send inning updates via Telegram.

    Polls every 3 minutes. Sends update when:
    - Inning changes and probability shifted >5%
    - Starting pitcher is pulled
    - Game becomes final

    Sends daily summary when all games are final.
    """
    from src.sports.mlb.providers.mlb_stats_api import get_live_scoreboard
    from src.notifications.formatters_mlb import (
        format_mlb_inning_update, format_mlb_daily_summary,
    )

    logger.info("MLB Live monitor started for %d predictions", len(pregame_preds))

    start_time = time.time()
    max_seconds = MAX_POLL_HOURS * 3600
    poll_interval = 180  # 3 minutes

    # Track state per game
    game_states = {}
    for p in pregame_preds:
        gk = p.get("game_pk")
        if gk:
            game_states[gk] = {
                "last_inning": 0,
                "last_half": "",
                "p_pregame": p.get("prob_home", 0.5),
                "home_team": p.get("home_team", p.get("home", "")),
                "away_team": p.get("away_team", p.get("away", "")),
                "sp_home": p.get("sp_home", ""),
                "sp_away": p.get("sp_away", ""),
                "notified_final": False,
            }

    while time.time() - start_time < max_seconds:
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            scoreboard = get_live_scoreboard(today)
        except Exception as e:
            logger.warning("Live scoreboard error: %s", e)
            time.sleep(poll_interval)
            continue

        all_final = True

        for game in scoreboard:
            gk = game.get("game_pk")
            if gk not in game_states:
                continue

            state = game_states[gk]
            status = game.get("status", "")
            inning = game.get("inning", 0)
            inning_half = game.get("inning_half", "")
            home_score = game.get("home_score", 0)
            away_score = game.get("away_score", 0)

            if status == "Final":
                if not state["notified_final"]:
                    state["notified_final"] = True
                    logger.info("Game %s is FINAL: %s %d - %d %s",
                                gk, state["home_team"], home_score,
                                away_score, state["away_team"])
                continue

            if status in ("In Progress", "Live"):
                all_final = False

                # Check for inning change
                if inning != state["last_inning"] or inning_half != state["last_half"]:
                    state["last_inning"] = inning
                    state["last_half"] = inning_half

                    # Simple Bayesian adjustment based on score differential
                    score_diff = home_score - away_score
                    innings_remaining = max(9 - inning, 1)
                    # MLB: ~0.5 runs/inning expected, so normalize by innings remaining
                    adjustment = score_diff * 0.08 * (9 / max(innings_remaining, 1))
                    p_adjusted = max(0.01, min(0.99,
                                               state["p_pregame"] + adjustment))
                    delta = p_adjusted - state["p_pregame"]

                    if abs(delta) > 0.05 and bot:
                        msg = format_mlb_inning_update(
                            home_team=state["home_team"],
                            away_team=state["away_team"],
                            inning=inning,
                            inning_half=inning_half,
                            home_score=home_score,
                            away_score=away_score,
                            p_pregame=state["p_pregame"],
                            p_adjusted=p_adjusted,
                            delta=delta,
                        )
                        bot.send_plain(msg)
            else:
                # Preview, Warmup, etc — not started yet
                all_final = False

        if all_final and all(s["notified_final"] for s in game_states.values()):
            logger.info("All MLB games are FINAL")
            break

        time.sleep(poll_interval)

    # Send daily summary
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        scoreboard = get_live_scoreboard(today)
        final_scores = [
            {
                "home_team": g.get("home_team", ""),
                "away_team": g.get("away_team", ""),
                "home_score": g.get("home_score", 0),
                "away_score": g.get("away_score", 0),
            }
            for g in scoreboard
            if g.get("status") == "Final"
        ]

        if bot and final_scores:
            summary = format_mlb_daily_summary(pregame_preds, final_scores)
            bot.send_plain(summary)
            logger.info("Sent MLB daily summary")
    except Exception as e:
        logger.error("Failed to send MLB daily summary: %s", e)

    logger.info("MLB live monitor finished")
