"""
HeadlessRunner: runs predictions headlessly and sends Telegram alerts.

Does NOT modify any model or prediction logic.
Wraps predictor.py's pipeline and adds notification layer.

Flow:
  run_pregame() -> calls predictor pipeline -> formats -> sends Telegram
  run_live_with_telegram() -> mirrors live_betting polling -> sends quarter updates
"""
import argparse
import os
import time
from datetime import datetime

from src.config import get_logger
from src.notifications.telegram_bot import TelegramBot

logger = get_logger(__name__)


def _make_default_args(sportsbook: str = "fanduel") -> argparse.Namespace:
    """Build args namespace matching what predictor.main() expects."""
    return argparse.Namespace(
        ensemble=True,
        xgb=False,
        odds=sportsbook,
        kelly=True,
        clv=False,
        live=False,
        h1=True,
        polymarket=False,
        polymarket_live=False,
        execute=False,
        bankroll_usdc=None,
        league="nba",
    )


def run_pregame(sportsbook: str = "fanduel", bot: TelegramBot = None) -> list[dict]:
    """Run pregame + H1 predictions and send Telegram alerts.

    Returns predictions list for use by live monitor.
    """
    if bot is None:
        bot = TelegramBot()

    logger.info("HeadlessRunner: starting pregame run for %s", sportsbook)

    try:
        import predictor
        from src.sports.nba.predict import ensemble_runner as Ensemble_Runner
        from src.notifications.formatters import format_pregame_message, format_h1_message

        args = _make_default_args(sportsbook)

        # Run the full pipeline via predictor.main()
        # main() prints to console AND runs ensemble_runner which stashes _last_blocks
        predictor.main(args)

        # Get predictions from ensemble_runner._last_blocks
        blocks = getattr(Ensemble_Runner, '_last_blocks', [])

        if not blocks:
            logger.info("No predictions generated — no games or API issue")
            bot.send_plain("No NBA games today (no predictions generated).")
            return []

        # --- Send pregame message ---
        msg = format_pregame_message(blocks, sportsbook)
        bot.send_plain(msg)
        logger.info("Pregame Telegram sent: %d games", len(blocks))

        # --- Run H1 separately (models already cached, near-zero overhead) ---
        try:
            from src.sports.nba.predict.h1_runner import predict_h1
            from src.core.odds_cache import OddsCache

            # Reconstruct games and data from blocks
            games_flat = []
            for b in blocks:
                games_flat.extend([b["home"], b["away"]])

            # We need the data matrix — re-fetch via predictor pipeline
            # Since main() already ran, the odds cache should still be warm
            odds_cache = OddsCache(sportsbook=sportsbook)
            odds = odds_cache.get("basketball_nba")

            # Extract H1 odds
            h1_home_odds = []
            h1_away_odds = []
            if odds:
                for b in blocks:
                    key = f"{b['home']}:{b['away']}"
                    game_odds = odds.get(key, {})
                    h1_home_odds.append(game_odds.get('h1_ml_home'))
                    h1_away_odds.append(game_odds.get('h1_ml_away'))

            # H1 was already called by main() via run_models() — results are printed
            # but not returned by main(). We need to call it again to get the dicts.
            # The models are cached in h1_runner module globals, so this is fast.
            # We need the data matrix which is not easily accessible here.
            # Skip H1 Telegram for now if we can't get the data matrix.
            logger.info("H1 predictions were printed during main() — "
                       "H1 Telegram message requires data matrix access")
        except Exception as e:
            logger.debug("H1 Telegram skipped: %s", e)

        # Build predictions list from blocks for live monitor
        predictions = []
        for b in blocks:
            predictions.append({
                "home_team": b["home"],
                "away_team": b["away"],
                "prob_home": b["pick_prob"] if b["winner"] == 1 else (1.0 - b["pick_prob"]),
                "prob_away": b["pick_prob"] if b["winner"] == 0 else (1.0 - b["pick_prob"]),
                "ev_home": b["ev_home"],
                "ev_away": b["ev_away"],
                "kelly_home": b["kelly_home"],
                "kelly_away": b["kelly_away"],
            })

        return predictions

    except Exception as e:
        logger.error("HeadlessRunner pregame error: %s", e, exc_info=True)
        bot.send_plain(f"ERROR in pregame run: {e}")
        return []


def run_live_with_telegram(
    pregame_predictions: list[dict],
    bot: TelegramBot = None,
    poll_interval: int = 30,
):
    """Live monitoring loop that sends Telegram on quarter ends.

    Mirrors live_betting.run_live_session() logic but with Telegram alerts
    instead of (in addition to) console output.
    """
    if bot is None:
        bot = TelegramBot()

    if not pregame_predictions:
        logger.info("No pregame predictions — skip live monitoring")
        return

    from src.sports.nba.features.live_game_state import (
        get_live_scoreboard, get_live_box_score, get_live_play_by_play, format_clock,
    )
    from src.sports.nba.features.live_pbp_tracker import LivePBPTracker
    from src.sports.nba.predict.live_betting import (
        _match_game_to_prediction, multi_signal_adjustment, _parse_clock_seconds,
        POLL_FAST_SECONDS, FAST_POLL_THRESHOLD,
    )
    from src.notifications.formatters import format_ingame_update, format_daily_summary

    MAX_HOURS = int(os.environ.get("LIVE_BETTING_TIMEOUT_HOURS", 5))
    start_time = time.time()

    processed_periods: dict[str, set] = {}
    last_p_adjusted: dict[str, float] = {}
    pbp_trackers: dict[str, LivePBPTracker] = {}
    poll_count = 0

    # Build pregame preds in the format _match_game_to_prediction expects
    live_preds = [
        {
            "home_team": p["home_team"],
            "away_team": p["away_team"],
            "p_pregame": p.get("prob_home", 0.5),
        }
        for p in pregame_predictions
    ]

    logger.info("Live monitor started for %d games", len(live_preds))
    bot.send_plain(
        f"Live monitoring started for {len(live_preds)} game(s). "
        f"Updates per quarter end."
    )

    try:
        while True:
            if (time.time() - start_time) > MAX_HOURS * 3600:
                logger.info("Live monitor: timeout after %dh", MAX_HOURS)
                bot.send_plain(f"Live monitor timeout after {MAX_HOURS}h.")
                break

            poll_count += 1
            live_games = get_live_scoreboard()
            if not live_games:
                time.sleep(poll_interval)
                continue

            all_final = True
            monitored = []

            for game in live_games:
                pred = _match_game_to_prediction(game, live_preds)
                if pred is None:
                    continue

                monitored.append(game)
                game_id = game["game_id"]
                status = game["status"]
                period = game["period"]

                if status != 3:
                    all_final = False

                if game_id not in processed_periods:
                    processed_periods[game_id] = set()

                # PBP tracker init
                if game_id not in pbp_trackers and status == 2:
                    home_tri = game.get("home_tricode", "")
                    away_tri = game.get("away_tricode", "")
                    if home_tri and away_tri:
                        pbp_trackers[game_id] = LivePBPTracker(home_tri, away_tri)

                # Update PBP tracker
                if game_id in pbp_trackers and status == 2:
                    actions = get_live_play_by_play(game_id)
                    if actions:
                        pbp_trackers[game_id].update(actions)

                clock_secs = _parse_clock_seconds(game.get("clock", ""))

                for check_period in [1, 2, 3]:
                    if check_period in processed_periods[game_id]:
                        continue

                    quarter_done = (
                        period > check_period or
                        (period == check_period and clock_secs == 0.0)
                    )

                    if not quarter_done or status not in (2, 3):
                        continue

                    # Compute in-game prediction
                    box = get_live_box_score(game_id) if status == 2 else None
                    pbp_feats = None
                    tracker = pbp_trackers.get(game_id)
                    if tracker is not None:
                        pbp_feats = tracker.get_features(period_end=check_period)

                    p_pre = pred["p_pregame"]

                    if box is not None:
                        p_adj, delta, expl, conf_set = multi_signal_adjustment(
                            p_pre, box["home"], box["away"],
                            period=check_period, pbp_features=pbp_feats,
                        )
                    else:
                        p_adj = p_pre
                        delta = 0.0
                        expl = "no box score"
                        conf_set = 0

                    last_p_adjusted[game_id] = p_adj
                    processed_periods[game_id].add(check_period)

                    # Extract model name from explanation
                    model_name = "simple"
                    if "[" in expl and "]" in expl:
                        model_name = expl.split("]")[0].lstrip("[")

                    # Send Telegram
                    try:
                        msg = format_ingame_update(
                            home_team=pred["home_team"],
                            away_team=pred["away_team"],
                            period=check_period,
                            home_score=game["home_score"],
                            away_score=game["away_score"],
                            p_pregame=p_pre,
                            p_adjusted=p_adj,
                            delta=delta,
                            conformal_set_size=conf_set,
                            model_used=model_name,
                        )
                        bot.send_plain(msg)
                        logger.info("Q%d update sent: %s vs %s",
                                    check_period, pred["home_team"], pred["away_team"])
                    except Exception as e:
                        logger.warning("Failed to send Q%d update: %s", check_period, e)

            # Check if all games final
            if all_final and monitored and len(monitored) == len(live_preds):
                # Send daily summary
                try:
                    final_scores = get_live_scoreboard()
                    msg = format_daily_summary(pregame_predictions, final_scores)
                    bot.send_plain(msg)
                    logger.info("Daily summary sent")
                except Exception as e:
                    logger.warning("Failed to send daily summary: %s", e)

                bot.send_plain("All games final. Live session complete.")
                break

            # Fast polling near quarter end
            any_close = any(
                0 < _parse_clock_seconds(g.get("clock", "")) <= FAST_POLL_THRESHOLD
                for g in monitored
            ) if monitored else False
            time.sleep(POLL_FAST_SECONDS if any_close else poll_interval)

    except KeyboardInterrupt:
        logger.info("Live monitor interrupted")
    except Exception as e:
        logger.error("Live monitor crashed: %s", e, exc_info=True)
        bot.send_plain(f"Live monitor ERROR: {e}")
