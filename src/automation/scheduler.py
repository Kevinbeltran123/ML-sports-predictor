"""
Scheduler: long-running process that orchestrates daily automation.

NBA Jobs:
  1. DAILY at 15:15 COT  -> run_pregame() + send Telegram
  2. Every 5 min from 16:00-23:59 COT -> check if any NBA game went live

MLB Jobs (April-October only):
  3. DAILY at 12:30 COT  -> run_mlb_pregame() + send Telegram
  4. Every 5 min from 12:00-23:00 COT -> check if any MLB game went live

Uses APScheduler BackgroundScheduler with Colombia timezone (UTC-5, no DST).
Entry point: python -m src.automation.scheduler
"""
import os
import time
import threading
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
import pytz

from src.config import get_logger
from src.notifications.telegram_bot import TelegramBot

logger = get_logger(__name__)

COLOMBIA = pytz.timezone("America/Bogota")
SPORTSBOOK = os.environ.get("DEFAULT_SPORTSBOOK", "fanduel")

# Singleton bot (shared across jobs)
_bot = TelegramBot()

# NBA module-level state
_pregame_results: list[dict] = []
_live_thread: threading.Thread | None = None
_live_notified = False

# MLB module-level state
_mlb_pregame_results: list[dict] = []
_mlb_live_thread: threading.Thread | None = None
_mlb_live_notified = False


def _is_mlb_season() -> bool:
    """Check if current month is within MLB season (March-October)."""
    return 3 <= datetime.now().month <= 10


def job_pregame():
    """3:15 PM COT daily: run pregame + H1 + send Telegram."""
    global _pregame_results, _live_notified
    _live_notified = False  # reset for new day
    logger.info("Scheduler: PREGAME job triggered (%s)", SPORTSBOOK)
    try:
        from src.automation.runner import run_pregame
        _pregame_results = run_pregame(sportsbook=SPORTSBOOK, bot=_bot)
        logger.info("Pregame job complete: %d predictions", len(_pregame_results))
    except Exception as e:
        logger.error("Pregame job failed: %s", e, exc_info=True)
        _bot.send_plain(f"Pregame job FAILED: {e}")


def job_game_watcher():
    """Polls every 5 min to detect when games go live, then starts live monitor.

    This replaces the fixed 4:30 PM trigger. It checks if any game has
    status=2 (in progress) and spawns the live monitor thread on first detection.
    Once the live thread is running, this job becomes a no-op until next day.
    """
    global _live_thread, _live_notified

    # Already running — skip
    if _live_thread is not None and _live_thread.is_alive():
        return

    # No predictions loaded — can't monitor
    if not _pregame_results:
        return

    from src.automation.game_calendar import any_game_live, has_games_today

    if not has_games_today():
        return

    if not any_game_live():
        return

    # A game just went live — start the monitor
    logger.info("Scheduler: game detected as LIVE — starting live monitor")

    if not _live_notified:
        _bot.send_plain(
            f"🏀 Game detected live! Starting quarter-by-quarter monitoring "
            f"for {len(_pregame_results)} game(s)."
        )
        _live_notified = True

    from src.automation.runner import run_live_with_telegram

    _live_thread = threading.Thread(
        target=run_live_with_telegram,
        args=(_pregame_results, _bot),
        daemon=True,
        name="live-monitor",
    )
    _live_thread.start()
    logger.info("Live monitor thread started")


def job_mlb_pregame():
    """12:30 PM COT daily (April-October): run MLB pregame + send Telegram."""
    global _mlb_pregame_results, _mlb_live_notified

    if not _is_mlb_season():
        logger.debug("MLB pregame skipped — off-season")
        return

    _mlb_live_notified = False  # reset for new day
    logger.info("Scheduler: MLB PREGAME job triggered (%s)", SPORTSBOOK)
    try:
        from src.automation.runner_mlb import run_mlb_pregame
        _mlb_pregame_results = run_mlb_pregame(sportsbook=SPORTSBOOK, bot=_bot)
        logger.info("MLB pregame job complete: %d predictions", len(_mlb_pregame_results))
    except Exception as e:
        logger.error("MLB pregame job failed: %s", e, exc_info=True)
        _bot.send_plain(f"⚾ MLB Pregame job FAILED: {e}")


def job_mlb_game_watcher():
    """Polls every 5 min to detect when MLB games go live (April-October)."""
    global _mlb_live_thread, _mlb_live_notified

    if not _is_mlb_season():
        return

    # Already running — skip
    if _mlb_live_thread is not None and _mlb_live_thread.is_alive():
        return

    # No predictions loaded — can't monitor
    if not _mlb_pregame_results:
        return

    try:
        from src.sports.mlb.providers.mlb_stats_api import get_schedule
        today_games = get_schedule()
        if not today_games:
            return

        # Check if any game is in progress (statusCode = 'I' or 'F' not started yet)
        any_live = any(
            g.get("status", {}).get("statusCode") == "I"
            for g in today_games
        )
        if not any_live:
            return
    except Exception as e:
        logger.debug("MLB game watcher check failed: %s", e)
        return

    # A game just went live — start the monitor
    logger.info("Scheduler: MLB game detected as LIVE — starting live monitor")

    if not _mlb_live_notified:
        _bot.send_plain(
            f"⚾ MLB game detected live! Starting inning-by-inning monitoring "
            f"for {len(_mlb_pregame_results)} game(s)."
        )
        _mlb_live_notified = True

    from src.automation.runner_mlb import run_mlb_live_with_telegram

    _mlb_live_thread = threading.Thread(
        target=run_mlb_live_with_telegram,
        args=(_mlb_pregame_results, _bot),
        daemon=True,
        name="mlb-live-monitor",
    )
    _mlb_live_thread.start()
    logger.info("MLB live monitor thread started")


def main():
    """Entry point: start scheduler and block forever."""
    # Load env vars
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    scheduler = BackgroundScheduler(timezone=COLOMBIA)

    # ---- NBA Jobs ----
    # 3:15 PM COT — pregame picks
    scheduler.add_job(
        job_pregame,
        trigger=CronTrigger(hour=15, minute=15, timezone=COLOMBIA),
        id="pregame",
        name="NBA Daily Pregame",
        replace_existing=True,
        misfire_grace_time=300,
    )

    # Every 5 min from 4 PM to midnight COT — watch for NBA games going live
    scheduler.add_job(
        job_game_watcher,
        trigger=CronTrigger(
            hour="16-23", minute="*/5", timezone=COLOMBIA,
        ),
        id="game_watcher",
        name="NBA Game Watcher (every 5 min)",
        replace_existing=True,
        misfire_grace_time=60,
    )

    # ---- MLB Jobs (April-October) ----
    # 12:30 PM COT — MLB pregame picks
    scheduler.add_job(
        job_mlb_pregame,
        trigger=CronTrigger(hour=12, minute=30, timezone=COLOMBIA),
        id="mlb_pregame",
        name="MLB Daily Pregame",
        replace_existing=True,
        misfire_grace_time=300,
    )

    # Every 5 min from 12 PM to 11 PM COT — watch for MLB games going live
    scheduler.add_job(
        job_mlb_game_watcher,
        trigger=CronTrigger(
            hour="12-23", minute="*/5", timezone=COLOMBIA,
        ),
        id="mlb_game_watcher",
        name="MLB Game Watcher (every 5 min)",
        replace_existing=True,
        misfire_grace_time=60,
    )

    scheduler.start()
    logger.info(
        "Scheduler started. NBA: Pregame@15:15, watcher 16-23. "
        "MLB: Pregame@12:30, watcher 12-23 (Apr-Oct). TZ=COT"
    )

    mlb_status = "active" if _is_mlb_season() else "off-season"
    _bot.send_plain(
        "Sports Bot online.\n"
        "NBA:\n"
        "  • Pregame picks: 3:15 PM COT\n"
        "  • Live monitor: auto-starts when games go live\n"
        f"MLB ({mlb_status}):\n"
        "  • Pregame picks: 12:30 PM COT\n"
        "  • Live monitor: auto-starts when games go live"
    )

    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("Scheduler stopped.")
        _bot.send_plain("Sports Bot offline.")


if __name__ == "__main__":
    main()
