"""
Scheduler: long-running process that orchestrates daily automation.

Jobs:
  1. DAILY at 15:15 COT  -> run_pregame() + send Telegram
  2. Every 5 min from 16:00-23:59 COT -> check if any game went live, start monitor

The live monitor is NOT on a fixed schedule. Instead, a lightweight "game watcher"
polls every 5 minutes. Once it detects status=2 (in progress), it spawns the
full live monitor thread which then runs until all games are final.

Uses APScheduler BackgroundScheduler with Colombia timezone (UTC-5, no DST).
Entry point: python -m src.automation.scheduler
"""
import os
import time
import threading

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

# Module-level state shared between jobs
_pregame_results: list[dict] = []
_live_thread: threading.Thread | None = None
_live_notified = False  # avoid sending "starting" message twice


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


def main():
    """Entry point: start scheduler and block forever."""
    # Load env vars
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    scheduler = BackgroundScheduler(timezone=COLOMBIA)

    # 3:15 PM COT — pregame picks
    scheduler.add_job(
        job_pregame,
        trigger=CronTrigger(hour=15, minute=15, timezone=COLOMBIA),
        id="pregame",
        name="Daily Pregame",
        replace_existing=True,
        misfire_grace_time=300,
    )

    # Every 5 min from 4 PM to midnight COT — watch for games going live
    scheduler.add_job(
        job_game_watcher,
        trigger=CronTrigger(
            hour="16-23", minute="*/5", timezone=COLOMBIA,
        ),
        id="game_watcher",
        name="Game Watcher (every 5 min)",
        replace_existing=True,
        misfire_grace_time=60,
    )

    scheduler.start()
    logger.info("Scheduler started. Pregame@15:15 COT, game watcher every 5 min 16:00-23:59 COT")
    _bot.send_plain(
        "NBA Bot online.\n"
        "• Pregame picks: 3:15 PM COT\n"
        "• Live monitor: auto-starts when games go live"
    )

    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("Scheduler stopped.")
        _bot.send_plain("NBA Bot offline.")


if __name__ == "__main__":
    main()
