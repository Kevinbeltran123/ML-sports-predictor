"""
GameCalendar: checks NBA schedule for today's games.

Uses the existing get_live_scoreboard() from live_game_state.py.
"""
from src.sports.nba.features.live_game_state import get_live_scoreboard
from src.config import get_logger

logger = get_logger(__name__)


def get_todays_games() -> list[dict]:
    """Returns today's NBA games from live scoreboard."""
    try:
        return get_live_scoreboard()
    except Exception as e:
        logger.warning("GameCalendar: could not get today's games: %s", e)
        return []


def has_games_today() -> bool:
    return len(get_todays_games()) > 0


def any_game_live() -> bool:
    """True if any game is currently in progress (status == 2)."""
    try:
        games = get_live_scoreboard()
        return any(g.get("status") == 2 for g in games)
    except Exception:
        return False


def all_games_final() -> bool:
    """True if all of today's games are complete (status == 3)."""
    try:
        games = get_live_scoreboard()
        if not games:
            return True
        return all(g.get("status") == 3 for g in games)
    except Exception:
        return False
