"""MLB Stats API wrapper using the MLB-StatsAPI package (statsapi).

Free, no authentication required.
Install: pip install MLB-StatsAPI

Uses statsapi library methods where available. For endpoints not covered
by the library, falls back to direct requests against statsapi.mlb.com/api/v1/.
"""

import logging
from datetime import date, datetime
from typing import Optional

import requests

try:
    import statsapi
except ImportError:
    statsapi = None  # graceful degradation; callers should check

logger = logging.getLogger(__name__)

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_statsapi():
    if statsapi is None:
        raise ImportError(
            "MLB-StatsAPI package is not installed. "
            "Run: pip install MLB-StatsAPI"
        )


def _api_get(endpoint: str, params: dict = None) -> dict:
    """Raw GET against the MLB Stats API v1 endpoint. Returns parsed JSON."""
    url = f"{MLB_API_BASE}/{endpoint.lstrip('/')}"
    try:
        resp = requests.get(url, params=params or {}, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as exc:
        logger.error("MLB API request failed [%s]: %s", url, exc)
        return {}


def _parse_date(d: str) -> str:
    """Normalise a date string to MM/DD/YYYY expected by statsapi."""
    try:
        return datetime.strptime(d, "%Y-%m-%d").strftime("%m/%d/%Y")
    except ValueError:
        return d  # already in another format; pass through


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_schedule(start_date: str, end_date: str) -> list[dict]:
    """Get games in a date range.

    Args:
        start_date: YYYY-MM-DD
        end_date:   YYYY-MM-DD

    Returns:
        List of dicts with keys:
          game_pk, game_date, home_team, away_team, status,
          home_score, away_score, home_pitcher, away_pitcher
    """
    _require_statsapi()
    try:
        raw = statsapi.schedule(
            start_date=_parse_date(start_date),
            end_date=_parse_date(end_date),
            sportId=1,
        )
    except Exception as exc:
        logger.error("get_schedule failed: %s", exc)
        return []

    results = []
    for game in raw:
        results.append({
            "game_pk":       game.get("game_id"),
            "game_date":     game.get("game_date"),
            "home_team":     game.get("home_name"),
            "away_team":     game.get("away_name"),
            "status":        game.get("status"),
            "home_score":    game.get("home_score"),
            "away_score":    game.get("away_score"),
            "home_pitcher":  game.get("home_probable_pitcher"),
            "away_pitcher":  game.get("away_probable_pitcher"),
        })
    return results


def get_boxscore(game_pk: int) -> dict:
    """Get full boxscore for a completed or in-progress game.

    Args:
        game_pk: MLB game primary key

    Returns:
        Dict with keys:
          game_pk, home_team, away_team,
          home_batting  (list of per-batter dicts),
          away_batting  (list of per-batter dicts),
          home_pitching (list of per-pitcher dicts),
          away_pitching (list of per-pitcher dicts),
          home_team_stats, away_team_stats
    """
    _require_statsapi()
    try:
        data = statsapi.boxscore_data(game_pk)
    except Exception as exc:
        logger.error("get_boxscore(%d) failed: %s", game_pk, exc)
        return {}

    def _extract_batters(side_data: dict) -> list[dict]:
        batters = []
        for player_id, info in side_data.get("players", {}).items():
            stats = info.get("stats", {}).get("batting", {})
            if not stats:
                continue
            batters.append({
                "player_id":   player_id,
                "name":        info.get("person", {}).get("fullName"),
                "position":    info.get("position", {}).get("abbreviation"),
                "ab":          stats.get("atBats"),
                "hits":        stats.get("hits"),
                "runs":        stats.get("runs"),
                "rbi":         stats.get("rbi"),
                "hr":          stats.get("homeRuns"),
                "bb":          stats.get("baseOnBalls"),
                "so":          stats.get("strikeOuts"),
                "avg":         stats.get("avg"),
            })
        return batters

    def _extract_pitchers(side_data: dict) -> list[dict]:
        pitchers = []
        for player_id, info in side_data.get("players", {}).items():
            stats = info.get("stats", {}).get("pitching", {})
            if not stats:
                continue
            pitchers.append({
                "player_id":    player_id,
                "name":         info.get("person", {}).get("fullName"),
                "ip":           stats.get("inningsPitched"),
                "hits":         stats.get("hits"),
                "runs":         stats.get("runs"),
                "er":           stats.get("earnedRuns"),
                "bb":           stats.get("baseOnBalls"),
                "so":           stats.get("strikeOuts"),
                "hr":           stats.get("homeRuns"),
                "era":          stats.get("era"),
                "pitches":      stats.get("numberOfPitches"),
                "strikes":      stats.get("strikes"),
            })
        return pitchers

    home = data.get("home", {})
    away = data.get("away", {})

    return {
        "game_pk":         game_pk,
        "home_team":       home.get("team", {}).get("name"),
        "away_team":       away.get("team", {}).get("name"),
        "home_batting":    _extract_batters(home),
        "away_batting":    _extract_batters(away),
        "home_pitching":   _extract_pitchers(home),
        "away_pitching":   _extract_pitchers(away),
        "home_team_stats": home.get("teamStats", {}),
        "away_team_stats": away.get("teamStats", {}),
    }


def get_probable_pitchers(date: str) -> list[dict]:
    """Get today's probable starting pitchers.

    Args:
        date: YYYY-MM-DD

    Returns:
        List of dicts with keys:
          game_pk, home_team, away_team,
          home_pitcher_id, home_pitcher_name,
          away_pitcher_id, away_pitcher_name
    """
    _require_statsapi()
    try:
        raw = statsapi.schedule(
            start_date=_parse_date(date),
            end_date=_parse_date(date),
            sportId=1,
        )
    except Exception as exc:
        logger.error("get_probable_pitchers(%s) failed: %s", date, exc)
        return []

    results = []
    for game in raw:
        home_name = game.get("home_probable_pitcher", "")
        away_name = game.get("away_probable_pitcher", "")

        # Resolve pitcher IDs via lookup when names are present
        home_id = _resolve_pitcher_id(home_name)
        away_id = _resolve_pitcher_id(away_name)

        results.append({
            "game_pk":            game.get("game_id"),
            "home_team":          game.get("home_name"),
            "away_team":          game.get("away_name"),
            "home_pitcher_id":    home_id,
            "home_pitcher_name":  home_name,
            "away_pitcher_id":    away_id,
            "away_pitcher_name":  away_name,
        })
    return results


def _resolve_pitcher_id(name: str) -> Optional[int]:
    """Best-effort lookup of a player ID from their full name."""
    if not name:
        return None
    _require_statsapi()
    try:
        results = statsapi.lookup_player(name, sportId=1)
        if results:
            return results[0].get("id")
    except Exception as exc:
        logger.debug("_resolve_pitcher_id('%s') failed: %s", name, exc)
    return None


def get_team_game_logs(team_id: int, season: int) -> list[dict]:
    """Get all game results for a team in a season.

    Args:
        team_id: MLB team ID (see get_all_team_ids())
        season:  four-digit year, e.g. 2024

    Returns:
        List of game dicts (one per game played).
    """
    try:
        data = _api_get(
            "schedule",
            {
                "teamId":   team_id,
                "season":   season,
                "sportId":  1,
                "gameType": "R",  # Regular Season
            },
        )
    except Exception as exc:
        logger.error("get_team_game_logs(%d, %d) failed: %s", team_id, season, exc)
        return []

    results = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            teams  = game.get("teams", {})
            home   = teams.get("home", {})
            away   = teams.get("away", {})
            results.append({
                "game_pk":    game.get("gamePk"),
                "game_date":  game.get("officialDate") or date_entry.get("date"),
                "status":     game.get("status", {}).get("detailedState"),
                "home_team":  home.get("team", {}).get("name"),
                "away_team":  away.get("team", {}).get("name"),
                "home_score": home.get("score"),
                "away_score": away.get("score"),
                "home_win":   home.get("isWinner"),
                "away_win":   away.get("isWinner"),
            })
    return results


def get_live_scoreboard(date: str = None) -> list[dict]:
    """Get live game states for all games on a given date (default: today).

    Args:
        date: YYYY-MM-DD  (optional, defaults to today)

    Returns:
        List of dicts with keys:
          game_pk, home_team, away_team, home_score, away_score,
          inning, inning_half, status
    """
    target_date = date or datetime.utcnow().strftime("%Y-%m-%d")
    try:
        data = _api_get(
            "schedule",
            {
                "sportId":   1,
                "date":      target_date,
                "hydrate":   "linescore",
                "gameType":  "R,P",  # Regular Season + Playoffs
            },
        )
    except Exception as exc:
        logger.error("get_live_scoreboard(%s) failed: %s", target_date, exc)
        return []

    results = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            teams     = game.get("teams", {})
            home      = teams.get("home", {})
            away      = teams.get("away", {})
            linescore = game.get("linescore", {})

            inning      = linescore.get("currentInning")
            inning_half = linescore.get("inningHalf", "")  # "Top" | "Bottom"

            results.append({
                "game_pk":     game.get("gamePk"),
                "home_team":   home.get("team", {}).get("name"),
                "away_team":   away.get("team", {}).get("name"),
                "home_score":  home.get("score"),
                "away_score":  away.get("score"),
                "inning":      inning,
                "inning_half": inning_half,
                "status":      game.get("status", {}).get("detailedState"),
            })
    return results


def get_live_game_feed(game_pk: int) -> dict:
    """Get detailed live game feed for a specific game.

    Returns current pitcher, recent plays, base runner state, and counts.

    Args:
        game_pk: MLB game primary key

    Returns:
        Dict with keys:
          game_pk, status, inning, inning_half,
          home_score, away_score,
          current_pitcher (dict),
          base_runners (list),
          recent_plays (list of last 5 play descriptions),
          balls, strikes, outs
    """
    _require_statsapi()
    try:
        data = statsapi.get("game", {"gamePk": game_pk})
    except Exception as exc:
        logger.error("get_live_game_feed(%d) failed: %s", game_pk, exc)
        return {}

    live = data.get("liveData", {})
    game_data = data.get("gameData", {})

    linescore = live.get("linescore", {})
    plays     = live.get("plays", {})
    boxscore  = live.get("boxscore", {})

    # Current pitcher
    current_play = plays.get("currentPlay", {})
    matchup      = current_play.get("matchup", {})
    pitcher_info = matchup.get("pitcher", {})
    current_pitcher = {
        "id":   pitcher_info.get("id"),
        "name": pitcher_info.get("fullName"),
        "hand": matchup.get("pitchHand", {}).get("code"),
    }

    # Base runners
    offense  = linescore.get("offense", {})
    runners  = []
    for base in ("first", "second", "third"):
        runner = offense.get(base)
        if runner:
            runners.append({
                "base":   base,
                "player": runner.get("fullName"),
                "id":     runner.get("id"),
            })

    # Recent plays (last 5 completed)
    all_plays   = plays.get("allPlays", [])
    recent_five = all_plays[-5:] if len(all_plays) >= 5 else all_plays
    recent_plays = [
        p.get("result", {}).get("description", "") for p in reversed(recent_five)
    ]

    # Score
    teams = game_data.get("teams", {})
    home_score = linescore.get("teams", {}).get("home", {}).get("runs")
    away_score = linescore.get("teams", {}).get("away", {}).get("runs")

    return {
        "game_pk":         game_pk,
        "status":          game_data.get("status", {}).get("detailedState"),
        "inning":          linescore.get("currentInning"),
        "inning_half":     linescore.get("inningHalf", ""),
        "home_score":      home_score,
        "away_score":      away_score,
        "current_pitcher": current_pitcher,
        "base_runners":    runners,
        "recent_plays":    recent_plays,
        "balls":           linescore.get("balls"),
        "strikes":         linescore.get("strikes"),
        "outs":            linescore.get("outs"),
    }


def get_all_team_ids() -> dict[str, int]:
    """Returns {team_name: team_id} mapping for all 30 MLB teams.

    Uses the statsapi library's lookup_team, with a hardcoded fallback in
    case the API call fails (team IDs are stable across seasons).
    """
    # Hardcoded stable IDs (MLB team IDs do not change)
    _HARDCODED = {
        "Arizona Diamondbacks":   109,
        "Atlanta Braves":         144,
        "Baltimore Orioles":      110,
        "Boston Red Sox":         111,
        "Chicago Cubs":           112,
        "Chicago White Sox":      145,
        "Cincinnati Reds":        113,
        "Cleveland Guardians":    114,
        "Colorado Rockies":       115,
        "Detroit Tigers":         116,
        "Houston Astros":         117,
        "Kansas City Royals":     118,
        "Los Angeles Angels":     108,
        "Los Angeles Dodgers":    119,
        "Miami Marlins":          146,
        "Milwaukee Brewers":      158,
        "Minnesota Twins":        142,
        "New York Mets":          121,
        "New York Yankees":       147,
        "Oakland Athletics":      133,
        "Philadelphia Phillies":  143,
        "Pittsburgh Pirates":     134,
        "San Diego Padres":       135,
        "San Francisco Giants":   137,
        "Seattle Mariners":       136,
        "St. Louis Cardinals":    138,
        "Tampa Bay Rays":         139,
        "Texas Rangers":          140,
        "Toronto Blue Jays":      141,
        "Washington Nationals":   120,
    }

    if statsapi is None:
        logger.warning("statsapi not installed; returning hardcoded team IDs")
        return _HARDCODED

    try:
        teams_data = statsapi.get(
            "teams",
            {"sportId": 1, "activeStatus": "Yes"},
        )
        result = {}
        for team in teams_data.get("teams", []):
            name = team.get("name")
            tid  = team.get("id")
            if name and tid:
                result[name] = tid
        if result:
            logger.debug("Loaded %d MLB team IDs from API", len(result))
            return result
    except Exception as exc:
        logger.warning("get_all_team_ids API call failed, using hardcoded: %s", exc)

    return _HARDCODED
