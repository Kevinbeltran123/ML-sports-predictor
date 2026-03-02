"""MLB odds provider for The Odds API.

Mirrors the pattern from src/sports/nba/providers/odds_api.py.
sport = "baseball_mlb"

Supports:
  - Full-game moneyline, spreads (run line), totals
  - First 5 innings (F5) moneyline
  - Quota tracking via response headers
  - Exponential backoff retries
"""

import logging
import os
import time
from datetime import datetime, timedelta, timezone

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MLB_SPORT = "baseball_mlb"
ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports/{sport}/odds/"
ODDS_API_EVENTS_URL = "https://api.the-odds-api.com/v4/sports/{sport}/events/"
ODDS_API_MLB_URL = ODDS_API_BASE.format(sport=MLB_SPORT)
ODDS_API_MLB_EVENTS_URL = ODDS_API_EVENTS_URL.format(sport=MLB_SPORT)

# Alert when fewer than 2,000 credits remain (same threshold as NBA provider)
QUOTA_WARNING_THRESHOLD = 2_000

BOOKMAKER_MAP = {
    "fanduel":    "fanduel",
    "draftkings": "draftkings",
    "betmgm":     "betmgm",
    "pointsbet":  "pointsbetus",
    "caesars":    "williamhill_us",
    "wynn":       "wynnbet",
    "bet_rivers_ny": "betrivers",
}

# F5 innings market key on The Odds API
F5_MARKET_KEY = "h2h_1st_5_innings"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _request_with_retry(url: str, params: dict, max_retries: int = 3,
                        timeout: int = 15) -> requests.Response:
    """GET with exponential backoff for transient errors."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as exc:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt  # 1s, 2s, 4s
            logger.warning(
                "Odds API MLB request failed (attempt %d/%d), retry in %ds: %s",
                attempt + 1, max_retries, wait, exc,
            )
            time.sleep(wait)


def _today_et() -> "date":
    """Return today's date in Eastern Time."""
    et = timezone(timedelta(hours=-5))
    return datetime.now(et).date()


def _event_date_et(commence_time: str):
    """Parse an ISO-8601 UTC timestamp and return its ET date."""
    try:
        et = timezone(timedelta(hours=-5))
        return (
            datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
            .astimezone(et)
            .date()
        )
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# MLBOddsProvider
# ---------------------------------------------------------------------------

class MLBOddsProvider:
    """Odds API wrapper for MLB (baseball_mlb).

    Usage:
        provider = MLBOddsProvider(sportsbook="fanduel")
        games = provider.get_odds()
        all_with_f5 = provider.get_all_odds_with_f5()
    """

    def __init__(self, sportsbook: str = "fanduel", api_key: str = None):
        self.api_key = api_key or os.environ.get("ODDS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ODDS_API_KEY not set. Pass api_key or set the ODDS_API_KEY "
                "environment variable."
            )
        self.sportsbook = sportsbook
        self.bookmaker_key = BOOKMAKER_MAP.get(sportsbook, sportsbook)
        self.sport = MLB_SPORT
        self._odds_url = ODDS_API_MLB_URL
        self._events_url = ODDS_API_MLB_EVENTS_URL
        self._event_odds_url = (
            f"https://api.the-odds-api.com/v4/sports/{MLB_SPORT}/events/{{event_id}}/odds/"
        )

        # Updated after each successful request
        self._quota_remaining = None
        self._quota_used = None

    # ------------------------------------------------------------------
    # Quota tracking
    # ------------------------------------------------------------------

    def _update_quota(self, response: requests.Response) -> None:
        """Read quota headers from response and update internal state."""
        remaining = response.headers.get("x-requests-remaining")
        used      = response.headers.get("x-requests-used")

        if remaining is not None:
            self._quota_remaining = int(remaining)
        if used is not None:
            self._quota_used = int(used)

        if self._quota_remaining is not None:
            logger.debug(
                "Odds API MLB quota: %d remaining, %d used",
                self._quota_remaining, self._quota_used or 0,
            )
            if self._quota_remaining < QUOTA_WARNING_THRESHOLD:
                logger.warning(
                    "Quota baja: %d creditos restantes (umbral: %d). "
                    "Considera reducir requests o esperar al siguiente ciclo.",
                    self._quota_remaining, QUOTA_WARNING_THRESHOLD,
                )

    def get_quota(self) -> dict:
        """Return current API quota state.

        Returns:
            Dict with keys: remaining, used, pct_used (or None if no data yet).
        """
        if self._quota_remaining is None:
            return {"remaining": None, "used": None, "pct_used": None}
        total   = (self._quota_remaining or 0) + (self._quota_used or 0)
        pct     = (self._quota_used / total * 100) if total > 0 else 0.0
        return {
            "remaining": self._quota_remaining,
            "used":      self._quota_used,
            "pct_used":  round(pct, 1),
        }

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def get_odds(self, markets: str = "h2h,spreads,totals") -> list[dict]:
        """Get odds for all of today's MLB games.

        Args:
            markets: comma-separated Odds API market keys.
                     Default: "h2h,spreads,totals"

        Returns:
            List of game dicts, one per game, with keys:
              game_pk (event_id), home_team, away_team, commence_time,
              ml_home, ml_away,
              run_line_home (spread), run_line_home_odds, run_line_away_odds,
              total, over_odds, under_odds
        """
        params = {
            "apiKey":     self.api_key,
            "regions":    "us",
            "markets":    markets,
            "oddsFormat": "american",
            "bookmakers": self.bookmaker_key,
        }
        try:
            resp = _request_with_retry(self._odds_url, params=params)
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 422:
                logger.warning(
                    "Market(s) '%s' not available for MLB, retrying with h2h only", markets
                )
                params["markets"] = "h2h"
                resp = _request_with_retry(self._odds_url, params=params)
            else:
                raise

        self._update_quota(resp)
        events = resp.json()

        today = _today_et()
        results = []

        for event in events:
            # Filter to today's games only
            event_date = _event_date_et(event.get("commence_time", ""))
            if event_date and event_date != today:
                continue

            home_team = event.get("home_team", "")
            away_team = event.get("away_team", "")

            ml_home              = None
            ml_away              = None
            run_line_home        = None
            run_line_home_odds   = None
            run_line_away_odds   = None
            total                = None
            over_odds            = None
            under_odds           = None

            for bookmaker in event.get("bookmakers", []):
                if bookmaker["key"] != self.bookmaker_key:
                    continue
                for market in bookmaker.get("markets", []):
                    mkey = market["key"]
                    if mkey == "h2h":
                        for outcome in market.get("outcomes", []):
                            if outcome["name"] == home_team:
                                ml_home = outcome.get("price")
                            elif outcome["name"] == away_team:
                                ml_away = outcome.get("price")
                    elif mkey == "spreads":
                        for outcome in market.get("outcomes", []):
                            if outcome["name"] == home_team:
                                run_line_home      = outcome.get("point")
                                run_line_home_odds = outcome.get("price")
                            elif outcome["name"] == away_team:
                                run_line_away_odds = outcome.get("price")
                    elif mkey == "totals":
                        for outcome in market.get("outcomes", []):
                            if outcome["name"] == "Over":
                                total     = outcome.get("point")
                                over_odds = outcome.get("price")
                            elif outcome["name"] == "Under":
                                under_odds = outcome.get("price")

            if ml_home is None or ml_away is None:
                logger.debug("Skipping %s vs %s — missing moneyline", home_team, away_team)
                continue

            results.append({
                "event_id":          event.get("id"),
                "home_team":         home_team,
                "away_team":         away_team,
                "commence_time":     event.get("commence_time"),
                "ml_home":           ml_home,
                "ml_away":           ml_away,
                "run_line_home":     run_line_home if run_line_home is not None else -1.5,
                "run_line_home_odds": run_line_home_odds if run_line_home_odds is not None else -110,
                "run_line_away_odds": run_line_away_odds if run_line_away_odds is not None else -110,
                "total":             total,
                "over_odds":         over_odds,
                "under_odds":        under_odds,
            })

        logger.info("MLB odds fetched: %d games today", len(results))
        return results

    def get_f5_odds(self, event_id: str) -> dict:
        """Get First 5 innings odds for a specific game.

        Market key: h2h_1st_5_innings

        Args:
            event_id: The Odds API event ID (from get_odds() or get_events())

        Returns:
            Dict with keys: event_id, ml_home_f5, ml_away_f5
            Returns empty dict on error or if market not available.
        """
        url = self._event_odds_url.format(event_id=event_id)
        params = {
            "apiKey":     self.api_key,
            "regions":    "us",
            "markets":    F5_MARKET_KEY,
            "oddsFormat": "american",
            "bookmakers": self.bookmaker_key,
        }
        try:
            resp = _request_with_retry(url, params=params)
            self._update_quota(resp)
            data = resp.json()
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            logger.warning("get_f5_odds(%s) HTTP %s — market may not be available", event_id, status)
            return {}
        except Exception as exc:
            logger.error("get_f5_odds(%s) failed: %s", event_id, exc)
            return {}

        home_team = data.get("home_team", "")
        away_team = data.get("away_team", "")
        ml_home_f5 = None
        ml_away_f5 = None

        for bookmaker in data.get("bookmakers", []):
            if bookmaker["key"] != self.bookmaker_key:
                continue
            for market in bookmaker.get("markets", []):
                if market["key"] != F5_MARKET_KEY:
                    continue
                for outcome in market.get("outcomes", []):
                    if outcome["name"] == home_team:
                        ml_home_f5 = outcome.get("price")
                    elif outcome["name"] == away_team:
                        ml_away_f5 = outcome.get("price")

        return {
            "event_id":   event_id,
            "home_team":  home_team,
            "away_team":  away_team,
            "ml_home_f5": ml_home_f5,
            "ml_away_f5": ml_away_f5,
        }

    def get_all_odds_with_f5(self) -> list[dict]:
        """Get all today's games with full-game AND F5 odds merged.

        Efficiency: 2 API credits for full-game odds (1 request) + 1 credit
        per game with F5 data, so typically 2–3 total credits when the
        sportsbook carries the F5 market.

        Strategy:
          1. Fetch full-game odds (1 credit).
          2. Attempt a single F5 request via get_odds(markets='h2h_1st_5_innings').
             This costs 1 credit and returns F5 lines for all games at once.
          3. Merge by matching home_team + away_team.

        Returns:
            List of game dicts (same as get_odds()) with additional keys:
              ml_home_f5, ml_away_f5  (None if F5 not available)
        """
        # Step 1 — full-game odds
        games = self.get_odds(markets="h2h,spreads,totals")
        if not games:
            return games

        # Step 2 — try bulk F5 request (1 credit)
        f5_lookup: dict[str, dict] = {}
        try:
            params = {
                "apiKey":     self.api_key,
                "regions":    "us",
                "markets":    F5_MARKET_KEY,
                "oddsFormat": "american",
                "bookmakers": self.bookmaker_key,
            }
            resp = _request_with_retry(self._odds_url, params=params)
            self._update_quota(resp)
            f5_events = resp.json()

            today = _today_et()
            for event in f5_events:
                event_date = _event_date_et(event.get("commence_time", ""))
                if event_date and event_date != today:
                    continue
                home = event.get("home_team", "")
                away = event.get("away_team", "")
                ml_home_f5 = None
                ml_away_f5 = None
                for bookmaker in event.get("bookmakers", []):
                    if bookmaker["key"] != self.bookmaker_key:
                        continue
                    for market in bookmaker.get("markets", []):
                        if market["key"] != F5_MARKET_KEY:
                            continue
                        for outcome in market.get("outcomes", []):
                            if outcome["name"] == home:
                                ml_home_f5 = outcome.get("price")
                            elif outcome["name"] == away:
                                ml_away_f5 = outcome.get("price")
                key = f"{home}:{away}"
                f5_lookup[key] = {"ml_home_f5": ml_home_f5, "ml_away_f5": ml_away_f5}

        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            logger.warning("F5 bulk request returned HTTP %s — F5 odds not available today", status)
        except Exception as exc:
            logger.warning("F5 bulk request failed: %s — continuing without F5 odds", exc)

        # Step 3 — merge
        for game in games:
            key = f"{game['home_team']}:{game['away_team']}"
            f5  = f5_lookup.get(key, {})
            game["ml_home_f5"] = f5.get("ml_home_f5")
            game["ml_away_f5"] = f5.get("ml_away_f5")

        f5_count = sum(1 for g in games if g.get("ml_home_f5") is not None)
        logger.info(
            "MLB odds with F5: %d games total, %d with F5 lines", len(games), f5_count
        )
        return games

    def get_events(self) -> list[dict]:
        """Get list of today's MLB events with their event IDs.

        Returns:
            List of dicts: {id, home_team, away_team, commence_time, key}
        """
        params = {"apiKey": self.api_key}
        try:
            resp = _request_with_retry(self._events_url, params=params)
            self._update_quota(resp)
            events = resp.json()
        except Exception as exc:
            logger.error("get_events() failed: %s", exc)
            return []

        today   = _today_et()
        results = []
        for event in events:
            event_date = _event_date_et(event.get("commence_time", ""))
            if event_date and event_date != today:
                continue
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            results.append({
                "id":            event.get("id"),
                "home_team":     home,
                "away_team":     away,
                "commence_time": event.get("commence_time"),
                "key":           f"{home}:{away}",
            })
        return results
