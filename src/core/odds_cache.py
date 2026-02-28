"""Single-fetch odds cache to avoid duplicate API calls.

Usage:
    cache = OddsCache(sportsbook="fanduel")
    odds = cache.get("basketball_nba")      # fetches on first call
    odds = cache.get("basketball_nba")      # returns cached, no API call
    odds_wnba = cache.get("basketball_wnba")  # fetches WNBA (different sport)
"""

import logging

from src.sports.nba.providers.odds_api import OddsApiProvider

logger = logging.getLogger(__name__)


class OddsCache:

    def __init__(self, sportsbook="fanduel", api_key=None):
        self.sportsbook = sportsbook
        self.api_key = api_key
        self._cache = {}  # sport -> odds dict

    def get(self, sport="basketball_nba"):
        """Return cached odds for a sport, fetching on first access."""
        if sport not in self._cache:
            provider = OddsApiProvider(
                sportsbook=self.sportsbook,
                api_key=self.api_key,
                sport=sport,
            )
            odds = provider.get_odds()
            self._cache[sport] = odds
            logger.info(
                "OddsCache: fetched %d games for %s via %s",
                len(odds) if odds else 0, sport, self.sportsbook,
            )
        return self._cache[sport]

    def get_best(self, sport="basketball_nba"):
        """Return best-odds-across-books data, cached."""
        key = f"{sport}__best"
        if key not in self._cache:
            provider = OddsApiProvider(
                sportsbook=self.sportsbook,
                api_key=self.api_key,
                sport=sport,
            )
            self._cache[key] = provider.get_best_odds()
        return self._cache[key]

    def inject(self, sport, odds_data):
        """Manually inject odds data (e.g. from a previous fetch)."""
        self._cache[sport] = odds_data
