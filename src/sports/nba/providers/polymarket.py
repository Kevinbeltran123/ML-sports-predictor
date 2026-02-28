"""PolymarketProvider: descubrimiento de mercados NBA y precios.

Usa APIs publicas (sin auth):
  - Gamma API: descubrimiento de eventos/mercados NBA
  - CLOB API: precios (midpoint, orderbook)

Rate limit: 60 req/min REST. Tracking de remaining.

Patron: sigue OddsApiProvider pero adaptado a prediction markets.
"""

import json
import logging
import time
from difflib import SequenceMatcher

import requests

logger = logging.getLogger(__name__)

# --- API endpoints ---
GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
CLOB_BASE_URL = "https://clob.polymarket.com"

# Rate limit: 60 req/min
MAX_REQUESTS_PER_MINUTE = 60
MIN_REQUEST_INTERVAL = 60.0 / MAX_REQUESTS_PER_MINUTE  # 1 segundo

# --- Team name normalization ---
# Polymarket usa nombres variados: "Lakers", "Los Angeles Lakers", "LA Lakers"
# Mapeamos a los nombres canonicos del proyecto (team_index_current keys)
_PM_TEAM_ALIASES = {
    # Variaciones comunes en Polymarket
    "lakers": "Los Angeles Lakers",
    "la lakers": "Los Angeles Lakers",
    "los angeles lakers": "Los Angeles Lakers",
    "celtics": "Boston Celtics",
    "boston celtics": "Boston Celtics",
    "warriors": "Golden State Warriors",
    "golden state warriors": "Golden State Warriors",
    "nuggets": "Denver Nuggets",
    "denver nuggets": "Denver Nuggets",
    "bucks": "Milwaukee Bucks",
    "milwaukee bucks": "Milwaukee Bucks",
    "76ers": "Philadelphia 76ers",
    "sixers": "Philadelphia 76ers",
    "philadelphia 76ers": "Philadelphia 76ers",
    "knicks": "New York Knicks",
    "new york knicks": "New York Knicks",
    "nets": "Brooklyn Nets",
    "brooklyn nets": "Brooklyn Nets",
    "heat": "Miami Heat",
    "miami heat": "Miami Heat",
    "bulls": "Chicago Bulls",
    "chicago bulls": "Chicago Bulls",
    "cavaliers": "Cleveland Cavaliers",
    "cavs": "Cleveland Cavaliers",
    "cleveland cavaliers": "Cleveland Cavaliers",
    "pistons": "Detroit Pistons",
    "detroit pistons": "Detroit Pistons",
    "pacers": "Indiana Pacers",
    "indiana pacers": "Indiana Pacers",
    "hawks": "Atlanta Hawks",
    "atlanta hawks": "Atlanta Hawks",
    "hornets": "Charlotte Hornets",
    "charlotte hornets": "Charlotte Hornets",
    "magic": "Orlando Magic",
    "orlando magic": "Orlando Magic",
    "wizards": "Washington Wizards",
    "washington wizards": "Washington Wizards",
    "raptors": "Toronto Raptors",
    "toronto raptors": "Toronto Raptors",
    "mavericks": "Dallas Mavericks",
    "mavs": "Dallas Mavericks",
    "dallas mavericks": "Dallas Mavericks",
    "rockets": "Houston Rockets",
    "houston rockets": "Houston Rockets",
    "grizzlies": "Memphis Grizzlies",
    "memphis grizzlies": "Memphis Grizzlies",
    "pelicans": "New Orleans Pelicans",
    "new orleans pelicans": "New Orleans Pelicans",
    "spurs": "San Antonio Spurs",
    "san antonio spurs": "San Antonio Spurs",
    "timberwolves": "Minnesota Timberwolves",
    "wolves": "Minnesota Timberwolves",
    "minnesota timberwolves": "Minnesota Timberwolves",
    "thunder": "Oklahoma City Thunder",
    "okc thunder": "Oklahoma City Thunder",
    "oklahoma city thunder": "Oklahoma City Thunder",
    "blazers": "Portland Trail Blazers",
    "trail blazers": "Portland Trail Blazers",
    "portland trail blazers": "Portland Trail Blazers",
    "kings": "Sacramento Kings",
    "sacramento kings": "Sacramento Kings",
    "suns": "Phoenix Suns",
    "phoenix suns": "Phoenix Suns",
    "jazz": "Utah Jazz",
    "utah jazz": "Utah Jazz",
    "clippers": "LA Clippers",
    "la clippers": "LA Clippers",
    "los angeles clippers": "LA Clippers",
}


def _normalize_team_name(raw_name: str) -> str | None:
    """Normaliza un nombre de equipo de Polymarket al formato canonico.

    Intenta:
    1. Match exacto (case-insensitive) en alias map
    2. Substring match (si el raw_name contiene un alias conocido)
    3. Fuzzy match con SequenceMatcher (cutoff 0.7)
    """
    key = raw_name.strip().lower()

    # 1. Match exacto
    if key in _PM_TEAM_ALIASES:
        return _PM_TEAM_ALIASES[key]

    # 2. Substring: buscar si algun alias esta contenido en el nombre
    for alias, canonical in _PM_TEAM_ALIASES.items():
        if alias in key or key in alias:
            return canonical

    # 3. Fuzzy match
    best_score = 0.0
    best_match = None
    for alias, canonical in _PM_TEAM_ALIASES.items():
        score = SequenceMatcher(None, key, alias).ratio()
        if score > best_score:
            best_score = score
            best_match = canonical
    if best_score >= 0.7:
        return best_match

    logger.warning("No se pudo normalizar equipo PM: '%s'", raw_name)
    return None


class PolymarketProvider:
    """Proveedor de mercados NBA en Polymarket.

    Uso:
        provider = PolymarketProvider()
        markets = provider.get_nba_markets()
        # markets = {"Boston Celtics:Miami Heat": {...}, ...}
    """

    def __init__(self):
        self._last_request_time = 0.0
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "User-Agent": "nba-wl-predictor/1.0",
        })

    def _rate_limit(self):
        """Espera si es necesario para respetar rate limit."""
        elapsed = time.time() - self._last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _get(self, url: str, params: dict = None) -> dict | list:
        """GET con rate limiting y error handling."""
        self._rate_limit()
        try:
            resp = self._session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error("Polymarket API error: %s", e)
            return []

    def get_nba_markets(self) -> dict:
        """Descubre mercados NBA activos en Polymarket.

        Usa Gamma API para buscar eventos NBA, luego extrae token IDs
        y precios de las outcomes.

        Returns:
            dict keyed by "Home:Away" (nombres canonicos) con:
              home_token_id, away_token_id, home_price, away_price,
              liquidity, volume, condition_id, slug
        """
        # Buscar eventos NBA en Gamma (tag_slug, no tag)
        params = {
            "tag_slug": "nba",
            "active": "true",
            "closed": "false",
            "limit": 100,
        }
        events = self._get(GAMMA_EVENTS_URL, params)

        if not events:
            # Fallback: buscar con tag_slug alternativo
            params["tag_slug"] = "basketball"
            events = self._get(GAMMA_EVENTS_URL, params)

        if not events:
            logger.warning("No NBA events found on Polymarket")
            return {}

        markets = {}
        for event in events:
            market = self._parse_nba_event(event)
            if market:
                key = f"{market['home_team']}:{market['away_team']}"
                markets[key] = market

        logger.info("Polymarket: %d NBA markets found", len(markets))
        return markets

    def _parse_nba_event(self, event: dict) -> dict | None:
        """Parsea un evento de Gamma API en formato de mercado.

        Los eventos NBA en Polymarket tipicamente tienen:
        - title: "Will the Lakers win against the Celtics?"
        - markets: lista con outcome tokens (YES/NO para cada equipo)
        """
        title = event.get("title", "")
        slug = event.get("slug", "")

        # Extraer equipos del titulo
        teams = self._extract_teams_from_title(title)
        if not teams:
            return None

        home_team, away_team = teams

        # Buscar markets dentro del evento
        event_markets = event.get("markets", [])
        if not event_markets:
            return None

        # En NBA, tipicamente hay 1 market binario (equipo A gana vs equipo B gana)
        mkt = event_markets[0]

        # Outcome tokens — API returns JSON strings, need to parse
        outcomes_raw = mkt.get("outcomes", [])
        prices_raw = mkt.get("outcomePrices", [])

        # Parse JSON strings if needed (API returns '["Yes","No"]' as string)
        if isinstance(outcomes_raw, str):
            try:
                outcomes_raw = json.loads(outcomes_raw)
            except (ValueError, TypeError):
                outcomes_raw = []
        if isinstance(prices_raw, str):
            try:
                prices_raw = json.loads(prices_raw)
            except (ValueError, TypeError):
                prices_raw = []

        outcomes = outcomes_raw
        outcome_prices = prices_raw

        if len(outcomes) < 2 or len(outcome_prices) < 2:
            return None

        try:
            prices = [float(p) for p in outcome_prices]
        except (ValueError, TypeError):
            return None

        # Determinar cual outcome es home y cual away
        home_idx, away_idx = self._match_outcomes_to_teams(
            outcomes, home_team, away_team
        )

        # Extraer token IDs del CLOB (tambien viene como JSON string)
        clob_token_ids = mkt.get("clobTokenIds", [])
        if isinstance(clob_token_ids, str):
            try:
                clob_token_ids = json.loads(clob_token_ids)
            except (ValueError, TypeError):
                clob_token_ids = []
        if len(clob_token_ids) < 2:
            clob_token_ids = [None, None]

        condition_id = mkt.get("conditionId", "")
        liquidity = float(mkt.get("liquidityNum", 0) or 0)
        volume = float(mkt.get("volumeNum", 0) or 0)

        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_token_id": clob_token_ids[home_idx] if clob_token_ids[home_idx] else None,
            "away_token_id": clob_token_ids[away_idx] if clob_token_ids[away_idx] else None,
            "home_price": prices[home_idx],
            "away_price": prices[away_idx],
            "liquidity": liquidity,
            "volume": volume,
            "condition_id": condition_id,
            "slug": slug,
            "question": mkt.get("question", title),
        }

    def _extract_teams_from_title(self, title: str) -> tuple[str, str] | None:
        """Extrae nombres de equipos del titulo del evento.

        Formatos comunes en Polymarket:
          "Nuggets vs. Grizzlies"
          "Lakers vs Warriors"
          "Will the Lakers beat the Celtics?"
          "NBA: Los Angeles Lakers vs Boston Celtics"
        """
        title_lower = title.lower()

        # Buscar todos los equipos mencionados con su posicion en el titulo
        found = []
        for alias, canonical in _PM_TEAM_ALIASES.items():
            pos = title_lower.find(alias)
            if pos >= 0 and canonical not in [c for _, c in found]:
                found.append((pos, canonical))

        if len(found) >= 2:
            # Ordenar por posicion en el titulo para respetar orden "TeamA vs. TeamB"
            found.sort(key=lambda x: x[0])
            return found[0][1], found[1][1]

        return None

    def _match_outcomes_to_teams(
        self, outcomes: list[str], home_team: str, away_team: str
    ) -> tuple[int, int]:
        """Determina cual outcome index corresponde a cada equipo.

        outcomes puede ser ["Yes", "No"] o ["Lakers", "Celtics"], etc.
        """
        if len(outcomes) == 2:
            o0 = _normalize_team_name(outcomes[0])
            o1 = _normalize_team_name(outcomes[1])

            if o0 == home_team:
                return 0, 1
            elif o1 == home_team:
                return 1, 0

        # Default: primer outcome = primer equipo encontrado en titulo
        return 0, 1

    def get_midpoint(self, token_id: str) -> float | None:
        """Obtiene el midpoint price de un token via CLOB API.

        El midpoint es (best_bid + best_ask) / 2.
        Mas preciso que el precio de Gamma para ejecucion.
        """
        if not token_id:
            return None

        url = f"{CLOB_BASE_URL}/midpoint"
        data = self._get(url, params={"token_id": token_id})

        if isinstance(data, dict) and "mid" in data:
            try:
                return float(data["mid"])
            except (ValueError, TypeError):
                pass
        return None

    def get_orderbook(self, token_id: str) -> dict | None:
        """Obtiene el orderbook completo para analisis de slippage.

        Returns:
            dict con 'bids' y 'asks', cada uno lista de [price, size].
            None si error.
        """
        if not token_id:
            return None

        url = f"{CLOB_BASE_URL}/book"
        data = self._get(url, params={"token_id": token_id})

        if isinstance(data, dict) and ("bids" in data or "asks" in data):
            return {
                "bids": data.get("bids", []),
                "asks": data.get("asks", []),
            }
        return None

    def estimate_slippage(self, token_id: str, size_usdc: float) -> float:
        """Estima slippage para un order de size_usdc dolares.

        Recorre el orderbook ask-side acumulando hasta cubrir el size.
        Retorna el precio promedio ponderado - midpoint.
        """
        book = self.get_orderbook(token_id)
        if not book or not book["asks"]:
            return 0.0

        midpoint = self.get_midpoint(token_id)
        if not midpoint:
            return 0.0

        total_cost = 0.0
        total_shares = 0.0

        for ask in book["asks"]:
            try:
                ask_price = float(ask.get("price", ask[0]) if isinstance(ask, dict) else ask[0])
                ask_size = float(ask.get("size", ask[1]) if isinstance(ask, dict) else ask[1])
            except (IndexError, ValueError, TypeError):
                continue

            cost_this_level = ask_price * ask_size
            if total_cost + cost_this_level >= size_usdc:
                remaining = size_usdc - total_cost
                shares_here = remaining / ask_price
                total_shares += shares_here
                total_cost += remaining
                break
            else:
                total_shares += ask_size
                total_cost += cost_this_level

        if total_shares <= 0:
            return 0.0

        avg_price = total_cost / total_shares
        return round(avg_price - midpoint, 4)

    def match_markets_to_games(
        self, games: list[tuple[str, str]], markets: dict
    ) -> dict:
        """Matchea mercados PM a los juegos del dia.

        Args:
            games: lista de (home_team, away_team) del pipeline
            markets: dict de get_nba_markets()

        Returns:
            dict: game_key -> market_data (solo juegos matcheados)
        """
        matched = {}
        for home, away in games:
            game_key = f"{home}:{away}"

            # Match directo
            if game_key in markets:
                matched[game_key] = markets[game_key]
                continue

            # Match inverso (PM pone equipos en orden diferente)
            inv_key = f"{away}:{home}"
            if inv_key in markets:
                # Swap prices para que home sea consistente
                mkt = markets[inv_key].copy()
                mkt["home_team"] = home
                mkt["away_team"] = away
                mkt["home_price"], mkt["away_price"] = mkt["away_price"], mkt["home_price"]
                mkt["home_token_id"], mkt["away_token_id"] = mkt["away_token_id"], mkt["home_token_id"]
                matched[game_key] = mkt
                continue

            # Fuzzy match por nombre parcial
            for mkt_key, mkt_data in markets.items():
                mkt_home = mkt_data["home_team"]
                mkt_away = mkt_data["away_team"]
                if (mkt_home == home and mkt_away == away) or \
                   (mkt_home == away and mkt_away == home):
                    if mkt_home == home:
                        matched[game_key] = mkt_data
                    else:
                        swapped = mkt_data.copy()
                        swapped["home_team"] = home
                        swapped["away_team"] = away
                        swapped["home_price"], swapped["away_price"] = swapped["away_price"], swapped["home_price"]
                        swapped["home_token_id"], swapped["away_token_id"] = swapped["away_token_id"], swapped["home_token_id"]
                        matched[game_key] = swapped
                    break

        logger.info("Polymarket: %d/%d games matched", len(matched), len(games))
        return matched
