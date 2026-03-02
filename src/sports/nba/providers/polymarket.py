"""PolymarketProvider: descubrimiento de mercados NBA y precios.

Usa APIs publicas (sin auth):
  - Gamma API: descubrimiento de eventos/mercados NBA
  - CLOB API: precios (midpoint, orderbook)

Rate limit: 60 req/min REST. Tracking de remaining.

Patron: sigue OddsApiProvider pero adaptado a prediction markets.
"""

import json
import logging
import re
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
        """Descubre mercados NBA activos en Polymarket (ML + AH).

        Cada evento de juego contiene sub-mercados: moneyline, spreads, O/U, props.
        Iteramos todos y clasificamos ML vs AH.

        Returns:
            dict con keys:
              "Home:Away" -> ML market data
              "Home:Away:AH:-3.5" -> AH market data (one per spread line)
        """
        params = {
            "tag_slug": "nba",
            "active": "true",
            "closed": "false",
            "limit": 100,
        }
        events = self._get(GAMMA_EVENTS_URL, params)

        if not events:
            params["tag_slug"] = "basketball"
            events = self._get(GAMMA_EVENTS_URL, params)

        if not events:
            logger.warning("No NBA events found on Polymarket")
            return {}

        markets = {}
        for event in events:
            parsed = self._parse_game_event(event)
            if parsed:
                markets.update(parsed)

        n_ml = sum(1 for k in markets if ":AH:" not in k)
        n_ah = sum(1 for k in markets if ":AH:" in k)
        logger.info("Polymarket: %d ML + %d AH markets found", n_ml, n_ah)
        return markets

    # Regex para spread markets: "Spread: TeamName (-3.5)"
    _SPREAD_RE = re.compile(r'^Spread:\s+(.+?)\s+\(([+-]?\d+\.?\d*)\)$')
    _H1_SPREAD_RE = re.compile(r'^1H\s+Spread:\s+(.+?)\s+\(([+-]?\d+\.?\d*)\)$')

    def _parse_game_event(self, event: dict) -> dict | None:
        """Parsea un evento de juego NBA en mercados ML + AH.

        Un evento como "Nuggets vs. Grizzlies" contiene sub-mercados:
        - ML: question = event title, outcomes = team names
        - AH: question = "Spread: TeamName (-X.5)"
        - O/U, props: ignorados por ahora
        """
        title = event.get("title", "")
        slug = event.get("slug", "")

        teams = self._extract_teams_from_title(title)
        if not teams:
            return None

        home_team, away_team = teams
        event_markets = event.get("markets", [])
        if not event_markets:
            return None

        result = {}
        game_key = f"{home_team}:{away_team}"

        for mkt in event_markets:
            question = mkt.get("question", "")
            parsed = self._parse_market_data(mkt)
            if not parsed:
                continue

            outcomes, prices, clob_token_ids, liquidity, volume = parsed

            # Clasificar mercado por tipo
            spread_match = self._SPREAD_RE.match(question)
            h1_spread_match = self._H1_SPREAD_RE.match(question)

            if spread_match:
                # AH/Spread market
                team_raw = spread_match.group(1)
                spread_val = float(spread_match.group(2))
                ah_market = self._build_ah_market(
                    home_team, away_team, team_raw, spread_val,
                    outcomes, prices, clob_token_ids, liquidity, volume,
                    mkt, slug,
                )
                if ah_market:
                    key = f"{game_key}:AH:{spread_val:+.1f}"
                    result[key] = ah_market

            elif h1_spread_match:
                # 1H Spread — skip for now (future feature)
                pass

            elif not any(kw in question.lower() for kw in ("spread", "o/u", "over", "under", "assists", "points", "rebounds", "three")):
                # Moneyline market: question matches event title or is generic "Team vs Team"
                ml_market = self._build_ml_market(
                    home_team, away_team,
                    outcomes, prices, clob_token_ids, liquidity, volume,
                    mkt, slug, question, title,
                )
                if ml_market and game_key not in result:
                    result[game_key] = ml_market

        return result if result else None

    def _parse_market_data(self, mkt: dict):
        """Extrae outcomes, prices, token IDs de un sub-mercado."""
        outcomes_raw = mkt.get("outcomes", [])
        prices_raw = mkt.get("outcomePrices", [])

        if isinstance(outcomes_raw, str):
            try:
                outcomes_raw = json.loads(outcomes_raw)
            except (ValueError, TypeError):
                return None
        if isinstance(prices_raw, str):
            try:
                prices_raw = json.loads(prices_raw)
            except (ValueError, TypeError):
                return None

        if len(outcomes_raw) < 2 or len(prices_raw) < 2:
            return None

        try:
            prices = [float(p) for p in prices_raw]
        except (ValueError, TypeError):
            return None

        clob_token_ids = mkt.get("clobTokenIds", [])
        if isinstance(clob_token_ids, str):
            try:
                clob_token_ids = json.loads(clob_token_ids)
            except (ValueError, TypeError):
                clob_token_ids = []
        if len(clob_token_ids) < 2:
            clob_token_ids = [None, None]

        liquidity = float(mkt.get("liquidityNum", 0) or 0)
        volume = float(mkt.get("volumeNum", 0) or 0)
        return outcomes_raw, prices, clob_token_ids, liquidity, volume

    def _build_ml_market(self, home_team, away_team, outcomes, prices,
                         clob_token_ids, liquidity, volume, mkt, slug,
                         question, title):
        """Construye dict de mercado ML."""
        home_idx, away_idx = self._match_outcomes_to_teams(outcomes, home_team, away_team)
        return {
            "home_team": home_team,
            "away_team": away_team,
            "market_type": "ML",
            "home_token_id": clob_token_ids[home_idx] if clob_token_ids[home_idx] else None,
            "away_token_id": clob_token_ids[away_idx] if clob_token_ids[away_idx] else None,
            "home_price": prices[home_idx],
            "away_price": prices[away_idx],
            "liquidity": liquidity,
            "volume": volume,
            "condition_id": mkt.get("conditionId", ""),
            "slug": slug,
            "question": question or title,
        }

    def _build_ah_market(self, home_team, away_team, team_raw, spread_val,
                         outcomes, prices, clob_token_ids, liquidity, volume,
                         mkt, slug):
        """Construye dict de mercado AH/spread.

        En Polymarket los spreads son: "Spread: Grizzlies (-3.5)"
        - outcomes[0] = team con el spread (Grizzlies), price = P(cubre)
        - outcomes[1] = otro team, price = P(no cubre)
        """
        cover_team = _normalize_team_name(team_raw)
        if not cover_team:
            return None

        # Asignar desde perspectiva home
        if cover_team == home_team:
            home_spread = spread_val
            home_price = prices[0]
            away_price = prices[1]
            home_token_id = clob_token_ids[0]
            away_token_id = clob_token_ids[1]
        elif cover_team == away_team:
            home_spread = -spread_val
            home_price = prices[1]
            away_price = prices[0]
            home_token_id = clob_token_ids[1]
            away_token_id = clob_token_ids[0]
        else:
            return None

        return {
            "home_team": home_team,
            "away_team": away_team,
            "market_type": "AH",
            "spread_line": home_spread,
            "home_token_id": home_token_id if home_token_id else None,
            "away_token_id": away_token_id if away_token_id else None,
            "home_price": home_price,
            "away_price": away_price,
            "liquidity": liquidity,
            "volume": volume,
            "condition_id": mkt.get("conditionId", ""),
            "slug": slug,
            "question": mkt.get("question", ""),
        }

    # Kept for backward compatibility (used by _build_ml_market)
    def _parse_nba_event(self, event: dict) -> dict | None:
        """Legacy: parsea solo el primer mercado ML de un evento."""
        result = self._parse_game_event(event)
        if not result:
            return None
        # Return first ML market found
        for key, mkt in result.items():
            if ":AH:" not in key:
                return mkt
        return None

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
            markets: dict de get_nba_markets() con keys "H:A" (ML) y "H:A:AH:-3.5" (AH)

        Returns:
            dict con keys:
              "Home:Away" -> ML market data
              "Home:Away:AH:-3.5" -> AH market data (best AH per game)
        """
        matched = {}
        for home, away in games:
            game_key = f"{home}:{away}"
            inv_key = f"{away}:{home}"

            # --- ML match ---
            ml_mkt = self._find_market(game_key, inv_key, markets, "ML")
            if ml_mkt:
                matched[game_key] = ml_mkt

            # --- AH matches: find all spread lines for this game ---
            for mkt_key, mkt_data in markets.items():
                if ":AH:" not in mkt_key:
                    continue
                mkt_home = mkt_data["home_team"]
                mkt_away = mkt_data["away_team"]
                is_direct = (mkt_home == home and mkt_away == away)
                is_inverse = (mkt_home == away and mkt_away == home)

                if is_direct:
                    spread_line = mkt_data["spread_line"]
                    matched[f"{game_key}:AH:{spread_line:+.1f}"] = mkt_data
                elif is_inverse:
                    swapped = self._swap_market(mkt_data, home, away)
                    spread_line = swapped["spread_line"]
                    matched[f"{game_key}:AH:{spread_line:+.1f}"] = swapped

        n_ml = sum(1 for k in matched if ":AH:" not in k)
        n_ah = sum(1 for k in matched if ":AH:" in k)
        logger.info("Polymarket: matched %d ML + %d AH / %d games", n_ml, n_ah, len(games))
        return matched

    def _find_market(self, game_key, inv_key, markets, mtype):
        """Busca mercado ML por game_key directo o inverso."""
        if game_key in markets and markets[game_key].get("market_type", "ML") == mtype:
            return markets[game_key]
        if inv_key in markets and markets[inv_key].get("market_type", "ML") == mtype:
            home, away = game_key.split(":")
            return self._swap_market(markets[inv_key], home, away)
        # Fuzzy match
        for mkt_key, mkt_data in markets.items():
            if ":AH:" in mkt_key:
                continue
            if mkt_data["home_team"] == game_key.split(":")[0] and mkt_data["away_team"] == game_key.split(":")[1]:
                return mkt_data
            if mkt_data["home_team"] == inv_key.split(":")[0] and mkt_data["away_team"] == inv_key.split(":")[1]:
                home, away = game_key.split(":")
                return self._swap_market(mkt_data, home, away)
        return None

    def _swap_market(self, mkt, home, away):
        """Swap home/away en un mercado para que matchee el game del pipeline."""
        swapped = mkt.copy()
        swapped["home_team"] = home
        swapped["away_team"] = away
        swapped["home_price"], swapped["away_price"] = swapped["away_price"], swapped["home_price"]
        swapped["home_token_id"], swapped["away_token_id"] = swapped["away_token_id"], swapped["home_token_id"]
        if "spread_line" in swapped:
            swapped["spread_line"] = -swapped["spread_line"]
        return swapped
