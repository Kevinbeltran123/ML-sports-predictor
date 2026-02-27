import logging
import os

import requests

logger = logging.getLogger(__name__)

# URL base del endpoint de odds por evento (para player props necesitamos event_id)
ODDS_API_EVENT_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds/"

BOOKMAKER_MAP = {
    'fanduel': 'fanduel',
    'draftkings': 'draftkings',
    'betmgm': 'betmgm',
    'pointsbet': 'pointsbetus',
    'caesars': 'williamhill_us',
    'wynn': 'wynnbet',
    'bet_rivers_ny': 'betrivers',
}

# 6 stats básicas: 6 créditos/evento (vs 11 con combos)
BASIC_PROP_MARKETS = [
    "player_points", "player_rebounds", "player_assists",
    "player_threes", "player_blocks", "player_steals",
]

# 11 stats (básicas + combos): 11 créditos/evento
ALL_PROP_MARKETS = BASIC_PROP_MARKETS + [
    "player_points_rebounds_assists", "player_points_rebounds",
    "player_points_assists", "player_rebounds_assists",
    "player_blocks_steals",
]

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"

# Umbral de alerta: si quedan menos del 10% de creditos, loguear WARNING
QUOTA_WARNING_THRESHOLD = 2_000  # 10% de 20,000 creditos del plan $30


class OddsApiProvider:

    def __init__(self, sportsbook="fanduel", api_key=None):
        self.api_key = api_key or os.environ.get("ODDS_API_KEY")
        if not self.api_key:
            raise ValueError("ODDS_API_KEY not set. Pass api_key or set the ODDS_API_KEY environment variable.")
        self.sportsbook = sportsbook
        self.bookmaker_key = BOOKMAKER_MAP.get(sportsbook, sportsbook)

        # Quota tracking: se actualiza con cada request a la API
        # x-requests-remaining = creditos disponibles este mes
        # x-requests-used = creditos consumidos este mes
        self._quota_remaining = None
        self._quota_used = None

    def _update_quota(self, response: requests.Response):
        """Lee los headers de quota de la respuesta y actualiza el estado.

        The Odds API incluye en CADA respuesta HTTP:
          x-requests-remaining: cuantos creditos te quedan
          x-requests-used: cuantos has usado este mes

        Esto permite monitorear el consumo sin hacer requests adicionales.
        """
        remaining = response.headers.get("x-requests-remaining")
        used = response.headers.get("x-requests-used")

        if remaining is not None:
            self._quota_remaining = int(remaining)
        if used is not None:
            self._quota_used = int(used)

        # Log siempre el estado de quota (nivel DEBUG para no saturar)
        if self._quota_remaining is not None:
            logger.debug("Quota API: %d restantes, %d usados",
                         self._quota_remaining, self._quota_used or 0)

            # Warning si estamos bajos
            if self._quota_remaining < QUOTA_WARNING_THRESHOLD:
                logger.warning(
                    "⚠ Quota baja: %d creditos restantes (umbral: %d). "
                    "Considera reducir requests o esperar al proximo ciclo.",
                    self._quota_remaining, QUOTA_WARNING_THRESHOLD,
                )

    def get_quota(self) -> dict:
        """Retorna el estado actual de la quota de la API.

        Solo tiene datos despues de al menos 1 request exitoso.
        Util para mostrar en el dashboard o verificar antes de operaciones costosas.

        Returns:
            dict con 'remaining', 'used' y 'pct_used' (o None si no hay datos)
        """
        if self._quota_remaining is None:
            return {"remaining": None, "used": None, "pct_used": None}

        total = (self._quota_remaining or 0) + (self._quota_used or 0)
        pct_used = (self._quota_used / total * 100) if total > 0 else 0.0
        return {
            "remaining": self._quota_remaining,
            "used": self._quota_used,
            "pct_used": round(pct_used, 1),
        }

    def get_odds(self):
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': 'h2h,totals,spreads',
            'oddsFormat': 'american',
            'bookmakers': self.bookmaker_key,
        }
        resp = requests.get(ODDS_API_URL, params=params, timeout=15)
        resp.raise_for_status()
        self._update_quota(resp)
        events = resp.json()

        # Filtrar solo eventos de hoy (ET) — la API devuelve eventos futuros
        from datetime import datetime as _dt, timezone as _tz, timedelta as _td
        _et = _tz(_td(hours=-5))
        _today_et = _dt.now(_et).date()

        dict_res = {}
        for event in events:
            commence = event.get('commence_time', '')
            if commence:
                try:
                    event_date_et = _dt.fromisoformat(
                        commence.replace('Z', '+00:00')
                    ).astimezone(_et).date()
                    if event_date_et != _today_et:
                        continue
                except (ValueError, TypeError):
                    pass
            home_team = event['home_team'].replace("Los Angeles Clippers", "LA Clippers")
            away_team = event['away_team'].replace("Los Angeles Clippers", "LA Clippers")

            money_line_home = None
            money_line_away = None
            totals_value = None
            # Convención canónica del proyecto: spread del local (home spread).
            spread_home = None
            spread_home_price = None
            spread_away_price = None

            for bookmaker in event.get('bookmakers', []):
                if bookmaker['key'] != self.bookmaker_key:
                    continue
                for market in bookmaker.get('markets', []):
                    if market['key'] == 'h2h':
                        for outcome in market['outcomes']:
                            name = outcome['name'].replace("Los Angeles Clippers", "LA Clippers")
                            if name == home_team:
                                money_line_home = outcome['price']
                            elif name == away_team:
                                money_line_away = outcome['price']
                    elif market['key'] == 'totals':
                        for outcome in market['outcomes']:
                            if outcome['name'] == 'Over':
                                totals_value = outcome.get('point')
                    elif market['key'] == 'spreads':
                        for outcome in market['outcomes']:
                            name = outcome['name'].replace("Los Angeles Clippers", "LA Clippers")
                            if name == home_team:
                                spread_home = outcome.get('point')
                                spread_home_price = outcome.get('price')
                            elif name == away_team:
                                spread_away_price = outcome.get('price')

            if money_line_home is None or money_line_away is None or totals_value is None:
                continue

            dict_res[f"{home_team}:{away_team}"] = {
                'under_over_odds': totals_value,
                # spread < 0: local favorito | spread > 0: local underdog
                'spread': spread_home if spread_home is not None else 0.0,
                # Odds del spread (americanas). Default -110 si no disponible.
                'spread_home_odds': spread_home_price if spread_home_price is not None else -110,
                'spread_away_odds': spread_away_price if spread_away_price is not None else -110,
                home_team: {'money_line_odds': money_line_home},
                away_team: {'money_line_odds': money_line_away},
            }

        return dict_res

    def get_events(self):
        """Obtiene la lista de eventos NBA de hoy con sus IDs.

        Necesario para llamar get_player_props() ya que ese endpoint
        requiere un event_id específico (no disponible en get_odds()).

        Returns:
            list of dicts con claves:
              - 'id':        event_id (str, ej: "abc123")
              - 'home_team': nombre completo del equipo local
              - 'away_team': nombre completo del equipo visitante
              - 'key':       "home_team:away_team" (para matching con get_odds())
        """
        params = {
            'apiKey': self.api_key,
        }
        url = "https://api.the-odds-api.com/v4/sports/basketball_nba/events/"
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        self._update_quota(resp)
        events = resp.json()

        result = []
        for event in events:
            home = event['home_team'].replace("Los Angeles Clippers", "LA Clippers")
            away = event['away_team'].replace("Los Angeles Clippers", "LA Clippers")
            result.append({
                'id':        event['id'],
                'home_team': home,
                'away_team': away,
                'key':       f"{home}:{away}",
            })
        return result

    def get_player_props(self, event_id, markets=None):
        """Obtiene líneas de player props para un evento específico.

        The Odds API requiere un event_id para acceder a props de jugadores.
        Obtén los event_ids desde get_events().

        Args:
            event_id: ID del evento obtenido de get_events() (str)
            markets:  lista de mercados a pedir. Default: 6 básicas (6 créditos/evento).
                      Pasar ALL_PROP_MARKETS para incluir combos (11 créditos/evento).

        Returns:
            dict[player_name -> dict[market -> dict]]
            Retorna {} si no hay datos o si el plan no incluye props.
        """
        if markets is None:
            markets = BASIC_PROP_MARKETS

        url = ODDS_API_EVENT_URL.format(event_id=event_id)
        params = {
            'apiKey':      self.api_key,
            'regions':     'us',
            'markets':     ','.join(markets),
            'oddsFormat':  'american',
            'bookmakers':  self.bookmaker_key,
        }

        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            self._update_quota(resp)
            data = resp.json()
        except Exception as e:
            logger.warning("Error obteniendo props para event %s: %s", event_id, e)
            return {}

        # The Odds API retorna una lista de bookmakers, cada uno con markets
        # Estructura: data['bookmakers'][0]['markets'][i]['outcomes']
        # Cada outcome tiene: {'name': 'Nikola Jokić Over 27.5', 'price': -110, 'point': 27.5}
        result = {}

        market_to_key = {
            "player_points":   "points",
            "player_rebounds": "rebounds",
            "player_assists":  "assists",
            "player_threes":   "threes",
            "player_blocks":   "blocks",
            "player_steals":   "steals",
            "player_points_rebounds_assists": "points_rebounds_assists",
            "player_points_rebounds": "points_rebounds",
            "player_points_assists":  "points_assists",
            "player_rebounds_assists": "rebounds_assists",
            "player_blocks_steals":   "blocks_steals",
        }

        for bookmaker in data.get('bookmakers', []):
            if bookmaker['key'] != self.bookmaker_key:
                continue
            for market in bookmaker.get('markets', []):
                market_key = market_to_key.get(market['key'])
                if market_key is None:
                    continue
                for outcome in market.get('outcomes', []):
                    price = outcome.get('price')
                    line = outcome.get('point')
                    if price is None or line is None:
                        continue

                    # La API tiene 2 formatos de outcome:
                    # Nuevo: name="Over", description="Ben Sheppard"
                    # Viejo: name="Nikola Jokić Over 27.5"
                    name = outcome.get('name', '')
                    description = outcome.get('description', '')

                    if name in ('Over', 'Under') and description:
                        player_name = description
                        direction = name.lower()
                    else:
                        parts = name.rsplit(' ', 2)
                        if len(parts) < 3 or parts[1] not in ('Over', 'Under'):
                            continue
                        player_name = parts[0]
                        direction = parts[1].lower()

                    if player_name not in result:
                        result[player_name] = {}
                    if market_key not in result[player_name]:
                        result[player_name][market_key] = {"line": line}

                    result[player_name][market_key][f"{direction}_odds"] = price

        return result

    # ── Raw data: retorna respuesta cruda sin procesar (para arbitrage) ──

    def get_all_books_raw(self, markets="h2h,totals,spreads"):
        """Retorna la respuesta cruda de la API con odds de TODOS los books.

        A diferencia de get_best_odds() que procesa y extrae los mejores precios,
        este metodo retorna la lista de eventos tal cual viene de la API.
        Util para el scanner de arbitraje que necesita TODOS los precios
        de TODOS los books para cada mercado.

        Costo: 1 credito API (misma request que get_best_odds).

        Returns:
            list[dict]: eventos crudos de la API con todos los bookmakers.
        """
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': markets,
            'oddsFormat': 'american',
        }
        resp = requests.get(ODDS_API_URL, params=params, timeout=15)
        resp.raise_for_status()
        self._update_quota(resp)
        return resp.json()

    def get_all_books_props_raw(self, event_id, markets=None):
        """Retorna props crudas de TODOS los books para un evento.

        Costo: 6 creditos (basicas) o 11 (todas).

        Returns:
            dict: respuesta cruda de la API con bookmakers y markets.
        """
        if markets is None:
            markets = BASIC_PROP_MARKETS

        url = ODDS_API_EVENT_URL.format(event_id=event_id)
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': ','.join(markets),
            'oddsFormat': 'american',
        }

        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            self._update_quota(resp)
            return resp.json()
        except Exception as e:
            logger.warning("Error obteniendo raw props para event %s: %s", event_id, e)
            return {}

    # ── Line Shopping: comparar precios entre TODOS los sportsbooks ──────

    def get_best_odds(self):
        """Line shopping para odds de equipos: busca el mejor precio en cada mercado.

        ¿Que es line shopping?
          Es comparar precios entre sportsbooks, igual que comparar precios en tiendas.
          La diferencia entre -150 y -135 en moneyline es ~1-3% de EV por apuesta.

          Ejemplo:
            FanDuel:    Lakers ML -150 (apuestas $150 para ganar $100)
            DraftKings: Lakers ML -140 (apuestas $140 para ganar $100)
            BetMGM:     Lakers ML -135 (apuestas $135 para ganar $100)
            → Apuesta en BetMGM, ahorras $15 por cada $100 de ganancia

        ¿Cual odds es "mejor"?
          Para el apostador, el numero MAS ALTO siempre es mejor:
            - Favoritos: -135 > -150 (pagas menos)
            - Underdogs: +155 > +140 (ganas mas)

        Costo: 1 request (mismo que get_odds), pero sin filtrar por bookmaker.
        El API retorna TODOS los books disponibles en la region 'us'.

        Returns:
            dict igual que get_odds() pero con campos adicionales:
              - 'best_home_ml': {'odds': -135, 'book': 'betmgm'}
              - 'best_away_ml': {'odds': +155, 'book': 'draftkings'}
              - 'best_over':    {'odds': -105, 'book': 'fanduel', 'line': 220.5}
              - 'best_under':   {'odds': -105, 'book': 'caesars', 'line': 220.5}
        """
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': 'h2h,totals,spreads',
            'oddsFormat': 'american',
            # NO especificamos 'bookmakers' → retorna TODOS los disponibles
        }
        resp = requests.get(ODDS_API_URL, params=params, timeout=15)
        resp.raise_for_status()
        self._update_quota(resp)
        events = resp.json()

        # Filtrar solo eventos de hoy (ET) — la API devuelve eventos futuros
        from datetime import datetime as _dt, timezone as _tz, timedelta as _td
        _et = _tz(_td(hours=-5))
        _today_et = _dt.now(_et).date()

        dict_res = {}
        for event in events:
            commence = event.get('commence_time', '')
            if commence:
                try:
                    event_date_et = _dt.fromisoformat(
                        commence.replace('Z', '+00:00')
                    ).astimezone(_et).date()
                    if event_date_et != _today_et:
                        continue
                except (ValueError, TypeError):
                    pass

            home_team = event['home_team'].replace("Los Angeles Clippers", "LA Clippers")
            away_team = event['away_team'].replace("Los Angeles Clippers", "LA Clippers")

            # Rastrear el mejor precio por mercado entre todos los books
            # Para ML: el odds mas alto es mejor para el apostador
            best_home_ml = {"odds": None, "book": None}
            best_away_ml = {"odds": None, "book": None}
            best_over = {"odds": None, "book": None, "line": None}
            best_under = {"odds": None, "book": None, "line": None}

            # Datos del sportsbook preferido (para compatibilidad con el pipeline actual)
            primary_home_ml = None
            primary_away_ml = None
            primary_totals = None
            primary_spread = None

            for bookmaker in event.get('bookmakers', []):
                book_key = bookmaker['key']
                is_primary = (book_key == self.bookmaker_key)

                for market in bookmaker.get('markets', []):
                    if market['key'] == 'h2h':
                        for outcome in market['outcomes']:
                            name = outcome['name'].replace("Los Angeles Clippers", "LA Clippers")
                            price = outcome.get('price')
                            if price is None:
                                continue
                            if name == home_team:
                                if is_primary:
                                    primary_home_ml = price
                                if best_home_ml["odds"] is None or price > best_home_ml["odds"]:
                                    best_home_ml = {"odds": price, "book": book_key}
                            elif name == away_team:
                                if is_primary:
                                    primary_away_ml = price
                                if best_away_ml["odds"] is None or price > best_away_ml["odds"]:
                                    best_away_ml = {"odds": price, "book": book_key}

                    elif market['key'] == 'totals':
                        for outcome in market['outcomes']:
                            price = outcome.get('price')
                            line = outcome.get('point')
                            if price is None or line is None:
                                continue
                            if outcome['name'] == 'Over':
                                if is_primary:
                                    primary_totals = line
                                if best_over["odds"] is None or price > best_over["odds"]:
                                    best_over = {"odds": price, "book": book_key, "line": line}
                            elif outcome['name'] == 'Under':
                                if best_under["odds"] is None or price > best_under["odds"]:
                                    best_under = {"odds": price, "book": book_key, "line": line}

                    elif market['key'] == 'spreads':
                        for outcome in market['outcomes']:
                            name = outcome['name'].replace("Los Angeles Clippers", "LA Clippers")
                            if name == home_team and is_primary:
                                primary_spread = outcome.get('point')

            # Necesitamos al menos ML de algun book para incluir el partido
            if best_home_ml["odds"] is None or best_away_ml["odds"] is None:
                continue

            # Usar datos del book primario si disponibles, sino el mejor encontrado
            home_ml = primary_home_ml if primary_home_ml is not None else best_home_ml["odds"]
            away_ml = primary_away_ml if primary_away_ml is not None else best_away_ml["odds"]
            totals = primary_totals if primary_totals is not None else (best_over.get("line") or 0.0)
            # Convención canónica: home spread.
            spread = primary_spread if primary_spread is not None else 0.0

            if totals is None or totals == 0.0:
                continue

            dict_res[f"{home_team}:{away_team}"] = {
                'under_over_odds': totals,
                'spread': spread,
                home_team: {'money_line_odds': home_ml},
                away_team: {'money_line_odds': away_ml},
                # Line shopping: mejor precio encontrado entre todos los books
                'best_home_ml': best_home_ml,
                'best_away_ml': best_away_ml,
                'best_over': best_over,
                'best_under': best_under,
            }

        n_games = len(dict_res)
        if n_games > 0:
            logger.info("Line shopping: %d partidos, mejores precios encontrados", n_games)
        return dict_res

    def get_best_player_props(self, event_id, markets=None):
        """Line shopping para props: busca las mejores odds entre todos los books.

        Misma logica que get_player_props() pero sin filtrar por un solo bookmaker.
        Para cada jugador/stat, guarda:
          - El mejor over_odds (el mas alto = mejor para apostar OVER)
          - El mejor under_odds (el mas alto = mejor para apostar UNDER)
          - La linea del book primario (para consistencia con el modelo)

        Returns:
            dict[player_name -> dict[market -> dict]] igual que get_player_props()
            pero con campos adicionales:
              - 'best_over_odds': int (mejor odds para OVER entre todos los books)
              - 'best_over_book': str (nombre del book con mejor over)
              - 'best_under_odds': int (mejor odds para UNDER)
              - 'best_under_book': str
        """
        if markets is None:
            markets = BASIC_PROP_MARKETS

        url = ODDS_API_EVENT_URL.format(event_id=event_id)
        params = {
            'apiKey':      self.api_key,
            'regions':     'us',
            'markets':     ','.join(markets),
            'oddsFormat':  'american',
            # NO especificamos 'bookmakers' → retorna TODOS
        }

        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            self._update_quota(resp)
            data = resp.json()
        except Exception as e:
            logger.warning("Error obteniendo best props para event %s: %s", event_id, e)
            return {}

        result = {}

        market_to_key = {
            "player_points":   "points",
            "player_rebounds": "rebounds",
            "player_assists":  "assists",
            "player_threes":   "threes",
            "player_blocks":   "blocks",
            "player_steals":   "steals",
            "player_points_rebounds_assists": "points_rebounds_assists",
            "player_points_rebounds": "points_rebounds",
            "player_points_assists":  "points_assists",
            "player_rebounds_assists": "rebounds_assists",
            "player_blocks_steals":   "blocks_steals",
        }

        for bookmaker in data.get('bookmakers', []):
            book_key = bookmaker['key']
            is_primary = (book_key == self.bookmaker_key)

            for market in bookmaker.get('markets', []):
                market_key = market_to_key.get(market['key'])
                if market_key is None:
                    continue

                for outcome in market.get('outcomes', []):
                    price = outcome.get('price')
                    line = outcome.get('point')
                    if price is None or line is None:
                        continue

                    # La API tiene 2 formatos de outcome:
                    # Nuevo: name="Over", description="Ben Sheppard"
                    # Viejo: name="Nikola Jokić Over 27.5"
                    name = outcome.get('name', '')
                    description = outcome.get('description', '')

                    if name in ('Over', 'Under') and description:
                        player_name = description
                        direction = name.lower()
                    else:
                        parts = name.rsplit(' ', 2)
                        if len(parts) < 3 or parts[1] not in ('Over', 'Under'):
                            continue
                        player_name = parts[0]
                        direction = parts[1].lower()

                    if player_name not in result:
                        result[player_name] = {}
                    if market_key not in result[player_name]:
                        result[player_name][market_key] = {
                            "line": None,
                            "over_odds": None, "under_odds": None,
                            "best_over_odds": None, "best_over_book": None,
                            "best_under_odds": None, "best_under_book": None,
                        }

                    entry = result[player_name][market_key]

                    # Guardar linea del book primario (consistencia con modelo)
                    if is_primary:
                        entry["line"] = line
                        entry[f"{direction}_odds"] = price

                    # Rastrear el mejor precio (el mas alto siempre es mejor)
                    best_key = f"best_{direction}_odds"
                    best_book_key = f"best_{direction}_book"
                    if entry[best_key] is None or price > entry[best_key]:
                        entry[best_key] = price
                        entry[best_book_key] = book_key

        # Fallback: si no habia book primario, usar la linea del mejor book
        for player_data in result.values():
            for mkt_data in player_data.values():
                if mkt_data["line"] is None and mkt_data["best_over_odds"] is not None:
                    # Tomamos la linea que ya vimos (todos los books usan la misma o similar)
                    # No tenemos la linea del best book guardada, asi que dejamos None
                    pass
                if mkt_data["over_odds"] is None:
                    mkt_data["over_odds"] = mkt_data["best_over_odds"]
                if mkt_data["under_odds"] is None:
                    mkt_data["under_odds"] = mkt_data["best_under_odds"]

        return result
