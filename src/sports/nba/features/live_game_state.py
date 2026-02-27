"""
LiveGameState: abstracción sobre nba_api.live para live betting.

Provee 3 funciones públicas:
  get_live_scoreboard()     → lista de partidos de hoy con status/score/periodo
  get_live_box_score(id)    → stats de ambos equipos si el juego está en vivo
  estimate_possessions(...) → estimación de posesiones usadas

Diseño robusto:
  - Retry automático 3 veces con 5 segundos de espera
  - Retorna [] o None en vez de lanzar excepción → el caller puede continuar
  - Compatible con nba_api 1.5+

Concepto clave — gameStatus:
  1 = programado (aún no empieza)
  2 = en vivo (podemos acceder al BoxScore)
  3 = final

Por qué nba_api.live y no stats.nba.com:
  nba_api.live es el endpoint OFICIAL de la app NBA (datos en tiempo real, sin auth).
  stats.nba.com solo tiene datos históricos (end-of-game o siguiente día).
"""

import time

# Nota: nba_api.live es diferente a nba_api.stats
# Importamos de forma lazy para poder testear sin datos live
try:
    from nba_api.live.nba.endpoints.scoreboard import ScoreBoard
    from nba_api.live.nba.endpoints.boxscore import BoxScore
    from nba_api.live.nba.endpoints.playbyplay import PlayByPlay
    _NBA_API_AVAILABLE = True
except ImportError:
    _NBA_API_AVAILABLE = False


def _retry(fn, retries=3, delay=5):
    """Ejecuta fn() hasta `retries` veces con `delay` segundos entre intentos.

    nba_api falla ~10% del tiempo por rate limiting o timeouts de la NBA.
    Con 3 reintentos el éxito sube a ~99.9%.

    Returns: resultado de fn() o None si todos los intentos fallan.
    """
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt < retries:
                time.sleep(delay)
            else:
                # Último intento fallido — retornar None silenciosamente
                return None


def get_live_scoreboard() -> list[dict]:
    """Retorna todos los partidos de HOY con su status, score y periodo.

    Funciona aunque no haya juegos en vivo (retorna lista vacía o solo programados).
    Llama a la NBA API una vez por invocación — usa polling externo si necesitas actualizar.

    Returns:
        Lista de dicts con claves:
          game_id    (str) — ej: "0022500824"
          home_team  (str) — ej: "Denver Nuggets"
          away_team  (str) — ej: "Boston Celtics"
          status     (int) — 1=programado, 2=en vivo, 3=final
          period     (int) — cuarto actual (1-4, 5+ para OT)
          clock      (str) — "PT05M30.00S" = 5:30 restantes
          home_score (int)
          away_score (int)
    """
    if not _NBA_API_AVAILABLE:
        return []

    raw = _retry(lambda: ScoreBoard().get_dict())
    if raw is None:
        return []

    games = raw.get("scoreboard", {}).get("games", [])
    result = []

    for g in games:
        try:
            result.append({
                "game_id":    g["gameId"],
                "home_team":  g["homeTeam"]["teamName"] if g["homeTeam"].get("teamName") else g["homeTeam"]["teamCity"],
                "away_team":  g["awayTeam"]["teamName"] if g["awayTeam"].get("teamName") else g["awayTeam"]["teamCity"],
                "home_team_city": g["homeTeam"].get("teamCity", ""),
                "away_team_city": g["awayTeam"].get("teamCity", ""),
                "home_tricode": g["homeTeam"].get("teamTricode", ""),
                "away_tricode": g["awayTeam"].get("teamTricode", ""),
                "status":     g.get("gameStatus", 1),
                "period":     g.get("period", 0),
                "clock":      g.get("gameClock", ""),
                "home_score": g["homeTeam"].get("score", 0),
                "away_score": g["awayTeam"].get("score", 0),
            })
        except (KeyError, TypeError):
            continue

    return result


def get_live_box_score(game_id: str) -> dict | None:
    """Retorna stats agregadas de ambos equipos si el partido está en vivo.

    Requiere que el juego tenga gameStatus == 2 (en vivo).
    Si el juego no está live o la API falla → retorna None.

    Args:
        game_id: ID del partido (ej: "0022500824") obtenido de get_live_scoreboard()

    Returns:
        dict con claves 'home' y 'away', cada una con:
          pts          (int)   — puntos anotados hasta ahora
          fgm, fga     (int)   — tiros de campo (intentados y encestados)
          fg3m, fg3a   (int)   — triples
          ftm, fta     (int)   — tiros libres
          tov          (int)   — pérdidas de balón (turnovers)
          reb          (int)   — rebotes totales
          possessions  (float) — estimación de posesiones

        None si el juego no es live o si hay error.
    """
    if not _NBA_API_AVAILABLE:
        return None

    raw = _retry(lambda: BoxScore(game_id).get_dict())
    if raw is None:
        return None

    try:
        game = raw.get("game", {})
        if not game:
            return None

        home = game.get("homeTeam", {})
        away = game.get("awayTeam", {})

        def _parse_team(t: dict) -> dict:
            stats = t.get("statistics", {})
            fgm  = int(stats.get("fieldGoalsMade",     0))
            fga  = int(stats.get("fieldGoalsAttempted", 0))
            fg3m = int(stats.get("threePointersMade",   0))
            fg3a = int(stats.get("threePointersAttempted", 0))
            ftm  = int(stats.get("freeThrowsMade",      0))
            fta  = int(stats.get("freeThrowsAttempted", 0))
            tov  = int(stats.get("turnovers",           0))
            reb  = int(stats.get("reboundsTotal",       0))
            pts  = int(stats.get("points",              0))
            poss = estimate_possessions(fgm, fga, tov, fta)
            return {
                "pts": pts, "fgm": fgm, "fga": fga,
                "fg3m": fg3m, "fg3a": fg3a,
                "ftm": ftm, "fta": fta,
                "tov": tov, "reb": reb,
                "possessions": poss,
            }

        return {
            "home": _parse_team(home),
            "away": _parse_team(away),
        }

    except (KeyError, TypeError, ValueError):
        return None


def estimate_possessions(fgm: int, fga: int, tov: int, fta: int) -> float:
    """Estima las posesiones usadas por un equipo.

    Fórmula estándar (Dean Oliver, Basketball on Paper):
        POSS = FGA + TOV + 0.44 × FTA

    Nota: La versión completa incluye ORB (FGA - FGM + ORB + TOV + 0.44×FTA),
    pero como no siempre tenemos ORB separado, usamos la aproximación sin él.
    El error es < 5% típicamente.

    Por qué 0.44 para FTA:
        En promedio un jugador "usa" 0.44 posesiones por tiro libre.
        Si hace 2 tiros libres (falta en jugada de 2), gasta 1 posesión.
        Si hace 1 tiro libre (falta en 3 = 3 intentos), gasta 0.33 posesiones.
        El promedio ponderado histórico NBA = 0.44.

    Args:
        fgm: tiros de campo anotados (no usado en la fórmula básica pero documentado)
        fga: tiros de campo intentados
        tov: turnovers (pérdidas de balón)
        fta: tiros libres intentados

    Returns:
        Número estimado de posesiones (float)
    """
    return float(fga + tov + 0.44 * fta)


def get_live_play_by_play(game_id: str) -> list[dict]:
    """Retorna las actions del play-by-play para un partido en vivo.

    Usa nba_api.live.PlayByPlay, que devuelve todas las jugadas del partido
    desde el inicio hasta el momento actual.

    Args:
        game_id: ID del partido (ej: "0022500824")

    Returns:
        Lista de dicts con las actions del PBP, o [] si no disponible.
        Cada action tiene: period, clock, teamTricode, actionType, subType,
        scoreHome, scoreAway, description, etc.
    """
    if not _NBA_API_AVAILABLE:
        return []

    raw = _retry(lambda: PlayByPlay(game_id).get_dict())
    if raw is None:
        return []

    try:
        return raw.get("game", {}).get("actions", [])
    except (KeyError, TypeError):
        return []


def format_clock(clock_str: str) -> str:
    """Convierte 'PT05M30.00S' a '5:30' para mostrar en pantalla.

    Args:
        clock_str: formato ISO de reloj de la NBA API (ej: "PT05M30.00S")

    Returns:
        Tiempo legible (ej: "5:30") o el string original si no parsea
    """
    try:
        # "PT05M30.00S" → minutos=5, segundos=30
        import re
        m = re.match(r"PT(\d+)M(\d+)", clock_str)
        if m:
            mins, secs = int(m.group(1)), int(m.group(2))
            return f"{mins}:{secs:02d}"
    except Exception:
        pass
    return clock_str
