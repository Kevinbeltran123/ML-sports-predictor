"""
Features de fatiga extendida para el modelo NBA ML.

Captura la densidad del calendario reciente de un equipo:
    - THREE_IN_FOUR: 1 si el equipo jugó 3 partidos en los últimos 4 días
    - TWO_IN_THREE: 1 si el equipo jugó 2 partidos en los últimos 3 días

Features de viaje (Fase 5.3):
    - TRAVEL_DIST_HOME/AWAY: millas desde la arena del último partido a la actual
    - TZ_CHANGE_HOME/AWAY: cambio de timezone absoluto (0-3 horas)

Features extendidos (Fase 5.4):
    - TZ_CHANGE_SIGNED_HOME/AWAY: dirección del cambio de timezone (+east/-west)
    - ALTITUDE_GAME: 1 si se juega en Denver (5,280 ft)
    - GAMES_IN_7_HOME/AWAY: partidos en los últimos 7 días
    - GAMES_IN_14_HOME/AWAY: partidos en los últimos 14 días
    - TRAVEL_7D_HOME/AWAY: millas acumuladas en 7 días

¿Por qué el viaje importa?
    Un equipo que voló Portland→Miami (~2,700 millas, 3h timezone) llega MUCHO
    más fatigado que uno que viajó Boston→Brooklyn (~200 millas, mismo timezone).
    Days_Rest=1 en ambos casos, pero el impacto es muy diferente.

    Estudios muestran que viajar al oeste (ganar horas) afecta menos que viajar
    al este (perder horas), porque el reloj circadiano se ajusta ~1h/día al oeste
    pero solo ~0.7h/día al este.

Usado por Create_Games.py (entrenamiento) y main.py (predicción).
"""

import math
from collections import defaultdict
from datetime import datetime, timedelta

import pandas as pd

# Aliases históricos de nombres de equipos
_TEAM_ALIASES = {
    "Los Angeles Clippers": "LA Clippers",
    "Charlotte Bobcats": "Charlotte Hornets",
    "New Orleans Hornets": "New Orleans Pelicans",
    "New Orleans/Oklahoma City Hornets": "New Orleans Pelicans",
    "New Jersey Nets": "Brooklyn Nets",
}


def _normalize_team(name):
    """Normaliza nombre de equipo usando aliases históricos."""
    return _TEAM_ALIASES.get(name, name)


def build_team_schedule(odds_con):
    """Construye calendario completo por equipo desde OddsData.sqlite.

    Itera TODAS las temporadas en OddsData para tener un historial completo
    de fechas de partido por equipo. Esto permite calcular fatiga para
    cualquier partido en el dataset.

    Args:
        odds_con: conexión SQLite a OddsData.sqlite

    Returns:
        dict[team_name] -> sorted list[datetime.date]
        Cada lista contiene las fechas en que el equipo jugó, en orden.
    """
    # Obtener todas las tablas de odds
    cursor = odds_con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = [row[0] for row in cursor.fetchall()]

    schedule = defaultdict(set)

    for table_name in tables:
        if not table_name.startswith("odds_") and "-" not in table_name:
            continue

        try:
            rows = odds_con.execute(
                f'SELECT Date, Home, Away FROM "{table_name}"'
            ).fetchall()
        except Exception:
            continue

        for row in rows:
            date_val = row[0]
            home = row[1]
            away = row[2]

            # Normalizar fecha a datetime.date
            if isinstance(date_val, str):
                try:
                    dt = datetime.strptime(date_val, "%Y-%m-%d").date()
                except ValueError:
                    try:
                        dt = datetime.strptime(date_val[:10], "%Y-%m-%d").date()
                    except ValueError:
                        continue
            elif hasattr(date_val, "date"):
                dt = date_val.date() if callable(getattr(date_val, "date")) else date_val
            else:
                continue

            # Normalizar nombres de equipo
            home_norm = _normalize_team(home)
            away_norm = _normalize_team(away)

            schedule[home_norm].add(dt)
            schedule[away_norm].add(dt)

            # Agregar aliases para que el lookup funcione con ambas versiones
            if home != home_norm:
                schedule[home].add(dt)
            if away != away_norm:
                schedule[away].add(dt)

    # Convertir sets a listas ordenadas para búsqueda eficiente
    return {team: sorted(dates) for team, dates in schedule.items()}


def compute_fatigue(team_schedule, team_name, game_date):
    """Calcula features de fatiga para un equipo en una fecha dada.

    THREE_IN_FOUR = 1 si el equipo jugó al menos 3 partidos en los últimos 4 días
    (incluyendo el partido de hoy).

    TWO_IN_THREE = 1 si el equipo jugó al menos 2 partidos en los últimos 3 días
    (incluyendo el partido de hoy).

    Ejemplo con THREE_IN_FOUR:
        Si hoy es Miércoles y el equipo también jugó Lunes y Domingo,
        eso son 3 partidos en 4 días (Dom, Lun, Mié) → THREE_IN_FOUR = 1.

    Args:
        team_schedule: dict[team] -> sorted list[date] de build_team_schedule()
        team_name: nombre del equipo
        game_date: fecha del partido (str 'YYYY-MM-DD' o datetime.date)

    Returns:
        dict con THREE_IN_FOUR (0/1) y TWO_IN_THREE (0/1)
    """
    # Normalizar fecha
    if isinstance(game_date, str):
        try:
            gd = datetime.strptime(game_date, "%Y-%m-%d").date()
        except ValueError:
            return {"THREE_IN_FOUR": 0, "TWO_IN_THREE": 0}
    else:
        gd = game_date

    # Buscar equipo (intentar alias si no se encuentra)
    dates = team_schedule.get(team_name)
    if dates is None:
        team_norm = _normalize_team(team_name)
        dates = team_schedule.get(team_norm, [])

    if not dates:
        return {"THREE_IN_FOUR": 0, "TWO_IN_THREE": 0}

    # Contar partidos en los últimos 4 días (gd-3 a gd, inclusivo)
    # Usando búsqueda lineal inversa (más rápido para listas ordenadas
    # cuando buscamos cerca del final)
    window_4_start = gd - timedelta(days=3)
    window_3_start = gd - timedelta(days=2)

    count_4 = 0  # partidos en ventana de 4 días
    count_3 = 0  # partidos en ventana de 3 días

    # Buscar desde el final (más eficiente para fechas recientes)
    for d in reversed(dates):
        if d > gd:
            continue
        if d < window_4_start:
            break
        count_4 += 1
        if d >= window_3_start:
            count_3 += 1

    return {
        "THREE_IN_FOUR": 1 if count_4 >= 3 else 0,
        "TWO_IN_THREE": 1 if count_3 >= 2 else 0,
    }


def get_game_fatigue(team_schedule, game_date, home_team, away_team):
    """Calcula features de fatiga para un partido completo.

    Devuelve 4 features: THREE_IN_FOUR y TWO_IN_THREE para ambos equipos.

    Args:
        team_schedule: de build_team_schedule()
        game_date: str 'YYYY-MM-DD'
        home_team: nombre del equipo local
        away_team: nombre del equipo visitante

    Returns:
        dict con 4 keys: THREE_IN_FOUR_HOME, THREE_IN_FOUR_AWAY,
                         TWO_IN_THREE_HOME, TWO_IN_THREE_AWAY
    """
    home_fatigue = compute_fatigue(team_schedule, home_team, game_date)
    away_fatigue = compute_fatigue(team_schedule, away_team, game_date)

    return {
        "THREE_IN_FOUR_HOME": home_fatigue["THREE_IN_FOUR"],
        "THREE_IN_FOUR_AWAY": away_fatigue["THREE_IN_FOUR"],
        "TWO_IN_THREE_HOME": home_fatigue["TWO_IN_THREE"],
        "TWO_IN_THREE_AWAY": away_fatigue["TWO_IN_THREE"],
    }


def add_fatigue_to_frame(frame, fatigue_features_list):
    """Agrega columnas de fatiga al DataFrame de juegos.

    Mismo patrón que add_availability_to_frame() en InjuryImpact.py.

    Args:
        frame: DataFrame con los juegos (ya construido)
        fatigue_features_list: list of dicts de get_game_fatigue()

    Returns:
        DataFrame con 4 columnas nuevas de fatiga
    """
    if not fatigue_features_list:
        return frame

    fatigue_df = pd.DataFrame(fatigue_features_list, index=frame.index)
    result = pd.concat([frame, fatigue_df], axis=1)
    return result


# =====================================================================
# Fase 5.3: Travel Distance + Timezone Change
# =====================================================================

# Coordenadas de las 30 arenas NBA (lat, lon) + offset de timezone vs Eastern
# tz_offset: 0=ET, -1=CT, -2=MT, -3=PT
# Las coordenadas son del centro de la arena (Google Maps)
NBA_ARENAS = {
    # --- Eastern Time (ET) ---
    "Atlanta Hawks":          (33.757, -84.396, 0),
    "Boston Celtics":         (42.366, -71.062, 0),
    "Brooklyn Nets":          (40.683, -73.975, 0),
    "Charlotte Hornets":      (35.225, -80.839, 0),
    "Cleveland Cavaliers":    (41.497, -81.688, 0),
    "Detroit Pistons":        (42.341, -83.055, 0),
    "Indiana Pacers":         (39.764, -86.156, 0),
    "Miami Heat":             (25.781, -80.187, 0),
    "New York Knicks":        (40.751, -73.994, 0),
    "Orlando Magic":          (28.539, -81.384, 0),
    "Philadelphia 76ers":     (39.901, -75.172, 0),
    "Toronto Raptors":        (43.643, -79.379, 0),
    "Washington Wizards":     (38.898, -77.021, 0),
    # --- Central Time (CT) ---
    "Chicago Bulls":          (41.881, -87.674, -1),
    "Dallas Mavericks":       (32.790, -96.810, -1),
    "Houston Rockets":        (29.751, -95.362, -1),
    "Memphis Grizzlies":      (35.138, -90.051, -1),
    "Milwaukee Bucks":        (43.045, -87.917, -1),
    "Minnesota Timberwolves": (44.980, -93.276, -1),
    "New Orleans Pelicans":   (29.949, -90.082, -1),
    "Oklahoma City Thunder":  (35.463, -97.515, -1),
    "San Antonio Spurs":      (29.427, -98.438, -1),
    # --- Mountain Time (MT) ---
    "Denver Nuggets":         (39.749, -105.008, -2),
    "Phoenix Suns":           (33.446, -112.071, -2),
    "Utah Jazz":              (40.768, -111.901, -2),
    # --- Pacific Time (PT) ---
    "Golden State Warriors":  (37.768, -122.388, -3),
    "LA Clippers":            (33.426, -118.261, -3),
    "Los Angeles Lakers":     (34.043, -118.267, -3),
    "Portland Trail Blazers": (45.532, -122.667, -3),
    "Sacramento Kings":       (38.580, -121.500, -3),
}

# Aliases que mapean nombres históricos a su arena actual
# (o a la arena más cercana si el equipo cambió de nombre)
_ARENA_ALIASES = {
    "Los Angeles Clippers": "LA Clippers",
    "Charlotte Bobcats": "Charlotte Hornets",
    "New Orleans Hornets": "New Orleans Pelicans",
    "New Orleans/Oklahoma City Hornets": "New Orleans Pelicans",
    "New Jersey Nets": "Brooklyn Nets",
}

# Radio de la Tierra en millas
_EARTH_RADIUS_MI = 3958.8


def haversine_miles(lat1, lon1, lat2, lon2):
    """Distancia en millas entre dos puntos (lat/lon en grados).

    Formula de Haversine:
        a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
        c = 2 × arcsin(√a)
        d = R × c

    Es la distancia "en línea recta" sobre la superficie de la Tierra.
    Para viajes en avión es una buena aproximación (error < 0.5%).
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_MI * math.asin(math.sqrt(a))


def _get_arena(team_name):
    """Devuelve (lat, lon, tz_offset) de la arena de un equipo.

    Busca primero el nombre exacto, luego aliases.
    Returns None si no se encuentra.
    """
    arena = NBA_ARENAS.get(team_name)
    if arena:
        return arena
    alias = _ARENA_ALIASES.get(team_name)
    if alias:
        return NBA_ARENAS.get(alias)
    return None


def build_team_travel_schedule(odds_con):
    """Construye calendario con ubicación para calcular distancia de viaje.

    Diferente de build_team_schedule() (que solo guarda fechas):
    aquí guardamos (fecha, equipo_anfitrión) para saber DÓNDE
    jugó cada equipo en cada fecha.

    Args:
        odds_con: conexión SQLite a OddsData.sqlite

    Returns:
        dict[team_name] -> sorted list[(datetime.date, host_team)]
        host_team = equipo cuya arena se usó (el local del partido)
    """
    cursor = odds_con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = [row[0] for row in cursor.fetchall()]

    # Usamos lista en vez de set porque necesitamos tuplas (date, host)
    schedule = defaultdict(list)
    seen = defaultdict(set)  # Para evitar duplicados

    for table_name in tables:
        if not table_name.startswith("odds_") and "-" not in table_name:
            continue

        try:
            rows = odds_con.execute(
                f'SELECT Date, Home, Away FROM "{table_name}"'
            ).fetchall()
        except Exception:
            continue

        for row in rows:
            date_val = row[0]
            home = _normalize_team(row[1])
            away = _normalize_team(row[2])

            # Normalizar fecha
            if isinstance(date_val, str):
                try:
                    dt = datetime.strptime(date_val, "%Y-%m-%d").date()
                except ValueError:
                    try:
                        dt = datetime.strptime(date_val[:10], "%Y-%m-%d").date()
                    except ValueError:
                        continue
            elif hasattr(date_val, "date"):
                dt = date_val.date() if callable(getattr(date_val, "date")) else date_val
            else:
                continue

            # Home team: estaba en su propia arena
            key_h = (home, dt, home)
            if key_h not in seen[home]:
                seen[home].add(key_h)
                schedule[home].append((dt, home))

            # Away team: estaba en la arena del home
            key_a = (away, dt, home)
            if key_a not in seen[away]:
                seen[away].add(key_a)
                schedule[away].append((dt, home))

    # Ordenar por fecha
    return {team: sorted(games, key=lambda x: x[0])
            for team, games in schedule.items()}


def compute_travel_features(travel_schedule, team_name, game_date,
                            is_home, opponent):
    """Calcula distancia de viaje y cambio de timezone para un equipo.

    Lógica:
        1. Encontrar el último partido ANTES de game_date en el calendario
        2. Determinar dónde fue ese partido (qué arena)
        3. Determinar dónde es el partido actual
        4. Calcular distancia Haversine + diferencia de timezone

    Args:
        travel_schedule: de build_team_travel_schedule()
        team_name: nombre del equipo
        game_date: str 'YYYY-MM-DD' o datetime.date
        is_home: True si el equipo juega de local
        opponent: nombre del oponente (para saber dónde juegan si es away)

    Returns:
        dict con TRAVEL_DIST (millas) y TZ_CHANGE (horas absolutas)
    """
    default = {"TRAVEL_DIST": 0.0, "TZ_CHANGE": 0}

    # Normalizar fecha
    if isinstance(game_date, str):
        try:
            gd = datetime.strptime(game_date, "%Y-%m-%d").date()
        except ValueError:
            return default
    else:
        gd = game_date

    # Arena del partido actual
    # Si es home → su propia arena. Si es away → arena del oponente.
    if is_home:
        current_arena = _get_arena(team_name)
    else:
        current_arena = _get_arena(opponent)

    if current_arena is None:
        return default

    # Buscar calendario del equipo
    team_norm = _normalize_team(team_name)
    games = travel_schedule.get(team_norm)
    if games is None:
        games = travel_schedule.get(team_name, [])

    if not games:
        return default

    # Encontrar el partido anterior (último antes de game_date)
    prev_host = None
    for dt, host in reversed(games):
        if dt < gd:
            prev_host = host
            break

    if prev_host is None:
        # No hay partido previo → inicio de temporada, sin viaje
        return default

    # Arena del partido anterior
    prev_arena = _get_arena(prev_host)
    if prev_arena is None:
        return default

    # Calcular distancia en millas
    dist = haversine_miles(
        prev_arena[0], prev_arena[1],
        current_arena[0], current_arena[1],
    )

    # Cambio de timezone (valor absoluto)
    tz_change = abs(current_arena[2] - prev_arena[2])

    return {
        "TRAVEL_DIST": round(dist, 1),
        "TZ_CHANGE": tz_change,
    }


def get_game_travel(travel_schedule, game_date, home_team, away_team):
    """Calcula features de viaje para un partido completo.

    Args:
        travel_schedule: de build_team_travel_schedule()
        game_date: str 'YYYY-MM-DD'
        home_team: nombre del equipo local
        away_team: nombre del equipo visitante

    Returns:
        dict con 4 keys: TRAVEL_DIST_HOME, TRAVEL_DIST_AWAY,
                         TZ_CHANGE_HOME, TZ_CHANGE_AWAY
    """
    home_travel = compute_travel_features(
        travel_schedule, home_team, game_date,
        is_home=True, opponent=away_team,
    )
    away_travel = compute_travel_features(
        travel_schedule, away_team, game_date,
        is_home=False, opponent=home_team,
    )

    return {
        "TRAVEL_DIST_HOME": home_travel["TRAVEL_DIST"],
        "TRAVEL_DIST_AWAY": away_travel["TRAVEL_DIST"],
        "TZ_CHANGE_HOME": home_travel["TZ_CHANGE"],
        "TZ_CHANGE_AWAY": away_travel["TZ_CHANGE"],
    }


def add_travel_to_frame(frame, travel_features_list):
    """Agrega columnas de viaje al DataFrame de juegos.

    Args:
        frame: DataFrame con los juegos (ya construido)
        travel_features_list: list of dicts de get_game_travel()

    Returns:
        DataFrame con 4 columnas nuevas de viaje
    """
    if not travel_features_list:
        return frame

    travel_df = pd.DataFrame(travel_features_list, index=frame.index)
    return pd.concat([frame, travel_df], axis=1)


# =====================================================================
# Fase 5.4: Extended Travel Features (TZ Signed, Altitude, Density)
# =====================================================================

# Denver es la única ciudad NBA a altitud significativa (5,280 ft / 1,609 m).
# La menor densidad de oxígeno reduce el rendimiento aeróbico de visitantes
# que no están aclimatados (~3-5% reducción en VO2max).
_ALTITUDE_TEAM = "Denver Nuggets"


def compute_travel_features_v2(travel_schedule, team_name, game_date,
                                is_home, opponent):
    """Features extendidos de viaje: TZ con signo + altitud.

    TZ_CHANGE_SIGNED: dirección del cambio de timezone.
        - Positivo = viajó al este (pierde horas → peor para rendimiento)
        - Negativo = viajó al oeste (gana horas → menor impacto)
        - Convención tz_offset: ET=0, CT=-1, MT=-2, PT=-3
        - Ejemplo: Portland(-3) → Miami(0) = 0 - (-3) = +3 (viajó al este)
        - Ejemplo: Miami(0) → Portland(-3) = -3 - 0 = -3 (viajó al oeste)
        Estudios de cronobiología muestran que el reloj circadiano se ajusta
        ~1h/día viajando al oeste pero solo ~0.7h/día al este, por eso la
        dirección importa.

    IS_AT_ALTITUDE: 1 si el partido se juega en Denver (5,280 ft).
        Solo Denver tiene altitud significativa en la NBA.

    Args:
        travel_schedule: de build_team_travel_schedule()
        team_name: nombre del equipo
        game_date: str 'YYYY-MM-DD' o datetime.date
        is_home: True si el equipo juega de local
        opponent: nombre del oponente

    Returns:
        dict con TZ_CHANGE_SIGNED (int) e IS_AT_ALTITUDE (0/1)
    """
    default = {"TZ_CHANGE_SIGNED": 0, "IS_AT_ALTITUDE": 0}

    # Normalizar fecha
    if isinstance(game_date, str):
        try:
            gd = datetime.strptime(game_date, "%Y-%m-%d").date()
        except ValueError:
            return default
    else:
        gd = game_date

    # Arena del partido actual
    # Si es home → su propia arena. Si es away → arena del oponente.
    if is_home:
        current_arena = _get_arena(team_name)
    else:
        current_arena = _get_arena(opponent)

    if current_arena is None:
        return default

    # Determinar si el partido es en Denver (altitud)
    # El equipo anfitrión es quien define la arena del partido
    host_team = _normalize_team(team_name) if is_home else _normalize_team(opponent)
    is_altitude = 1 if host_team == _ALTITUDE_TEAM else 0

    # Buscar calendario del equipo para encontrar partido anterior
    team_norm = _normalize_team(team_name)
    games = travel_schedule.get(team_norm)
    if games is None:
        games = travel_schedule.get(team_name, [])

    if not games:
        # Sin historial → solo devolver altitud, sin cambio de TZ
        return {"TZ_CHANGE_SIGNED": 0, "IS_AT_ALTITUDE": is_altitude}

    # Encontrar el partido anterior (último antes de game_date)
    prev_host = None
    for dt, host in reversed(games):
        if dt < gd:
            prev_host = host
            break

    if prev_host is None:
        # No hay partido previo → inicio de temporada, sin cambio de TZ
        return {"TZ_CHANGE_SIGNED": 0, "IS_AT_ALTITUDE": is_altitude}

    # Arena del partido anterior
    prev_arena = _get_arena(prev_host)
    if prev_arena is None:
        return {"TZ_CHANGE_SIGNED": 0, "IS_AT_ALTITUDE": is_altitude}

    # Cambio de timezone CON SIGNO
    # current - prev: positivo = viajó al este, negativo = viajó al oeste
    tz_signed = current_arena[2] - prev_arena[2]

    return {
        "TZ_CHANGE_SIGNED": tz_signed,
        "IS_AT_ALTITUDE": is_altitude,
    }


def compute_schedule_density(team_schedule, travel_schedule, team_name,
                              game_date):
    """Features de densidad de calendario: carga acumulada reciente.

    GAMES_IN_7: partidos jugados en los últimos 7 días (sin contar hoy).
        Rango típico: 0-4. Un equipo con 4 partidos en 7 días está
        significativamente más fatigado que uno con 2.

    GAMES_IN_14: partidos jugados en los últimos 14 días (sin contar hoy).
        Rango típico: 0-8. Captura fatiga acumulada de mediano plazo.

    TRAVEL_7D: millas totales viajadas en los últimos 7 días.
        Suma las distancias Haversine entre arenas consecutivas para
        todos los trayectos dentro de la ventana de 7 días.
        Ejemplo: un equipo que hizo BOS→MIA→LAL en 7 días acumula
        ~1,200 + ~2,300 = ~3,500 millas.

    Args:
        team_schedule: dict[team] -> sorted list[date] de build_team_schedule()
        travel_schedule: dict[team] -> sorted list[(date, host)] de
                         build_team_travel_schedule()
        team_name: nombre del equipo
        game_date: str 'YYYY-MM-DD' o datetime.date

    Returns:
        dict con GAMES_IN_7 (int), GAMES_IN_14 (int), TRAVEL_7D (float millas)
    """
    default = {"GAMES_IN_7": 0, "GAMES_IN_14": 0, "TRAVEL_7D": 0.0}

    # Normalizar fecha
    if isinstance(game_date, str):
        try:
            gd = datetime.strptime(game_date, "%Y-%m-%d").date()
        except ValueError:
            return default
    else:
        gd = game_date

    # --- Contar partidos en ventanas de 7 y 14 días ---
    # Usamos team_schedule (solo fechas) para conteo rápido
    team_norm = _normalize_team(team_name)
    dates = team_schedule.get(team_norm)
    if dates is None:
        dates = team_schedule.get(team_name, [])

    window_7_start = gd - timedelta(days=7)    # últimos 7 días (sin hoy)
    window_14_start = gd - timedelta(days=14)   # últimos 14 días (sin hoy)

    games_7 = 0
    games_14 = 0

    # Búsqueda inversa: más eficiente para fechas recientes
    for d in reversed(dates):
        if d >= gd:
            # No contar el partido de hoy ni futuros
            continue
        if d < window_14_start:
            # Ya pasamos la ventana de 14 días, salir
            break
        # Está dentro de los 14 días
        games_14 += 1
        if d > window_7_start:
            # También está dentro de los 7 días (estrictamente después del inicio)
            games_7 += 1

    # --- Calcular millas acumuladas en 7 días ---
    # Usamos travel_schedule para saber DÓNDE jugó cada partido
    travel_games = travel_schedule.get(team_norm)
    if travel_games is None:
        travel_games = travel_schedule.get(team_name, [])

    # Recolectar partidos en la ventana de 7 días (sin hoy) + el último
    # partido ANTES de la ventana como punto de partida del primer viaje.
    # Necesitamos saber de dónde venía el equipo para calcular la primera
    # distancia dentro de la ventana.
    recent_hosts = []
    for dt, host in reversed(travel_games):
        if dt >= gd:
            # No incluir el partido de hoy ni futuros
            continue
        if dt <= window_7_start:
            # Este partido está fuera de la ventana, pero lo incluimos
            # como punto de origen del primer viaje dentro de la ventana
            recent_hosts.append(host)
            break
        recent_hosts.append(host)

    # Revertir para tener orden cronológico
    recent_hosts.reverse()

    # Sumar distancias entre arenas consecutivas
    travel_miles = 0.0
    for i in range(1, len(recent_hosts)):
        arena_from = _get_arena(recent_hosts[i - 1])
        arena_to = _get_arena(recent_hosts[i])
        if arena_from is not None and arena_to is not None:
            travel_miles += haversine_miles(
                arena_from[0], arena_from[1],
                arena_to[0], arena_to[1],
            )

    return {
        "GAMES_IN_7": games_7,
        "GAMES_IN_14": games_14,
        "TRAVEL_7D": round(travel_miles, 1),
    }


def get_game_extended_fatigue(team_schedule, travel_schedule, game_date,
                               home_team, away_team):
    """Calcula TODAS las features extendidas de fatiga para un partido.

    Combina:
        - TZ_CHANGE_SIGNED (por equipo): dirección del cambio de timezone
        - ALTITUDE_GAME (nivel de partido): 1 si se juega en Denver
        - GAMES_IN_7 / GAMES_IN_14 (por equipo): densidad de calendario
        - TRAVEL_7D (por equipo): millas acumuladas en 7 días

    Total: 9 features nuevas por partido.

    Args:
        team_schedule: de build_team_schedule()
        travel_schedule: de build_team_travel_schedule()
        game_date: str 'YYYY-MM-DD' o datetime.date
        home_team: nombre del equipo local
        away_team: nombre del equipo visitante

    Returns:
        dict con 9 keys:
            TZ_CHANGE_SIGNED_HOME, TZ_CHANGE_SIGNED_AWAY,
            ALTITUDE_GAME,
            GAMES_IN_7_HOME, GAMES_IN_7_AWAY,
            GAMES_IN_14_HOME, GAMES_IN_14_AWAY,
            TRAVEL_7D_HOME, TRAVEL_7D_AWAY
    """
    # Features de viaje extendidos (TZ signed + altitud)
    home_travel_v2 = compute_travel_features_v2(
        travel_schedule, home_team, game_date,
        is_home=True, opponent=away_team,
    )
    away_travel_v2 = compute_travel_features_v2(
        travel_schedule, away_team, game_date,
        is_home=False, opponent=home_team,
    )

    # ALTITUDE_GAME es a nivel de partido: 1 si se juega en Denver.
    # Ambas llamadas a compute_travel_features_v2 dan el mismo resultado
    # para IS_AT_ALTITUDE cuando el host es el home_team (que determina
    # la arena del partido). Usamos el valor del home.
    altitude_game = home_travel_v2["IS_AT_ALTITUDE"]

    # Densidad de calendario (por equipo)
    home_density = compute_schedule_density(
        team_schedule, travel_schedule, home_team, game_date,
    )
    away_density = compute_schedule_density(
        team_schedule, travel_schedule, away_team, game_date,
    )

    return {
        # Dirección del cambio de timezone
        "TZ_CHANGE_SIGNED_HOME": home_travel_v2["TZ_CHANGE_SIGNED"],
        "TZ_CHANGE_SIGNED_AWAY": away_travel_v2["TZ_CHANGE_SIGNED"],
        # Altitud (nivel de partido)
        "ALTITUDE_GAME": altitude_game,
        # Densidad de calendario
        "GAMES_IN_7_HOME": home_density["GAMES_IN_7"],
        "GAMES_IN_7_AWAY": away_density["GAMES_IN_7"],
        "GAMES_IN_14_HOME": home_density["GAMES_IN_14"],
        "GAMES_IN_14_AWAY": away_density["GAMES_IN_14"],
        # Millas acumuladas en 7 días
        "TRAVEL_7D_HOME": home_density["TRAVEL_7D"],
        "TRAVEL_7D_AWAY": away_density["TRAVEL_7D"],
    }


def add_extended_fatigue_to_frame(frame, features_list):
    """Agrega columnas de fatiga extendida al DataFrame de juegos.

    Mismo patrón que add_fatigue_to_frame() y add_travel_to_frame().

    Args:
        frame: DataFrame con los juegos (ya construido)
        features_list: list of dicts de get_game_extended_fatigue()

    Returns:
        DataFrame con 9 columnas nuevas de fatiga extendida
    """
    if not features_list:
        return frame

    ext_df = pd.DataFrame(features_list, index=frame.index)
    return pd.concat([frame, ext_df], axis=1)


# ── Phase 5.5: Interaction features (B2B × Travel combos) ──────────


def add_fatigue_combo_to_frame(frame):
    """Agrega interaction features de fatiga × viaje al DataFrame.

    Requiere que las columnas base ya existan en el frame:
        TWO_IN_THREE_*, TRAVEL_DIST_*, TZ_CHANGE_*, GAMES_IN_7_*, TRAVEL_7D_*

    Nuevas features (6 columnas):
        B2B_TRAVEL_HOME/AWAY  — B2B con viaje largo (TWO_IN_THREE × TRAVEL_DIST)
        B2B_TZ_HOME/AWAY      — B2B con jet lag (TWO_IN_THREE × |TZ_CHANGE|)
        FATIGUE_INDEX_HOME/AWAY — densidad + distancia (GAMES_IN_7 × TRAVEL_7D / 1000)

    Args:
        frame: DataFrame con features de fatiga y viaje ya agregadas

    Returns:
        DataFrame con 6 columnas nuevas de interacción
    """
    frame["B2B_TRAVEL_HOME"] = frame["TWO_IN_THREE_HOME"] * frame["TRAVEL_DIST_HOME"]
    frame["B2B_TRAVEL_AWAY"] = frame["TWO_IN_THREE_AWAY"] * frame["TRAVEL_DIST_AWAY"]

    frame["B2B_TZ_HOME"] = frame["TWO_IN_THREE_HOME"] * frame["TZ_CHANGE_HOME"].abs()
    frame["B2B_TZ_AWAY"] = frame["TWO_IN_THREE_AWAY"] * frame["TZ_CHANGE_AWAY"].abs()

    frame["FATIGUE_INDEX_HOME"] = frame["GAMES_IN_7_HOME"] * frame["TRAVEL_7D_HOME"] / 1000
    frame["FATIGUE_INDEX_AWAY"] = frame["GAMES_IN_7_AWAY"] * frame["TRAVEL_7D_AWAY"] / 1000

    return frame
