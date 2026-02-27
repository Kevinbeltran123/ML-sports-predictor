"""
Sistema de ratings Elo para equipos NBA.

Elo es un sistema de clasificacion dinamico inventado para ajedrez (Arpad Elo, 1960)
y adaptado al deporte. La idea central:

    Cada equipo tiene un rating numerico (empieza en 1500).
    Despues de cada partido, el ganador "roba" puntos al perdedor.
    La cantidad de puntos transferidos depende de la SORPRESA del resultado:
        - Victoria esperada = pocos puntos
        - Upset (sorpresa) = muchos puntos

Esto genera un ranking dinamico que se ajusta automaticamente por:
    - Calidad del rival (ganarle a un rival fuerte vale mas)
    - Forma reciente (rachas ganadoras suben el Elo)
    - Ventaja de local (ajustada en la prediccion)

Features que genera (4 por partido):
    - ELO_HOME: Rating Elo del equipo local ANTES del partido
    - ELO_AWAY: Rating Elo del visitante ANTES del partido
    - ELO_DIFF: Diferencia de Elo (incluye ventaja de local)
    - ELO_PROB: Probabilidad de victoria del local segun Elo

Usado por Create_Games.py (entrenamiento) y main.py (prediccion).
"""

import math
import sqlite3

import pandas as pd

# --- Constantes del sistema Elo ---

# Rating inicial para equipos nuevos
INITIAL_ELO = 1500

# K-factor: controla la velocidad de actualizacion
# K=20 es estandar para NBA (reactivo pero estable)
K_FACTOR = 20

# Ventaja de local en puntos Elo
# +100 Elo ≈ 60% probabilidad de ganar (cercano al ~58% real de la NBA)
HOME_ADVANTAGE = 100

# Factor de regresion a la media entre temporadas
# 0.75 = mantener 75% del Elo anterior, 25% vuelve al promedio
# Razon: cambios de roster en offseason hacen que el Elo anterior
# no sea 100% representativo del equipo nuevo
SEASON_CARRY = 0.75

# Mapeo de nombres historicos (igual que en RollingAverages.py)
_TEAM_ALIASES = {
    "Los Angeles Clippers": "LA Clippers",
    "Charlotte Bobcats": "Charlotte Hornets",
    "New Orleans Hornets": "New Orleans Pelicans",
    "New Orleans/Oklahoma City Hornets": "New Orleans Pelicans",
}


def _expected_score(elo_a, elo_b):
    """Calcula la probabilidad de victoria de A contra B.

    Formula: E_A = 1 / (1 + 10^((R_B - R_A) / 400))

    El 400 es una constante de escala. Con 400 puntos de diferencia,
    el equipo fuerte tiene ~91% de probabilidad de ganar.

    Ejemplo NBA:
        Warriors (1700) vs Wizards (1300) → diferencia = 400
        E = 1 / (1 + 10^(-400/400)) = 1 / (1 + 0.1) = 0.909 → 91%
    """
    return 1.0 / (1.0 + math.pow(10, (elo_b - elo_a) / 400.0))


def _mov_multiplier(win_margin, elo_diff):
    """Multiplicador por margen de victoria (Margin of Victory).

    Adaptado del modelo de FiveThirtyEight. La idea:
    - Una paliza (ej: +25 pts) deberia actualizar mas que una victoria
      por 1 punto — refleja una diferencia de calidad mas clara.
    - PERO: si un equipo fuerte (elo_diff grande) aplasta a uno debil,
      no debe recibir tanto credito (era esperado).

    Formula: ln(|MOV| + 1) * 2.2 / (elo_diff * 0.001 + 2.2)

    ln(|MOV| + 1) → escala logaritmica: ganar por 20 no es el doble de
                     informativo que ganar por 10
    2.2 / (...)   → autocorreccion: reduce el bonus si el favorito gana
    """
    abs_margin = abs(win_margin)
    # Escala logaritmica del margen
    log_part = math.log(abs_margin + 1)
    # Autocorreccion por diferencia de Elo
    # elo_diff positivo = el ganador era favorito → reduce multiplicador
    autocorrect = 2.2 / (elo_diff * 0.001 + 2.2)
    return log_part * autocorrect


def _update_elo(elo_winner, elo_loser, win_margin, winner_is_home):
    """Actualiza los ratings Elo despues de un partido.

    Pasos:
    1. Calcular probabilidad esperada (con ventaja de local)
    2. Calcular multiplicador por margen de victoria
    3. Aplicar actualizacion: Elo_new = Elo_old + K * MOV * (Resultado - Esperado)

    Args:
        elo_winner: Elo actual del equipo ganador
        elo_loser: Elo actual del equipo perdedor
        win_margin: Margen de victoria (siempre positivo)
        winner_is_home: True si el ganador jugaba de local

    Returns:
        (nuevo_elo_winner, nuevo_elo_loser)
    """
    # Ajustar por ventaja de local para calcular probabilidad
    if winner_is_home:
        expected_win = _expected_score(elo_winner + HOME_ADVANTAGE, elo_loser)
        elo_diff = (elo_winner + HOME_ADVANTAGE) - elo_loser
    else:
        expected_win = _expected_score(elo_winner, elo_loser + HOME_ADVANTAGE)
        elo_diff = elo_winner - (elo_loser + HOME_ADVANTAGE)

    # Multiplicador por margen de victoria
    mov_mult = _mov_multiplier(win_margin, elo_diff)

    # Actualizacion: K * MOV * (1 - E) para ganador, K * MOV * (0 - E) para perdedor
    shift = K_FACTOR * mov_mult * (1.0 - expected_win)

    return elo_winner + shift, elo_loser - shift


def _apply_season_reset(elo_ratings):
    """Aplica regresion a la media al inicio de nueva temporada.

    En el offseason, los equipos cambian: trades, draft, agentes libres.
    El Elo de la temporada pasada sigue siendo informativo (la base del
    equipo se mantiene), pero no es perfecto.

    Solucion: mover cada rating un 25% hacia la media (1500).
        Elo_nuevo = Elo_viejo * 0.75 + 1500 * 0.25

    Ejemplo:
        Warriors (1700) → 1700 * 0.75 + 1500 * 0.25 = 1275 + 375 = 1650
        Wizards  (1300) → 1300 * 0.75 + 1500 * 0.25 = 975 + 375 = 1350

    Los buenos siguen arriba, los malos abajo, pero todos se acercan un poco.
    """
    for team in elo_ratings:
        elo_ratings[team] = (
            elo_ratings[team] * SEASON_CARRY + INITIAL_ELO * (1 - SEASON_CARRY)
        )


def _normalize_team(name):
    """Normaliza nombre de equipo usando aliases conocidos."""
    return _TEAM_ALIASES.get(name, name)


def _get_season_tables(odds_con):
    """Obtiene todas las tablas de odds ordenadas cronologicamente por temporada.

    Las tablas en OddsData.sqlite siguen el patron:
        odds_YYYY-YY o odds_YYYY-YY_new (o simplemente YYYY-YY)

    Returns:
        list of (season_key, table_name) ordenada por temporada
    """
    cursor = odds_con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )
    tables = [row[0] for row in cursor.fetchall()]

    # Extraer season key y elegir la mejor tabla por temporada
    # Preferir tablas _new sobre las originales (datos mas limpios)
    season_tables = {}
    for table in tables:
        # Extraer el season key (ej: "2007-08" de "odds_2007-08_new")
        clean = table.replace("odds_", "").replace("_new", "")
        # Verificar que parece un season key (YYYY-YY)
        parts = clean.split("-")
        if len(parts) == 2 and len(parts[0]) == 4 and parts[0].isdigit():
            season_key = clean
            # Preferir _new si existe
            if season_key not in season_tables or "_new" in table:
                season_tables[season_key] = table

    # Ordenar por ano de inicio
    sorted_seasons = sorted(
        season_tables.items(), key=lambda x: x[0]
    )
    return sorted_seasons


def build_elo_history(odds_con):
    """Construye el historial completo de Elo desde todas las temporadas.

    Procesa TODOS los partidos en OddsData.sqlite cronologicamente,
    empezando desde 2007-08. Esto da ~5 temporadas de "calentamiento"
    antes de que el dataset real empiece en 2012-13.

    CLAVE para evitar data leakage:
        Para cada partido, registramos el Elo ANTES de jugar.
        Luego actualizamos. Asi el modelo solo ve informacion pasada.

    Args:
        odds_con: conexion SQLite a OddsData.sqlite

    Returns:
        dict[(date_str, home_team, away_team)] -> {
            'ELO_HOME': float,
            'ELO_AWAY': float,
            'ELO_DIFF': float,  # home_elo + HOME_ADV - away_elo
            'ELO_PROB': float,  # prob de victoria local
        }
    """
    elo_ratings = {}  # team_name -> current_elo
    elo_lookup = {}   # (date, home, away) -> dict de features

    season_tables = _get_season_tables(odds_con)

    prev_season = None
    for season_key, table_name in season_tables:
        # Aplicar regresion a la media al cambiar de temporada
        if prev_season is not None and elo_ratings:
            _apply_season_reset(elo_ratings)
        prev_season = season_key

        # Leer todos los partidos de esta temporada
        try:
            df = pd.read_sql_query(
                f'SELECT Date, Home, Away, Win_Margin FROM "{table_name}"',
                odds_con,
            )
        except Exception:
            continue

        if df.empty:
            continue

        # Ordenar por fecha para procesar cronologicamente
        df = df.sort_values("Date").reset_index(drop=True)

        for row in df.itertuples(index=False):
            home = _normalize_team(row.Home)
            away = _normalize_team(row.Away)

            # Inicializar equipos nuevos con Elo base
            if home not in elo_ratings:
                elo_ratings[home] = INITIAL_ELO
            if away not in elo_ratings:
                elo_ratings[away] = INITIAL_ELO

            home_elo = elo_ratings[home]
            away_elo = elo_ratings[away]

            # --- Registrar Elo ANTES del partido (para el modelo) ---
            elo_diff = (home_elo + HOME_ADVANTAGE) - away_elo
            elo_prob = _expected_score(home_elo + HOME_ADVANTAGE, away_elo)

            # Guardar con ambas versiones del nombre (original y normalizado)
            for h_name in [row.Home, home]:
                for a_name in [row.Away, away]:
                    elo_lookup[(str(row.Date), h_name, a_name)] = {
                        "ELO_HOME": home_elo,
                        "ELO_AWAY": away_elo,
                        "ELO_DIFF": elo_diff,
                        "ELO_PROB": elo_prob,
                    }

            # --- Actualizar Elo DESPUES del partido ---
            win_margin = float(row.Win_Margin)
            if win_margin > 0:
                # Home gano
                new_home, new_away = _update_elo(
                    home_elo, away_elo, win_margin, winner_is_home=True
                )
            elif win_margin < 0:
                # Away gano
                new_away, new_home = _update_elo(
                    away_elo, home_elo, abs(win_margin), winner_is_home=False
                )
            else:
                # Empate (raro en NBA, pero por si acaso: no cambiar)
                new_home, new_away = home_elo, away_elo

            elo_ratings[home] = new_home
            elo_ratings[away] = new_away

    return elo_lookup, elo_ratings


def get_current_elos(odds_db_path):
    """Obtiene los ratings Elo actuales de todos los equipos.

    Procesa todo el historial y devuelve el estado final.
    Usado por main.py para predicciones del dia.

    Args:
        odds_db_path: ruta al archivo OddsData.sqlite

    Returns:
        dict[team_name] -> float (Elo actual)
    """
    with sqlite3.connect(odds_db_path) as con:
        _, elo_ratings = build_elo_history(con)
    return elo_ratings


def get_game_elo_features(elo_ratings, home_team, away_team):
    """Calcula features Elo para un partido especifico (prediccion en vivo).

    Usado por main.py para partidos de hoy.

    Args:
        elo_ratings: dict[team] -> Elo actual
        home_team: nombre del equipo local
        away_team: nombre del equipo visitante

    Returns:
        dict con ELO_HOME, ELO_AWAY, ELO_DIFF, ELO_PROB
    """
    home = _normalize_team(home_team)
    away = _normalize_team(away_team)

    home_elo = elo_ratings.get(home, INITIAL_ELO)
    away_elo = elo_ratings.get(away, INITIAL_ELO)

    elo_diff = (home_elo + HOME_ADVANTAGE) - away_elo
    elo_prob = _expected_score(home_elo + HOME_ADVANTAGE, away_elo)

    return {
        "ELO_HOME": home_elo,
        "ELO_AWAY": away_elo,
        "ELO_DIFF": elo_diff,
        "ELO_PROB": elo_prob,
    }


def add_elo_features_to_frame(frame, elo_features_list):
    """Agrega columnas de Elo al DataFrame de juegos.

    Args:
        frame: DataFrame con los juegos (ya construido)
        elo_features_list: list of dicts (uno por juego)

    Returns:
        DataFrame con 4 nuevas columnas de Elo
    """
    if not elo_features_list:
        return frame

    elo_df = pd.DataFrame(elo_features_list, index=frame.index)
    result = pd.concat([frame, elo_df], axis=1)
    return result


# ===================================================================
# Simple Rating System (SRS)
# ===================================================================
#
# SRS es un sistema de rating ajustado por oponente inventado por
# Doug Lillibridge y popularizado por Basketball-Reference.
#
# La diferencia clave con Elo:
#   - Elo actualiza INCREMENTALMENTE despues de cada partido (como un stream)
#   - SRS resuelve un SISTEMA DE ECUACIONES con todos los datos de la temporada
#
# La ecuacion para cada equipo:
#   SRS_i = MOV_i + SOS_i
#
# Donde:
#   MOV_i = margen de victoria promedio del equipo i
#   SOS_i = promedio de SRS de los oponentes del equipo i (Strength of Schedule)
#
# Esto crea dependencias circulares (el SRS de A depende del SRS de B
# que depende del SRS de A...) que se resuelven iterativamente.
#
# Propiedad clave: la media de todos los SRS es siempre 0.
#   - SRS = +5 significa "5 puntos mejor que el equipo promedio"
#   - SRS = -3 significa "3 puntos peor que el equipo promedio"
#
# Ejemplo NBA:
#   Si los Celtics tienen MOV=+8 pero jugaron contra rivales debiles (SOS=-2),
#   su SRS = +8 + (-2) = +6. Es decir, son buenos pero no TAN buenos como
#   su margen bruto sugiere.
# ===================================================================

# Minimo de partidos para calcular SRS confiable.
# Con menos de 5 partidos, los margenes son muy ruidosos
# y el SRS no tiene suficiente informacion para ajustar.
_SRS_MIN_GAMES = 5


def _compute_srs_ratings(team_margins, team_opponents, n_iter=50):
    """Calcula SRS para todos los equipos con datos actuales.

    Sistema iterativo:
        1. Inicializar SRS_i = MOV_i (margen promedio de victoria)
        2. SOS_i = promedio de SRS de oponentes de i
        3. SRS_i = MOV_i + SOS_i
        4. Repetir hasta convergencia
        5. Centrar en 0 (la media de todos los SRS = 0 por definicion)

    Por que converge?
        Porque es una contraccion: en cada iteracion, SOS promedia
        multiples valores, lo que reduce la varianza. Matematicamente,
        la matriz de adyacencia del calendario tiene radio espectral < 1
        (cada equipo juega contra muchos rivales), asi que el punto fijo
        existe y es unico.

    Args:
        team_margins: dict[team] -> list[float] (todos los margenes, + para victorias)
        team_opponents: dict[team] -> list[str] (todos los oponentes enfrentados)
        n_iter: iteraciones maximas (50 es mas que suficiente; tipicamente converge en 10-20)

    Returns:
        dict[team] -> float (SRS rating, centrado en 0)
    """
    # Equipos con suficientes partidos para calcular SRS
    teams = [
        t for t in team_margins
        if len(team_margins[t]) >= _SRS_MIN_GAMES
    ]

    if not teams:
        return {}

    # Paso 1: Calcular MOV (margen promedio) para cada equipo
    mov = {}
    for team in teams:
        margins = team_margins[team]
        mov[team] = sum(margins) / len(margins)

    # Paso 2: Inicializar SRS = MOV (antes de ajustar por oponentes)
    srs = {team: mov[team] for team in teams}

    # Paso 3: Iterar hasta convergencia
    for iteration in range(n_iter):
        max_change = 0.0
        new_srs = {}

        for team in teams:
            # Calcular SOS = promedio de SRS de los oponentes
            opponents = team_opponents[team]
            # Solo contar oponentes que tienen SRS calculado
            opp_ratings = [
                srs[opp] for opp in opponents
                if opp in srs
            ]

            if opp_ratings:
                sos = sum(opp_ratings) / len(opp_ratings)
            else:
                sos = 0.0

            # SRS = MOV + SOS
            new_srs[team] = mov[team] + sos

            # Rastrear convergencia
            change = abs(new_srs[team] - srs[team])
            if change > max_change:
                max_change = change

        srs = new_srs

        # Convergencia: cambio maximo < 0.01 puntos
        if max_change < 0.01:
            break

    # Paso 4: Centrar en 0 (propiedad matematica del SRS)
    # La media deberia ser ~0 naturalmente, pero forzamos por precision numerica
    if srs:
        mean_srs = sum(srs.values()) / len(srs)
        srs = {team: rating - mean_srs for team, rating in srs.items()}

    return srs


def build_srs_history(odds_con):
    """Construye historial de SRS (Simple Rating System) para todos los partidos.

    SRS = MOV + SOS, donde SOS = promedio de SRS de los oponentes.
    Es un sistema de ecuaciones simultaneas que se resuelve iterativamente.

    Ventaja sobre Elo:
        - Ajusta por calidad del rival Y por margen de victoria simultaneamente
        - No tiene hiperparametros (K-factor, home advantage)
        - Converge matematicamente (garantizado)

    Ventaja sobre SOS_W_PCT_10 (actual):
        - Usa margenes en vez de W/L binario (mas informacion)
        - Ajuste transitivo: la calidad de los rivales de tus rivales importa

    Para cada partido, calcula SRS con datos ANTERIORES al partido (sin leakage).
    Se recalcula al inicio de cada temporada.

    Args:
        odds_con: conexion SQLite a OddsData.sqlite

    Returns:
        dict[(date_str, home, away)] -> {
            "SRS_HOME": float,  # SRS del equipo local
            "SRS_AWAY": float,  # SRS del visitante
            "SRS_DIFF": float,  # SRS_HOME - SRS_AWAY
        }
    """
    srs_lookup = {}  # (date, home, away) -> dict de features

    season_tables = _get_season_tables(odds_con)

    for _season_key, table_name in season_tables:
        # --- Reiniciar registros al inicio de cada temporada ---
        # SRS solo usa datos de la temporada actual (no hay carry-over)
        # porque los rosters cambian demasiado entre temporadas.
        team_margins = {}    # team -> [margin1, margin2, ...]
        team_opponents = {}  # team -> [opp1, opp2, ...]

        # Leer todos los partidos de esta temporada
        try:
            df = pd.read_sql_query(
                f'SELECT Date, Home, Away, Win_Margin FROM "{table_name}"',
                odds_con,
            )
        except Exception:
            continue

        if df.empty:
            continue

        # Ordenar por fecha para procesar cronologicamente
        df = df.sort_values("Date").reset_index(drop=True)

        for row in df.itertuples(index=False):
            home = _normalize_team(row.Home)
            away = _normalize_team(row.Away)

            # --- Calcular SRS ANTES del partido (para el modelo) ---
            # Resolver el sistema con TODOS los datos acumulados hasta ahora
            srs_ratings = _compute_srs_ratings(team_margins, team_opponents)

            # Obtener SRS pre-partido (0.0 si el equipo no tiene suficientes partidos)
            home_srs = srs_ratings.get(home, 0.0)
            away_srs = srs_ratings.get(away, 0.0)

            # Guardar con ambas versiones del nombre (original y normalizado)
            srs_features = {
                "SRS_HOME": round(home_srs, 3),
                "SRS_AWAY": round(away_srs, 3),
                "SRS_DIFF": round(home_srs - away_srs, 3),
            }

            for h_name in [row.Home, home]:
                for a_name in [row.Away, away]:
                    srs_lookup[(str(row.Date), h_name, a_name)] = srs_features

            # --- Actualizar registros DESPUES del partido ---
            win_margin = float(row.Win_Margin)

            # Registrar margen desde la perspectiva de cada equipo
            # Home: margen positivo = victoria, negativo = derrota
            # Away: margen invertido
            if home not in team_margins:
                team_margins[home] = []
                team_opponents[home] = []
            if away not in team_margins:
                team_margins[away] = []
                team_opponents[away] = []

            team_margins[home].append(win_margin)
            team_margins[away].append(-win_margin)
            team_opponents[home].append(away)
            team_opponents[away].append(home)

    return srs_lookup


def get_game_srs_features(srs_lookup, game_date, home_team, away_team):
    """Obtiene features de SRS para un partido.

    Busca en el lookup pre-calculado. Si no encuentra el partido
    (ej: equipo nuevo o inicio de temporada), devuelve SRS = 0 (neutral).

    Mismo patron que get_game_elo_features pero usando el lookup historico.

    Args:
        srs_lookup: dict generado por build_srs_history()
        game_date: fecha del partido (str, formato YYYY-MM-DD o similar)
        home_team: nombre del equipo local
        away_team: nombre del equipo visitante

    Returns:
        dict con SRS_HOME, SRS_AWAY, SRS_DIFF
    """
    home = _normalize_team(home_team)
    away = _normalize_team(away_team)

    # Intentar buscar con nombres normalizados primero, luego originales
    key = (str(game_date), home, away)
    features = srs_lookup.get(key)

    if features is None:
        # Intentar con nombres originales
        key_orig = (str(game_date), home_team, away_team)
        features = srs_lookup.get(key_orig)

    if features is None:
        # Partido no encontrado: devolver valores neutrales
        return {
            "SRS_HOME": 0.0,
            "SRS_AWAY": 0.0,
            "SRS_DIFF": 0.0,
        }

    return features


def add_srs_features_to_frame(frame, srs_features_list):
    """Agrega columnas de SRS al DataFrame de juegos.

    Mismo patron que add_elo_features_to_frame.

    Args:
        frame: DataFrame con los juegos (ya construido)
        srs_features_list: list of dicts (uno por juego, generados por get_game_srs_features)

    Returns:
        DataFrame con 3 nuevas columnas: SRS_HOME, SRS_AWAY, SRS_DIFF
    """
    if not srs_features_list:
        return frame

    srs_df = pd.DataFrame(srs_features_list, index=frame.index)
    result = pd.concat([frame, srs_df], axis=1)
    return result
