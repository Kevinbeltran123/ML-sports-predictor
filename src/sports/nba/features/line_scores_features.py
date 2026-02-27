"""Features de Quarter Scoring Patterns extraidas de BRefData.sqlite (line_scores).

Los patrones de anotacion por cuarto revelan tendencias importantes:
    - Equipos que cierran fuerte (Q4) vs equipos que arrancan rapido (Q1)
    - Capacidad de remontada (2H > 1H)
    - Consistencia ofensiva (baja varianza entre cuartos)

Features generadas (8 por juego, 4 por equipo):
    LS_Q4_PCT_HOME, LS_Q4_PCT_AWAY         — % de puntos en Q4 (clutch scoring)
    LS_2H_RATIO_HOME, LS_2H_RATIO_AWAY     — ratio (Q3+Q4)/(Q1+Q2) (cierre de juegos)
    LS_Q1_PCT_HOME, LS_Q1_PCT_AWAY         — % de puntos en Q1 (arranque rapido)
    LS_SCORING_VAR_HOME, LS_SCORING_VAR_AWAY — varianza normalizada de [Q1-Q4] (consistencia)

Patron: build -> get -> add (mismo que bref_game_features.py).

Calculo:
    - Rolling 10 juegos con shift(1) para T-1 estricto (sin leakage)
    - min_periods=3: necesitamos al menos 3 juegos previos
    - Datos de BRefData.sqlite, tablas line_scores_YYYY-ZZ (2014-15 a 2025-26)
    - team_name esta vacio en line_scores; se obtiene el equipo
      cruzando game_id + team_side contra la tabla schedules
"""

import sqlite3

import numpy as np
import pandas as pd

from src.config import BREF_DB, get_logger
from src.sports.nba.features.team_mappings import TEAM_NAME_TO_BREF

logger = get_logger(__name__)

# Temporadas disponibles en BRefData.sqlite
SEASONS = [
    "2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24",
    "2024-25", "2025-26",
]

# Ventana de rolling para promedios
_ROLLING_WINDOW = 10

# Minimo de juegos previos para generar features
_MIN_PERIODS = 3

# Columnas intermedias de stats calculadas por juego
_STATS_COLS = ["q4_pct", "q1_pct", "half_ratio", "scoring_var"]

# Mapeo de columna intermedia -> nombre de feature final
_COL_TO_FEATURE = {
    "q4_pct": "LS_Q4_PCT",
    "half_ratio": "LS_2H_RATIO",
    "q1_pct": "LS_Q1_PCT",
    "scoring_var": "LS_SCORING_VAR",
}

# Valores default cuando no hay datos suficientes
# Q4_PCT y Q1_PCT: ~0.25 (cada cuarto aporta ~25% en promedio)
# 2H_RATIO: 1.0 (mitades equilibradas)
# SCORING_VAR: 0.01 (varianza tipica normalizada)
_DEFAULTS = {
    "LS_Q4_PCT": 0.25,
    "LS_2H_RATIO": 1.0,
    "LS_Q1_PCT": 0.25,
    "LS_SCORING_VAR": 0.01,
}


def build_line_scores_history(bref_db_path=None):
    """Pre-construye promedios rolling de 10 juegos de patrones de anotacion por cuarto.

    Estrategia:
        1. Cargar tabla schedules para mapear game_id -> equipos (BRef codes)
        2. Para cada temporada, cargar line_scores y asignar identidad de equipo
           cruzando game_id + team_side contra schedules
        3. Calcular per-game: q4_pct, half_ratio, q1_pct, scoring_var
        4. Rolling 10 juegos con shift(1) para T-1 estricto

    Args:
        bref_db_path: ruta al archivo BRefData.sqlite.
            Si es None, usa la ruta de src.config.BREF_DB.

    Returns:
        dict[(date_str, bref_code)] -> {
            "LS_Q4_PCT": float, "LS_2H_RATIO": float,
            "LS_Q1_PCT": float, "LS_SCORING_VAR": float,
        }
    """
    if bref_db_path is None:
        bref_db_path = BREF_DB

    lookup = {}

    with sqlite3.connect(bref_db_path) as con:
        # --- Paso 1: Construir mapeo game_id -> {home: bref_code, away: bref_code} ---
        # team_name esta vacio en line_scores, asi que necesitamos schedules
        game_teams = _build_game_teams_mapping(con)

        if not game_teams:
            logger.warning(
                "No se pudo construir mapeo game_id->equipos desde schedules"
            )
            return lookup

        logger.info(
            "Mapeo schedules: %d juegos con equipos identificados", len(game_teams)
        )

        # Verificar que tablas de line_scores existen en la BD
        cursor = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name LIKE 'line_scores_%'"
        )
        existing_tables = {row[0] for row in cursor.fetchall()}

        for season in SEASONS:
            table_name = f"line_scores_{season}"
            if table_name not in existing_tables:
                logger.debug("Tabla %s no existe, saltando", table_name)
                continue

            try:
                df = pd.read_sql_query(
                    f'SELECT game_id, game_date, team_side, '
                    f'q1, q2, q3, q4, final '
                    f'FROM "{table_name}"',
                    con,
                )
            except Exception as e:
                logger.warning("Error leyendo %s: %s", table_name, e)
                continue

            if df.empty:
                logger.debug("Tabla %s vacia", table_name)
                continue

            # Asignar identidad de equipo via game_id + team_side
            df["bref_code"] = df.apply(
                lambda r: game_teams.get(r["game_id"], {}).get(r["team_side"]),
                axis=1,
            )

            # Eliminar filas sin equipo identificado
            before = len(df)
            df = df.dropna(subset=["bref_code"])
            dropped = before - len(df)
            if dropped > 0:
                logger.debug(
                    "%s: %d filas sin equipo identificado (descartadas)",
                    table_name, dropped,
                )

            if df.empty:
                continue

            # Convertir columnas de cuartos y final a numerico
            for col in ["q1", "q2", "q3", "q4", "final"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Parsear fechas y eliminar filas invalidas
            df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
            df = df.dropna(subset=["game_date"])

            # Eliminar juegos con final = 0 o NaN (datos corruptos)
            df = df[df["final"].fillna(0) > 0]

            if df.empty:
                continue

            # --- Paso 3: Calcular features per-game ---
            _compute_per_game_features(df)

            # --- Paso 4: Rolling por equipo con T-1 estricto ---
            n_entries = _process_season_rolling(df, lookup)
            logger.info(
                "Line Scores %s: %d entradas generadas", season, n_entries
            )

    logger.info(
        "Line Scores total: %d entradas (fecha, equipo)", len(lookup)
    )
    return lookup


def _build_game_teams_mapping(con):
    """Construye mapeo game_id -> {'home': bref_code, 'away': bref_code} desde schedules.

    La tabla schedules tiene home_team/away_team como nombres completos
    (ej. "Boston Celtics"). Los convertimos a codigos BRef (ej. "BOS")
    usando TEAM_NAME_TO_BREF.

    Args:
        con: conexion sqlite3 a BRefData.sqlite

    Returns:
        dict[str, dict[str, str]]: game_id -> {'home': bref_code, 'away': bref_code}
    """
    try:
        schedules = pd.read_sql_query(
            "SELECT game_id, home_team, away_team FROM schedules", con
        )
    except Exception as e:
        logger.warning("Error leyendo tabla schedules: %s", e)
        return {}

    game_teams = {}
    for _, row in schedules.iterrows():
        h_code = TEAM_NAME_TO_BREF.get(row["home_team"])
        a_code = TEAM_NAME_TO_BREF.get(row["away_team"])
        if h_code and a_code:
            game_teams[row["game_id"]] = {"home": h_code, "away": a_code}

    return game_teams


def _compute_per_game_features(df):
    """Calcula features per-game directamente sobre el DataFrame (in-place).

    Para cada fila (un equipo en un juego):
        - q4_pct: porcentaje de puntos anotados en Q4 (Q4 / final)
        - q1_pct: porcentaje de puntos anotados en Q1 (Q1 / final)
        - half_ratio: ratio segunda mitad / primera mitad ((Q3+Q4)/(Q1+Q2))
        - scoring_var: varianza de [Q1,Q2,Q3,Q4] normalizada por final²

    La normalizacion por final² hace que la varianza sea comparable
    entre juegos de distinto puntaje total.

    Args:
        df: DataFrame con columnas q1, q2, q3, q4, final (modificado in-place)
    """
    # clip(lower=1) para evitar division por cero en juegos con datos raros
    final_safe = df["final"].clip(lower=1)

    # Porcentaje de puntos en Q4 — mide capacidad clutch
    df["q4_pct"] = df["q4"] / final_safe

    # Porcentaje de puntos en Q1 — mide arranque rapido
    df["q1_pct"] = df["q1"] / final_safe

    # Ratio segunda mitad / primera mitad
    # >1 significa que anotan mas en la segunda mitad (closers)
    # <1 significa que arrancan fuerte pero bajan
    first_half = df["q1"] + df["q2"]
    second_half = df["q3"] + df["q4"]
    df["half_ratio"] = second_half / first_half.clip(lower=1)

    # Varianza de anotacion por cuarto, normalizada por final²
    # Mide consistencia: varianza baja = equipo estable en los 4 cuartos
    q_values = df[["q1", "q2", "q3", "q4"]].values
    df["scoring_var"] = np.var(q_values, axis=1) / (final_safe ** 2)


def _process_season_rolling(df, lookup):
    """Calcula rolling por equipo para una temporada y llena el lookup.

    Para cada equipo:
    1. Ordena por fecha
    2. shift(1) para excluir el juego actual (T-1)
    3. rolling(10, min_periods=3) para promediar los ultimos 10 juegos previos
    4. Almacena cada fila con rolling valido en el lookup

    Args:
        df: DataFrame con game_date, bref_code y columnas de _STATS_COLS
        lookup: dict a llenar con (date_str, bref_code) -> features

    Returns:
        int: numero de entradas generadas
    """
    entries_added = 0

    for bref_code, team_df in df.groupby("bref_code"):
        # Ordenar cronologicamente
        team_df = team_df.sort_values("game_date").copy()

        # Calcular rolling con T-1 estricto:
        # shift(1) desplaza una fila -> el valor en fila i corresponde
        # al dato del juego i-1 (juego anterior).
        # rolling(10) promedia las ultimas 10 filas desplazadas,
        # es decir, los ultimos 10 juegos ANTES del actual.
        feature_names = list(_COL_TO_FEATURE.values())
        for col in _STATS_COLS:
            feature_name = _COL_TO_FEATURE[col]
            team_df[feature_name] = (
                team_df[col]
                .shift(1)
                .rolling(_ROLLING_WINDOW, min_periods=_MIN_PERIODS)
                .mean()
            )

        # Almacenar filas con rolling valido en el lookup
        valid_mask = team_df[feature_names].notna().all(axis=1)

        for _, row in team_df[valid_mask].iterrows():
            date_str = row["game_date"].strftime("%Y-%m-%d")
            lookup[(date_str, bref_code)] = {
                feat: round(float(row[feat]), 4) for feat in feature_names
            }
            entries_added += 1

    return entries_added


def get_game_line_scores(ls_lookup, date_str, home_team, away_team):
    """Obtiene features de Line Scores para un partido especifico.

    Recibe nombres completos de equipos (como vienen de OddsData/create_games)
    y los traduce a codigos BRef internamente.

    Args:
        ls_lookup: dict de build_line_scores_history()
        date_str: fecha del juego como 'YYYY-MM-DD'
        home_team: nombre completo del equipo local (ej. "Boston Celtics")
        away_team: nombre completo del equipo visitante (ej. "New York Knicks")

    Returns:
        dict con 8 features: 4 por equipo con sufijo HOME/AWAY.
        Si no hay datos, retorna valores default (promedios neutrales).
    """
    home_bref = TEAM_NAME_TO_BREF.get(home_team)
    away_bref = TEAM_NAME_TO_BREF.get(away_team)

    if home_bref is None:
        logger.debug("Equipo local no encontrado en mapeo BRef: '%s'", home_team)
    if away_bref is None:
        logger.debug("Equipo visitante no encontrado en mapeo BRef: '%s'", away_team)

    # Obtener features del lookup o defaults
    home_feats = (
        ls_lookup.get((date_str, home_bref), _DEFAULTS)
        if home_bref
        else _DEFAULTS
    )
    away_feats = (
        ls_lookup.get((date_str, away_bref), _DEFAULTS)
        if away_bref
        else _DEFAULTS
    )

    return {
        "LS_Q4_PCT_HOME": home_feats["LS_Q4_PCT"],
        "LS_Q4_PCT_AWAY": away_feats["LS_Q4_PCT"],
        "LS_2H_RATIO_HOME": home_feats["LS_2H_RATIO"],
        "LS_2H_RATIO_AWAY": away_feats["LS_2H_RATIO"],
        "LS_Q1_PCT_HOME": home_feats["LS_Q1_PCT"],
        "LS_Q1_PCT_AWAY": away_feats["LS_Q1_PCT"],
        "LS_SCORING_VAR_HOME": home_feats["LS_SCORING_VAR"],
        "LS_SCORING_VAR_AWAY": away_feats["LS_SCORING_VAR"],
    }


def add_line_scores_to_frame(frame, ls_features_list):
    """Agrega columnas de Line Scores al DataFrame de juegos.

    Convierte la lista de dicts (uno por juego) en un DataFrame
    y lo concatena horizontalmente al frame existente.

    Args:
        frame: DataFrame de juegos (de create_games.py)
        ls_features_list: lista de dicts de get_game_line_scores(),
            uno por cada fila del frame

    Returns:
        DataFrame con las 8 columnas de Line Scores agregadas
    """
    if not ls_features_list:
        return frame
    ls_df = pd.DataFrame(ls_features_list, index=frame.index)
    return pd.concat([frame, ls_df], axis=1)
