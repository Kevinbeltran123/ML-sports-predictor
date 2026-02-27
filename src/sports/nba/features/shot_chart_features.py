"""Features de Shot Chart extraidas de BRefData.sqlite.

Analiza ~2.6M tiros con coordenadas x/y para extraer el perfil de
distribucion de tiros de cada equipo. La distribucion de tiros revela
la estrategia ofensiva: equipos que atacan el aro vs equipos perimetrales.

Zonas de tiro:
    1. Restricted Area (RA)  — distancia <= 4 ft (mates, bandejas)
    2. Paint (non-RA)        — 4 < distancia <= 10 ft (floaters, hooks)
    3. Midrange              — 10 < distancia <= 22 ft AND valor == 2
    4. Corner 3              — valor == 3 AND y_ft <= 7.8
    5. Above Break 3         — valor == 3 AND y_ft > 7.8 (no se usa como feature)

Features generadas (12 por juego, 6 por equipo):
    SC_RA_RATE_HOME/AWAY      — fraccion de tiros en area restringida
    SC_RA_FG_PCT_HOME/AWAY    — FG% en area restringida
    SC_PAINT_RATE_HOME/AWAY   — fraccion de tiros en pintura (no RA)
    SC_MID_RATE_HOME/AWAY     — fraccion de tiros de media distancia
    SC_CORNER3_RATE_HOME/AWAY — fraccion de tiros de esquina 3
    SC_AVG_DIST_HOME/AWAY     — distancia promedio de tiro

Patron: build -> get -> add (mismo que bref_game_features.py).

Calculo:
    - Rolling 10 juegos con shift(1) para T-1 estricto (sin leakage)
    - min_periods=3: necesitamos al menos 3 juegos previos
    - Datos de BRefData.sqlite, tablas shot_chart_YYYY-ZZ (2014-15 a 2025-26)

NOTA: Las tablas son grandes (~234K filas por temporada). Se cargan solo
las columnas necesarias y se procesan por temporada para ser eficientes
en memoria.
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

# Columnas minimas a leer de la tabla (eficiencia de memoria)
_READ_COLS = ["game_date", "team_code", "distance_ft", "point_value", "made", "y_ft"]

# Nombres de features finales (sin sufijo HOME/AWAY)
_FEATURE_NAMES = [
    "SC_RA_RATE",
    "SC_RA_FG_PCT",
    "SC_PAINT_RATE",
    "SC_MID_RATE",
    "SC_CORNER3_RATE",
    "SC_AVG_DIST",
]

# Columnas intermedias de agregacion por juego
# (se calculan por game_date + team_code antes del rolling)
_AGG_COLS = [
    "ra_rate",
    "ra_fg_pct",
    "paint_rate",
    "mid_rate",
    "corner3_rate",
    "avg_dist",
]

# Mapeo de columna agregada -> nombre de feature
_AGG_TO_FEATURE = dict(zip(_AGG_COLS, _FEATURE_NAMES))

# Valores default cuando no hay datos suficientes
# (promedios historicos aproximados de la NBA)
_DEFAULTS = {
    "SC_RA_RATE": 0.30,
    "SC_RA_FG_PCT": 0.62,
    "SC_PAINT_RATE": 0.15,
    "SC_MID_RATE": 0.15,
    "SC_CORNER3_RATE": 0.10,
    "SC_AVG_DIST": 14.0,
}


def build_shot_chart_history(bref_db_path=None):
    """Pre-construye promedios rolling de 10 juegos de shot chart para todos los equipos.

    Lee todas las tablas shot_chart_YYYY-ZZ de BRefData.sqlite,
    clasifica cada tiro por zona, agrega por partido, calcula rolling
    means con T-1 estricto (shift + rolling) y almacena en un dict
    para lookup rapido.

    Estrategia de memoria:
        - Solo carga columnas necesarias (6 de ~15)
        - Procesa una temporada a la vez y descarta el DataFrame
        - Clasifica zonas con operaciones vectorizadas (numpy)

    Args:
        bref_db_path: ruta al archivo BRefData.sqlite.
            Si es None, usa la ruta de src.config.BREF_DB.

    Returns:
        dict[(date_str, bref_code)] -> {
            "SC_RA_RATE": float, "SC_RA_FG_PCT": float,
            "SC_PAINT_RATE": float, "SC_MID_RATE": float,
            "SC_CORNER3_RATE": float, "SC_AVG_DIST": float,
        }
    """
    if bref_db_path is None:
        bref_db_path = BREF_DB

    lookup = {}

    with sqlite3.connect(bref_db_path) as con:
        # Verificar que tablas existen en la BD
        cursor = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name LIKE 'shot_chart_%'"
        )
        existing_tables = {row[0] for row in cursor.fetchall()}

        for season in SEASONS:
            table_name = f"shot_chart_{season}"
            if table_name not in existing_tables:
                logger.debug("Tabla %s no existe, saltando", table_name)
                continue

            try:
                # Solo leemos columnas necesarias para ahorrar memoria
                cols_str = ", ".join(_READ_COLS)
                df = pd.read_sql_query(
                    f'SELECT {cols_str} FROM "{table_name}"',
                    con,
                )
            except Exception as e:
                logger.warning("Error leyendo %s: %s", table_name, e)
                continue

            if df.empty:
                logger.debug("Tabla %s vacia", table_name)
                continue

            # Agregar tiros por partido y clasificar zonas
            game_agg = _aggregate_game_shots(df)

            # Liberar memoria del DataFrame crudo
            del df

            # Calcular rolling por equipo y llenar lookup
            n_entries = _process_season_rolling(game_agg, lookup)
            logger.info(
                "Shot Chart %s: %d entradas generadas", season, n_entries
            )

    logger.info(
        "Shot Chart total: %d entradas (fecha, equipo)", len(lookup)
    )
    return lookup


def _aggregate_game_shots(df):
    """Clasifica tiros por zona y agrega metricas por partido.

    Cada tiro se clasifica en una zona basandose en distancia,
    punto y coordenada y. Luego se agrupan por (game_date, team_code)
    para obtener conteos y tasas por partido.

    Args:
        df: DataFrame crudo de shot_chart con columnas _READ_COLS

    Returns:
        DataFrame con una fila por (game_date, team_code) y columnas
        de tasas/porcentajes por zona.
    """
    # Convertir columnas numericas por seguridad
    df["distance_ft"] = pd.to_numeric(df["distance_ft"], errors="coerce")
    df["point_value"] = pd.to_numeric(df["point_value"], errors="coerce")
    df["made"] = pd.to_numeric(df["made"], errors="coerce")
    df["y_ft"] = pd.to_numeric(df["y_ft"], errors="coerce")

    # Clasificacion vectorizada de zonas usando condiciones numpy
    # Cada tiro pertenece a exactamente una zona
    dist = df["distance_ft"].values
    pv = df["point_value"].values
    y = df["y_ft"].values
    made = df["made"].values

    is_ra = dist <= 4
    is_paint = (dist > 4) & (dist <= 10)
    is_mid = (dist > 10) & (dist <= 22) & (pv == 2)
    is_corner3 = (pv == 3) & (y <= 7.8)
    # above_break_3 = (pv == 3) & (y > 7.8) — no se usa como feature individual

    # Asignar columnas de clasificacion al DataFrame
    df["is_ra"] = is_ra
    df["ra_made"] = is_ra & (made == 1)
    df["is_paint"] = is_paint
    df["is_mid"] = is_mid
    df["is_corner3"] = is_corner3

    # Agrupar por partido (game_date, team_code) y sumar
    grouped = df.groupby(["game_date", "team_code"]).agg(
        total_shots=("is_ra", "size"),  # cuenta total de filas
        ra_shots=("is_ra", "sum"),
        ra_makes=("ra_made", "sum"),
        paint_shots=("is_paint", "sum"),
        mid_shots=("is_mid", "sum"),
        corner3_shots=("is_corner3", "sum"),
        sum_distance=("distance_ft", "sum"),
    ).reset_index()

    # Calcular tasas y porcentajes
    # Evitar division por cero con np.where
    total = grouped["total_shots"].values.astype(float)

    grouped["ra_rate"] = np.where(total > 0, grouped["ra_shots"] / total, 0.0)
    grouped["ra_fg_pct"] = np.where(
        grouped["ra_shots"] > 0,
        grouped["ra_makes"] / grouped["ra_shots"].astype(float),
        0.0,
    )
    grouped["paint_rate"] = np.where(total > 0, grouped["paint_shots"] / total, 0.0)
    grouped["mid_rate"] = np.where(total > 0, grouped["mid_shots"] / total, 0.0)
    grouped["corner3_rate"] = np.where(total > 0, grouped["corner3_shots"] / total, 0.0)
    grouped["avg_dist"] = np.where(total > 0, grouped["sum_distance"] / total, 0.0)

    return grouped[["game_date", "team_code"] + _AGG_COLS]


def _process_season_rolling(game_agg, lookup):
    """Calcula rolling por equipo para una temporada y llena el lookup.

    Para cada equipo:
    1. Ordena por fecha
    2. shift(1) para excluir el juego actual (T-1)
    3. rolling(10, min_periods=3) para promediar los ultimos 10 juegos previos
    4. Almacena cada fila con rolling valido en el lookup

    Args:
        game_agg: DataFrame con game_date, team_code y columnas de tasas
        lookup: dict a llenar con (date_str, bref_code) -> features

    Returns:
        int: numero de entradas generadas
    """
    entries_added = 0

    # Parsear fechas
    game_agg["game_date"] = pd.to_datetime(game_agg["game_date"], errors="coerce")
    game_agg = game_agg.dropna(subset=["game_date"])

    for team_code, team_df in game_agg.groupby("team_code"):
        # Ordenar cronologicamente
        team_df = team_df.sort_values("game_date").copy()

        # Calcular rolling con T-1 estricto:
        # shift(1) desplaza una fila hacia abajo -> el valor en fila i
        # corresponde al dato de la fila i-1 (juego anterior).
        # Luego rolling(10) promedia las ultimas 10 filas desplazadas,
        # es decir, los ultimos 10 juegos ANTES del juego actual.
        roll_cols = []
        for col in _AGG_COLS:
            roll_name = f"roll_{col}"
            team_df[roll_name] = (
                team_df[col]
                .shift(1)
                .rolling(_ROLLING_WINDOW, min_periods=_MIN_PERIODS)
                .mean()
            )
            roll_cols.append(roll_name)

        # Almacenar filas con rolling valido en el lookup
        valid_mask = team_df[roll_cols].notna().all(axis=1)

        for _, row in team_df[valid_mask].iterrows():
            date_str = row["game_date"].strftime("%Y-%m-%d")
            lookup[(date_str, team_code)] = {
                _AGG_TO_FEATURE[col]: round(float(row[f"roll_{col}"]), 4)
                for col in _AGG_COLS
            }
            entries_added += 1

    return entries_added


def get_game_shot_chart(sc_lookup, date_str, home_team, away_team):
    """Obtiene features de Shot Chart para un partido especifico.

    Recibe nombres completos de equipos (como vienen de OddsData/create_games)
    y los traduce a codigos BRef internamente.

    Args:
        sc_lookup: dict de build_shot_chart_history()
        date_str: fecha del juego como 'YYYY-MM-DD'
        home_team: nombre completo del equipo local (ej. "Boston Celtics")
        away_team: nombre completo del equipo visitante (ej. "New York Knicks")

    Returns:
        dict con 12 features: 6 por equipo con sufijo HOME/AWAY.
        Si no hay datos, retorna valores default (promedios historicos).
    """
    home_bref = TEAM_NAME_TO_BREF.get(home_team)
    away_bref = TEAM_NAME_TO_BREF.get(away_team)

    if home_bref is None:
        logger.debug("Equipo local no encontrado en mapeo BRef: '%s'", home_team)
    if away_bref is None:
        logger.debug("Equipo visitante no encontrado en mapeo BRef: '%s'", away_team)

    # Obtener features del lookup o defaults
    home_feats = (
        sc_lookup.get((date_str, home_bref), _DEFAULTS)
        if home_bref
        else _DEFAULTS
    )
    away_feats = (
        sc_lookup.get((date_str, away_bref), _DEFAULTS)
        if away_bref
        else _DEFAULTS
    )

    result = {}
    for feat_name in _FEATURE_NAMES:
        result[f"{feat_name}_HOME"] = home_feats[feat_name]
        result[f"{feat_name}_AWAY"] = away_feats[feat_name]

    return result


def add_shot_chart_to_frame(frame, sc_features_list):
    """Agrega columnas de Shot Chart al DataFrame de juegos.

    Convierte la lista de dicts (uno por juego) en un DataFrame
    y lo concatena horizontalmente al frame existente.

    Args:
        frame: DataFrame de juegos (de create_games.py)
        sc_features_list: lista de dicts de get_game_shot_chart(),
            uno por cada fila del frame

    Returns:
        DataFrame con las 12 columnas de Shot Chart agregadas
    """
    if not sc_features_list:
        return frame
    sc_df = pd.DataFrame(sc_features_list, index=frame.index)
    return pd.concat([frame, sc_df], axis=1)
