"""Features de Four Factors extraidas de BRefData.sqlite.

Los Four Factors (Oliver, 2004) son las 4 metricas que mas explican
victorias en basketball:
    1. eFG% (Effective Field Goal %) — eficiencia de tiro
    2. TOV% (Turnover %)             — cuidado del balon
    3. ORB% (Offensive Rebound %)    — segundas oportunidades
    4. FT/FGA (Free Throw Rate)      — llegar a la linea

Ademas incluimos:
    5. Pace — posesiones por 48 min (ritmo de juego)
    6. ORtg — puntos por 100 posesiones (eficiencia ofensiva global)

Features generadas (12 por juego, 6 por equipo):
    FF_PACE_HOME, FF_PACE_AWAY      — ritmo
    FF_ORTG_HOME, FF_ORTG_AWAY      — eficiencia ofensiva
    FF_EFG_HOME, FF_EFG_AWAY        — eficiencia de tiro
    FF_TOV_HOME, FF_TOV_AWAY        — perdidas de balon
    FF_ORB_HOME, FF_ORB_AWAY        — rebotes ofensivos
    FF_FT_FGA_HOME, FF_FT_FGA_AWAY  — tasa de tiros libres

Patron: build -> get -> add (mismo que lineup_strength.py).

Calculo:
    - Rolling 10 juegos con shift(1) para T-1 estricto (sin leakage)
    - min_periods=3: necesitamos al menos 3 juegos previos
    - Datos de BRefData.sqlite, tablas four_factors_YYYY-ZZ (2014-15 a 2025-26)
"""

import sqlite3

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

# Columnas de stats en la tabla de BRef
_STATS_COLS = ["pace", "efg_pct", "tov_pct", "orb_pct", "ft_fga", "ortg"]

# Mapeo de columna BRef -> nombre de feature
_COL_TO_FEATURE = {
    "pace": "FF_PACE",
    "ortg": "FF_ORTG",
    "efg_pct": "FF_EFG",
    "tov_pct": "FF_TOV",
    "orb_pct": "FF_ORB",
    "ft_fga": "FF_FT_FGA",
}

# Valores default cuando no hay datos suficientes
# (promedios historicos aproximados de la NBA)
_DEFAULTS = {
    "FF_PACE": 100.0,
    "FF_ORTG": 110.0,
    "FF_EFG": 0.50,
    "FF_TOV": 13.0,
    "FF_ORB": 25.0,
    "FF_FT_FGA": 0.20,
}


def build_four_factors_history(bref_db_path=None):
    """Pre-construye promedios rolling de 10 juegos de Four Factors para todos los equipos.

    Lee todas las tablas four_factors_YYYY-ZZ de BRefData.sqlite,
    calcula rolling means con T-1 estricto (shift + rolling) y
    almacena en un dict para lookup rapido.

    Args:
        bref_db_path: ruta al archivo BRefData.sqlite.
            Si es None, usa la ruta de src.config.BREF_DB.

    Returns:
        dict[(date_str, bref_code)] -> {
            "FF_PACE": float, "FF_ORTG": float, "FF_EFG": float,
            "FF_TOV": float, "FF_ORB": float, "FF_FT_FGA": float,
        }
    """
    if bref_db_path is None:
        bref_db_path = BREF_DB

    lookup = {}

    with sqlite3.connect(bref_db_path) as con:
        # Verificar que tablas existen en la BD
        cursor = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name LIKE 'four_factors_%'"
        )
        existing_tables = {row[0] for row in cursor.fetchall()}

        for season in SEASONS:
            table_name = f"four_factors_{season}"
            if table_name not in existing_tables:
                logger.debug("Tabla %s no existe, saltando", table_name)
                continue

            try:
                df = pd.read_sql_query(
                    f'SELECT game_date, team_name, '
                    f'pace, efg_pct, tov_pct, orb_pct, ft_fga, ortg '
                    f'FROM "{table_name}"',
                    con,
                )
            except Exception as e:
                logger.warning("Error leyendo %s: %s", table_name, e)
                continue

            if df.empty:
                logger.debug("Tabla %s vacia", table_name)
                continue

            # Convertir columnas de stats a float por seguridad
            for col in _STATS_COLS:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Parsear fechas y eliminar filas invalidas
            df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
            df = df.dropna(subset=["game_date"])

            # Procesar cada equipo por separado para calcular rolling
            n_entries = _process_season_rolling(df, lookup)
            logger.info(
                "Four Factors %s: %d entradas generadas", season, n_entries
            )

    logger.info(
        "Four Factors total: %d entradas (fecha, equipo)", len(lookup)
    )
    return lookup


def _process_season_rolling(df, lookup):
    """Calcula rolling por equipo para una temporada y llena el lookup.

    Para cada equipo:
    1. Ordena por fecha
    2. shift(1) para excluir el juego actual (T-1)
    3. rolling(10, min_periods=3) para promediar los ultimos 10 juegos previos
    4. Almacena cada fila con rolling valido en el lookup

    Args:
        df: DataFrame con game_date, team_name y columnas de stats
        lookup: dict a llenar con (date_str, bref_code) -> features

    Returns:
        int: numero de entradas generadas
    """
    entries_added = 0

    for bref_code, team_df in df.groupby("team_name"):
        # Ordenar cronologicamente
        team_df = team_df.sort_values("game_date").copy()

        # Calcular rolling con T-1 estricto:
        # shift(1) desplaza una fila hacia abajo -> el valor en fila i
        # corresponde al dato de la fila i-1 (juego anterior).
        # Luego rolling(10) promedia las ultimas 10 filas desplazadas,
        # es decir, los ultimos 10 juegos ANTES del juego actual.
        for col in _STATS_COLS:
            feature_name = _COL_TO_FEATURE[col]
            team_df[feature_name] = (
                team_df[col]
                .shift(1)
                .rolling(_ROLLING_WINDOW, min_periods=_MIN_PERIODS)
                .mean()
            )

        # Almacenar filas con rolling valido en el lookup
        feature_names = list(_COL_TO_FEATURE.values())
        valid_mask = team_df[feature_names].notna().all(axis=1)

        for _, row in team_df[valid_mask].iterrows():
            date_str = row["game_date"].strftime("%Y-%m-%d")
            lookup[(date_str, bref_code)] = {
                feat: round(float(row[feat]), 4) for feat in feature_names
            }
            entries_added += 1

    return entries_added


def get_game_four_factors(ff_lookup, date_str, home_team, away_team):
    """Obtiene features de Four Factors para un partido especifico.

    Recibe nombres completos de equipos (como vienen de OddsData/create_games)
    y los traduce a codigos BRef internamente.

    Args:
        ff_lookup: dict de build_four_factors_history()
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
        ff_lookup.get((date_str, home_bref), _DEFAULTS)
        if home_bref
        else _DEFAULTS
    )
    away_feats = (
        ff_lookup.get((date_str, away_bref), _DEFAULTS)
        if away_bref
        else _DEFAULTS
    )

    return {
        "FF_PACE_HOME": home_feats["FF_PACE"],
        "FF_PACE_AWAY": away_feats["FF_PACE"],
        "FF_ORTG_HOME": home_feats["FF_ORTG"],
        "FF_ORTG_AWAY": away_feats["FF_ORTG"],
        "FF_EFG_HOME": home_feats["FF_EFG"],
        "FF_EFG_AWAY": away_feats["FF_EFG"],
        "FF_TOV_HOME": home_feats["FF_TOV"],
        "FF_TOV_AWAY": away_feats["FF_TOV"],
        "FF_ORB_HOME": home_feats["FF_ORB"],
        "FF_ORB_AWAY": away_feats["FF_ORB"],
        "FF_FT_FGA_HOME": home_feats["FF_FT_FGA"],
        "FF_FT_FGA_AWAY": away_feats["FF_FT_FGA"],
    }


def add_four_factors_to_frame(frame, ff_features_list):
    """Agrega columnas de Four Factors al DataFrame de juegos.

    Convierte la lista de dicts (uno por juego) en un DataFrame
    y lo concatena horizontalmente al frame existente.

    Args:
        frame: DataFrame de juegos (de create_games.py)
        ff_features_list: lista de dicts de get_game_four_factors(),
            uno por cada fila del frame

    Returns:
        DataFrame con las 12 columnas de Four Factors agregadas
    """
    if not ff_features_list:
        return frame
    ff_df = pd.DataFrame(ff_features_list, index=frame.index)
    return pd.concat([frame, ff_df], axis=1)
