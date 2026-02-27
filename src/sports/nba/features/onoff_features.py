"""Features de On/Off Court Plus/Minus extraidas de BRefData.sqlite.

El on/off court plus/minus mide cuanto cambia el rendimiento del equipo
cuando un jugador esta en cancha vs en banca. Esto captura:
    - Dependencia de estrellas (si el equipo colapsa sin cierto jugador)
    - Varianza de impacto (si un solo jugador domina vs impacto distribuido)

Features generadas (4 por juego, 2 por equipo):
    ONOFF_NET_TOP5_HOME, ONOFF_NET_TOP5_AWAY
        Promedio del net (on - off) de los top 5 jugadores (por abs(on_court))
        -> Mide dependencia de las estrellas del equipo

    ONOFF_SPREAD_HOME, ONOFF_SPREAD_AWAY
        max(net) - min(net) por juego
        -> Alta varianza = un jugador domina, baja = impacto distribuido

Datos fuente:
    - BRefData.sqlite, tablas plus_minus_YYYY-ZZ (2014-15 a 2025-26)
    - Schema: game_id, game_date, team_side, player_name, on_court, off_court, net
    - team_side es 'home' o 'away' (NO un codigo de equipo)
    - Necesitamos tabla schedules para mapear game_id + team_side -> bref_code

Patron: build -> get -> add (mismo que bref_game_features.py).

Calculo:
    - Rolling 10 juegos con shift(1) para T-1 estricto (sin leakage)
    - min_periods=3: necesitamos al menos 3 juegos previos
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

# Cuantos jugadores top considerar para ONOFF_NET_TOP5
_TOP_N_PLAYERS = 5

# Columnas de features calculadas por juego
_STAT_COLS = ["ONOFF_NET_TOP5", "ONOFF_SPREAD"]

# Valores default cuando no hay datos suficientes
# NET_TOP5=0: impacto neutral; SPREAD=10: varianza tipica en la NBA
_DEFAULTS = {
    "ONOFF_NET_TOP5": 0.0,
    "ONOFF_SPREAD": 10.0,
}

# Mapeo inverso: nombre completo de schedules -> codigo BRef
# Reutiliza TEAM_NAME_TO_BREF que ya incluye aliases historicos
_SCHEDULE_TEAM_TO_BREF = TEAM_NAME_TO_BREF


def _build_game_map(con):
    """Construye mapeo game_id -> {'home': bref_code, 'away': bref_code} desde schedules.

    Lee la tabla schedules y convierte los nombres completos de equipos
    (ej. "Boston Celtics") a codigos BRef (ej. "BOS") usando el mapeo existente.

    Args:
        con: conexion sqlite3 abierta a BRefData.sqlite

    Returns:
        dict[str, dict]: game_id -> {'home': str, 'away': str}
    """
    try:
        schedules = pd.read_sql_query(
            "SELECT game_id, home_team, away_team FROM schedules", con
        )
    except Exception as e:
        logger.warning("No se pudo leer tabla schedules: %s", e)
        return {}

    game_map = {}
    for _, row in schedules.iterrows():
        gid = row["game_id"]
        home_bref = _SCHEDULE_TEAM_TO_BREF.get(row["home_team"])
        away_bref = _SCHEDULE_TEAM_TO_BREF.get(row["away_team"])

        if home_bref is None or away_bref is None:
            # Equipos no mapeados (raro, pero posible con datos historicos)
            continue

        game_map[gid] = {"home": home_bref, "away": away_bref}

    logger.info("Game map construido: %d juegos con equipos mapeados", len(game_map))
    return game_map


def _compute_game_features(group_df):
    """Calcula ONOFF_NET_TOP5 y ONOFF_SPREAD para un grupo (un juego, un equipo).

    ONOFF_NET_TOP5:
        Toma los top 5 jugadores por abs(on_court) — los que mas minutos/impacto
        tienen — y promedia su net (on_court - off_court).
        Un valor positivo alto = estrellas aportan mucho mas que el banco.

    ONOFF_SPREAD:
        max(net) - min(net) entre TODOS los jugadores del juego.
        Alto = un jugador domina vs el resto. Bajo = impacto parejo.

    Args:
        group_df: DataFrame con columnas on_court, off_court, net para UN juego y UN equipo

    Returns:
        pd.Series con ONOFF_NET_TOP5 y ONOFF_SPREAD
    """
    if group_df.empty:
        return pd.Series(_DEFAULTS)

    # Top 5 por abs(on_court): jugadores con mayor impacto cuando estan en cancha
    sorted_by_impact = group_df.reindex(
        group_df["on_court"].abs().sort_values(ascending=False).index
    )
    top_n = sorted_by_impact.head(_TOP_N_PLAYERS)

    net_top5 = top_n["net"].mean() if len(top_n) > 0 else 0.0
    spread = group_df["net"].max() - group_df["net"].min() if len(group_df) > 1 else 0.0

    return pd.Series({
        "ONOFF_NET_TOP5": net_top5,
        "ONOFF_SPREAD": spread,
    })


def build_onoff_history(bref_db_path=None):
    """Pre-construye promedios rolling de 10 juegos de On/Off Court features.

    Estrategia:
    1. Carga tabla schedules para crear mapeo game_id -> {home_bref, away_bref}
    2. Para cada temporada, carga plus_minus_YYYY-ZZ
    3. Mapea cada fila a su codigo BRef via game_id + team_side
    4. Por juego por equipo: calcula ONOFF_NET_TOP5 y ONOFF_SPREAD
    5. Rolling 10 juegos T-1 estricto por equipo

    Args:
        bref_db_path: ruta al archivo BRefData.sqlite.
            Si es None, usa la ruta de src.config.BREF_DB.

    Returns:
        dict[(date_str, bref_code)] -> {
            "ONOFF_NET_TOP5": float,
            "ONOFF_SPREAD": float,
        }
    """
    if bref_db_path is None:
        bref_db_path = BREF_DB

    lookup = {}

    with sqlite3.connect(bref_db_path) as con:
        # Paso 1: construir mapeo game_id -> codigos BRef
        game_map = _build_game_map(con)
        if not game_map:
            logger.warning("Game map vacio — no se generaran features on/off")
            return lookup

        # Verificar tablas plus_minus existentes
        cursor = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name LIKE 'plus_minus_%'"
        )
        existing_tables = {row[0] for row in cursor.fetchall()}

        all_game_stats = []  # Acumular stats de todos los juegos

        for season in SEASONS:
            table_name = f"plus_minus_{season}"
            if table_name not in existing_tables:
                logger.debug("Tabla %s no existe, saltando", table_name)
                continue

            try:
                df = pd.read_sql_query(
                    f'SELECT game_id, game_date, team_side, player_name, '
                    f'on_court, off_court, net '
                    f'FROM "{table_name}"',
                    con,
                )
            except Exception as e:
                logger.warning("Error leyendo %s: %s", table_name, e)
                continue

            if df.empty:
                logger.debug("Tabla %s vacia", table_name)
                continue

            # Convertir columnas numericas
            for col in ["on_court", "off_court", "net"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Parsear fechas
            df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
            df = df.dropna(subset=["game_date", "on_court", "net"])

            # Paso 2: mapear team_side a codigo BRef usando game_map
            # Enfoque vectorizado para rendimiento
            home_codes = df["game_id"].map(
                lambda gid: game_map.get(gid, {}).get("home")
            )
            away_codes = df["game_id"].map(
                lambda gid: game_map.get(gid, {}).get("away")
            )
            df["bref_code"] = np.where(
                df["team_side"] == "home", home_codes, away_codes
            )

            # Descartar filas sin mapeo (juegos no encontrados en schedules)
            df = df.dropna(subset=["bref_code"])
            if df.empty:
                logger.debug("Sin mapeos validos en %s", table_name)
                continue

            # Paso 3: calcular features por juego por equipo
            game_stats = (
                df.groupby(["game_date", "bref_code"])
                .apply(_compute_game_features, include_groups=False)
                .reset_index()
            )

            all_game_stats.append(game_stats)
            logger.info(
                "On/Off %s: %d juegos-equipo procesados", season, len(game_stats)
            )

        if not all_game_stats:
            logger.warning("No se encontraron datos de plus_minus en ninguna temporada")
            return lookup

        # Concatenar todas las temporadas
        all_df = pd.concat(all_game_stats, ignore_index=True)

        # Paso 4: rolling T-1 por equipo
        n_entries = _compute_rolling(all_df, lookup)
        logger.info("On/Off total: %d entradas (fecha, equipo)", n_entries)

    return lookup


def _compute_rolling(all_df, lookup):
    """Calcula rolling T-1 de 10 juegos por equipo y llena el lookup.

    Para cada equipo:
    1. Ordena por fecha
    2. shift(1) para excluir el juego actual (T-1 estricto)
    3. rolling(10, min_periods=3) para promediar ultimos 10 juegos previos
    4. Almacena en lookup con clave (date_str, bref_code)

    Args:
        all_df: DataFrame con game_date, bref_code, ONOFF_NET_TOP5, ONOFF_SPREAD
        lookup: dict a llenar

    Returns:
        int: numero de entradas generadas
    """
    entries_added = 0

    for bref_code, team_df in all_df.groupby("bref_code"):
        team_df = team_df.sort_values("game_date").copy()

        # Rolling T-1: shift(1) desplaza para no incluir juego actual
        for col in _STAT_COLS:
            team_df[f"roll_{col}"] = (
                team_df[col]
                .shift(1)
                .rolling(_ROLLING_WINDOW, min_periods=_MIN_PERIODS)
                .mean()
            )

        # Filtrar filas con rolling valido
        roll_cols = [f"roll_{col}" for col in _STAT_COLS]
        valid_mask = team_df[roll_cols].notna().all(axis=1)

        for _, row in team_df[valid_mask].iterrows():
            date_str = row["game_date"].strftime("%Y-%m-%d")
            lookup[(date_str, bref_code)] = {
                col: round(float(row[f"roll_{col}"]), 4)
                for col in _STAT_COLS
            }
            entries_added += 1

    return entries_added


def get_game_onoff(onoff_lookup, date_str, home_team, away_team):
    """Obtiene features de On/Off Court para un partido especifico.

    Recibe nombres completos de equipos (como vienen de OddsData/create_games)
    y los traduce a codigos BRef internamente.

    Args:
        onoff_lookup: dict de build_onoff_history()
        date_str: fecha del juego como 'YYYY-MM-DD'
        home_team: nombre completo del equipo local (ej. "Boston Celtics")
        away_team: nombre completo del equipo visitante (ej. "New York Knicks")

    Returns:
        dict con 4 features: 2 por equipo con sufijo HOME/AWAY.
        Si no hay datos, retorna valores default.
    """
    home_bref = TEAM_NAME_TO_BREF.get(home_team)
    away_bref = TEAM_NAME_TO_BREF.get(away_team)

    if home_bref is None:
        logger.debug("Equipo local no encontrado en mapeo BRef: '%s'", home_team)
    if away_bref is None:
        logger.debug("Equipo visitante no encontrado en mapeo BRef: '%s'", away_team)

    # Obtener features del lookup o defaults
    home_feats = (
        onoff_lookup.get((date_str, home_bref), _DEFAULTS)
        if home_bref
        else _DEFAULTS
    )
    away_feats = (
        onoff_lookup.get((date_str, away_bref), _DEFAULTS)
        if away_bref
        else _DEFAULTS
    )

    return {
        "ONOFF_NET_TOP5_HOME": home_feats["ONOFF_NET_TOP5"],
        "ONOFF_NET_TOP5_AWAY": away_feats["ONOFF_NET_TOP5"],
        "ONOFF_SPREAD_HOME": home_feats["ONOFF_SPREAD"],
        "ONOFF_SPREAD_AWAY": away_feats["ONOFF_SPREAD"],
    }


def add_onoff_to_frame(frame, onoff_features_list):
    """Agrega columnas de On/Off Court al DataFrame de juegos.

    Convierte la lista de dicts (uno por juego) en un DataFrame
    y lo concatena horizontalmente al frame existente.

    Args:
        frame: DataFrame de juegos (de create_games.py)
        onoff_features_list: lista de dicts de get_game_onoff(),
            uno por cada fila del frame

    Returns:
        DataFrame con las 4 columnas de On/Off Court agregadas
    """
    if not onoff_features_list:
        return frame
    onoff_df = pd.DataFrame(onoff_features_list, index=frame.index)
    return pd.concat([frame, onoff_df], axis=1)
