"""Features de agregacion a nivel jugador para el modelo de equipos.

En vez de usar promedios de equipo (que tardan 60 juegos en reflejar una lesion),
agregamos stats individuales de los jugadores que realmente jugaron recientemente,
ponderados por minutos.

Concepto: "Player Performance Vectors" (Hubacek et al., 2019)
    Para cada equipo, calcular:
    - TOP5_PM36: media de plus/minus por 36 min de los 5 con mas minutos
    - TOP8_PM36: media de los 8 de rotacion
    - DEPTH_SCORE: TOP8/TOP5 ratio (profundidad del banco)
    - STAR_POWER: max PM36 individual (impacto del mejor jugador)
    - HHI_MINUTES: Herfindahl-Hirschman Index de distribucion de minutos
      - HHI alto (>0.15): produccion concentrada en 1-2 jugadores (mas volatil)
      - HHI bajo (<0.08): distribucion uniforme (mas consistente)

Features generadas (por equipo, home y away):
    - TOP5_PM36_HOME, TOP5_PM36_AWAY
    - DEPTH_SCORE_HOME, DEPTH_SCORE_AWAY
    - STAR_POWER_HOME, STAR_POWER_AWAY
    - HHI_MINUTES_HOME, HHI_MINUTES_AWAY

Usado por create_games.py (entrenamiento) y main.py (prediccion).

Nota sobre datos disponibles:
    - Temporadas 2012-2021: solo tienen MIN (7 columnas), NO tienen PLUS_MINUS
    - Temporadas 2022+: tienen 22 columnas incluyendo PLUS_MINUS
    - Para temporadas sin PLUS_MINUS, solo calculamos HHI_MINUTES
      (las demas features necesitan PLUS_MINUS y quedan como defaults)
"""

import sqlite3

import numpy as np
import pandas as pd

from src.config import PLAYER_LOGS_DB, get_logger
from src.sports.nba.features.team_mappings import TEAM_NAME_TO_ABBR as _TEAM_NAME_TO_ABBR

logger = get_logger(__name__)

# Temporadas con PLUS_MINUS disponible (22 columnas completas)
# Las temporadas 2012-2021 solo tienen 7 columnas (sin stats individuales)
_FULL_STATS_SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]

# Ventana de juegos recientes para calcular stats de jugadores
_ROLLING_WINDOW = 10

# Minimo de filas de jugadores para calcular features de un equipo
_MIN_TEAM_ROWS = 20

# Minimo de juegos por jugador para incluirlo en el calculo
_MIN_PLAYER_GAMES = 2


def build_lineup_strength_history(player_logs_db_path=None):
    """Construye historial de lineup strength para todos los equipos.

    Pre-procesa PlayerGameLogs.sqlite para computar rolling player stats
    por equipo y fecha. Para cada fecha, calcula las features de lineup
    usando los ultimos 10 juegos de cada jugador.

    Solo procesa temporadas con PLUS_MINUS (2022+). Para temporadas
    anteriores, create_games.py usara los valores default.

    Args:
        player_logs_db_path: ruta al archivo PlayerGameLogs.sqlite.
            Si es None, usa la ruta de src.config.PLAYER_LOGS_DB.

    Returns:
        dict[(date_str, team_abbr)] -> {
            "TOP5_PM36": float,
            "DEPTH_SCORE": float,
            "STAR_POWER": float,
            "HHI_MINUTES": float,
        }
    """
    if player_logs_db_path is None:
        player_logs_db_path = PLAYER_LOGS_DB

    lookup = {}

    with sqlite3.connect(player_logs_db_path) as con:
        # Obtener tablas de game logs existentes
        cursor = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'player_logs_%'"
        )
        tables = sorted([row[0] for row in cursor.fetchall()])

        for table_name in tables:
            # Extraer temporada del nombre de tabla: "player_logs_2022-23" -> "2022-23"
            season = table_name.replace("player_logs_", "")

            # Solo temporadas con PLUS_MINUS (22 columnas)
            if season not in _FULL_STATS_SEASONS:
                continue

            # Verificar que la tabla tiene PLUS_MINUS
            cols = [c[1] for c in con.execute(f'PRAGMA table_info("{table_name}")').fetchall()]
            if "PLUS_MINUS" not in cols:
                logger.warning("%s: sin columna PLUS_MINUS, saltando", season)
                continue

            try:
                df = pd.read_sql_query(
                    f'SELECT PLAYER_NAME, TEAM_ABBREVIATION, GAME_DATE, MIN, PLUS_MINUS '
                    f'FROM "{table_name}"',
                    con,
                )
            except Exception as e:
                logger.warning("Error leyendo %s: %s", table_name, e)
                continue

            if df.empty:
                continue

            # Convertir MIN a float (en PlayerGameLogs ya viene como float,
            # pero por seguridad manejamos el formato "MM:SS" tambien)
            df["MIN_FLOAT"] = df["MIN"].apply(_parse_minutes)
            df = df.dropna(subset=["MIN_FLOAT"])
            df = df[df["MIN_FLOAT"] > 0]

            # Parsear fechas
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
            df = df.dropna(subset=["GAME_DATE"])
            df = df.sort_values("GAME_DATE")

            # PM36: plus/minus normalizado a 36 minutos
            # Un jugador con +10 en 18 minutos es mas impactante que +10 en 36 minutos
            # PM36 = PLUS_MINUS * (36 / MIN)
            df["PM36"] = df["PLUS_MINUS"].astype(float) * 36.0 / df["MIN_FLOAT"]

            # Clipear PM36 a [-50, 50] para evitar outliers extremos
            # (jugadores con 2 minutos y +/- grande distorsionan el promedio)
            df["PM36"] = df["PM36"].clip(-50, 50)

            # Procesar cada fecha unica del equipo
            n_entries = _process_season_dates(df, lookup)
            logger.info(
                "Lineup strength %s: %d entradas generadas", season, n_entries
            )

    logger.info(
        "Lineup strength total: %d entradas (fecha, equipo)", len(lookup)
    )
    return lookup


def _process_season_dates(df, lookup):
    """Procesa todas las fechas de una temporada y llena el lookup.

    Para cada fecha de juego, calcula features de lineup usando
    SOLO datos ANTERIORES a esa fecha (evitar data leakage).

    Args:
        df: DataFrame con columnas PLAYER_NAME, TEAM_ABBREVIATION,
            GAME_DATE, MIN_FLOAT, PM36
        lookup: dict a llenar con (date_str, team_abbr) -> features

    Returns:
        int: numero de entradas generadas
    """
    entries_added = 0
    unique_dates = sorted(df["GAME_DATE"].dt.date.unique())

    for game_date in unique_dates:
        # Datos ANTES de esta fecha (T-1, evitar leakage)
        cutoff = pd.Timestamp(game_date)
        before = df[df["GAME_DATE"] < cutoff]

        if before.empty:
            continue

        # Equipos que juegan HOY (para saber cuales necesitan features)
        teams_today = df[
            df["GAME_DATE"].dt.date == game_date
        ]["TEAM_ABBREVIATION"].unique()

        for team_abbr in teams_today:
            team_data = before[before["TEAM_ABBREVIATION"] == team_abbr]

            # Necesitamos suficientes datos para calcular features confiables
            if len(team_data) < _MIN_TEAM_ROWS:
                continue

            # Ultimos N juegos del equipo (por fecha, no por jugador)
            recent_dates = sorted(team_data["GAME_DATE"].unique())[
                -_ROLLING_WINDOW:
            ]
            recent = team_data[team_data["GAME_DATE"].isin(recent_dates)]

            features = _compute_lineup_features(recent)
            if features:
                date_str = game_date.isoformat()
                lookup[(date_str, team_abbr)] = features
                entries_added += 1

    return entries_added


def _parse_minutes(min_val):
    """Convierte MIN a float. Maneja formato 'MM:SS' y float.

    En PlayerGameLogs.sqlite, MIN ya viene como float (ej. 32.0),
    pero por robustez manejamos el caso 'MM:SS' (ej. '32:15' -> 32.25).
    """
    if pd.isna(min_val):
        return np.nan
    if isinstance(min_val, (int, float)):
        return float(min_val)
    try:
        if ":" in str(min_val):
            parts = str(min_val).split(":")
            return float(parts[0]) + float(parts[1]) / 60.0
        return float(min_val)
    except (ValueError, IndexError):
        return np.nan


def _compute_lineup_features(team_recent_df):
    """Calcula features de lineup a partir de datos recientes del equipo.

    Concepto de cada feature:
    - TOP5_PM36: impacto de los 5 titulares (los que mas minutos juegan)
    - DEPTH_SCORE: que tan bueno es el banco relativo a los titulares
      > 1.0: banco mejor que titulares (raro, equipo atipico)
      ~ 0.8-0.9: banco solido (equipos profundos como BOS, OKC)
      < 0.7: drop-off grande cuando salen titulares (equipos top-heavy)
    - STAR_POWER: impacto del mejor jugador individual
    - HHI_MINUTES: concentracion de minutos (indice Herfindahl-Hirschman)
      > 0.15: produccion concentrada en 1-2 jugadores (mas volatil)
      < 0.08: distribucion uniforme (mas consistente)

    Args:
        team_recent_df: DataFrame con columnas MIN_FLOAT, PM36, PLAYER_NAME
                       para los ultimos ~10 juegos del equipo

    Returns:
        dict con TOP5_PM36, DEPTH_SCORE, STAR_POWER, HHI_MINUTES
        None si datos insuficientes
    """
    # Agregar stats por jugador: media de PM36, total de minutos, conteo de juegos
    player_stats = (
        team_recent_df.groupby("PLAYER_NAME")
        .agg(
            avg_pm36=("PM36", "mean"),
            total_min=("MIN_FLOAT", "sum"),
            games=("MIN_FLOAT", "count"),
        )
        .reset_index()
    )

    # Filtrar jugadores con al menos N juegos (evitar outliers de 1 juego)
    player_stats = player_stats[player_stats["games"] >= _MIN_PLAYER_GAMES]

    if len(player_stats) < 5:
        return None

    # Ordenar por minutos totales: los que mas juegan = rotacion principal
    player_stats = player_stats.sort_values("total_min", ascending=False)

    # TOP5: media PM36 de los 5 con mas minutos (~ titulares)
    top5 = player_stats.head(5)
    top5_pm36 = top5["avg_pm36"].mean()

    # TOP8: media PM36 de los 8 de rotacion (titulares + banco clave)
    top8 = player_stats.head(min(8, len(player_stats)))
    top8_pm36 = top8["avg_pm36"].mean()

    # DEPTH_SCORE: ratio TOP8/TOP5
    # Cuando TOP5_PM36 ≈ 0, el ratio explota (kurtosis 7,052 en EDA).
    # Fix: threshold más alto (0.1) y clip a [0.0, 3.0] para evitar outliers extremos.
    if abs(top5_pm36) < 0.1:
        depth = 1.0  # Sin diferencia significativa entre titulares y banco
    else:
        depth = max(0.0, min(3.0, top8_pm36 / top5_pm36))

    # STAR_POWER: max PM36 individual (el mejor jugador del equipo)
    star_power = player_stats["avg_pm36"].max()

    # HHI de distribucion de minutos (concentracion)
    # Formula: HHI = sum(s_i^2) donde s_i = share de minutos del jugador i
    # Ejemplo: 5 jugadores con 20% cada uno -> HHI = 5 * 0.04 = 0.20
    # Ejemplo: 1 jugador con 50%, 9 con ~5.5% -> HHI = 0.25 + 9*0.003 = 0.277
    total_minutes = player_stats["total_min"].sum()
    if total_minutes > 0:
        shares = player_stats["total_min"] / total_minutes
        hhi = (shares**2).sum()
    else:
        hhi = 0.1  # default neutral

    return {
        "TOP5_PM36": round(float(top5_pm36), 3),
        "DEPTH_SCORE": round(float(depth), 3),
        "STAR_POWER": round(float(star_power), 3),
        "HHI_MINUTES": round(float(hhi), 4),
    }


def get_game_lineup_features(lineup_lookup, game_date, home_team, away_team):
    """Obtiene features de lineup strength para un partido especifico.

    Esta funcion es el punto de entrada para create_games.py y main.py.
    Recibe nombres completos (OddsData) y los traduce a abreviaturas
    (PlayerGameLogs) internamente.

    Args:
        lineup_lookup: dict de build_lineup_strength_history()
        game_date: str 'YYYY-MM-DD'
        home_team: nombre completo del equipo local (ej. "Boston Celtics")
        away_team: nombre completo del equipo visitante (ej. "Los Angeles Lakers")

    Returns:
        dict con 8 features: 4 por equipo con sufijo HOME/AWAY
        Si no hay datos, retorna valores default (neutrales)
    """
    home_abbr = _TEAM_NAME_TO_ABBR.get(home_team)
    away_abbr = _TEAM_NAME_TO_ABBR.get(away_team)

    # Valores default neutrales para equipos sin datos
    # TOP5_PM36=0: impacto neutro, DEPTH_SCORE=1: banco igual a titulares,
    # STAR_POWER=0: sin estrella dominante, HHI=0.1: distribucion neutral
    defaults = {
        "TOP5_PM36": 0.0,
        "DEPTH_SCORE": 1.0,
        "STAR_POWER": 0.0,
        "HHI_MINUTES": 0.1,
    }

    if home_abbr is None:
        logger.debug("Equipo local no encontrado en mapeo: '%s'", home_team)
    if away_abbr is None:
        logger.debug("Equipo visitante no encontrado en mapeo: '%s'", away_team)

    home_feats = (
        lineup_lookup.get((game_date, home_abbr), defaults)
        if home_abbr
        else defaults
    )
    away_feats = (
        lineup_lookup.get((game_date, away_abbr), defaults)
        if away_abbr
        else defaults
    )

    return {
        "TOP5_PM36_HOME": home_feats["TOP5_PM36"],
        "TOP5_PM36_AWAY": away_feats["TOP5_PM36"],
        "DEPTH_SCORE_HOME": home_feats["DEPTH_SCORE"],
        "DEPTH_SCORE_AWAY": away_feats["DEPTH_SCORE"],
        "STAR_POWER_HOME": home_feats["STAR_POWER"],
        "STAR_POWER_AWAY": away_feats["STAR_POWER"],
        "HHI_MINUTES_HOME": home_feats["HHI_MINUTES"],
        "HHI_MINUTES_AWAY": away_feats["HHI_MINUTES"],
    }


def add_lineup_features_to_frame(frame, lineup_features_list):
    """Agrega columnas de lineup strength al DataFrame de juegos.

    Convierte la lista de dicts (una por juego) en un DataFrame
    y lo concatena horizontalmente al frame existente.

    Args:
        frame: DataFrame de juegos (de create_games.py)
        lineup_features_list: lista de dicts de get_game_lineup_features(),
            uno por cada fila del frame

    Returns:
        DataFrame con las 8 columnas de lineup agregadas
    """
    if not lineup_features_list:
        return frame
    lineup_df = pd.DataFrame(lineup_features_list, index=frame.index)
    return pd.concat([frame, lineup_df], axis=1)
