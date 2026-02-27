"""Features de Zone Shooting Profile extraidas de BRefData.sqlite.

El perfil de tiro por zona de un equipo captura COMO y DONDE ataca ofensivamente:
  - Distancia promedio de tiro (AVG_DIST)
  - Tasa de intentos de 3 (FG3A_RATE)
  - Eficiencia en area restringida (PAINT_FG_PCT)
  - Eficiencia en medio rango cercano (CLOSE_MID_FG_PCT)
  - Eficiencia en medio rango lejano (MID_FG_PCT)
  - Eficiencia en largo 2 (LONG2_FG_PCT)
  - Eficiencia desde esquina 3 (CORNER3_PCT)
  - Tasa de mates (DUNK_RATE) — indicador de atletismo y ataque al aro

Features generadas (16 por juego, 8 por equipo):
    ZONE_AVG_DIST_HOME, ZONE_AVG_DIST_AWAY
    ZONE_FG3A_RATE_HOME, ZONE_FG3A_RATE_AWAY
    ZONE_PAINT_FG_PCT_HOME, ZONE_PAINT_FG_PCT_AWAY
    ZONE_CLOSE_MID_FG_PCT_HOME, ZONE_CLOSE_MID_FG_PCT_AWAY
    ZONE_MID_FG_PCT_HOME, ZONE_MID_FG_PCT_AWAY
    ZONE_LONG2_FG_PCT_HOME, ZONE_LONG2_FG_PCT_AWAY
    ZONE_CORNER3_PCT_HOME, ZONE_CORNER3_PCT_AWAY
    ZONE_DUNK_RATE_HOME, ZONE_DUNK_RATE_AWAY

Patron: build -> get -> add (mismo que bref_game_features.py).

Calculo:
    - Agregado por temporada (no rolling game-by-game)
    - Los datos son por jugador, se agregan por team_code ponderando por mp (minutos)
    - Solo jugadores con games > 10 para evitar ruido de muestras pequenas
    - Para T-1: se usa la temporada ANTERIOR (los datos de zona son agregados de temporada,
      no tenemos granularidad por juego, asi que usar la temporada actual seria leakage)
"""

import sqlite3

import pandas as pd

from src.config import BREF_DB, get_logger
from src.sports.nba.features.team_mappings import TEAM_NAME_TO_BREF

logger = get_logger(__name__)

# ── Temporadas disponibles en BRefData.sqlite ──────────────────────────
SEASONS = [
    "2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24",
    "2024-25", "2025-26",
]

# Mapeo temporada actual -> temporada previa (para T-1)
_PREV_SEASON = {
    "2015-16": "2014-15",
    "2016-17": "2015-16",
    "2017-18": "2016-17",
    "2018-19": "2017-18",
    "2019-20": "2018-19",
    "2020-21": "2019-20",
    "2021-22": "2020-21",
    "2022-23": "2021-22",
    "2023-24": "2022-23",
    "2024-25": "2023-24",
    "2025-26": "2024-25",
}

# ── Columnas BRef -> feature names ─────────────────────────────────────
# Cada tupla: (columna_bref, nombre_feature, es_fg_pct_o_rate)
_ZONE_COLS = [
    ("avg_dist",       "ZONE_AVG_DIST"),
    ("pct_fga_fg3a",   "ZONE_FG3A_RATE"),
    ("fg_pct_00_03",   "ZONE_PAINT_FG_PCT"),
    ("fg_pct_03_10",   "ZONE_CLOSE_MID_FG_PCT"),
    ("fg_pct_10_16",   "ZONE_MID_FG_PCT"),
    ("fg_pct_16_xx",   "ZONE_LONG2_FG_PCT"),
    ("fg_pct_corner3", "ZONE_CORNER3_PCT"),
    ("pct_fga_dunk",   "ZONE_DUNK_RATE"),
]

# Columnas fuente de BRef que necesitamos leer
_BREF_COLS = [col for col, _ in _ZONE_COLS]

# Nombres de features finales (sin sufijo HOME/AWAY)
_FEATURE_NAMES = [feat for _, feat in _ZONE_COLS]

# ── Valores default (promedios historicos NBA) ─────────────────────────
# Se usan cuando no hay datos de temporada previa (ej: 2014-15 no tiene T-1)
_DEFAULTS = {
    "ZONE_AVG_DIST":          14.0,
    "ZONE_FG3A_RATE":         0.35,
    "ZONE_PAINT_FG_PCT":      0.60,
    "ZONE_CLOSE_MID_FG_PCT":  0.40,
    "ZONE_MID_FG_PCT":        0.38,
    "ZONE_LONG2_FG_PCT":      0.38,
    "ZONE_CORNER3_PCT":       0.38,
    "ZONE_DUNK_RATE":         0.06,
}

# Defaults por columna BRef (para fillna antes de promediar)
# Jugadores con 0 intentos en una zona tendran None en fg_pct de esa zona
_COL_DEFAULTS = {
    "avg_dist":       14.0,
    "pct_fga_fg3a":   0.35,
    "fg_pct_00_03":   0.60,
    "fg_pct_03_10":   0.40,
    "fg_pct_10_16":   0.38,
    "fg_pct_16_xx":   0.38,
    "fg_pct_corner3": 0.38,
    "pct_fga_dunk":   0.06,
}

# Minimo de juegos para incluir a un jugador
_MIN_GAMES = 10


# ── Funciones auxiliares ───────────────────────────────────────────────

def _weighted_avg(group, col, weight_col="mp"):
    """Promedio ponderado por minutos jugados.

    Si todos los pesos son 0 o NaN, retorna el promedio simple como fallback.
    """
    w = group[weight_col]
    v = group[col]
    total_w = w.sum()
    if total_w == 0:
        return v.mean()
    return (v * w).sum() / total_w


def _season_to_table(season_str):
    """Convierte '2024-25' -> 'team_shooting_2024-25'."""
    return f"team_shooting_{season_str}"


# ── Funciones principales (build -> get -> add) ───────────────────────

def build_zone_shooting_lookup(bref_db_path=None):
    """Pre-construye perfiles de tiro por zona a nivel de temporada para todos los equipos.

    Lee las tablas team_shooting_YYYY-ZZ de BRefData.sqlite. Para cada temporada:
      1. Filtra jugadores con games > 10 y mp no nulo
      2. Rellena NaN en columnas de eficiencia con defaults por zona
      3. Agrupa por team_code y calcula promedios ponderados por mp

    Args:
        bref_db_path: ruta al BRefData.sqlite. Si None, usa BREF_DB de config.

    Returns:
        dict[(season_str, bref_code)] -> {
            "ZONE_AVG_DIST": float,
            "ZONE_FG3A_RATE": float,
            "ZONE_PAINT_FG_PCT": float,
            "ZONE_CLOSE_MID_FG_PCT": float,
            "ZONE_MID_FG_PCT": float,
            "ZONE_LONG2_FG_PCT": float,
            "ZONE_CORNER3_PCT": float,
            "ZONE_DUNK_RATE": float,
        }
    """
    db_path = bref_db_path or str(BREF_DB)
    lookup = {}

    try:
        conn = sqlite3.connect(db_path)
    except Exception as e:
        logger.error("No se pudo conectar a BRefData.sqlite: %s", e)
        return lookup

    try:
        for season in SEASONS:
            table = _season_to_table(season)

            # Verificar que la tabla existe
            check = pd.read_sql_query(
                f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'",
                conn,
            )
            if check.empty:
                logger.debug("Tabla %s no encontrada, saltando temporada", table)
                continue

            # Columnas necesarias: team_code, mp, games + columnas de zona
            cols_needed = ["team_code", "mp", "games"] + _BREF_COLS
            try:
                df = pd.read_sql_query(f'SELECT {", ".join(cols_needed)} FROM "{table}"', conn)
            except Exception as e:
                logger.warning("Error leyendo %s: %s", table, e)
                continue

            if df.empty:
                logger.debug("Tabla %s vacia", table)
                continue

            # --- Limpieza ---
            # Filtrar filas sin minutos (mp=None o NaN)
            df["mp"] = pd.to_numeric(df["mp"], errors="coerce")
            df = df.dropna(subset=["mp"])
            df = df[df["mp"] > 0]

            # Filtrar jugadores con pocas apariciones (muestra muy chica)
            df["games"] = pd.to_numeric(df["games"], errors="coerce")
            df = df[df["games"] > _MIN_GAMES]

            if df.empty:
                logger.debug("Temporada %s sin jugadores con games > %d", season, _MIN_GAMES)
                continue

            # Convertir columnas de zona a numerico y rellenar NaN con defaults
            for col in _BREF_COLS:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(_COL_DEFAULTS.get(col, 0.0))

            # --- Agregacion por equipo (promedio ponderado por mp) ---
            for team_code, grp in df.groupby("team_code"):
                profile = {}
                for bref_col, feat_name in _ZONE_COLS:
                    profile[feat_name] = round(_weighted_avg(grp, bref_col), 4)

                lookup[(season, team_code)] = profile

            logger.debug(
                "Temporada %s: %d equipos procesados",
                season,
                df["team_code"].nunique(),
            )

    finally:
        conn.close()

    logger.info(
        "Zone shooting lookup construido: %d entradas (equipo-temporada)", len(lookup)
    )
    return lookup


def get_game_zone_shooting(zone_lookup, season_key, date_str, home_team, away_team):
    """Obtiene 16 features de zone shooting para un juego (8 por equipo).

    Para evitar data leakage, usa la temporada ANTERIOR al season_key dado.
    Esto es porque los datos de zone shooting son agregados por temporada (no por juego),
    asi que usar la temporada actual significaria incluir informacion futura.

    Args:
        zone_lookup: dict construido por build_zone_shooting_lookup()
        season_key: temporada actual del juego, ej: "2024-25"
        date_str: fecha del juego (formato YYYY-MM-DD). No se usa directamente para el
                  lookup, pero se preserva por consistencia con la interfaz de otros modulos.
        home_team: nombre completo del equipo local, ej: "Boston Celtics"
        away_team: nombre completo del equipo visitante, ej: "New York Knicks"

    Returns:
        dict con 16 features:
            ZONE_AVG_DIST_HOME, ZONE_AVG_DIST_AWAY,
            ZONE_FG3A_RATE_HOME, ZONE_FG3A_RATE_AWAY,
            ZONE_PAINT_FG_PCT_HOME, ZONE_PAINT_FG_PCT_AWAY,
            ZONE_CLOSE_MID_FG_PCT_HOME, ZONE_CLOSE_MID_FG_PCT_AWAY,
            ZONE_MID_FG_PCT_HOME, ZONE_MID_FG_PCT_AWAY,
            ZONE_LONG2_FG_PCT_HOME, ZONE_LONG2_FG_PCT_AWAY,
            ZONE_CORNER3_PCT_HOME, ZONE_CORNER3_PCT_AWAY,
            ZONE_DUNK_RATE_HOME, ZONE_DUNK_RATE_AWAY
    """
    result = {}

    # Determinar temporada previa para T-1
    prev_season = _PREV_SEASON.get(season_key)

    for team_name, suffix in [(home_team, "HOME"), (away_team, "AWAY")]:
        # Convertir nombre completo a codigo BRef
        bref_code = TEAM_NAME_TO_BREF.get(team_name)

        if bref_code is None:
            logger.warning("Equipo '%s' no encontrado en TEAM_NAME_TO_BREF", team_name)
            # Usar defaults para equipo desconocido
            for feat_name in _FEATURE_NAMES:
                result[f"{feat_name}_{suffix}"] = _DEFAULTS[feat_name]
            continue

        # Buscar perfil de la temporada previa
        profile = None
        if prev_season is not None:
            profile = zone_lookup.get((prev_season, bref_code))

        if profile is None:
            # Sin temporada previa (ej: 2014-15) o equipo no encontrado -> defaults
            if prev_season is not None:
                logger.debug(
                    "Sin datos de zone shooting para %s en %s, usando defaults",
                    bref_code, prev_season,
                )
            for feat_name in _FEATURE_NAMES:
                result[f"{feat_name}_{suffix}"] = _DEFAULTS[feat_name]
        else:
            # Asignar features del perfil de la temporada previa
            for feat_name in _FEATURE_NAMES:
                result[f"{feat_name}_{suffix}"] = profile.get(
                    feat_name, _DEFAULTS[feat_name]
                )

    return result


def add_zone_shooting_to_frame(frame, zone_features_list):
    """Agrega columnas de zone shooting a un DataFrame existente.

    Args:
        frame: pd.DataFrame al que se le agregan las columnas (tipicamente el dataset
               de create_games.py con una fila por juego).
        zone_features_list: lista de dicts, uno por fila del frame, con las 16 features
                            generadas por get_game_zone_shooting(). Debe tener la misma
                            longitud que frame.

    Returns:
        pd.DataFrame con las 16 columnas de zone shooting agregadas.
    """
    if len(zone_features_list) != len(frame):
        logger.error(
            "Longitud de zone_features_list (%d) != longitud del frame (%d)",
            len(zone_features_list), len(frame),
        )
        return frame

    zone_df = pd.DataFrame(zone_features_list, index=frame.index)
    return pd.concat([frame, zone_df], axis=1)
