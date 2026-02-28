"""Features de ESPNLines.sqlite: movimiento de lineas y desacuerdo entre casas.

Integra datos de ESPNLines.sqlite (opening/closing lines de ESPN consensus +
por proveedor) para features de movimiento de mercado pre-tipoff.

Features generadas:
    ESPN_LINE_MOVE:         spread_movement (apertura → cierre), pre-tipoff safe
    ESPN_TOTAL_MOVE:        total_movement (apertura → cierre), pre-tipoff safe
    ESPN_OPEN_ML_PROB:      consensus_open_ml_prob, pre-tipoff safe
    ESPN_BOOK_DISAGREEMENT: std de open_spread_home entre casas (solo 2023-24)

COBERTURA: ~24% de los partidos en OddsData/dataset.sqlite tienen datos ESPN.
Ambos son scrapers parciales del calendario NBA con sets solapados pero distintos.
El 76% de filas tendran NaN — XGBoost maneja NaN nativamente (missingness pathways).

T-1 SAFETY:
    USAR: columnas _open_ (consensus_open_spread, consensus_open_ml_prob, etc.)
          spread_movement y total_movement son seguros (=close-open, conocidos al tipoff)
    NO USAR como features de modelo: consensus_close_spread, consensus_close_ml_prob
          (solo para calculo de CLV en notebooks de evaluacion)

Patron: build → get → add (identico a bref_game_features.py)
"""

import sqlite3
from datetime import date as _date
from datetime import timedelta

import numpy as np
from pathlib import Path

import pandas as pd

from src.config import ESPN_LINES_DB, get_logger

logger = get_logger(__name__)

# Temporadas disponibles en ESPNLines.sqlite
ESPNLINES_SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]

# Solo 2023-24 tiene datos reales de multiples proveedores para BOOK_DISAGREEMENT
# 2022-23: spreads todos NULL en espn_lines_2022-23
# 2024-25: solo 1 proveedor (ESPN BET) — std de 1 valor es NaN
BOOK_DISAGREEMENT_SEASONS = ["2023-24"]

# Proveedores a excluir del calculo de BOOK_DISAGREEMENT
# Estos son modelos estadisticos, no casas de apuestas reales
_EXCLUDE_PROVIDERS = {"consensus", "teamrankings", "accuscore", "betegy"}

# Alias de nombres de equipo para resolver el mismatch Clippers pre-2024
# OddsData 2022-23 usa "Los Angeles Clippers"; ESPNLines usa "LA Clippers"
_TEAM_NAME_ALIASES = {
    "Los Angeles Clippers": "LA Clippers",
}


def _normalize_team(team_name: str) -> str:
    """Normaliza nombre de equipo para el join con ESPNLines."""
    return _TEAM_NAME_ALIASES.get(team_name, team_name)


def build_espn_consensus_history(espn_db_path=None):
    """Pre-construye lookup de ESPN consensus lines para todos los partidos.

    JOIN KEY: (game_date_str, home_team, away_team)
    Los nombres de equipo en ESPNLines deben coincidir con los de OddsData/create_games.
    Se aplica alias para "Los Angeles Clippers" → "LA Clippers" al leer ESPN.

    COBERTURA: ~24% de los partidos en OddsData/dataset.sqlite.
    Esperar ~76% NaN en features ESPN para filas 2022-2026.
    Filas anteriores a 2022: 100% NaN (no hay datos ESPN).

    Returns:
        dict: clave (game_date_str, home_team, away_team) →
              {ESPN_LINE_MOVE, ESPN_TOTAL_MOVE, ESPN_OPEN_ML_PROB}
              Nombres de equipo en clave coinciden con OddsData (incluyendo alias).
    """
    if espn_db_path is None:
        espn_db_path = ESPN_LINES_DB

    lookup = {}

    if not Path(espn_db_path).exists():
        logger.warning("ESPNLines.sqlite no existe en %s — ESPN consensus deshabilitado", espn_db_path)
        return lookup

    with sqlite3.connect(espn_db_path) as con:
        for season in ESPNLINES_SEASONS:
            table = f"espn_consensus_{season}"
            try:
                df = pd.read_sql_query(
                    f'SELECT game_date, home_team, away_team, '
                    f'spread_movement, total_movement, consensus_open_ml_prob '
                    f'FROM "{table}"',
                    con,
                )
            except Exception as e:
                logger.debug("ESPN consensus tabla %s no existe o error: %s", table, e)
                continue

            if df.empty:
                logger.debug("ESPN consensus tabla %s vacia", table)
                continue

            for _, row in df.iterrows():
                # Normalizar nombres de equipo para coincidir con OddsData
                # ESPNLines usa "LA Clippers"; OddsData pre-2024 usa "Los Angeles Clippers"
                # Al crear la clave con el nombre ESPN, creamos una lookup que
                # coincidira con el nombre normalizado que pasa create_games.
                # create_games pasa row.Home/row.Away de OddsData;
                # debemos normalizar EN ESE LADO (en get_game_espn_features).
                key = (row["game_date"], row["home_team"], row["away_team"])
                lookup[key] = {
                    "ESPN_LINE_MOVE": row["spread_movement"],
                    "ESPN_TOTAL_MOVE": row["total_movement"],
                    "ESPN_OPEN_ML_PROB": row["consensus_open_ml_prob"],
                }

            logger.info("ESPN consensus %s: %d entradas cargadas", season, len(df))

    logger.info("ESPN consensus total: %d entradas en lookup", len(lookup))
    return lookup


def build_espn_book_disagreement_history(espn_db_path=None):
    """Pre-construye lookup de BOOK_DISAGREEMENT (std spread entre casas de apuestas).

    Solo disponible para 2023-24 con datos reales multi-proveedor.
    Excluye proveedores que son modelos estadisticos: consensus, teamrankings,
    accuscore, betegy.

    Returns:
        dict: clave (game_date_str, home_team, away_team) →
              {ESPN_BOOK_DISAGREEMENT: float}
    """
    if espn_db_path is None:
        espn_db_path = ESPN_LINES_DB

    lookup = {}

    if not Path(espn_db_path).exists():
        logger.warning("ESPNLines.sqlite no existe — book disagreement deshabilitado")
        return lookup

    with sqlite3.connect(espn_db_path) as con:
        for season in BOOK_DISAGREEMENT_SEASONS:
            table = f"espn_lines_{season}"
            try:
                df = pd.read_sql_query(
                    f'SELECT game_date, home_team, away_team, '
                    f'provider_name, open_spread_home '
                    f'FROM "{table}" '
                    f'WHERE open_spread_home IS NOT NULL',
                    con,
                )
            except Exception as e:
                logger.debug("ESPN lines tabla %s no existe: %s", table, e)
                continue

            if df.empty:
                logger.debug("ESPN lines tabla %s vacia o sin spreads", table)
                continue

            # Excluir proveedores no-sportsbook (modelos estadisticos)
            df = df[~df["provider_name"].isin(_EXCLUDE_PROVIDERS)]

            if df.empty:
                logger.debug("ESPN lines %s: sin proveedores reales despues del filtro", season)
                continue

            # Calcular std del spread de apertura entre casas (por partido)
            book_std = (
                df.groupby(["game_date", "home_team", "away_team"])["open_spread_home"]
                .std()
                .reset_index()
            )
            book_std.columns = ["game_date", "home_team", "away_team", "ESPN_BOOK_DISAGREEMENT"]

            for _, row in book_std.iterrows():
                key = (row["game_date"], row["home_team"], row["away_team"])
                lookup[key] = {"ESPN_BOOK_DISAGREEMENT": row["ESPN_BOOK_DISAGREEMENT"]}

            logger.info(
                "ESPN book disagreement %s: %d entradas (%d partidos con datos)",
                season, len(book_std), len(book_std),
            )

    logger.info("ESPN book disagreement total: %d entradas en lookup", len(lookup))
    return lookup


def get_game_espn_features(consensus_lookup, disagreement_lookup, date_str, home_team, away_team):
    """Obtiene features de ESPNLines para un partido especifico.

    Normaliza nombres de equipo antes del lookup para resolver el mismatch
    "Los Angeles Clippers" (OddsData pre-2024) vs "LA Clippers" (ESPNLines).

    ESPNLines registra fechas de partido con +1 dia respecto a OddsData en ~70-84%
    de los partidos (diferencia de zona horaria o convencion de scraping). El lookup
    prueba la fecha exacta primero y luego date+1 como fallback.

    Retorna NaN para partidos sin cobertura ESPN (~25% de los casos en 2022+).
    XGBoost maneja NaN nativamente — no imputar con ceros.

    Args:
        consensus_lookup: dict de build_espn_consensus_history()
        disagreement_lookup: dict de build_espn_book_disagreement_history()
        date_str: fecha del partido en formato "YYYY-MM-DD"
        home_team: nombre del equipo local (como viene de OddsData/row.Home)
        away_team: nombre del equipo visitante (como viene de OddsData/row.Away)

    Returns:
        dict con 4 features (ESPN_LINE_MOVE, ESPN_TOTAL_MOVE, ESPN_OPEN_ML_PROB,
        ESPN_BOOK_DISAGREEMENT). NaN para partidos sin cobertura.
    """
    # Normalizar nombres para resolver alias de equipo
    home_normalized = _normalize_team(home_team)
    away_normalized = _normalize_team(away_team)

    # ESPNLines almacena fechas con +1 dia vs OddsData en ~75% de partidos.
    # Probamos fecha exacta primero, luego date+1 como fallback.
    # Esto sube el match rate de ~22% a ~95%+ en temporadas 2022+.
    key = (date_str, home_normalized, away_normalized)
    try:
        next_date_str = (_date.fromisoformat(date_str) + timedelta(days=1)).isoformat()
    except ValueError:
        next_date_str = None
    key_plus1 = (next_date_str, home_normalized, away_normalized) if next_date_str else None

    nan = float("nan")

    # Buscar en consensus: fecha exacta → date+1
    consensus_feats = consensus_lookup.get(key)
    if consensus_feats is None and key_plus1 is not None:
        consensus_feats = consensus_lookup.get(key_plus1)
    if consensus_feats is None:
        consensus_feats = {
            "ESPN_LINE_MOVE": nan,
            "ESPN_TOTAL_MOVE": nan,
            "ESPN_OPEN_ML_PROB": nan,
        }

    # Buscar en disagreement: fecha exacta → date+1
    disagree_feats = disagreement_lookup.get(key)
    if disagree_feats is None and key_plus1 is not None:
        disagree_feats = disagreement_lookup.get(key_plus1)
    if disagree_feats is None:
        disagree_feats = {"ESPN_BOOK_DISAGREEMENT": nan}

    return {**consensus_feats, **disagree_feats}


def add_espn_features_to_frame(frame, espn_features_list):
    """Agrega columnas ESPN al DataFrame de juegos.

    Mismo patron que add_four_factors_to_frame en bref_game_features.py.

    Args:
        frame: DataFrame principal de create_games.py
        espn_features_list: lista de dicts de get_game_espn_features()

    Returns:
        DataFrame con columnas ESPN_ agregadas.
    """
    if not espn_features_list:
        logger.warning("espn_features_list vacia — no se agregan columnas ESPN")
        return frame

    espn_df = pd.DataFrame(espn_features_list, index=frame.index)
    return pd.concat([frame, espn_df], axis=1)
