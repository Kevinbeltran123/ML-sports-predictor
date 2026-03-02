"""
Recoleccion de game logs de jugadores NBA via nba_api.

Patron ETL (Extract-Transform-Load):
    Extract:   Llama a stats.nba.com via nba_api (1 llamada por temporada)
    Transform: Parsea MATCHUP para determinar home/away, limpia minutos
    Load:      Guarda en Data/PlayerGameLogs.sqlite (1 tabla por temporada)

Usado por InjuryImpact.py para calcular el Weighted Availability Index:
    Para saber si un jugador jugo en un partido especifico, consultamos
    la tabla correspondiente y buscamos si aparece con MIN > 0.

Ejecucion:
    PYTHONPATH=/path/to/project python -u src/Process-Data/Collect_Player_Logs.py

    Flags opcionales:
        --force    Re-descarga temporadas ya cacheadas
        --season   Descarga solo una temporada (ej: --season 2024-25)
"""

import argparse
import sqlite3
import time
from pathlib import Path

import pandas as pd

# nba_api: wrapper para stats.nba.com que maneja headers, rate limiting, etc.
from nba_api.stats.endpoints import LeagueGameLog

from src.config import PLAYER_LOGS_DB as DB_PATH, CONFIG_PATH, get_logger

logger = get_logger(__name__)

# Temporadas que necesitamos para el dataset (2012-13 en adelante).
# IMPORTANTE: incluir SIEMPRE la temporada actual para que InjuryImpact.py
# pueda calcular la rotación real (si falta, todos los equipos caen al fallback AVAIL=1.0).
SEASONS = [
    "2012-13", "2013-14", "2014-15", "2015-16", "2016-17",
    "2017-18", "2018-19", "2019-20", "2020-21", "2021-22",
    "2022-23", "2023-24", "2024-25", "2025-26",
]

# Columnas que guardamos.
# MIN: usada por InjuryImpact.py (weighted availability index).
# PTS/REB/AST/...: usadas por PlayerPropsFeatures.py (predicción de props).
# Backward compatibility: tablas antiguas (7 cols) siguen funcionando con InjuryImpact.
KEEP_COLS = [
    "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION",
    "GAME_DATE", "MATCHUP",
    "MIN",          # Minutos jugados
    "PTS",          # Puntos
    "REB", "OREB", "DREB",  # Rebotes (totales, ofensivos, defensivos)
    "AST",          # Asistencias
    "STL", "BLK",   # Robos y bloqueos
    "TOV",          # Pérdidas de balón
    "FGM", "FGA",   # Intentos y canastas de campo
    "FG3M", "FG3A", # Intentos y canastas de 3 puntos
    "FTM", "FTA",   # Intentos y tiros libres
    "PLUS_MINUS",   # Diferencial de puntos mientras jugó
]


def table_name_for_season(season):
    """Nombre de tabla en SQLite para una temporada (ej: 'player_logs_2024-25')."""
    return f"player_logs_{season}"


def season_exists(con, season):
    """Verifica si ya tenemos datos cacheados para esta temporada."""
    tbl = table_name_for_season(season)
    cursor = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (tbl,)
    )
    if cursor.fetchone() is None:
        return False
    # Verificar que tiene filas (no solo la tabla vacia)
    count = con.execute(f'SELECT COUNT(*) FROM "{tbl}"').fetchone()[0]
    return count > 0


def parse_is_home(matchup):
    """Determina si el equipo jugo de local basado en el campo MATCHUP.

    nba_api retorna MATCHUP en dos formatos:
        "BOS vs. NYK"  -> BOS es HOME (contiene 'vs.')
        "BOS @ NYK"    -> BOS es AWAY (contiene '@')

    Returns:
        1 si home, 0 si away
    """
    if "vs." in str(matchup):
        return 1
    return 0


def parse_minutes(min_val):
    """Convierte el campo MIN a float.

    nba_api retorna MIN como string 'MM:SS' o como None.
    Algunos registros tienen MIN=0 (DNP con cero minutos).
    """
    if min_val is None or min_val == "" or min_val == "None":
        return 0.0
    # Puede venir como "34:12" (minutos:segundos) o como float directo
    s = str(min_val)
    if ":" in s:
        parts = s.split(":")
        try:
            return float(parts[0]) + float(parts[1]) / 60.0
        except (ValueError, IndexError):
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def download_season(season, max_retries=3):
    """Descarga todos los game logs de jugadores para una temporada.

    Usa LeagueGameLog con player_or_team_abbreviation='P' que retorna
    TODOS los jugadores que jugaron en la temporada en una sola llamada.
    ~25,000-40,000 filas por temporada.

    Args:
        season: string en formato nba_api (ej: '2024-25')
        max_retries: intentos maximos si falla la API

    Returns:
        DataFrame con columnas KEEP_COLS + IS_HOME, o None si falla
    """
    for attempt in range(max_retries):
        try:
            # LeagueGameLog: endpoint principal de stats.nba.com
            # player_or_team_abbreviation='P' = game logs de jugadores (no equipos)
            # season_type_all_star='Regular Season' = solo temporada regular
            log = LeagueGameLog(
                season=season,
                player_or_team_abbreviation="P",
                season_type_all_star="Regular Season",
                timeout=180,
                headers={
                    "Host": "stats.nba.com",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "application/json, text/plain, */*",
                    "Accept-Language": "en-US,en;q=0.9",
                    "x-nba-stats-origin": "stats",
                    "x-nba-stats-token": "true",
                    "Referer": "https://www.nba.com/",
                    "Origin": "https://www.nba.com",
                    "Connection": "keep-alive",
                },
            )
            df = log.get_data_frames()[0]

            if df.empty:
                logger.warning("Sin datos para %s", season)
                return None

            # Seleccionar solo columnas necesarias
            df = df[KEEP_COLS].copy()

            # Transformar: parsear home/away desde MATCHUP
            df["IS_HOME"] = df["MATCHUP"].apply(parse_is_home)

            # Transformar: parsear minutos a float
            df["MIN"] = df["MIN"].apply(parse_minutes)

            return df

        except Exception as e:
            wait_time = 30 * (attempt + 1)
            logger.error("Error en intento %d/%d: %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                logger.info("Reintentando en %ds...", wait_time)
                time.sleep(wait_time)
            else:
                logger.error("No se pudo descargar %s despues de %d intentos", season, max_retries)
                return None


def _get_current_season() -> str:
    """Lee la última temporada definida en config.toml (sección get-data).

    Esto evita hardcodear '2025-26' — cuando empiece una nueva temporada,
    solo hay que agregar la entrada en config.toml.
    """
    import toml
    config = toml.load(CONFIG_PATH)
    seasons = list(config.get("get-data", {}).keys())
    if not seasons:
        raise RuntimeError("No hay temporadas en config.toml [get-data]")
    return seasons[-1]  # última = temporada actual


def update_current_season(season: str = None, force: bool = True) -> int:
    """Actualiza los game logs de la temporada actual (o la indicada).

    Función importable para el scheduler diario. Equivalente a:
      python collect_player_logs.py --season 2025-26 --force

    ¿Por qué force=True por defecto?
      La temporada en curso crece cada día con partidos nuevos.
      Siempre queremos reemplazar la tabla con los datos más frescos.

    Args:
        season: temporada a actualizar. None = lee de config.toml (última).
        force: True = reemplaza tabla existente (siempre True para temp. actual).

    Returns:
        Número de filas guardadas.

    Raises:
        RuntimeError: si no se pudo descargar la temporada.
    """
    if season is None:
        season = _get_current_season()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as con:
        if not force and season_exists(con, season):
            count = con.execute(
                f'SELECT COUNT(*) FROM "{table_name_for_season(season)}"'
            ).fetchone()[0]
            logger.info("%s: Ya cacheada (%s filas). Saltando.", season, f"{count:,}")
            return count

        logger.info("%s: Descargando (force=%s)...", season, force)
        df = download_season(season)
        if df is None:
            raise RuntimeError(f"No se pudo descargar temporada {season}")

        tbl = table_name_for_season(season)
        df.to_sql(tbl, con, if_exists="replace", index=False)
        logger.info("%s: %s filas guardadas", season, f"{len(df):,}")
        return len(df)


def main():
    parser = argparse.ArgumentParser(description="Descargar game logs de jugadores NBA")
    parser.add_argument("--force", action="store_true", help="Re-descargar temporadas ya cacheadas")
    parser.add_argument("--season", type=str, help="Descargar solo una temporada (ej: 2024-25)")
    args = parser.parse_args()

    seasons_to_download = [args.season] if args.season else SEASONS

    # Crear directorio Data si no existe
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as con:
        for i, season in enumerate(seasons_to_download):
            # Verificar si ya esta cacheada
            if not args.force and season_exists(con, season):
                count = con.execute(
                    f'SELECT COUNT(*) FROM "{table_name_for_season(season)}"'
                ).fetchone()[0]
                logger.info("%s: Ya cacheada (%s filas). Saltando.", season, f"{count:,}")
                continue

            logger.info("%s: Descargando... (%d/%d)", season, i + 1, len(seasons_to_download))

            df = download_season(season)
            if df is None:
                continue

            # Load: guardar en SQLite
            tbl = table_name_for_season(season)
            df.to_sql(tbl, con, if_exists="replace", index=False)
            logger.info("%s: %s filas guardadas en %s", season, f"{len(df):,}", tbl)

            # Rate limiting: esperar entre llamadas para no saturar stats.nba.com
            # 3 segundos es suficiente para evitar 429 (Too Many Requests)
            if i < len(seasons_to_download) - 1:
                logger.debug("Esperando 3s (rate limit)...")
                time.sleep(3)

    logger.info("Datos guardados en %s", DB_PATH)

    # Resumen final
    with sqlite3.connect(DB_PATH) as con:
        tables = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        total = 0
        for (tbl,) in tables:
            count = con.execute(f'SELECT COUNT(*) FROM "{tbl}"').fetchone()[0]
            total += count
            logger.debug("%s: %s filas", tbl, f"{count:,}")
        logger.info("TOTAL: %s filas", f"{total:,}")


if __name__ == "__main__":
    main()
