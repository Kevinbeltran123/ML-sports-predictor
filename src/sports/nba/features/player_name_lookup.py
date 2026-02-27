"""Lookup de nombres BRef -> nba_api PLAYER_ID.

Por que esto es necesario:
  - Archetypes (GMM) se entrenan con PLAYER_ID de nba_api
  - player_basic (BRef) usa nombres como "LeBron James", "Stephen Curry"
  - Para asignar archetypes a starters de BRef, necesitamos el mapping

Estrategia:
  - Unir por nombre normalizado (quitar acentos, Jr./Sr., lower)
  - Deduplicar por (nombre_normalizado, temporada) -> primer PLAYER_ID
  - Cache persistente en data/nba/research/player_name_lookup.pkl
"""

import sqlite3
import unicodedata
import re
from pathlib import Path

import pandas as pd
import joblib

from src.config import RESEARCH_DIR, get_logger

logger = get_logger(__name__)

BREF_DB = Path("data/BRefData.sqlite")
PLAYER_LOGS_DB = Path("data/nba/training/PlayerGameLogs.sqlite")
LOOKUP_CACHE = RESEARCH_DIR / "player_name_lookup.pkl"


def _normalize_name(name: str) -> str:
    """Normalizar nombre para matching."""
    if not isinstance(name, str):
        return ""
    # Quitar acentos (Sengun -> Sengun, Doncic -> Doncic)
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    # Quitar sufijos comunes
    name = re.sub(r"\s+(Jr\.|Sr\.|III|II|IV)\s*$", "", name, flags=re.IGNORECASE)
    # Lower y strip
    return name.strip().lower()


def build_player_name_lookup(
    seasons: list[str] | None = None,
    force_rebuild: bool = False,
) -> dict:
    """Construir mapping {(bref_name, season) -> player_id}.

    Para cada temporada, cargar player_basic (BRef) y player_logs (nba_api),
    unir por nombre normalizado. Muchos nombres son identicos entre fuentes;
    los que difieren (acentos, Jr.) se resuelven por normalizacion.

    Args:
        seasons: lista de temporadas a procesar. None = todas 2014-2026.
        force_rebuild: si True, ignora cache existente.

    Returns:
        dict: {(bref_player_name, season_str) -> nba_api_player_id}
    """
    if not force_rebuild and LOOKUP_CACHE.exists():
        logger.info("  Cargando lookup cacheado: %s", LOOKUP_CACHE)
        return joblib.load(LOOKUP_CACHE)

    if seasons is None:
        seasons = [f"{y}-{str(y+1)[-2:]}" for y in range(2014, 2026)]

    lookup = {}

    for season in seasons:
        # --- Cargar BRef player_basic ---
        try:
            bref_con = sqlite3.connect(str(BREF_DB))
            bref_df = pd.read_sql(
                f'SELECT DISTINCT player_name FROM "player_basic_{season}"',
                bref_con,
            )
            bref_con.close()
        except Exception:
            continue

        # --- Cargar nba_api player_logs ---
        try:
            logs_con = sqlite3.connect(str(PLAYER_LOGS_DB))
            logs_df = pd.read_sql(
                f'SELECT DISTINCT PLAYER_ID, PLAYER_NAME '
                f'FROM "player_logs_{season}"',
                logs_con,
            )
            logs_con.close()
        except Exception:
            continue

        # Normalizar nombres en ambas fuentes
        bref_df["norm_name"] = bref_df["player_name"].apply(_normalize_name)
        logs_df["norm_name"] = logs_df["PLAYER_NAME"].apply(_normalize_name)

        # Dedup: tomar primer PLAYER_ID por nombre normalizado
        id_map = (
            logs_df.drop_duplicates("norm_name")
            .set_index("norm_name")["PLAYER_ID"]
            .to_dict()
        )

        # Asignar: para cada nombre BRef unico, buscar en id_map
        matched = 0
        for _, row in bref_df.iterrows():
            pid = id_map.get(row["norm_name"])
            if pid is not None:
                lookup[(row["player_name"], season)] = int(pid)
                matched += 1

        logger.debug("  %s: %d/%d nombres matched", season, matched, len(bref_df))

    # Guardar cache
    LOOKUP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(lookup, LOOKUP_CACHE)
    logger.info("  Lookup construido: %d mappings -> %s", len(lookup), LOOKUP_CACHE)

    return lookup
