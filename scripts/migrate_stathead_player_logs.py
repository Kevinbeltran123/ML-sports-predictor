"""Migra player_game_finder_{season} de StatheadData.sqlite → PlayerGameLogs.sqlite.

Stathead Player Game Finder tiene los mismos datos que el NBA Stats API
(player logs por partido) pero con nombres de columnas distintos.

Mapeo de columnas:
    Stathead              → PlayerGameLogs
    player_link           → PLAYER_ID   (hash del link, consistente entre partidos)
    player_name           → PLAYER_NAME
    team_id               → TEAM_ABBREVIATION
    date_game             → GAME_DATE
    opp_id + is_away      → MATCHUP     (ej: "BOS vs. MIA" o "BOS @ MIA")
    mp_dec                → MIN
    pts                   → PTS
    trb                   → REB
    orb                   → OREB
    drb                   → DREB
    ast                   → AST
    stl                   → STL
    blk                   → BLK
    tov                   → TOV
    fg                    → FGM
    fga                   → FGA
    fg3                   → FG3M
    fg3a                  → FG3A
    ft                    → FTM
    fta                   → FTA
    plus_minus            → PLUS_MINUS
    1 - is_away           → IS_HOME

Uso:
    PYTHONPATH=. python scripts/migrate_stathead_player_logs.py
    PYTHONPATH=. python scripts/migrate_stathead_player_logs.py --season 2024-25
    PYTHONPATH=. python scripts/migrate_stathead_player_logs.py --season 2025-26 --force
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

from src.config import STATHEAD_DB, PLAYER_LOGS_DB, get_logger

logger = get_logger(__name__)

SEASONS = [
    "2012-13", "2013-14", "2014-15", "2015-16", "2016-17",
    "2017-18", "2018-19", "2019-20", "2020-21", "2021-22",
    "2022-23", "2023-24", "2024-25", "2025-26",
]


def _player_link_to_id(link: str) -> int:
    """Convierte '/players/b/birchbi01.html' → ID numérico estable via hash."""
    if not link:
        return 0
    # Extraer slug: 'birchbi01'
    slug = link.rstrip("/").split("/")[-1].replace(".html", "")
    # Hash positivo de 9 dígitos — estable entre runs
    return abs(hash(slug)) % 900_000_000 + 100_000_000


def migrate_season(src_con: sqlite3.Connection, dst_con: sqlite3.Connection,
                   season: str, force: bool = False) -> int:
    """Migra una temporada. Retorna filas insertadas."""
    src_table = f"player_game_finder_{season}"
    dst_table = f"player_logs_{season}"

    # Verificar tabla origen
    exists = src_con.execute(
        f"SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (src_table,)
    ).fetchone()
    if not exists:
        logger.warning("[%s] Tabla '%s' no existe en StatheadData.sqlite. "
                       "Corre scrape_stathead.py --phase player_game_finder --season %s primero.",
                       season, src_table, season)
        return 0

    src_count = src_con.execute(f'SELECT COUNT(*) FROM "{src_table}"').fetchone()[0]
    if src_count == 0:
        logger.warning("[%s] Tabla origen vacía.", season)
        return 0

    # Verificar si ya migrado
    dst_exists = dst_con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (dst_table,)
    ).fetchone()
    if dst_exists and not force:
        dst_count = dst_con.execute(f'SELECT COUNT(*) FROM "{dst_table}"').fetchone()[0]
        # Si ya tiene datos comparables (>90% del origen), skip
        if dst_count >= int(src_count * 0.9):
            logger.info("[%s] Ya migrado (%d filas). --force para re-migrar.", season, dst_count)
            return 0
        logger.info("[%s] Destino tiene %d/%d filas — actualizando.", season, dst_count, src_count)

    # Leer datos de Stathead
    df = pd.read_sql(f'SELECT * FROM "{src_table}"', src_con)
    logger.info("[%s] %d filas leídas de Stathead.", season, len(df))

    # Mapear columnas → formato PlayerGameLogs
    out = pd.DataFrame()
    out["PLAYER_ID"]          = df["player_link"].apply(_player_link_to_id)
    out["PLAYER_NAME"]        = df["player_name"]
    out["TEAM_ABBREVIATION"]  = df["team_id"]
    out["GAME_DATE"]          = df["date_game"]

    # MATCHUP: "TEAM vs. OPP" (home) o "TEAM @ OPP" (away)
    def _matchup(row):
        if row["is_away"]:
            return f"{row['team_id']} @ {row['opp_id']}"
        return f"{row['team_id']} vs. {row['opp_id']}"
    out["MATCHUP"]   = df.apply(_matchup, axis=1)

    out["MIN"]        = pd.to_numeric(df["mp_dec"],    errors="coerce").fillna(0.0)
    out["PTS"]        = pd.to_numeric(df["pts"],       errors="coerce").fillna(0).astype(int)
    out["REB"]        = pd.to_numeric(df["trb"],       errors="coerce").fillna(0).astype(int)
    out["OREB"]       = pd.to_numeric(df["orb"],       errors="coerce").fillna(0).astype(int)
    out["DREB"]       = pd.to_numeric(df["drb"],       errors="coerce").fillna(0).astype(int)
    out["AST"]        = pd.to_numeric(df["ast"],       errors="coerce").fillna(0).astype(int)
    out["STL"]        = pd.to_numeric(df["stl"],       errors="coerce").fillna(0).astype(int)
    out["BLK"]        = pd.to_numeric(df["blk"],       errors="coerce").fillna(0).astype(int)
    out["TOV"]        = pd.to_numeric(df["tov"],       errors="coerce").fillna(0).astype(int)
    out["FGM"]        = pd.to_numeric(df["fg"],        errors="coerce").fillna(0).astype(int)
    out["FGA"]        = pd.to_numeric(df["fga"],       errors="coerce").fillna(0).astype(int)
    out["FG3M"]       = pd.to_numeric(df["fg3"],       errors="coerce").fillna(0).astype(int)
    out["FG3A"]       = pd.to_numeric(df["fg3a"],      errors="coerce").fillna(0).astype(int)
    out["FTM"]        = pd.to_numeric(df["ft"],        errors="coerce").fillna(0).astype(int)
    out["FTA"]        = pd.to_numeric(df["fta"],       errors="coerce").fillna(0).astype(int)
    out["PLUS_MINUS"] = pd.to_numeric(df["plus_minus"], errors="coerce").fillna(0).astype(int)
    out["IS_HOME"]    = (1 - df["is_away"].fillna(0)).astype(int)

    # Escribir en destino (replace para fuerza, append si parcial)
    out.to_sql(dst_table, dst_con, if_exists="replace", index=False)
    dst_con.commit()

    inserted = dst_con.execute(f'SELECT COUNT(*) FROM "{dst_table}"').fetchone()[0]
    max_date = dst_con.execute(f'SELECT MAX(GAME_DATE) FROM "{dst_table}"').fetchone()[0]
    logger.info("[%s] ✓ Migrados %d filas → '%s' (max_date=%s)", season, inserted, dst_table, max_date)
    return inserted


def main():
    ap = argparse.ArgumentParser(description="Migra Stathead PGF → PlayerGameLogs.sqlite")
    ap.add_argument("--season", default=None, help="Ej: 2025-26. Default: todas.")
    ap.add_argument("--force", action="store_true", help="Re-migrar aunque ya exista.")
    args = ap.parse_args()

    seasons = [args.season] if args.season else SEASONS

    if not STATHEAD_DB.exists():
        logger.error("StatheadData.sqlite no encontrado en %s", STATHEAD_DB)
        logger.error("Corre: PYTHONPATH=. python scripts/scrape_stathead.py --phase player_game_finder")
        sys.exit(1)

    src_con = sqlite3.connect(str(STATHEAD_DB))
    dst_con = sqlite3.connect(str(PLAYER_LOGS_DB))

    total = 0
    for season in seasons:
        total += migrate_season(src_con, dst_con, season, force=args.force)

    src_con.close()

    # Resumen final
    cur = dst_con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'player_logs_%'")
    tables = [r[0] for r in cur.fetchall()]
    grand_total = sum(
        dst_con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in tables
    )
    dst_con.close()

    logger.info("=" * 50)
    logger.info("Migración completada: +%d filas insertadas", total)
    logger.info("PlayerGameLogs.sqlite total: %d filas (%d tablas)", grand_total, len(tables))


if __name__ == "__main__":
    main()
