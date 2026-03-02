"""Scrapea Basketball Reference para datos historicos NBA 2014-15 a 2025-26.

Se conecta a tu Chrome real via CDP (remote debugging) para evitar deteccion.
Requiere tener Chrome abierto con debugging habilitado y logueado en BRef/Stathead.

Prerequisito (una sola vez):
    # Cerrar Chrome completamente (Cmd+Q), luego en terminal:
    pkill -9 -f "Google Chrome"
    sleep 2
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \\
        --remote-debugging-port=9222 \\
        --remote-allow-origins="*" \\
        --user-data-dir="/tmp/chrome-debug-profile" \\
        --no-first-run
    # En el Chrome que se abre, loguear en stathead.com

Fases:
  --phase schedules     → recolecta calendarios, genera lista de juegos
  --phase boxscores     → recolecta box scores (four factors, line scores, player stats)
  --phase plus-minus    → on/off court +/- por jugador por cuarto
  --phase pbp           → play-by-play (cada posesion)
  --phase shot-chart    → ubicacion X,Y de cada tiro (make/miss)
  --phase team-shooting → shooting % por distancia (team season pages)
  --phase subpages      → alias: plus-minus + pbp + shot-chart
  --phase all           → todas las fases en orden

Uso:
    source .env && PYTHONPATH=. python scripts/scrape_basketball_reference.py --phase schedules
    source .env && PYTHONPATH=. python scripts/scrape_basketball_reference.py --phase schedules --season 2025-26
    source .env && PYTHONPATH=. python scripts/scrape_basketball_reference.py --phase boxscores --season 2024-25 --max-games 5
    source .env && PYTHONPATH=. python scripts/scrape_basketball_reference.py --phase boxscores --force
    source .env && PYTHONPATH=. python scripts/scrape_basketball_reference.py --phase plus-minus --season 2024-25 --max-games 2
    source .env && PYTHONPATH=. python scripts/scrape_basketball_reference.py --phase team-shooting
    source .env && PYTHONPATH=. python scripts/scrape_basketball_reference.py --phase subpages
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pandas as pd
from patchright.sync_api import sync_playwright, Page, BrowserContext

from src.config import BREF_DB, BREF_SESSION_DIR, get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------ #
# Caffeinate — previene sleep de macOS durante scraping largo
# ------------------------------------------------------------------ #
_caffeinate_proc: subprocess.Popen | None = None


def _start_caffeinate():
    """Inicia caffeinate para mantener Mac despierto. -dims = display+idle+system+disk."""
    global _caffeinate_proc
    if sys.platform != "darwin":
        return
    try:
        _caffeinate_proc = subprocess.Popen(
            ["caffeinate", "-dims"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        logger.info("caffeinate iniciado (PID %d) — Mac no dormirá", _caffeinate_proc.pid)
    except FileNotFoundError:
        logger.warning("caffeinate no encontrado (¿no es macOS?)")


def _stop_caffeinate():
    """Detiene caffeinate si esta corriendo."""
    global _caffeinate_proc
    if _caffeinate_proc is not None:
        try:
            _caffeinate_proc.terminate()
            _caffeinate_proc.wait(timeout=5)
            logger.info("caffeinate detenido")
        except Exception:
            try:
                _caffeinate_proc.kill()
            except Exception:
                pass
        _caffeinate_proc = None


# ------------------------------------------------------------------ #
# Graceful shutdown — SIGINT/SIGTERM
# ------------------------------------------------------------------ #
_shutdown_requested = False
_active_browser: "BRefBrowser | None" = None
_active_connection: sqlite3.Connection | None = None


def _signal_handler(signum, frame):
    """Maneja Ctrl+C y SIGTERM: marca shutdown, no mata inmediatamente."""
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    if _shutdown_requested:
        # Segunda señal → salir de verdad
        logger.warning("Segunda señal (%s), forzando salida...", sig_name)
        _cleanup()
        sys.exit(1)
    _shutdown_requested = True
    logger.warning(
        "Señal %s recibida. Terminando después del juego actual... "
        "(Ctrl+C de nuevo para forzar)", sig_name)


def _cleanup():
    """Limpieza final: cerrar browser, commit DB, matar caffeinate."""
    global _active_browser, _active_connection
    if _active_browser is not None:
        try:
            _active_browser.close()
        except Exception:
            pass
        _active_browser = None
    if _active_connection is not None:
        try:
            _active_connection.commit()
            _active_connection.close()
        except Exception:
            pass
        _active_connection = None
    _stop_caffeinate()


def should_stop() -> bool:
    """Retorna True si se pidio shutdown."""
    return _shutdown_requested


# ------------------------------------------------------------------ #
# Constantes
# ------------------------------------------------------------------ #
BASE_URL = "https://www.basketball-reference.com"
SESSION_DIR = BREF_SESSION_DIR  # cookies persistentes

REQUEST_DELAY = 3.0       # segundos entre requests (premium ~20 req/min)
MAX_RETRIES = 3
RETRY_DELAY = 5            # backoff lineal: 5s, 10s, 15s

ALL_SEASONS = [
    "2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24",
    "2024-25", "2025-26",
]

SEASON_MONTHS = [
    "october", "november", "december", "january",
    "february", "march", "april", "may", "june",
]

BREF_TEAM_CODES = [
    "ATL", "BOS", "BRK", "CHO", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHO", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
]

# Aliases historicos (Charlotte Bobcats → Hornets, etc.)
BREF_TEAM_ALIASES: dict[str, dict[str, str]] = {
    # season → {old_code: new_code}
    "2014-15": {"CHA": "CHO"},
}


def season_to_bref_year(season: str) -> int:
    """Convierte '2014-15' → 2015 (el year que usa BRef en URLs)."""
    return int(season.split("-")[0]) + 1


# ------------------------------------------------------------------ #
# Browser management — conecta a Chrome real via CDP
# ------------------------------------------------------------------ #
CDP_URL = "http://localhost:9222"


class BRefBrowser:
    """Se conecta a un Chrome real ya abierto via CDP (remote debugging).

    Prerrequisito: lanzar Chrome con --remote-debugging-port=9222
    y estar logueado en Stathead/BRef manualmente.

    Esto evita toda deteccion porque es literalmente tu Chrome real.
    """

    def __init__(self):
        self._pw = sync_playwright().start()
        self._browser = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    def start(self) -> Page:
        """Conecta al Chrome real via CDP."""
        try:
            self._browser = self._pw.chromium.connect_over_cdp(CDP_URL)
        except Exception as e:
            logger.error(
                "No se pudo conectar a Chrome en %s.\n"
                "Asegurate de lanzar Chrome asi:\n"
                "  /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome "
                "--remote-debugging-port=9222\n"
                "Error: %s", CDP_URL, e)
            raise SystemExit(1)

        # Usar el primer contexto (perfil por defecto del Chrome)
        contexts = self._browser.contexts
        if contexts:
            self._context = contexts[0]
        else:
            self._context = self._browser.new_context()

        # Abrir nueva tab para el scraping (no tocar las tabs del usuario)
        self._page = self._context.new_page()
        logger.info("Conectado a Chrome via CDP (%s)", CDP_URL)
        return self._page

    def fetch_html(self, url: str) -> str | None:
        """Descarga HTML usando Chrome real. Respeta rate limiting."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self._page.goto(url, wait_until="domcontentloaded", timeout=30000)
                if resp and resp.status == 429:
                    wait = RETRY_DELAY * attempt * 2
                    logger.warning("Rate limited (429). Esperando %ds...", wait)
                    time.sleep(wait)
                    continue
                if resp and resp.status >= 400:
                    if attempt < MAX_RETRIES:
                        wait = RETRY_DELAY * attempt
                        logger.warning("HTTP %d en %s. Reintento en %ds...",
                                       resp.status, url, wait)
                        time.sleep(wait)
                        continue
                    logger.warning("HTTP %d tras %d intentos: %s", resp.status, MAX_RETRIES, url)
                    return None

                html = self._page.content()
                time.sleep(REQUEST_DELAY)
                return html

            except Exception as e:
                if attempt < MAX_RETRIES:
                    wait = RETRY_DELAY * attempt
                    logger.warning("Error %d/%d: %s. Reintento en %ds...",
                                   attempt, MAX_RETRIES, e, wait)
                    time.sleep(wait)
                else:
                    logger.warning("Fallaron %d intentos: %s", MAX_RETRIES, e)
                    return None
        return None

    def close(self):
        """Cierra la tab de scraping. NO cierra Chrome (el usuario lo controla)."""
        if self._page:
            try:
                self._page.close()
            except Exception:
                pass
        # NO cerrar browser ni context — es el Chrome del usuario
        self._pw.stop()


# ------------------------------------------------------------------ #
# HTML Parsers
# ------------------------------------------------------------------ #
class CommentExtractor(HTMLParser):
    """Extrae bloques <!-- ... --> del HTML.

    BRef esconde tablas de stats en comentarios para lazy-loading.
    """
    def __init__(self):
        super().__init__()
        self.comments: list[str] = []

    def handle_comment(self, data: str):
        self.comments.append(data)


def extract_commented_html(html: str, table_id: str) -> str | None:
    """Busca en comentarios HTML el bloque que contiene table_id."""
    extractor = CommentExtractor()
    extractor.feed(html)
    for comment in extractor.comments:
        if f'id="{table_id}"' in comment:
            return comment
    return None


class BRefTableParser(HTMLParser):
    """Parser generico para tablas HTML de BRef.

    Extrae headers (thead) y filas de datos (tbody).
    Usa data-stat como column ID (BRef lo pone en cada celda).
    """
    def __init__(self, target_id: str | None = None):
        super().__init__()
        self.target_id = target_id
        self._in_target = target_id is None
        self._in_thead = False
        self._in_tbody = False
        self._in_tfoot = False
        self._in_cell = False
        self._cell_data: list[str] = []
        self._cell_attrs: dict = {}
        self._current_row: list[dict] = []

        self.headers: list[str] = []
        self.rows: list[list[dict]] = []
        self._header_done = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table":
            if self.target_id is None or a.get("id") == self.target_id:
                self._in_target = True
            return
        if not self._in_target:
            return
        if tag == "thead":
            self._in_thead = True
        elif tag == "tbody":
            self._in_thead = False
            self._in_tbody = True
        elif tag == "tfoot":
            self._in_tfoot = True
        elif tag == "tr":
            self._current_row = []
        elif tag in ("th", "td") and (self._in_thead or self._in_tbody):
            self._in_cell = True
            self._cell_data = []
            self._cell_attrs = a
        elif tag == "a" and self._in_cell:
            # Capturar href para links de jugadores/equipos
            pass

    def handle_endtag(self, tag):
        if not self._in_target:
            return
        if tag == "table":
            self._in_target = False
        elif tag == "thead":
            self._in_thead = False
            self._header_done = True
        elif tag == "tbody":
            self._in_tbody = False
        elif tag == "tfoot":
            self._in_tfoot = False
        elif tag == "tr":
            if self._current_row and self._in_tbody:
                self.rows.append(self._current_row)
            self._current_row = []
        elif tag in ("th", "td") and self._in_cell:
            text = "".join(self._cell_data).strip()
            ds = self._cell_attrs.get("data-stat", "")
            if self._in_thead and not self._header_done:
                self.headers.append(ds)
            elif self._in_tbody:
                self._current_row.append({"text": text, "data_stat": ds})
            self._in_cell = False

    def handle_data(self, data):
        if self._in_cell:
            self._cell_data.append(data)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame()
        records = []
        for row in self.rows:
            record = {}
            for cell in row:
                if cell["data_stat"]:
                    record[cell["data_stat"]] = cell["text"]
            if record:
                records.append(record)
        return pd.DataFrame(records)


# ------------------------------------------------------------------ #
# Numeric helpers
# ------------------------------------------------------------------ #
def _pf(val: str) -> float | None:
    """Parse float, None si vacio o NaN."""
    if not val or val in ("", "\u2014", "-"):
        return None
    try:
        f = float(val)
        import math
        return None if math.isnan(f) else f
    except (ValueError, TypeError):
        return None


def _pi(val: str) -> int | None:
    f = _pf(val)
    if f is None:
        return None
    import math
    if math.isnan(f):
        return None
    return int(f)


# ------------------------------------------------------------------ #
# PHASE 1: SCHEDULES
# ------------------------------------------------------------------ #
class ScheduleParser(HTMLParser):
    """Parsea tabla de calendario de BRef (id='schedule')."""
    def __init__(self):
        super().__init__()
        self._in_table = False
        self._in_tbody = False
        self._in_row = False
        self._in_cell = False
        self._current_stat: str | None = None
        self._cell_text: list[str] = []
        self._current_row: dict = {}
        self._current_href: str | None = None
        self.games: list[dict] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table" and a.get("id") == "schedule":
            self._in_table = True
        if not self._in_table:
            return
        if tag == "tbody":
            self._in_tbody = True
        elif tag == "tr" and self._in_tbody:
            self._in_row = True
            self._current_row = {}
        elif tag in ("td", "th") and self._in_row:
            self._in_cell = True
            self._current_stat = a.get("data-stat")
            self._cell_text = []
            self._current_href = None
        elif tag == "a" and self._in_cell:
            self._current_href = a.get("href")

    def handle_endtag(self, tag):
        if not self._in_table:
            return
        if tag == "table":
            self._in_table = False
        elif tag == "tbody":
            self._in_tbody = False
        elif tag == "tr" and self._in_row:
            if self._current_row.get("date_game"):
                self.games.append(self._current_row)
            self._in_row = False
        elif tag in ("td", "th") and self._in_cell:
            text = "".join(self._cell_text).strip()
            if self._current_stat:
                self._current_row[self._current_stat] = text
            if self._current_stat == "box_score_text" and self._current_href:
                self._current_row["boxscore_url"] = self._current_href
            self._in_cell = False

    def handle_data(self, data):
        if self._in_cell:
            self._cell_text.append(data)


def scrape_season_schedule(season: str, browser: BRefBrowser) -> list[dict]:
    """Recolecta calendario de una temporada."""
    bref_year = season_to_bref_year(season)
    all_games = []

    for month in SEASON_MONTHS:
        url = f"{BASE_URL}/leagues/NBA_{bref_year}_games-{month}.html"
        logger.info("[%s] Calendario: %s", season, month)

        html = browser.fetch_html(url)
        if html is None:
            continue
        if "Page Not Found" in html[:2000] or "<title>404" in html[:2000]:
            continue

        parser = ScheduleParser()
        parser.feed(html)

        for g in parser.games:
            date_raw = g.get("date_game", "")
            if not date_raw or date_raw == "Playoffs":
                continue
            try:
                date_clean = re.sub(r"^[A-Za-z]+,\s*", "", date_raw)
                dt = datetime.strptime(date_clean, "%b %d, %Y")
                game_date = dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

            boxscore_url = g.get("boxscore_url", "")
            game_id = ""
            if boxscore_url:
                m = re.search(r"/boxscores/([^.]+)\.html", boxscore_url)
                if m:
                    game_id = m.group(1)

            all_games.append({
                "game_id": game_id,
                "season": season,
                "game_date": game_date,
                "home_team": g.get("home_team_name", ""),
                "away_team": g.get("visitor_team_name", ""),
                "home_score": _pi(g.get("home_pts", "")),
                "away_score": _pi(g.get("visitor_pts", "")),
                "boxscore_url": boxscore_url,
            })

        logger.info("[%s/%s] %d juegos", season, month, len(parser.games))

    logger.info("[%s] Total calendario: %d juegos", season, len(all_games))
    return all_games


def phase_schedules(seasons: list[str], browser: BRefBrowser,
                    con: sqlite3.Connection, force: bool = False):
    """Fase 1: recolectar calendarios."""
    existing_seasons: set[str] = set()
    if not force:
        try:
            df = pd.read_sql_query("SELECT DISTINCT season FROM schedules", con)
            existing_seasons = set(df["season"].tolist())
        except Exception:
            pass

    for season in seasons:
        if should_stop():
            logger.info("Shutdown solicitado durante schedules.")
            return
        if season in existing_seasons:
            logger.info("[%s] Calendario ya existe. --force para re-scraping.", season)
            continue
        games = scrape_season_schedule(season, browser)
        if not games:
            continue
        df = pd.DataFrame(games)
        # Upsert via temp table
        df.to_sql("_tmp_sched", con, if_exists="replace", index=False)
        con.execute("INSERT OR REPLACE INTO schedules SELECT * FROM _tmp_sched")
        con.execute("DROP TABLE IF EXISTS _tmp_sched")
        con.commit()
        logger.info("[%s] Guardados %d juegos en schedules", season, len(df))


# ------------------------------------------------------------------ #
# PHASE 2: BOX SCORES
# ------------------------------------------------------------------ #
def _find_team_codes(html: str) -> list[str]:
    """Detecta team codes de los IDs box-{CODE}-game-basic en el HTML."""
    pattern = r'id="box-([A-Z]{2,3})-game-basic"'
    return re.findall(pattern, html)


def parse_four_factors(html: str, game_id: str, game_date: str) -> list[dict]:
    """Extrae Four Factors. Tabla en comentarios, id='four_factors'."""
    comment = extract_commented_html(html, "four_factors")
    source = comment if comment else html

    parser = BRefTableParser(target_id="four_factors")
    parser.feed(source)
    df = parser.to_dataframe()

    if df.empty or len(df) < 2:
        return []

    results = []
    for i, side in enumerate(["away", "home"]):
        if i >= len(df):
            break
        row = df.iloc[i]
        results.append({
            "game_id": game_id,
            "game_date": game_date,
            "team_side": side,
            "team_name": str(row.get("team_id", "")).strip(),
            "pace": _pf(row.get("pace", "")),
            "efg_pct": _pf(row.get("efg_pct", "")),
            "tov_pct": _pf(row.get("tov_pct", "")),
            "orb_pct": _pf(row.get("orb_pct", "")),
            "ft_fga": _pf(row.get("ft_rate", "")),
            "ortg": _pf(row.get("off_rtg", "")),
        })
    return results


def parse_line_score(html: str, game_id: str, game_date: str) -> list[dict]:
    """Extrae Line Scores (Q1-Q4, OT)."""
    comment = extract_commented_html(html, "line_score")
    source = comment if comment else html

    parser = BRefTableParser(target_id="line_score")
    parser.feed(source)
    df = parser.to_dataframe()

    if df.empty or len(df) < 2:
        return []

    results = []
    for i, side in enumerate(["away", "home"]):
        if i >= len(df):
            break
        row = df.iloc[i]
        results.append({
            "game_id": game_id,
            "game_date": game_date,
            "team_side": side,
            "team_name": str(row.get("team_id", "")).strip(),
            "q1": _pi(row.get("1", row.get("q1", ""))),
            "q2": _pi(row.get("2", row.get("q2", ""))),
            "q3": _pi(row.get("3", row.get("q3", ""))),
            "q4": _pi(row.get("4", row.get("q4", ""))),
            "ot1": _pi(row.get("ot1", row.get("5", ""))),
            "ot2": _pi(row.get("ot2", row.get("6", ""))),
            "ot3": _pi(row.get("ot3", row.get("7", ""))),
            "final": _pi(row.get("T", row.get("total", ""))),
        })
    return results


def parse_player_box(html: str, game_id: str, game_date: str,
                     team_code: str, team_side: str,
                     advanced: bool = False) -> list[dict]:
    """Extrae box score de jugadores (basic o advanced) de un equipo.

    Tablas en comentarios: box-{CODE}-game-basic / box-{CODE}-game-advanced
    """
    suffix = "advanced" if advanced else "basic"
    table_id = f"box-{team_code}-game-{suffix}"

    # Buscar primero en comentarios (BRef esconde tablas advanced en <!-- -->)
    # Si no esta en comentario, buscar en el HTML directo (tablas basic)
    comment = extract_commented_html(html, table_id)
    source = comment if comment else html

    parser = BRefTableParser(target_id=table_id)
    parser.feed(source)
    df = parser.to_dataframe()

    if df.empty:
        return []

    results = []
    is_starter = True
    starter_count = 0

    for _, row in df.iterrows():
        player = str(row.get("player", "")).strip()
        if player in ("Team Totals", "", "Reserves", "Starters"):
            if player == "Reserves":
                is_starter = False
            continue

        reason = str(row.get("reason", "")).strip()
        mp_val = str(row.get("mp", "")).strip()
        is_dnp = bool(reason) or mp_val in (
            "Did Not Play", "Did Not Dress", "Not With Team",
            "Player Suspended", "Inactive",
        )

        if not is_dnp and is_starter:
            starter_count += 1
            if starter_count > 5:
                is_starter = False

        if advanced:
            results.append({
                "game_id": game_id,
                "game_date": game_date,
                "team_side": team_side,
                "team_name": team_code,
                "player_name": player,
                "mp": mp_val if not is_dnp else None,
                "ts_pct": _pf(row.get("ts_pct", "")),
                "efg_pct": _pf(row.get("efg_pct", "")),
                "fg3par": _pf(row.get("fg3a_per_fga_pct", "")),
                "ftr": _pf(row.get("fta_per_fga_pct", "")),
                "orb_pct": _pf(row.get("orb_pct", "")),
                "drb_pct": _pf(row.get("drb_pct", "")),
                "trb_pct": _pf(row.get("trb_pct", "")),
                "ast_pct": _pf(row.get("ast_pct", "")),
                "stl_pct": _pf(row.get("stl_pct", "")),
                "blk_pct": _pf(row.get("blk_pct", "")),
                "tov_pct": _pf(row.get("tov_pct", "")),
                "usg_pct": _pf(row.get("usg_pct", "")),
                "ortg": _pi(row.get("off_rtg", "")),
                "drtg": _pi(row.get("def_rtg", "")),
                "bpm": _pf(row.get("bpm", "")),
            })
        else:
            results.append({
                "game_id": game_id,
                "game_date": game_date,
                "team_side": team_side,
                "team_name": team_code,
                "player_name": player,
                "is_starter": 1 if (is_starter and not is_dnp and starter_count <= 5) else 0,
                "mp": mp_val if not is_dnp else None,
                "fg": _pi(row.get("fg", "")),
                "fga": _pi(row.get("fga", "")),
                "fg_pct": _pf(row.get("fg_pct", "")),
                "fg3": _pi(row.get("fg3", "")),
                "fg3a": _pi(row.get("fg3a", "")),
                "fg3_pct": _pf(row.get("fg3_pct", "")),
                "ft": _pi(row.get("ft", "")),
                "fta": _pi(row.get("fta", "")),
                "ft_pct": _pf(row.get("ft_pct", "")),
                "orb": _pi(row.get("orb", "")),
                "drb": _pi(row.get("drb", "")),
                "trb": _pi(row.get("trb", "")),
                "ast": _pi(row.get("ast", "")),
                "stl": _pi(row.get("stl", "")),
                "blk": _pi(row.get("blk", "")),
                "tov": _pi(row.get("tov", "")),
                "pf": _pi(row.get("pf", "")),
                "pts": _pi(row.get("pts", "")),
                "gmsc": _pf(row.get("game_score", "")),
                "plus_minus": _pi(row.get("plus_minus", "")),
                "dnp": 1 if is_dnp else 0,
                "dnp_reason": reason if is_dnp else None,
            })
    return results


def scrape_boxscore(browser: BRefBrowser, game_id: str,
                    game_date: str, boxscore_url: str) -> dict | None:
    """Scrapea un boxscore completo: four factors, line score, player stats."""
    url = f"{BASE_URL}{boxscore_url}"
    html = browser.fetch_html(url)
    if html is None:
        return None

    codes = _find_team_codes(html)
    if len(codes) < 2:
        logger.warning("[%s] Solo %d team codes detectados", game_id, len(codes))
        return None

    away_code, home_code = codes[0], codes[1]

    return {
        "four_factors": parse_four_factors(html, game_id, game_date),
        "line_scores": parse_line_score(html, game_id, game_date),
        "player_basic": (
            parse_player_box(html, game_id, game_date, away_code, "away", advanced=False) +
            parse_player_box(html, game_id, game_date, home_code, "home", advanced=False)
        ),
        "player_advanced": (
            parse_player_box(html, game_id, game_date, away_code, "away", advanced=True) +
            parse_player_box(html, game_id, game_date, home_code, "home", advanced=True)
        ),
    }


# ------------------------------------------------------------------ #
# DB helpers
# ------------------------------------------------------------------ #
def _create_tables(con: sqlite3.Connection):
    """Crea tabla schedules."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS schedules (
            game_id      TEXT PRIMARY KEY,
            season       TEXT NOT NULL,
            game_date    TEXT NOT NULL,
            home_team    TEXT NOT NULL,
            away_team    TEXT NOT NULL,
            home_score   INTEGER,
            away_score   INTEGER,
            boxscore_url TEXT
        )
    """)
    con.commit()


def _create_season_tables(con: sqlite3.Connection, season: str):
    """Crea tablas por temporada."""
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS "four_factors_{season}" (
            game_id TEXT NOT NULL, game_date TEXT NOT NULL,
            team_side TEXT NOT NULL, team_name TEXT,
            pace REAL, efg_pct REAL, tov_pct REAL,
            orb_pct REAL, ft_fga REAL, ortg REAL,
            PRIMARY KEY (game_id, team_side)
        )""")
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS "line_scores_{season}" (
            game_id TEXT NOT NULL, game_date TEXT NOT NULL,
            team_side TEXT NOT NULL, team_name TEXT,
            q1 INTEGER, q2 INTEGER, q3 INTEGER, q4 INTEGER,
            ot1 INTEGER, ot2 INTEGER, ot3 INTEGER,
            final INTEGER,
            PRIMARY KEY (game_id, team_side)
        )""")
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS "player_basic_{season}" (
            game_id TEXT NOT NULL, game_date TEXT NOT NULL,
            team_side TEXT NOT NULL, team_name TEXT,
            player_name TEXT NOT NULL, is_starter INTEGER,
            mp TEXT, fg INTEGER, fga INTEGER, fg_pct REAL,
            fg3 INTEGER, fg3a INTEGER, fg3_pct REAL,
            ft INTEGER, fta INTEGER, ft_pct REAL,
            orb INTEGER, drb INTEGER, trb INTEGER,
            ast INTEGER, stl INTEGER, blk INTEGER,
            tov INTEGER, pf INTEGER, pts INTEGER,
            gmsc REAL, plus_minus INTEGER,
            dnp INTEGER DEFAULT 0, dnp_reason TEXT,
            PRIMARY KEY (game_id, team_side, player_name)
        )""")
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS "player_advanced_{season}" (
            game_id TEXT NOT NULL, game_date TEXT NOT NULL,
            team_side TEXT NOT NULL, team_name TEXT,
            player_name TEXT NOT NULL, mp TEXT,
            ts_pct REAL, efg_pct REAL, fg3par REAL, ftr REAL,
            orb_pct REAL, drb_pct REAL, trb_pct REAL, ast_pct REAL,
            stl_pct REAL, blk_pct REAL, tov_pct REAL, usg_pct REAL,
            ortg INTEGER, drtg INTEGER, bpm REAL,
            PRIMARY KEY (game_id, team_side, player_name)
        )""")
    con.commit()


def _upsert_rows(con: sqlite3.Connection, table: str, rows: list[dict]):
    """INSERT OR REPLACE filas via tabla temporal."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    tmp = f"_tmp_{table.replace('-', '_')}"
    df.to_sql(tmp, con, if_exists="replace", index=False)
    cols = ", ".join(df.columns.tolist())
    con.execute(f'INSERT OR REPLACE INTO "{table}" ({cols}) SELECT {cols} FROM "{tmp}"')
    con.execute(f'DROP TABLE IF EXISTS "{tmp}"')


def _save_boxscore(con: sqlite3.Connection, season: str, data: dict):
    """Guarda datos de un boxscore en las 4 tablas."""
    mapping = {
        "four_factors": f"four_factors_{season}",
        "line_scores": f"line_scores_{season}",
        "player_basic": f"player_basic_{season}",
        "player_advanced": f"player_advanced_{season}",
    }
    for key, table in mapping.items():
        _upsert_rows(con, table, data.get(key, []))
    con.commit()


def _get_scraped_ids(con: sqlite3.Connection, season: str) -> set[str]:
    """Retorna game_ids ya scrapeados."""
    try:
        df = pd.read_sql_query(
            f'SELECT DISTINCT game_id FROM "four_factors_{season}"', con)
        return set(df["game_id"].tolist())
    except Exception:
        return set()


def phase_boxscores(seasons: list[str], browser: BRefBrowser,
                    con: sqlite3.Connection, force: bool = False,
                    max_games: int | None = None):
    """Fase 2: scrapear box scores."""
    for season in seasons:
        if should_stop():
            logger.info("Shutdown solicitado entre temporadas.")
            return

        logger.info("=" * 60)
        logger.info("  Phase 2: Boxscores %s", season)
        logger.info("=" * 60)

        _create_season_tables(con, season)

        try:
            games_df = pd.read_sql_query(
                "SELECT game_id, game_date, boxscore_url FROM schedules "
                "WHERE season = ? AND boxscore_url IS NOT NULL AND boxscore_url != ''",
                con, params=(season,))
        except Exception:
            logger.error("[%s] Sin calendario. Ejecuta --phase schedules primero.", season)
            continue

        if games_df.empty:
            logger.warning("[%s] Sin juegos con boxscore URL.", season)
            continue

        if not force:
            existing = _get_scraped_ids(con, season)
            before = len(games_df)
            games_df = games_df[~games_df["game_id"].isin(existing)]
            logger.info("[%s] %d pendientes (skip %d ya scrapeados)",
                        season, len(games_df), before - len(games_df))

        if max_games:
            games_df = games_df.head(max_games)

        total = len(games_df)
        if total == 0:
            logger.info("[%s] Nada que scrapear.", season)
            continue

        ok = fail = 0
        t0 = time.time()

        for _, row in games_df.iterrows():
            # Checkear shutdown entre juegos (permite terminar limpiamente)
            if should_stop():
                logger.info("[%s] Shutdown solicitado. %d ok, %d fail hasta ahora.",
                            season, ok, fail)
                return

            gid = row["game_id"]
            data = scrape_boxscore(browser, gid, row["game_date"], row["boxscore_url"])

            if data is None:
                fail += 1
                logger.warning("[%s] FAIL %s", season, gid)
                continue

            _save_boxscore(con, season, data)
            ok += 1

            if ok % 25 == 0:
                elapsed = time.time() - t0
                avg = elapsed / ok
                eta = avg * (total - ok - fail) / 60
                pct = (ok + fail) / total * 100
                logger.info(
                    "[%s] %d/%d (%.1f%%) | OK:%d FAIL:%d | %.1fs/game | ETA: %.1fm",
                    season, ok + fail, total, pct, ok, fail, avg, eta)

        elapsed = time.time() - t0
        logger.info("[%s] Completada: %d ok, %d fail de %d (%.1f min)",
                    season, ok, fail, total, elapsed / 60)


# ------------------------------------------------------------------ #
# PHASE 3: PLUS-MINUS (on/off court por cuarto)
# ------------------------------------------------------------------ #
_PM_PLAYER_RE = re.compile(
    r'<div class="player"><span>(.*?)</span>\s*\((.*?)\)</div>',
)
_PM_SUMMARY_RE = re.compile(
    r'On:\s*([\+\-]?\d+).*?Off:\s*([\+\-]?\d+).*?Net:\s*([\+\-]?\d+)',
)


def parse_plus_minus(html: str, game_id: str, game_date: str) -> list[dict]:
    """Extrae plus-minus de la pagina BRef.

    BRef plus-minus page usa divs (NO tablas):
    - <div class="player"><span>Name</span> (On: +N · Off: -N · Net: +N)</div>
    - El primer grupo de jugadores = away, el segundo = home
    - Separados por un header de cuartos (1st Qtr, 2nd Qtr...)
    """
    matches = list(_PM_PLAYER_RE.finditer(html))
    if not matches:
        return []

    # Encontrar el separador entre equipos: buscar el segundo header de cuartos
    # El header aparece como: <div style="width:250px;">1st Qtr</div>
    # Solo lo buscamos entre el primer y ultimo jugador
    qtr_headers = list(re.finditer(r'1st\s+Qtr', html))

    # El segundo header de cuartos marca el inicio del segundo equipo (home)
    split_pos = None
    if len(qtr_headers) >= 2:
        split_pos = qtr_headers[1].start()
    elif len(qtr_headers) == 1:
        # Solo 1 header → todos los jugadores de antes son away, despues home
        split_pos = qtr_headers[0].start()

    results = []
    for m in matches:
        name = m.group(1).strip()
        summary = m.group(2).replace('\xa0', ' ')

        sm = _PM_SUMMARY_RE.search(summary)
        if not sm:
            continue

        # Determinar team_side segun posicion relativa al header
        if split_pos is not None:
            team_side = "away" if m.start() < split_pos else "home"
        else:
            # Sin header → asumir primera mitad away, segunda home
            mid = len(matches) // 2
            idx = matches.index(m)
            team_side = "away" if idx < mid else "home"

        results.append({
            "game_id": game_id,
            "game_date": game_date,
            "team_side": team_side,
            "player_name": name,
            "on_court": int(sm.group(1)),
            "off_court": int(sm.group(2)),
            "net": int(sm.group(3)),
        })
    return results


def _create_plus_minus_table(con: sqlite3.Connection, season: str):
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS "plus_minus_{season}" (
            game_id TEXT NOT NULL, game_date TEXT NOT NULL,
            team_side TEXT NOT NULL,
            player_name TEXT NOT NULL,
            on_court INTEGER, off_court INTEGER, net INTEGER,
            PRIMARY KEY (game_id, team_side, player_name)
        )""")
    con.commit()


def _get_pm_scraped_ids(con: sqlite3.Connection, season: str) -> set[str]:
    try:
        df = pd.read_sql_query(
            f'SELECT DISTINCT game_id FROM "plus_minus_{season}"', con)
        return set(df["game_id"].tolist())
    except Exception:
        return set()


def phase_plus_minus(seasons: list[str], browser: BRefBrowser,
                     con: sqlite3.Connection, force: bool = False,
                     max_games: int | None = None):
    """Fase 3: scrapear plus-minus por cuarto."""
    for season in seasons:
        if should_stop():
            return

        logger.info("=" * 60)
        logger.info("  Phase: Plus-Minus %s", season)
        logger.info("=" * 60)

        _create_plus_minus_table(con, season)

        try:
            games_df = pd.read_sql_query(
                "SELECT game_id, game_date, boxscore_url FROM schedules "
                "WHERE season = ? AND boxscore_url IS NOT NULL AND boxscore_url != ''",
                con, params=(season,))
        except Exception:
            logger.error("[%s] Sin calendario. Ejecuta --phase schedules primero.", season)
            continue

        if games_df.empty:
            continue

        if not force:
            existing = _get_pm_scraped_ids(con, season)
            before = len(games_df)
            games_df = games_df[~games_df["game_id"].isin(existing)]
            logger.info("[%s] %d pendientes (skip %d ya scrapeados)",
                        season, len(games_df), before - len(games_df))

        if max_games:
            games_df = games_df.head(max_games)

        total = len(games_df)
        if total == 0:
            logger.info("[%s] Plus-minus: nada que scrapear.", season)
            continue

        ok = fail = 0
        t0 = time.time()

        for _, row in games_df.iterrows():
            if should_stop():
                logger.info("[%s] Shutdown. %d ok, %d fail.", season, ok, fail)
                return

            gid = row["game_id"]
            url = f"{BASE_URL}/boxscores/plus-minus/{gid}.html"
            html = browser.fetch_html(url)
            if html is None:
                fail += 1
                continue

            pm_data = parse_plus_minus(html, gid, row["game_date"])
            if not pm_data:
                fail += 1
                logger.warning("[%s] Plus-minus vacio", gid)
                continue

            _upsert_rows(con, f"plus_minus_{season}", pm_data)
            con.commit()
            ok += 1

            if ok % 25 == 0:
                elapsed = time.time() - t0
                avg = elapsed / ok
                eta = avg * (total - ok - fail) / 60
                pct = (ok + fail) / total * 100
                logger.info(
                    "[%s] PM %d/%d (%.1f%%) | OK:%d FAIL:%d | %.1fs/game | ETA: %.1fm",
                    season, ok + fail, total, pct, ok, fail, avg, eta)

        elapsed = time.time() - t0
        logger.info("[%s] Plus-Minus completada: %d ok, %d fail de %d (%.1f min)",
                    season, ok, fail, total, elapsed / 60)


# ------------------------------------------------------------------ #
# PHASE 4: PLAY-BY-PLAY
# ------------------------------------------------------------------ #
class PBPParser(HTMLParser):
    """Parser custom para la tabla de play-by-play (id='pbp').

    La tabla PBP tiene 6 columnas: time, away_action, (spacer), score, (spacer), home_action.
    Los cuartos se separan con filas header dentro del tbody.
    """
    def __init__(self):
        super().__init__()
        self._in_table = False
        self._in_tbody = False
        self._in_row = False
        self._in_cell = False
        self._cell_data: list[str] = []
        self._cell_tag = ""
        self._cell_attrs: dict = {}
        self._cell_href: str | None = None
        self._current_row_cells: list[dict] = []
        self._current_row_is_header = False

        self.quarter = 1
        self.plays: list[dict] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table" and a.get("id") == "pbp":
            self._in_table = True
            return
        if not self._in_table:
            return
        if tag == "tbody":
            self._in_tbody = True
        elif tag == "tr" and self._in_tbody:
            self._in_row = True
            self._current_row_cells = []
            self._current_row_is_header = False
        elif tag in ("td", "th") and self._in_row:
            self._in_cell = True
            self._cell_data = []
            self._cell_tag = tag
            self._cell_attrs = a
            self._cell_href = None
            # Detectar si es una fila de header de cuarto
            colspan = a.get("colspan", "")
            if colspan and int(colspan) > 2:
                self._current_row_is_header = True
        elif tag == "a" and self._in_cell:
            self._cell_href = a.get("href")

    def handle_endtag(self, tag):
        if not self._in_table:
            return
        if tag == "table":
            self._in_table = False
        elif tag == "tbody":
            self._in_tbody = False
        elif tag == "tr" and self._in_row:
            self._in_row = False
            if self._current_row_is_header:
                # Detectar cuarto del texto del header
                if self._current_row_cells:
                    txt = self._current_row_cells[0].get("text", "").lower()
                    if "1st" in txt:
                        self.quarter = 1
                    elif "2nd" in txt:
                        self.quarter = 2
                    elif "3rd" in txt:
                        self.quarter = 3
                    elif "4th" in txt:
                        self.quarter = 4
                    elif "ot" in txt or "overtime" in txt:
                        # Parsear numero de OT
                        m = re.search(r'(\d+)', txt)
                        if m:
                            self.quarter = 4 + int(m.group(1))
                        else:
                            self.quarter = 5  # 1er OT
            elif len(self._current_row_cells) >= 2:
                self.plays.append({
                    "cells": self._current_row_cells,
                    "quarter": self.quarter,
                })
        elif tag in ("td", "th") and self._in_cell:
            text = "".join(self._cell_data).strip()
            self._current_row_cells.append({
                "text": text,
                "tag": self._cell_tag,
                "data_stat": self._cell_attrs.get("data-stat", ""),
                "href": self._cell_href,
            })
            self._in_cell = False

    def handle_data(self, data):
        if self._in_cell:
            self._cell_data.append(data)


def parse_pbp(html: str, game_id: str, game_date: str) -> list[dict]:
    """Extrae play-by-play de una pagina BRef.

    Formato de la tabla PBP (6 columnas):
      [time, away_action, away_delta, score, home_delta, home_action]
    Filas de jump ball/quarter header tienen 2 columnas (colspan).
    """
    parser = PBPParser()
    parser.feed(html)

    if not parser.plays:
        return []

    results = []
    play_number = 0

    for play in parser.plays:
        cells = play["cells"]
        quarter = play["quarter"]
        n = len(cells)

        if n < 2:
            continue

        time_remaining = cells[0]["text"].strip()
        if not time_remaining or time_remaining == "Time":
            continue

        # Filas con 2 cells = jump ball / evento general (colspan)
        if n == 2:
            play_number += 1
            results.append({
                "game_id": game_id,
                "game_date": game_date,
                "play_number": play_number,
                "quarter": quarter,
                "time_remaining": time_remaining,
                "team_side": "neutral",
                "action": cells[1]["text"].strip(),
                "away_score": None,
                "home_score": None,
            })
            continue

        # Filas normales: 6 cells
        # cells[1]=away_action, cells[2]=away_delta, cells[3]=score,
        # cells[4]=home_delta, cells[5]=home_action
        away_action = cells[1]["text"].strip() if n > 1 else ""
        home_action = cells[5]["text"].strip() if n > 5 else ""
        score_text = cells[3]["text"].strip() if n > 3 else ""

        # Limpiar &nbsp; (se renderiza como espacio)
        away_action = "" if away_action in ("\xa0", "&nbsp;", "") else away_action
        home_action = "" if home_action in ("\xa0", "&nbsp;", "") else home_action

        # Parsear score
        away_score = home_score = None
        if re.match(r'^\d+-\d+$', score_text):
            parts = score_text.split("-")
            away_score = int(parts[0])
            home_score = int(parts[1])

        # Determinar equipo con accion
        if away_action and not home_action:
            team_side = "away"
            action = away_action
        elif home_action and not away_action:
            team_side = "home"
            action = home_action
        elif away_action and home_action:
            # Ambos tienen texto (raro, puede ser scoring delta en away)
            team_side = "away"
            action = away_action
        else:
            continue  # Sin accion

        # Ignorar deltas sueltos (+2, +3) que no son acciones reales
        if re.match(r'^[+\-]\d+$', action):
            continue

        play_number += 1
        results.append({
            "game_id": game_id,
            "game_date": game_date,
            "play_number": play_number,
            "quarter": quarter,
            "time_remaining": time_remaining,
            "team_side": team_side,
            "action": action,
            "away_score": away_score,
            "home_score": home_score,
        })
    return results


def _create_pbp_table(con: sqlite3.Connection, season: str):
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS "pbp_{season}" (
            game_id TEXT NOT NULL, game_date TEXT NOT NULL,
            play_number INTEGER NOT NULL,
            quarter INTEGER, time_remaining TEXT,
            team_side TEXT, action TEXT,
            away_score INTEGER, home_score INTEGER,
            PRIMARY KEY (game_id, play_number)
        )""")
    con.commit()


def _get_pbp_scraped_ids(con: sqlite3.Connection, season: str) -> set[str]:
    try:
        df = pd.read_sql_query(
            f'SELECT DISTINCT game_id FROM "pbp_{season}"', con)
        return set(df["game_id"].tolist())
    except Exception:
        return set()


def phase_pbp(seasons: list[str], browser: BRefBrowser,
              con: sqlite3.Connection, force: bool = False,
              max_games: int | None = None):
    """Fase 4: scrapear play-by-play."""
    for season in seasons:
        if should_stop():
            return

        logger.info("=" * 60)
        logger.info("  Phase: Play-by-Play %s", season)
        logger.info("=" * 60)

        _create_pbp_table(con, season)

        try:
            games_df = pd.read_sql_query(
                "SELECT game_id, game_date, boxscore_url FROM schedules "
                "WHERE season = ? AND boxscore_url IS NOT NULL AND boxscore_url != ''",
                con, params=(season,))
        except Exception:
            logger.error("[%s] Sin calendario.", season)
            continue

        if games_df.empty:
            continue

        if not force:
            existing = _get_pbp_scraped_ids(con, season)
            before = len(games_df)
            games_df = games_df[~games_df["game_id"].isin(existing)]
            logger.info("[%s] %d pendientes (skip %d ya scrapeados)",
                        season, len(games_df), before - len(games_df))

        if max_games:
            games_df = games_df.head(max_games)

        total = len(games_df)
        if total == 0:
            logger.info("[%s] PBP: nada que scrapear.", season)
            continue

        ok = fail = 0
        t0 = time.time()

        for _, row in games_df.iterrows():
            if should_stop():
                logger.info("[%s] Shutdown. %d ok, %d fail.", season, ok, fail)
                return

            gid = row["game_id"]
            url = f"{BASE_URL}/boxscores/pbp/{gid}.html"
            html = browser.fetch_html(url)
            if html is None:
                fail += 1
                continue

            pbp_data = parse_pbp(html, gid, row["game_date"])
            if not pbp_data:
                fail += 1
                logger.warning("[%s] PBP vacio", gid)
                continue

            _upsert_rows(con, f"pbp_{season}", pbp_data)
            con.commit()
            ok += 1

            if ok % 25 == 0:
                elapsed = time.time() - t0
                avg = elapsed / ok
                eta = avg * (total - ok - fail) / 60
                pct = (ok + fail) / total * 100
                logger.info(
                    "[%s] PBP %d/%d (%.1f%%) | OK:%d FAIL:%d | %.1fs/game | ETA: %.1fm",
                    season, ok + fail, total, pct, ok, fail, avg, eta)

        elapsed = time.time() - t0
        logger.info("[%s] PBP completada: %d ok, %d fail de %d (%.1f min)",
                    season, ok, fail, total, elapsed / 60)


# ------------------------------------------------------------------ #
# PHASE 5: SHOT CHART
# ------------------------------------------------------------------ #
_SHOT_TIP_RE = re.compile(
    r'(?:(\d)[a-z]{2}\s+quarter|(\d*)\s*(?:st|nd|rd|th)?\s*overtime)'
    r',\s*([\d:\.]+)\s+remaining'
    r'.*?<br>(.*?)\s+(missed|made)\s+(\d)-pointer\s+from\s+(\d+)\s+ft',
    re.IGNORECASE,
)

# Regex simplificada para tips que no matchean la primera
_SHOT_TIP_SIMPLE = re.compile(
    r'(.*?)\s+(missed|made)\s+(\d)-pointer\s+from\s+(\d+)\s+ft',
    re.IGNORECASE,
)

# Dimensiones del court SVG en BRef (pixeles)
COURT_WIDTH_PX = 500.0
COURT_HEIGHT_PX = 472.0
COURT_WIDTH_FT = 50.0
COURT_HEIGHT_FT = 47.0


class ShotChartParser(HTMLParser):
    """Parser custom para shot chart — extrae divs con class='tooltip'."""
    def __init__(self):
        super().__init__()
        self._in_shots_container = False
        self._current_team = ""
        self._in_tooltip_div = False
        self._div_depth = 0
        self._current_attrs: dict = {}
        self._current_text: list[str] = []
        self.shots: list[dict] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "div":
            self._div_depth += 1
            div_id = a.get("id", "")
            # Container de equipo: id="shots-{CODE}"
            if div_id.startswith("shots-"):
                self._in_shots_container = True
                self._current_team = div_id.replace("shots-", "")
            # Shot individual: class contiene "tooltip"
            cls = a.get("class", "")
            if "tooltip" in cls and self._in_shots_container:
                self._in_tooltip_div = True
                self._current_attrs = a
                self._current_text = []

    def handle_endtag(self, tag):
        if tag == "div":
            if self._in_tooltip_div:
                self._in_tooltip_div = False
                self._process_shot()
            self._div_depth -= 1
            if self._div_depth <= 0:
                self._in_shots_container = False
                self._div_depth = 0

    def handle_data(self, data):
        if self._in_tooltip_div:
            self._current_text.append(data)

    def _process_shot(self):
        a = self._current_attrs
        cls = a.get("class", "")
        style = a.get("style", "")
        tip = a.get("tip", "")
        text = "".join(self._current_text).strip()

        # Determinar make/miss del inner text
        made = None
        if "●" in text or "•" in text:
            made = 1
        elif "×" in text or "✕" in text or "x" in text.lower():
            made = 0

        # Extraer player_id y quarter de class
        player_id = ""
        quarter = None
        for part in cls.split():
            if part.startswith("p-"):
                player_id = part[2:]
            elif part.startswith("q-"):
                try:
                    quarter = int(part[2:])
                except ValueError:
                    pass

        # Extraer coordenadas de style
        x_px = y_px = None
        top_m = re.search(r'top:\s*([\d.]+)px', style)
        left_m = re.search(r'left:\s*([\d.]+)px', style)
        if top_m:
            y_px = float(top_m.group(1))
        if left_m:
            x_px = float(left_m.group(1))

        # Convertir a pies
        x_ft = round(x_px / COURT_WIDTH_PX * COURT_WIDTH_FT, 1) if x_px is not None else None
        y_ft = round(y_px / COURT_HEIGHT_PX * COURT_HEIGHT_FT, 1) if y_px is not None else None

        # Parsear tip para metadata
        player_name = ""
        time_remaining = ""
        point_value = None
        distance_ft = None

        # Intentar regex completa
        m = _SHOT_TIP_RE.search(tip)
        if m:
            if m.group(1):  # quarter
                quarter = quarter or int(m.group(1))
            elif m.group(2):  # overtime
                ot_num = int(m.group(2)) if m.group(2) else 1
                quarter = quarter or (4 + ot_num)
            time_remaining = m.group(3)
            player_name = m.group(4).strip()
            if m.group(5).lower() == "made":
                made = 1
            else:
                made = 0
            point_value = int(m.group(6))
            distance_ft = int(m.group(7))
        else:
            # Regex simplificada
            m2 = _SHOT_TIP_SIMPLE.search(tip)
            if m2:
                player_name = m2.group(1).strip()
                made = 1 if m2.group(2).lower() == "made" else 0
                point_value = int(m2.group(3))
                distance_ft = int(m2.group(4))

        self.shots.append({
            "team_code": self._current_team,
            "player_id": player_id,
            "player_name": player_name,
            "quarter": quarter,
            "time_remaining": time_remaining,
            "x_px": x_px,
            "y_px": y_px,
            "x_ft": x_ft,
            "y_ft": y_ft,
            "distance_ft": distance_ft,
            "point_value": point_value,
            "made": made,
        })


def parse_shot_chart(html: str, game_id: str, game_date: str) -> list[dict]:
    """Extrae shot chart de una pagina BRef."""
    parser = ShotChartParser()
    parser.feed(html)

    if not parser.shots:
        return []

    results = []
    for i, shot in enumerate(parser.shots, 1):
        shot["game_id"] = game_id
        shot["game_date"] = game_date
        shot["shot_number"] = i
        results.append(shot)
    return results


def _create_shot_chart_table(con: sqlite3.Connection, season: str):
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS "shot_chart_{season}" (
            game_id TEXT NOT NULL, game_date TEXT NOT NULL,
            shot_number INTEGER NOT NULL,
            team_code TEXT, player_id TEXT, player_name TEXT,
            quarter INTEGER, time_remaining TEXT,
            x_px REAL, y_px REAL, x_ft REAL, y_ft REAL,
            distance_ft INTEGER, point_value INTEGER, made INTEGER,
            PRIMARY KEY (game_id, shot_number)
        )""")
    con.commit()


def _get_sc_scraped_ids(con: sqlite3.Connection, season: str) -> set[str]:
    try:
        df = pd.read_sql_query(
            f'SELECT DISTINCT game_id FROM "shot_chart_{season}"', con)
        return set(df["game_id"].tolist())
    except Exception:
        return set()


def phase_shot_chart(seasons: list[str], browser: BRefBrowser,
                     con: sqlite3.Connection, force: bool = False,
                     max_games: int | None = None):
    """Fase 5: scrapear shot charts."""
    for season in seasons:
        if should_stop():
            return

        logger.info("=" * 60)
        logger.info("  Phase: Shot Chart %s", season)
        logger.info("=" * 60)

        _create_shot_chart_table(con, season)

        try:
            games_df = pd.read_sql_query(
                "SELECT game_id, game_date, boxscore_url FROM schedules "
                "WHERE season = ? AND boxscore_url IS NOT NULL AND boxscore_url != ''",
                con, params=(season,))
        except Exception:
            logger.error("[%s] Sin calendario.", season)
            continue

        if games_df.empty:
            continue

        if not force:
            existing = _get_sc_scraped_ids(con, season)
            before = len(games_df)
            games_df = games_df[~games_df["game_id"].isin(existing)]
            logger.info("[%s] %d pendientes (skip %d ya scrapeados)",
                        season, len(games_df), before - len(games_df))

        if max_games:
            games_df = games_df.head(max_games)

        total = len(games_df)
        if total == 0:
            logger.info("[%s] Shot Chart: nada que scrapear.", season)
            continue

        ok = fail = 0
        t0 = time.time()

        for _, row in games_df.iterrows():
            if should_stop():
                logger.info("[%s] Shutdown. %d ok, %d fail.", season, ok, fail)
                return

            gid = row["game_id"]
            url = f"{BASE_URL}/boxscores/shot-chart/{gid}.html"
            html = browser.fetch_html(url)
            if html is None:
                fail += 1
                continue

            sc_data = parse_shot_chart(html, gid, row["game_date"])
            if not sc_data:
                fail += 1
                logger.warning("[%s] Shot chart vacio", gid)
                continue

            _upsert_rows(con, f"shot_chart_{season}", sc_data)
            con.commit()
            ok += 1

            if ok % 25 == 0:
                elapsed = time.time() - t0
                avg = elapsed / ok
                eta = avg * (total - ok - fail) / 60
                pct = (ok + fail) / total * 100
                logger.info(
                    "[%s] SC %d/%d (%.1f%%) | OK:%d FAIL:%d | %.1fs/game | ETA: %.1fm",
                    season, ok + fail, total, pct, ok, fail, avg, eta)

        elapsed = time.time() - t0
        logger.info("[%s] Shot Chart completada: %d ok, %d fail de %d (%.1f min)",
                    season, ok, fail, total, elapsed / 60)


# ------------------------------------------------------------------ #
# PHASE 6: TEAM SHOOTING (season pages)
# ------------------------------------------------------------------ #
def parse_team_shooting(html: str, season: str, team_code: str) -> list[dict]:
    """Extrae tabla de shooting de una team season page."""
    comment = extract_commented_html(html, "shooting")
    source = comment if comment else html

    parser = BRefTableParser(target_id="shooting")
    parser.feed(source)
    df = parser.to_dataframe()

    if df.empty:
        return []

    results = []
    for _, row in df.iterrows():
        # BRef usa "name_display" en shooting tables, "player" en box scores
        player = str(row.get("name_display", row.get("player", ""))).strip()
        if not player or player in ("Team Totals", "League Average", ""):
            continue

        rec = {
            "season": season,
            "team_code": team_code,
            "player_name": player,
        }
        # Capturar todas las columnas numericas que el parser encontro
        skip = {"name_display", "player", "ranker", "awards", "pos"}
        for col_name in row.index:
            if col_name in skip:
                continue
            val = str(row.get(col_name, "")).strip()
            if val:
                rec[col_name] = _pf(val)
        results.append(rec)
    return results


def _create_team_shooting_table(con: sqlite3.Connection, season: str,
                                extra_cols: list[str] | None = None):
    """Crea tabla team_shooting dinamicamente.

    Como los data-stat exactos se descubren al parsear, creamos con columnas base
    y agregamos nuevas columnas si aparecen.
    """
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS "team_shooting_{season}" (
            season TEXT NOT NULL, team_code TEXT NOT NULL,
            player_name TEXT NOT NULL,
            PRIMARY KEY (season, team_code, player_name)
        )""")
    if extra_cols:
        existing = {r[1] for r in con.execute(
            f'PRAGMA table_info("team_shooting_{season}")').fetchall()}
        for col in extra_cols:
            if col not in existing and col not in ("season", "team_code", "player_name"):
                con.execute(
                    f'ALTER TABLE "team_shooting_{season}" ADD COLUMN "{col}" REAL')
    con.commit()


def _get_ts_scraped_teams(con: sqlite3.Connection, season: str) -> set[str]:
    try:
        df = pd.read_sql_query(
            f'SELECT DISTINCT team_code FROM "team_shooting_{season}"', con)
        return set(df["team_code"].tolist())
    except Exception:
        return set()


def phase_team_shooting(seasons: list[str], browser: BRefBrowser,
                        con: sqlite3.Connection, force: bool = False,
                        max_games: int | None = None):
    """Fase 6: scrapear shooting de team season pages."""
    for season in seasons:
        if should_stop():
            return

        logger.info("=" * 60)
        logger.info("  Phase: Team Shooting %s", season)
        logger.info("=" * 60)

        _create_team_shooting_table(con, season)
        bref_year = season_to_bref_year(season)

        teams = BREF_TEAM_CODES[:]
        if not force:
            existing = _get_ts_scraped_teams(con, season)
            teams = [t for t in teams if t not in existing]
            logger.info("[%s] %d equipos pendientes (skip %d ya scrapeados)",
                        season, len(teams), len(BREF_TEAM_CODES) - len(teams))

        if max_games:
            teams = teams[:max_games]

        total = len(teams)
        if total == 0:
            logger.info("[%s] Team Shooting: nada que scrapear.", season)
            continue

        ok = fail = 0
        t0 = time.time()

        for team_code in teams:
            if should_stop():
                logger.info("[%s] Shutdown. %d ok, %d fail.", season, ok, fail)
                return

            url = f"{BASE_URL}/teams/{team_code}/{bref_year}.html"
            html = browser.fetch_html(url)
            if html is None:
                fail += 1
                logger.warning("[%s/%s] FAIL fetch", season, team_code)
                continue

            ts_data = parse_team_shooting(html, season, team_code)
            if not ts_data:
                fail += 1
                logger.warning("[%s/%s] Shooting table vacia", season, team_code)
                continue

            # Descubrir columnas nuevas y agregarlas al schema
            all_cols = set()
            for r in ts_data:
                all_cols.update(r.keys())
            _create_team_shooting_table(con, season, list(all_cols))
            _upsert_rows(con, f"team_shooting_{season}", ts_data)
            con.commit()
            ok += 1

            logger.info("[%s] %s: %d jugadores", season, team_code, len(ts_data))

        elapsed = time.time() - t0
        logger.info("[%s] Team Shooting completada: %d ok, %d fail de %d (%.1f min)",
                    season, ok, fail, total, elapsed / 60)


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    global _active_browser, _active_connection

    ap = argparse.ArgumentParser(
        description="Scrapea Basketball Reference: datos historicos NBA")
    ap.add_argument("--phase", choices=[
        "schedules", "boxscores", "plus-minus", "pbp",
        "shot-chart", "team-shooting", "subpages", "all",
    ], help="Fase a ejecutar")
    ap.add_argument("--season", type=str, default=None,
                    help='Temporada (ej: "2025-26"). Default: todas 2014-15..2025-26')
    ap.add_argument("--force", action="store_true", help="Re-scraping")
    ap.add_argument("--max-games", type=int, default=None, help="Max juegos (testing)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.phase:
        ap.error("--phase es requerido")

    seasons = [args.season] if args.season else ALL_SEASONS

    # Registrar signal handlers y caffeinate
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    atexit.register(_cleanup)
    _start_caffeinate()

    logger.info("=" * 60)
    logger.info("  BASKETBALL REFERENCE SCRAPER (Playwright)")
    logger.info("  Phase: %s | Seasons: %s", args.phase, ", ".join(seasons))
    logger.info("  DB: %s", BREF_DB)
    logger.info("  Force: %s | Max-games: %s", args.force, args.max_games)
    logger.info("  caffeinate: ON | Resume: automático")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("[DRY RUN] %d temporadas: %s", len(seasons), seasons)
        _stop_caffeinate()
        return

    con = sqlite3.connect(str(BREF_DB))
    _create_tables(con)
    _active_connection = con

    # Conectar al Chrome real via CDP (debe estar abierto con --remote-debugging-port=9222)
    browser = BRefBrowser()
    browser.start()
    _active_browser = browser

    # Expandir alias de fases
    phases_to_run = []
    if args.phase == "all":
        phases_to_run = ["schedules", "boxscores", "plus-minus", "pbp",
                         "shot-chart", "team-shooting"]
    elif args.phase == "subpages":
        phases_to_run = ["plus-minus", "pbp", "shot-chart"]
    else:
        phases_to_run = [args.phase]

    try:
        for phase in phases_to_run:
            if should_stop():
                break
            if phase == "schedules":
                phase_schedules(seasons, browser, con, force=args.force)
            elif phase == "boxscores":
                phase_boxscores(seasons, browser, con,
                                force=args.force, max_games=args.max_games)
            elif phase == "plus-minus":
                phase_plus_minus(seasons, browser, con,
                                 force=args.force, max_games=args.max_games)
            elif phase == "pbp":
                phase_pbp(seasons, browser, con,
                          force=args.force, max_games=args.max_games)
            elif phase == "shot-chart":
                phase_shot_chart(seasons, browser, con,
                                 force=args.force, max_games=args.max_games)
            elif phase == "team-shooting":
                phase_team_shooting(seasons, browser, con,
                                    force=args.force, max_games=args.max_games)
    finally:
        _cleanup()
        if should_stop():
            logger.info("Scraping interrumpido. Re-ejecuta el mismo comando para continuar.")
        else:
            logger.info("Scraping completado.")


if __name__ == "__main__":
    main()
