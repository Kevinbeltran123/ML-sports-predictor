"""Scrapea Stathead (premium) para datos NBA: Quarter Finder, Player/Team Game Finder.

Se conecta al mismo Chrome real via CDP que usa scrape_basketball_reference.py.
Requiere estar logueado en stathead.com con cuenta premium.

Prerequisito (mismo que BRef scraper):
    pkill -9 -f "Google Chrome"
    sleep 2
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
        --remote-debugging-port=9222 \
        --remote-allow-origins="*" \
        --user-data-dir="/tmp/chrome-debug-profile" \
        --no-first-run
    # En el Chrome que se abre, loguear en stathead.com

Fases:
  --phase quarter_finder       → stats por cuarto (Q1-Q4, 1H, 2H) por jugador
  --phase player_game_finder   → game logs por jugador con stats completas
  --phase team_game_finder     → game logs por equipo con advanced stats
  --phase all                  → las 3 fases en orden

Uso:
    PYTHONPATH=. python scripts/scrape_stathead.py --phase team_game_finder --season 2024-25 --max-pages 2
    PYTHONPATH=. python scripts/scrape_stathead.py --phase quarter_finder
    PYTHONPATH=. python scripts/scrape_stathead.py --phase all --dry-run
"""

from __future__ import annotations

import argparse
import atexit
import math
import re
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from html.parser import HTMLParser

import pandas as pd
from patchright.sync_api import sync_playwright, Page, BrowserContext

from src.config import STATHEAD_DB, get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------ #
# Caffeinate — previene sleep de macOS durante scraping largo
# ------------------------------------------------------------------ #
_caffeinate_proc: subprocess.Popen | None = None


def _start_caffeinate():
    global _caffeinate_proc
    if sys.platform != "darwin":
        return
    try:
        _caffeinate_proc = subprocess.Popen(
            ["caffeinate", "-dims"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        logger.info("caffeinate iniciado (PID %d)", _caffeinate_proc.pid)
    except FileNotFoundError:
        pass


def _stop_caffeinate():
    global _caffeinate_proc
    if _caffeinate_proc is not None:
        try:
            _caffeinate_proc.terminate()
            _caffeinate_proc.wait(timeout=5)
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
_active_browser: "StatheadBrowser | None" = None
_active_connection: sqlite3.Connection | None = None


def _signal_handler(signum, frame):
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    if _shutdown_requested:
        logger.warning("Segunda señal (%s), forzando salida...", sig_name)
        _cleanup()
        sys.exit(1)
    _shutdown_requested = True
    logger.warning(
        "Señal %s recibida. Terminando después de la página actual... "
        "(Ctrl+C de nuevo para forzar)", sig_name)


def _cleanup():
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
    return _shutdown_requested


# ------------------------------------------------------------------ #
# Constantes
# ------------------------------------------------------------------ #
STATHEAD_BASE = "https://stathead.com"
CDP_URL = "http://localhost:9222"

REQUEST_DELAY = 5.0     # Stathead más estricto que BRef
MAX_RETRIES = 3
RETRY_DELAY = 8          # backoff lineal: 8s, 16s, 24s
RESULTS_PER_PAGE = 200   # fijo de Stathead

ALL_SEASONS = [
    "2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24",
    "2024-25", "2025-26",
]

# Quarters a scrapear (OT omitido: muy pocos datos, ruidoso)
QUARTERS = ["q1", "q2", "q3", "q4", "1h", "2h"]


def season_to_year(season: str) -> int:
    """'2014-15' → 2015 (año que usa Stathead en URLs)."""
    return int(season.split("-")[0]) + 1


# ------------------------------------------------------------------ #
# Browser — conecta a Chrome real via CDP (igual que BRef scraper)
# ------------------------------------------------------------------ #
class StatheadBrowser:
    """Navegador via CDP para scrapear Stathead.

    Reutiliza el mismo Chrome con debugging abierto en puerto 9222.
    Detecta logout y CAPTCHA para no perder requests inútiles.
    """

    def __init__(self):
        self._pw = sync_playwright().start()
        self._browser = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    def start(self) -> Page:
        try:
            self._browser = self._pw.chromium.connect_over_cdp(CDP_URL)
        except Exception as e:
            logger.error(
                "No se pudo conectar a Chrome en %s.\n"
                "Asegurate de lanzar Chrome con --remote-debugging-port=9222\n"
                "Error: %s", CDP_URL, e)
            raise SystemExit(1)

        contexts = self._browser.contexts
        self._context = contexts[0] if contexts else self._browser.new_context()
        self._page = self._context.new_page()
        logger.info("Conectado a Chrome via CDP (%s)", CDP_URL)
        return self._page

    def fetch_html(self, url: str) -> str | None:
        """Descarga HTML. Usa networkidle porque Stathead puede renderizar con JS."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self._page.goto(url, wait_until="networkidle", timeout=45000)
                if resp and resp.status == 429:
                    wait = RETRY_DELAY * attempt * 2
                    logger.warning("Rate limited (429). Esperando %ds...", wait)
                    time.sleep(wait)
                    continue
                if resp and resp.status >= 400:
                    if attempt < MAX_RETRIES:
                        wait = RETRY_DELAY * attempt
                        logger.warning("HTTP %d. Reintento en %ds...", resp.status, wait)
                        time.sleep(wait)
                        continue
                    logger.warning("HTTP %d tras %d intentos: %s", resp.status, MAX_RETRIES, url)
                    return None

                html = self._page.content()

                # Detectar logout de Stathead
                if self._is_logged_out(html):
                    logger.error(
                        "Sesión de Stathead expirada. Vuelve a loguear en Chrome y re-ejecuta.")
                    raise SystemExit(1)

                # Detectar bloqueo/CAPTCHA
                if self._is_blocked(html):
                    if attempt < MAX_RETRIES:
                        wait = 30 * attempt  # 30s, 60s, 90s
                        logger.warning("Posible CAPTCHA/bloqueo. Esperando %ds...", wait)
                        time.sleep(wait)
                        continue
                    logger.error("Bloqueado por Stathead tras %d intentos.", MAX_RETRIES)
                    return None

                time.sleep(REQUEST_DELAY)
                return html

            except SystemExit:
                raise
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

    def _is_logged_out(self, html: str) -> bool:
        """Detecta si Stathead redirigió a login."""
        snippet = html[:5000]
        return (
            'id="login"' in snippet
            or "stathead.com/users/login" in snippet
            or "Please log in" in snippet
            or "You must be logged in" in snippet
        )

    def _is_blocked(self, html: str) -> bool:
        """Detecta CAPTCHA o página de bloqueo."""
        snippet = html[:3000].lower()
        return (
            "captcha" in snippet
            or "access denied" in snippet
            or (len(html) < 500 and "<table" not in html.lower())
        )

    def close(self):
        if self._page:
            try:
                self._page.close()
            except Exception:
                pass
        self._pw.stop()


# ------------------------------------------------------------------ #
# HTML Parsers
# ------------------------------------------------------------------ #
class StatheadTableParser(HTMLParser):
    """Parser para tablas de resultados de Stathead.

    Las tablas están directas en el DOM (no en comentarios como BRef).
    La tabla de resultados usa id que contiene "results".
    Filas con class="thead" son separadores (repetición de headers) → se ignoran.
    Captura data-stat de cada celda y href de links a jugadores/equipos.
    """

    def __init__(self):
        super().__init__()
        self._in_table = False
        self._in_thead = False
        self._in_tbody = False
        self._in_tfoot = False
        self._in_cell = False
        self._skip_row = False
        self._cell_data: list[str] = []
        self._cell_attrs: dict = {}
        self._current_row: list[dict] = []
        self._current_link: str | None = None

        self.headers: list[str] = []
        self.rows: list[list[dict]] = []
        self._header_done = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table":
            tid = a.get("id", "")
            # Stathead usa IDs como "results", "stats", etc.
            if "results" in tid or "stats" in tid or tid.startswith("result"):
                self._in_table = True
            return
        if not self._in_table:
            return

        if tag == "thead":
            self._in_thead = True
        elif tag == "tbody":
            self._in_thead = False
            self._in_tbody = True
        elif tag == "tfoot":
            self._in_tfoot = True
        elif tag == "tr":
            cls = a.get("class", "")
            # Separadores de header repetido dentro de tbody
            self._skip_row = "thead" in cls or "partial_table" in cls
            self._current_row = []
            self._current_link = None
        elif tag in ("th", "td"):
            if (self._in_thead or self._in_tbody) and not self._skip_row:
                self._in_cell = True
                self._cell_data = []
                self._cell_attrs = a
                self._current_link = None
        elif tag == "a" and self._in_cell:
            href = a.get("href", "")
            # Capturar links de jugadores (/players/) y equipos (/teams/)
            if "/players/" in href or "/teams/" in href:
                self._current_link = href

    def handle_endtag(self, tag):
        if not self._in_table:
            return
        if tag == "table":
            self._in_table = False
        elif tag == "thead":
            self._in_thead = False
            self._header_done = True
        elif tag == "tbody":
            self._in_tbody = False
        elif tag == "tfoot":
            self._in_tfoot = False
        elif tag == "tr":
            if self._current_row and self._in_tbody and not self._skip_row:
                self.rows.append(self._current_row)
            self._current_row = []
            self._skip_row = False
        elif tag in ("th", "td") and self._in_cell:
            text = "".join(self._cell_data).strip()
            ds = self._cell_attrs.get("data-stat", "")
            if self._in_thead and not self._header_done:
                self.headers.append(ds)
            elif self._in_tbody and not self._skip_row:
                cell = {"text": text, "data_stat": ds}
                if self._current_link:
                    cell["link"] = self._current_link
                self._current_row.append(cell)
            self._in_cell = False
            self._current_link = None

    def handle_data(self, data):
        if self._in_cell:
            self._cell_data.append(data)


class ResultsCountParser(HTMLParser):
    """Extrae total de resultados de la página.

    Stathead muestra "Showing 1-200 of 36,748 results" o similar.
    """

    def __init__(self):
        super().__init__()
        self.total: int | None = None
        self._capture = False
        self._text: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        # Buscar divs/spans con info de resultados
        cid = a.get("id", "")
        cls = a.get("class", "")
        if "result" in cid.lower() or "result" in cls.lower() or "paging" in cls.lower():
            self._capture = True
            self._text = []

    def handle_endtag(self, tag):
        if self._capture and tag in ("div", "span", "p"):
            text = "".join(self._text)
            m = re.search(r"of\s+([\d,]+)\s+result", text, re.IGNORECASE)
            if m:
                self.total = int(m.group(1).replace(",", ""))
            self._capture = False

    def handle_data(self, data):
        if self._capture:
            self._text.append(data)


# ------------------------------------------------------------------ #
# Numeric helpers
# ------------------------------------------------------------------ #
def _pf(val: str) -> float | None:
    """Parse float, None si vacío o NaN."""
    if not val or val in ("", "\u2014", "-", "—"):
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (ValueError, TypeError):
        return None


def _pi(val: str) -> int | None:
    """Parse int, None si vacío o NaN."""
    f = _pf(val)
    return None if f is None else int(f)


def _parse_mp(mp_str: str) -> float | None:
    """Convierte '32:45' → 32.75. None si vacío/DNP."""
    if not mp_str or mp_str in ("Did Not Play", "Did Not Dress", "", "\u2014"):
        return None
    try:
        parts = mp_str.split(":")
        return int(parts[0]) + int(parts[1]) / 60 if len(parts) == 2 else float(mp_str)
    except (ValueError, IndexError):
        return _pf(mp_str)


def _parse_result(result_str: str) -> tuple[int | None, int | None]:
    """Parsea resultado de Stathead.

    Formatos posibles:
      'W, 162-109'         → (1, 53)
      'L, 139-140 (2OT)'  → (0, -1)
      'W (+15)'            → (1, 15)    (formato alternativo)
      'W'                  → (1, None)  (QF sin scores)
    """
    if not result_str:
        return None, None
    s = result_str.strip()
    # Formato Stathead principal: "W, 162-109" o "L, 139-140 (2OT)"
    m = re.match(r"([WL]),?\s*(\d+)-(\d+)", s)
    if m:
        win = 1 if m.group(1) == "W" else 0
        pts1, pts2 = int(m.group(2)), int(m.group(3))
        margin = pts1 - pts2
        return win, margin
    # Formato alternativo: "W (+15)"
    m2 = re.match(r"([WL])\s*\(([+-]?\d+)\)", s)
    if m2:
        win = 1 if m2.group(1) == "W" else 0
        return win, int(m2.group(2))
    # Solo W/L sin score
    if s in ("W", "L"):
        return (1 if s == "W" else 0), None
    return None, None


# ------------------------------------------------------------------ #
# DB helpers
# ------------------------------------------------------------------ #
def _create_all_tables(con: sqlite3.Connection):
    """Crea tablas base: scrape_progress y quarter_finder."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS scrape_progress (
            phase       TEXT NOT NULL,
            season      TEXT NOT NULL,
            sub_key     TEXT NOT NULL DEFAULT '',
            last_offset INTEGER NOT NULL DEFAULT 0,
            total_rows  INTEGER,
            rows_saved  INTEGER NOT NULL DEFAULT 0,
            completed   INTEGER NOT NULL DEFAULT 0,
            updated_at  TEXT NOT NULL,
            PRIMARY KEY (phase, season, sub_key)
        )
    """)
    # Quarter finder: por juego (cada fila = 1 jugador, 1 cuarto, 1 partido)
    con.execute("""
        CREATE TABLE IF NOT EXISTS quarter_finder (
            season       TEXT NOT NULL,
            quarter      TEXT NOT NULL,
            player_name  TEXT NOT NULL,
            player_link  TEXT,
            date_game    TEXT NOT NULL,
            pos          TEXT,
            team_id      TEXT,
            is_away      INTEGER,
            opp_id       TEXT,
            game_result  TEXT,
            win          INTEGER,
            is_starter   INTEGER,
            mp           INTEGER,
            fg           INTEGER,
            fga          INTEGER,
            fg_pct       REAL,
            fg2          INTEGER,
            fg2a         INTEGER,
            fg2_pct      REAL,
            fg3          INTEGER,
            fg3a         INTEGER,
            fg3_pct      REAL,
            ft           INTEGER,
            fta          INTEGER,
            ft_pct       REAL,
            orb          INTEGER,
            drb          INTEGER,
            trb          INTEGER,
            ast          INTEGER,
            stl          INTEGER,
            blk          INTEGER,
            tov          INTEGER,
            pf           INTEGER,
            pts          INTEGER,
            PRIMARY KEY (player_link, date_game, quarter)
        )
    """)
    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_qf_player_season
        ON quarter_finder (player_link, season)
    """)
    con.commit()


def _create_pgf_table(con: sqlite3.Connection, season: str):
    """Crea tabla player_game_finder_{season}."""
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS "player_game_finder_{season}" (
            player_name  TEXT NOT NULL,
            player_link  TEXT,
            age          TEXT,
            team_id      TEXT NOT NULL,
            is_away      INTEGER,
            opp_id       TEXT,
            game_result  TEXT,
            win          INTEGER,
            margin       INTEGER,
            is_starter   INTEGER,
            date_game    TEXT NOT NULL,
            mp           TEXT,
            mp_dec       REAL,
            fg           INTEGER,
            fga          INTEGER,
            fg_pct       REAL,
            fg3          INTEGER,
            fg3a         INTEGER,
            fg3_pct      REAL,
            ft           INTEGER,
            fta          INTEGER,
            ft_pct       REAL,
            orb          INTEGER,
            drb          INTEGER,
            trb          INTEGER,
            ast          INTEGER,
            stl          INTEGER,
            blk          INTEGER,
            tov          INTEGER,
            pf           INTEGER,
            pts          INTEGER,
            game_score   REAL,
            plus_minus   INTEGER,
            PRIMARY KEY (player_link, date_game, team_id)
        )
    """)
    con.execute(f"""
        CREATE INDEX IF NOT EXISTS "idx_pgf_{season}_date"
        ON "player_game_finder_{season}" (date_game)
    """)
    con.commit()


def _create_tgf_table(con: sqlite3.Connection, season: str):
    """Crea tabla team_game_finder_{season}."""
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS "team_game_finder_{season}" (
            team_id      TEXT NOT NULL,
            date_game    TEXT NOT NULL,
            is_away      INTEGER,
            opp_id       TEXT,
            game_result  TEXT,
            win          INTEGER,
            margin       INTEGER,
            fg           INTEGER,
            fga          INTEGER,
            fg_pct       REAL,
            fg2          INTEGER,
            fg2a         INTEGER,
            fg2_pct      REAL,
            fg3          INTEGER,
            fg3a         INTEGER,
            fg3_pct      REAL,
            ft           INTEGER,
            fta          INTEGER,
            ft_pct       REAL,
            pts          INTEGER,
            opp_fg       INTEGER,
            opp_fga      INTEGER,
            opp_fg_pct   REAL,
            opp_fg3      INTEGER,
            opp_fg3a     INTEGER,
            opp_fg3_pct  REAL,
            opp_ft       INTEGER,
            opp_fta      INTEGER,
            opp_ft_pct   REAL,
            pts_opp      INTEGER,
            PRIMARY KEY (team_id, date_game)
        )
    """)
    con.execute(f"""
        CREATE INDEX IF NOT EXISTS "idx_tgf_{season}_date"
        ON "team_game_finder_{season}" (date_game)
    """)
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


# ------------------------------------------------------------------ #
# Progreso / Resumabilidad
# ------------------------------------------------------------------ #
def _get_last_offset(con: sqlite3.Connection, phase: str, season: str,
                     sub_key: str = "") -> int:
    """Retorna último offset procesado. -1 si ya completó."""
    row = con.execute(
        "SELECT last_offset, completed FROM scrape_progress "
        "WHERE phase=? AND season=? AND sub_key=?",
        (phase, season, sub_key)
    ).fetchone()
    if row is None:
        return 0
    if row[1] == 1:  # completed
        return -1
    return row[0]


def _save_progress(con: sqlite3.Connection, phase: str, season: str,
                   sub_key: str, offset: int, total_rows: int | None,
                   rows_saved: int, completed: int = 0):
    con.execute("""
        INSERT OR REPLACE INTO scrape_progress
        (phase, season, sub_key, last_offset, total_rows, rows_saved, completed, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
    """, (phase, season, sub_key, offset, total_rows, rows_saved, completed))
    con.commit()


# ------------------------------------------------------------------ #
# URL builders
# ------------------------------------------------------------------ #
def _build_qf_url(year: int, quarter: str, offset: int) -> str:
    """URL para Player Quarter Finder."""
    return (
        f"{STATHEAD_BASE}/basketball/quarter_finder.cgi"
        f"?request=1&lg_id=NBA"
        f"&year_min={year}&year_max={year}"
        f"&quarter={quarter}"
        f"&is_playoffs=N"
        f"&offset={offset}"
    )


def _build_pgf_url(year: int, offset: int) -> str:
    """URL para Player Game Finder."""
    return (
        f"{STATHEAD_BASE}/basketball/player-game-finder.cgi"
        f"?request=1&lg_id=NBA"
        f"&year_min={year}&year_max={year}"
        f"&is_playoffs=N"
        f"&offset={offset}"
    )


def _build_tgf_url(year: int, offset: int) -> str:
    """URL para Team Game Finder."""
    return (
        f"{STATHEAD_BASE}/basketball/team-game-finder.cgi"
        f"?request=1&lg_id=NBA"
        f"&year_min={year}&year_max={year}"
        f"&comp_type=reg"
        f"&is_playoffs=N"
        f"&offset={offset}"
    )


# ------------------------------------------------------------------ #
# Row parsers — convierten filas del HTML a dicts para DB
# ------------------------------------------------------------------ #
def _cells_to_dict(cells: list[dict]) -> dict[str, str]:
    """Convierte lista de celdas [{text, data_stat, link?}] a {stat: text}."""
    d: dict[str, str] = {}
    for c in cells:
        ds = c.get("data_stat", "")
        if ds:
            d[ds] = c["text"]
            if "link" in c:
                d[f"{ds}_link"] = c["link"]
    return d


def _parse_qf_rows(rows: list[list[dict]], season: str,
                    quarter: str) -> list[dict]:
    """Parsea filas de Quarter Finder a dicts para DB.

    data-stat reales de Stathead QF (por juego, no agregado):
      player, date_game, pos, quarter, team_id, game_location,
      opp_id, game_result (W/L), gs (1=starter),
      mp, fg/fga/fg_pct, fg2/..., fg3/..., ft/...,
      orb, drb, trb, ast, stl, blk, tov, pf, pts
    """
    results = []
    for cells in rows:
        d = _cells_to_dict(cells)
        player = d.get("player", "").strip()
        date = d.get("date_game", "").strip()
        if not player or not date or player in ("Player", ""):
            continue

        result_str = d.get("game_result", "")
        win, _ = _parse_result(result_str)

        loc = d.get("game_location", "").strip()
        is_away = 1 if loc == "@" else 0

        gs = d.get("gs", "").strip()
        is_starter = 1 if gs in ("1", "*") else 0

        results.append({
            "season": season,
            "quarter": quarter,
            "player_name": player,
            "player_link": d.get("player_link", ""),
            "date_game": date,
            "pos": d.get("pos", ""),
            "team_id": d.get("team_id", ""),
            "is_away": is_away,
            "opp_id": d.get("opp_id", ""),
            "game_result": result_str,
            "win": win,
            "is_starter": is_starter,
            "mp": _pi(d.get("mp", "")),
            "fg": _pi(d.get("fg", "")),
            "fga": _pi(d.get("fga", "")),
            "fg_pct": _pf(d.get("fg_pct", "")),
            "fg2": _pi(d.get("fg2", "")),
            "fg2a": _pi(d.get("fg2a", "")),
            "fg2_pct": _pf(d.get("fg2_pct", "")),
            "fg3": _pi(d.get("fg3", "")),
            "fg3a": _pi(d.get("fg3a", "")),
            "fg3_pct": _pf(d.get("fg3_pct", "")),
            "ft": _pi(d.get("ft", "")),
            "fta": _pi(d.get("fta", "")),
            "ft_pct": _pf(d.get("ft_pct", "")),
            "orb": _pi(d.get("orb", "")),
            "drb": _pi(d.get("drb", "")),
            "trb": _pi(d.get("trb", "")),
            "ast": _pi(d.get("ast", "")),
            "stl": _pi(d.get("stl", "")),
            "blk": _pi(d.get("blk", "")),
            "tov": _pi(d.get("tov", "")),
            "pf": _pi(d.get("pf", "")),
            "pts": _pi(d.get("pts", "")),
        })
    return results


def _parse_pgf_rows(rows: list[list[dict]], season: str) -> list[dict]:
    """Parsea filas de Player Game Finder a dicts para DB.

    data-stat reales de Stathead PGF:
      name_display, pts, date, age_on_day, team_name_abbr, game_location,
      opp_name_abbr, game_result, is_starter (*=starter), mp,
      fg/fga/fg_pct, fg2/fg2a/fg2_pct, fg3/fg3a/fg3_pct, ft/fta/ft_pct,
      ts_pct, orb, drb, trb, ast, stl, blk, tov, pf, pts,
      game_score, bpm, plus_minus, pos_game
    """
    results = []
    for cells in rows:
        d = _cells_to_dict(cells)
        player = d.get("name_display", "").strip()
        date = d.get("date", "").strip()
        if not player or not date or player in ("Player", "Name"):
            continue

        result_str = d.get("game_result", "")
        win, margin = _parse_result(result_str)

        loc = d.get("game_location", "").strip()
        is_away = 1 if loc == "@" else 0

        # is_starter: "*" = starter, vacío = bench
        gs = d.get("is_starter", "").strip()
        is_starter = 1 if gs == "*" else 0

        mp_raw = d.get("mp", "")

        results.append({
            "player_name": player,
            "player_link": d.get("name_display_link", ""),
            "age": d.get("age_on_day", ""),
            "team_id": d.get("team_name_abbr", ""),
            "is_away": is_away,
            "opp_id": d.get("opp_name_abbr", ""),
            "game_result": result_str,
            "win": win,
            "margin": margin,
            "is_starter": is_starter,
            "date_game": date,
            "mp": mp_raw,
            "mp_dec": _parse_mp(mp_raw),
            "fg": _pi(d.get("fg", "")),
            "fga": _pi(d.get("fga", "")),
            "fg_pct": _pf(d.get("fg_pct", "")),
            "fg3": _pi(d.get("fg3", "")),
            "fg3a": _pi(d.get("fg3a", "")),
            "fg3_pct": _pf(d.get("fg3_pct", "")),
            "ft": _pi(d.get("ft", "")),
            "fta": _pi(d.get("fta", "")),
            "ft_pct": _pf(d.get("ft_pct", "")),
            "orb": _pi(d.get("orb", "")),
            "drb": _pi(d.get("drb", "")),
            "trb": _pi(d.get("trb", "")),
            "ast": _pi(d.get("ast", "")),
            "stl": _pi(d.get("stl", "")),
            "blk": _pi(d.get("blk", "")),
            "tov": _pi(d.get("tov", "")),
            "pf": _pi(d.get("pf", "")),
            "pts": _pi(d.get("pts", "")),
            "game_score": _pf(d.get("game_score", "")),
            "plus_minus": _pi(d.get("plus_minus", "")),
        })
    return results


def _parse_tgf_rows(rows: list[list[dict]], season: str) -> list[dict]:
    """Parsea filas de Team Game Finder a dicts para DB.

    data-stat reales de Stathead TGF:
      team_name_abbr, date, pts, game_location, opp_name_abbr, game_result,
      mp, fg/fga/fg_pct, fg2/fg2a/fg2_pct, fg3/fg3a/fg3_pct, ft/fta/ft_pct,
      opp_fg/opp_fga/opp_fg_pct, opp_fg2/..., opp_fg3/..., opp_ft/..., opp_pts
    Nota: NO incluye ORtg/DRtg/Pace/eFG%/TS% (esos vienen de Four Factors).
    """
    results = []
    for cells in rows:
        d = _cells_to_dict(cells)
        team = d.get("team_name_abbr", "").strip()
        date = d.get("date", "").strip()
        if not team or not date or team == "Team":
            continue

        result_str = d.get("game_result", "")
        win, margin = _parse_result(result_str)

        loc = d.get("game_location", "").strip()
        is_away = 1 if loc == "@" else 0

        results.append({
            "team_id": team,
            "date_game": date,
            "is_away": is_away,
            "opp_id": d.get("opp_name_abbr", ""),
            "game_result": result_str,
            "win": win,
            "margin": margin,
            "fg": _pi(d.get("fg", "")),
            "fga": _pi(d.get("fga", "")),
            "fg_pct": _pf(d.get("fg_pct", "")),
            "fg2": _pi(d.get("fg2", "")),
            "fg2a": _pi(d.get("fg2a", "")),
            "fg2_pct": _pf(d.get("fg2_pct", "")),
            "fg3": _pi(d.get("fg3", "")),
            "fg3a": _pi(d.get("fg3a", "")),
            "fg3_pct": _pf(d.get("fg3_pct", "")),
            "ft": _pi(d.get("ft", "")),
            "fta": _pi(d.get("fta", "")),
            "ft_pct": _pf(d.get("ft_pct", "")),
            "pts": _pi(d.get("pts", "")),
            "opp_fg": _pi(d.get("opp_fg", "")),
            "opp_fga": _pi(d.get("opp_fga", "")),
            "opp_fg_pct": _pf(d.get("opp_fg_pct", "")),
            "opp_fg3": _pi(d.get("opp_fg3", "")),
            "opp_fg3a": _pi(d.get("opp_fg3a", "")),
            "opp_fg3_pct": _pf(d.get("opp_fg3_pct", "")),
            "opp_ft": _pi(d.get("opp_ft", "")),
            "opp_fta": _pi(d.get("opp_fta", "")),
            "opp_ft_pct": _pf(d.get("opp_ft_pct", "")),
            "pts_opp": _pi(d.get("opp_pts", "")),
        })
    return results


# ------------------------------------------------------------------ #
# Paginación genérica
# ------------------------------------------------------------------ #
def _paginate(
    browser: StatheadBrowser,
    con: sqlite3.Connection,
    phase: str,
    season: str,
    sub_key: str,
    build_url_fn,
    parse_rows_fn,
    table_name: str,
    year: int,
    start_offset: int = 0,
    max_pages: int | None = None,
) -> int:
    """Loop de paginación genérico para todos los finders de Stathead.

    Retorna número de filas guardadas.
    """
    rows_saved = 0
    offset = start_offset
    total_rows = None
    pages_done = 0

    while True:
        if should_stop():
            logger.info("[%s/%s] Shutdown en offset=%d", season, sub_key or phase, offset)
            break

        if max_pages is not None and pages_done >= max_pages:
            logger.info("[%s/%s] Límite de %d páginas alcanzado.", season, sub_key or phase, max_pages)
            break

        url = build_url_fn(year, offset)
        logger.debug("[%s/%s] Pag offset=%d", season, sub_key or phase, offset)

        html = browser.fetch_html(url)
        if html is None:
            logger.warning("[%s/%s] Error al obtener offset=%d. Saltando.", season, sub_key or phase, offset)
            break

        # Extraer total de resultados (solo primera página útil)
        if offset == start_offset:
            rcp = ResultsCountParser()
            rcp.feed(html)
            total_rows = rcp.total
            if total_rows is not None:
                total_pages = (total_rows + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE
                logger.info("[%s/%s] Total: %s filas = %d páginas",
                           season, sub_key or phase, f"{total_rows:,}", total_pages)
            else:
                logger.info("[%s/%s] Total de resultados no detectado (continuando por páginas)",
                           season, sub_key or phase)

        # Parsear tabla
        parser = StatheadTableParser()
        parser.feed(html)

        if not parser.rows:
            # Verificar si es un falso vacío (total_rows > 0 pero 0 filas parseadas)
            if total_rows and total_rows > 0 and offset == start_offset:
                logger.warning(
                    "[%s/%s] 0 filas parseadas pero total=%d. "
                    "¿Cambió el HTML de Stathead? Primeros 2000 chars:\n%s",
                    season, sub_key or phase, total_rows, html[:2000])
            else:
                logger.info("[%s/%s] 0 filas en offset=%d — fin de datos.",
                           season, sub_key or phase, offset)
            _save_progress(con, phase, season, sub_key, offset, total_rows, rows_saved, completed=1)
            break

        # Log headers en primera página (para debug de data-stat names)
        if offset == start_offset and parser.headers:
            logger.debug("[%s/%s] Headers: %s", season, sub_key or phase, parser.headers)

        rows = parse_rows_fn(parser.rows)

        if not rows:
            logger.info("[%s/%s] 0 filas válidas en offset=%d.", season, sub_key or phase, offset)
            _save_progress(con, phase, season, sub_key, offset, total_rows, rows_saved, completed=1)
            break

        _upsert_rows(con, table_name, rows)
        rows_saved += len(rows)
        pages_done += 1
        next_offset = offset + RESULTS_PER_PAGE

        _save_progress(con, phase, season, sub_key, next_offset, total_rows, rows_saved)

        logger.info("[%s/%s] offset=%d: +%d filas (%d acumuladas)",
                   season, sub_key or phase, offset, len(rows), rows_saved)

        # ¿Última página?
        if len(parser.rows) < RESULTS_PER_PAGE:
            _save_progress(con, phase, season, sub_key, next_offset, total_rows, rows_saved, completed=1)
            logger.info("[%s/%s] Completado: %d filas.", season, sub_key or phase, rows_saved)
            break

        offset = next_offset

    return rows_saved


# ------------------------------------------------------------------ #
# PHASE 1: Quarter Finder
# ------------------------------------------------------------------ #
def phase_quarter_finder(seasons: list[str], browser: StatheadBrowser,
                         con: sqlite3.Connection, force: bool = False,
                         max_pages: int | None = None):
    """Scrapea stats por cuarto (Q1-Q4, 1H, 2H) por jugador por temporada."""
    for season in seasons:
        year = season_to_year(season)
        for quarter in QUARTERS:
            if should_stop():
                return

            offset = _get_last_offset(con, "quarter_finder", season, quarter)
            if offset == -1 and not force:
                logger.info("[QF] %s/%s ya completado.", season, quarter)
                continue
            if force:
                offset = 0

            logger.info("[QF] %s / %s (desde offset=%d)", season, quarter, offset)

            def _build(y, q, o):
                return _build_qf_url(y, q, o)

            _paginate(
                browser, con,
                phase="quarter_finder",
                season=season,
                sub_key=quarter,
                build_url_fn=lambda y, o, _q=quarter: _build_qf_url(y, _q, o),
                parse_rows_fn=lambda rows, _s=season, _q=quarter: _parse_qf_rows(rows, _s, _q),
                table_name="quarter_finder",
                year=year,
                start_offset=offset,
                max_pages=max_pages,
            )


# ------------------------------------------------------------------ #
# PHASE 2: Player Game Finder
# ------------------------------------------------------------------ #
def phase_player_game_finder(seasons: list[str], browser: StatheadBrowser,
                             con: sqlite3.Connection, force: bool = False,
                             max_pages: int | None = None):
    """Scrapea game logs de todos los jugadores por temporada."""
    for season in seasons:
        if should_stop():
            return

        year = season_to_year(season)
        table = f"player_game_finder_{season}"
        _create_pgf_table(con, season)

        offset = _get_last_offset(con, "player_game_finder", season)
        if offset == -1 and not force:
            logger.info("[PGF] %s ya completado.", season)
            continue
        if force:
            offset = 0

        remaining = ""
        if offset > 0:
            remaining = f" (retomando desde offset={offset})"
        logger.info("[PGF] %s%s", season, remaining)

        _paginate(
            browser, con,
            phase="player_game_finder",
            season=season,
            sub_key="",
            build_url_fn=lambda y, o: _build_pgf_url(y, o),
            parse_rows_fn=lambda rows, _s=season: _parse_pgf_rows(rows, _s),
            table_name=table,
            year=year,
            start_offset=offset,
            max_pages=max_pages,
        )


# ------------------------------------------------------------------ #
# PHASE 3: Team Game Finder
# ------------------------------------------------------------------ #
def phase_team_game_finder(seasons: list[str], browser: StatheadBrowser,
                           con: sqlite3.Connection, force: bool = False,
                           max_pages: int | None = None):
    """Scrapea game logs de todos los equipos por temporada."""
    for season in seasons:
        if should_stop():
            return

        year = season_to_year(season)
        table = f"team_game_finder_{season}"
        _create_tgf_table(con, season)

        offset = _get_last_offset(con, "team_game_finder", season)
        if offset == -1 and not force:
            logger.info("[TGF] %s ya completado.", season)
            continue
        if force:
            offset = 0

        logger.info("[TGF] %s (desde offset=%d)", season, offset)

        _paginate(
            browser, con,
            phase="team_game_finder",
            season=season,
            sub_key="",
            build_url_fn=lambda y, o: _build_tgf_url(y, o),
            parse_rows_fn=lambda rows, _s=season: _parse_tgf_rows(rows, _s),
            table_name=table,
            year=year,
            start_offset=offset,
            max_pages=max_pages,
        )


# ------------------------------------------------------------------ #
# main
# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser(
        description="Scrapea Stathead: Quarter Finder, Player/Team Game Finder"
    )
    ap.add_argument(
        "--phase",
        choices=["quarter_finder", "player_game_finder", "team_game_finder", "all"],
        required=True,
    )
    ap.add_argument("--season", type=str, default=None,
                    help='Temporada (ej: "2024-25"). Default: todas.')
    ap.add_argument("--force", action="store_true",
                    help="Re-scraping desde cero (ignora progreso guardado)")
    ap.add_argument("--max-pages", type=int, default=None,
                    help="Máximo de páginas por query (para testing)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Solo muestra lo que haría")
    args = ap.parse_args()

    seasons = [args.season] if args.season else ALL_SEASONS

    # Validar temporada
    if args.season and args.season not in ALL_SEASONS:
        logger.error("Temporada '%s' no reconocida. Válidas: %s", args.season, ALL_SEASONS)
        sys.exit(1)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    atexit.register(_cleanup)

    if args.dry_run:
        logger.info("[DRY RUN] Phase: %s | Seasons: %s", args.phase, seasons)
        for s in seasons:
            if args.phase in ("quarter_finder", "all"):
                logger.info("  QF  %s: ~%d queries (6 quarters), ~3-5 pag c/u", s, 6)
            if args.phase in ("player_game_finder", "all"):
                logger.info("  PGF %s: ~185 páginas (~15min @5s)", s)
            if args.phase in ("team_game_finder", "all"):
                logger.info("  TGF %s: ~13 páginas (~1min @5s)", s)
        return

    _start_caffeinate()

    global _active_browser, _active_connection

    con = sqlite3.connect(str(STATHEAD_DB))
    _active_connection = con
    _create_all_tables(con)

    browser = StatheadBrowser()
    browser.start()
    _active_browser = browser

    try:
        phases = (
            ["quarter_finder", "team_game_finder", "player_game_finder"]
            if args.phase == "all"
            else [args.phase]
        )
        for phase in phases:
            if should_stop():
                break
            logger.info("=" * 60)
            logger.info("  FASE: %s", phase.upper())
            logger.info("=" * 60)
            if phase == "quarter_finder":
                phase_quarter_finder(seasons, browser, con, args.force, args.max_pages)
            elif phase == "player_game_finder":
                phase_player_game_finder(seasons, browser, con, args.force, args.max_pages)
            elif phase == "team_game_finder":
                phase_team_game_finder(seasons, browser, con, args.force, args.max_pages)
    finally:
        _cleanup()
        if should_stop():
            logger.info("Scraping interrumpido. Re-ejecuta para continuar desde donde paró.")
        else:
            logger.info("Scraping completado.")


if __name__ == "__main__":
    main()
