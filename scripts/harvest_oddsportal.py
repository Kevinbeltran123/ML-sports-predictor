"""
Scrapes historical NBA opening + closing moneyline odds from OddsPortal.

Almacena en data/training/OddsPortalHistory.sqlite:
  tabla 'odds_history': game_date, home_team, away_team, bookmaker,
                        ml_home_open, ml_away_open, ml_home_close, ml_away_close,
                        open_ts, close_ts

Uso:
  python scripts/harvest_oddsportal.py --seasons 2022-2023 2023-2024
  python scripts/harvest_oddsportal.py --seasons 2024-2025 --bookmaker bet365
  python scripts/harvest_oddsportal.py --seasons 2024-2025 --results-only   # solo cierre, sin ir a detalle
"""

import argparse
import asyncio
import json
import logging
import re
import sqlite3
import time
from pathlib import Path

from playwright.async_api import async_playwright

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_URL = "https://www.oddsportal.com"
DB_PATH = Path(__file__).parent.parent / "data" / "training" / "OddsPortalHistory.sqlite"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
PAGE_DELAY_S = 4       # espera tras cargar página de resultados
MATCH_DELAY_S = 3      # espera tras cargar página de partido

# Patrones de URL que suelen contener datos de odds en OddsPortal
AJAX_ODD_PATTERNS = [
    "feed/match-page",
    "ajax-event-header",
    "ajax-sport-country",
    "/x/feed/",
    "betoffers",
    "odds-history",
    "openingodds",
]

log = logging.getLogger("harvest")


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------
def init_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS odds_history (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            season        TEXT NOT NULL,
            game_date     TEXT NOT NULL,
            home_team     TEXT NOT NULL,
            away_team     TEXT NOT NULL,
            bookmaker     TEXT NOT NULL,
            ml_home_close INTEGER,
            ml_away_close INTEGER,
            ml_home_open  INTEGER,
            ml_away_open  INTEGER,
            open_ts       TEXT,
            close_ts      TEXT,
            scraped_at    TEXT DEFAULT (datetime('now')),
            UNIQUE(season, game_date, home_team, away_team, bookmaker)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS scraped_matches (
            match_url TEXT PRIMARY KEY,
            season    TEXT NOT NULL,
            scraped_at TEXT DEFAULT (datetime('now'))
        )
    """)
    con.commit()
    return con


def is_already_scraped(con: sqlite3.Connection, match_url: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM scraped_matches WHERE match_url = ?", (match_url,)
    ).fetchone()
    return row is not None


def mark_scraped(con: sqlite3.Connection, match_url: str, season: str):
    con.execute(
        "INSERT OR IGNORE INTO scraped_matches (match_url, season) VALUES (?,?)",
        (match_url, season),
    )
    con.commit()


def save_odds(con: sqlite3.Connection, row: dict):
    con.execute(
        """INSERT OR REPLACE INTO odds_history
           (season, game_date, home_team, away_team, bookmaker,
            ml_home_close, ml_away_close, ml_home_open, ml_away_open,
            open_ts, close_ts)
           VALUES (:season,:game_date,:home_team,:away_team,:bookmaker,
                   :ml_home_close,:ml_away_close,:ml_home_open,:ml_away_open,
                   :open_ts,:close_ts)""",
        row,
    )
    con.commit()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def parse_american(text: str) -> int | None:
    """Convierte texto de cuota americana a int. Acepta '-263', '+225', '225'."""
    if text is None:
        return None
    text = str(text).strip().replace(",", "")
    try:
        return int(text)
    except ValueError:
        return None


def _to_int(val) -> int | None:
    """Convierte cualquier representación de odd (int/float/str) a int."""
    if val is None:
        return None
    try:
        # Las odds europeas/decimales suelen ser float; americanas son int
        s = str(val).strip().replace(",", "")
        return int(s)
    except (ValueError, TypeError):
        return None


def _deep_search_odds(obj, depth: int = 0) -> list[dict]:
    """
    Búsqueda recursiva en JSON de OddsPortal para encontrar entradas de bookmaker.
    Retorna lista de dicts con bookmaker y odds.
    """
    results = []
    if depth > 10 or obj is None:
        return results

    if isinstance(obj, dict):
        bookie = (
            obj.get("bookName")
            or obj.get("name")
            or obj.get("bookie")
        )
        if bookie and isinstance(bookie, str) and len(bookie) > 1:
            entry = _extract_odds_entry(obj, bookie)
            if entry:
                results.append(entry)
                return results  # no seguir si ya lo parseamos

        for v in obj.values():
            results.extend(_deep_search_odds(v, depth + 1))

    elif isinstance(obj, list):
        for item in obj:
            results.extend(_deep_search_odds(item, depth + 1))

    return results


def _extract_odds_entry(data: dict, bookie: str) -> dict | None:
    """Extrae campos de odds de un dict que sabemos contiene un bookmaker."""
    close_home = close_away = open_home = open_away = None
    open_ts = close_ts = None

    # --- Cierre ---
    # Formato: {"odds": {"0": "-263", "1": "+221"}}  o lista
    odds = data.get("odds") or data.get("oddsTxt") or data.get("closingOdds") or {}
    if isinstance(odds, dict):
        close_home = _to_int(odds.get("0"))
        close_away = _to_int(odds.get("1"))
    elif isinstance(odds, list) and len(odds) >= 2:
        close_home = _to_int(odds[0])
        close_away = _to_int(odds[1])

    # Fallback campos directos
    if close_home is None:
        close_home = _to_int(
            data.get("closeOdd") or data.get("odd1") or data.get("home_odds")
        )
    if close_away is None:
        close_away = _to_int(
            data.get("closeOdd2") or data.get("odd2") or data.get("away_odds")
        )

    # --- Apertura ---
    open_odds = (
        data.get("openOdds")
        or data.get("openOddsTxt")
        or data.get("opening")
        or data.get("openingOdds")
        or {}
    )
    if isinstance(open_odds, dict):
        open_home = _to_int(open_odds.get("0"))
        open_away = _to_int(open_odds.get("1"))
    elif isinstance(open_odds, list) and len(open_odds) >= 2:
        open_home = _to_int(open_odds[0])
        open_away = _to_int(open_odds[1])

    if open_home is None:
        open_home = _to_int(
            data.get("openOdd") or data.get("openOdd1") or data.get("open_home")
        )
    if open_away is None:
        open_away = _to_int(
            data.get("openOdd2") or data.get("open_away")
        )

    # --- Timestamps ---
    open_ts = data.get("openDate") or data.get("open_ts") or data.get("openingDate")
    close_ts = data.get("closeDate") or data.get("close_ts") or data.get("closingDate")

    # Descartar si no hay ningún dato útil
    if all(v is None for v in [close_home, close_away, open_home, open_away]):
        return None

    return {
        "bookmaker": bookie,
        "ml_home_close": close_home,
        "ml_away_close": close_away,
        "ml_home_open": open_home,
        "ml_away_open": open_away,
        "open_ts": str(open_ts) if open_ts else None,
        "close_ts": str(close_ts) if close_ts else None,
    }


def _parse_oddsportal_responses(responses: list[dict], target_bookie: str | None) -> list[dict]:
    """
    Intenta parsear todas las respuestas AJAX capturadas y retorna odds por bookmaker.
    Maneja múltiples estructuras conocidas de OddsPortal.
    """
    all_entries: list[dict] = []

    for resp in responses:
        url = resp["url"]
        data = resp["data"]

        # Estructura 1: d.oddsdata.back.{key}: {bookName, odds, openOdds, ...}
        oddsdata = None
        try:
            oddsdata = data["d"]["oddsdata"]["back"]
        except (KeyError, TypeError):
            pass

        if oddsdata and isinstance(oddsdata, dict):
            for key, entry in oddsdata.items():
                if not isinstance(entry, dict):
                    continue
                bookie = entry.get("bookName") or entry.get("name") or ""
                if not bookie:
                    continue
                e = _extract_odds_entry(entry, bookie)
                if e:
                    all_entries.append(e)
            continue  # estructura encontrada, no seguir buscando en este response

        # Estructura 2: búsqueda recursiva genérica
        found = _deep_search_odds(data)
        if found:
            log.debug("AJAX genérico → %d bookmakers en %s", len(found), url)
            all_entries.extend(found)

    # Filtrar por bookmaker si se especificó
    if target_bookie:
        all_entries = [
            e for e in all_entries
            if target_bookie.lower() in e["bookmaker"].lower()
        ]

    # Deduplicar por bookmaker (quedarse con la entrada que tenga más datos)
    seen: dict[str, dict] = {}
    for e in all_entries:
        key = e["bookmaker"].lower()
        if key not in seen:
            seen[key] = e
        else:
            prev = seen[key]
            # Preferir la entrada con opening odds
            if e["ml_home_open"] is not None and prev["ml_home_open"] is None:
                seen[key] = e

    return list(seen.values())


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------
async def get_results_page(page, season_url: str, page_num: int) -> list[dict]:
    """Carga una página de resultados y retorna lista de {url, home, away, date}."""
    url = f"{season_url}#/page/{page_num}"
    await page.goto(url, timeout=30000, wait_until="networkidle")
    await asyncio.sleep(PAGE_DELAY_S)

    rows = await page.query_selector_all("div[data-testid='game-row']")
    log.info("  Página %d: %d filas encontradas", page_num, len(rows))

    matches = []
    for row in rows:
        # Link del partido
        link_el = await row.query_selector("a[href*='/basketball/usa/nba-']")
        if not link_el:
            continue
        href = await link_el.get_attribute("href")
        if not href or len(href) < 30:
            continue

        # Equipos vía title
        team_els = await row.query_selector_all("a[title]")
        teams = []
        for el in team_els:
            t = await el.get_attribute("title")
            if t and t not in teams:
                teams.append(t)

        if len(teams) < 2:
            continue

        # Fecha y cuotas de cierre visibles en la fila
        row_text = (await row.inner_text()).replace("\n", " ")
        # Extraer cuotas del texto: patrones +-NNNNN
        ml_vals = re.findall(r"[+-]\d{3,4}", row_text)

        matches.append({
            "url": BASE_URL + href if href.startswith("/") else href,
            "home_team": teams[0],
            "away_team": teams[1],
            "ml_home_close": parse_american(ml_vals[0]) if len(ml_vals) > 0 else None,
            "ml_away_close": parse_american(ml_vals[1]) if len(ml_vals) > 1 else None,
        })

    return matches


async def get_match_opening_odds(
    page,
    match_url: str,
    target_bookie: str | None,
    debug_dir: Path | None = None,
) -> list[dict]:
    """
    Intercepta respuestas AJAX al navegar a la página de partido para extraer
    odds de apertura y cierre. No usa hover (OddsPortal lo detecta).

    Si debug_dir está definido, vuelca todos los JSON capturados en esa carpeta.
    """
    captured: list[dict] = []

    async def on_response(response):
        url = response.url
        # Solo respuestas que parecen contener datos de odds
        if not any(pat in url for pat in AJAX_ODD_PATTERNS):
            return
        ct = response.headers.get("content-type", "")
        if "json" not in ct and "javascript" not in ct:
            return
        try:
            body = await response.json()
            captured.append({"url": url, "data": body})
            log.debug("AJAX capturado: %s", url)
        except Exception as exc:
            log.debug("No se pudo parsear JSON de %s: %s", url, exc)

    page.on("response", on_response)
    try:
        await page.goto(match_url, timeout=30000, wait_until="networkidle")
        await asyncio.sleep(MATCH_DELAY_S)

        # También buscar en __NEXT_DATA__ embebido en el HTML
        next_data = await page.evaluate(
            "() => { try { return JSON.parse(document.getElementById('__NEXT_DATA__').textContent); } catch(e) { return null; } }"
        )
        if next_data:
            captured.append({"url": "__NEXT_DATA__", "data": next_data})
            log.debug("__NEXT_DATA__ capturado (%d bytes)", len(json.dumps(next_data)))

    finally:
        page.remove_listener("response", on_response)

    log.info("  Respuestas AJAX capturadas: %d", len(captured))

    # Dump para debug
    if debug_dir and captured:
        debug_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", match_url.split("/")[-2] or "match")
        dump_path = debug_dir / f"{slug}.json"
        with open(dump_path, "w") as f:
            json.dump(captured, f, indent=2, default=str)
        log.info("  Debug AJAX volcado en: %s", dump_path)

    if not captured:
        log.warning("  Sin datos AJAX en %s — sin opening odds", match_url)
        return []

    results = _parse_oddsportal_responses(captured, target_bookie)
    log.info(
        "  Bookmakers parseados: %d  (con open: %d)",
        len(results),
        sum(1 for r in results if r["ml_home_open"] is not None),
    )
    return results


async def get_total_pages(page, season_url: str) -> int:
    await page.goto(season_url, timeout=30000, wait_until="networkidle")
    await asyncio.sleep(PAGE_DELAY_S)
    links = await page.query_selector_all("a.pagination-link:not([rel='next'])")
    nums = []
    for lnk in links:
        txt = (await lnk.inner_text()).strip()
        if txt.isdigit():
            nums.append(int(txt))
    return max(nums) if nums else 1


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
async def harvest(
    seasons: list[str],
    bookmaker: str | None,
    results_only: bool,
    max_matches: int | None = None,
    debug_ajax: bool = False,
):
    con = init_db(DB_PATH)

    debug_dir = (
        Path(__file__).parent.parent / "data" / "ajax_debug"
        if debug_ajax
        else None
    )

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        ctx = await browser.new_context(user_agent=USER_AGENT)
        page = await ctx.new_page()

        session_count = 0  # partidos procesados en esta ejecución

        for season in seasons:
            # OddsPortal URL format: nba-2024-2025 (current season = nba)
            parts = season.split("-")
            if len(parts) == 2:
                season_slug = f"nba-{season}"
            else:
                season_slug = "nba"

            season_url = f"{BASE_URL}/basketball/usa/{season_slug}/results/"
            log.info("=== Temporada %s → %s ===", season, season_url)

            total_pages = await get_total_pages(page, season_url)
            log.info("Total páginas: %d", total_pages)

            for pg in range(1, total_pages + 1):
                log.info("--- Página %d/%d ---", pg, total_pages)
                try:
                    matches = await get_results_page(page, season_url, pg)
                except Exception as e:
                    log.error("Error en página %d: %s", pg, e)
                    continue

                for m in matches:
                    if is_already_scraped(con, m["url"]):
                        log.debug("Skip (ya procesado): %s", m["url"])
                        continue

                    if max_matches is not None and session_count >= max_matches:
                        log.info("Alcanzado límite de %d partidos en esta sesión. Saliendo.", max_matches)
                        return

                    if results_only:
                        # Solo guardar cierre desde la fila de resultados
                        # Necesitamos la fecha — la extraemos del URL slug
                        save_odds(con, {
                            "season": season,
                            "game_date": "",
                            "home_team": m["home_team"],
                            "away_team": m["away_team"],
                            "bookmaker": "oddsportal_avg",
                            "ml_home_close": m.get("ml_home_close"),
                            "ml_away_close": m.get("ml_away_close"),
                            "ml_home_open": None,
                            "ml_away_open": None,
                            "open_ts": None,
                            "close_ts": None,
                        })
                        mark_scraped(con, m["url"], season)
                        session_count += 1
                        continue

                    # Ir a la página del partido para obtener apertura
                    try:
                        detail_tab = await ctx.new_page()
                        odds_rows = await get_match_opening_odds(
                            detail_tab, m["url"], bookmaker, debug_dir
                        )
                        await detail_tab.close()

                        # Extraer la fecha del URL slug
                        date_match = re.search(
                            r"/nba(?:-\d{4}-\d{4})?/(.+?)-([A-Za-z0-9]{8})/$",
                            m["url"],
                        )
                        game_date = ""  # OddsPortal no incluye fecha en el URL

                        if not odds_rows:
                            # Fallback: guardar solo el cierre de la fila
                            save_odds(con, {
                                "season": season,
                                "game_date": game_date,
                                "home_team": m["home_team"],
                                "away_team": m["away_team"],
                                "bookmaker": bookmaker or "unknown",
                                "ml_home_close": m.get("ml_home_close"),
                                "ml_away_close": m.get("ml_away_close"),
                                "ml_home_open": None,
                                "ml_away_open": None,
                                "open_ts": None,
                                "close_ts": None,
                            })
                        else:
                            for od in odds_rows:
                                save_odds(con, {
                                    "season": season,
                                    "game_date": game_date,
                                    "home_team": m["home_team"],
                                    "away_team": m["away_team"],
                                    "bookmaker": od["bookmaker"],
                                    "ml_home_close": od["ml_home_close"],
                                    "ml_away_close": od["ml_away_close"],
                                    "ml_home_open": od["ml_home_open"],
                                    "ml_away_open": od["ml_away_open"],
                                    "open_ts": od.get("open_ts"),
                                    "close_ts": od.get("close_ts"),
                                })
                                log.info(
                                    "Saved: %s vs %s [%s] open=%s/%s close=%s/%s",
                                    m["home_team"], m["away_team"], od["bookmaker"],
                                    od["ml_home_open"], od["ml_away_open"],
                                    od["ml_home_close"], od["ml_away_close"],
                                )

                        mark_scraped(con, m["url"], season)
                        session_count += 1

                    except Exception as e:
                        log.error("Error en partido %s: %s", m["url"], e)
                        try:
                            await detail_tab.close()
                        except Exception:
                            pass

                    # Pausa anti-detección entre partidos
                    await asyncio.sleep(2)

        await browser.close()

    con.close()
    log.info("Terminado. DB: %s", DB_PATH)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Scrape OddsPortal NBA historical odds")
    parser.add_argument(
        "--seasons", nargs="+", required=True,
        help="Temporadas a scrapear: 2022-2023 2023-2024 2024-2025",
    )
    parser.add_argument(
        "--bookmaker", default=None,
        help="Filtrar por bookmaker (ej: 'bet365', 'Pinnacle'). Por defecto: todos.",
    )
    parser.add_argument(
        "--results-only", action="store_true",
        help="Solo guardar cuotas de cierre desde la página de resultados (más rápido).",
    )
    parser.add_argument(
        "--max-matches", type=int, default=None,
        help="Límite de partidos a procesar (útil para pruebas).",
    )
    parser.add_argument(
        "--debug-ajax", action="store_true",
        help="Vuelca todas las respuestas AJAX JSON capturadas en data/ajax_debug/ para inspección.",
    )
    args = parser.parse_args()

    asyncio.run(harvest(
        args.seasons, args.bookmaker, args.results_only,
        args.max_matches, args.debug_ajax,
    ))


if __name__ == "__main__":
    main()
