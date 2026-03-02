"""collect_park_factors.py — MLB park factors for all 30 ballparks.

Stores hardcoded 3-year average park factors (2022-2024) into
MLBParkFactors.sqlite. Park factors are stable year-to-year and
don't require frequent API calls.

If --fetch is passed, attempts to scrape Baseball Savant for live data
(requires requests + BeautifulSoup or selenium), falling back to the
hardcoded values automatically on failure.

Usage:
    PYTHONPATH=. python scripts/collect_park_factors.py
    PYTHONPATH=. python scripts/collect_park_factors.py --overwrite
    PYTHONPATH=. python scripts/collect_park_factors.py --fetch --year 2024
"""

import argparse
import sqlite3
import sys
from typing import Any

try:
    from src.sports.mlb.config_paths import MLB_PARK_DB, MLB_TRAINING_DIR
except ImportError:
    print("ERROR: run with PYTHONPATH=. from the project root")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Hardcoded 3-year average park factors (2022-2024)
# Source: Baseball Savant / FanGraphs park factors leaderboard
# 100 = league average, >100 = hitter-friendly, <100 = pitcher-friendly
#
# Format: ABBREV → (PF_RUNS, PF_HR, PF_H, PF_1B, PF_2B, PF_3B, PF_BB)
# Abbreviations match MLB_ABBREV in config_mlb.py
# ---------------------------------------------------------------------------
PARK_FACTORS: dict[str, tuple[int, int, int, int, int, int, int]] = {
    #         RUNS   HR    H     1B    2B    3B    BB
    "ARI": (  102,  106,  101,  100,  103,  104,  101),   # Chase Field (retractable)
    "ATL": (  100,  100,  100,  100,  101,   97,  100),   # Truist Park
    "BAL": (  101,  105,  100,  100,  102,   95,  100),   # Camden Yards
    "BOS": (  105,  110,  106,  106,  110,   82,  101),   # Fenway Park (Green Monster)
    "CHC": (  102,  105,  102,  101,  103,   97,  101),   # Wrigley Field (wind-affected)
    "CWS": (  100,  102,  100,  100,  101,   98,  100),   # Guaranteed Rate Field
    "CIN": (  104,  112,  102,  101,  103,   92,  101),   # Great American Ball Park
    "CLE": (   97,   94,   98,   99,   97,   97,   99),   # Progressive Field
    "COL": (  115,  130,  113,  114,  113,  121,  108),   # Coors Field (altitude)
    "DET": (   98,   97,   99,   99,   98,   97,   99),   # Comerica Park (spacious OF)
    "HOU": (   99,   99,   99,   100,   99,   97,   99),   # Minute Maid Park (retractable)
    "KC":  (   99,   97,   99,   100,  100,   99,   99),   # Kauffman Stadium
    "LAA": (   99,  100,  100,  101,   99,   97,  100),   # Angel Stadium
    "LAD": (   98,   98,   99,  100,   98,   95,   99),   # Dodger Stadium
    "MIA": (   96,   93,   97,   98,   96,   94,   97),   # loanDepot Park (retractable dome)
    "MIL": (   99,   98,  100,  100,   99,   97,   99),   # American Family Field (retractable)
    "MIN": (  102,  107,  101,  101,  102,   94,  100),   # Target Field
    "NYM": (   98,   96,   99,   99,   99,   97,   99),   # Citi Field (large OF)
    "NYY": (  102,  108,  100,   99,  101,   93,  101),   # Yankee Stadium (short RF porch)
    "OAK": (   96,   92,   97,   98,   96,   98,   98),   # Oakland Coliseum (large foul territory)
    "PHI": (  101,  103,  101,  101,  101,   97,  100),   # Citizens Bank Park
    "PIT": (   97,   94,   97,   98,   97,  102,   98),   # PNC Park (large OF, but nice 3B gaps)
    "SD":  (   92,   88,   93,   94,   92,   93,   95),   # Petco Park (marine layer)
    "SF":  (   93,   85,   94,   95,   93,   99,   96),   # Oracle Park (marine layer, deep OF)
    "SEA": (   96,   93,   97,   97,   96,   96,   98),   # T-Mobile Park (retractable)
    "STL": (   98,   96,   99,  100,   98,   97,   99),   # Busch Stadium
    "TB":  (   95,   93,   96,   97,   95,   92,   97),   # Tropicana Field (dome)
    "TEX": (  103,  108,  102,  101,  103,   96,  101),   # Globe Life Field (retractable)
    "TOR": (  101,  104,  101,  101,  102,   94,  101),   # Rogers Centre (dome, turf)
    "WSH": (   99,   98,   100,  100,   99,   96,   99),  # Nationals Park
}

SEASON_RANGE = "2022-2024"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS park_factors (
    TEAM          TEXT PRIMARY KEY,
    PF_RUNS       INTEGER,
    PF_HR         INTEGER,
    PF_H          INTEGER,
    PF_1B         INTEGER,
    PF_2B         INTEGER,
    PF_3B         INTEGER,
    PF_BB         INTEGER,
    SEASON_RANGE  TEXT
)
"""

UPSERT_SQL = """
INSERT OR REPLACE INTO park_factors
    (TEAM, PF_RUNS, PF_HR, PF_H, PF_1B, PF_2B, PF_3B, PF_BB, SEASON_RANGE)
VALUES
    (:TEAM, :PF_RUNS, :PF_HR, :PF_H, :PF_1B, :PF_2B, :PF_3B, :PF_BB, :SEASON_RANGE)
"""


# ---------------------------------------------------------------------------
# Optional: Baseball Savant scraper (best-effort, falls back to hardcoded)
# ---------------------------------------------------------------------------
def _fetch_savant_year(year: int) -> dict[str, dict[str, Any]] | None:
    """
    Attempt to pull park factor data from Baseball Savant's statcast leaderboard.
    Returns dict of {abbrev: {PF_RUNS, PF_HR, ...}} or None on failure.
    """
    url = (
        f"https://baseballsavant.mlb.com/leaderboard/statcast-park-factors"
        f"?type=year&year={year}"
    )
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        print("  requests/beautifulsoup4 not available for Savant scraping")
        return None

    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception as exc:
        print(f"  WARNING: Savant request failed: {exc}")
        return None

    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table")
        if table is None:
            print("  WARNING: No table found in Savant response")
            return None

        headers = [th.get_text(strip=True) for th in table.find("thead").find_all("th")]
        results: dict[str, dict[str, Any]] = {}

        for tr in table.find("tbody").find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(cells) < len(headers):
                continue
            row = dict(zip(headers, cells))

            # Attempt to map Savant column names to our schema
            abbrev = row.get("Team", "").strip()
            if not abbrev:
                continue

            def _int(col: str) -> int:
                try:
                    return int(row.get(col, 100) or 100)
                except (ValueError, TypeError):
                    return 100

            results[abbrev] = {
                "PF_RUNS": _int("Basic"),
                "PF_HR":   _int("HR"),
                "PF_H":    _int("H"),
                "PF_1B":   _int("1B"),
                "PF_2B":   _int("2B"),
                "PF_3B":   _int("3B"),
                "PF_BB":   _int("BB"),
            }

        return results if results else None

    except Exception as exc:
        print(f"  WARNING: Savant parse failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Store to SQLite
# ---------------------------------------------------------------------------
def store_park_factors(
    conn: sqlite3.Connection,
    data: dict[str, tuple[int, int, int, int, int, int, int]],
    season_range: str,
) -> int:
    """Insert/replace park factor rows. Returns row count."""
    conn.execute(CREATE_TABLE_SQL)
    inserted = 0
    for abbrev, (pf_runs, pf_hr, pf_h, pf_1b, pf_2b, pf_3b, pf_bb) in data.items():
        rec = {
            "TEAM":         abbrev,
            "PF_RUNS":      pf_runs,
            "PF_HR":        pf_hr,
            "PF_H":         pf_h,
            "PF_1B":        pf_1b,
            "PF_2B":        pf_2b,
            "PF_3B":        pf_3b,
            "PF_BB":        pf_bb,
            "SEASON_RANGE": season_range,
        }
        conn.execute(UPSERT_SQL, rec)
        inserted += 1
    conn.commit()
    return inserted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Store MLB park factors into MLBParkFactors.sqlite"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Drop and recreate the park_factors table before inserting",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Attempt to scrape Baseball Savant for live park factor data",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        metavar="YYYY",
        help="Year to fetch from Baseball Savant when --fetch is used (default: 2024)",
    )
    args = parser.parse_args()

    MLB_TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Park factors → {MLB_PARK_DB}")

    conn = sqlite3.connect(MLB_PARK_DB)

    if args.overwrite:
        conn.execute("DROP TABLE IF EXISTS park_factors")
        conn.commit()
        print("Dropped existing park_factors table")

    # Check existing rows
    conn.execute(CREATE_TABLE_SQL)
    existing = conn.execute("SELECT COUNT(*) FROM park_factors").fetchone()[0]
    if not args.overwrite and existing > 0:
        print(f"Table already has {existing} rows. Use --overwrite to refresh.")
        conn.close()
        return

    # Try fetching live data from Baseball Savant if requested
    live_data: dict[str, dict] | None = None
    if args.fetch:
        print(f"Attempting to fetch {args.year} park factors from Baseball Savant...")
        live_data = _fetch_savant_year(args.year)
        if live_data:
            print(f"  Got {len(live_data)} park factor records from Savant")

    if live_data and len(live_data) >= 28:
        # Convert Savant format to our tuple format
        converted: dict[str, tuple[int, int, int, int, int, int, int]] = {}
        for abbrev, factors in live_data.items():
            converted[abbrev] = (
                factors["PF_RUNS"],
                factors["PF_HR"],
                factors["PF_H"],
                factors["PF_1B"],
                factors["PF_2B"],
                factors["PF_3B"],
                factors["PF_BB"],
            )
        season_range = str(args.year)
        source = "Baseball Savant (scraped)"
    else:
        if args.fetch:
            print("  Savant scrape incomplete or failed — falling back to hardcoded values")
        converted = PARK_FACTORS
        season_range = SEASON_RANGE
        source = "hardcoded (2022-2024 avg)"

    n = store_park_factors(conn, converted, season_range)
    conn.close()

    print(f"\nDone. {n} park factor rows stored (source: {source})")
    print(f"Season range: {season_range}")
    print(f"Database: {MLB_PARK_DB}")
    print()
    print("Park factor highlights (100 = avg, >100 = hitter-friendly):")
    highlights = {
        "COL": "Coors Field     ",
        "BOS": "Fenway Park     ",
        "CIN": "GABP            ",
        "TEX": "Globe Life Field",
        "NYY": "Yankee Stadium  ",
        "SD":  "Petco Park      ",
        "SF":  "Oracle Park     ",
        "TB":  "Tropicana Field ",
    }
    for abbrev, name in highlights.items():
        if abbrev in converted:
            pf = converted[abbrev]
            print(f"  {name} ({abbrev}): Runs={pf[0]}, HR={pf[1]}, H={pf[2]}")


if __name__ == "__main__":
    main()
