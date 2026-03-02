"""collect_pitcher_stats.py — Pitcher season stats from FanGraphs via pybaseball.

Stores one table per season in MLBPitcherData.sqlite:
  - `pitcher_stats_YYYY`   — qualified starters (GS >= 1)
  - `reliever_stats_YYYY`  — frequent relievers (GS=0, G >= 10)

Usage:
    PYTHONPATH=. python scripts/collect_pitcher_stats.py
    PYTHONPATH=. python scripts/collect_pitcher_stats.py --seasons 2024 2025
    PYTHONPATH=. python scripts/collect_pitcher_stats.py --seasons 2024 --overwrite
    PYTHONPATH=. python scripts/collect_pitcher_stats.py --no-cache
"""

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

try:
    from pybaseball import pitching_stats
    from pybaseball import cache as pb_cache
except ImportError:
    print("ERROR: pybaseball not installed. Run: pip install pybaseball")
    sys.exit(1)

from src.sports.mlb.config_paths import MLB_PITCHER_DB, MLB_TRAINING_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_SEASONS = list(range(2018, 2026))
RATE_LIMIT_SLEEP = 2.0   # FanGraphs rate limit — 2 s between seasons

# Minimum games played thresholds
MIN_GS_STARTER  = 1
MIN_G_RELIEVER  = 10

# ---------------------------------------------------------------------------
# Column mapping: pybaseball → our schema
# pybaseball returns mixed-case/special-char column names.
# ---------------------------------------------------------------------------

# Starters schema
STARTERS_COLS: dict[str, str] = {
    "IDfg":      "PITCHER_ID",
    "Name":      "PITCHER_NAME",
    "Team":      "TEAM",
    # season injected separately
    "GS":        "GS",
    "IP":        "IP",
    "ERA":       "ERA",
    "FIP":       "FIP",
    "xFIP":      "xFIP",
    "SIERA":     "SIERA",
    "K%":        "K_PCT",
    "BB%":       "BB_PCT",
    "K-BB%":     "K_BB_PCT",
    "GB%":       "GB_PCT",
    "HR/9":      "HR_9",
    "WHIP":      "WHIP",
    "LOB%":      "LOB_PCT",
    "Hard%":     "HARD_PCT",
    "WAR":       "WAR",
}

CREATE_STARTERS_SQL = """
CREATE TABLE IF NOT EXISTS {table} (
    PITCHER_ID   TEXT,
    PITCHER_NAME TEXT,
    TEAM         TEXT,
    SEASON       INTEGER,
    GS           INTEGER,
    IP           REAL,
    ERA          REAL,
    FIP          REAL,
    xFIP         REAL,
    SIERA        REAL,
    K_PCT        REAL,
    BB_PCT       REAL,
    K_BB_PCT     REAL,
    GB_PCT       REAL,
    HR_9         REAL,
    WHIP         REAL,
    LOB_PCT      REAL,
    HARD_PCT     REAL,
    WAR          REAL
)
"""

CREATE_RELIEVERS_SQL = """
CREATE TABLE IF NOT EXISTS {table} (
    PITCHER_ID   TEXT,
    PITCHER_NAME TEXT,
    TEAM         TEXT,
    SEASON       INTEGER,
    G            INTEGER,
    IP           REAL,
    ERA          REAL,
    FIP          REAL,
    xFIP         REAL,
    SIERA        REAL,
    K_PCT        REAL,
    BB_PCT       REAL,
    K_BB_PCT     REAL,
    GB_PCT       REAL,
    HR_9         REAL,
    WHIP         REAL,
    LOB_PCT      REAL,
    HARD_PCT     REAL,
    WAR          REAL
)
"""

INSERT_STARTERS_SQL = """
INSERT INTO {table}
    (PITCHER_ID, PITCHER_NAME, TEAM, SEASON, GS, IP, ERA, FIP, xFIP, SIERA,
     K_PCT, BB_PCT, K_BB_PCT, GB_PCT, HR_9, WHIP, LOB_PCT, HARD_PCT, WAR)
VALUES
    (:PITCHER_ID, :PITCHER_NAME, :TEAM, :SEASON, :GS, :IP, :ERA, :FIP, :xFIP,
     :SIERA, :K_PCT, :BB_PCT, :K_BB_PCT, :GB_PCT, :HR_9, :WHIP, :LOB_PCT,
     :HARD_PCT, :WAR)
"""

INSERT_RELIEVERS_SQL = """
INSERT INTO {table}
    (PITCHER_ID, PITCHER_NAME, TEAM, SEASON, G, IP, ERA, FIP, xFIP, SIERA,
     K_PCT, BB_PCT, K_BB_PCT, GB_PCT, HR_9, WHIP, LOB_PCT, HARD_PCT, WAR)
VALUES
    (:PITCHER_ID, :PITCHER_NAME, :TEAM, :SEASON, :G, :IP, :ERA, :FIP, :xFIP,
     :SIERA, :K_PCT, :BB_PCT, :K_BB_PCT, :GB_PCT, :HR_9, :WHIP, :LOB_PCT,
     :HARD_PCT, :WAR)
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_float(val) -> float | None:
    """Convert to float, returning None if not parseable."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(str(val).replace("%", ""))
    except (ValueError, TypeError):
        return None


def _pct_to_decimal(val) -> float | None:
    """
    pybaseball K% and BB% are sometimes returned as decimals (0.25) and
    sometimes as percentages (25.0). Values > 1.5 are treated as raw percentages
    and divided by 100.
    """
    f = _safe_float(val)
    if f is None:
        return None
    return f / 100.0 if f > 1.5 else f


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column name that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _map_row(row: pd.Series, season: int, is_reliever: bool = False) -> dict:
    """Map a pybaseball row to our schema dict."""

    def g(candidates: list[str], converter=_safe_float):
        for c in candidates:
            if c in row.index and row[c] is not None:
                v = row[c]
                if isinstance(v, float) and pd.isna(v):
                    return None
                return converter(v) if converter else v
        return None

    rec = {
        "PITCHER_ID":   str(g(["IDfg", "playerid", "xMLBAMID"], converter=None) or ""),
        "PITCHER_NAME": str(g(["Name"], converter=None) or ""),
        "TEAM":         str(g(["Team"], converter=None) or ""),
        "SEASON":       season,
        "IP":           g(["IP"]),
        "ERA":          g(["ERA"]),
        "FIP":          g(["FIP"]),
        "xFIP":         g(["xFIP"]),
        "SIERA":        g(["SIERA"]),
        "K_PCT":        g(["K%"], converter=_pct_to_decimal),
        "BB_PCT":       g(["BB%"], converter=_pct_to_decimal),
        "K_BB_PCT":     g(["K-BB%"], converter=_pct_to_decimal),
        "GB_PCT":       g(["GB%"], converter=_pct_to_decimal),
        "HR_9":         g(["HR/9", "HR9"]),
        "WHIP":         g(["WHIP"]),
        "LOB_PCT":      g(["LOB%"], converter=_pct_to_decimal),
        "HARD_PCT":     g(["Hard%", "Hard%+"], converter=_pct_to_decimal),
        "WAR":          g(["WAR"]),
    }

    if is_reliever:
        rec["G"] = int(g(["G"], converter=lambda x: x) or 0)
    else:
        rec["GS"] = int(g(["GS"], converter=lambda x: x) or 0)

    return rec


# ---------------------------------------------------------------------------
# Season fetcher
# ---------------------------------------------------------------------------
def fetch_and_store_season(
    season: int,
    conn: sqlite3.Connection,
    overwrite: bool = False,
) -> tuple[int, int]:
    """
    Fetch all pitcher stats for one season, split into starters/relievers,
    and write to SQLite.

    Returns (starters_count, relievers_count).
    """
    s_table = f"pitcher_stats_{season}"
    r_table = f"reliever_stats_{season}"

    # Create tables
    conn.execute(CREATE_STARTERS_SQL.format(table=s_table))
    conn.execute(CREATE_RELIEVERS_SQL.format(table=r_table))
    conn.commit()

    # Check existing
    s_count = conn.execute(f"SELECT COUNT(*) FROM {s_table}").fetchone()[0]
    r_count = conn.execute(f"SELECT COUNT(*) FROM {r_table}").fetchone()[0]
    if not overwrite and (s_count > 0 or r_count > 0):
        print(
            f"  Season {season}: {s_table} ({s_count} rows), "
            f"{r_table} ({r_count} rows) — skipping"
        )
        return s_count, r_count

    if overwrite:
        conn.execute(f"DELETE FROM {s_table}")
        conn.execute(f"DELETE FROM {r_table}")
        conn.commit()

    print(f"  Fetching FanGraphs pitching stats for {season}...")
    try:
        df = pitching_stats(season, season, qual=0)
    except Exception as exc:
        print(f"  ERROR fetching season {season}: {exc}")
        return 0, 0

    if df is None or df.empty:
        print(f"  WARNING: No data returned for season {season}")
        return 0, 0

    print(f"  {len(df)} total pitcher records returned")

    # Split starters vs relievers
    # GS column might be labelled differently
    gs_col = _find_col(df, ["GS"])
    g_col  = _find_col(df, ["G"])

    if gs_col is None:
        print(f"  WARNING: No GS column found for {season}. Columns: {list(df.columns[:20])}")
        return 0, 0

    starters  = df[df[gs_col] >= MIN_GS_STARTER].copy()
    relievers = df[(df[gs_col] == 0) & (df[g_col] >= MIN_G_RELIEVER)].copy() if g_col else pd.DataFrame()

    # Insert starters
    s_inserted = 0
    insert_s = INSERT_STARTERS_SQL.format(table=s_table)
    for _, row in starters.iterrows():
        rec = _map_row(row, season, is_reliever=False)
        try:
            conn.execute(insert_s, rec)
            s_inserted += 1
        except Exception as exc:
            print(f"  WARNING: starter insert failed ({rec.get('PITCHER_NAME')}): {exc}")

    # Insert relievers
    r_inserted = 0
    if not relievers.empty:
        insert_r = INSERT_RELIEVERS_SQL.format(table=r_table)
        for _, row in relievers.iterrows():
            rec = _map_row(row, season, is_reliever=True)
            try:
                conn.execute(insert_r, rec)
                r_inserted += 1
            except Exception as exc:
                print(f"  WARNING: reliever insert failed ({rec.get('PITCHER_NAME')}): {exc}")

    conn.commit()
    print(f"  Season {season}: {s_inserted} starters, {r_inserted} relievers stored")
    return s_inserted, r_inserted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect pitcher season stats from FanGraphs into MLBPitcherData.sqlite"
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=DEFAULT_SEASONS,
        metavar="YYYY",
        help="Seasons to collect (default: 2018-2025)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing season tables",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable pybaseball caching (re-downloads from FanGraphs)",
    )
    args = parser.parse_args()

    # Enable pybaseball cache by default (avoids re-downloading on reruns)
    if not args.no_cache:
        try:
            pb_cache.enable()
            print("pybaseball cache enabled")
        except Exception:
            pass

    MLB_TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Pitcher stats collection → {MLB_PITCHER_DB}")
    print(f"Seasons to process: {args.seasons}\n")

    conn = sqlite3.connect(MLB_PITCHER_DB)

    total_s = total_r = 0
    for idx, season in enumerate(sorted(args.seasons)):
        print(f"=== Season {season} ===")
        s, r = fetch_and_store_season(season, conn, overwrite=args.overwrite)
        total_s += s
        total_r += r
        print()

        # Rate limit between seasons (not needed after the last one)
        if idx < len(args.seasons) - 1:
            print(f"  Sleeping {RATE_LIMIT_SLEEP}s (FanGraphs rate limit)...")
            time.sleep(RATE_LIMIT_SLEEP)

    conn.close()
    print(f"Done. Total starters: {total_s}, Total relievers: {total_r}")
    print(f"Database: {MLB_PITCHER_DB}")


if __name__ == "__main__":
    main()
