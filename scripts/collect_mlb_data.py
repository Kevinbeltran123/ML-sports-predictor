"""collect_mlb_data.py — Historical MLB game log collector (2018-2025).

Collects per-game batting and pitching lines for both teams via MLB-StatsAPI
and stores them in MLBTeamData.sqlite with one table per season.

Row model: 2 rows per game (home team perspective + away team perspective),
matching the pattern used by the NBA TeamData.sqlite pipeline.

Usage:
    PYTHONPATH=. python scripts/collect_mlb_data.py
    PYTHONPATH=. python scripts/collect_mlb_data.py --seasons 2024 2025
    PYTHONPATH=. python scripts/collect_mlb_data.py --seasons 2024 --overwrite
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

try:
    import statsapi
except ImportError:
    print("ERROR: mlb-statsapi not installed. Run: pip install mlb-statsapi")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Import from src so PYTHONPATH=. is respected
from src.sports.mlb.config_paths import MLB_TEAMS_DB, MLB_TRAINING_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_SEASONS = list(range(2018, 2026))   # 2018-2025 inclusive
RATE_LIMIT_SLEEP = 0.5                      # seconds between boxscore requests
PRINT_EVERY = 100                           # progress interval

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {table} (
    GAME_PK         INTEGER,
    GAME_DATE       TEXT,
    SEASON          INTEGER,
    TEAM_NAME       TEXT,
    TEAM_ID         INTEGER,
    OPPONENT_NAME   TEXT,
    OPPONENT_ID     INTEGER,
    HOME_AWAY       TEXT,
    WIN             INTEGER,
    RUNS            INTEGER,
    HITS            INTEGER,
    ERRORS          INTEGER,
    AB              INTEGER,
    R               INTEGER,
    H               INTEGER,
    HR              INTEGER,
    RBI             INTEGER,
    BB              INTEGER,
    SO              INTEGER,
    AVG             REAL,
    OBP             REAL,
    SLG             REAL,
    OPS             REAL,
    LOB             INTEGER,
    SP_NAME         TEXT,
    SP_ID           INTEGER,
    SP_IP           REAL,
    SP_H            INTEGER,
    SP_R            INTEGER,
    SP_ER           INTEGER,
    SP_BB           INTEGER,
    SP_SO           INTEGER,
    SP_HR           INTEGER,
    SP_PITCHES      INTEGER,
    BULLPEN_IP      REAL,
    BULLPEN_ER      INTEGER,
    BULLPEN_H       INTEGER,
    BULLPEN_BB      INTEGER,
    BULLPEN_SO      INTEGER,
    INNING_RUNS     TEXT,
    F5_RUNS         INTEGER
)
"""

INSERT_SQL = """
INSERT INTO {table} VALUES (
    :GAME_PK, :GAME_DATE, :SEASON,
    :TEAM_NAME, :TEAM_ID, :OPPONENT_NAME, :OPPONENT_ID,
    :HOME_AWAY, :WIN,
    :RUNS, :HITS, :ERRORS,
    :AB, :R, :H, :HR, :RBI, :BB, :SO,
    :AVG, :OBP, :SLG, :OPS,
    :LOB,
    :SP_NAME, :SP_ID,
    :SP_IP, :SP_H, :SP_R, :SP_ER, :SP_BB, :SP_SO, :SP_HR, :SP_PITCHES,
    :BULLPEN_IP, :BULLPEN_ER, :BULLPEN_H, :BULLPEN_BB, :BULLPEN_SO,
    :INNING_RUNS, :F5_RUNS
)
"""


# ---------------------------------------------------------------------------
# IP conversion  (e.g. "5.2" outs notation → float 5.667)
# ---------------------------------------------------------------------------
def ip_to_float(ip_str: Any) -> float:
    """Convert MLB innings-pitched string/number to a decimal float."""
    if ip_str is None:
        return 0.0
    try:
        val = float(ip_str)
        full = int(val)
        outs = round((val - full) * 10)   # .1 → 1 out, .2 → 2 outs
        return full + outs / 3.0
    except (ValueError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# Live feed parser helpers
# ---------------------------------------------------------------------------
def _safe(d: dict, *keys, default=None):
    """Safely navigate nested dict."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is None:
            return default
    return cur


def _parse_pitcher_line(pitcher_data: dict) -> dict:
    """Extract SP/RP stat line from a player dict inside liveData.boxscore."""
    stats = _safe(pitcher_data, "stats", "pitching", default={})
    season = _safe(pitcher_data, "seasonStats", "pitching", default={})
    person = _safe(pitcher_data, "person", default={})
    return {
        "id":      person.get("id"),
        "name":    person.get("fullName", ""),
        "ip":      ip_to_float(stats.get("inningsPitched", 0)),
        "h":       stats.get("hits", 0),
        "r":       stats.get("runs", 0),
        "er":      stats.get("earnedRuns", 0),
        "bb":      stats.get("baseOnBalls", 0),
        "so":      stats.get("strikeOuts", 0),
        "hr":      stats.get("homeRuns", 0),
        "pitches": stats.get("pitchesThrown", 0),
    }


def _parse_team_side(side_data: dict, inning_runs: list[int]) -> dict:
    """Parse one team's boxscore side from the live feed JSON."""
    batting  = _safe(side_data, "teamStats", "batting",  default={})
    fielding = _safe(side_data, "teamStats", "fielding", default={})
    team_info = _safe(side_data, "team", default={})

    total_runs  = batting.get("runs", 0)
    total_hits  = batting.get("hits", 0)
    total_errors = fielding.get("errors", 0)

    # Batting line
    ab  = batting.get("atBats", 0)
    r   = batting.get("runs", 0)
    h   = batting.get("hits", 0)
    hr  = batting.get("homeRuns", 0)
    rbi = batting.get("rbi", 0)
    bb  = batting.get("baseOnBalls", 0)
    so  = batting.get("strikeOuts", 0)
    lob = batting.get("leftOnBase", 0)

    # Derived rates (guard div-by-zero)
    avg = h / ab if ab else 0.0
    obp_denom = ab + bb + batting.get("sacFlies", 0)
    obp = (h + bb + batting.get("hitByPitch", 0)) / obp_denom if obp_denom else 0.0
    tb  = h + batting.get("doubles", 0) + 2 * batting.get("triples", 0) + 3 * hr
    slg = tb / ab if ab else 0.0
    ops = obp + slg

    # Pitchers: identify SP (gameSequence=1) vs bullpen
    players = side_data.get("players", {})
    pitchers = [
        p for p in players.values()
        if _safe(p, "position", "abbreviation") == "P"
        and _safe(p, "stats", "pitching") is not None
        and _safe(p, "stats", "pitching") != {}
    ]
    # Sort by game sequence (1 = SP)
    pitchers.sort(key=lambda p: p.get("gameSequence", 99))

    sp_line  = {}
    bp_ip    = bp_er = bp_h = bp_bb = bp_so = 0
    if pitchers:
        sp_raw  = pitchers[0]
        sp_line = _parse_pitcher_line(sp_raw)
        for rp in pitchers[1:]:
            rp_line = _parse_pitcher_line(rp)
            bp_ip  += rp_line["ip"]
            bp_er  += rp_line["er"]
            bp_h   += rp_line["h"]
            bp_bb  += rp_line["bb"]
            bp_so  += rp_line["so"]

    f5_runs = sum(inning_runs[:5])

    return {
        "TEAM_NAME":   team_info.get("name", ""),
        "TEAM_ID":     team_info.get("id"),
        "RUNS":        total_runs,
        "HITS":        total_hits,
        "ERRORS":      total_errors,
        "AB":          ab,
        "R":           r,
        "H":           h,
        "HR":          hr,
        "RBI":         rbi,
        "BB":          bb,
        "SO":          so,
        "AVG":         round(avg, 4),
        "OBP":         round(obp, 4),
        "SLG":         round(slg, 4),
        "OPS":         round(ops, 4),
        "LOB":         lob,
        "SP_NAME":     sp_line.get("name", ""),
        "SP_ID":       sp_line.get("id"),
        "SP_IP":       round(sp_line.get("ip", 0.0), 3),
        "SP_H":        sp_line.get("h", 0),
        "SP_R":        sp_line.get("r", 0),
        "SP_ER":       sp_line.get("er", 0),
        "SP_BB":       sp_line.get("bb", 0),
        "SP_SO":       sp_line.get("so", 0),
        "SP_HR":       sp_line.get("hr", 0),
        "SP_PITCHES":  sp_line.get("pitches", 0),
        "BULLPEN_IP":  round(bp_ip, 3),
        "BULLPEN_ER":  bp_er,
        "BULLPEN_H":   bp_h,
        "BULLPEN_BB":  bp_bb,
        "BULLPEN_SO":  bp_so,
        "INNING_RUNS": json.dumps(inning_runs),
        "F5_RUNS":     f5_runs,
    }


# ---------------------------------------------------------------------------
# Game fetcher
# ---------------------------------------------------------------------------
def fetch_game(game_pk: int) -> tuple[dict, dict] | None:
    """
    Fetch and parse both team sides for a completed game.
    Returns (home_row_partial, away_row_partial) or None on failure.
    """
    try:
        feed = statsapi.get("game", {"gamePk": game_pk})
    except Exception as exc:
        print(f"    WARNING: could not fetch game {game_pk}: {exc}")
        return None

    try:
        live  = feed.get("liveData", {})
        box   = live.get("boxscore", {}).get("teams", {})
        lscore = live.get("linescore", {})

        home_side = box.get("home", {})
        away_side = box.get("away", {})

        # Inning-by-inning runs
        innings = lscore.get("innings", [])
        home_inning_runs = [inn.get("home", {}).get("runs", 0) for inn in innings]
        away_inning_runs = [inn.get("away", {}).get("runs", 0) for inn in innings]

        home_partial = _parse_team_side(home_side, home_inning_runs)
        away_partial = _parse_team_side(away_side, away_inning_runs)
        return home_partial, away_partial

    except Exception as exc:
        print(f"    WARNING: parse error on game {game_pk}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Season collector
# ---------------------------------------------------------------------------
def collect_season(season: int, conn: sqlite3.Connection) -> int:
    """Collect all regular-season + playoff games for one MLB season."""
    table = f"season_{season}"

    # Create table
    conn.execute(CREATE_TABLE_SQL.format(table=table))
    conn.commit()

    # Check if already populated
    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    if count > 0:
        print(f"  Season {season}: {count} rows already exist — skipping (use --overwrite to redo)")
        return 0

    # Build date ranges for regular season + playoffs
    date_ranges = [
        (f"{season}-03-15", f"{season}-11-15"),   # covers all games incl. postseason
    ]

    all_game_ids: list[tuple[int, str]] = []  # (game_pk, game_date)
    for start, end in date_ranges:
        try:
            schedule = statsapi.schedule(
                start_date=start,
                end_date=end,
                sportId=1,
            )
        except Exception as exc:
            print(f"  WARNING: schedule fetch failed for {season}: {exc}")
            continue

        for game in schedule:
            status = game.get("status", "")
            # Only final / completed games
            if status not in ("Final", "Game Over", "Completed Early"):
                continue
            gid = game.get("game_id")
            gdate = game.get("game_date", "")
            if gid:
                all_game_ids.append((gid, gdate))

    # Deduplicate (schedule endpoint can return duplicates for doubleheaders)
    seen: set[int] = set()
    unique_games: list[tuple[int, str]] = []
    for gid, gdate in all_game_ids:
        if gid not in seen:
            seen.add(gid)
            unique_games.append((gid, gdate))

    print(f"  Season {season}: {len(unique_games)} completed games found")
    if not unique_games:
        return 0

    rows_written = 0
    insert_sql = INSERT_SQL.format(table=table)

    for idx, (game_pk, game_date) in enumerate(unique_games, 1):
        if idx % PRINT_EVERY == 0:
            print(f"    ... processed {idx}/{len(unique_games)} games ({rows_written} rows)")

        result = fetch_game(game_pk)
        time.sleep(RATE_LIMIT_SLEEP)

        if result is None:
            continue

        home_partial, away_partial = result
        home_runs = home_partial["RUNS"]
        away_runs = away_partial["RUNS"]

        # Compose full row for home team
        home_row = {
            "GAME_PK":       game_pk,
            "GAME_DATE":     game_date,
            "SEASON":        season,
            "HOME_AWAY":     "home",
            "WIN":           1 if home_runs > away_runs else 0,
            "OPPONENT_NAME": away_partial["TEAM_NAME"],
            "OPPONENT_ID":   away_partial["TEAM_ID"],
            **home_partial,
        }

        # Compose full row for away team
        away_row = {
            "GAME_PK":       game_pk,
            "GAME_DATE":     game_date,
            "SEASON":        season,
            "HOME_AWAY":     "away",
            "WIN":           1 if away_runs > home_runs else 0,
            "OPPONENT_NAME": home_partial["TEAM_NAME"],
            "OPPONENT_ID":   home_partial["TEAM_ID"],
            **away_partial,
        }

        try:
            conn.execute(insert_sql, home_row)
            conn.execute(insert_sql, away_row)
            conn.commit()
            rows_written += 2
        except Exception as exc:
            print(f"    WARNING: DB insert failed for game {game_pk}: {exc}")

    return rows_written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect historical MLB game logs into MLBTeamData.sqlite"
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
        help="Drop and recreate tables even if data already exists",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    MLB_TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    print(f"MLB data collection → {MLB_TEAMS_DB}")
    print(f"Seasons to process: {args.seasons}\n")

    conn = sqlite3.connect(MLB_TEAMS_DB)

    if args.overwrite:
        for season in args.seasons:
            table = f"season_{season}"
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.commit()
            print(f"Dropped table {table}")

    total_rows = 0
    for season in sorted(args.seasons):
        print(f"=== Season {season} ===")
        rows = collect_season(season, conn)
        total_rows += rows
        print(f"  Season {season}: wrote {rows} rows\n")

    conn.close()
    print(f"Done. Total rows written: {total_rows}")
    print(f"Database: {MLB_TEAMS_DB}")


if __name__ == "__main__":
    main()
