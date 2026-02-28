"""Collect halftime scores from NBA CDN boxscore API.

Fetches quarter-by-quarter scores for each game and computes H1 winner.
Saves to data/training/HalftimeScores.sqlite (resumable).

Uso:
    PYTHONPATH=. python scripts/collect_halftime_scores.py
    PYTHONPATH=. python scripts/collect_halftime_scores.py --seasons 2024-25 2025-26
"""

import argparse
import os
import sqlite3
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import requests

from src.config import get_logger

logger = get_logger(__name__)

DATA_DIR = Path("data/training")
H1_DB = DATA_DIR / "HalftimeScores.sqlite"
H1_TABLE = "halftime_scores"

CDN_BASE = "https://cdn.nba.com/static/json"
CDN_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
}

DEFAULT_SEASONS = [
    "2012-13", "2013-14", "2014-15", "2015-16", "2016-17",
    "2017-18", "2018-19", "2019-20", "2020-21", "2021-22",
    "2022-23", "2023-24", "2024-25", "2025-26",
]


def _cdn_get(url: str, retries=3, delay=3) -> dict | None:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=CDN_HEADERS, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            logger.warning("CDN %d for %s (attempt %d)", r.status_code, url.split("/")[-1], attempt)
        except Exception as e:
            if attempt < retries:
                logger.warning("CDN error (attempt %d/%d): %s", attempt, retries, str(e)[:60])
                time.sleep(delay)
            else:
                return None
    return None


def _get_season_games(season: str, season_type: str = "Regular Season") -> list[dict]:
    """Get game_ids via stats.nba.com leaguegamelog."""
    url = "https://stats.nba.com/stats/leaguegamelog"
    params = {
        "LeagueID": "00",
        "Season": season,
        "SeasonType": season_type,
        "PlayerOrTeam": "T",
        "Direction": "ASC",
        "Sorter": "DATE",
    }
    headers = {
        **CDN_HEADERS,
        "Referer": "https://www.nba.com/",
    }

    for attempt in range(1, 4):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            if r.status_code != 200:
                logger.warning("Stats API %d for %s (attempt %d)", r.status_code, season, attempt)
                time.sleep(3)
                continue
            data = r.json()
            rs = data["resultSets"][0]
            cols = rs["headers"]
            rows = rs["rowSet"]
            break
        except Exception as e:
            logger.warning("Stats API attempt %d: %s", attempt, str(e)[:60])
            if attempt < 3:
                time.sleep(5)
            else:
                return []
    else:
        return []

    gid_idx = cols.index("GAME_ID")
    team_idx = cols.index("TEAM_ABBREVIATION")
    matchup_idx = cols.index("MATCHUP")
    date_idx = cols.index("GAME_DATE")

    games = []
    seen = set()
    for row in rows:
        gid = row[gid_idx]
        if gid in seen:
            continue
        matchup = row[matchup_idx]
        if "vs." not in matchup:
            continue
        seen.add(gid)
        games.append({
            "game_id": gid,
            "date": row[date_idx],
            "home_team": row[team_idx],
            "away_team": matchup.split("vs.")[-1].strip(),
        })

    logger.info("Season %s (%s): %d games", season, season_type, len(games))
    return games


def _get_boxscore_halftime(game_id: str) -> dict | None:
    """Fetch CDN boxscore and extract halftime scores from periods array."""
    time.sleep(0.3)
    data = _cdn_get(
        f"{CDN_BASE}/liveData/boxscore/boxscore_{game_id}.json",
        retries=2, delay=3,
    )
    if not data:
        return None

    game = data.get("game", {})
    home = game.get("homeTeam", {})
    away = game.get("awayTeam", {})

    home_periods = home.get("periods", [])
    away_periods = away.get("periods", [])

    if len(home_periods) < 2 or len(away_periods) < 2:
        return None

    h1_home = home_periods[0].get("score", 0) + home_periods[1].get("score", 0)
    h1_away = away_periods[0].get("score", 0) + away_periods[1].get("score", 0)

    return {
        "h1_home_score": h1_home,
        "h1_away_score": h1_away,
        "home_tricode": home.get("teamTricode", ""),
        "away_tricode": away.get("teamTricode", ""),
    }


def _init_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(H1_DB) as con:
        con.execute(f"""
            CREATE TABLE IF NOT EXISTS {H1_TABLE} (
                game_id TEXT PRIMARY KEY,
                date TEXT,
                home_team TEXT,
                away_team TEXT,
                h1_home_score INTEGER,
                h1_away_score INTEGER,
                h1_home_win INTEGER
            )
        """)
    return H1_DB


def _get_existing_ids() -> set:
    if not H1_DB.exists():
        return set()
    with sqlite3.connect(H1_DB) as con:
        rows = con.execute(f"SELECT game_id FROM {H1_TABLE}").fetchall()
    return {r[0] for r in rows}


def collect_season(season: str, existing_ids: set) -> int:
    """Collect halftime scores for one season. Returns count of new games."""
    games = _get_season_games(season, "Regular Season")
    time.sleep(1)
    playoff_games = _get_season_games(season, "Playoffs")
    if playoff_games:
        games.extend(playoff_games)

    new_games = [g for g in games if g["game_id"] not in existing_ids]
    if not new_games:
        print(f"  {season}: all {len(games)} games already collected")
        return 0

    print(f"  {season}: {len(new_games)} new games to fetch (of {len(games)} total)")

    collected = 0
    batch = []

    for i, g in enumerate(new_games):
        ht = _get_boxscore_halftime(g["game_id"])
        if ht is None:
            continue

        h1_home_win = 1 if ht["h1_home_score"] > ht["h1_away_score"] else 0

        batch.append((
            g["game_id"], g["date"], g["home_team"], g["away_team"],
            ht["h1_home_score"], ht["h1_away_score"], h1_home_win,
        ))
        collected += 1

        # Save every 100 games
        if len(batch) >= 100:
            with sqlite3.connect(H1_DB) as con:
                con.executemany(
                    f"INSERT OR IGNORE INTO {H1_TABLE} VALUES (?,?,?,?,?,?,?)",
                    batch,
                )
            print(f"    checkpoint: {collected}/{len(new_games)} ({collected/len(new_games):.0%})")
            batch = []

    # Final batch
    if batch:
        with sqlite3.connect(H1_DB) as con:
            con.executemany(
                f"INSERT OR IGNORE INTO {H1_TABLE} VALUES (?,?,?,?,?,?,?)",
                batch,
            )

    print(f"  {season}: collected {collected} halftime scores")
    return collected


def main():
    parser = argparse.ArgumentParser(description="Collect halftime scores from NBA CDN")
    parser.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  COLLECT HALFTIME SCORES — NBA CDN Boxscore")
    print(f"{'='*60}")

    _init_db()
    existing = _get_existing_ids()
    print(f"  Existing: {len(existing)} games in DB")
    print(f"  Seasons: {', '.join(args.seasons)}")

    total = 0
    for season in args.seasons:
        try:
            n = collect_season(season, existing)
            total += n
            existing = _get_existing_ids()  # refresh
            time.sleep(2)  # pause between seasons
        except Exception as e:
            logger.error("Season %s failed: %s", season, e)
            continue

    print(f"\n  Total new: {total}")
    print(f"  Total in DB: {len(_get_existing_ids())}")

    # Quick stats
    with sqlite3.connect(H1_DB) as con:
        total_rows = con.execute(f"SELECT COUNT(*) FROM {H1_TABLE}").fetchone()[0]
        h1_home_wins = con.execute(
            f"SELECT SUM(h1_home_win) FROM {H1_TABLE}"
        ).fetchone()[0] or 0
        ties = con.execute(
            f"SELECT COUNT(*) FROM {H1_TABLE} WHERE h1_home_score = h1_away_score"
        ).fetchone()[0]

    pct = h1_home_wins / total_rows if total_rows else 0
    print(f"\n  H1 Home Win Rate: {h1_home_wins}/{total_rows} ({pct:.1%})")
    print(f"  H1 Ties: {ties} ({ties/total_rows:.1%})" if total_rows else "")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
