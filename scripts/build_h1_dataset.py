"""Build H1 (First Half) dataset by joining halftime scores to existing dataset.

Reads the existing dataset_2012-26 from dataset.sqlite, joins with
HalftimeScores.sqlite to add H1-Home-Win column, and saves as new table.

Uso:
    PYTHONPATH=. python scripts/build_h1_dataset.py
"""

import sqlite3

import numpy as np
import pandas as pd

from src.config import DATASET_DB, get_logger

logger = get_logger(__name__)

H1_DB = "data/training/HalftimeScores.sqlite"
SOURCE_TABLE = "dataset_2012-26"
TARGET_TABLE = "dataset_h1_2012-26"

# Team name mapping: dataset uses full names, H1 DB uses tricodes
TRICODE_TO_FULL = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}

# Reverse: full name -> tricode
FULL_TO_TRICODE = {v: k for k, v in TRICODE_TO_FULL.items()}
# Also add common variations
FULL_TO_TRICODE.update({
    "Los Angeles Clippers": "LAC",
    "Charlotte Bobcats": "CHA",
    "New Jersey Nets": "BKN",
    "Seattle SuperSonics": "SEA",
})


def _normalize_team(name: str) -> str:
    """Get tricode from various name formats."""
    if len(name) == 3:
        return name.upper()
    # Try full name match
    tri = FULL_TO_TRICODE.get(name)
    if tri:
        return tri
    # Try last word match (e.g., "Hawks" -> ATL)
    last = name.strip().split()[-1]
    for full, tri in FULL_TO_TRICODE.items():
        if full.endswith(last):
            return tri
    return name.upper()[:3]


def main():
    print(f"\n{'='*60}")
    print(f"  BUILD H1 DATASET")
    print(f"{'='*60}")

    # Load existing dataset
    with sqlite3.connect(DATASET_DB) as con:
        df = pd.read_sql_query(f'SELECT * FROM "{SOURCE_TABLE}"', con)
    print(f"  Source dataset: {len(df)} rows, {len(df.columns)} columns")

    # Load halftime scores
    with sqlite3.connect(H1_DB) as con:
        h1 = pd.read_sql_query("SELECT * FROM halftime_scores", con)
    print(f"  Halftime scores: {len(h1)} games")

    # Normalize dates
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    h1["date"] = pd.to_datetime(h1["date"], errors="coerce")

    # Normalize team names in dataset to tricodes
    if "TEAM_NAME" in df.columns:
        df["_home_tri"] = df["TEAM_NAME"].apply(_normalize_team)
    elif "Home" in df.columns:
        df["_home_tri"] = df["Home"].apply(_normalize_team)
    else:
        # Try to find team column
        print("  ERROR: No team name column found in dataset")
        return

    h1["_home_tri"] = h1["home_team"].apply(_normalize_team)

    # Create date string for matching
    df["_date_str"] = df["Date"].dt.strftime("%Y-%m-%d")
    h1["_date_str"] = h1["date"].dt.strftime("%Y-%m-%d")

    # Merge on (date, home_tricode)
    h1_lookup = h1.set_index(["_date_str", "_home_tri"])["h1_home_win"].to_dict()

    matched = 0
    ties = 0
    h1_values = []

    for _, row in df.iterrows():
        key = (row["_date_str"], row["_home_tri"])
        h1_win = h1_lookup.get(key)
        if h1_win is not None:
            matched += 1
            h1_values.append(int(h1_win))
        else:
            h1_values.append(np.nan)

    df["H1-Home-Win"] = h1_values

    # Drop temp columns
    df = df.drop(columns=["_home_tri", "_date_str"], errors="ignore")

    # Stats
    n_valid = df["H1-Home-Win"].notna().sum()
    n_missing = df["H1-Home-Win"].isna().sum()
    n_h1_wins = (df["H1-Home-Win"] == 1).sum()

    print(f"\n  Matched: {matched}/{len(df)} ({matched/len(df):.1%})")
    print(f"  Missing H1 data: {n_missing}")
    print(f"  H1 Home Win Rate: {n_h1_wins}/{n_valid} ({n_h1_wins/n_valid:.1%})" if n_valid > 0 else "")

    # Check ties in H1 scores
    h1_ties = h1[h1["h1_home_score"] == h1["h1_away_score"]]
    print(f"  H1 Ties (excluded as NaN): {len(h1_ties)}")

    # Save — keep NaN rows (they'll be excluded during training by dropna on target)
    with sqlite3.connect(DATASET_DB) as con:
        df.to_sql(TARGET_TABLE, con, if_exists="replace", index=False)

    print(f"\n  Saved: {TARGET_TABLE} ({len(df)} rows) to {DATASET_DB.name}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
