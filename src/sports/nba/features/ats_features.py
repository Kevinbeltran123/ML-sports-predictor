"""ATS (Against The Spread) rolling features por equipo.

Algunos equipos consistentemente cubren o fallan spreads. Un rolling ATS rate
captura esta tendencia para mejorar predicciones de Asian Handicap.

Features (4):
  ATS_RATE_HOME  — rolling 20-game ATS cover rate del equipo home (shift 1)
  ATS_RATE_AWAY  — rolling 20-game ATS cover rate del equipo away (shift 1)
  ATS_STREAK_HOME — racha ATS actual (positiva = cubriendo, negativa = fallando)
  ATS_STREAK_AWAY — racha ATS actual del away

Data source: OddsData.sqlite (Win_Margin + Spread por juego).
ATS cover: Win_Margin + Spread > 0 para home.
"""

import sqlite3
from collections import defaultdict

import numpy as np
import pandas as pd

from src.config import ODDS_DB, get_logger

logger = get_logger(__name__)

ATS_WINDOW = 20  # rolling window de juegos
ATS_FEATURES = ["ATS_RATE_HOME", "ATS_RATE_AWAY", "ATS_STREAK_HOME", "ATS_STREAK_AWAY"]


def build_ats_lookup():
    """Construye lookup de ATS rate por equipo y fecha desde OddsData.sqlite.

    Returns:
        dict: {team_name: DataFrame con Date, ATS_RATE, ATS_STREAK}
    """
    if not ODDS_DB.exists():
        logger.warning("OddsData.sqlite not found: %s", ODDS_DB)
        return {}

    all_games = []
    with sqlite3.connect(ODDS_DB) as con:
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", con)
        for table in tables["name"]:
            try:
                df = pd.read_sql(f'SELECT Date, Home, Away, Spread, Win_Margin FROM "{table}"', con)
                if len(df) > 0 and "Spread" in df.columns and "Win_Margin" in df.columns:
                    all_games.append(df)
            except Exception:
                continue

    if not all_games:
        logger.warning("No ATS data found in OddsData.sqlite")
        return {}

    games = pd.concat(all_games, ignore_index=True)
    games["Date"] = pd.to_datetime(games["Date"], errors="coerce")
    games = games.dropna(subset=["Date", "Spread", "Win_Margin"])
    games["Spread"] = pd.to_numeric(games["Spread"], errors="coerce")
    games["Win_Margin"] = pd.to_numeric(games["Win_Margin"], errors="coerce")
    games = games.dropna(subset=["Spread", "Win_Margin"])
    games = games.sort_values("Date").reset_index(drop=True)

    # ATS cover: home covers when Win_Margin + Spread > 0
    games["home_cover"] = (games["Win_Margin"] + games["Spread"] > 0).astype(int)

    # Build per-team ATS history
    team_lookup = {}
    teams = set(games["Home"].unique()) | set(games["Away"].unique())

    for team in teams:
        # Home games
        home_mask = games["Home"] == team
        # Away games (team covered = home did NOT cover)
        away_mask = games["Away"] == team

        rows = []
        for _, g in games[home_mask | away_mask].iterrows():
            if g["Home"] == team:
                covered = g["home_cover"]
            else:
                covered = 1 - g["home_cover"]  # away cover = not home cover
            rows.append({"Date": g["Date"], "covered": covered})

        if not rows:
            continue

        tdf = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)

        # Rolling ATS rate (shift 1 to avoid leakage)
        tdf["ATS_RATE"] = tdf["covered"].rolling(ATS_WINDOW, min_periods=5).mean().shift(1)

        # ATS streak
        streaks = []
        streak = 0
        for i, row in tdf.iterrows():
            streaks.append(streak)
            if row["covered"] == 1:
                streak = streak + 1 if streak > 0 else 1
            else:
                streak = streak - 1 if streak < 0 else -1
        tdf["ATS_STREAK"] = streaks  # already shifted (streak before this game)

        team_lookup[team] = tdf[["Date", "ATS_RATE", "ATS_STREAK"]].copy()

    logger.info("ATS lookup built: %d teams, %d total games", len(team_lookup), len(games))
    return team_lookup


def get_game_ats_features(home_team, away_team, game_date, ats_lookup):
    """Retorna ATS features para un partido específico.

    Args:
        home_team: nombre del equipo local
        away_team: nombre del equipo visitante
        game_date: fecha del partido (datetime o string)
        ats_lookup: dict del build_ats_lookup()

    Returns:
        dict con ATS_RATE_HOME, ATS_RATE_AWAY, ATS_STREAK_HOME, ATS_STREAK_AWAY
    """
    result = {f: 0.5 if "RATE" in f else 0.0 for f in ATS_FEATURES}

    if not ats_lookup:
        return result

    game_date = pd.to_datetime(game_date)

    for team, prefix in [(home_team, "HOME"), (away_team, "AWAY")]:
        tdf = ats_lookup.get(team)
        if tdf is None:
            continue
        # Find most recent entry before game_date
        mask = tdf["Date"] < game_date
        if mask.any():
            last = tdf[mask].iloc[-1]
            rate = last["ATS_RATE"]
            streak = last["ATS_STREAK"]
            result[f"ATS_RATE_{prefix}"] = rate if pd.notna(rate) else 0.5
            result[f"ATS_STREAK_{prefix}"] = int(streak) if pd.notna(streak) else 0

    return result


def add_ats_to_frame(frame, ats_lookup=None):
    """Agrega ATS features al DataFrame de entrenamiento.

    Args:
        frame: DataFrame con columnas Home, Away (o Home-Team, Away-Team), Date
        ats_lookup: pre-built lookup, o None para construir on-the-fly

    Returns:
        frame con 4 columnas adicionales
    """
    if ats_lookup is None:
        ats_lookup = build_ats_lookup()

    if not ats_lookup:
        for f in ATS_FEATURES:
            frame[f] = 0.5 if "RATE" in f else 0.0
        return frame

    # Determine column names
    home_col = "Home" if "Home" in frame.columns else "TEAM_NAME"
    away_col = "Away" if "Away" in frame.columns else "TEAM_NAME.1"
    date_col = "Date"

    features = {f: [] for f in ATS_FEATURES}

    for _, row in frame.iterrows():
        feats = get_game_ats_features(
            row[home_col], row[away_col], row[date_col], ats_lookup
        )
        for f in ATS_FEATURES:
            features[f].append(feats[f])

    for f in ATS_FEATURES:
        frame[f] = features[f]

    logger.info("ATS features added: %d rows, %d with data",
                len(frame), frame["ATS_RATE_HOME"].notna().sum())
    return frame
