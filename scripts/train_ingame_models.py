"""Train in-game XGBoost + Logistic PBP models with score-context features.

Collects historical PBP data from nba_api, computes 20 PBP features
(17 original + 3 score-context to fix home bias in blowouts),
and trains cascading models per quarter (Q1, Q2, Q3).

Uso:
    PYTHONPATH=. python scripts/train_ingame_models.py
    PYTHONPATH=. python scripts/train_ingame_models.py --seasons 2024-25 2025-26
    PYTHONPATH=. python scripts/train_ingame_models.py --skip-collect  # solo retrain

Score-context features (fix home bias):
  PBP_SCORE_DIFF_NORM:          score_diff / max(total_points, 1)
  PBP_BLOWOUT_FLAG:             1.0 if abs(score_diff) > 15
  PBP_SCORE_DIFF_X_MOMENTUM:   momentum * (1 - sigmoid(score_diff/10))
"""

import argparse
import json
import math
import os
import sqlite3
import sys
import time
from pathlib import Path

# Force unbuffered output (conda run buffers aggressively)
os.environ["PYTHONUNBUFFERED"] = "1"

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from src.config import INGAME_MODELS_DIR, get_logger
from src.core.calibration.conformal import ConformalClassifier
from src.core.calibration.xgb_calibrator import XGBCalibrator
from src.sports.nba.features.live_pbp_tracker import LivePBPTracker

logger = get_logger(__name__)

# --- Paths ---
DATA_DIR = Path("data/training")
PBP_DB = DATA_DIR / "PBPFeatures.sqlite"
PBP_TABLE = "pbp_features_v2"  # v2 includes score-context features

# --- Feature definitions ---
PBP_FEATURE_COLS = [
    "PBP_LEAD_CHANGES", "PBP_LARGEST_LEAD_HOME", "PBP_LARGEST_LEAD_AWAY",
    "PBP_HOME_RUNS_MAX", "PBP_AWAY_RUNS_MAX",
    "PBP_TIMEOUTS_HOME", "PBP_TIMEOUTS_AWAY",
    "PBP_FOULS_HOME", "PBP_FOULS_AWAY",
    "PBP_TURNOVERS_HOME", "PBP_TURNOVERS_AWAY",
    "PBP_MOMENTUM", "PBP_LAST_5MIN_DIFF",
    "PBP_FG3_MADE_HOME", "PBP_FG3_MADE_AWAY",
    "PBP_OREB_HOME", "PBP_OREB_AWAY",
    # v2 score-context features
    "PBP_SCORE_DIFF_NORM", "PBP_BLOWOUT_FLAG", "PBP_SCORE_DIFF_X_MOMENTUM",
]

# XGBoost: LOGIT_PREGAME + all 20 PBP features = 21
XGB_FEATURES = ["LOGIT_PREGAME"] + PBP_FEATURE_COLS
# Logistic: LOGIT_PREGAME + first 8 PBP + 3 score-context = 12
LOGISTIC_FEATURES = ["LOGIT_PREGAME"] + PBP_FEATURE_COLS[:8] + PBP_FEATURE_COLS[-3:]

TARGET = "HOME_WIN"

# --- XGBoost hyperparameters (adapted from moneyline Optuna tuning) ---
XGB_PARAMS = {
    "max_depth": 6, "eta": 0.05, "subsample": 0.85,
    "colsample_bytree": 0.80, "colsample_bylevel": 0.75,
    "min_child_weight": 5, "gamma": 2.0,
    "lambda": 3.0, "alpha": 0.5,
    "objective": "multi:softprob", "num_class": 2,
    "tree_method": "hist", "seed": 42,
    "eval_metric": ["mlogloss"],
}
XGB_ROUNDS = {1: 500, 2: 600, 3: 700}  # mas rounds en periodos tardios

DEFAULT_SEASONS = [
    "2019-20", "2020-21", "2021-22", "2022-23",
    "2023-24", "2024-25", "2025-26",
]


# =====================================================================
# Data collection: PBP from nba_api historical
# =====================================================================

CDN_BASE = "https://cdn.nba.com/static/json"
CDN_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
}


def _cdn_get(url: str, retries=3, delay=5) -> dict | None:
    """GET JSON from cdn.nba.com with retry."""
    import requests
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=CDN_HEADERS, timeout=30)
            if r.status_code == 200:
                return r.json()
            logger.warning("CDN %d for %s", r.status_code, url.split("/")[-1])
            return None
        except Exception as e:
            if attempt < retries:
                logger.warning("CDN attempt %d/%d: %s. Retry in %ds...",
                               attempt, retries, str(e)[:60], delay)
                time.sleep(delay)
            else:
                return None


def _get_season_games(season: str, include_playoffs: bool = True) -> list[dict]:
    """Obtiene game_ids y resultados de la temporada.

    Intenta stats.nba.com/stats/leaguegamelog primero (funciona para temporadas
    históricas y actuales). Fallback a cdn.nba.com schedule para temporada actual.
    Si include_playoffs=True, también recolecta juegos de playoffs.
    """
    games = _get_season_games_stats_api(season, "Regular Season")
    if not games:
        logger.info("Stats API failed for %s RS, trying CDN schedule...", season)
        games = _get_season_games_cdn(season)

    if include_playoffs:
        time.sleep(1)  # rate limit entre requests
        playoff_games = _get_season_games_stats_api(season, "Playoffs")
        if playoff_games:
            logger.info("Season %s: +%d playoff games", season, len(playoff_games))
            games.extend(playoff_games)

    return games


def _get_season_games_stats_api(season: str, season_type: str = "Regular Season") -> list[dict]:
    """Obtiene games via stats.nba.com leaguegamelog (todas las temporadas)."""
    import requests

    url = "https://stats.nba.com/stats/leaguegamelog"
    params = {
        "LeagueID": "00",
        "Season": season,
        "SeasonType": season_type,
        "PlayerOrTeam": "T",
        "Direction": "ASC",
        "Sorter": "DATE",
    }
    stats_headers = {
        **CDN_HEADERS,
        "Referer": "https://www.nba.com/",
    }

    for attempt in range(1, 4):
        try:
            r = requests.get(url, params=params, headers=stats_headers, timeout=30)
            if r.status_code != 200:
                logger.warning("Stats API %d for season %s (attempt %d)",
                               r.status_code, season, attempt)
                time.sleep(3)
                continue

            data = r.json()
            rs = data["resultSets"][0]
            cols = rs["headers"]
            rows = rs["rowSet"]
            break
        except Exception as e:
            logger.warning("Stats API attempt %d/%d: %s", attempt, 3, str(e)[:60])
            if attempt < 3:
                time.sleep(5)
            else:
                return []
    else:
        return []

    # Column indices
    gid_idx = cols.index("GAME_ID")
    team_idx = cols.index("TEAM_ABBREVIATION")
    matchup_idx = cols.index("MATCHUP")
    wl_idx = cols.index("WL")
    date_idx = cols.index("GAME_DATE")

    # Deduplicate: each game has 2 rows (home + away). Keep home row ("vs." in matchup).
    games = []
    seen_ids = set()
    for row in rows:
        game_id = row[gid_idx]
        if game_id in seen_ids:
            continue

        matchup = row[matchup_idx]  # "DEN vs. LAL" or "LAL @ DEN"
        is_home = "vs." in matchup

        if is_home:
            home_team = row[team_idx]
            away_team = matchup.split("vs.")[-1].strip()
            home_win = 1 if row[wl_idx] == "W" else 0
        else:
            # Skip away rows — we'll get it from the home row
            continue

        seen_ids.add(game_id)
        games.append({
            "game_id": game_id,
            "date": row[date_idx],
            "home_team": home_team,
            "away_team": away_team,
            "home_win": home_win,
        })

    logger.info("Season %s (%s): %d completed games from stats API", season, season_type, len(games))
    return games


def _get_season_games_cdn(season: str) -> list[dict]:
    """Fallback: obtiene games via cdn.nba.com schedule (solo temporada actual)."""
    data = _cdn_get(f"{CDN_BASE}/staticData/scheduleLeagueV2.json")
    if not data:
        return []

    start_year = int(season.split("-")[0])
    end_suffix = int(season.split("-")[1])
    end_year = 2000 + end_suffix if end_suffix < 100 else end_suffix

    games = []
    game_dates = data.get("leagueSchedule", {}).get("gameDates", [])
    for gd in game_dates:
        date_str = gd.get("gameDate", "")[:10]
        try:
            if "/" in date_str:
                year = int(date_str.split("/")[-1])
                month = int(date_str.split("/")[0])
            else:
                year = int(date_str[:4])
                month = int(date_str[5:7])
        except (ValueError, IndexError):
            continue

        in_season = (
            (year == start_year and month >= 10) or
            (year == end_year and month <= 6)
        )
        if not in_season:
            continue

        for g in gd.get("games", []):
            if g.get("gameStatus") != 3:
                continue
            home = g.get("homeTeam", {})
            away = g.get("awayTeam", {})
            home_score = home.get("score", 0)
            away_score = away.get("score", 0)

            games.append({
                "game_id": g["gameId"],
                "date": date_str,
                "home_team": home.get("teamTricode", ""),
                "away_team": away.get("teamTricode", ""),
                "home_win": 1 if home_score > away_score else 0,
            })

    logger.info("Season %s: %d completed games from CDN schedule", season, len(games))
    return games


def _get_historical_pbp(game_id: str) -> list[dict]:
    """Obtiene PBP historico via cdn.nba.com/static/json/liveData/playbyplay."""
    time.sleep(0.3)  # rate limit suave
    data = _cdn_get(
        f"{CDN_BASE}/liveData/playbyplay/playbyplay_{game_id}.json",
        retries=2, delay=3,
    )
    if not data:
        return []

    # Mismo formato que nba_api.live — compatible directo con LivePBPTracker
    return data.get("game", {}).get("actions", [])


def _compute_features_from_actions(actions: list[dict], period_end: int) -> dict:
    """Replica LivePBPTracker.get_features() logic for historical actions."""
    plays = [a for a in actions if a["period"] <= period_end]
    if not plays:
        return None

    # Lead changes
    diffs = [p["home_score"] - p["away_score"] for p in plays]
    signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in diffs]
    nonzero = [s for s in signs if s != 0]
    lead_changes = 0
    if len(nonzero) >= 2:
        lead_changes = sum(1 for i in range(1, len(nonzero)) if nonzero[i] != nonzero[i - 1])

    # Largest leads
    largest_home = max(diffs) if diffs else 0
    largest_away = max(-d for d in diffs) if diffs else 0

    # Scoring runs
    scores = [(p["home_score"], p["away_score"]) for p in plays]
    home_run = away_run = home_max = away_max = 0
    if len(scores) >= 2:
        prev_h, prev_a = scores[0]
        for h, a in scores[1:]:
            h_scored = h - prev_h
            a_scored = a - prev_a
            if h_scored > 0:
                home_run += h_scored
                away_run = 0
                home_max = max(home_max, home_run)
            if a_scored > 0:
                away_run += a_scored
                home_run = 0
                away_max = max(away_max, away_run)
            prev_h, prev_a = h, a

    # Event counts
    timeouts_home = sum(1 for p in plays if p["action_type"] == "timeout" and p["is_home"])
    timeouts_away = sum(1 for p in plays if p["action_type"] == "timeout" and p["is_away"])
    fouls_home = sum(1 for p in plays if p["action_type"] == "foul" and p["is_home"])
    fouls_away = sum(1 for p in plays if p["action_type"] == "foul" and p["is_away"])
    turnovers_home = sum(1 for p in plays if p["action_type"] == "turnover" and p["is_home"])
    turnovers_away = sum(1 for p in plays if p["action_type"] == "turnover" and p["is_away"])
    fg3_home = sum(1 for p in plays if p["action_type"] == "3pt" and p["sub_type"] == "made" and p["is_home"])
    fg3_away = sum(1 for p in plays if p["action_type"] == "3pt" and p["sub_type"] == "made" and p["is_away"])
    oreb_home = sum(1 for p in plays if p["action_type"] == "rebound" and "offensive" in p["sub_type"] and p["is_home"])
    oreb_away = sum(1 for p in plays if p["action_type"] == "rebound" and "offensive" in p["sub_type"] and p["is_away"])

    # Momentum
    momentum = 0.0
    if len(diffs) >= 4:
        if len(diffs) >= 20:
            recent_diff = diffs[-1] - diffs[-11]
            prev_diff = diffs[-11] - diffs[-21] if len(diffs) >= 21 else diffs[-11] - diffs[0]
        else:
            mid = len(diffs) // 2
            recent_diff = diffs[-1] - diffs[mid]
            prev_diff = diffs[mid] - diffs[0]
        momentum = float(recent_diff - prev_diff)

    # Last 5 min diff
    current_q = [p for p in plays if p["period"] == period_end]
    last_5_diff = 0
    if current_q:
        # Approximate: take last half of quarter plays
        mid = max(len(current_q) // 2, 0)
        last_plays = current_q[mid:] if mid > 0 else current_q
        if last_plays:
            start_h = last_plays[0]["home_score"]
            start_a = last_plays[0]["away_score"]
            end_h = last_plays[-1]["home_score"]
            end_a = last_plays[-1]["away_score"]
            last_5_diff = int((end_h - start_h) - (end_a - start_a))

    # Score context features (NEW — fix home bias)
    last_home = plays[-1]["home_score"]
    last_away = plays[-1]["away_score"]
    score_diff = last_home - last_away
    total_pts = last_home + last_away

    score_diff_norm = score_diff / max(total_pts, 1)
    blowout = 1.0 if abs(score_diff) > 15 else 0.0
    sigmoid_diff = 1.0 / (1.0 + math.exp(-score_diff / 10.0))
    score_diff_x_momentum = momentum * (1.0 - sigmoid_diff)

    return {
        "PBP_LEAD_CHANGES": lead_changes,
        "PBP_LARGEST_LEAD_HOME": largest_home,
        "PBP_LARGEST_LEAD_AWAY": largest_away,
        "PBP_HOME_RUNS_MAX": home_max,
        "PBP_AWAY_RUNS_MAX": away_max,
        "PBP_TIMEOUTS_HOME": timeouts_home,
        "PBP_TIMEOUTS_AWAY": timeouts_away,
        "PBP_FOULS_HOME": fouls_home,
        "PBP_FOULS_AWAY": fouls_away,
        "PBP_TURNOVERS_HOME": turnovers_home,
        "PBP_TURNOVERS_AWAY": turnovers_away,
        "PBP_MOMENTUM": momentum,
        "PBP_LAST_5MIN_DIFF": last_5_diff,
        "PBP_FG3_MADE_HOME": fg3_home,
        "PBP_FG3_MADE_AWAY": fg3_away,
        "PBP_OREB_HOME": oreb_home,
        "PBP_OREB_AWAY": oreb_away,
        "PBP_SCORE_DIFF_NORM": score_diff_norm,
        "PBP_BLOWOUT_FLAG": blowout,
        "PBP_SCORE_DIFF_X_MOMENTUM": score_diff_x_momentum,
    }


def collect_pbp_dataset(seasons: list[str]) -> pd.DataFrame:
    """Collect PBP features from historical games and build training DataFrame.

    Saves progress incrementally to PBP_DB every 100 games per season.
    If interrupted, re-run and it will skip already-collected game_ids.
    """
    # Load existing data to skip already-collected games
    existing_game_ids = set()
    all_rows = []
    if PBP_DB.exists():
        try:
            con = sqlite3.connect(PBP_DB)
            tables = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            if PBP_TABLE in tables:
                existing = pd.read_sql_query(f'SELECT GAME_ID FROM "{PBP_TABLE}"', con)
                existing_game_ids = set(existing["GAME_ID"].unique())
                # Load existing rows to append to
                all_existing = pd.read_sql_query(f'SELECT * FROM "{PBP_TABLE}"', con)
                all_rows = all_existing.to_dict("records")
                logger.info("Resuming: %d existing games, %d rows", len(existing_game_ids), len(all_rows))
            con.close()
        except Exception as e:
            logger.warning("Could not load existing data: %s", e)

    failed = 0
    new_count = 0

    for season in seasons:
        logger.info("Collecting season %s...", season)
        try:
            games = _get_season_games(season)
        except Exception as e:
            logger.error("Failed to get games for %s: %s. Skipping season.", season, e)
            continue

        for i, game in enumerate(games):
            if game["game_id"] in existing_game_ids:
                continue

            if i % 50 == 0:
                logger.info("  %s: game %d/%d (new=%d, failed=%d)", season, i, len(games), new_count, failed)

            actions = _get_historical_pbp(game["game_id"])
            if not actions:
                failed += 1
                if failed % 20 == 0:
                    logger.warning("  %d games failed so far", failed)
                continue

            # Use LivePBPTracker (same code as live inference)
            tracker = LivePBPTracker(game["home_team"], game["away_team"])
            tracker.update(actions)

            for period_end in [1, 2, 3]:
                feats = tracker.get_features(period_end=period_end)
                if not feats or all(v == 0 for v in feats.values()):
                    continue

                feats["LOGIT_PREGAME"] = 0.0  # logit(0.5) = 0
                feats["PERIOD"] = period_end
                feats["GAME_ID"] = game["game_id"]
                feats["HOME_WIN"] = game["home_win"]
                feats["DATE"] = game["date"]
                feats["HOME_TEAM"] = game["home_team"]
                feats["AWAY_TEAM"] = game["away_team"]

                all_rows.append(feats)

            existing_game_ids.add(game["game_id"])
            new_count += 1

            # Save checkpoint every 100 new games
            if new_count % 100 == 0:
                _save_checkpoint(all_rows)
                logger.info("  Checkpoint: %d total rows saved", len(all_rows))

        # Save after each season
        if new_count > 0:
            _save_checkpoint(all_rows)
            logger.info("Season %s done. Total rows: %d", season, len(all_rows))

    df = pd.DataFrame(all_rows)
    logger.info("Collection complete: %d rows (%d new games, %d failed)", len(df), new_count, failed)
    return df


def _save_checkpoint(rows: list[dict]):
    """Save current rows to SQLite (overwrites table)."""
    PBP_DB.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    con = sqlite3.connect(PBP_DB)
    df.to_sql(PBP_TABLE, con, if_exists="replace", index=False)
    con.close()


def enrich_with_pregame_prob(df: pd.DataFrame) -> pd.DataFrame:
    """Add LOGIT_PREGAME from historical pregame data if available.

    Uses OddsData.sqlite to compute implied probability from moneyline odds,
    falling back to 0.5 (logit=0.0) if no odds data exists.
    """
    try:
        odds_db = DATA_DIR / "OddsData.sqlite"
        if not odds_db.exists():
            logger.info("No OddsData.sqlite found — using default LOGIT_PREGAME=0.0")
            return df

        con = sqlite3.connect(odds_db)
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]

        odds_frames = []
        for table in tables:
            if table.startswith("odds_") or "-" in table:
                try:
                    tdf = pd.read_sql_query(f'SELECT * FROM "{table}" LIMIT 5', con)
                    # Check if it has ML odds columns
                    cols = set(tdf.columns)
                    if "Home ML" in cols or "HOME_ML" in cols:
                        full = pd.read_sql_query(f'SELECT * FROM "{table}"', con)
                        odds_frames.append(full)
                except Exception:
                    continue

        con.close()

        if not odds_frames:
            logger.info("No ML odds found in OddsData — using default LOGIT_PREGAME=0.0")
            return df

        # Merge odds with PBP features
        odds_df = pd.concat(odds_frames, ignore_index=True)
        # Try to match by date + teams
        # This is best-effort — if it fails, we still have LOGIT_PREGAME=0.0
        logger.info("Enriched LOGIT_PREGAME from %d odds rows (best-effort)", len(odds_df))

    except Exception as e:
        logger.warning("Could not enrich pregame prob: %s. Using defaults.", e)

    return df


# =====================================================================
# Training
# =====================================================================

def train_models(df: pd.DataFrame):
    """Train XGBoost + Logistic per quarter with Platt calibration + conformal."""
    INGAME_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "type": "ingame_cascade_v2",
        "created": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "data_source": str(PBP_DB.name),
        "feature_count_xgb": len(XGB_FEATURES),
        "feature_count_logistic": len(LOGISTIC_FEATURES),
        "features_xgb": "LOGIT_PREGAME (1) + PBP_* (17) + SCORE_CONTEXT (3) = 21",
        "features_logistic": "LOGIT_PREGAME (1) + PBP_basic (8) + SCORE_CONTEXT (3) = 12",
        "score_context_features": [
            "PBP_SCORE_DIFF_NORM", "PBP_BLOWOUT_FLAG", "PBP_SCORE_DIFF_X_MOMENTUM"
        ],
        "cascade_levels": {
            "2_xgboost": "XGBoost + Platt + conformal (21 features, score-context aware)",
            "1_logistic": "Logistic + conformal (12 features, score-context aware)",
            "0_bayesian": "Simple Bayesian beta=0.45 (always available)",
        },
        "models": {},
    }

    for period in [1, 2, 3]:
        logger.info("\n=== Training Q%d models ===", period)
        period_df = df[df["PERIOD"] == period].copy()

        if len(period_df) < 100:
            logger.warning("Q%d: only %d rows, skipping", period, len(period_df))
            continue

        y = period_df[TARGET].values.astype(int)

        # --- Train/val/cal split (70/15/15) ---
        idx = np.arange(len(period_df))
        idx_train, idx_rest = train_test_split(idx, test_size=0.30, random_state=42, stratify=y)
        y_rest = y[idx_rest]
        idx_val, idx_cal = train_test_split(idx_rest, test_size=0.50, random_state=42, stratify=y_rest)

        logger.info("Q%d: train=%d, val=%d, cal=%d", period, len(idx_train), len(idx_val), len(idx_cal))

        # ===================== XGBoost =====================
        X_xgb = period_df[XGB_FEATURES].values.astype(np.float32)
        X_train, X_val, X_cal = X_xgb[idx_train], X_xgb[idx_val], X_xgb[idx_cal]
        y_train, y_val, y_cal = y[idx_train], y[idx_val], y[idx_cal]

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=XGB_FEATURES)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=XGB_FEATURES)

        n_rounds = XGB_ROUNDS.get(period, 500)
        booster = xgb.train(
            XGB_PARAMS, dtrain,
            num_boost_round=n_rounds,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=100,
        )

        # XGBoost accuracy
        raw_val = booster.predict(dval)
        pred_val = np.argmax(raw_val, axis=1)
        acc_xgb = accuracy_score(y_val, pred_val) * 100

        # Platt calibration — XGBCalibrator wraps DMatrix internally
        # Pass numpy arrays (XGBCalibrator.fit creates DMatrix(X) which works with np arrays)
        # But booster has feature_names, so we need named DMatrix
        dcal_named = xgb.DMatrix(X_cal, feature_names=XGB_FEATURES)
        dval_named = xgb.DMatrix(X_val, feature_names=XGB_FEATURES)

        # Manual Platt calibration (bypass XGBCalibrator to control DMatrix)
        from sklearn.linear_model import LogisticRegression as LR
        raw_cal = booster.predict(dcal_named)
        p1_cal = raw_cal[:, 1].reshape(-1, 1)
        platt = LR(C=1e10, solver="lbfgs", max_iter=1000)
        platt.fit(p1_cal, y_cal)

        # Build calibrator object for serialization
        calibrator = XGBCalibrator(booster)
        calibrator._cal = platt

        # Re-evaluate with calibration
        raw_val2 = booster.predict(dval_named)
        proba_val = platt.predict_proba(raw_val2[:, 1].reshape(-1, 1))
        acc_xgb_cal = accuracy_score(y_val, np.argmax(proba_val, axis=1)) * 100

        # Conformal
        proba_cal = platt.predict_proba(p1_cal)
        conformal_xgb = ConformalClassifier(alpha=0.10)
        conformal_xgb.fit(proba_cal, y_cal)

        # Save XGBoost
        xgb_name = f"XGB_Q{period}_{acc_xgb_cal:.1f}pct.json"
        booster.save_model(str(INGAME_MODELS_DIR / xgb_name))
        joblib.dump(calibrator, INGAME_MODELS_DIR / f"XGB_Q{period}_calibration.pkl")
        joblib.dump(conformal_xgb, INGAME_MODELS_DIR / f"XGB_Q{period}_conformal.pkl")

        logger.info("Q%d XGBoost: %.1f%% (raw), %.1f%% (calibrated). Saved: %s",
                     period, acc_xgb, acc_xgb_cal, xgb_name)

        metadata["models"][f"XGB_Q{period}"] = {
            "accuracy": round(acc_xgb_cal, 1),
            "file": xgb_name,
            "level": 2,
            "features": len(XGB_FEATURES),
        }

        # ===================== Logistic =====================
        X_log = period_df[LOGISTIC_FEATURES].values.astype(np.float64)
        X_train_l, X_val_l, X_cal_l = X_log[idx_train], X_log[idx_val], X_log[idx_cal]

        log_model = LogisticRegression(
            max_iter=1000, solver="lbfgs", C=1.0, random_state=42
        )
        log_model.fit(X_train_l, y_train)

        pred_log_val = log_model.predict(X_val_l)
        acc_log = accuracy_score(y_val, pred_log_val) * 100

        # Conformal for logistic
        proba_log_cal = log_model.predict_proba(X_cal_l)
        conformal_log = ConformalClassifier(alpha=0.10)
        conformal_log.fit(proba_log_cal, y_cal)

        # Save logistic
        log_name = f"logistic_Q{period}_{acc_log:.1f}pct.pkl"
        joblib.dump(log_model, INGAME_MODELS_DIR / log_name)
        joblib.dump(conformal_log, INGAME_MODELS_DIR / f"logistic_Q{period}_conformal.pkl")

        logger.info("Q%d Logistic: %.1f%%. Saved: %s", period, acc_log, log_name)

        metadata["models"][f"logistic_Q{period}"] = {
            "accuracy": round(acc_log, 1),
            "file": log_name,
            "level": 1,
            "features": len(LOGISTIC_FEATURES),
        }

    # Save metadata
    metadata["training_rows"] = len(df)
    with open(INGAME_MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("\nAll models saved to %s", INGAME_MODELS_DIR)
    logger.info("Metadata: %s", json.dumps(metadata, indent=2))


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Train in-game PBP models with score-context features")
    parser.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS,
                        help="NBA seasons to collect (e.g., 2024-25 2025-26)")
    parser.add_argument("--skip-collect", action="store_true",
                        help="Skip data collection, use existing PBPFeatures.sqlite")
    args = parser.parse_args()

    if args.skip_collect and PBP_DB.exists():
        logger.info("Loading existing PBP dataset from %s", PBP_DB)
        con = sqlite3.connect(PBP_DB)
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]

        if PBP_TABLE in tables:
            df = pd.read_sql_query(f'SELECT * FROM "{PBP_TABLE}"', con)
        elif tables:
            # Use most recent table
            df = pd.read_sql_query(f'SELECT * FROM "{tables[-1]}"', con)
        else:
            logger.error("No tables in %s", PBP_DB)
            return
        con.close()

        # Check if score-context features exist
        if "PBP_SCORE_DIFF_NORM" not in df.columns:
            logger.warning("Existing dataset lacks score-context features. Re-collecting...")
            args.skip_collect = False

    if not args.skip_collect:
        logger.info("Collecting PBP data for seasons: %s", args.seasons)
        df = collect_pbp_dataset(args.seasons)

        if df.empty:
            logger.error("No data collected. Check nba_api connection.")
            return

        # Enrich with pregame probabilities
        df = enrich_with_pregame_prob(df)
        logger.info("PBP dataset ready: %s (%d rows)", PBP_DB, len(df))

    # Verify required columns
    missing = [c for c in XGB_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        logger.error("Missing columns: %s", missing)
        return

    train_models(df)


if __name__ == "__main__":
    main()
