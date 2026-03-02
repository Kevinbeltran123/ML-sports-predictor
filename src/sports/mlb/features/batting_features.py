"""Team-level batting features for MLB W/L predictor.

Dual rolling windows (15-game and 30-game) capture both recent form and
stable trend.  MOMENTUM_OPS = OPS_ROLL15 - OPS_ROLL30 signals whether a
lineup is currently running hot or cold relative to its baseline.

32 features total (16 per side):
    OPS_ROLL15, OPS_ROLL30, AVG_ROLL15, OBP_ROLL15, SLG_ROLL15, ISO_ROLL15,
    K_PCT_BAT, BB_PCT_BAT, HR_RATE, RUN_RATE, LOB_RATE, H_RATE,
    MOMENTUM_OPS, RBI_RATE, AB_PER_GAME, ERRORS_RATE

Usage (training):
    batting_lookup = build_batting_lookup(teams_db_path)
    df = add_batting_features_to_frame(df, batting_lookup)

Usage (live):
    feats = get_team_batting_features(batting_lookup, team_name, game_date)
"""

import sqlite3
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.config import get_logger

logger = get_logger(__name__)

# League-average defaults — used when no history is available
_BAT_DEFAULTS: Dict[str, float] = {
    "OPS_ROLL15": 0.720,
    "OPS_ROLL30": 0.720,
    "AVG_ROLL15": 0.250,
    "OBP_ROLL15": 0.320,
    "SLG_ROLL15": 0.400,
    "ISO_ROLL15": 0.150,
    "K_PCT_BAT": 0.220,
    "BB_PCT_BAT": 0.085,
    "HR_RATE": 1.10,
    "RUN_RATE": 4.50,
    "LOB_RATE": 5.80,
    "H_RATE": 8.50,
    "MOMENTUM_OPS": 0.0,
    "RBI_RATE": 4.20,
    "AB_PER_GAME": 34.0,
    "ERRORS_RATE": 0.60,
}

_WIN_15 = 15
_WIN_30 = 30


def _parse_date(date_val) -> Optional[datetime.date]:
    if isinstance(date_val, str):
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
            try:
                return datetime.strptime(date_val[:10], fmt).date()
            except ValueError:
                continue
        return None
    if hasattr(date_val, "date"):
        return date_val.date() if callable(getattr(date_val, "date")) else date_val
    return None


def build_batting_lookup(teams_db_path: str) -> Dict:
    """Build team batting history from MLBTeamData.sqlite.

    Returns:
        {team_name: [(game_date, stat_dict), ...]}
        Sorted chronologically.  stat_dict contains:
            R, H, HR, RBI, BB, SO, AB, AVG, OBP, SLG, OPS, LOB, ERRORS
    """
    lookup: Dict[str, List[Tuple]] = defaultdict(list)

    try:
        con = sqlite3.connect(teams_db_path)
    except Exception as exc:
        logger.warning("Cannot open teams DB %s: %s", teams_db_path, exc)
        return lookup

    try:
        cursor = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            if "team" not in table.lower():
                continue

            try:
                col_info = con.execute(f'PRAGMA table_info("{table}")').fetchall()
                col_names = [d[0].upper() for d in col_info]
                rows = con.execute(f'SELECT * FROM "{table}"').fetchall()
            except Exception:
                continue

            def _col(name: str) -> Optional[int]:
                try:
                    return col_names.index(name.upper())
                except ValueError:
                    return None

            idx_team = _col("TEAM") or _col("TEAM_NAME")
            idx_date = _col("GAME_DATE") or _col("DATE")
            if idx_team is None or idx_date is None:
                continue

            def _safe(idx):
                if idx is None:
                    return None
                try:
                    v = rows  # placeholder; overridden inside loop
                    return idx
                except Exception:
                    return None

            idx_r = _col("R") or _col("RUNS")
            idx_h = _col("H") or _col("HITS")
            idx_hr = _col("HR")
            idx_rbi = _col("RBI")
            idx_bb = _col("BB") or _col("WALKS")
            idx_so = _col("SO") or _col("K") or _col("STRIKEOUTS")
            idx_ab = _col("AB")
            idx_avg = _col("AVG") or _col("BA")
            idx_obp = _col("OBP")
            idx_slg = _col("SLG")
            idx_ops = _col("OPS")
            idx_lob = _col("LOB")
            idx_err = _col("ERRORS") or _col("E")

            for row in rows:
                team = row[idx_team]
                if not isinstance(team, str):
                    continue
                team = team.strip()
                game_date = _parse_date(row[idx_date])
                if game_date is None:
                    continue

                def _val(idx):
                    if idx is None:
                        return None
                    try:
                        v = row[idx]
                        return float(v) if v is not None else None
                    except (TypeError, ValueError):
                        return None

                stats = {
                    "R": _val(idx_r),
                    "H": _val(idx_h),
                    "HR": _val(idx_hr),
                    "RBI": _val(idx_rbi),
                    "BB": _val(idx_bb),
                    "SO": _val(idx_so),
                    "AB": _val(idx_ab),
                    "AVG": _val(idx_avg),
                    "OBP": _val(idx_obp),
                    "SLG": _val(idx_slg),
                    "OPS": _val(idx_ops),
                    "LOB": _val(idx_lob),
                    "ERRORS": _val(idx_err),
                }
                lookup[team].append((game_date, stats))

        for team in lookup:
            lookup[team].sort(key=lambda x: x[0])

    finally:
        con.close()

    logger.info("Batting lookup built: %d teams", len(lookup))
    return lookup


def get_team_batting_features(
    batting_lookup: Dict,
    team_name: str,
    game_date,
) -> Dict[str, float]:
    """Compute rolling batting features for a team before game_date.

    Uses games strictly before game_date to avoid leakage.

    Returns dict with 16 batting feature keys (no HOME/AWAY suffix here;
    callers apply the suffix when merging home/away).
    """
    result = dict(_BAT_DEFAULTS)

    if not team_name:
        return result

    gd = _parse_date(game_date) if not hasattr(game_date, "year") else game_date
    if gd is None:
        return result

    history = batting_lookup.get(team_name, [])
    if not history:
        return result

    # Games strictly before game_date
    prior = [(d, s) for d, s in history if d < gd]
    if not prior:
        return result

    def _vals(key: str, games) -> List[float]:
        return [s[key] for _, s in games if s.get(key) is not None]

    def _mean(vals: List[float], n: int) -> Optional[float]:
        if not vals:
            return None
        window = vals[-n:]
        return sum(window) / len(window)

    # Rolling window helpers
    def _roll(key: str, n: int) -> Optional[float]:
        return _mean(_vals(key, prior), n)

    # OPS and components (15-game rolling)
    ops15 = _roll("OPS", _WIN_15)
    ops30 = _roll("OPS", _WIN_30)
    avg15 = _roll("AVG", _WIN_15)
    obp15 = _roll("OBP", _WIN_15)
    slg15 = _roll("SLG", _WIN_15)

    if ops15 is not None:
        result["OPS_ROLL15"] = round(ops15, 4)
    if ops30 is not None:
        result["OPS_ROLL30"] = round(ops30, 4)
    if avg15 is not None:
        result["AVG_ROLL15"] = round(avg15, 4)
    if obp15 is not None:
        result["OBP_ROLL15"] = round(obp15, 4)
    if slg15 is not None:
        result["SLG_ROLL15"] = round(slg15, 4)
    if slg15 is not None and avg15 is not None:
        result["ISO_ROLL15"] = round(slg15 - avg15, 4)

    # K% and BB% from raw counts (30-game window)
    so_30 = _vals("SO", prior)[-_WIN_30:]
    ab_30 = _vals("AB", prior)[-_WIN_30:]
    bb_30 = _vals("BB", prior)[-_WIN_30:]

    if so_30 and ab_30:
        total_ab = sum(ab_30)
        if total_ab > 0:
            result["K_PCT_BAT"] = round(sum(so_30) / total_ab, 4)
    if bb_30 and ab_30:
        total_ab30 = sum(ab_30)
        total_bb = sum(bb_30)
        denom = total_ab30 + total_bb
        if denom > 0:
            result["BB_PCT_BAT"] = round(total_bb / denom, 4)

    # Rate stats (per game, 30-day)
    hr_r = _roll("HR", _WIN_30)
    run_r = _roll("R", _WIN_30)
    lob_r = _roll("LOB", _WIN_30)
    rbi_r = _roll("RBI", _WIN_30)
    ab_per = _roll("AB", _WIN_30)
    err_r = _roll("ERRORS", _WIN_30)

    if hr_r is not None:
        result["HR_RATE"] = round(hr_r, 3)
    if run_r is not None:
        result["RUN_RATE"] = round(run_r, 3)
    if lob_r is not None:
        result["LOB_RATE"] = round(lob_r, 3)
    if rbi_r is not None:
        result["RBI_RATE"] = round(rbi_r, 3)
    if ab_per is not None:
        result["AB_PER_GAME"] = round(ab_per, 2)
    if err_r is not None:
        result["ERRORS_RATE"] = round(err_r, 3)

    # Hits rate (15-game)
    h_r15 = _roll("H", _WIN_15)
    if h_r15 is not None:
        result["H_RATE"] = round(h_r15, 3)

    # Momentum: OPS_ROLL15 - OPS_ROLL30 (positive = hot streak)
    if ops15 is not None and ops30 is not None:
        result["MOMENTUM_OPS"] = round(ops15 - ops30, 4)

    return result


def add_batting_features_to_frame(
    df: pd.DataFrame,
    batting_lookup: Dict,
) -> pd.DataFrame:
    """Add rolling batting features for home and away teams to training df.

    Expects columns HOME_TEAM / AWAY_TEAM (or TEAM_HOME/TEAM_AWAY) and
    a date column (GAME_DATE or DATE).

    Returns DataFrame with 32 new batting columns (_HOME / _AWAY suffix).
    """
    date_col = next(
        (c for c in ("GAME_DATE", "Date", "DATE", "date") if c in df.columns), None
    )
    home_col = next(
        (c for c in ("HOME_TEAM", "TEAM_HOME", "Home", "HOME") if c in df.columns), None
    )
    away_col = next(
        (c for c in ("AWAY_TEAM", "TEAM_AWAY", "Away", "AWAY") if c in df.columns), None
    )

    if date_col is None or home_col is None or away_col is None:
        logger.warning(
            "add_batting_features_to_frame: required columns not found — returning unchanged df"
        )
        return df

    records = []
    for _, row in df.iterrows():
        home_feats = get_team_batting_features(
            batting_lookup, row.get(home_col, ""), row[date_col]
        )
        away_feats = get_team_batting_features(
            batting_lookup, row.get(away_col, ""), row[date_col]
        )
        merged = {}
        for k, v in home_feats.items():
            merged[f"{k}_HOME"] = v
        for k, v in away_feats.items():
            merged[f"{k}_AWAY"] = v
        records.append(merged)

    if not records:
        return df

    bat_df = pd.DataFrame(records, index=df.index)
    result = pd.concat([df, bat_df], axis=1)
    logger.info(
        "add_batting_features_to_frame: added %d batting columns for %d rows",
        len(bat_df.columns), len(df),
    )
    return result
