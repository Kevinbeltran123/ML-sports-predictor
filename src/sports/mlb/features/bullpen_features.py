"""Bullpen features for MLB W/L predictor.

Bullpen quality and fatigue matter in close games.  A team whose bullpen
has thrown 20+ IP in the last 7 days (high BP_IP_7D / BP_USAGE_3D) is
significantly more vulnerable in the 6th inning onward.

16 features total (8 per side):
    BP_ERA_7D, BP_ERA_30D, BP_WHIP_7D, BP_IP_7D,
    BP_K_PCT, BP_BB_PCT, BP_HR_RATE, BP_USAGE_3D

Data source: MLBTeamData.sqlite — expects columns BULLPEN_IP, BULLPEN_ER,
BULLPEN_H, BULLPEN_BB, BULLPEN_SO, BULLPEN_HR per game row.

Usage (training):
    bullpen_lookup = build_bullpen_lookup(teams_db_path)
    df = add_bullpen_features_to_frame(df, bullpen_lookup)

Usage (live):
    feats = get_bullpen_features(bullpen_lookup, team_name, game_date)
"""

import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.config import get_logger

logger = get_logger(__name__)

# League-average defaults
_BP_DEFAULTS: Dict[str, float] = {
    "BP_ERA_7D": 4.20,
    "BP_ERA_30D": 4.20,
    "BP_WHIP_7D": 1.30,
    "BP_IP_7D": 12.0,
    "BP_K_PCT": 0.245,
    "BP_BB_PCT": 0.095,
    "BP_HR_RATE": 1.20,
    "BP_USAGE_3D": 4.5,
}

_WIN_7 = 7    # days
_WIN_30 = 30  # days
_WIN_3 = 3    # days for usage signal


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


def _era_from_er_ip(er_list: List[float], ip_list: List[float]) -> Optional[float]:
    """Compute ERA = 9 * sum(ER) / sum(IP).  Returns None if total IP == 0."""
    total_ip = sum(ip_list)
    if total_ip <= 0:
        return None
    return 9.0 * sum(er_list) / total_ip


def _whip_from_counts(h_list, bb_list, ip_list) -> Optional[float]:
    """Compute WHIP = (H + BB) / IP."""
    total_ip = sum(ip_list)
    if total_ip <= 0:
        return None
    return (sum(h_list) + sum(bb_list)) / total_ip


def build_bullpen_lookup(teams_db_path: str) -> Dict:
    """Build per-team bullpen history from MLBTeamData.sqlite.

    Returns:
        {team_name: [(game_date, {BP_IP, BP_ER, BP_H, BP_BB, BP_SO, BP_HR}), ...]}
        Sorted chronologically.
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

            # Bullpen-specific columns
            idx_bp_ip = _col("BULLPEN_IP") or _col("BP_IP")
            idx_bp_er = _col("BULLPEN_ER") or _col("BP_ER")
            idx_bp_h = _col("BULLPEN_H") or _col("BP_H")
            idx_bp_bb = _col("BULLPEN_BB") or _col("BP_BB")
            idx_bp_so = _col("BULLPEN_SO") or _col("BP_SO") or _col("BULLPEN_K")
            idx_bp_hr = _col("BULLPEN_HR") or _col("BP_HR")

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

                bp_stats = {
                    "BP_IP": _val(idx_bp_ip),
                    "BP_ER": _val(idx_bp_er),
                    "BP_H": _val(idx_bp_h),
                    "BP_BB": _val(idx_bp_bb),
                    "BP_SO": _val(idx_bp_so),
                    "BP_HR": _val(idx_bp_hr),
                }
                lookup[team].append((game_date, bp_stats))

        for team in lookup:
            lookup[team].sort(key=lambda x: x[0])

    finally:
        con.close()

    logger.info("Bullpen lookup built: %d teams", len(lookup))
    return lookup


def get_bullpen_features(
    bullpen_lookup: Dict,
    team_name: str,
    game_date,
) -> Dict[str, float]:
    """Compute bullpen features for a team before game_date.

    All windows use games strictly before game_date (no leakage).

    Returns dict with 8 BP feature keys (no suffix; callers add _HOME/_AWAY).
    """
    result = dict(_BP_DEFAULTS)

    if not team_name:
        return result

    gd = _parse_date(game_date) if not hasattr(game_date, "year") else game_date
    if gd is None:
        return result

    history = bullpen_lookup.get(team_name, [])
    if not history:
        return result

    cutoff_7 = gd - timedelta(days=_WIN_7)
    cutoff_30 = gd - timedelta(days=_WIN_30)
    cutoff_3 = gd - timedelta(days=_WIN_3)

    # Games in each window (strictly before game_date)
    games_7 = [(d, s) for d, s in history if cutoff_7 <= d < gd]
    games_30 = [(d, s) for d, s in history if cutoff_30 <= d < gd]
    games_3 = [(d, s) for d, s in history if cutoff_3 <= d < gd]

    def _collect(games, key):
        return [s[key] for _, s in games if s.get(key) is not None]

    # --- 7-day ERA ---
    er_7 = _collect(games_7, "BP_ER")
    ip_7 = _collect(games_7, "BP_IP")
    if er_7 and ip_7 and len(er_7) == len(ip_7):
        era_7 = _era_from_er_ip(er_7, ip_7)
        if era_7 is not None:
            result["BP_ERA_7D"] = round(era_7, 3)

    # --- 30-day ERA ---
    er_30 = _collect(games_30, "BP_ER")
    ip_30 = _collect(games_30, "BP_IP")
    if er_30 and ip_30 and len(er_30) == len(ip_30):
        era_30 = _era_from_er_ip(er_30, ip_30)
        if era_30 is not None:
            result["BP_ERA_30D"] = round(era_30, 3)

    # --- 7-day WHIP ---
    h_7 = _collect(games_7, "BP_H")
    bb_7 = _collect(games_7, "BP_BB")
    if h_7 and bb_7 and ip_7:
        whip_7 = _whip_from_counts(h_7, bb_7, ip_7)
        if whip_7 is not None:
            result["BP_WHIP_7D"] = round(whip_7, 3)

    # --- IP in last 7 days (usage/fatigue signal) ---
    if ip_7:
        result["BP_IP_7D"] = round(sum(ip_7), 1)

    # --- K% and BB% from 30-day counts ---
    so_30 = _collect(games_30, "BP_SO")
    bb_30 = _collect(games_30, "BP_BB")
    ab_proxy = ip_30  # IP as batter-faced proxy; not ideal but available

    if so_30 and ip_30:
        # BF ≈ IP * 4.3 (rough — use actual SO/BF when available)
        total_ip30 = sum(ip_30)
        bf_approx = total_ip30 * 4.3
        if bf_approx > 0:
            result["BP_K_PCT"] = round(sum(so_30) / bf_approx, 4)

    if bb_30 and ip_30:
        total_ip30 = sum(ip_30)
        bf_approx = total_ip30 * 4.3
        if bf_approx > 0:
            result["BP_BB_PCT"] = round(sum(bb_30) / bf_approx, 4)

    # --- HR per 9 IP (30-day) ---
    hr_30 = _collect(games_30, "BP_HR")
    if hr_30 and ip_30:
        total_ip30 = sum(ip_30)
        if total_ip30 > 0:
            result["BP_HR_RATE"] = round(9.0 * sum(hr_30) / total_ip30, 3)

    # --- Usage in last 3 days (availability signal) ---
    ip_3 = _collect(games_3, "BP_IP")
    if ip_3:
        result["BP_USAGE_3D"] = round(sum(ip_3), 1)
    else:
        result["BP_USAGE_3D"] = 0.0

    return result


def add_bullpen_features_to_frame(
    df: pd.DataFrame,
    bullpen_lookup: Dict,
) -> pd.DataFrame:
    """Add bullpen features for both home and away teams to training df.

    Expects HOME_TEAM / AWAY_TEAM (or TEAM_HOME/TEAM_AWAY) and a date column.

    Returns DataFrame with 16 new BP columns (_HOME / _AWAY suffix).
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
            "add_bullpen_features_to_frame: required columns not found — returning unchanged df"
        )
        return df

    records = []
    for _, row in df.iterrows():
        home_feats = get_bullpen_features(
            bullpen_lookup, row.get(home_col, ""), row[date_col]
        )
        away_feats = get_bullpen_features(
            bullpen_lookup, row.get(away_col, ""), row[date_col]
        )
        merged = {}
        for k, v in home_feats.items():
            merged[f"{k}_HOME"] = v
        for k, v in away_feats.items():
            merged[f"{k}_AWAY"] = v
        records.append(merged)

    if not records:
        return df

    bp_df = pd.DataFrame(records, index=df.index)
    result = pd.concat([df, bp_df], axis=1)
    logger.info(
        "add_bullpen_features_to_frame: added %d BP columns for %d rows",
        len(bp_df.columns), len(df),
    )
    return result
