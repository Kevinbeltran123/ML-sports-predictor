"""Configuracion centralizada del proyecto nba-wl-predictor.

Solo contiene lo necesario para el modelo W/L (moneyline).
Sin props, SaaS, in-game ni multi-sport.
"""

import logging
import os
import re
from pathlib import Path

# --- Logging ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    level=getattr(logging, LOG_LEVEL, logging.INFO),
)


def _redact_sensitive_text(value: str) -> str:
    """Redacta patrones comunes de secretos antes de escribir logs."""
    redacted = value
    kv_patterns = [
        re.compile(r"(?i)\b(password|passwd|pass|token|api[_-]?key|secret)\b\s*[:=]\s*([^\s,;]+)"),
        re.compile(r"(?i)\b(session|authorization)\b\s*[:=]\s*([^\s,;]+)"),
    ]
    for pattern in kv_patterns:
        redacted = pattern.sub(lambda m: f"{m.group(1)}=<redacted>", redacted)
    redacted = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}\b", "Bearer <redacted>", redacted)
    return redacted


class SensitiveDataFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
            redacted = _redact_sensitive_text(message)
            if redacted != message:
                record.msg = redacted
                record.args = ()
        except Exception:
            return True
        return True


_sensitive_filter = SensitiveDataFilter()
for _handler in logging.getLogger().handlers:
    _handler.addFilter(_sensitive_filter)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# Raiz del proyecto (1 nivel arriba de src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Directorios ---
DATA_DIR   = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# --- Modelos W/L ---
NBA_ML_MODELS_DIR = MODELS_DIR / "moneyline"

# Aliases de compatibilidad
XGBOOST_MODELS_DIR = NBA_ML_MODELS_DIR
CATBOOST_MODELS_DIR = NBA_ML_MODELS_DIR
# H1 (First Half) moneyline
NBA_H1_MODELS_DIR = MODELS_DIR / "h1moneyline"
# Totals and margin models
NBA_UO_MODELS_DIR = MODELS_DIR / "totals"
NBA_MARGIN_MODELS_DIR = MODELS_DIR / "margin"

# --- Datos de entrenamiento ---
TRAINING_DIR = DATA_DIR / "training"

# --- WNBA ---
WNBA_ML_MODELS_DIR = MODELS_DIR / "wnba" / "moneyline"
WNBA_TEAMS_DB = TRAINING_DIR / "WNBATeamData.sqlite"
WNBA_DATASET_DB = TRAINING_DIR / "wnba_dataset.sqlite"
WNBA_BETS_DB = DATA_DIR / "nba" / "predictions" / "WNBABetsTracking.sqlite"

# --- Bases de datos SQLite ---
TEAMS_DB   = TRAINING_DIR / "TeamData.sqlite"
ODDS_DB    = TRAINING_DIR / "OddsData.sqlite"
DATASET_DB = TRAINING_DIR / "dataset.sqlite"

# PLAYER_LOGS_DB: necesario para injury impact (AVAIL features)
PLAYER_LOGS_DB = TRAINING_DIR / "PlayerGameLogs.sqlite"

# --- Directorios auxiliares ---
NBA_DIR       = DATA_DIR / "nba"
SCHEDULES_DIR = NBA_DIR / "schedules"
SCRAPERS_DIR  = NBA_DIR / "scrapers"
RESEARCH_DIR  = NBA_DIR / "research"
PREDICTIONS_DIR     = NBA_DIR / "predictions"
HISTORICAL_LINES_DB = RESEARCH_DIR / "HistoricalLines.sqlite"
BETS_DB             = PREDICTIONS_DIR / "BetsTracking.sqlite"
INGAME_MODELS_DIR   = MODELS_DIR / "ingame"

# --- Polymarket ---
POLYMARKET_DB = PREDICTIONS_DIR / "PolymarketTracking.sqlite"
POLYMARKET_DEFAULT_BANKROLL = 50.0

# Bases de datos opcionales (para features avanzadas)
LINEUP_DB     = TRAINING_DIR / "LineupData.sqlite"
REFEREE_DB    = TRAINING_DIR / "RefereeData.sqlite"
ARCHETYPES_MODEL_PATH = MODELS_DIR / "archetypes"
ESPN_LINES_DB = TRAINING_DIR / "ESPNLines.sqlite"
_bref_new    = SCRAPERS_DIR / "BRefData.sqlite"
BREF_DB = (
    _bref_new
    if (_bref_new.exists() and _bref_new.stat().st_size > 1024)
    else TRAINING_DIR / "BRefData.sqlite"
)
STATHEAD_DB      = SCRAPERS_DIR / "StatheadData.sqlite"
BREF_SESSION_DIR = NBA_DIR / ".bref_session"

# --- Archivo de configuracion ---
CONFIG_PATH = PROJECT_ROOT / "config.toml"

# --- Columnas a eliminar en prediccion/entrenamiento (W/L moneyline) ---
# 158 features finales tras ablacion Feb-2026.
DROP_COLUMNS_ML = [
    # --- Metadata y targets ---
    "index", "Score", "Home-Team-Win", "H1-Home-Win", "Margin", "TEAM_NAME", "Date",
    "index.1", "TEAM_NAME.1", "Date.1", "OU-Cover", "OU",
    # --- Feature Selection original: redundancia manual ---
    "Net_Rtg", "Net_Rtg.1",
    "eFG_PCT", "eFG_PCT.1",
    "ELO_PROB",
    "FG3M", "FG3M.1",
    "W_RANK", "W_RANK.1",
    "L_RANK", "L_RANK.1",
    "GP", "GP.1",
    # --- Feature Selection automatica (Ablation Study Feb-2026) ---
    "AVAIL_AWAY", "AVAIL_HOME",
    "Avg_Pace",
    "BLKA", "BLKA.1",
    "BLK_RANK", "BLK_RANK.1",
    "DIFF_FG_PCT",
    "FG3A", "FG3A.1",
    "FG3A_RANK", "FG3A_RANK.1",
    "FGM", "FGM.1",
    "FG_PCT.1",
    "FTA", "FTA.1",
    "FTA_RANK", "FTA_RANK.1",
    "FTM",
    "FT_PCT", "FT_PCT.1",
    "FT_Rate.1",
    "MARKET_SPREAD",
    "OREB", "OREB_RANK.1",
    "PLUS_MINUS", "PLUS_MINUS.1",
    "PLUS_MINUS_RANK", "PLUS_MINUS_RANK.1",
    "PTS", "PTS.1",
    "REB", "REB_RANK.1",
    "STL", "STL.1",
    "TOV", "TOV_PCT.1", "TOV_RANK.1",
    "TS_PCT", "TS_PCT.1",
    "TZ_CHANGE_HOME",
    "W_PCT", "W_PCT.1",
    "W_PCT_RANK", "W_PCT_RANK.1",
    # --- Phase 4: Correlation pruning ---
    "ORB_PCT.1",
    "SRS_AWAY",
    "SRS_DIFF",
    # --- Phase 4: Ablation-driven group drops ---
    "SC_RA_RATE_HOME", "SC_RA_RATE_AWAY",
    "SC_RA_FG_PCT_HOME", "SC_RA_FG_PCT_AWAY",
    "SC_PAINT_RATE_HOME", "SC_PAINT_RATE_AWAY",
    "SC_MID_RATE_HOME", "SC_MID_RATE_AWAY",
    "SC_CORNER3_RATE_HOME", "SC_CORNER3_RATE_AWAY",
    "SC_AVG_DIST_HOME", "SC_AVG_DIST_AWAY",
    "ZONE_AVG_DIST_HOME", "ZONE_FG3A_RATE_HOME",
    "ZONE_PAINT_FG_PCT_HOME", "ZONE_CLOSE_MID_FG_PCT_HOME",
    "ZONE_MID_FG_PCT_HOME", "ZONE_LONG2_FG_PCT_HOME",
    "ZONE_CORNER3_PCT_HOME", "ZONE_DUNK_RATE_HOME",
    "ZONE_AVG_DIST_AWAY", "ZONE_FG3A_RATE_AWAY",
    "ZONE_PAINT_FG_PCT_AWAY", "ZONE_CLOSE_MID_FG_PCT_AWAY",
    "ZONE_MID_FG_PCT_AWAY", "ZONE_LONG2_FG_PCT_AWAY",
    "ZONE_CORNER3_PCT_AWAY", "ZONE_DUNK_RATE_AWAY",
    "ONOFF_NET_TOP5_HOME", "ONOFF_NET_TOP5_AWAY",
    "ONOFF_SPREAD_HOME", "ONOFF_SPREAD_AWAY",
    "ADV_BPM_TOP5_HOME", "ADV_BPM_TOP5_AWAY",
    "ADV_MAX_BPM_HOME", "ADV_MAX_BPM_AWAY",
    "ADV_USG_CONCENTRATION_HOME", "ADV_USG_CONCENTRATION_AWAY",
    "ADV_TS_TEAM_HOME", "ADV_TS_TEAM_AWAY",
    "LS_Q4_PCT_HOME", "LS_Q4_PCT_AWAY",
    "LS_2H_RATIO_HOME", "LS_2H_RATIO_AWAY",
    "LS_Q1_PCT_HOME", "LS_Q1_PCT_AWAY",
    "LS_SCORING_VAR_HOME", "LS_SCORING_VAR_AWAY",
    "ESPN_LINE_MOVE", "ESPN_TOTAL_MOVE",
    "ESPN_OPEN_ML_PROB", "ESPN_BOOK_DISAGREEMENT",
    # --- Phase 5: Lineup composition drops ---
    "LINEUP_DIVERSITY_HOME", "LINEUP_DIVERSITY_AWAY",
    "LINEUP_STAR_FRAC_HOME", "LINEUP_STAR_FRAC_AWAY",
    "BENCH_PPG_GAP_HOME", "BENCH_PPG_GAP_AWAY",
    "BENCH_DEPTH_HOME", "BENCH_DEPTH_AWAY",
    # --- Referee features (sin datos de entrenamiento) ---
    "REF_CREW_TOTAL_TENDENCY", "REF_CREW_OVER_PCT", "REF_CREW_HOME_WIN_PCT",
    # --- Injury granulares (ablacion Feb-2026): -1.3pp ---
    "STAR_MISSING_HOME", "STAR_MISSING_AWAY",
    "N_ROTATION_OUT_HOME", "N_ROTATION_OUT_AWAY",
    "MISSING_BPM_HOME", "MISSING_BPM_AWAY",
    "AVAIL_DIFF",
]

# Features dropped from ML but RETAINED for margin regression.
# These help predict margin SIZE even if they don't improve W/L accuracy.
# Injury impacts, star quality, line movement, and lineup depth all affect
# how large a team wins/loses by, independent of the binary W/L outcome.
_MARGIN_KEEP = {
    # Raw net rating — direct predictor of expected margin
    "Net_Rtg", "Net_Rtg.1",
    # Injury availability — missing players widen/narrow margins
    "AVAIL_AWAY", "AVAIL_HOME",
    "STAR_MISSING_HOME", "STAR_MISSING_AWAY",
    "N_ROTATION_OUT_HOME", "N_ROTATION_OUT_AWAY",
    "MISSING_BPM_HOME", "MISSING_BPM_AWAY",
    "AVAIL_DIFF",
    # Travel/timezone fatigue — affects margin more than binary W/L
    "TZ_CHANGE_HOME",
    # ESPN line movement — captures late injury/lineup info
    "ESPN_LINE_MOVE", "ESPN_TOTAL_MOVE",
    "ESPN_OPEN_ML_PROB", "ESPN_BOOK_DISAGREEMENT",
    # Advanced player metrics — star power predicts blowout probability
    "ADV_BPM_TOP5_HOME", "ADV_BPM_TOP5_AWAY",
    "ADV_MAX_BPM_HOME", "ADV_MAX_BPM_AWAY",
    "ADV_USG_CONCENTRATION_HOME", "ADV_USG_CONCENTRATION_AWAY",
    "ADV_TS_TEAM_HOME", "ADV_TS_TEAM_AWAY",
    # On/off net rating — lineup cohesion affects margin magnitude
    "ONOFF_NET_TOP5_HOME", "ONOFF_NET_TOP5_AWAY",
    "ONOFF_SPREAD_HOME", "ONOFF_SPREAD_AWAY",
    # Line score patterns — Q4/late-game scoring predicts final margin
    "LS_Q4_PCT_HOME", "LS_Q4_PCT_AWAY",
    "LS_2H_RATIO_HOME", "LS_2H_RATIO_AWAY",
    "LS_Q1_PCT_HOME", "LS_Q1_PCT_AWAY",
    "LS_SCORING_VAR_HOME", "LS_SCORING_VAR_AWAY",
    # Lineup/bench depth — depth matters in blowouts
    "LINEUP_DIVERSITY_HOME", "LINEUP_DIVERSITY_AWAY",
    "LINEUP_STAR_FRAC_HOME", "LINEUP_STAR_FRAC_AWAY",
    "BENCH_PPG_GAP_HOME", "BENCH_PPG_GAP_AWAY",
    "BENCH_DEPTH_HOME", "BENCH_DEPTH_AWAY",
    # Strength of schedule — SRS predicts expected margin vs opponent quality
    "SRS_AWAY", "SRS_DIFF",
    "ORB_PCT.1",
    # Shot chart — paint/RA efficiency predicts margin via high-percentage scoring
    "SC_RA_RATE_HOME", "SC_RA_RATE_AWAY",
    "SC_RA_FG_PCT_HOME", "SC_RA_FG_PCT_AWAY",
    "SC_PAINT_RATE_HOME", "SC_PAINT_RATE_AWAY",
    "ZONE_AVG_DIST_HOME", "ZONE_FG3A_RATE_HOME",
    "ZONE_PAINT_FG_PCT_HOME", "ZONE_CLOSE_MID_FG_PCT_HOME",
    "ZONE_AVG_DIST_AWAY", "ZONE_FG3A_RATE_AWAY",
    "ZONE_PAINT_FG_PCT_AWAY", "ZONE_CLOSE_MID_FG_PCT_AWAY",
}

# Drop list for margin regression: same as ML but retains margin-specific signals.
# "Residual" and "MARKET_SPREAD" handled separately in train_margin_models.py.
DROP_COLUMNS_MARGIN = [c for c in DROP_COLUMNS_ML if c not in _MARGIN_KEEP]
