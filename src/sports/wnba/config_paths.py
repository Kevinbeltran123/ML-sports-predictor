"""WNBA path constants derived from the central project config.

Import from here instead of hard-coding paths in WNBA modules.

Usage:
    from src.sports.wnba.config_paths import WNBA_ML_MODELS_DIR, WNBA_DATASET_DB
"""

from src.config import PROJECT_ROOT, MODELS_DIR, DATA_DIR  # noqa: F401 (re-exported)

TRAINING_DIR = DATA_DIR / "training"

# Model artefacts
WNBA_ML_MODELS_DIR = MODELS_DIR / "wnba" / "moneyline"

# Raw team-stat database (per game, fetched from stats.nba.com with LeagueID=10)
WNBA_TEAMS_DB = TRAINING_DIR / "WNBATeamData.sqlite"

# Pre-built feature dataset used for training
WNBA_DATASET_DB = TRAINING_DIR / "wnba_dataset.sqlite"

# Bet-tracking persistence (separate from NBA bets)
WNBA_BETS_DB = DATA_DIR / "nba" / "predictions" / "WNBABetsTracking.sqlite"
