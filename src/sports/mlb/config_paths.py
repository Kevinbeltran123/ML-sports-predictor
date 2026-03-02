"""MLB path constants derived from the central project config.

Usage:
    from src.sports.mlb.config_paths import MLB_ML_MODELS_DIR, MLB_DATASET_DB
"""

from src.config import PROJECT_ROOT, MODELS_DIR, DATA_DIR  # noqa: F401

MLB_DATA_DIR = DATA_DIR / "mlb"
MLB_TRAINING_DIR = MLB_DATA_DIR / "training"
MLB_PREDICTIONS_DIR = MLB_DATA_DIR / "predictions"

# Model artefacts
MLB_ML_MODELS_DIR = MODELS_DIR / "mlb" / "moneyline"
MLB_F5_MODELS_DIR = MODELS_DIR / "mlb" / "f5"
MLB_TOTALS_MODELS_DIR = MODELS_DIR / "mlb" / "totals"

# Raw data databases
MLB_TEAMS_DB = MLB_TRAINING_DIR / "MLBTeamData.sqlite"
MLB_PITCHER_DB = MLB_TRAINING_DIR / "MLBPitcherData.sqlite"
MLB_ODDS_DB = MLB_TRAINING_DIR / "MLBOddsData.sqlite"
MLB_PARK_DB = MLB_TRAINING_DIR / "MLBParkFactors.sqlite"
MLB_UMPIRE_DB = MLB_TRAINING_DIR / "MLBUmpireData.sqlite"

# Pre-built feature dataset
MLB_DATASET_DB = MLB_TRAINING_DIR / "mlb_dataset.sqlite"

# Bet tracking
MLB_BETS_DB = MLB_PREDICTIONS_DIR / "MLBBetsTracking.sqlite"
