import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from colorama import Fore, Style, init, deinit
from src.core.betting import expected_value as Expected_Value
from src.core.betting import kelly_criterion as kc


init()

from src.config import XGBOOST_MODELS_DIR as MODEL_DIR, NBA_UO_MODELS_DIR, get_logger
ACCURACY_PATTERN = re.compile(r"XGBoost_(\d+(?:\.\d+)?)%_")

logger = get_logger(__name__)

xgb_ml = None
xgb_uo = None
xgb_ml_calibrator = None
xgb_uo_calibrator = None


def _select_model_path(kind):
    # ML models en moneyline/, UO models en totals/
    search_dir = NBA_UO_MODELS_DIR if kind == "UO" else MODEL_DIR
    candidates = list(search_dir.glob(f"*{kind}*.json"))
    if not candidates:
        raise FileNotFoundError(f"No XGBoost {kind} model found in {search_dir}")

    def score(path):
        match = ACCURACY_PATTERN.search(path.name)
        accuracy = float(match.group(1)) if match else 0.0
        return (path.stat().st_mtime, accuracy)

    return max(candidates, key=score)


def _load_calibrator(model_path):
    calibration_path = model_path.with_name(f"{model_path.stem}_calibration.pkl")
    if not calibration_path.exists():
        return None
    try:
        return joblib.load(calibration_path)
    except Exception:
        return None


def _load_models():
    """Carga modelos ML y UO usando el registry cuando esta disponible.

    Orden de preferencia:
    1. registry.load_production() — activa validacion de FeatureMismatchError
    2. _select_model_path() — fallback si el registry no esta inicializado aun

    El fallback asegura compatibilidad durante el primer setup (antes de bootstrap_production).
    """
    global xgb_ml, xgb_uo, xgb_ml_calibrator, xgb_uo_calibrator

    if xgb_ml is None:
        xgb_ml = _load_ml_via_registry()

    if xgb_uo is None:
        xgb_uo = _load_uo_via_registry()


def _load_ml_via_registry():
    """Carga modelo ML a traves del registry (con validacion de features)."""
    from src.core.models.registry import ModelRegistry
    from src.core.models.feature_schema import FeatureMismatchError

    registry = ModelRegistry()
    try:
        model = registry.load_production("moneyline")
        logger.info("Modelo ML cargado via registry (produccion).")

        # Cargar calibracion: buscar en production/ primero
        from src.config import NBA_ML_MODELS_DIR
        metadata = registry.get_metadata("moneyline")
        model_file = metadata.get("production", {}).get("model_file", "")
        if model_file:
            # Buscar calibracion en production/ o flat
            for search_dir in [NBA_ML_MODELS_DIR / "production", NBA_ML_MODELS_DIR]:
                cal_path = search_dir / (Path(model_file).stem + "_calibration.pkl")
                if cal_path.exists():
                    global xgb_ml_calibrator
                    xgb_ml_calibrator = joblib.load(cal_path)
                    break

        return model

    except FileNotFoundError as exc:
        logger.warning(
            "Registry ML no inicializado, usando _select_model_path() como fallback: %s", exc
        )
        return _load_ml_fallback()
    except FeatureMismatchError:
        # Re-lanzar: es un error critico de seguridad
        raise
    except Exception as exc:
        logger.warning(
            "Error cargando via registry ML, usando fallback: %s", exc
        )
        return _load_ml_fallback()


def _load_ml_fallback():
    """Fallback: carga ML usando _select_model_path() (sin validacion de features)."""
    global xgb_ml_calibrator
    ml_path = _select_model_path("ML")
    model = xgb.Booster()
    model.load_model(str(ml_path))
    xgb_ml_calibrator = _load_calibrator(ml_path)
    logger.warning(
        "Fallback: modelo ML cargado desde %s sin validacion de features.", ml_path.name
    )
    return model


def _load_uo_via_registry():
    """Carga modelo UO a traves del registry (con validacion de features)."""
    from src.core.models.registry import ModelRegistry
    from src.core.models.feature_schema import FeatureMismatchError

    registry = ModelRegistry()
    try:
        model = registry.load_production("totals")
        logger.info("Modelo UO cargado via registry (produccion).")

        from src.config import NBA_UO_MODELS_DIR
        metadata = registry.get_metadata("totals")
        model_file = metadata.get("production", {}).get("model_file", "")
        if model_file:
            for search_dir in [NBA_UO_MODELS_DIR / "production", NBA_UO_MODELS_DIR]:
                cal_path = search_dir / (Path(model_file).stem + "_calibration.pkl")
                if cal_path.exists():
                    global xgb_uo_calibrator
                    xgb_uo_calibrator = joblib.load(cal_path)
                    break

        return model

    except FileNotFoundError as exc:
        logger.warning(
            "Registry UO no inicializado, usando _select_model_path() como fallback: %s", exc
        )
        return _load_uo_fallback()
    except FeatureMismatchError:
        raise
    except Exception as exc:
        logger.warning(
            "Error cargando via registry UO, usando fallback: %s", exc
        )
        return _load_uo_fallback()


def _load_uo_fallback():
    """Fallback: carga UO usando _select_model_path() (sin validacion de features)."""
    global xgb_uo_calibrator
    uo_path = _select_model_path("UO")
    model = xgb.Booster()
    model.load_model(str(uo_path))
    xgb_uo_calibrator = _load_calibrator(uo_path)
    logger.warning(
        "Fallback: modelo UO cargado desde %s sin validacion de features.", uo_path.name
    )
    return model


def _predict_probs(model, data, calibrator=None):
    if calibrator is not None:
        return calibrator.predict_proba(data)
    return model.predict(xgb.DMatrix(data))


def _format_game_line(home_team, away_team, winner_is_home, winner_confidence, under_over, ou_value, ou_confidence):
    winner_team = home_team if winner_is_home else away_team
    loser_team = away_team if winner_is_home else home_team
    winner_color = Fore.GREEN if winner_is_home else Fore.RED
    loser_color = Fore.RED if winner_is_home else Fore.GREEN
    ou_label = "UNDER" if under_over == 0 else "OVER"
    ou_color = Fore.MAGENTA if under_over == 0 else Fore.BLUE
    return (
        f"{winner_color}{winner_team}{Style.RESET_ALL}"
        f"{Fore.CYAN} ({winner_confidence}%)"
        f"{Style.RESET_ALL} vs {loser_color}{loser_team}{Style.RESET_ALL}: "
        f"{ou_color}{ou_label} {Style.RESET_ALL}{ou_value}"
        f"{Style.RESET_ALL}{Fore.CYAN} ({ou_confidence}%)"
        f"{Style.RESET_ALL}"
    )


def _print_expected_value(
    games,
    ml_predictions_array,
    home_team_odds,
    away_team_odds,
    kelly_criterion,
):
    if kelly_criterion:
        print("------------Expected Value & Kelly Criterion-----------")
    else:
        print("---------------------Expected Value--------------------")
    for idx, game in enumerate(games):
        home_team, away_team = game
        ev_home = ev_away = 0
        if home_team_odds[idx] and away_team_odds[idx]:
            ev_home = float(
                Expected_Value.expected_value(
                    ml_predictions_array[idx][1],
                    int(home_team_odds[idx]),
                )
            )
            ev_away = float(
                Expected_Value.expected_value(
                    ml_predictions_array[idx][0],
                    int(away_team_odds[idx]),
                )
            )
        expected_value_colors = {
            "home_color": Fore.GREEN if ev_home > 0 else Fore.RED,
            "away_color": Fore.GREEN if ev_away > 0 else Fore.RED,
        }
        bankroll_descriptor = " Fraction of Bankroll: "
        bankroll_fraction_home = bankroll_descriptor + str(
            kc.calculate_kelly_criterion(home_team_odds[idx], ml_predictions_array[idx][1])
        ) + "%"
        bankroll_fraction_away = bankroll_descriptor + str(
            kc.calculate_kelly_criterion(away_team_odds[idx], ml_predictions_array[idx][0])
        ) + "%"

        print(
            home_team
            + " EV: "
            + expected_value_colors["home_color"]
            + str(ev_home)
            + Style.RESET_ALL
            + (bankroll_fraction_home if kelly_criterion else "")
        )
        print(
            away_team
            + " EV: "
            + expected_value_colors["away_color"]
            + str(ev_away)
            + Style.RESET_ALL
            + (bankroll_fraction_away if kelly_criterion else "")
        )


def xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion):
    _load_models()

    frame_uo = frame_ml.copy()
    frame_uo["OU"] = np.asarray(todays_games_uo, dtype=float)

    try:
        ml_predictions_array = _predict_probs(xgb_ml, data, xgb_ml_calibrator)
        ou_predictions_array = _predict_probs(
            xgb_uo,
            frame_uo.values.astype(float),
            xgb_uo_calibrator,
        )

        for idx, game in enumerate(games):
            home_team, away_team = game
            winner = int(np.argmax(ml_predictions_array[idx]))
            under_over = int(np.argmax(ou_predictions_array[idx]))
            winner_confidence = round(ml_predictions_array[idx][winner] * 100, 1)
            ou_confidence = round(ou_predictions_array[idx][under_over] * 100, 1)

            print(
                _format_game_line(
                    home_team,
                    away_team,
                    winner_is_home=(winner == 1),
                    winner_confidence=winner_confidence,
                    under_over=under_over,
                    ou_value=todays_games_uo[idx],
                    ou_confidence=ou_confidence,
                )
            )

        _print_expected_value(
            games,
            ml_predictions_array,
            home_team_odds,
            away_team_odds,
            kelly_criterion,
        )
    finally:
        deinit()
