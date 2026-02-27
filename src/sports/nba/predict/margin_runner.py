"""Margin Runner: carga modelo XGBoost de margen y predice μ per-game.

El modelo predice home_score - away_score (margen continuo).
Positivo = home gana por X puntos.
Negativo = home pierde por X puntos.

Uso:
    from src.sports.nba.predict.margin_runner import predict_margins
    margins = predict_margins(data)  # array (N,) o None si no hay modelo
"""

import re

import xgboost as xgb

from src.config import NBA_MARGIN_MODELS_DIR, get_logger

logger = get_logger(__name__)

_margin_model = None
_load_attempted = False


def _load_margin_model():
    """Carga el modelo de margen (lazy, una sola vez)."""
    global _margin_model, _load_attempted
    if _load_attempted:
        return _margin_model
    _load_attempted = True

    # Intentar via registry primero
    try:
        from src.core.models.registry import ModelRegistry
        registry = ModelRegistry()
        _margin_model = registry.load_production("margin")
        logger.info("Margin model loaded via registry")
        return _margin_model
    except (FileNotFoundError, ValueError, KeyError):
        pass

    # Fallback: buscar mejor modelo por RMSE en el nombre
    candidates = list(NBA_MARGIN_MODELS_DIR.glob("*MARGIN*.json"))
    if not candidates:
        logger.debug("No margin model found in %s", NBA_MARGIN_MODELS_DIR)
        return None

    rmse_re = re.compile(r"_(\d+(?:\.\d+)?)rmse_")
    def _extract_rmse(path):
        m = rmse_re.search(path.name)
        return float(m.group(1)) if m else 999.0

    best = min(candidates, key=_extract_rmse)
    booster = xgb.Booster()
    booster.load_model(str(best))
    _margin_model = booster
    logger.info("Margin model loaded (fallback): %s", best.name)
    return _margin_model


def predict_margins(data):
    """Predice margen esperado para cada juego.

    Args:
        data: numpy array (N, num_features), mismas features que el clasificador.

    Returns:
        numpy array (N,) con margen esperado (positivo = home gana por X).
        None si el modelo no esta disponible.
    """
    model = _load_margin_model()
    if model is None:
        return None
    return model.predict(xgb.DMatrix(data))
