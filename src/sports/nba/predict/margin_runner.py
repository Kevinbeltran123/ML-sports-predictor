"""Margin Runner: carga modelos XGBoost + CatBoost de margen y predice per-game.

Camino B: el modelo predice Residual = Margin + MARKET_SPREAD (spread residual).
  Residual > 0 → home covers.  Residual < 0 → away covers.
  Para obtener raw margin: margin = residual - spread.

Legacy: si metadata.target != "residual", el modelo predice raw margin.

Ensemble: XGB 60% + CatBoost 40% (mismos pesos que moneyline).

Uso:
    from src.sports.nba.predict.margin_runner import predict_margins, predict_margin_sigma, is_residual_model
    preds = predict_margins(data)  # array (N,) — residuals or raw margins
    sigmas = predict_margin_sigma(spreads)  # sigma calibrado por bucket
    if is_residual_model():
        raw_margin = preds - spread  # convert back
"""

import json
import re

import joblib
import numpy as np
import xgboost as xgb

from src.config import NBA_MARGIN_MODELS_DIR, get_logger

logger = get_logger(__name__)

_xgb_margin = None
_cat_margin = None
_conformal_reg = None
_sigma_buckets = None
_is_residual = False
_load_attempted = False

W_XGB = 0.60
W_CAT = 0.40

# Fallback global sigma (NBA historical)
FALLBACK_SIGMA = 14.3


def _load_margin_models():
    """Carga modelos de margen XGBoost + CatBoost (lazy, una sola vez)."""
    global _xgb_margin, _cat_margin, _conformal_reg, _sigma_buckets, _is_residual, _load_attempted
    if _load_attempted:
        return _xgb_margin is not None
    _load_attempted = True

    if not NBA_MARGIN_MODELS_DIR.exists():
        logger.debug("No margin models dir: %s", NBA_MARGIN_MODELS_DIR)
        return False

    # XGBoost: best by RMSE in filename
    xgb_candidates = list(NBA_MARGIN_MODELS_DIR.glob("*MARGIN*.json"))
    if xgb_candidates:
        rmse_re = re.compile(r"_(\d+(?:\.\d+)?)rmse_")
        def _extract_rmse(path):
            m = rmse_re.search(path.name)
            return float(m.group(1)) if m else 999.0
        best_xgb = min(xgb_candidates, key=_extract_rmse)
        booster = xgb.Booster()
        booster.load_model(str(best_xgb))
        _xgb_margin = booster
        logger.info("Margin XGBoost loaded: %s", best_xgb.name)
    else:
        logger.debug("No margin XGBoost model found in %s", NBA_MARGIN_MODELS_DIR)
        return False

    # CatBoost: best by RMSE in filename
    cat_candidates = list(NBA_MARGIN_MODELS_DIR.glob("*MARGIN*.pkl"))
    # Exclude conformal pkl
    cat_candidates = [p for p in cat_candidates if "conformal" not in p.name.lower()]
    if cat_candidates:
        rmse_re_cat = re.compile(r"_(\d+(?:\.\d+)?)rmse_")
        def _extract_rmse_cat(path):
            m = rmse_re_cat.search(path.name)
            return float(m.group(1)) if m else 999.0
        best_cat = min(cat_candidates, key=_extract_rmse_cat)
        _cat_margin = joblib.load(best_cat)
        logger.info("Margin CatBoost loaded: %s", best_cat.name)
    else:
        logger.debug("No margin CatBoost model found — using XGBoost only")

    # Conformal regression
    conf_path = NBA_MARGIN_MODELS_DIR / "margin_conformal.pkl"
    if conf_path.exists():
        try:
            _conformal_reg = joblib.load(conf_path)
            logger.info("Margin conformal loaded: q̂=%.2f", _conformal_reg.quantile_)
        except Exception as e:
            logger.warning("Error loading margin conformal: %s", e)

    # Sigma buckets + target type from metadata
    meta_path = NBA_MARGIN_MODELS_DIR / "metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            _sigma_buckets = meta.get("sigma_buckets", {})
            logger.info("Margin sigma buckets loaded: %d buckets", len(_sigma_buckets) - 1)
            # Check if model predicts residuals (Camino B) vs raw margin
            if meta.get("target") == "residual":
                _is_residual = True
                logger.info("Margin model: RESIDUAL target (Camino B)")
            else:
                logger.info("Margin model: raw margin target (legacy)")
        except Exception as e:
            logger.warning("Error loading margin metadata: %s", e)

    return True


def is_residual_model():
    """Returns True if the loaded model predicts residuals (Camino B).

    When True, predict_margins() returns residuals (Margin + MARKET_SPREAD).
    Downstream must subtract spread to get raw margin.
    """
    _load_margin_models()
    return _is_residual


def predict_margins(data):
    """Predice margen o residual para cada juego usando ensemble.

    If is_residual_model() → returns residuals (Margin + MARKET_SPREAD).
      Residual > 0 means home covers. Convert: margin = residual - spread.
    Else → returns raw margin (home_score - away_score).

    Args:
        data: numpy array (N, num_features), mismas features que el clasificador.

    Returns:
        numpy array (N,) con prediccion. None si no hay modelo.
    """
    if not _load_margin_models():
        return None

    dmatrix = xgb.DMatrix(data)
    p_xgb = _xgb_margin.predict(dmatrix)

    if _cat_margin is not None:
        p_cat = _cat_margin.predict(data)
        return W_XGB * p_xgb + W_CAT * p_cat
    else:
        return p_xgb


def predict_margin_sigma(spread_lines):
    """Retorna sigma calibrado por bucket de spread.

    Args:
        spread_lines: array (N,) de spreads (negativo = home favorito).

    Returns:
        array (N,) de sigmas calibrados.
    """
    _load_margin_models()
    spread_lines = np.asarray(spread_lines, dtype=float)
    sigmas = np.full(len(spread_lines), FALLBACK_SIGMA)

    if _sigma_buckets is None:
        return sigmas

    abs_spreads = np.abs(spread_lines)
    prev_edge = 0.0
    for label, info in _sigma_buckets.items():
        if label == "global":
            continue
        edge = info["edge"]
        sigma = info["sigma"]
        mask = (abs_spreads > prev_edge) & (abs_spreads <= edge)
        sigmas[mask] = sigma
        prev_edge = edge

    return sigmas


def get_conformal_regressor():
    """Retorna el ConformalRegressor cargado (o None)."""
    _load_margin_models()
    return _conformal_reg
