"""Totals Runner: carga modelos XGBoost + CatBoost de totals y predice score total.

El modelo predice Score = home_score + away_score (total de puntos).
P(over) se calcula via normal CDF: P(over) = Phi((predicted - line) / sigma).

Uso:
    from src.sports.nba.predict.totals_runner import predict_totals, p_over
    totals = predict_totals(data)  # array (N,) o None si no hay modelo
    probs = p_over(totals, ou_lines, sigma=12.0)  # array (N,) de P(over)
"""

import json
import re

import joblib
import numpy as np
import xgboost as xgb
from scipy.stats import norm

from src.config import NBA_UO_MODELS_DIR, get_logger

logger = get_logger(__name__)

_xgb_totals = None
_cat_totals = None
_conformal_reg = None
_residual_sigma = None
_load_attempted = False

W_XGB = 0.60
W_CAT = 0.40

# Fallback sigma for totals (NBA historical ~12 pts)
FALLBACK_SIGMA = 12.0


def _load_totals_models():
    """Carga modelos de totals XGBoost + CatBoost (lazy, una sola vez)."""
    global _xgb_totals, _cat_totals, _conformal_reg, _residual_sigma, _load_attempted
    if _load_attempted:
        return _xgb_totals is not None
    _load_attempted = True

    if not NBA_UO_MODELS_DIR.exists():
        logger.debug("No totals models dir: %s", NBA_UO_MODELS_DIR)
        return False

    # XGBoost: best by RMSE in filename
    xgb_candidates = list(NBA_UO_MODELS_DIR.glob("*TOTAL*.json"))
    if xgb_candidates:
        rmse_re = re.compile(r"_(\d+(?:\.\d+)?)rmse_")
        def _extract_rmse(path):
            m = rmse_re.search(path.name)
            return float(m.group(1)) if m else 999.0
        best_xgb = min(xgb_candidates, key=_extract_rmse)
        booster = xgb.Booster()
        booster.load_model(str(best_xgb))
        _xgb_totals = booster
        logger.info("Totals XGBoost loaded: %s", best_xgb.name)
    else:
        logger.debug("No totals XGBoost model found in %s", NBA_UO_MODELS_DIR)
        return False

    # CatBoost: best by RMSE in filename
    cat_candidates = list(NBA_UO_MODELS_DIR.glob("*TOTAL*.pkl"))
    cat_candidates = [p for p in cat_candidates if "conformal" not in p.name.lower()]
    if cat_candidates:
        rmse_re_cat = re.compile(r"_(\d+(?:\.\d+)?)rmse_")
        def _extract_rmse_cat(path):
            m = rmse_re_cat.search(path.name)
            return float(m.group(1)) if m else 999.0
        best_cat = min(cat_candidates, key=_extract_rmse_cat)
        _cat_totals = joblib.load(best_cat)
        logger.info("Totals CatBoost loaded: %s", best_cat.name)
    else:
        logger.debug("No totals CatBoost model found — using XGBoost only")

    # Conformal regression
    conf_path = NBA_UO_MODELS_DIR / "totals_conformal.pkl"
    if conf_path.exists():
        try:
            _conformal_reg = joblib.load(conf_path)
            logger.info("Totals conformal loaded: q̂=%.2f", _conformal_reg.quantile_)
        except Exception as e:
            logger.warning("Error loading totals conformal: %s", e)

    # Residual sigma from metadata
    meta_path = NBA_UO_MODELS_DIR / "metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            _residual_sigma = meta.get("residual_sigma", FALLBACK_SIGMA)
            logger.info("Totals residual σ=%.1f", _residual_sigma)
        except Exception as e:
            logger.warning("Error loading totals metadata: %s", e)

    return True


def predict_totals(data):
    """Predice total de puntos para cada juego usando ensemble.

    Args:
        data: numpy array (N, num_features).

    Returns:
        numpy array (N,) con total predicho.
        None si el modelo no esta disponible.
    """
    if not _load_totals_models():
        return None

    dmatrix = xgb.DMatrix(data)
    p_xgb = _xgb_totals.predict(dmatrix)

    if _cat_totals is not None:
        p_cat = _cat_totals.predict(data)
        return W_XGB * p_xgb + W_CAT * p_cat
    else:
        return p_xgb


def p_over(predicted_total, ou_line, sigma=None):
    """P(over) via normal CDF: P(total > line) = Phi((predicted - line) / sigma).

    Args:
        predicted_total: float or array, total predicho.
        ou_line: float or array, linea O/U del mercado.
        sigma: residual sigma. Si None, usa calibrado o fallback.

    Returns:
        float or array, P(over) clipeada a [0.01, 0.99].
    """
    if sigma is None:
        _load_totals_models()
        sigma = _residual_sigma if _residual_sigma is not None else FALLBACK_SIGMA

    predicted_total = np.asarray(predicted_total, dtype=float)
    ou_line = np.asarray(ou_line, dtype=float)
    z = (predicted_total - ou_line) / sigma
    return np.clip(norm.cdf(z), 0.01, 0.99)


def get_totals_conformal():
    """Retorna el ConformalRegressor de totals (o None)."""
    _load_totals_models()
    return _conformal_reg


def get_residual_sigma():
    """Retorna sigma residual calibrado del modelo de totals."""
    _load_totals_models()
    return _residual_sigma if _residual_sigma is not None else FALLBACK_SIGMA
