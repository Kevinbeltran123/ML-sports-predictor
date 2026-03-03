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
_xgb_q10 = None  # Quantile 10% model
_xgb_q90 = None  # Quantile 90% model
_conformal_reg = None
_sigma_buckets = None
_feature_columns = None  # Expected feature order from training
_is_residual = False
_load_attempted = False

W_XGB = 0.60  # Default, overridden by metadata.json if present
W_CAT = 0.40

# Fallback global sigma (NBA historical)
FALLBACK_SIGMA = 14.3


def _load_margin_models():
    """Carga modelos de margen XGBoost + CatBoost + Quantile Q10/Q90 (lazy, una sola vez)."""
    global _xgb_margin, _cat_margin, _xgb_q10, _xgb_q90, _conformal_reg, _sigma_buckets, _feature_columns, _is_residual, _load_attempted, W_XGB, W_CAT
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

    # Quantile models (Q10 / Q90) — optional, trained by train_margin_models.py
    q10_candidates = list(NBA_MARGIN_MODELS_DIR.glob("*MARGIN_Q10*.json"))
    if q10_candidates:
        best_q10 = max(q10_candidates, key=lambda p: p.stat().st_mtime)
        b10 = xgb.Booster()
        b10.load_model(str(best_q10))
        _xgb_q10 = b10
        logger.info("Margin Q10 loaded: %s", best_q10.name)

    q90_candidates = list(NBA_MARGIN_MODELS_DIR.glob("*MARGIN_Q90*.json"))
    if q90_candidates:
        best_q90 = max(q90_candidates, key=lambda p: p.stat().st_mtime)
        b90 = xgb.Booster()
        b90.load_model(str(best_q90))
        _xgb_q90 = b90
        logger.info("Margin Q90 loaded: %s", best_q90.name)

    # Conformal regression
    conf_path = NBA_MARGIN_MODELS_DIR / "margin_conformal.pkl"
    if conf_path.exists():
        try:
            _conformal_reg = joblib.load(conf_path)
            logger.info("Margin conformal loaded: q̂=%.2f", _conformal_reg.quantile_)
        except Exception as e:
            logger.warning("Error loading margin conformal: %s", e)

    # Sigma buckets + target type + weights from metadata
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
            # Load ensemble weights from metadata (Optuna-tuned)
            weights = meta.get("weights", {})
            if "xgb" in weights and "cat" in weights:
                W_XGB = weights["xgb"]
                W_CAT = weights["cat"]
                logger.info("Margin weights from metadata: XGB=%.2f, CAT=%.2f", W_XGB, W_CAT)
            # Load feature column order (critical for positional alignment)
            fc = meta.get("feature_columns")
            if fc:
                _feature_columns = fc
                logger.info("Margin feature columns loaded: %d", len(_feature_columns))
        except Exception as e:
            logger.warning("Error loading margin metadata: %s", e)

    return True


def get_feature_columns():
    """Returns expected feature column order from training, or None if not available."""
    _load_margin_models()
    return _feature_columns


def is_residual_model():
    """Returns True if the loaded model predicts residuals (Camino B).

    When True, predict_margins() returns residuals (Margin + MARKET_SPREAD).
    Downstream must subtract spread to get raw margin.
    """
    _load_margin_models()
    return _is_residual


def predict_margins(data, feature_names=None):
    """Predice margen o residual para cada juego usando ensemble.

    If is_residual_model() → returns residuals (Margin + MARKET_SPREAD).
      Residual > 0 means home covers. Convert: margin = residual - spread.
    Else → returns raw margin (home_score - away_score).

    Args:
        data: numpy array (N, num_features), mismas features que el clasificador.
        feature_names: list of feature names matching data columns (optional).
            Used to reorder columns to match training feature order for CatBoost.

    Returns:
        numpy array (N,) con prediccion. None si no hay modelo.
    """
    if not _load_margin_models():
        return None

    dmatrix = xgb.DMatrix(data)
    p_xgb = _xgb_margin.predict(dmatrix)

    if _cat_margin is not None:
        try:
            cat_data = data
            # Reorder columns to match training feature order if possible
            if _feature_columns and feature_names:
                import pandas as pd
                df_tmp = pd.DataFrame(data, columns=feature_names)
                # Only use columns that exist in both
                common = [c for c in _feature_columns if c in df_tmp.columns]
                if len(common) >= len(_feature_columns) * 0.9:
                    cat_data = df_tmp[_feature_columns].fillna(0).to_numpy(dtype=float)
                else:
                    logger.warning("Feature overlap too low (%d/%d) — using XGB only",
                                   len(common), len(_feature_columns))
                    return p_xgb
            p_cat = _cat_margin.predict(cat_data)
            return W_XGB * p_xgb + W_CAT * p_cat
        except Exception as e:
            logger.warning("CatBoost margin predict failed: %s — using XGB only", e)
            return p_xgb
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


def predict_margin_interval(data):
    """Retorna intervalo de prediccion [Q10, Q90] para cada juego.

    Usa modelos XGBoost quantile (entrenados con objective=reg:quantileerror).
    El interval_width = Q90 - Q10 sirve como medida de incertidumbre:
      - Ancho < 15pts → alta certeza → mayor confianza AH
      - Ancho > 25pts → alta incertidumbre → reducir Kelly AH

    Args:
        data: numpy array (N, num_features), mismas features que predict_margins().

    Returns:
        tuple: (q10, q90, interval_width) — arrays (N,) o None si no hay modelos.
    """
    _load_margin_models()
    if _xgb_q10 is None or _xgb_q90 is None:
        return None

    dmatrix = xgb.DMatrix(data)
    q10 = _xgb_q10.predict(dmatrix)
    q90 = _xgb_q90.predict(dmatrix)
    interval_width = q90 - q10
    return q10, q90, interval_width


def get_conformal_regressor():
    """Retorna el ConformalRegressor cargado (o None)."""
    _load_margin_models()
    return _conformal_reg
