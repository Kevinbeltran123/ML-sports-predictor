"""Runner para predicciones in-game con cascading fallback.

TRES FAMILIAS DE PREDICCION:
=============================
1. Moneyline (ML): predict_ingame() — ¿quien gana?
2. Spread (ATS):   predict_ingame_spread() — ¿cubre el handicap?
3. Total (O/U):    predict_ingame_total() — ¿pasan la linea de puntos?

CASCADING FALLBACK (aplica a las 3 familias):
===============================================
Para cada periodo (Q1, Q2, Q3), el runner intenta modelos en orden:
  1. XGBoost Q{period} (si existe) — Nivel 2, accuracy mas alta
  2. Logistic Q{period} (si existe) — Nivel 1, mas simple pero calibrado
  3. Fallback simple (siempre disponible):
     - ML:     bayesian_q1_adjustment() con B=0.45
     - Spread: proyeccion lineal de margen vs spread
     - Total:  proyeccion lineal de scoring pace vs linea

Esto garantiza que SIEMPRE hay una prediccion disponible, incluso si
ningun modelo fue entrenado aun.

Uso:
    from src.sports.nba.predict.ingame_runner import predict_ingame, predict_ingame_spread, predict_ingame_total

    # Moneyline
    ml = predict_ingame(p_pregame=0.68, box_home={...}, box_away={...}, period=1)

    # Spread
    spread = predict_ingame_spread(
        p_pregame=0.68, box_home={...}, box_away={...},
        pregame_spread=-5.0, period=1
    )

    # Total
    total = predict_ingame_total(
        box_home={...}, box_away={...},
        pregame_total=215.5, period=1
    )
"""

import logging
import math
import re
from pathlib import Path

import joblib
import numpy as np

from src.config import INGAME_MODELS_DIR
# XGBCalibrator debe importarse aqui para que joblib pueda deserializar
# pickles del calibrador independientemente del contexto de llamada
from src.core.calibration.xgb_calibrator import XGBCalibrator  # noqa: F401

logger = logging.getLogger(__name__)

# =====================================================================
# Cache de modelos (cargados una sola vez por sesion)
# =====================================================================

# --- Moneyline ---
_xgb_cache: dict[int, object] = {}       # period -> XGBoost Booster
_xgb_cal_cache: dict[int, object] = {}   # period -> _XGBCalibrator
_xgb_conf_cache: dict[int, object] = {}  # period -> ConformalClassifier
_log_cache: dict[int, object] = {}       # period -> LogisticRegression
_log_conf_cache: dict[int, object] = {}  # period -> ConformalClassifier

# --- Spread ---
_spread_xgb_cache: dict[int, object] = {}
_spread_cal_cache: dict[int, object] = {}
_spread_conf_cache: dict[int, object] = {}

# --- Total ---
_total_xgb_cache: dict[int, object] = {}
_total_cal_cache: dict[int, object] = {}
_total_conf_cache: dict[int, object] = {}


# =====================================================================
# Loaders de modelos — Moneyline (existente)
# =====================================================================

def _load_xgboost(period: int):
    """Carga modelo XGBoost in-game ML para un periodo dado."""
    if period in _xgb_cache:
        return _xgb_cache[period], _xgb_cal_cache.get(period), _xgb_conf_cache.get(period)

    try:
        import xgboost as xgb

        if not INGAME_MODELS_DIR.exists():
            return None, None, None

        # Buscar el modelo mas reciente
        pattern = f"XGB_Q{period}_*pct.json"
        candidates = sorted(INGAME_MODELS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
        if not candidates:
            return None, None, None

        model_path = candidates[-1]
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        _xgb_cache[period] = booster
        logger.info("XGBoost in-game cargado: %s", model_path.name)

        # Calibrador
        cal_path = INGAME_MODELS_DIR / f"XGB_Q{period}_calibration.pkl"
        calibrator = None
        if cal_path.exists():
            calibrator = joblib.load(cal_path)
            _xgb_cal_cache[period] = calibrator

        # Conformal
        conf_path = INGAME_MODELS_DIR / f"XGB_Q{period}_conformal.pkl"
        conformal = None
        if conf_path.exists():
            conformal = joblib.load(conf_path)
            _xgb_conf_cache[period] = conformal

        return booster, calibrator, conformal

    except Exception as e:
        logger.warning("Error cargando XGBoost in-game Q%d: %s", period, e)
        return None, None, None


def _load_logistic(period: int):
    """Carga modelo logistico in-game ML para un periodo dado."""
    if period in _log_cache:
        return _log_cache[period], _log_conf_cache.get(period)

    try:
        if not INGAME_MODELS_DIR.exists():
            return None, None

        pattern = f"logistic_Q{period}_*pct.pkl"
        candidates = sorted(INGAME_MODELS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
        if not candidates:
            return None, None

        model_path = candidates[-1]
        model = joblib.load(model_path)
        _log_cache[period] = model
        logger.info("Logistic in-game cargado: %s", model_path.name)

        conf_path = INGAME_MODELS_DIR / f"logistic_Q{period}_conformal.pkl"
        conformal = None
        if conf_path.exists():
            conformal = joblib.load(conf_path)
            _log_conf_cache[period] = conformal

        return model, conformal

    except Exception as e:
        logger.warning("Error cargando logistic in-game Q%d: %s", period, e)
        return None, None


# =====================================================================
# Loaders de modelos — Spread
# =====================================================================

def _load_spread_xgboost(period: int):
    """Carga modelo XGBoost de spread in-game para un periodo dado."""
    if period in _spread_xgb_cache:
        return (
            _spread_xgb_cache[period],
            _spread_cal_cache.get(period),
            _spread_conf_cache.get(period),
        )

    try:
        import xgboost as xgb

        if not INGAME_MODELS_DIR.exists():
            return None, None, None

        pattern = f"SPREAD_XGB_Q{period}_*pct.json"
        candidates = sorted(INGAME_MODELS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
        if not candidates:
            return None, None, None

        model_path = candidates[-1]
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        _spread_xgb_cache[period] = booster
        logger.info("Spread XGBoost cargado: %s", model_path.name)

        # Calibrador Platt
        cal_path = INGAME_MODELS_DIR / f"SPREAD_XGB_Q{period}_calibration.pkl"
        calibrator = None
        if cal_path.exists():
            calibrator = joblib.load(cal_path)
            _spread_cal_cache[period] = calibrator

        # Conformal
        conf_path = INGAME_MODELS_DIR / f"SPREAD_XGB_Q{period}_conformal.pkl"
        conformal = None
        if conf_path.exists():
            conformal = joblib.load(conf_path)
            _spread_conf_cache[period] = conformal

        return booster, calibrator, conformal

    except Exception as e:
        logger.warning("Error cargando Spread XGBoost Q%d: %s", period, e)
        return None, None, None


# =====================================================================
# Loaders de modelos — Total
# =====================================================================

def _load_total_xgboost(period: int):
    """Carga modelo XGBoost de total O/U in-game para un periodo dado."""
    if period in _total_xgb_cache:
        return (
            _total_xgb_cache[period],
            _total_cal_cache.get(period),
            _total_conf_cache.get(period),
        )

    try:
        import xgboost as xgb

        if not INGAME_MODELS_DIR.exists():
            return None, None, None

        pattern = f"TOTAL_XGB_Q{period}_*pct.json"
        candidates = sorted(INGAME_MODELS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
        if not candidates:
            return None, None, None

        model_path = candidates[-1]
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        _total_xgb_cache[period] = booster
        logger.info("Total XGBoost cargado: %s", model_path.name)

        # Calibrador Platt
        cal_path = INGAME_MODELS_DIR / f"TOTAL_XGB_Q{period}_calibration.pkl"
        calibrator = None
        if cal_path.exists():
            calibrator = joblib.load(cal_path)
            _total_cal_cache[period] = calibrator

        # Conformal
        conf_path = INGAME_MODELS_DIR / f"TOTAL_XGB_Q{period}_conformal.pkl"
        conformal = None
        if conf_path.exists():
            conformal = joblib.load(conf_path)
            _total_conf_cache[period] = conformal

        return booster, calibrator, conformal

    except Exception as e:
        logger.warning("Error cargando Total XGBoost Q%d: %s", period, e)
        return None, None, None


# =====================================================================
# FAMILIA 1: Moneyline (ML) — predict_ingame()
# =====================================================================

def predict_ingame(
    p_pregame: float,
    box_home: dict,
    box_away: dict,
    period: int = 1,
    elo_diff: float = 0.0,
    market_prob: float | None = None,
    pbp_features: dict | None = None,
) -> dict:
    """Pipeline completo de prediccion in-game con cascading fallback.

    Intenta modelos en orden: XGBoost -> Logistic -> Simple Bayesian.
    Retorna un dict estandarizado con la prediccion y metadata.

    Args:
        p_pregame:    probabilidad pre-partido del local (0-1)
        box_home:     dict de stats del local (formato live API, keys minusculas)
        box_away:     dict de stats del visitante
        period:       periodo completado (1, 2, o 3)
        elo_diff:     diferencial Elo pre-partido (0 si no disponible)
        market_prob:  probabilidad implicita del mercado (None = usar p_pregame)
        pbp_features: dict con 17 features PBP (PBP_LEAD_CHANGES, etc.) ya
                      computadas desde el tracker en vivo. Si se provee, se usa
                      directamente con los modelos XGBoost/Logistic PBP-based.
                      Si es None, el XGBoost/Logistic se saltan y se usa Bayesian.

    Returns:
        dict con claves:
          p_home:              float — probabilidad ajustada del local
          confidence:          str — "HIGH", "LOW", o "N/A"
          conformal_set_size:  int — 1=confiado, 2=incierto, 0=sin conformal
          model_used:          str — "xgboost_Q1", "logistic_Q1", "simple"
          delta:               float — cambio vs p_pregame
          features:            dict — features usadas (para debug)
    """
    # Feature lists for PBP models — v1 (18 features, original) and v2 (21, with score context)
    PBP_V1_FEATURES = [
        "LOGIT_PREGAME",
        "PBP_LEAD_CHANGES", "PBP_LARGEST_LEAD_HOME", "PBP_LARGEST_LEAD_AWAY",
        "PBP_HOME_RUNS_MAX", "PBP_AWAY_RUNS_MAX",
        "PBP_TIMEOUTS_HOME", "PBP_TIMEOUTS_AWAY",
        "PBP_FOULS_HOME", "PBP_FOULS_AWAY",
        "PBP_TURNOVERS_HOME", "PBP_TURNOVERS_AWAY",
        "PBP_MOMENTUM", "PBP_LAST_5MIN_DIFF",
        "PBP_FG3_MADE_HOME", "PBP_FG3_MADE_AWAY",
        "PBP_OREB_HOME", "PBP_OREB_AWAY",
    ]
    # v2: score-context features that fix home bias in blowouts
    PBP_ALL_FEATURES = PBP_V1_FEATURES + [
        "PBP_SCORE_DIFF_NORM",
        "PBP_BLOWOUT_FLAG",
        "PBP_SCORE_DIFF_X_MOMENTUM",
    ]
    PBP_LOGISTIC_V1 = PBP_V1_FEATURES[:9]  # v1: 9 features
    # v2: 9 basic + 3 score-context = 12
    PBP_LOGISTIC_FEATURES = PBP_LOGISTIC_V1 + [
        "PBP_SCORE_DIFF_NORM",
        "PBP_BLOWOUT_FLAG",
        "PBP_SCORE_DIFF_X_MOMENTUM",
    ]

    if market_prob is None:
        market_prob = p_pregame

    # LOGIT_PREGAME: anclaje pre-partido para ambos modelos PBP
    p_clip = max(0.05, min(0.95, p_pregame))
    logit_pregame = float(np.log(p_clip / (1.0 - p_clip)))

    # --- Nivel 2: XGBoost (PBP-based, requiere pbp_features) ---
    booster, calibrator, conformal_xgb = _load_xgboost(period)
    if booster is not None and pbp_features is not None:
        try:
            import xgboost as xgb

            # Detect model version by feature count (v1=18, v2=21)
            try:
                model_num_features = int(booster.num_features())
            except Exception:
                model_num_features = len(PBP_V1_FEATURES)  # assume v1

            if model_num_features == len(PBP_ALL_FEATURES):
                feature_list = PBP_ALL_FEATURES
            else:
                feature_list = PBP_V1_FEATURES

            # Construir vector de features en orden correcto para el modelo
            features_with_anchor = dict(pbp_features)
            features_with_anchor["LOGIT_PREGAME"] = logit_pregame
            X = np.array(
                [[features_with_anchor.get(name, 0.0) for name in feature_list]],
                dtype=np.float32,
            )
            dmat = xgb.DMatrix(X, feature_names=feature_list)

            # Predecir con calibracion Platt
            if calibrator is not None:
                proba = calibrator.predict_proba(dmat)  # (1, 2)
                p_home = float(proba[0, 1])
            else:
                raw = booster.predict(dmat)  # (1, 2)
                p_home = float(raw[0, 1])
                proba = raw

            # Conformal — batch call 2D (1, 2)
            set_size = 0
            if conformal_xgb is not None:
                set_sizes, _ = conformal_xgb.predict_confidence(proba)
                set_size = int(set_sizes[0])

            confidence = "HIGH" if set_size == 1 else "LOW" if set_size == 2 else "N/A"

            return {
                "p_home": p_home,
                "confidence": confidence,
                "conformal_set_size": set_size,
                "model_used": f"xgboost_Q{period}",
                "delta": p_home - p_pregame,
                "features": features_with_anchor,
            }

        except Exception as e:
            logger.warning("XGBoost Q%d prediction failed: %s. Falling back.", period, e)

    # --- Nivel 1: Logistic (PBP-based, requiere pbp_features) ---
    log_model, conformal_log = _load_logistic(period)
    if log_model is not None and pbp_features is not None:
        try:
            # Detect logistic model version by feature count
            try:
                log_num_features = log_model.n_features_in_
            except AttributeError:
                log_num_features = len(PBP_LOGISTIC_V1)  # assume v1

            if log_num_features == len(PBP_LOGISTIC_FEATURES):
                log_feature_list = PBP_LOGISTIC_FEATURES
            else:
                log_feature_list = PBP_LOGISTIC_V1

            features_with_anchor = dict(pbp_features)
            features_with_anchor["LOGIT_PREGAME"] = logit_pregame
            X = np.array(
                [[features_with_anchor.get(name, 0.0) for name in log_feature_list]]
            )

            proba = log_model.predict_proba(X)  # (1, 2)
            p_home = float(proba[0, 1])

            # Conformal — batch call 2D (1, 2)
            set_size = 0
            if conformal_log is not None:
                set_sizes, _ = conformal_log.predict_confidence(proba)
                set_size = int(set_sizes[0])

            confidence = "HIGH" if set_size == 1 else "LOW" if set_size == 2 else "N/A"

            return {
                "p_home": p_home,
                "confidence": confidence,
                "conformal_set_size": set_size,
                "model_used": f"logistic_Q{period}",
                "delta": p_home - p_pregame,
                "features": features_with_anchor,
            }

        except Exception as e:
            logger.warning("Logistic Q%d prediction failed: %s. Falling back.", period, e)

    # --- Nivel 0: Simple Bayesian (siempre disponible) ---
    from src.sports.nba.predict.live_betting import bayesian_q1_adjustment
    from src.sports.nba.features.ingame_features import box_score_to_stats_dict

    home_stats = box_score_to_stats_dict(box_home)
    away_stats = box_score_to_stats_dict(box_away)

    home_pts = float(home_stats.get("PTS", 0))
    away_pts = float(away_stats.get("PTS", 0))
    score_diff = int(home_pts - away_pts)

    home_poss = float(home_stats.get("POSS", 25))
    away_poss = float(away_stats.get("POSS", 25))
    total_poss = (home_poss + away_poss) / 2.0

    p_adj, expl = bayesian_q1_adjustment(p_pregame, score_diff, total_poss)

    return {
        "p_home": p_adj,
        "confidence": "N/A",
        "conformal_set_size": 0,
        "model_used": "simple",
        "delta": p_adj - p_pregame,
        "features": {"score_diff": score_diff, "total_poss": total_poss},
    }


# =====================================================================
# FAMILIA 2: Spread (ATS) — predict_ingame_spread()
# =====================================================================

def predict_ingame_spread(
    p_pregame: float,
    box_home: dict,
    box_away: dict,
    pregame_spread: float,
    period: int = 1,
    elo_diff: float = 0.0,
    market_prob: float | None = None,
) -> dict:
    """Prediccion in-game de spread cover con cascading fallback.

    Pregunta: el equipo local cubrira el handicap?
    Ejemplo: spread=-6.0 -> Lakers ganara por mas de 6 puntos?

    Cascading fallback:
      1. XGBoost spread (22 features, Optuna-tuned)
      2. Fallback simple: proyeccion lineal del margen vs spread
         (no hay modelo logistico para spread — el XGBoost es el unico ML)

    Args:
        p_pregame:      probabilidad pre-partido del local (0-1)
        box_home:       dict de stats del local
        box_away:       dict de stats del visitante
        pregame_spread: spread del local (negativo=favorito). Ej: -5.0
        period:         periodo completado (1, 2, o 3)
        elo_diff:       diferencial Elo pre-partido
        market_prob:    probabilidad implicita del mercado

    Returns:
        dict con claves:
          p_cover:             float — P(home cubre el spread)
          confidence:          str — "HIGH", "LOW", o "N/A"
          conformal_set_size:  int — 1=confiado, 2=incierto, 0=sin conformal
          model_used:          str — "spread_xgb_Q1" o "spread_simple"
          spread_pace:         float — score actual vs expectativa del mercado
          projected_margin:    float — margen proyectado al final
          features:            dict — features usadas (debug)
    """
    from src.sports.nba.features.ingame_features import (
        compute_spread_features,
        box_score_to_stats_dict,
        INGAME_SPREAD_FEATURES,
        EXPECTED_TOTAL_POSS,
    )

    if market_prob is None:
        market_prob = p_pregame

    home_stats = box_score_to_stats_dict(box_home)
    away_stats = box_score_to_stats_dict(box_away)

    # Stats basicas para fallback y metadata
    home_pts = float(home_stats.get("PTS", 0))
    away_pts = float(away_stats.get("PTS", 0))
    score_diff = home_pts - away_pts
    home_poss = max(1.0, float(home_stats.get("POSS", 25)))
    away_poss = max(1.0, float(away_stats.get("POSS", 25)))
    combined_poss = home_poss + away_poss
    period_fraction = min(period / 4.0, 1.0)

    # Spread pace: desviacion del score vs expectativa del mercado
    expected_diff = (-pregame_spread) * period_fraction
    spread_pace = score_diff - expected_diff

    # Margen proyectado: si el ritmo continua, cual sera el margen final
    if combined_poss > 0:
        margin_per_poss = score_diff / (combined_poss / 2.0)
        projected_margin = margin_per_poss * (EXPECTED_TOTAL_POSS / 2.0)
    else:
        projected_margin = 0.0

    # --- Nivel 2: XGBoost Spread ---
    booster, calibrator, conformal = _load_spread_xgboost(period)
    if booster is not None:
        try:
            import xgboost as xgb

            features = compute_spread_features(
                home_stats, away_stats,
                p_pregame=p_pregame,
                market_ml_prob=market_prob,
                elo_diff=elo_diff,
                pregame_spread=pregame_spread,
                period=period,
            )
            X = np.array([[features[name] for name in INGAME_SPREAD_FEATURES]],
                         dtype=np.float32)

            # Predecir (con calibracion si disponible)
            if calibrator is not None:
                proba = calibrator.predict_proba(X)
                p_cover = float(proba[0, 1])
            else:
                dmat = xgb.DMatrix(X)
                raw = booster.predict(dmat)
                p_cover = float(raw[0, 1])

            # Conformal
            set_size = 0
            if conformal is not None:
                if calibrator is not None:
                    prob_2d = calibrator.predict_proba(X)[0]
                else:
                    prob_2d = raw[0]
                set_size, _ = conformal.predict_confidence(prob_2d)

            confidence = "HIGH" if set_size == 1 else "LOW" if set_size == 2 else "N/A"

            return {
                "p_cover": p_cover,
                "confidence": confidence,
                "conformal_set_size": set_size,
                "model_used": f"spread_xgb_Q{period}",
                "spread_pace": spread_pace,
                "projected_margin": projected_margin,
                "features": dict(features),
            }

        except Exception as e:
            logger.warning("Spread XGBoost Q%d failed: %s. Falling back.", period, e)

    # --- Nivel 0: Fallback simple (proyeccion lineal + logistica) ---
    remaining_poss = max(1.0, EXPECTED_TOTAL_POSS - combined_poss)
    margin_vs_spread = score_diff - (-pregame_spread)  # positivo = ya cubriendo
    z = margin_vs_spread / math.sqrt(remaining_poss)

    k = 2.0
    p_cover = 1.0 / (1.0 + math.exp(-k * z))

    # Incorporar prior del pregame (spread=0 -> P(cover)=0.5)
    prior = 0.5
    p_cover = 0.80 * p_cover + 0.20 * prior

    return {
        "p_cover": max(0.01, min(0.99, p_cover)),
        "confidence": "N/A",
        "conformal_set_size": 0,
        "model_used": "spread_simple",
        "spread_pace": spread_pace,
        "projected_margin": projected_margin,
        "features": {
            "score_diff": score_diff,
            "margin_vs_spread": margin_vs_spread,
            "remaining_poss": remaining_poss,
            "z_score": z,
        },
    }


# =====================================================================
# FAMILIA 3: Total (O/U) — predict_ingame_total()
# =====================================================================

def predict_ingame_total(
    box_home: dict,
    box_away: dict,
    pregame_total: float,
    period: int = 1,
    p_pregame: float = 0.5,
    market_prob: float | None = None,
) -> dict:
    """Prediccion in-game de total O/U con cascading fallback.

    Pregunta: el total combinado de puntos superara la linea pregame?
    Ejemplo: total=215.5, Q1 termina 32-30 (62 pts) -> on pace for 248 -> OVER

    Cascading fallback:
      1. XGBoost total (15 features, ritmo/scoring pace)
      2. Fallback simple: proyeccion lineal de scoring pace vs linea

    Args:
        box_home:       dict de stats del local
        box_away:       dict de stats del visitante
        pregame_total:  linea O/U pregame. Ej: 215.5
        period:         periodo completado (1, 2, o 3)
        p_pregame:      probabilidad pre-partido del local (proxy de calidad)
        market_prob:    probabilidad implicita del mercado

    Returns:
        dict con claves:
          p_over:              float — P(total > linea)
          confidence:          str — "HIGH", "LOW", o "N/A"
          conformal_set_size:  int — 1=confiado, 2=incierto, 0=sin conformal
          model_used:          str — "total_xgb_Q1" o "total_simple"
          projected_total:     float — total proyectado al final
          pace_vs_line:        float — projected - pregame_total
          scoring_pace:        float — pts combinados / poss combinadas
          features:            dict — features usadas (debug)
    """
    from src.sports.nba.features.ingame_features import (
        compute_total_features,
        box_score_to_stats_dict,
        INGAME_TOTAL_FEATURES,
        EXPECTED_TOTAL_POSS,
        HISTORICAL_FG_PCT,
    )

    if market_prob is None:
        market_prob = p_pregame

    home_stats = box_score_to_stats_dict(box_home)
    away_stats = box_score_to_stats_dict(box_away)

    # Stats basicas para fallback y metadata
    home_pts = float(home_stats.get("PTS", 0))
    away_pts = float(away_stats.get("PTS", 0))
    combined_pts = home_pts + away_pts
    home_poss = max(1.0, float(home_stats.get("POSS", 25)))
    away_poss = max(1.0, float(away_stats.get("POSS", 25)))
    combined_poss = home_poss + away_poss

    # Scoring pace y proyeccion
    scoring_pace = combined_pts / combined_poss if combined_poss > 0 else 1.0
    projected_total = scoring_pace * EXPECTED_TOTAL_POSS
    pace_vs_line = projected_total - pregame_total

    # --- Nivel 2: XGBoost Total ---
    booster, calibrator, conformal = _load_total_xgboost(period)
    if booster is not None:
        try:
            import xgboost as xgb

            features = compute_total_features(
                home_stats, away_stats,
                pregame_total=pregame_total,
                period=period,
                p_pregame=p_pregame,
                market_ml_prob=market_prob,
            )
            X = np.array([[features[name] for name in INGAME_TOTAL_FEATURES]],
                         dtype=np.float32)

            # Predecir (con calibracion si disponible)
            if calibrator is not None:
                proba = calibrator.predict_proba(X)
                p_over = float(proba[0, 1])
            else:
                dmat = xgb.DMatrix(X)
                raw = booster.predict(dmat)
                p_over = float(raw[0, 1])

            # Conformal
            set_size = 0
            if conformal is not None:
                if calibrator is not None:
                    prob_2d = calibrator.predict_proba(X)[0]
                else:
                    prob_2d = raw[0]
                set_size, _ = conformal.predict_confidence(prob_2d)

            confidence = "HIGH" if set_size == 1 else "LOW" if set_size == 2 else "N/A"

            return {
                "p_over": p_over,
                "confidence": confidence,
                "conformal_set_size": set_size,
                "model_used": f"total_xgb_Q{period}",
                "projected_total": projected_total,
                "pace_vs_line": pace_vs_line,
                "scoring_pace": scoring_pace,
                "features": dict(features),
            }

        except Exception as e:
            logger.warning("Total XGBoost Q%d failed: %s. Falling back.", period, e)

    # --- Nivel 0: Fallback simple (proyeccion lineal + regresion a la media) ---
    pace_prior = pregame_total / EXPECTED_TOTAL_POSS

    n_prior = 50.0
    w = combined_poss / (combined_poss + n_prior)
    pace_adjusted = w * scoring_pace + (1.0 - w) * pace_prior

    projected_adjusted = pace_adjusted * EXPECTED_TOTAL_POSS
    diff_vs_line = projected_adjusted - pregame_total

    remaining_poss = max(1.0, EXPECTED_TOTAL_POSS - combined_poss)
    z = diff_vs_line / math.sqrt(remaining_poss)

    k = 1.5
    p_over = 1.0 / (1.0 + math.exp(-k * z))

    return {
        "p_over": max(0.01, min(0.99, p_over)),
        "confidence": "N/A",
        "conformal_set_size": 0,
        "model_used": "total_simple",
        "projected_total": projected_adjusted,
        "pace_vs_line": diff_vs_line,
        "scoring_pace": scoring_pace,
        "features": {
            "combined_pts": combined_pts,
            "combined_poss": combined_poss,
            "pace_observed": scoring_pace,
            "pace_prior": pace_prior,
            "pace_adjusted": pace_adjusted,
            "bayesian_weight": w,
            "z_score": z,
        },
    }


# =====================================================================
# Utilidad: modelos disponibles
# =====================================================================

def get_available_models() -> dict[str, dict[int, str]]:
    """Retorna los modelos in-game disponibles por familia y periodo.

    Returns:
        dict con 3 llaves (ml, spread, total), cada una con dict {period: model_name}.
    """
    available = {"ml": {}, "spread": {}, "total": {}}

    if not INGAME_MODELS_DIR.exists():
        for period in [1, 2, 3]:
            available["ml"][period] = "simple"
            available["spread"][period] = "spread_simple"
            available["total"][period] = "total_simple"
        return available

    for period in [1, 2, 3]:
        # --- Moneyline ---
        xgb_ml = list(INGAME_MODELS_DIR.glob(f"XGB_Q{period}_*pct.json"))
        if xgb_ml:
            available["ml"][period] = f"xgboost_Q{period}"
        elif list(INGAME_MODELS_DIR.glob(f"logistic_Q{period}_*pct.pkl")):
            available["ml"][period] = f"logistic_Q{period}"
        else:
            available["ml"][period] = "simple"

        # --- Spread ---
        spread_xgb = list(INGAME_MODELS_DIR.glob(f"SPREAD_XGB_Q{period}_*pct.json"))
        if spread_xgb:
            available["spread"][period] = f"spread_xgb_Q{period}"
        else:
            available["spread"][period] = "spread_simple"

        # --- Total ---
        total_xgb = list(INGAME_MODELS_DIR.glob(f"TOTAL_XGB_Q{period}_*pct.json"))
        if total_xgb:
            available["total"][period] = f"total_xgb_Q{period}"
        else:
            available["total"][period] = "total_simple"

    return available
