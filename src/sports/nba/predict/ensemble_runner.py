"""Ensemble Runner: combina predicciones de XGBoost + CatBoost.

Usa weighted average con pesos fijos:
  ML: XGB 60% + CatBoost 40% (mejor calibracion para Kelly sizing)
  O/U: XGB solo (unico modelo funcional — NN no mejora sobre ~50%)

Por que estos pesos?
  - XGB+CatBoost cometen errores diferentes (depth-wise vs symmetric trees)
  - 60/40 tiene el mejor cw-ECE (0.037) y log_loss (0.6238) en test
  - CatBoost solo tiene mejor accuracy (66.3%) pero peor calibracion
  - Para apuestas, calibracion > accuracy (Kelly sizing depende de probabilidades exactas)

Nota: NN fue retirado del ensemble porque:
  1. ML: en OOF stacking colapsaba a 56.7% (arrastraba al meta-learner)
  2. O/U: ~50% accuracy, no mejor que random
  Esto elimina la dependencia de TensorFlow del pipeline de prediccion.
"""
import re
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from colorama import Fore, Style, init, deinit
from src.sports.nba.predict import xgboost_runner as XGBoost_Runner
from src.core.betting import expected_value as Expected_Value
from src.core.betting import kelly_criterion as kc
from src.core.betting.robust_kelly import calculate_robust_kelly_simple
from src.core.betting.spread_math import p_cover, expected_margin, ah_probabilities, p_cover_regression, p_win_from_margin, p_cover_from_residual
from src.core.betting.expected_value import ah_expected_value
from src.sports.nba.predict.margin_runner import predict_margins, predict_margin_sigma, predict_margin_interval, get_conformal_regressor, is_residual_model
from src.sports.nba.predict.totals_runner import predict_totals, p_over as totals_p_over, get_totals_conformal

init()

from src.config import CATBOOST_MODELS_DIR, NBA_ML_MODELS_DIR, get_logger

logger = get_logger(__name__)

# --- Conformal prediction para filtrar juegos inciertos ---
_ensemble_conformal = None

# --- Modelo de varianza per-game para Kelly adaptativo ---
_variance_model = None

# --- Cache de blocks para Telegram formatter ---
_last_blocks = []

# Rango valido de epsilon (mismo que fit_variance_model.py)
SIGMA_MIN = 0.02
SIGMA_MAX = 0.20

# --- AH confidence filters ---
# Filtros para reducir picks AH de baja confianza (basado en analisis de resultados):
AH_MIN_PCOVER = 0.54     # minimo P(cover) para recomendar AH
AH_MAX_SPREAD = 12.0     # no recomendar AH con |spread| > 12 (demasiado volatil)
AH_MIN_KELLY = 0.3       # minimo Kelly % para mostrar pick AH
AH_SIGMA_MAX = 0.12      # si sigma del ensemble > threshold, skip AH (modelos no concuerdan)
AH_DOG_SPREAD_MAX = 8.0  # para underdogs: no recomendar si |spread| > 8 AND P(cover) < 58%
AH_DOG_PCOVER_MIN = 0.58 # umbral P(cover) para underdogs con spread grande
AH_REG_BLEND_W = 0.6     # peso del modelo de margen en blend (0.6 reg + 0.4 clf)

# --- Pesos del ensemble ---
# Evaluacion v2 en test set (809 juegos, 2025-10 a 2026-02), 188 features:
#   XGB 64.8% + CatBoost 65.6% + Diff_TS_PCT feature
#   XGB 25/Cat 75: acc=65.6%, ECE=0.0355  <- MEJOR (grid search optimo)
#   XGB 60/Cat 40: acc=64.8%, ECE=0.0437  (anterior)
#   Optuna-tuned: 95/5 XGB/Cat — mejor calibracion (ECE 0.031)
# Pesos se cargan desde metadata.json si disponible (ver _load_ensemble_weights)
W_XGB_ML = 0.95
W_CAT_ML = 0.05

# --- CatBoost model loading ---
CATBOOST_ACCURACY_PATTERN = re.compile(r"CatBoost_(\d+(?:\.\d+)?)%_")
LIGHTGBM_ACCURACY_PATTERN = re.compile(r"LightGBM_(\d+(?:\.\d+)?)%_")

# Modelos cacheados (se cargan una vez)
_catboost_ml = None
_catboost_calibrator = None
_lightgbm_ml = None
_lightgbm_calibrator = None


def _select_catboost_path(kind="ML"):
    """Selecciona el mejor modelo CatBoost por accuracy + fecha de modificacion."""
    candidates = list(CATBOOST_MODELS_DIR.glob(f"*{kind}*.pkl"))
    # Excluir archivos de calibracion
    candidates = [c for c in candidates if "calibration" not in c.name]
    if not candidates:
        raise FileNotFoundError(f"No CatBoost {kind} model found in {CATBOOST_MODELS_DIR}")

    def score(path):
        match = CATBOOST_ACCURACY_PATTERN.search(path.name)
        accuracy = float(match.group(1)) if match else 0.0
        return (accuracy, path.stat().st_mtime)

    return max(candidates, key=score)


def _load_catboost():
    """Carga el modelo CatBoost ML + calibrator opcional (lazy, una sola vez)."""
    global _catboost_ml, _catboost_calibrator
    if _catboost_ml is None:
        path = _select_catboost_path("ML")
        _catboost_ml = joblib.load(path)
        logger.info("CatBoost ML loaded: %s", path.name)

        # Intentar cargar calibrator
        cal_path = path.with_name(f"{path.stem}_calibration.pkl")
        if cal_path.exists():
            try:
                _catboost_calibrator = joblib.load(cal_path)
                logger.info("CatBoost calibrator loaded: %s", cal_path.name)
            except Exception as e:
                logger.warning("Error loading CatBoost calibrator: %s", e)
    return _catboost_ml


def _load_lightgbm():
    """Carga el modelo LightGBM ML + calibrator opcional (lazy, una sola vez).

    Retorna None si no existe ningun modelo LightGBM (opcional en el ensemble).
    """
    global _lightgbm_ml, _lightgbm_calibrator
    if _lightgbm_ml is None:
        candidates = list(NBA_ML_MODELS_DIR.glob("LightGBM_*ML*.txt"))
        if not candidates:
            _lightgbm_ml = False
            logger.debug("No LightGBM model found — 2-model ensemble")
            return None

        def score(path):
            match = LIGHTGBM_ACCURACY_PATTERN.search(path.name)
            accuracy = float(match.group(1)) if match else 0.0
            return (accuracy, path.stat().st_mtime)

        path = max(candidates, key=score)
        try:
            import lightgbm as _lgb
            _lightgbm_ml = _lgb.Booster(model_file=str(path))
            logger.info("LightGBM ML loaded: %s", path.name)

            cal_path = path.with_name(f"{path.stem}_calibration.pkl")
            if cal_path.exists():
                try:
                    _lightgbm_calibrator = joblib.load(cal_path)
                    logger.info("LightGBM calibrator loaded: %s", cal_path.name)
                except Exception as e:
                    logger.warning("Error loading LightGBM calibrator: %s", e)
        except Exception as e:
            logger.warning("Error loading LightGBM: %s", e)
            _lightgbm_ml = False

    return _lightgbm_ml if _lightgbm_ml is not False else None


def _load_ensemble_conformal(sportsbook=None):
    """Carga el ConformalClassifier del ensemble (lazy, una sola vez).

    Busca ensemble_conformal_{sportsbook}.pkl primero, fallback a ensemble_conformal.pkl.
    Si no existe ningun pkl, retorna None (conformal es opcional).
    """
    global _ensemble_conformal
    if _ensemble_conformal is None:
        # Try per-book first
        if sportsbook:
            book_path = NBA_ML_MODELS_DIR / f"ensemble_conformal_{sportsbook}.pkl"
            if book_path.exists():
                try:
                    _ensemble_conformal = joblib.load(book_path)
                    logger.info("Ensemble conformal (per-book %s) loaded: threshold=%.4f",
                                sportsbook, _ensemble_conformal.threshold_)
                    return _ensemble_conformal
                except Exception as e:
                    logger.warning("Error loading per-book conformal: %s", e)

        # Fallback to global
        conformal_path = NBA_ML_MODELS_DIR / "ensemble_conformal.pkl"
        if conformal_path.exists():
            try:
                _ensemble_conformal = joblib.load(conformal_path)
                logger.info("Ensemble conformal loaded: threshold=%.4f, coverage=%.1f%%",
                            _ensemble_conformal.threshold_, _ensemble_conformal.coverage_ * 100)
            except Exception as e:
                logger.warning("Error loading ensemble conformal: %s", e)
                _ensemble_conformal = False
        else:
            logger.debug("No ensemble_conformal.pkl found — conformal filtering disabled")
            _ensemble_conformal = False
    return _ensemble_conformal if _ensemble_conformal is not False else None


def _load_variance_model():
    """Carga las estadísticas de varianza del ensemble (lazy, una sola vez).

    ensemble_variance.json contiene mean_sigma y sigma_percentiles calculados
    durante el training como |P_xgb - P_cat| (disagreement entre modelos).
    Se usa para mapear disagreement per-game → epsilon para DRO-Kelly:
      disagreement bajo → sigma bajo → Kelly agresivo
      disagreement alto → sigma alto → Kelly conservador

    Si no existe ensemble_variance.json, retorna None (sigma es opcional).
    """
    global _variance_model
    if _variance_model is None:
        variance_path = NBA_ML_MODELS_DIR / "ensemble_variance.json"
        if variance_path.exists():
            try:
                import json
                with open(variance_path) as f:
                    _variance_model = json.load(f)
                logger.info("Variance stats loaded: mean_sigma=%.4f",
                            _variance_model["mean_sigma"])
            except Exception as e:
                logger.warning("Error loading variance stats: %s", e)
                _variance_model = False
        else:
            logger.debug("No ensemble_variance.json found — adaptive Kelly disabled")
            _variance_model = False
    return _variance_model if _variance_model is not False else None


def _predict_sigmas(variance_stats, data, ml_probs, xgb_ml_probs, cat_ml_probs):
    """Estima sigma per-game desde el disagreement XGB↔CatBoost.

    Usa las estadísticas de varianza del training (percentiles de |P_xgb - P_cat|)
    para mapear el disagreement actual a un epsilon calibrado:
      - disagreement < p25 → sigma bajo (modelo confiado, Kelly agresivo)
      - disagreement > p75 → sigma alto (modelos discrepan, Kelly conservador)

    El mapeo lineal interpola entre SIGMA_MIN y SIGMA_MAX usando el rango
    de percentiles observado en training como referencia.

    Output: sigmas clipeados a [SIGMA_MIN, SIGMA_MAX].
    """
    disagreement = np.abs(xgb_ml_probs[:, 1] - cat_ml_probs[:, 1])

    # Usar percentiles del training como referencia de escala
    p25 = variance_stats["sigma_percentiles"]["25"]
    p75 = variance_stats["sigma_percentiles"]["75"]
    denom = p75 - p25 if p75 > p25 else 0.01

    # Mapeo lineal: disagreement → [SIGMA_MIN, SIGMA_MAX]
    # disagreement == p25 → SIGMA_MIN, disagreement == p75 → SIGMA_MAX
    sigmas = SIGMA_MIN + (SIGMA_MAX - SIGMA_MIN) * (disagreement - p25) / denom

    return np.clip(sigmas, SIGMA_MIN, SIGMA_MAX)


def _catboost_predict_proba(model, data):
    """Predice probabilidades con CatBoost, usando Platt si disponible.

    CatBoost.predict_proba() retorna array (N, 2) con [P(away), P(home)].
    Si hay calibrator, aplica Platt scaling para mejorar calibracion.
    """
    raw = model.predict_proba(data)
    if _catboost_calibrator is not None:
        return _catboost_calibrator.calibrate(raw[:, 1])
    return raw


def _load_ensemble_weights():
    """Carga pesos del ensemble desde metadata.json, fallback a defaults."""
    meta_path = NBA_ML_MODELS_DIR / "metadata.json"
    if meta_path.exists():
        try:
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            weights = meta.get("weights", {})
            if weights:
                logger.info("Ensemble weights from metadata: %s", weights)
                return weights
        except Exception:
            pass
    return {"xgb": W_XGB_ML, "cat": W_CAT_ML}


def _generate_all_predictions(data, todays_games_uo, frame_ml):
    """Genera predicciones combinadas XGB+CatBoost(+LightGBM) para ML y XGB para O/U.

    ML: weighted average con pesos de metadata.json (o defaults 60/40).
    Si LightGBM esta disponible y metadata incluye peso lgb, se incluye.
    O/U: Totals model o XGBoost solo.

    Retorna (ml_probs, ou_probs, xgb_ml_probs, cat_ml_probs).
    """
    # --- Cargar modelos ---
    XGBoost_Runner._load_models()
    cat_model = _load_catboost()
    lgb_model = _load_lightgbm()

    # --- XGBoost ML ---
    xgb_ml_probs = XGBoost_Runner._predict_probs(
        XGBoost_Runner.xgb_ml, data, XGBoost_Runner.xgb_ml_calibrator
    )

    # --- CatBoost ML ---
    cat_ml_probs = _catboost_predict_proba(cat_model, data)

    # --- LightGBM ML (opcional) ---
    lgb_ml_probs = None
    if lgb_model is not None:
        p1 = lgb_model.predict(data)
        if _lightgbm_calibrator is not None:
            lgb_ml_probs = _lightgbm_calibrator.calibrate(p1)
        else:
            lgb_ml_probs = np.column_stack([1.0 - p1, p1])
        logger.info("LightGBM predictions: mean_P(home)=%.3f", lgb_ml_probs[:, 1].mean())

    # --- Combinar ML: weighted average ---
    weights = _load_ensemble_weights()
    w_xgb = weights.get("xgb", W_XGB_ML)
    w_cat = weights.get("cat", W_CAT_ML)
    w_lgb = weights.get("lgb", 0.0)

    if lgb_ml_probs is not None and w_lgb > 0:
        ml_probs = w_xgb * xgb_ml_probs + w_cat * cat_ml_probs + w_lgb * lgb_ml_probs
        logger.info("3-model ensemble: XGB %.0f%% + Cat %.0f%% + LGB %.0f%%",
                     w_xgb * 100, w_cat * 100, w_lgb * 100)
    else:
        # Renormalizar pesos si LightGBM no esta disponible
        total = w_xgb + w_cat
        ml_probs = (w_xgb / total) * xgb_ml_probs + (w_cat / total) * cat_ml_probs

    # --- O/U: Totals regression model (preferred) or XGBoost classifier (legacy) ---
    predicted_totals = predict_totals(data)
    if predicted_totals is not None:
        ou_lines = np.asarray(todays_games_uo, dtype=float)
        p_over_arr = totals_p_over(predicted_totals, ou_lines)
        ou_probs = np.column_stack([1.0 - p_over_arr, p_over_arr])
        logger.info("Totals model: mean=%.1f, range=[%.1f, %.1f]",
                     predicted_totals.mean(), predicted_totals.min(), predicted_totals.max())
    elif XGBoost_Runner.xgb_uo is not None:
        frame_uo = frame_ml.copy()
        frame_uo["OU"] = np.asarray(todays_games_uo, dtype=float)
        ou_probs = XGBoost_Runner._predict_probs(
            XGBoost_Runner.xgb_uo,
            frame_uo.values.astype(float),
            XGBoost_Runner.xgb_uo_calibrator,
        )
    else:
        ou_probs = np.full((len(todays_games_uo), 2), 0.5)

    return ml_probs, ou_probs, xgb_ml_probs, cat_ml_probs, predicted_totals


def _build_game_blocks(games, ml_probs, ou_probs, xgb_ml_probs, cat_ml_probs,
                       todays_games_uo, home_team_odds, away_team_odds,
                       kelly_flag, market_info, spread_home_odds, spread_away_odds,
                       conformal_set_sizes=None, conformal_margins=None,
                       sigmas=None, reg_margins=None, reg_sigmas=None,
                       predicted_totals=None):
    """Construye bloques de output por partido con toda la info consolidada.

    Retorna lista de dicts con datos pre-calculados, rankeados por max EV descendente.
    """
    spreads = market_info.get("MARKET_SPREAD", np.zeros(len(games)))
    blocks = []

    for idx, (home_team, away_team) in enumerate(games):
        b = {"idx": idx, "home": home_team, "away": away_team}

        # --- ML pick ---
        winner = int(np.argmax(ml_probs[idx]))
        b["winner"] = winner
        b["pick"] = home_team if winner == 1 else away_team
        b["pick_prob"] = float(ml_probs[idx][winner])
        b["xgb_agree"] = int(np.argmax(xgb_ml_probs[idx])) == int(np.argmax(cat_ml_probs[idx]))
        b["xgb_conf"] = round(xgb_ml_probs[idx][winner] * 100, 1)
        b["cat_conf"] = round(cat_ml_probs[idx][winner] * 100, 1)

        # --- O/U ---
        under_over = int(np.argmax(ou_probs[idx]))
        b["ou_label"] = "OVER" if under_over == 1 else "UNDER"
        b["ou_line"] = todays_games_uo[idx]
        b["ou_conf"] = round(ou_probs[idx][under_over] * 100, 1)
        b["predicted_total"] = float(predicted_totals[idx]) if predicted_totals is not None else None

        # --- EV + Kelly (ML) ---
        ev_home = ev_away = 0.0
        kelly_h = kelly_a = 0.0
        sigma_i = float(sigmas[idx]) if sigmas is not None else 0.05
        b["sigma"] = sigma_i

        if home_team_odds[idx] and away_team_odds[idx]:
            h_odds = int(home_team_odds[idx])
            a_odds = int(away_team_odds[idx])
            ev_home = float(Expected_Value.expected_value(ml_probs[idx][1], h_odds))
            ev_away = float(Expected_Value.expected_value(ml_probs[idx][0], a_odds))
            if sigmas is not None:
                kelly_h = float(calculate_robust_kelly_simple(h_odds, ml_probs[idx][1], epsilon=sigma_i))
                kelly_a = float(calculate_robust_kelly_simple(a_odds, ml_probs[idx][0], epsilon=sigma_i))
            elif kelly_flag:
                kelly_h = float(kc.calculate_eighth_kelly(h_odds, ml_probs[idx][1]))
                kelly_a = float(kc.calculate_eighth_kelly(a_odds, ml_probs[idx][0]))

        b["ev_home"] = ev_home
        b["ev_away"] = ev_away
        b["kelly_home"] = kelly_h
        b["kelly_away"] = kelly_a
        b["max_ev"] = max(ev_home, ev_away)

        # --- Traps ---
        b["trap_home"] = _is_underdog_trap(ml_probs[idx][1], ev_home)
        b["trap_away"] = _is_underdog_trap(ml_probs[idx][0], ev_away)

        # --- Conformal ---
        if conformal_set_sizes is not None:
            ss = int(conformal_set_sizes[idx])
            margin = float(conformal_margins[idx]) if conformal_margins is not None else 0.0
            b["conf_ss"] = ss
            b["conf_margin"] = margin
        else:
            b["conf_ss"] = None

        # --- AH (spread) ---
        line = float(spreads[idx])
        prob_home = float(ml_probs[idx][1])
        b["spread"] = line
        b["margin"] = expected_margin(prob_home)

        sh_odds = int(spread_home_odds[idx]) if spread_home_odds and spread_home_odds[idx] else -110
        sa_odds = int(spread_away_odds[idx]) if spread_away_odds and spread_away_odds[idx] else -110

        # --- Margin regression (compute first, needed for AH blend) ---
        reg_p_home_cover = None
        reg_p_away_cover = None
        if reg_margins is not None:
            raw_pred = float(reg_margins[idx])
            reg_sigma_val = float(reg_sigmas[idx]) if reg_sigmas is not None else None
            b["reg_sigma"] = reg_sigma_val

            if is_residual_model():
                b["reg_residual"] = raw_pred
                b["reg_margin"] = raw_pred - line
                b["reg_p_cover"] = p_cover_from_residual(raw_pred, sigma=reg_sigma_val, line=line) if reg_sigma_val else 0.5
                reg_p_home_cover = b["reg_p_cover"]
                reg_p_away_cover = p_cover_from_residual(-raw_pred, sigma=reg_sigma_val, line=-line) if reg_sigma_val else 0.5

                conformal_reg = get_conformal_regressor()
                if conformal_reg is not None:
                    b["reg_confident"] = conformal_reg.is_confident_residual(raw_pred)
                    b["reg_conf_margin"] = conformal_reg.confidence_margin_residual(raw_pred)
                else:
                    b["reg_confident"] = None
                    b["reg_conf_margin"] = None
            else:
                mu_reg = raw_pred
                b["reg_residual"] = None
                b["reg_margin"] = mu_reg
                b["reg_p_cover"] = p_cover_regression(mu_reg, line, sigma=reg_sigma_val)
                reg_p_home_cover = b["reg_p_cover"]
                reg_p_away_cover = p_cover_regression(-mu_reg, -line, sigma=reg_sigma_val)

                conformal_reg = get_conformal_regressor()
                if conformal_reg is not None:
                    b["reg_confident"] = conformal_reg.is_confident(mu_reg, line)
                    b["reg_conf_margin"] = conformal_reg.confidence_margin(mu_reg, line)
                else:
                    b["reg_confident"] = None
                    b["reg_conf_margin"] = None

            reg_ah_ev_home = float(ah_expected_value(
                {"p_full_win": reg_p_home_cover, "p_half_win": 0.0,
                 "p_half_loss": 0.0, "p_full_loss": 1.0 - reg_p_home_cover,
                 "is_quarter": False}, sh_odds))
            reg_ah_ev_away = float(ah_expected_value(
                {"p_full_win": reg_p_away_cover, "p_half_win": 0.0,
                 "p_half_loss": 0.0, "p_full_loss": 1.0 - reg_p_away_cover,
                 "is_quarter": False}, sa_odds))
            b["reg_ah_ev_home"] = reg_ah_ev_home
            b["reg_ah_ev_away"] = reg_ah_ev_away
            b["reg_p_home_cover"] = reg_p_home_cover
            b["reg_p_away_cover"] = reg_p_away_cover
        else:
            b["reg_margin"] = None
            b["reg_residual"] = None
            b["reg_sigma"] = None

        # --- AH probabilities (ML classifier-based) ---
        home_ah = ah_probabilities(prob_home, line)
        away_ah = ah_probabilities(1.0 - prob_home, -line)

        clf_p_home_cover = home_ah["p_full_win"] + home_ah["p_half_win"]
        clf_p_away_cover = away_ah["p_full_win"] + away_ah["p_half_win"]

        # Blend: if margin model is available and confident, weight it 60/40
        if reg_p_home_cover is not None and b.get("reg_confident"):
            p_home_cover = AH_REG_BLEND_W * reg_p_home_cover + (1 - AH_REG_BLEND_W) * clf_p_home_cover
            p_away_cover = AH_REG_BLEND_W * reg_p_away_cover + (1 - AH_REG_BLEND_W) * clf_p_away_cover
            b["ah_blend"] = "REG+CLF"
        else:
            p_home_cover = clf_p_home_cover
            p_away_cover = clf_p_away_cover
            b["ah_blend"] = "CLF"

        # Recompute AH EV with blended probabilities
        blended_home_ah = {"p_full_win": p_home_cover, "p_half_win": 0.0,
                           "p_half_loss": 0.0, "p_full_loss": 1.0 - p_home_cover,
                           "is_quarter": home_ah["is_quarter"]}
        blended_away_ah = {"p_full_win": p_away_cover, "p_half_win": 0.0,
                           "p_half_loss": 0.0, "p_full_loss": 1.0 - p_away_cover,
                           "is_quarter": away_ah["is_quarter"]}

        ah_ev_home = float(ah_expected_value(blended_home_ah, sh_odds))
        ah_ev_away = float(ah_expected_value(blended_away_ah, sa_odds))
        ah_kelly_home = float(calculate_robust_kelly_simple(sh_odds, p_home_cover, epsilon=sigma_i))
        ah_kelly_away = float(calculate_robust_kelly_simple(sa_odds, p_away_cover, epsilon=sigma_i))

        # Pick best side
        if ah_ev_home > ah_ev_away and ah_ev_home > 0:
            b["ah_side"] = home_team
            b["ah_line"] = f"{line:+.1f}"
            b["ah_ev"] = ah_ev_home
            b["ah_p"] = p_home_cover
            b["ah_kelly"] = ah_kelly_home
            ah_is_home_pick = True
        elif ah_ev_away > 0:
            b["ah_side"] = away_team
            b["ah_line"] = f"{-line:+.1f}"
            b["ah_ev"] = ah_ev_away
            b["ah_p"] = p_away_cover
            b["ah_kelly"] = ah_kelly_away
            ah_is_home_pick = False
        else:
            b["ah_side"] = None
            b["ah_ev"] = max(ah_ev_home, ah_ev_away)
            b["ah_p"] = max(p_home_cover, p_away_cover)
            b["ah_kelly"] = 0.0
            b["ah_line"] = f"{line:+.1f}"
            ah_is_home_pick = None

        b["ah_is_quarter"] = home_ah["is_quarter"]

        # --- AH confidence tag (independent of ML tag) ---
        ah_skip_reasons = []

        if b["ah_side"] is None:
            b["ah_tag"] = "AH-PASS"
        else:
            # Filter 1: spread too large
            if abs(line) > AH_MAX_SPREAD:
                ah_skip_reasons.append(f"|spread|={abs(line):.1f}>{AH_MAX_SPREAD}")

            # Filter 2: P(cover) too low
            if b["ah_p"] < AH_MIN_PCOVER:
                ah_skip_reasons.append(f"P={b['ah_p']:.1%}<{AH_MIN_PCOVER:.0%}")

            # Filter 3: Kelly too low
            if b["ah_kelly"] < AH_MIN_KELLY:
                ah_skip_reasons.append(f"Kelly={b['ah_kelly']:.2f}%<{AH_MIN_KELLY}%")

            # Filter 4: ensemble sigma too high (models disagree)
            if sigma_i > AH_SIGMA_MAX:
                ah_skip_reasons.append(f"σ={sigma_i:.3f}>{AH_SIGMA_MAX}")

            # Filter 5: underdog with large spread
            is_underdog_pick = (ah_is_home_pick and line > 0) or (ah_is_home_pick is False and line < 0)
            if is_underdog_pick and abs(line) > AH_DOG_SPREAD_MAX and b["ah_p"] < AH_DOG_PCOVER_MIN:
                ah_skip_reasons.append(f"dog |spread|={abs(line):.1f} P={b['ah_p']:.1%}")

            # Filter 6: ML and margin model disagree on AH side
            if reg_p_home_cover is not None:
                clf_side_home = clf_p_home_cover > clf_p_away_cover
                reg_side_home = reg_p_home_cover > reg_p_away_cover
                if clf_side_home != reg_side_home:
                    ah_skip_reasons.append("CLF↔REG disagree")

            if ah_skip_reasons:
                b["ah_tag"] = "AH-SKIP"
                b["ah_skip_reasons"] = ah_skip_reasons
            else:
                b["ah_tag"] = "AH-BET"
                b["ah_skip_reasons"] = []

        blocks.append(b)

    # Rankear por max EV descendente
    blocks.sort(key=lambda x: x["max_ev"], reverse=True)
    return blocks


def _print_compact_output(blocks, kelly_flag, conformal=None, has_sigma=False):
    """Imprime bloques compactos por partido, rankeados por EV."""
    # Threshold de Kelly para permitir BET con conformal incierto (set_size=2)
    CONF2_KELLY_THRESHOLD = 0.5  # % del bankroll

    # Header
    n_total = len(blocks)
    n_bet = sum(
        1 for b in blocks
        if b["conf_ss"] == 1 or (b["conf_ss"] == 2 and max(b["kelly_home"], b["kelly_away"]) >= CONF2_KELLY_THRESHOLD)
    ) if blocks[0]["conf_ss"] is not None else n_total
    sigma_label = " | DRO-Kelly" if has_sigma else ""
    conf_label = f" | Conformal {n_bet}/{n_total}" if blocks[0]["conf_ss"] is not None else ""
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"  PICKS ranked by EV{conf_label}{sigma_label}")
    print(f"{'='*60}{Style.RESET_ALL}\n")

    # Threshold de Kelly para permitir BET con conformal incierto (set_size=2)
    CONF2_KELLY_THRESHOLD = 0.5  # % del bankroll

    for rank, b in enumerate(blocks, 1):
        # --- Status tag: BET / SKIP / TRAP ---
        has_trap = b["trap_home"] or b["trap_away"]
        max_kelly = max(b["kelly_home"], b["kelly_away"])
        conf_uncertain = b["conf_ss"] is not None and b["conf_ss"] != 1
        # Permitir BET con conf=2 si Kelly supera el umbral (edge robusto a pesar de incertidumbre)
        conf_override = conf_uncertain and max_kelly >= CONF2_KELLY_THRESHOLD

        if has_trap:
            tag = f"{Fore.YELLOW}TRAP{Style.RESET_ALL}"
        elif conf_uncertain and not conf_override:
            tag = f"{Fore.YELLOW}SKIP{Style.RESET_ALL}"
        elif b["max_ev"] > 0:
            tag = f"{Fore.GREEN}BET{Style.RESET_ALL}"
        else:
            tag = f"{Fore.RED}PASS{Style.RESET_ALL}"

        # --- Sigma color ---
        s = b["sigma"]
        sigma_color = Fore.GREEN if s < 0.07 else (Fore.YELLOW if s < 0.10 else Fore.RED)

        # --- Agreement ---
        agree = "+" if b["xgb_agree"] else "~"

        # --- Line 1: pick + confidence + tag ---
        pick_color = Fore.GREEN if b["winner"] == 1 else Fore.RED
        loser = b["away"] if b["winner"] == 1 else b["home"]
        loser_color = Fore.RED if b["winner"] == 1 else Fore.GREEN
        print(
            f"  {Fore.CYAN}#{rank}{Style.RESET_ALL}  "
            f"{pick_color}{b['pick']}{Style.RESET_ALL} "
            f"({b['pick_prob']*100:.1f}%) vs "
            f"{loser_color}{loser}{Style.RESET_ALL}  "
            f"[{tag}]"
        )

        # --- Line 2: ML EV + Kelly + models ---
        ev_best_side = "home" if b["ev_home"] >= b["ev_away"] else "away"
        ev_val = b[f"ev_{ev_best_side}"]
        kelly_val = b[f"kelly_{ev_best_side}"]
        ev_color = Fore.GREEN if ev_val > 0 else Fore.RED
        ev_team = b["home"] if ev_best_side == "home" else b["away"]
        kelly_str = f"  Kelly={kelly_val:.2f}%" if kelly_flag else ""
        print(
            f"       ML  {ev_team}: EV={ev_color}{ev_val:+.1f}{Style.RESET_ALL}{kelly_str}"
            f"  [{agree} XGB:{b['xgb_conf']}% Cat:{b['cat_conf']}%]"
            f"  {sigma_color}σ={s:.3f}{Style.RESET_ALL}"
        )

        # --- Line 3: AH with tag ---
        ah_tag = b.get("ah_tag", "AH-PASS")
        if ah_tag == "AH-BET":
            ah_tag_str = f"{Fore.GREEN}AH-BET{Style.RESET_ALL}"
        elif ah_tag == "AH-SKIP":
            ah_tag_str = f"{Fore.YELLOW}AH-SKIP{Style.RESET_ALL}"
        else:
            ah_tag_str = f"{Fore.RED}AH-PASS{Style.RESET_ALL}"

        blend_str = f" ({b.get('ah_blend', 'CLF')})" if b.get("ah_blend") == "REG+CLF" else ""

        if b["ah_side"]:
            ah_ev_color = Fore.GREEN if b["ah_ev"] > 0 else Fore.RED
            q_tag = " Q" if b["ah_is_quarter"] else ""
            print(
                f"       AH  [{ah_tag_str}] {b['ah_side']} ({b['ah_line']}{q_tag}): "
                f"P={b['ah_p']:.1%} EV={ah_ev_color}{b['ah_ev']:+.1f}{Style.RESET_ALL} "
                f"Kelly={b['ah_kelly']:.2f}%"
                f"  margin={b['margin']:+.1f}{blend_str}"
            )
            # Show skip reasons if AH-SKIP
            if ah_tag == "AH-SKIP" and b.get("ah_skip_reasons"):
                reasons = ", ".join(b["ah_skip_reasons"])
                print(f"           {Fore.YELLOW}! AH skip: {reasons}{Style.RESET_ALL}")
        else:
            print(
                f"       AH  [{ah_tag_str}] spread {b['ah_line']}, margin={b['margin']:+.1f}"
                f"  EV={b['ah_ev']:+.1f}"
            )

        # --- Line 3b: Margin regression comparison (if available) ---
        if b["reg_margin"] is not None:
            reg_conf_str = ""
            if b.get("reg_confident") is not None:
                conf_tag = f"{Fore.GREEN}CONF" if b["reg_confident"] else f"{Fore.YELLOW}WEAK"
                reg_conf_str = f"  [{conf_tag}{Style.RESET_ALL} Δ={b['reg_conf_margin']:+.1f}]"
            sigma_str = f" σ={b['reg_sigma']:.1f}" if b.get("reg_sigma") else ""
            if b.get("reg_residual") is not None:
                print(
                    f"           CLF: μ={b['margin']:+.1f}  |  "
                    f"REG: res={b['reg_residual']:+.1f} μ={b['reg_margin']:+.1f} "
                    f"P(cover)={b['reg_p_cover']:.1%}{sigma_str}{reg_conf_str}"
                )
            else:
                print(
                    f"           CLF: μ={b['margin']:+.1f}  |  "
                    f"REG: μ={b['reg_margin']:+.1f} P(cover)={b['reg_p_cover']:.1%}{sigma_str}{reg_conf_str}"
                )

        # --- Line 4: TRAP redirect ---
        if has_trap:
            trap_side = "away" if b["trap_away"] else "home"
            trap_team = b[trap_side]
            print(
                f"       {Fore.YELLOW}! underdog {trap_team} EV+ es trampa — usar AH del favorito{Style.RESET_ALL}"
            )

        # --- O/U ---
        ou_color = Fore.MAGENTA if b["ou_label"] == "UNDER" else Fore.BLUE
        pred_str = f" pred={b['predicted_total']:.0f}" if b.get("predicted_total") is not None else ""
        print(
            f"       O/U {ou_color}{b['ou_label']}{Style.RESET_ALL} "
            f"{b['ou_line']} ({b['ou_conf']}%){pred_str}"
        )

        # --- Conformal margin ---
        if b["conf_ss"] is not None:
            if b["conf_ss"] == 1:
                print(f"       {Fore.GREEN}conformal: margin={b['conf_margin']:.2f}{Style.RESET_ALL}")
            elif conf_override:
                print(f"       {Fore.CYAN}conformal: set_size=2 pero Kelly={max_kelly:.2f}% >= {CONF2_KELLY_THRESHOLD}% → BET override{Style.RESET_ALL}")
            else:
                print(f"       {Fore.YELLOW}conformal: skip (set_size={b['conf_ss']}){Style.RESET_ALL}")

        print()  # blank line between games


def _is_underdog_trap(prob, ev):
    """Detecta EV+ en underdog con prob < 35% — empiricamente nunca se da."""
    return prob < 0.35 and ev > 0


def _build_prediction_results(games, ml_probs, ou_probs, todays_games_uo, home_team_odds,
                              away_team_odds, market_info, conformal_set_sizes=None,
                              conformal_margins=None, sigmas=None,
                              spread_home_odds=None, spread_away_odds=None,
                              reg_margins=None, reg_sigmas=None):
    """Empaqueta predicciones en lista de dicts para BetTracker.

    market_info contiene MARKET_SPREAD y MARKET_ML_PROB extraidos antes
    del drop de features (MARKET_SPREAD es feature redundante pero se
    necesita para display/tracking).

    Conformal fields:
      conformal_set_size: 1=confiado (apostar), 2=incierto (skip), 0=vacio
      conformal_margin: max_prob - threshold (mayor = mas confianza)

    Sigma (varianza per-game):
      sigma: epsilon para DRO-Kelly. Bajo = Kelly agresivo, alto = conservador.
      Cuando sigma esta disponible, kelly_home/away se calculan con robust_kelly.
    """
    spreads = market_info.get("MARKET_SPREAD", np.zeros(len(games)))
    market_probs = market_info.get("MARKET_ML_PROB", np.full(len(games), 0.5))

    result = []
    for idx, (home_team, away_team) in enumerate(games):
        if not home_team_odds[idx] or not away_team_odds[idx]:
            continue
        h_odds = int(home_team_odds[idx])
        a_odds = int(away_team_odds[idx])

        # Kelly: adaptativo (sigma) si disponible, fijo (eighth) si no
        if sigmas is not None:
            sigma_i = float(sigmas[idx])
            kelly_h = float(calculate_robust_kelly_simple(h_odds, ml_probs[idx][1], epsilon=sigma_i))
            kelly_a = float(calculate_robust_kelly_simple(a_odds, ml_probs[idx][0], epsilon=sigma_i))
        else:
            sigma_i = None
            kelly_h = float(kc.calculate_eighth_kelly(h_odds, ml_probs[idx][1]))
            kelly_a = float(kc.calculate_eighth_kelly(a_odds, ml_probs[idx][0]))

        entry = {
            "home_team": home_team,
            "away_team": away_team,
            "prob_home": float(ml_probs[idx][1]),
            "prob_away": float(ml_probs[idx][0]),
            "prob_over": float(ou_probs[idx][1]),
            "prob_under": float(ou_probs[idx][0]),
            "ou_line": float(todays_games_uo[idx]),
            "ml_home_odds": h_odds,
            "ml_away_odds": a_odds,
            "spread": float(spreads[idx]),
            "market_prob_home": float(market_probs[idx]),
            "ev_home": float(Expected_Value.expected_value(ml_probs[idx][1], h_odds)),
            "ev_away": float(Expected_Value.expected_value(ml_probs[idx][0], a_odds)),
            "kelly_home": kelly_h,
            "kelly_away": kelly_a,
        }
        # Conformal: anotar cada prediccion
        if conformal_set_sizes is not None:
            entry["conformal_set_size"] = int(conformal_set_sizes[idx])
            entry["conformal_margin"] = float(conformal_margins[idx])
        # Sigma (varianza per-game)
        if sigma_i is not None:
            entry["sigma"] = sigma_i

        # --- Underdog trap flags ---
        entry["trap_home"] = _is_underdog_trap(ml_probs[idx][1], entry["ev_home"])
        entry["trap_away"] = _is_underdog_trap(ml_probs[idx][0], entry["ev_away"])

        # --- Asian Handicap (Spread) ---
        line = float(spreads[idx])
        prob_home = float(ml_probs[idx][1])

        # Settlement completo: quarter lines (.25, .75) tienen half win/loss
        home_ah = ah_probabilities(prob_home, line)
        # Para away: invertir la línea
        away_ah = ah_probabilities(1.0 - prob_home, -line)

        entry["ah_spread"] = line
        entry["ah_prob_home_cover"] = home_ah["p_full_win"] + home_ah["p_half_win"]
        entry["ah_prob_away_cover"] = away_ah["p_full_win"] + away_ah["p_half_win"]
        entry["ah_expected_margin"] = expected_margin(prob_home)

        sh_odds = int(spread_home_odds[idx]) if spread_home_odds and spread_home_odds[idx] else -110
        sa_odds = int(spread_away_odds[idx]) if spread_away_odds and spread_away_odds[idx] else -110
        entry["ah_home_odds"] = sh_odds
        entry["ah_away_odds"] = sa_odds

        # EV con settlement correcto (half win/loss para quarter lines)
        entry["ah_ev_home"] = float(ah_expected_value(home_ah, sh_odds))
        entry["ah_ev_away"] = float(ah_expected_value(away_ah, sa_odds))

        # Kelly usa P(full_win + half_win) como proxy de "probability of profit"
        eps = sigma_i if sigma_i is not None else 0.05
        p_home_profit = home_ah["p_full_win"] + home_ah["p_half_win"]
        p_away_profit = away_ah["p_full_win"] + away_ah["p_half_win"]
        entry["ah_kelly_home"] = float(calculate_robust_kelly_simple(sh_odds, p_home_profit, epsilon=eps))
        entry["ah_kelly_away"] = float(calculate_robust_kelly_simple(sa_odds, p_away_profit, epsilon=eps))

        # --- AH tag (confidence filter) ---
        ah_best_p = max(p_home_profit, p_away_profit)
        ah_best_kelly = max(entry["ah_kelly_home"], entry["ah_kelly_away"])
        ah_best_ev = max(entry["ah_ev_home"], entry["ah_ev_away"])
        ah_sigma = sigma_i if sigma_i is not None else 0.05
        ah_skip = (
            ah_best_ev <= 0
            or abs(line) > AH_MAX_SPREAD
            or ah_best_p < AH_MIN_PCOVER
            or ah_best_kelly < AH_MIN_KELLY
            or ah_sigma > AH_SIGMA_MAX
        )
        entry["ah_tag"] = "AH-PASS" if ah_best_ev <= 0 else ("AH-SKIP" if ah_skip else "AH-BET")

        # --- Margin Regression (si disponible) ---
        if reg_margins is not None:
            raw_pred = float(reg_margins[idx])
            reg_sigma_val = float(reg_sigmas[idx]) if reg_sigmas is not None else None
            entry["reg_sigma"] = reg_sigma_val

            if is_residual_model():
                entry["reg_residual"] = raw_pred
                entry["reg_margin"] = raw_pred - line
                entry["reg_p_win"] = p_win_from_margin(raw_pred - line)
                entry["reg_p_cover_home"] = p_cover_from_residual(raw_pred, sigma=reg_sigma_val) if reg_sigma_val else 0.5
            else:
                entry["reg_margin"] = raw_pred
                entry["reg_p_win"] = p_win_from_margin(raw_pred)
                entry["reg_p_cover_home"] = p_cover_regression(raw_pred, line, sigma=reg_sigma_val)

        result.append(entry)
    return result


def ensemble_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion, market_info=None, spread_home_odds=None, spread_away_odds=None, sportsbook=None, data_margin=None):
    """Ejecuta predicciones combinadas de XGBoost 60% + CatBoost 40%.

    Con conformal prediction (si ensemble_conformal.pkl existe):
      - set_size=1: juego confiado → apostar (solo una clase plausible)
      - set_size=2: juego incierto → skip (ambas clases plausibles)
      - Filtra juegos inciertos → menos apuestas pero mayor hit rate

    Con variance model (si ensemble_variance.json existe):
      - sigma per-game → epsilon para DRO-Kelly
      - sigma bajo → Kelly agresivo, sigma alto → Kelly conservador

    Asian Handicap: convierte P(win) → P(cover spread) analíticamente.
    """
    if market_info is None:
        market_info = {}
    ml_probs, ou_probs, xgb_ml_probs, cat_ml_probs, predicted_totals = _generate_all_predictions(
        data, todays_games_uo, frame_ml
    )

    # --- Conformal prediction sets ---
    conformal_set_sizes = None
    conformal_margins = None
    conformal = _load_ensemble_conformal(sportsbook=sportsbook)
    if conformal is not None:
        conformal_set_sizes, conformal_margins = conformal.predict_confidence(ml_probs)
        n_bet = int((conformal_set_sizes == 1).sum())
        n_skip = len(games) - n_bet
        logger.info("Conformal: %d/%d confiados (threshold=%.4f)",
                     n_bet, len(games), conformal.threshold_)

    # --- Variance model → sigma per-game ---
    sigmas = None
    variance_model = _load_variance_model()
    if variance_model is not None:
        sigmas = _predict_sigmas(variance_model, data, ml_probs, xgb_ml_probs, cat_ml_probs)
        logger.info("Sigma stats: mean=%.4f, std=%.4f, range=[%.4f, %.4f]",
                     sigmas.mean(), sigmas.std(), sigmas.min(), sigmas.max())

    # --- Margin regression model (opcional) ---
    reg_margins = None
    if data_margin is not None:
        logger.info("Margin model: using dedicated margin features (%d cols)", data_margin.shape[1])
        try:
            reg_margins = predict_margins(data_margin)
        except Exception as e:
            logger.warning("Margin prediction failed: %s", e)
    else:
        logger.warning("Margin model: no dedicated features available, skipping")
    reg_sigmas = None
    if reg_margins is not None:
        logger.info("Margin model: mean=%.1f, std=%.1f, range=[%.1f, %.1f]",
                     reg_margins.mean(), reg_margins.std(), reg_margins.min(), reg_margins.max())
        # Sigma calibrado por bucket de spread
        spreads = market_info.get("MARKET_SPREAD", np.zeros(len(games)))
        reg_sigmas = predict_margin_sigma(spreads)
        # Quantile interval (Q10/Q90) for uncertainty estimation
        interval_result = predict_margin_interval(margin_input)
        if interval_result is not None:
            q10, q90, interval_width = interval_result
            logger.info("Margin interval: mean_width=%.1f, range=[%.1f, %.1f]",
                         interval_width.mean(), interval_width.min(), interval_width.max())

    # --- Output compacto: un bloque por partido, rankeado por EV ---
    blocks = _build_game_blocks(
        games, ml_probs, ou_probs, xgb_ml_probs, cat_ml_probs,
        todays_games_uo, home_team_odds, away_team_odds,
        kelly_criterion, market_info, spread_home_odds, spread_away_odds,
        conformal_set_sizes, conformal_margins, sigmas, reg_margins, reg_sigmas,
        predicted_totals,
    )
    _print_compact_output(blocks, kelly_criterion, conformal, sigmas is not None)

    global _last_blocks
    _last_blocks = blocks

    deinit()
    return _build_prediction_results(
        games, ml_probs, ou_probs, todays_games_uo,
        home_team_odds, away_team_odds, market_info,
        conformal_set_sizes, conformal_margins, sigmas,
        spread_home_odds, spread_away_odds, reg_margins, reg_sigmas,
    )
