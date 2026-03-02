"""MLB First 5 Innings (F5) runner.

F5 es el mercado con mayor edge en MLB porque:
  1. Aísla completamente al lanzador abridor (el predictor #1 en beisbol)
  2. Elimina el bullpen — la mayor fuente de varianza impredecible
  3. F5 lanzadores con FIP bajo vs bullpens malos = edge significativo
  4. Lineas de F5 suelen ser menos eficientes que full-game

El modelo F5 es un ensemble independiente del modelo full-game:
  - Mismas features de SP, batting, ELO, park
  - Target: equipo ganando despues de 5 innings (run differential > 0)
  - Calibrado con conformal per-sportsbook en models/mlb/f5/

Mercado en Odds API: h2h_1st_5_innings
"""
import re
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.core.betting import expected_value as Expected_Value
from src.core.betting import kelly_criterion as kc
from src.core.betting.robust_kelly import calculate_robust_kelly_simple
from src.core.calibration.conformal import ConformalClassifier
from src.sports.mlb.config_paths import MLB_F5_MODELS_DIR
from src.config import get_logger

logger = get_logger(__name__)

# --- Accuracy patterns (mismo formato que ensemble_runner.py) ---
XGB_ACCURACY_PATTERN = re.compile(r"XGBoost_(\d+(?:\.\d+)?)%_")
CATBOOST_ACCURACY_PATTERN = re.compile(r"CatBoost_(\d+(?:\.\d+)?)%_")

# --- Default weights para F5 ensemble ---
F5_W_XGB = 0.65
F5_W_CAT = 0.35

# --- Cache de modelos (lazy loading) ---
_f5_xgb = None
_f5_xgb_calibrator = None
_f5_cat = None
_f5_cat_calibrator = None
_f5_conformal = None
_f5_weights = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _select_model_path(model_dir: Path, pattern: re.Pattern, kind: str,
                       ext: str = None) -> Optional[Path]:
    """Selecciona el mejor modelo F5 por accuracy. Retorna None si no existe."""
    if not model_dir.exists():
        return None

    if ext:
        candidates = list(model_dir.glob(f"*{kind}*{ext}"))
    else:
        candidates = (
            list(model_dir.glob(f"*{kind}*.json"))
            + list(model_dir.glob(f"*{kind}*.pkl"))
        )
    candidates = [c for c in candidates if "calibrat" not in c.name.lower() and "conformal" not in c.name.lower()]

    if not candidates:
        return None

    def score(path: Path):
        match = pattern.search(path.name)
        acc = float(match.group(1)) if match else 0.0
        return (acc, path.stat().st_mtime)

    return max(candidates, key=score)


def _load_f5_weights(model_dir: Path) -> dict:
    """Carga pesos desde f5_metadata.json si existe, sino usa defaults."""
    global _f5_weights
    if _f5_weights is not None:
        return _f5_weights

    metadata_path = model_dir / "metadata.json"
    if metadata_path.exists():
        import json
        try:
            with open(metadata_path) as f:
                meta = json.load(f)
            w = meta.get("ensemble_weights", {})
            _f5_weights = {
                "xgb": float(w.get("xgb", F5_W_XGB)),
                "cat": float(w.get("cat", F5_W_CAT)),
            }
            logger.info(
                "F5 weights from metadata.json: XGB=%.0f%% Cat=%.0f%%",
                _f5_weights["xgb"] * 100, _f5_weights["cat"] * 100,
            )
        except Exception as e:
            logger.debug("Could not load F5 metadata.json: %s", e)
            _f5_weights = {"xgb": F5_W_XGB, "cat": F5_W_CAT}
    else:
        _f5_weights = {"xgb": F5_W_XGB, "cat": F5_W_CAT}

    return _f5_weights


def _load_f5_conformal(model_dir: Path, sportsbook: str = None) -> Optional[ConformalClassifier]:
    """Carga conformal per-sportsbook para F5."""
    global _f5_conformal
    if _f5_conformal is not None:
        return _f5_conformal

    candidates = []
    if sportsbook:
        p = model_dir / f"f5_conformal_{sportsbook}.pkl"
        if p.exists():
            candidates.append(p)
    generic = model_dir / "f5_conformal.pkl"
    if generic.exists():
        candidates.append(generic)

    if candidates:
        try:
            _f5_conformal = joblib.load(candidates[0])
            logger.info("F5 conformal loaded: %s", candidates[0].name)
        except Exception as e:
            logger.warning("Could not load F5 conformal: %s", e)

    return _f5_conformal


# ---------------------------------------------------------------------------
# Public: load_f5_models
# ---------------------------------------------------------------------------

def load_f5_models(model_dir: Path = None) -> Optional[tuple]:
    """Carga modelos F5 desde models/mlb/f5/.

    Returns:
        Tuple (xgb_model, cat_model, weights_dict, conformal_or_None)
        si los modelos existen, None si el directorio no tiene modelos.

    Falla silenciosamente — F5 es un mercado opcional, no bloquea
    el pipeline principal de full-game.
    """
    global _f5_xgb, _f5_xgb_calibrator, _f5_cat, _f5_cat_calibrator

    d = Path(model_dir or MLB_F5_MODELS_DIR)

    if not d.exists():
        logger.debug("F5 model directory not found: %s (market optional)", d)
        return None

    # --- XGBoost F5 ---
    xgb_path = _select_model_path(d, XGB_ACCURACY_PATTERN, "F5", ext=".json")
    if xgb_path is None:
        xgb_path = _select_model_path(d, XGB_ACCURACY_PATTERN, "ML", ext=".json")

    if xgb_path is None:
        logger.debug("No XGBoost F5 model found in %s", d)
        return None

    if _f5_xgb is None:
        _f5_xgb = xgb.Booster()
        _f5_xgb.load_model(str(xgb_path))
        logger.info("XGBoost F5 loaded: %s", xgb_path.name)

        cal_path = xgb_path.with_name(f"{xgb_path.stem}_calibration.pkl")
        if cal_path.exists():
            _f5_xgb_calibrator = joblib.load(cal_path)

    # --- CatBoost F5 ---
    cat_path = _select_model_path(d, CATBOOST_ACCURACY_PATTERN, "F5", ext=".pkl")
    if cat_path is None:
        cat_path = _select_model_path(d, CATBOOST_ACCURACY_PATTERN, "ML", ext=".pkl")

    if cat_path is not None and _f5_cat is None:
        try:
            _f5_cat = joblib.load(cat_path)
            logger.info("CatBoost F5 loaded: %s", cat_path.name)
            cal_path = cat_path.with_name(f"{cat_path.stem}_calibration.pkl")
            if cal_path.exists():
                _f5_cat_calibrator = joblib.load(cal_path)
        except Exception as e:
            logger.warning("Could not load CatBoost F5: %s", e)

    weights = _load_f5_weights(d)
    conformal = _load_f5_conformal(d)

    return _f5_xgb, _f5_cat, weights, conformal


# ---------------------------------------------------------------------------
# Public: predict_f5
# ---------------------------------------------------------------------------

def predict_f5(features: Optional[pd.DataFrame], odds_data: list[dict] = None,
               sportsbook: str = "fanduel") -> list[dict]:
    """Ejecuta predicciones F5 para los juegos de hoy.

    Si features es None y odds_data contiene f5 odds, usa solo las odds
    para calcular un EV simple sin modelo (fallback de mercado).

    Si features no es None, corre el ensemble F5 y calcula EV con modelo.

    Args:
        features:   DataFrame con features por juego (puede ser None).
                    Si se provee, debe tener columnas 'home_team' y 'away_team'.
        odds_data:  Lista de dicts con odds (de MLBOddsProvider.get_all_odds_with_f5()).
                    Debe tener 'f5_ml_home', 'f5_ml_away' por juego.
        sportsbook: Nombre del sportsbook para conformal per-book.

    Returns:
        Lista de dicts, uno por juego:
          {home_team, away_team, f5_prob_home, f5_prob_away,
           f5_ev_home, f5_ev_away, f5_ev, f5_kelly, f5_tag, f5_conf_set_size}
    """
    results = []

    # Construir lookup de F5 odds por (home, away)
    f5_odds_by_key = {}
    for game in (odds_data or []):
        k = (game.get("home_team", ""), game.get("away_team", ""))
        if game.get("f5_ml_home") is not None or game.get("f5_ml_away") is not None:
            f5_odds_by_key[k] = game

    if not f5_odds_by_key:
        logger.debug("No F5 odds available in odds_data")
        return []

    # --- Path A: modelo F5 disponible ---
    models_tuple = load_f5_models()
    use_model = models_tuple is not None and features is not None and len(features) > 0

    if use_model:
        xgb_model, cat_model, weights, conformal = models_tuple

        # Recargar conformal per-sportsbook si es necesario
        global _f5_conformal
        _f5_conformal = None
        d = Path(MLB_F5_MODELS_DIR)
        conformal = _load_f5_conformal(d, sportsbook=sportsbook)

        meta_cols = ["home_team", "away_team"]
        teams = features[meta_cols].copy() if all(c in features.columns for c in meta_cols) else None
        feature_matrix = features.drop(
            columns=[c for c in meta_cols if c in features.columns], errors="ignore"
        ).values.astype(float)

        # XGBoost predict
        dmat = xgb.DMatrix(feature_matrix)
        xgb_raw = xgb_model.predict(dmat)
        xgb_probs = np.column_stack([1.0 - xgb_raw, xgb_raw])
        if _f5_xgb_calibrator is not None:
            try:
                xgb_probs = _f5_xgb_calibrator.predict_proba(xgb_probs)
            except Exception:
                pass

        # CatBoost predict (optional)
        cat_probs = None
        if cat_model is not None:
            try:
                cat_probs = cat_model.predict_proba(feature_matrix)
                if _f5_cat_calibrator is not None:
                    cat_probs = _f5_cat_calibrator.predict_proba(cat_probs)
            except Exception as e:
                logger.debug("CatBoost F5 predict failed: %s", e)

        # Ensemble
        w_xgb = weights.get("xgb", F5_W_XGB)
        w_cat = weights.get("cat", F5_W_CAT)
        if cat_probs is not None:
            total_w = w_xgb + w_cat
            ml_probs = (w_xgb / total_w) * xgb_probs + (w_cat / total_w) * cat_probs
        else:
            ml_probs = xgb_probs

        # Conformal prediction sets
        conf_set_sizes = None
        conf_margins = None
        if conformal is not None:
            try:
                conf_set_sizes, conf_margins = conformal.predict_confidence(ml_probs)
            except Exception as e:
                logger.debug("F5 conformal failed: %s", e)

        for i in range(len(feature_matrix)):
            home = str(teams.iloc[i]["home_team"]) if teams is not None else f"Home{i}"
            away = str(teams.iloc[i]["away_team"]) if teams is not None else f"Away{i}"
            key = (home, away)

            f5_game = f5_odds_by_key.get(key, {})
            f5_ml_home = f5_game.get("f5_ml_home")
            f5_ml_away = f5_game.get("f5_ml_away")

            prob_home = float(ml_probs[i, 1])
            prob_away = float(ml_probs[i, 0])

            ev_home = ev_away = 0.0
            kelly_h = kelly_a = 0.0
            sigma = 0.08  # MLB F5 default sigma

            if f5_ml_home is not None and f5_ml_away is not None:
                h_odds = int(f5_ml_home)
                a_odds = int(f5_ml_away)
                ev_home = float(Expected_Value.expected_value(prob_home, h_odds))
                ev_away = float(Expected_Value.expected_value(prob_away, a_odds))
                kelly_h = float(calculate_robust_kelly_simple(h_odds, prob_home, epsilon=sigma))
                kelly_a = float(calculate_robust_kelly_simple(a_odds, prob_away, epsilon=sigma))

            max_ev = max(ev_home, ev_away)
            best_ev = ev_home if ev_home >= ev_away else ev_away
            best_kelly = kelly_h if ev_home >= ev_away else kelly_a

            conf_ss = int(conf_set_sizes[i]) if conf_set_sizes is not None else None
            conf_uncertain = conf_ss is not None and conf_ss != 1

            if best_ev > 0 and not conf_uncertain:
                f5_tag = "BET"
            elif best_ev > 0 and conf_uncertain and best_kelly >= 0.5:
                f5_tag = "BET"  # Kelly override
            elif best_ev > 0:
                f5_tag = "SKIP"
            else:
                f5_tag = "PASS"

            results.append({
                "home_team": home,
                "away_team": away,
                "f5_prob_home": prob_home,
                "f5_prob_away": prob_away,
                "f5_ev_home": ev_home,
                "f5_ev_away": ev_away,
                "f5_ev": best_ev,
                "f5_kelly": best_kelly,
                "f5_tag": f5_tag,
                "f5_conf_set_size": conf_ss,
                "f5_ml_home_odds": f5_ml_home,
                "f5_ml_away_odds": f5_ml_away,
            })

    else:
        # --- Path B: sin modelo, solo EV desde odds de F5 ---
        # Usa implied probability del libro como estimacion del modelo
        for key, f5_game in f5_odds_by_key.items():
            home, away = key
            f5_ml_home = f5_game.get("f5_ml_home")
            f5_ml_away = f5_game.get("f5_ml_away")

            if f5_ml_home is None or f5_ml_away is None:
                continue

            h_odds = int(f5_ml_home)
            a_odds = int(f5_ml_away)

            # Implied probability (con vig eliminado)
            def implied(odds: int) -> float:
                if odds > 0:
                    return 100.0 / (odds + 100.0)
                else:
                    return abs(odds) / (abs(odds) + 100.0)

            p_h_raw = implied(h_odds)
            p_a_raw = implied(a_odds)
            total_vig = p_h_raw + p_a_raw
            # Remover vig proporcional
            prob_home = p_h_raw / total_vig
            prob_away = p_a_raw / total_vig

            # Con solo odds, no hay edge — modelo no disponible
            # Reportamos la linea pero marcamos como N/A para EV
            logger.debug(
                "F5 odds-only mode for %s @ %s: model not available", away, home
            )

            results.append({
                "home_team": home,
                "away_team": away,
                "f5_prob_home": prob_home,
                "f5_prob_away": prob_away,
                "f5_ev_home": 0.0,
                "f5_ev_away": 0.0,
                "f5_ev": 0.0,
                "f5_kelly": 0.0,
                "f5_tag": "N/A",
                "f5_conf_set_size": None,
                "f5_ml_home_odds": h_odds,
                "f5_ml_away_odds": a_odds,
            })

    return results
