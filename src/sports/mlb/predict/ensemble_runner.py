"""MLB Ensemble Runner: combina predicciones de XGBoost + CatBoost.

Weighted average con pesos cargados desde metadata.json:
  Default: XGB 60% + CatBoost 40%

Por que estos pesos?
  - XGB+CatBoost cometen errores diferentes (depth-wise vs symmetric trees)
  - Pesos optimizados por calibracion (ECE) sobre test set MLB
  - Para apuestas, calibracion > accuracy (Kelly sizing depende de probabilidades exactas)

MLB es mas impredecible que NBA por la varianza inherente del beisbol:
  - 30% de la varianza en MLB es ruido puro (Pythagorean wins)
  - El lanzador abridor es el factor #1 — debe estar en features
  - Sigma tipico mas alto: 0.08-0.15 (vs 0.04-0.10 en NBA)
"""
import re
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from colorama import Fore, Style, init, deinit

from src.core.betting import expected_value as Expected_Value
from src.core.betting import kelly_criterion as kc
from src.core.betting.robust_kelly import calculate_robust_kelly_simple
from src.core.calibration.conformal import ConformalClassifier
from src.sports.mlb.config_mlb import MLB_SIGMA_BUCKETS
from src.sports.mlb.config_paths import MLB_ML_MODELS_DIR, MLB_F5_MODELS_DIR
from src.config import get_logger

logger = get_logger(__name__)

init()

# --- Module-level cache para Telegram formatter ---
_last_blocks = []

# --- Rango valido de sigma (DRO-Kelly) ---
SIGMA_MIN = 0.03
SIGMA_MAX = 0.20

# --- Pesos default (se sobreescriben con metadata.json) ---
W_XGB_ML = 0.60
W_CAT_ML = 0.40

# --- Patterns para seleccion de modelo por accuracy en filename ---
XGB_ACCURACY_PATTERN = re.compile(r"XGBoost_(\d+(?:\.\d+)?)%_")
CATBOOST_ACCURACY_PATTERN = re.compile(r"CatBoost_(\d+(?:\.\d+)?)%_")

# --- Modelos cacheados (se cargan una vez por sesion) ---
_xgb_ml = None
_xgb_calibrator = None
_catboost_ml = None
_catboost_calibrator = None
_ensemble_conformal = None
_ensemble_weights = None


# ---------------------------------------------------------------------------
# Model selection helpers
# ---------------------------------------------------------------------------

def _select_model_path(model_dir: Path, pattern: re.Pattern, kind: str = "ML",
                       ext: str = None) -> Path:
    """Selecciona el mejor modelo por accuracy + fecha de modificacion.

    Busca archivos que contengan `kind` en el nombre dentro de `model_dir`.
    Si hay varios candidatos, prioriza el de mayor accuracy (numero en filename).
    En empate, usa el mas reciente (mtime).
    """
    if ext:
        candidates = list(model_dir.glob(f"*{kind}*{ext}"))
    else:
        candidates = list(model_dir.glob(f"*{kind}*.json")) + list(model_dir.glob(f"*{kind}*.pkl"))

    # Excluir archivos de calibracion
    candidates = [c for c in candidates if "calibrat" not in c.name.lower() and "conformal" not in c.name.lower()]

    if not candidates:
        raise FileNotFoundError(
            f"No model file matching '*{kind}*' found in {model_dir}. "
            "Run scripts/train_models_mlb.py to train MLB models first."
        )

    def score(path: Path):
        match = pattern.search(path.name)
        accuracy = float(match.group(1)) if match else 0.0
        return (accuracy, path.stat().st_mtime)

    return max(candidates, key=score)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_xgb(model_dir: Path = None):
    """Carga el modelo XGBoost ML + calibrator opcional (lazy, una sola vez)."""
    global _xgb_ml, _xgb_calibrator
    if _xgb_ml is None:
        d = model_dir or MLB_ML_MODELS_DIR
        path = _select_model_path(d, XGB_ACCURACY_PATTERN, "ML", ext=".json")
        _xgb_ml = xgb.Booster()
        _xgb_ml.load_model(str(path))
        logger.info("XGBoost MLB ML loaded: %s", path.name)

        cal_path = path.with_name(f"{path.stem}_calibration.pkl")
        if cal_path.exists():
            _xgb_calibrator = joblib.load(cal_path)
            logger.info("XGBoost MLB calibrator loaded: %s", cal_path.name)
    return _xgb_ml, _xgb_calibrator


def _load_catboost(model_dir: Path = None):
    """Carga el modelo CatBoost ML + calibrator opcional (lazy, una sola vez)."""
    global _catboost_ml, _catboost_calibrator
    if _catboost_ml is None:
        d = model_dir or MLB_ML_MODELS_DIR
        path = _select_model_path(d, CATBOOST_ACCURACY_PATTERN, "ML", ext=".pkl")
        _catboost_ml = joblib.load(path)
        logger.info("CatBoost MLB ML loaded: %s", path.name)

        cal_path = path.with_name(f"{path.stem}_calibration.pkl")
        if cal_path.exists():
            _catboost_calibrator = joblib.load(cal_path)
            logger.info("CatBoost MLB calibrator loaded: %s", cal_path.name)
    return _catboost_ml, _catboost_calibrator


def _load_ensemble_weights(model_dir: Path = None) -> dict:
    """Carga pesos desde metadata.json si existe, sino usa defaults."""
    global _ensemble_weights
    if _ensemble_weights is None:
        d = model_dir or MLB_ML_MODELS_DIR
        metadata_path = d / "metadata.json"
        if metadata_path.exists():
            import json
            try:
                with open(metadata_path) as f:
                    meta = json.load(f)
                weights = meta.get("ensemble_weights", {})
                _ensemble_weights = {
                    "xgb": float(weights.get("xgb", W_XGB_ML)),
                    "cat": float(weights.get("cat", W_CAT_ML)),
                }
                logger.info(
                    "Ensemble weights from metadata.json: XGB=%.0f%% Cat=%.0f%%",
                    _ensemble_weights["xgb"] * 100, _ensemble_weights["cat"] * 100,
                )
            except Exception as e:
                logger.warning("Could not load metadata.json: %s — using defaults", e)
                _ensemble_weights = {"xgb": W_XGB_ML, "cat": W_CAT_ML}
        else:
            _ensemble_weights = {"xgb": W_XGB_ML, "cat": W_CAT_ML}
    return _ensemble_weights


def _load_conformal(model_dir: Path = None, sportsbook: str = None) -> Optional[ConformalClassifier]:
    """Carga conformal per-sportsbook si existe, sino generico."""
    global _ensemble_conformal
    if _ensemble_conformal is not None:
        return _ensemble_conformal

    d = model_dir or MLB_ML_MODELS_DIR
    candidates = []

    # Intentar per-book primero
    if sportsbook:
        book_path = d / f"ensemble_conformal_{sportsbook}.pkl"
        if book_path.exists():
            candidates.append(book_path)

    # Fallback generico
    generic_path = d / "ensemble_conformal.pkl"
    if generic_path.exists():
        candidates.append(generic_path)

    if candidates:
        try:
            _ensemble_conformal = joblib.load(candidates[0])
            logger.info("MLB conformal loaded: %s", candidates[0].name)
        except Exception as e:
            logger.warning("Could not load MLB conformal: %s", e)
    else:
        logger.debug("No MLB conformal predictor found in %s", d)

    return _ensemble_conformal


def load_mlb_models(model_dir: Path = None) -> tuple:
    """Carga XGBoost + CatBoost models + metadata + conformal.

    Returns:
        (xgb_model, cat_model, weights_dict, conformal_or_None)
    """
    d = model_dir or MLB_ML_MODELS_DIR
    d = Path(d)

    if not d.exists():
        raise FileNotFoundError(
            f"MLB moneyline model directory not found: {d}\n"
            "Run: PYTHONPATH=. python scripts/train_models_mlb.py"
        )

    xgb_model, _xgb_cal = _load_xgb(d)
    cat_model, _cat_cal = _load_catboost(d)
    weights = _load_ensemble_weights(d)
    conformal = _load_conformal(d)

    return xgb_model, cat_model, weights, conformal


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _xgb_predict_proba(model, features: np.ndarray, calibrator=None) -> np.ndarray:
    """Retorna probabilidades (N, 2) desde XGBoost Booster."""
    dmat = xgb.DMatrix(features)
    raw = model.predict(dmat)
    # XGBoost binary classifier retorna P(class=1)
    probs = np.column_stack([1.0 - raw, raw])
    if calibrator is not None:
        try:
            probs = calibrator.predict_proba(probs)
        except Exception as e:
            logger.debug("XGBoost calibrator failed: %s", e)
    return probs


def _catboost_predict_proba(model, features: np.ndarray, calibrator=None) -> np.ndarray:
    """Retorna probabilidades (N, 2) desde CatBoost."""
    try:
        probs = model.predict_proba(features)
    except Exception:
        # Algunos wrappers de CatBoost exponen predict_proba directamente
        probs = np.column_stack([
            1.0 - model.predict(features),
            model.predict(features),
        ])
    if calibrator is not None:
        try:
            probs = calibrator.predict_proba(probs)
        except Exception as e:
            logger.debug("CatBoost calibrator failed: %s", e)
    return probs


def _compute_sigma(xgb_probs: np.ndarray, cat_probs: np.ndarray,
                   buckets: list = None) -> np.ndarray:
    """Calcula sigma per-game como disagreement entre XGB y CatBoost.

    Mapea el disagreement (|p_xgb - p_cat|) a un epsilon de DRO-Kelly
    usando MLB_SIGMA_BUCKETS.

    Returns:
        Array de shape (N,) con sigma per-game, clipeado en [SIGMA_MIN, SIGMA_MAX].
    """
    if buckets is None:
        buckets = MLB_SIGMA_BUCKETS

    disagreement = np.abs(xgb_probs[:, 1] - cat_probs[:, 1])
    sigmas = np.zeros(len(disagreement))

    for i, d in enumerate(disagreement):
        for threshold, sigma_val in buckets:
            if d <= threshold:
                sigmas[i] = sigma_val
                break

    return np.clip(sigmas, SIGMA_MIN, SIGMA_MAX)


# ---------------------------------------------------------------------------
# Public: predict_ensemble
# ---------------------------------------------------------------------------

def predict_ensemble(features: pd.DataFrame, models: tuple,
                     weights: dict, conformal=None) -> list[dict]:
    """Ejecuta prediccion ensemble XGB + CatBoost para MLB moneyline.

    Args:
        features:  DataFrame con features para cada juego (una fila por juego).
                   El orden de columnas debe coincidir con el set de entrenamiento.
        models:    Tuple (xgb_model, cat_model) desde load_mlb_models().
        weights:   Dict {"xgb": 0.60, "cat": 0.40} desde _load_ensemble_weights().
        conformal: ConformalClassifier opcional.

    Returns:
        Lista de dicts, uno por fila de features:
          {home_team, away_team, prob_home, prob_away,
           xgb_prob, cat_prob, sigma, conf_set_size, conf_margin}

        home_team / away_team son None si no estan en features — el caller
        los debe inyectar desde odds_data.
    """
    xgb_model, cat_model = models
    xgb_cal = _xgb_calibrator
    cat_cal = _catboost_calibrator

    X = features.values.astype(float)

    xgb_probs = _xgb_predict_proba(xgb_model, X, xgb_cal)
    cat_probs = _catboost_predict_proba(cat_model, X, cat_cal)

    w_xgb = weights.get("xgb", W_XGB_ML)
    w_cat = weights.get("cat", W_CAT_ML)
    total_w = w_xgb + w_cat
    ml_probs = (w_xgb / total_w) * xgb_probs + (w_cat / total_w) * cat_probs

    sigmas = _compute_sigma(xgb_probs, cat_probs)

    # Conformal prediction sets
    conf_set_sizes = None
    conf_margins = None
    if conformal is not None:
        try:
            conf_set_sizes, conf_margins = conformal.predict_confidence(ml_probs)
        except Exception as e:
            logger.warning("Conformal prediction failed: %s", e)

    results = []
    for i in range(len(X)):
        entry = {
            "home_team": None,
            "away_team": None,
            "prob_home": float(ml_probs[i, 1]),
            "prob_away": float(ml_probs[i, 0]),
            "xgb_prob": float(xgb_probs[i, 1]),
            "cat_prob": float(cat_probs[i, 1]),
            "sigma": float(sigmas[i]),
            "conf_set_size": int(conf_set_sizes[i]) if conf_set_sizes is not None else None,
            "conf_margin": float(conf_margins[i]) if conf_margins is not None else None,
        }
        results.append(entry)

    return results


# ---------------------------------------------------------------------------
# Block builder
# ---------------------------------------------------------------------------

def _build_game_blocks(predictions: list[dict], odds_data: list[dict],
                       sportsbook: str) -> list[dict]:
    """Construye bloques de output por partido con toda la info consolidada.

    Combina predicciones del ensemble con odds_data (proveniente de MLBOddsProvider).
    Importa resultados de F5 y totals si estan disponibles.

    Returns:
        Lista de dicts con datos pre-calculados, rankeados por max_ev descendente.
    """
    # Intentar importar resultados de F5 y totals runners
    f5_results = {}
    totals_results = {}

    # Construir lookup por (home_team, away_team) para acceso rapido
    odds_by_matchup = {}
    for game in (odds_data or []):
        key = (game.get("home_team", ""), game.get("away_team", ""))
        odds_by_matchup[key] = game

    blocks = []
    for pred in predictions:
        home = pred.get("home_team") or ""
        away = pred.get("away_team") or ""
        key = (home, away)

        game_odds = odds_by_matchup.get(key, {})

        # --- ML odds ---
        ml_home = game_odds.get("ml_home")
        ml_away = game_odds.get("ml_away")

        prob_home = pred["prob_home"]
        prob_away = pred["prob_away"]
        xgb_prob = pred["xgb_prob"]
        cat_prob = pred["cat_prob"]
        sigma = pred["sigma"]

        ev_home = ev_away = 0.0
        kelly_h = kelly_a = 0.0

        if ml_home is not None and ml_away is not None:
            h_odds = int(ml_home)
            a_odds = int(ml_away)
            ev_home = float(Expected_Value.expected_value(prob_home, h_odds))
            ev_away = float(Expected_Value.expected_value(prob_away, a_odds))
            kelly_h = float(calculate_robust_kelly_simple(h_odds, prob_home, epsilon=sigma))
            kelly_a = float(calculate_robust_kelly_simple(a_odds, prob_away, epsilon=sigma))

        max_ev = max(ev_home, ev_away)

        # --- Pick ---
        winner = 1 if prob_home >= prob_away else 0
        pick = home if winner == 1 else away
        pick_prob = prob_home if winner == 1 else prob_away
        xgb_agree = (xgb_prob >= 0.5) == (cat_prob >= 0.5)
        xgb_conf = round(xgb_prob * 100 if winner == 1 else (1 - xgb_prob) * 100, 1)
        cat_conf = round(cat_prob * 100 if winner == 1 else (1 - cat_prob) * 100, 1)

        # --- Underdog trap ---
        trap_home = prob_home < 0.35 and ev_home > 0
        trap_away = prob_away < 0.35 and ev_away > 0

        # --- Run line (like NBA AH spread) ---
        run_line = game_odds.get("run_line_home", -1.5)  # MLB default: -1.5 favorite
        rl_home_odds = game_odds.get("run_line_home_odds", -110)
        rl_away_odds = game_odds.get("run_line_away_odds", -110)

        # P(favorite covers -1.5): use normal approximation on run differential
        # MLB avg runs = 4.5 per game per team; std of margin ~3.5 runs
        MLB_MARGIN_STD = 3.5
        from scipy import stats as scipy_stats
        if run_line is not None:
            # P(home margin > |run_line|) for home -1.5
            expected_margin = (prob_home - 0.5) * 7.0  # rough linear mapping
            p_rl_home = float(scipy_stats.norm.sf(abs(float(run_line)) - 0.5, loc=expected_margin, scale=MLB_MARGIN_STD))
            p_rl_away = float(scipy_stats.norm.sf(abs(float(run_line)) - 0.5, loc=-expected_margin, scale=MLB_MARGIN_STD))
            rl_ev_home = float(Expected_Value.expected_value(p_rl_home, int(rl_home_odds) if rl_home_odds else -110))
            rl_ev_away = float(Expected_Value.expected_value(p_rl_away, int(rl_away_odds) if rl_away_odds else -110))
            rl_kelly_home = float(calculate_robust_kelly_simple(
                int(rl_home_odds) if rl_home_odds else -110, p_rl_home, epsilon=sigma))
            rl_kelly_away = float(calculate_robust_kelly_simple(
                int(rl_away_odds) if rl_away_odds else -110, p_rl_away, epsilon=sigma))
            if rl_ev_home > rl_ev_away and rl_ev_home > 0:
                ah_side = home
                ah_ev = rl_ev_home
                ah_p = p_rl_home
                ah_kelly = rl_kelly_home
                ah_line = f"{float(run_line):+.1f}"
            elif rl_ev_away > 0:
                ah_side = away
                ah_ev = rl_ev_away
                ah_p = p_rl_away
                ah_kelly = rl_kelly_away
                ah_line = f"{-float(run_line):+.1f}"
            else:
                ah_side = None
                ah_ev = max(rl_ev_home, rl_ev_away)
                ah_p = max(p_rl_home, p_rl_away)
                ah_kelly = 0.0
                ah_line = f"{float(run_line):+.1f}"
        else:
            ah_side = None
            ah_ev = 0.0
            ah_p = 0.5
            ah_kelly = 0.0
            ah_line = "N/A"

        # --- Totals (O/U) ---
        ou_line = game_odds.get("total")
        ou_over_odds = game_odds.get("over_odds", -110)
        ou_under_odds = game_odds.get("under_odds", -110)
        totals_key = key
        totals_pred = totals_results.get(totals_key, {})
        ou_label = totals_pred.get("ou_label", "---")
        ou_prob = totals_pred.get("ou_prob", 0.5)
        ou_ev = totals_pred.get("ou_ev", 0.0)
        ou_kelly = totals_pred.get("ou_kelly", 0.0)
        predicted_total = totals_pred.get("predicted_total")

        # --- F5 ---
        f5_key = key
        f5_pred = f5_results.get(f5_key, {})
        f5_prob = f5_pred.get("f5_prob_home", None)
        f5_ev = f5_pred.get("f5_ev", 0.0)
        f5_kelly = f5_pred.get("f5_kelly", 0.0)
        f5_tag = f5_pred.get("f5_tag", None)
        f5_conf_ss = f5_pred.get("f5_conf_set_size", None)

        # --- Game info ---
        commence_time = game_odds.get("commence_time", "")
        sp_home = game_odds.get("home_pitcher", "TBD")
        sp_away = game_odds.get("away_pitcher", "TBD")
        park_factor = game_odds.get("park_factor", 100)
        temp = game_odds.get("temp_f")
        wind_desc = game_odds.get("wind_desc", "")

        # --- Conformal ---
        conf_ss = pred.get("conf_set_size")
        conf_margin = pred.get("conf_margin", 0.0)

        # --- Tag ---
        has_trap = trap_home or trap_away
        max_kelly = max(kelly_h, kelly_a)
        conf_uncertain = conf_ss is not None and conf_ss != 1
        conf_override = conf_uncertain and max_kelly >= 0.5

        if has_trap:
            tag = "TRAP"
        elif conf_uncertain and not conf_override:
            tag = "SKIP"
        elif max_ev > 0:
            tag = "BET"
        else:
            tag = "PASS"

        b = {
            "home": home,
            "away": away,
            "pick": pick,
            "pick_prob": pick_prob,
            "winner": winner,
            "prob_home": prob_home,
            "prob_away": prob_away,
            "xgb_conf": xgb_conf,
            "cat_conf": cat_conf,
            "xgb_agree": xgb_agree,
            "sigma": sigma,
            # ML
            "ev_home": ev_home,
            "ev_away": ev_away,
            "kelly_home": kelly_h,
            "kelly_away": kelly_a,
            "ml_home_odds": ml_home,
            "ml_away_odds": ml_away,
            # Run line (MLB equivalent of AH spread)
            "ah_side": ah_side,
            "ah_line": ah_line,
            "ah_p": ah_p,
            "ah_ev": ah_ev,
            "ah_kelly": ah_kelly,
            # Totals
            "ou_line": ou_line,
            "ou_label": ou_label,
            "ou_prob": ou_prob,
            "ou_ev": ou_ev,
            "ou_kelly": ou_kelly,
            "predicted_total": predicted_total,
            # F5
            "f5_prob": f5_prob,
            "f5_ev": f5_ev,
            "f5_kelly": f5_kelly,
            "f5_tag": f5_tag,
            "f5_conf_ss": f5_conf_ss,
            # Game info
            "commence_time": commence_time,
            "sp_home": sp_home,
            "sp_away": sp_away,
            "park_factor": park_factor,
            "temp": temp,
            "wind_desc": wind_desc,
            # Conformal
            "conf_ss": conf_ss,
            "conf_margin": conf_margin if conf_margin is not None else 0.0,
            # Summary
            "max_ev": max_ev,
            "tag": tag,
            "trap_home": trap_home,
            "trap_away": trap_away,
        }
        blocks.append(b)

    # Rankear por max_ev descendente
    blocks.sort(key=lambda x: x["max_ev"], reverse=True)
    return blocks


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def _print_compact_output(blocks: list[dict], sportsbook: str):
    """Imprime output MLB compacto, un bloque por partido, rankeado por EV.

    Formato ejemplo:
      #1  Boston Red Sox (54.2%) vs New York Yankees  [BET]
           ML:  Boston -115 / NYY +105  EV=+4.2%  Kelly=1.1%
           F5:  Boston 56.1%  F5 -108/-108  EV=+6.8%  Kelly=1.4%  [BET]
           O/U: UNDER 8.5  P(under)=58%  EV=+3.1%  Kelly=0.8%  pred=8.1
           RL:  Boston (+1.5) P=62%  EV=+2.8%  Kelly=0.7%
           Park: Fenway PF=104  62F  Wind 8mph OUT
           Conf: set=1 margin=0.09  sigma=0.08  XGB:55% Cat:53%
    """
    CONF2_KELLY_THRESHOLD = 0.5  # % bankroll — permite BET con conformal=2 si Kelly alto

    n_total = len(blocks)
    if n_total == 0:
        print(f"\n{Fore.YELLOW}No MLB games found for today.{Style.RESET_ALL}\n")
        return

    n_bet = sum(
        1 for b in blocks
        if b["conf_ss"] is None
        or b["conf_ss"] == 1
        or (b["conf_ss"] == 2 and max(b["kelly_home"], b["kelly_away"]) >= CONF2_KELLY_THRESHOLD)
    )

    conf_label = f" | Conformal {n_bet}/{n_total}"
    print(f"\n{Fore.CYAN}{'='*62}")
    print(f"  MLB PICKS ranked by EV | {sportsbook.upper()}{conf_label}")
    print(f"{'='*62}{Style.RESET_ALL}\n")

    for rank, b in enumerate(blocks, 1):
        max_kelly = max(b["kelly_home"], b["kelly_away"])
        conf_uncertain = b["conf_ss"] is not None and b["conf_ss"] != 1
        conf_override = conf_uncertain and max_kelly >= CONF2_KELLY_THRESHOLD

        # --- Tag color ---
        tag_str = b["tag"]
        if tag_str == "BET":
            tag_colored = f"{Fore.GREEN}BET{Style.RESET_ALL}"
        elif tag_str == "TRAP":
            tag_colored = f"{Fore.YELLOW}TRAP{Style.RESET_ALL}"
        elif tag_str == "SKIP":
            tag_colored = f"{Fore.YELLOW}SKIP{Style.RESET_ALL}"
        else:
            tag_colored = f"{Fore.RED}PASS{Style.RESET_ALL}"

        # --- Sigma color ---
        s = b["sigma"]
        sigma_color = Fore.GREEN if s < 0.07 else (Fore.YELLOW if s < 0.12 else Fore.RED)

        # --- Agreement ---
        agree = "+" if b["xgb_agree"] else "~"

        # --- Pitcher matchup ---
        pitcher_str = ""
        if b["sp_away"] and b["sp_away"] != "TBD" and b["sp_home"] and b["sp_home"] != "TBD":
            pitcher_str = f"  {b['sp_away']} vs {b['sp_home']}"
        elif b["sp_home"] and b["sp_home"] != "TBD":
            pitcher_str = f"  TBD vs {b['sp_home']}"

        # --- Time ---
        time_str = ""
        if b["commence_time"]:
            try:
                from datetime import datetime, timezone, timedelta
                et = timezone(timedelta(hours=-5))
                dt = datetime.fromisoformat(str(b["commence_time"]).replace("Z", "+00:00")).astimezone(et)
                time_str = f"  {dt.strftime('%H:%M ET')}"
            except Exception:
                time_str = ""

        # --- Line 1: header ---
        pick_color = Fore.GREEN if b["winner"] == 1 else Fore.RED
        loser = b["away"] if b["winner"] == 1 else b["home"]
        loser_color = Fore.RED if b["winner"] == 1 else Fore.GREEN
        print(
            f"  {Fore.CYAN}#{rank}{Style.RESET_ALL}  "
            f"{pick_color}{b['pick']}{Style.RESET_ALL} "
            f"({b['pick_prob']*100:.1f}%) vs "
            f"{loser_color}{loser}{Style.RESET_ALL}"
            f"{time_str}{pitcher_str}  [{tag_colored}]"
        )

        # --- Line 2: ML EV + Kelly ---
        ev_side = "home" if b["ev_home"] >= b["ev_away"] else "away"
        ev_val = b[f"ev_{ev_side}"]
        kelly_val = b[f"kelly_{ev_side}"]
        ev_team = b["home"] if ev_side == "home" else b["away"]
        ev_color = Fore.GREEN if ev_val > 0 else Fore.RED
        ml_h_str = f"{b['ml_home_odds']:+d}" if b["ml_home_odds"] else "N/A"
        ml_a_str = f"{b['ml_away_odds']:+d}" if b["ml_away_odds"] else "N/A"
        print(
            f"       ML:  {ev_team}: {b['home']} {ml_h_str} / {b['away']} {ml_a_str}  "
            f"EV={ev_color}{ev_val:+.1f}{Style.RESET_ALL}  Kelly={kelly_val:.2f}%  "
            f"[{agree} XGB:{b['xgb_conf']}% Cat:{b['cat_conf']}%]  "
            f"{sigma_color}sigma={s:.3f}{Style.RESET_ALL}"
        )

        # --- Line 3: F5 ---
        if b["f5_ev"] and abs(b["f5_ev"]) > 0:
            f5_ev_color = Fore.GREEN if b["f5_ev"] > 0 else Fore.RED
            f5_tag_str = f"  [{Fore.GREEN}BET{Style.RESET_ALL}]" if b["f5_tag"] == "BET" else ""
            f5_prob_str = f"{b['f5_prob']*100:.1f}%" if b["f5_prob"] is not None else "N/A"
            print(
                f"       F5:  {f5_prob_str}  "
                f"EV={f5_ev_color}{b['f5_ev']:+.1f}{Style.RESET_ALL}  "
                f"Kelly={b['f5_kelly']:.2f}%{f5_tag_str}"
            )

        # --- Line 4: Run line ---
        if b["ah_side"]:
            rl_ev_color = Fore.GREEN if b["ah_ev"] > 0 else Fore.RED
            print(
                f"       RL:  {b['ah_side']} ({b['ah_line']})  "
                f"P={b['ah_p']:.1%}  "
                f"EV={rl_ev_color}{b['ah_ev']:+.1f}{Style.RESET_ALL}  "
                f"Kelly={b['ah_kelly']:.2f}%"
            )
        else:
            print(f"       RL:  --- run line {b['ah_line']}  EV={b['ah_ev']:+.1f}")

        # --- Line 5: O/U ---
        if b["ou_label"] and b["ou_label"] != "---" and b["ou_line"]:
            ou_color = Fore.MAGENTA if b["ou_label"] == "UNDER" else Fore.BLUE
            pred_str = f"  pred={b['predicted_total']:.1f}" if b["predicted_total"] is not None else ""
            ou_ev_color = Fore.GREEN if b["ou_ev"] > 0 else Fore.RED
            print(
                f"       O/U: {ou_color}{b['ou_label']}{Style.RESET_ALL} "
                f"{b['ou_line']}  P={b['ou_prob']:.1%}  "
                f"EV={ou_ev_color}{b['ou_ev']:+.1f}{Style.RESET_ALL}  "
                f"Kelly={b['ou_kelly']:.2f}%{pred_str}"
            )

        # --- Line 6: Park + weather ---
        park_str = ""
        if b["park_factor"]:
            park_str = f"PF={b['park_factor']}"
        temp_str = f"  {b['temp']}F" if b["temp"] else ""
        wind_str = f"  {b['wind_desc']}" if b["wind_desc"] else ""
        if park_str or temp_str or wind_str:
            print(f"       Park: {park_str}{temp_str}{wind_str}")

        # --- Line 7: Conformal ---
        if b["conf_ss"] is not None:
            if b["conf_ss"] == 1:
                print(f"       {Fore.GREEN}Conf: set=1 margin={b['conf_margin']:.2f}{Style.RESET_ALL}")
            elif conf_override:
                print(
                    f"       {Fore.CYAN}Conf: set=2 pero Kelly={max_kelly:.2f}% "
                    f">= {CONF2_KELLY_THRESHOLD}% -> BET override{Style.RESET_ALL}"
                )
            else:
                print(f"       {Fore.YELLOW}Conf: skip (set_size={b['conf_ss']}){Style.RESET_ALL}")

        # --- Trap warning ---
        if b["trap_home"] or b["trap_away"]:
            trap_team = b["away"] if b["trap_away"] else b["home"]
            print(
                f"       {Fore.YELLOW}! underdog {trap_team} EV+ es trampa -> usar RL del favorito{Style.RESET_ALL}"
            )

        print()  # blank line between games


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run_mlb_ensemble(games_df: pd.DataFrame, odds_data: list[dict],
                     sportsbook: str = "fanduel") -> list[dict]:
    """Punto de entrada principal para predicciones MLB pregame.

    Carga modelos, ejecuta ensemble, construye bloques de output,
    imprime resultado compacto en consola, y cachea _last_blocks para Telegram.

    Args:
        games_df:   DataFrame con features para los juegos de hoy.
                    Cada fila corresponde a un juego.
                    Debe tener columnas 'home_team' y 'away_team' (se extraen y dropean antes de predict).
        odds_data:  Lista de dicts con odds del dia (de MLBOddsProvider.get_odds()).
        sportsbook: Nombre del sportsbook para conformal per-book.

    Returns:
        Lista de dicts con predicciones enriquecidas (una por juego).
    """
    global _last_blocks

    if games_df is None or len(games_df) == 0:
        logger.warning("No MLB games to predict.")
        return []

    # --- Cargar modelos ---
    try:
        xgb_model, cat_model, weights, conformal = load_mlb_models()
    except FileNotFoundError as e:
        logger.error("MLB models not found: %s", e)
        print(f"\n{Fore.RED}[MLB] Modelos no encontrados: {e}{Style.RESET_ALL}")
        print("  Entrena primero: PYTHONPATH=. python scripts/train_models_mlb.py\n")
        return []

    # --- Recargar conformal per-sportsbook ---
    global _ensemble_conformal
    _ensemble_conformal = None  # forzar recarga con sportsbook correcto
    conformal = _load_conformal(sportsbook=sportsbook)

    # --- Extraer home/away antes de predecir ---
    meta_cols = ["home_team", "away_team"]
    teams = games_df[meta_cols].copy() if all(c in games_df.columns for c in meta_cols) else None

    feature_df = games_df.drop(columns=[c for c in meta_cols if c in games_df.columns], errors="ignore")

    # --- Ejecutar ensemble ---
    predictions = predict_ensemble(feature_df, (xgb_model, cat_model), weights, conformal)

    # --- Inyectar nombres de equipo ---
    if teams is not None:
        for i, pred in enumerate(predictions):
            if i < len(teams):
                pred["home_team"] = str(teams.iloc[i]["home_team"])
                pred["away_team"] = str(teams.iloc[i]["away_team"])
    else:
        # Intentar obtener equipos de odds_data en orden
        for i, (pred, game_odds) in enumerate(zip(predictions, odds_data or [])):
            pred["home_team"] = game_odds.get("home_team", f"Home{i}")
            pred["away_team"] = game_odds.get("away_team", f"Away{i}")

    # --- Integrar resultados de F5 y totals si estan disponibles ---
    _integrate_f5_totals(predictions, odds_data)

    # --- Construir bloques ---
    blocks = _build_game_blocks(predictions, odds_data, sportsbook)
    _print_compact_output(blocks, sportsbook)

    _last_blocks = blocks

    deinit()

    # --- Construir resultado final ---
    results = []
    for pred in predictions:
        home = pred.get("home_team", "")
        away = pred.get("away_team", "")

        # Buscar block correspondiente
        block = next((b for b in blocks if b["home"] == home and b["away"] == away), {})

        result = {
            "home_team": home,
            "away_team": away,
            "prob_home": pred["prob_home"],
            "prob_away": pred["prob_away"],
            "xgb_prob": pred["xgb_prob"],
            "cat_prob": pred["cat_prob"],
            "sigma": pred["sigma"],
            "ev_home": block.get("ev_home", 0.0),
            "ev_away": block.get("ev_away", 0.0),
            "kelly_home": block.get("kelly_home", 0.0),
            "kelly_away": block.get("kelly_away", 0.0),
            "ml_home_odds": block.get("ml_home_odds"),
            "ml_away_odds": block.get("ml_away_odds"),
            "f5_ev": block.get("f5_ev", 0.0),
            "f5_kelly": block.get("f5_kelly", 0.0),
            "ou_label": block.get("ou_label", "---"),
            "ou_line": block.get("ou_line"),
            "ou_ev": block.get("ou_ev", 0.0),
            "conformal_set_size": pred.get("conf_set_size"),
            "conformal_margin": pred.get("conf_margin"),
            "tag": block.get("tag", "PASS"),
        }
        results.append(result)

    return results


def _integrate_f5_totals(predictions: list[dict], odds_data: list[dict]):
    """Intenta correr F5 runner y totals runner e inyecta resultados en predictions.

    Falla silenciosamente si los modelos no existen — no bloquea el ensemble principal.
    """
    from src.sports.mlb.predict.f5_runner import predict_f5, load_f5_models
    from src.sports.mlb.predict.totals_runner import predict_totals, load_totals_model

    # --- F5 ---
    try:
        f5_models = load_f5_models()
        if f5_models is not None:
            f5_preds = predict_f5(None, odds_data)  # features=None -> usa solo odds
            f5_by_key = {(p["home_team"], p["away_team"]): p for p in (f5_preds or [])}
            for pred in predictions:
                k = (pred.get("home_team", ""), pred.get("away_team", ""))
                if k in f5_by_key:
                    f5 = f5_by_key[k]
                    pred["f5_prob_home"] = f5.get("f5_prob_home")
                    pred["f5_ev"] = f5.get("f5_ev", 0.0)
                    pred["f5_kelly"] = f5.get("f5_kelly", 0.0)
                    pred["f5_tag"] = f5.get("f5_tag")
                    pred["f5_conf_set_size"] = f5.get("f5_conf_set_size")
    except Exception as e:
        logger.debug("F5 integration skipped: %s", e)

    # --- Totals ---
    try:
        totals_model = load_totals_model()
        if totals_model is not None:
            totals_preds = predict_totals(None, odds_data)  # features=None -> usa odds line
            totals_by_key = {(p["home_team"], p["away_team"]): p for p in (totals_preds or [])}
            for pred in predictions:
                k = (pred.get("home_team", ""), pred.get("away_team", ""))
                if k in totals_by_key:
                    t = totals_by_key[k]
                    pred["ou_label"] = t.get("ou_label", "---")
                    pred["ou_prob"] = t.get("ou_prob", 0.5)
                    pred["ou_ev"] = t.get("ou_ev", 0.0)
                    pred["ou_kelly"] = t.get("ou_kelly", 0.0)
                    pred["predicted_total"] = t.get("predicted_total")
    except Exception as e:
        logger.debug("Totals integration skipped: %s", e)
