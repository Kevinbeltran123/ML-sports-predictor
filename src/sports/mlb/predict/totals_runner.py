"""MLB Totals (O/U) runner — prediccion de carreras totales.

El modelo de totals es una regresion XGBoost que predice el total de carreras
esperado para el juego (home_runs + away_runs).

Pipeline:
  1. Predecir total esperado con XGBoost regressor
  2. Comparar con la linea del libro (ou_line)
  3. Calcular P(over/under) usando distribucion normal con std=MLB_RUNS_STD
  4. Calcular EV y Kelly para over/under

Por que normal distribution?
  - Distribuciones de carreras en MLB se aproximan bien con N(mu, sigma)
  - Tipico: mu = 8.5 carreras, sigma = 2.1 (varianza real observada en MLB 2019-2024)
  - Permite calcular P(over X) = P(Z > (X - mu) / sigma) analiticamente

Mercado en Odds API: 'totals' (over/under carreras totales)
"""
import re
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.core.betting import expected_value as Expected_Value
from src.core.betting.robust_kelly import calculate_robust_kelly_simple
from src.sports.mlb.config_paths import MLB_TOTALS_MODELS_DIR
from src.config import get_logger

logger = get_logger(__name__)

# --- Standard deviation de carreras totales en MLB (historico 2019-2024) ---
# Distribucion de runlines totales en MLB: std ~2.1 carreras por partido
MLB_RUNS_STD = 2.1

# --- Sigma DRO-Kelly para totals (menos calibrado que ML, usar mas conservador) ---
TOTALS_SIGMA = 0.10

# --- Cache de modelo ---
_totals_model = None
_totals_calibrator = None

# --- Accuracy pattern ---
XGB_TOTALS_PATTERN = re.compile(r"XGBoost_(\d+(?:\.\d+)?)%_")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_totals_model(model_dir: Path = None):
    """Carga el modelo XGBoost de regresion de totals desde models/mlb/totals/.

    Returns:
        El modelo cargado (XGBoost Booster o scikit-learn), o None si no existe.

    Falla silenciosamente — totals es un mercado secundario.
    """
    global _totals_model, _totals_calibrator

    if _totals_model is not None:
        return _totals_model

    d = Path(model_dir or MLB_TOTALS_MODELS_DIR)

    if not d.exists():
        logger.debug("Totals model directory not found: %s", d)
        return None

    # Buscar modelo JSON (XGBoost nativo) o PKL (sklearn wrapper)
    candidates_json = list(d.glob("*Totals*Reg*.json")) + list(d.glob("*XGBoost*Reg*.json"))
    candidates_pkl = list(d.glob("*Totals*.pkl")) + list(d.glob("*XGBoost*Reg*.pkl"))
    # Excluir calibracion y conformal
    candidates = [
        c for c in candidates_json + candidates_pkl
        if "calibrat" not in c.name.lower() and "conformal" not in c.name.lower()
    ]

    if not candidates:
        logger.debug("No totals regression model found in %s", d)
        return None

    def score(p: Path):
        match = re.search(r"(\d+(?:\.\d+)?)(?:%|_rmse)", p.name)
        acc = float(match.group(1)) if match else 0.0
        return (acc, p.stat().st_mtime)

    best = max(candidates, key=score)

    try:
        if best.suffix == ".json":
            import xgboost as xgb
            _totals_model = xgb.Booster()
            _totals_model.load_model(str(best))
            logger.info("Totals XGBoost regressor loaded: %s", best.name)
        else:
            _totals_model = joblib.load(best)
            logger.info("Totals model loaded: %s", best.name)

        # Calibrador opcional (solo para clasificadores, no regresores)
        cal_path = best.with_name(f"{best.stem}_calibration.pkl")
        if cal_path.exists():
            _totals_calibrator = joblib.load(cal_path)

    except Exception as e:
        logger.warning("Could not load totals model: %s", e)
        return None

    return _totals_model


# ---------------------------------------------------------------------------
# P(over) computation
# ---------------------------------------------------------------------------

def p_over(predicted_totals: np.ndarray, ou_lines: np.ndarray,
           std: float = MLB_RUNS_STD) -> np.ndarray:
    """Calcula P(total > ou_line) usando distribucion normal.

    P(over) = P(X > line) = 1 - CDF(line; mu=predicted, sigma=std)
            = 1 - Phi((line - predicted) / std)

    Args:
        predicted_totals: Array de totales predichos por el modelo (N,).
        ou_lines:         Array de lineas O/U del libro (N,).
        std:              Desviacion estandar de carreras totales (default 2.1).

    Returns:
        Array (N,) con P(over) para cada juego.
    """
    from scipy import stats as scipy_stats
    z = (ou_lines - predicted_totals) / std
    p_over_arr = scipy_stats.norm.sf(0, loc=predicted_totals - ou_lines, scale=std)
    # Equivalente: P(X > ou_line) = P(X - ou_line > 0) = 1 - CDF(ou_line; mu, std)
    return np.clip(p_over_arr, 0.01, 0.99)


# ---------------------------------------------------------------------------
# Public: predict_totals
# ---------------------------------------------------------------------------

def predict_totals(features: Optional[pd.DataFrame],
                   odds_data: list[dict] = None,
                   std: float = MLB_RUNS_STD) -> list[dict]:
    """Predice el total de carreras y calcula EV/Kelly para over/under.

    Si features es None o el modelo no existe, usa la linea del libro
    y retorna predicciones basadas en la probabilidad implicita del mercado
    (EV~=0, Kelly=0) — util para mostrar la linea sin edge propio.

    Args:
        features:   DataFrame con features por juego (puede ser None).
                    Si se provee, debe tener columnas 'home_team' y 'away_team'.
        odds_data:  Lista de dicts con 'total', 'over_odds', 'under_odds'.
        std:        Desviacion estandar de carreras (default 2.1).

    Returns:
        Lista de dicts, uno por juego:
          {home_team, away_team, predicted_total, ou_line,
           ou_label, ou_prob, ou_ev, ou_kelly,
           ou_over_prob, ou_over_ev, ou_over_kelly,
           ou_under_prob, ou_under_ev, ou_under_kelly}
    """
    from scipy import stats as scipy_stats

    # Construir lookup de odds por (home, away)
    odds_by_key = {}
    for game in (odds_data or []):
        k = (game.get("home_team", ""), game.get("away_team", ""))
        if game.get("total") is not None:
            odds_by_key[k] = game

    if not odds_by_key:
        logger.debug("No totals odds available in odds_data")
        return []

    # Intentar cargar el modelo
    model = load_totals_model()

    results = []

    # --- Path A: modelo disponible + features ---
    use_model = (
        model is not None
        and features is not None
        and len(features) > 0
    )

    if use_model:
        import xgboost as xgb

        meta_cols = ["home_team", "away_team"]
        teams_df = features[meta_cols].copy() if all(c in features.columns for c in meta_cols) else None
        feature_matrix = features.drop(
            columns=[c for c in meta_cols if c in features.columns], errors="ignore"
        ).values.astype(float)

        # Prediccion del regressor
        if hasattr(model, "predict"):
            if isinstance(model, xgb.Booster):
                dmat = xgb.DMatrix(feature_matrix)
                predicted_totals = model.predict(dmat)
            else:
                predicted_totals = model.predict(feature_matrix)
        else:
            logger.warning("Totals model has no predict() method, falling back to odds-only")
            use_model = False

    if use_model:
        for i in range(len(feature_matrix)):
            home = str(teams_df.iloc[i]["home_team"]) if teams_df is not None else f"Home{i}"
            away = str(teams_df.iloc[i]["away_team"]) if teams_df is not None else f"Away{i}"
            key = (home, away)

            game_odds = odds_by_key.get(key, {})
            ou_line = game_odds.get("total")
            over_odds = game_odds.get("over_odds", -110)
            under_odds = game_odds.get("under_odds", -110)

            if ou_line is None:
                continue

            ou_line = float(ou_line)
            predicted = float(predicted_totals[i])

            # P(over) y P(under) via distribucion normal
            p_ov = float(scipy_stats.norm.sf(ou_line, loc=predicted, scale=std))
            p_un = 1.0 - p_ov
            p_ov = np.clip(p_ov, 0.01, 0.99)
            p_un = np.clip(p_un, 0.01, 0.99)

            # EV y Kelly
            over_o = int(over_odds) if over_odds else -110
            under_o = int(under_odds) if under_odds else -110

            ev_over = float(Expected_Value.expected_value(p_ov, over_o))
            ev_under = float(Expected_Value.expected_value(p_un, under_o))
            kelly_over = float(calculate_robust_kelly_simple(over_o, p_ov, epsilon=TOTALS_SIGMA))
            kelly_under = float(calculate_robust_kelly_simple(under_o, p_un, epsilon=TOTALS_SIGMA))

            # Label: la apuesta con mayor EV
            if ev_over >= ev_under:
                ou_label = "OVER"
                ou_prob = p_ov
                ou_ev = ev_over
                ou_kelly = kelly_over
            else:
                ou_label = "UNDER"
                ou_prob = p_un
                ou_ev = ev_under
                ou_kelly = kelly_under

            results.append({
                "home_team": home,
                "away_team": away,
                "predicted_total": predicted,
                "ou_line": ou_line,
                "ou_label": ou_label,
                "ou_prob": ou_prob,
                "ou_ev": ou_ev,
                "ou_kelly": ou_kelly,
                # Detalle de ambos lados
                "ou_over_prob": p_ov,
                "ou_over_ev": ev_over,
                "ou_over_kelly": kelly_over,
                "ou_under_prob": p_un,
                "ou_under_ev": ev_under,
                "ou_under_kelly": kelly_under,
            })

    else:
        # --- Path B: sin modelo, usar linea del libro + implied probability ---
        # Reporta la linea sin edge real (EV=0, Kelly=0)
        for key, game_odds in odds_by_key.items():
            home, away = key
            ou_line = game_odds.get("total")
            over_odds = game_odds.get("over_odds", -110)
            under_odds = game_odds.get("under_odds", -110)

            if ou_line is None:
                continue

            ou_line = float(ou_line)

            # Probabilidad implicita (sin vig) desde el libro
            def _implied(odds: int) -> float:
                odds = float(odds)
                if odds > 0:
                    return 100.0 / (odds + 100.0)
                else:
                    return abs(odds) / (abs(odds) + 100.0)

            over_o = int(over_odds) if over_odds else -110
            under_o = int(under_odds) if under_odds else -110

            p_ov_raw = _implied(over_o)
            p_un_raw = _implied(under_o)
            total_vig = p_ov_raw + p_un_raw
            p_ov = p_ov_raw / total_vig
            p_un = p_un_raw / total_vig

            # Sin modelo no hay edge
            if p_ov >= p_un:
                ou_label = "OVER"
                ou_prob = p_ov
            else:
                ou_label = "UNDER"
                ou_prob = p_un

            logger.debug(
                "Totals odds-only mode for %s @ %s: model not available, line=%.1f",
                away, home, ou_line,
            )

            results.append({
                "home_team": home,
                "away_team": away,
                "predicted_total": None,  # modelo no disponible
                "ou_line": ou_line,
                "ou_label": ou_label,
                "ou_prob": ou_prob,
                "ou_ev": 0.0,
                "ou_kelly": 0.0,
                "ou_over_prob": p_ov,
                "ou_over_ev": 0.0,
                "ou_over_kelly": 0.0,
                "ou_under_prob": p_un,
                "ou_under_ev": 0.0,
                "ou_under_kelly": 0.0,
            })

    return results
