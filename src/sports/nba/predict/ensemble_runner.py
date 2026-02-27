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
from src.core.betting.spread_math import p_cover, expected_margin, ah_probabilities, p_cover_regression, p_win_from_margin
from src.core.betting.expected_value import ah_expected_value
from src.sports.nba.predict.margin_runner import predict_margins

init()

from src.config import CATBOOST_MODELS_DIR, NBA_ML_MODELS_DIR, get_logger

logger = get_logger(__name__)

# --- Conformal prediction para filtrar juegos inciertos ---
_ensemble_conformal = None

# --- Modelo de varianza per-game para Kelly adaptativo ---
_variance_model = None

# Rango valido de epsilon (mismo que fit_variance_model.py)
SIGMA_MIN = 0.02
SIGMA_MAX = 0.20

# --- Pesos del ensemble ---
# Evaluacion Phase 7 en test set (809 juegos, 2025-10 a 2026-02), 179-feature golden set:
#   XGB 65.0% (179-feat) + CatBoost 65.9% (179-feat):
#   XGB 50/Cat 50: acc=65.27%, ECE=0.0365
#   XGB 60/Cat 40: acc=66.38%, ECE=0.0313  <- MEJOR (supera meta 66.3%)
#   XGB 70/Cat 30: acc=65.88%, ECE=0.0303
#   60/40 da mejor accuracy Y mejor calibracion (ECE mas bajo)
W_XGB_ML = 0.60
W_CAT_ML = 0.40

# --- CatBoost model loading ---
CATBOOST_ACCURACY_PATTERN = re.compile(r"CatBoost_(\d+(?:\.\d+)?)%_")

# Modelo CatBoost cacheado (se carga una vez)
_catboost_ml = None


def _select_catboost_path(kind="ML"):
    """Selecciona el mejor modelo CatBoost por accuracy + fecha de modificacion."""
    candidates = list(CATBOOST_MODELS_DIR.glob(f"*{kind}*.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No CatBoost {kind} model found in {CATBOOST_MODELS_DIR}")

    def score(path):
        match = CATBOOST_ACCURACY_PATTERN.search(path.name)
        accuracy = float(match.group(1)) if match else 0.0
        # Accuracy es el criterio principal; mtime como desempate entre modelos de igual accuracy
        return (accuracy, path.stat().st_mtime)

    return max(candidates, key=score)


def _load_catboost():
    """Carga el modelo CatBoost ML (lazy, una sola vez)."""
    global _catboost_ml
    if _catboost_ml is None:
        path = _select_catboost_path("ML")
        _catboost_ml = joblib.load(path)
        logger.info("CatBoost ML loaded: %s", path.name)
    return _catboost_ml


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
    """Carga el modelo de varianza per-game (lazy, una sola vez).

    El modelo predice sigma(features) para cada partido:
      sigma bajo → juego predecible → Kelly agresivo
      sigma alto → juego incierto → Kelly conservador

    Si no existe ensemble_variance.json, retorna None (sigma es opcional).
    """
    global _variance_model
    if _variance_model is None:
        variance_path = NBA_ML_MODELS_DIR / "ensemble_variance.json"
        if variance_path.exists():
            try:
                booster = xgb.Booster()
                booster.load_model(str(variance_path))
                _variance_model = booster
                logger.info("Variance model loaded: %s", variance_path.name)
            except Exception as e:
                logger.warning("Error loading variance model: %s", e)
                _variance_model = False
        else:
            logger.debug("No ensemble_variance.json found — adaptive Kelly disabled")
            _variance_model = False
    return _variance_model if _variance_model is not False else None


def _predict_sigmas(variance_model, data, ml_probs, xgb_ml_probs, cat_ml_probs):
    """Predice sigma per-game usando el modelo de varianza.

    Features: datos base + 2 meta-features (margin, disagreement).
    Output: sigmas clipeados a [SIGMA_MIN, SIGMA_MAX].
    """
    margin = np.max(ml_probs, axis=1) - 0.5
    disagreement = np.abs(xgb_ml_probs[:, 1] - cat_ml_probs[:, 1])
    features_aug = np.column_stack([data, margin, disagreement])
    sigmas = variance_model.predict(xgb.DMatrix(features_aug))
    return np.clip(sigmas, SIGMA_MIN, SIGMA_MAX)


def _catboost_predict_proba(model, data):
    """Predice probabilidades con CatBoost.

    CatBoost.predict_proba() retorna array (N, 2) con [P(away), P(home)],
    mismo formato que XGBoost calibrado.
    """
    return model.predict_proba(data)


def _generate_all_predictions(data, todays_games_uo, frame_ml):
    """Genera predicciones combinadas XGB+CatBoost para ML y XGB para O/U.

    ML: weighted average XGB 60% + CatBoost 40%
    O/U: XGBoost solo

    Retorna (ml_probs, ou_probs, xgb_ml_probs, cat_ml_probs).
    """
    # --- Cargar modelos ---
    XGBoost_Runner._load_models()
    cat_model = _load_catboost()

    # --- XGBoost ML ---
    xgb_ml_probs = XGBoost_Runner._predict_probs(
        XGBoost_Runner.xgb_ml, data, XGBoost_Runner.xgb_ml_calibrator
    )

    # --- CatBoost ML ---
    # CatBoost usa los mismos datos crudos que XGBoost (sin normalizar)
    cat_ml_probs = _catboost_predict_proba(cat_model, data)

    # --- Combinar ML: weighted average ---
    ml_probs = W_XGB_ML * xgb_ml_probs + W_CAT_ML * cat_ml_probs

    # --- O/U: XGBoost solo (opcional) ---
    if XGBoost_Runner.xgb_uo is not None:
        frame_uo = frame_ml.copy()
        frame_uo["OU"] = np.asarray(todays_games_uo, dtype=float)
        ou_probs = XGBoost_Runner._predict_probs(
            XGBoost_Runner.xgb_uo,
            frame_uo.values.astype(float),
            XGBoost_Runner.xgb_uo_calibrator,
        )
    else:
        ou_probs = np.full((len(todays_games_uo), 2), 0.5)

    return ml_probs, ou_probs, xgb_ml_probs, cat_ml_probs


def _build_game_blocks(games, ml_probs, ou_probs, xgb_ml_probs, cat_ml_probs,
                       todays_games_uo, home_team_odds, away_team_odds,
                       kelly_flag, market_info, spread_home_odds, spread_away_odds,
                       conformal_set_sizes=None, conformal_margins=None,
                       sigmas=None, reg_margins=None):
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

        home_ah = ah_probabilities(prob_home, line)
        away_ah = ah_probabilities(1.0 - prob_home, -line)

        ah_ev_home = float(ah_expected_value(home_ah, sh_odds))
        ah_ev_away = float(ah_expected_value(away_ah, sa_odds))
        p_home_cover = home_ah["p_full_win"] + home_ah["p_half_win"]
        p_away_cover = away_ah["p_full_win"] + away_ah["p_half_win"]
        ah_kelly_home = float(calculate_robust_kelly_simple(sh_odds, p_home_cover, epsilon=sigma_i))
        ah_kelly_away = float(calculate_robust_kelly_simple(sa_odds, p_away_cover, epsilon=sigma_i))

        if ah_ev_home > ah_ev_away and ah_ev_home > 0:
            b["ah_side"] = home_team
            b["ah_line"] = f"{line:+.1f}"
            b["ah_ev"] = ah_ev_home
            b["ah_p"] = p_home_cover
            b["ah_kelly"] = ah_kelly_home
        elif ah_ev_away > 0:
            b["ah_side"] = away_team
            b["ah_line"] = f"{-line:+.1f}"
            b["ah_ev"] = ah_ev_away
            b["ah_p"] = p_away_cover
            b["ah_kelly"] = ah_kelly_away
        else:
            b["ah_side"] = None
            b["ah_ev"] = max(ah_ev_home, ah_ev_away)
            b["ah_p"] = max(p_home_cover, p_away_cover)
            b["ah_kelly"] = 0.0
            b["ah_line"] = f"{line:+.1f}"

        b["ah_is_quarter"] = home_ah["is_quarter"]

        # --- Margin regression ---
        if reg_margins is not None:
            mu_reg = float(reg_margins[idx])
            b["reg_margin"] = mu_reg
            b["reg_p_cover"] = p_cover_regression(mu_reg, line)
        else:
            b["reg_margin"] = None

        blocks.append(b)

    # Rankear por max EV descendente
    blocks.sort(key=lambda x: x["max_ev"], reverse=True)
    return blocks


def _print_compact_output(blocks, kelly_flag, conformal=None, has_sigma=False):
    """Imprime bloques compactos por partido, rankeados por EV."""
    # Header
    n_total = len(blocks)
    n_bet = sum(1 for b in blocks if b["conf_ss"] == 1) if blocks[0]["conf_ss"] is not None else n_total
    sigma_label = " | DRO-Kelly" if has_sigma else ""
    conf_label = f" | Conformal {n_bet}/{n_total}" if blocks[0]["conf_ss"] is not None else ""
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"  PICKS ranked by EV{conf_label}{sigma_label}")
    print(f"{'='*60}{Style.RESET_ALL}\n")

    for rank, b in enumerate(blocks, 1):
        # --- Status tag: BET / SKIP / TRAP ---
        has_trap = b["trap_home"] or b["trap_away"]
        if has_trap:
            tag = f"{Fore.YELLOW}TRAP{Style.RESET_ALL}"
        elif b["conf_ss"] is not None and b["conf_ss"] != 1:
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

        # --- Line 3: AH ---
        if b["ah_side"]:
            ah_ev_color = Fore.GREEN if b["ah_ev"] > 0 else Fore.RED
            q_tag = " Q" if b["ah_is_quarter"] else ""
            print(
                f"       AH  {b['ah_side']} ({b['ah_line']}{q_tag}): "
                f"P={b['ah_p']:.1%} EV={ah_ev_color}{b['ah_ev']:+.1f}{Style.RESET_ALL} "
                f"Kelly={b['ah_kelly']:.2f}%"
                f"  margin={b['margin']:+.1f}"
            )
        else:
            print(
                f"       AH  --- spread {b['ah_line']}, margin={b['margin']:+.1f}"
                f"  EV={b['ah_ev']:+.1f}"
            )

        # --- Line 3b: Margin regression comparison (if available) ---
        if b["reg_margin"] is not None:
            print(
                f"           CLF: μ={b['margin']:+.1f}  |  "
                f"REG: μ={b['reg_margin']:+.1f} P(cover)={b['reg_p_cover']:.1%}"
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
        print(
            f"       O/U {ou_color}{b['ou_label']}{Style.RESET_ALL} "
            f"{b['ou_line']} ({b['ou_conf']}%)"
        )

        # --- Conformal margin ---
        if b["conf_ss"] is not None:
            if b["conf_ss"] == 1:
                print(f"       {Fore.GREEN}conformal: margin={b['conf_margin']:.2f}{Style.RESET_ALL}")
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
                              reg_margins=None):
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

        # --- Margin Regression (si disponible) ---
        if reg_margins is not None:
            mu_reg = float(reg_margins[idx])
            entry["reg_margin"] = mu_reg
            entry["reg_p_win"] = p_win_from_margin(mu_reg)
            entry["reg_p_cover"] = p_cover_regression(mu_reg, line)

        result.append(entry)
    return result


def ensemble_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion, market_info=None, spread_home_odds=None, spread_away_odds=None, sportsbook=None):
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
    ml_probs, ou_probs, xgb_ml_probs, cat_ml_probs = _generate_all_predictions(
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
    reg_margins = predict_margins(data)
    if reg_margins is not None:
        logger.info("Margin model: mean=%.1f, std=%.1f, range=[%.1f, %.1f]",
                     reg_margins.mean(), reg_margins.std(), reg_margins.min(), reg_margins.max())

    # --- Output compacto: un bloque por partido, rankeado por EV ---
    blocks = _build_game_blocks(
        games, ml_probs, ou_probs, xgb_ml_probs, cat_ml_probs,
        todays_games_uo, home_team_odds, away_team_odds,
        kelly_criterion, market_info, spread_home_odds, spread_away_odds,
        conformal_set_sizes, conformal_margins, sigmas, reg_margins,
    )
    _print_compact_output(blocks, kelly_criterion, conformal, sigmas is not None)

    deinit()
    return _build_prediction_results(
        games, ml_probs, ou_probs, todays_games_uo,
        home_team_odds, away_team_odds, market_info,
        conformal_set_sizes, conformal_margins, sigmas,
        spread_home_odds, spread_away_odds, reg_margins,
    )
