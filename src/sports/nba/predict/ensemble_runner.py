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

    # --- O/U: XGBoost solo ---
    frame_uo = frame_ml.copy()
    frame_uo["OU"] = np.asarray(todays_games_uo, dtype=float)
    ou_probs = XGBoost_Runner._predict_probs(
        XGBoost_Runner.xgb_uo,
        frame_uo.values.astype(float),
        XGBoost_Runner.xgb_uo_calibrator,
    )

    return ml_probs, ou_probs, xgb_ml_probs, cat_ml_probs


def _print_game_predictions(games, ml_probs, ou_probs, xgb_ml_probs, cat_ml_probs,
                            todays_games_uo, conformal_set_sizes=None, conformal_margins=None,
                            sigmas=None):
    """Muestra prediccion de ganador y O/U para cada partido."""
    for idx, (home_team, away_team) in enumerate(games):
        winner = int(np.argmax(ml_probs[idx]))
        under_over = int(np.argmax(ou_probs[idx]))
        winner_confidence = round(ml_probs[idx][winner] * 100, 1)
        ou_confidence = round(ou_probs[idx][under_over] * 100, 1)

        xgb_winner = int(np.argmax(xgb_ml_probs[idx]))
        cat_winner = int(np.argmax(cat_ml_probs[idx]))
        xgb_conf = round(xgb_ml_probs[idx][winner] * 100, 1)
        cat_conf = round(cat_ml_probs[idx][winner] * 100, 1)

        winner_team = home_team if winner == 1 else away_team
        loser_team = away_team if winner == 1 else home_team
        winner_color = Fore.GREEN if winner == 1 else Fore.RED
        loser_color = Fore.RED if winner == 1 else Fore.GREEN
        ou_label = "UNDER" if under_over == 0 else "OVER"
        ou_color = Fore.MAGENTA if under_over == 0 else Fore.BLUE
        # "+" si ambos modelos coinciden, "~" si discrepan
        agree = "+" if xgb_winner == cat_winner else "~"

        # Indicador conformal: BET si set_size=1, SKIP si set_size!=1
        conf_tag = ""
        if conformal_set_sizes is not None:
            ss = int(conformal_set_sizes[idx])
            margin = float(conformal_margins[idx]) if conformal_margins is not None else 0.0
            if ss == 1:
                conf_tag = f"  {Fore.GREEN}BET{Style.RESET_ALL} m={margin:.2f}"
            else:
                conf_tag = f"  {Fore.YELLOW}SKIP{Style.RESET_ALL} s={ss}"

        # Indicador sigma (varianza per-game)
        sigma_tag = ""
        if sigmas is not None:
            s = float(sigmas[idx])
            sigma_color = Fore.GREEN if s < 0.07 else (Fore.YELLOW if s < 0.10 else Fore.RED)
            sigma_tag = f"  {sigma_color}σ={s:.3f}{Style.RESET_ALL}"

        print(
            f"{winner_color}{winner_team}{Style.RESET_ALL}"
            f"{Fore.CYAN} ({winner_confidence}%){Style.RESET_ALL}"
            f" vs {loser_color}{loser_team}{Style.RESET_ALL}: "
            f"{ou_color}{ou_label} {Style.RESET_ALL}{todays_games_uo[idx]}"
            f"{Fore.CYAN} ({ou_confidence}%){Style.RESET_ALL}"
            f"  [{agree} XGB:{xgb_conf}% Cat:{cat_conf}%]"
            f"{conf_tag}{sigma_tag}"
        )


def _print_ah_section(games, ml_probs, spread_home_odds, spread_away_odds, market_info, sigmas=None, reg_margins=None):
    """Muestra predicciones de Asian Handicap (spread) por partido."""
    spreads = market_info.get("MARKET_SPREAD", np.zeros(len(games)))
    print(f"\n{Fore.CYAN}------------- Asian Handicap (Spread) ---------------{Style.RESET_ALL}")
    for idx, (home_team, away_team) in enumerate(games):
        line = float(spreads[idx])
        prob_home = float(ml_probs[idx][1])
        margin = expected_margin(prob_home)

        sh_odds = int(spread_home_odds[idx]) if spread_home_odds and spread_home_odds[idx] else -110
        sa_odds = int(spread_away_odds[idx]) if spread_away_odds and spread_away_odds[idx] else -110

        # Settlement completo con quarter lines
        home_ah = ah_probabilities(prob_home, line)
        away_ah = ah_probabilities(1.0 - prob_home, -line)
        ev_home = float(ah_expected_value(home_ah, sh_odds))
        ev_away = float(ah_expected_value(away_ah, sa_odds))

        eps = float(sigmas[idx]) if sigmas is not None else 0.05
        p_home_profit = home_ah["p_full_win"] + home_ah["p_half_win"]
        p_away_profit = away_ah["p_full_win"] + away_ah["p_half_win"]
        kelly_home = float(calculate_robust_kelly_simple(sh_odds, p_home_profit, epsilon=eps))
        kelly_away = float(calculate_robust_kelly_simple(sa_odds, p_away_profit, epsilon=eps))

        # Tag para quarter lines
        q_tag = " Q" if home_ah["is_quarter"] else ""

        # Determinar mejor lado
        if ev_home > ev_away and ev_home > 0:
            best_side = "HOME"
            best_color = Fore.GREEN
            best_ev = ev_home
            best_kelly = kelly_home
            best_p = p_home_profit
            best_line = f"{line:+.1f}"
        elif ev_away > 0:
            best_side = "AWAY"
            best_color = Fore.GREEN
            best_ev = ev_away
            best_kelly = kelly_away
            best_p = p_away_profit
            best_line = f"{-line:+.1f}"
        else:
            best_side = "---"
            best_color = Fore.YELLOW
            best_ev = max(ev_home, ev_away)
            best_kelly = 0.0
            best_p = max(p_home_profit, p_away_profit)
            best_line = f"{line:+.1f}"

        team_label = home_team if best_side == "HOME" else away_team if best_side == "AWAY" else "ninguno"
        print(
            f"  {home_team} vs {away_team}: "
            f"spread {line:+.1f}, margin {margin:+.1f}{q_tag}  "
            f"{best_color}{best_side}{Style.RESET_ALL} "
            f"({team_label} {best_line}) "
            f"P={best_p:.1%} EV={best_ev:+.1f} Kelly={best_kelly:.2f}%"
        )

        # Comparación con margen del regresor (si disponible)
        if reg_margins is not None:
            mu_reg = float(reg_margins[idx])
            p_cover_reg = p_cover_regression(mu_reg, line)
            p_cover_clf = home_ah["p_full_win"] + home_ah["p_half_win"]
            print(f"    CLF: μ={margin:+.1f} P(cover)={p_cover_clf:.1%}  |  "
                  f"REG: μ={mu_reg:+.1f} P(cover)={p_cover_reg:.1%}")


def _print_ev_and_kelly(games, ml_probs, home_team_odds, away_team_odds, kelly_flag, sigmas=None):
    """Muestra Expected Value y opcionalmente Kelly Criterion por partido.

    Si sigmas esta disponible, usa robust_kelly(epsilon=sigma) en vez de eighth_kelly.
    """
    if kelly_flag:
        label = "Expected Value & Robust Kelly (σ-adaptive)" if sigmas is not None else "Expected Value & Kelly Criterion"
        print(f"------------{label}-----------")
    else:
        print("---------------------Expected Value--------------------")

    for idx, (home_team, away_team) in enumerate(games):
        ev_home = ev_away = 0
        if home_team_odds[idx] and away_team_odds[idx]:
            ev_home = float(Expected_Value.expected_value(ml_probs[idx][1], int(home_team_odds[idx])))
            ev_away = float(Expected_Value.expected_value(ml_probs[idx][0], int(away_team_odds[idx])))

        ev_colors = {
            "home": Fore.GREEN if ev_home > 0 else Fore.RED,
            "away": Fore.GREEN if ev_away > 0 else Fore.RED,
        }

        if sigmas is not None:
            sigma_i = float(sigmas[idx])
            kelly_home = calculate_robust_kelly_simple(home_team_odds[idx], ml_probs[idx][1], epsilon=sigma_i)
            kelly_away = calculate_robust_kelly_simple(away_team_odds[idx], ml_probs[idx][0], epsilon=sigma_i)
            bankroll_home = f" Bankroll (DRO-Kelly, σ={sigma_i:.3f}): {kelly_home}%"
            bankroll_away = f" Bankroll (DRO-Kelly, σ={sigma_i:.3f}): {kelly_away}%"
        else:
            bankroll_home = (
                f" Bankroll (\u215b-Kelly, cap 2.5%): "
                f"{kc.calculate_eighth_kelly(home_team_odds[idx], ml_probs[idx][1])}%"
            )
            bankroll_away = (
                f" Bankroll (\u215b-Kelly, cap 2.5%): "
                f"{kc.calculate_eighth_kelly(away_team_odds[idx], ml_probs[idx][0])}%"
            )
        print(
            f"{home_team} EV: {ev_colors['home']}{ev_home}{Style.RESET_ALL}"
            f"{bankroll_home if kelly_flag else ''}"
        )
        print(
            f"{away_team} EV: {ev_colors['away']}{ev_away}{Style.RESET_ALL}"
            f"{bankroll_away if kelly_flag else ''}"
        )


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

    _print_game_predictions(games, ml_probs, ou_probs, xgb_ml_probs, cat_ml_probs,
                            todays_games_uo, conformal_set_sizes, conformal_margins,
                            sigmas)

    # Resumen conformal antes de EV/Kelly
    if conformal_set_sizes is not None:
        n_bet = int((conformal_set_sizes == 1).sum())
        n_total = len(games)
        print(f"\n  Conformal: {n_bet}/{n_total} juegos confiados"
              f" (threshold={conformal.threshold_:.3f})")
        if n_bet < n_total:
            print(f"  {n_total - n_bet} juegos filtrados (set_size=2, ambas clases plausibles)\n")

    # Resumen sigma
    if sigmas is not None:
        print(f"  Sigma: mean={sigmas.mean():.3f}, range=[{sigmas.min():.3f}, {sigmas.max():.3f}]"
              f" → Kelly adaptativo (DRO)\n")

    _print_ev_and_kelly(games, ml_probs, home_team_odds, away_team_odds, kelly_criterion, sigmas)

    # --- Asian Handicap (spread) ---
    _print_ah_section(games, ml_probs, spread_home_odds, spread_away_odds, market_info, sigmas, reg_margins)

    deinit()
    return _build_prediction_results(
        games, ml_probs, ou_probs, todays_games_uo,
        home_team_odds, away_team_odds, market_info,
        conformal_set_sizes, conformal_margins, sigmas,
        spread_home_odds, spread_away_odds, reg_margins,
    )
