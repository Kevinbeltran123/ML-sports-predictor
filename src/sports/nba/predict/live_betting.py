"""
LiveBetting: ajuste Bayesiano de predicciones in-game (post-Q1, Q2, Q3).

CONCEPTO CLAVE — Por que log-odds y no suma directa:
==================================================
Si P_pregame = 0.68 y sumamos +0.05 por punto de ventaja:
  - Ventaja de +6  -> P = 0.98 (exagerado)
  - Ventaja de +7  -> P = 1.03 (imposible)

El problema: probabilidad esta limitada a [0, 1].

Solucion: trabajar en espacio log-odds -> (-inf, +inf)

    logit(p) = log(p / (1-p))       # transforma [0,1] -> (-inf, +inf)
    sigmoid(x) = 1 / (1+e^{-x})    # transforma (-inf, +inf) -> [0,1]

NIVEL 1 — Formula simple (Stern 1994):
    logit_adj = logit(P_pregame) + B * (score_diff / sqrt(possessions))
    P_adjusted = sigmoid(logit_adj)
    B = 0.45 calibrado para NBA

NIVEL 2 — Multi-senal (regresion logistica entrenada):
    logit(P) = B0 + B1*LOGIT_PREGAME + B2*SCORE_DIFF_NORM + B3*FG_PCT + ...
    Usa 8 features diferenciales aprendidas de datos historicos.
    Fallback a Nivel 1 si el modelo no esta disponible.

Uso:
    # Formula simple:
    from src.sports.nba.predict.live_betting import bayesian_q1_adjustment
    p, msg = bayesian_q1_adjustment(0.68, score_diff=8, total_possessions=50)

    # Multi-senal (usa modelo si existe, fallback a simple):
    from src.sports.nba.predict.live_betting import multi_signal_adjustment
    p, delta, expl, conf = multi_signal_adjustment(0.68, box_home, box_away, period=1)

    # Loop completo (desde main.py):
    from src.sports.nba.predict.live_betting import run_live_session
    run_live_session(pregame_predictions)  # monitorea Q1, Q2 y Q3
"""

import logging
import math
import time
from datetime import datetime
from pathlib import Path

from colorama import Fore, Style, init, deinit

from src.sports.nba.features.live_game_state import get_live_scoreboard, get_live_box_score, format_clock

logger = logging.getLogger(__name__)

# Factor de calibracion Bayesiana (Stern 1994, NBA-especifico)
# El valor exacto importa menos que la idea: mas senal = mas peso
BETA = 0.45

# Cuantos segundos esperar entre polls
POLL_INTERVAL_SECONDS = 30

# Tiempo maximo de polling en horas (evita loop infinito si algo falla)
MAX_POLL_HOURS = 4

# --- Cache de modelos logisticos in-game (cargados una sola vez) ---
_model_cache: dict[int, object] = {}        # period -> LogisticRegression
_conformal_cache: dict[int, object] = {}    # period -> ConformalClassifier


def bayesian_q1_adjustment(
    p_pregame: float,
    score_diff: int,
    total_possessions: float,
) -> tuple[float, str]:
    """Ajusta la probabilidad de victoria del local usando el score de Q1.

    Formula:
        logit_adj = logit(P_pregame) + B * (score_diff / sqrt(possessions))
        P_adjusted = sigmoid(logit_adj)

    Args:
        p_pregame:        probabilidad pre-partido del equipo local (0.0 a 1.0)
        score_diff:       puntos_local - puntos_visitante al final de Q1
                          (positivo = local gana Q1, negativo = visitante gana Q1)
        total_possessions: posesiones estimadas del partido hasta el final de Q1

    Returns:
        (p_adjusted, explanation_string)

    Ejemplos:
        bayesian_q1_adjustment(0.68, 0, 50) -> (~0.680, "sin cambio")
        bayesian_q1_adjustment(0.50, 8, 25) -> (~0.660, "local +8 con 25 poss")
        bayesian_q1_adjustment(0.68, 8, 50) -> (~0.736)
    """
    # Caso extremo: probabilidad ya en los bordes (0 o 1)
    p_clamped = max(0.001, min(0.999, p_pregame))

    # Paso 1: Convertir a log-odds (espacio lineal donde podemos sumar)
    logit_pre = math.log(p_clamped / (1.0 - p_clamped))

    # Paso 2: Normalizar la senal del score por sqrt(posesiones)
    poss = max(1.0, total_possessions)
    normalized_signal = score_diff / math.sqrt(poss)

    # Paso 3: Actualizar log-odds con la senal ponderada
    delta_logit = BETA * normalized_signal
    logit_adj = logit_pre + delta_logit

    # Paso 4: Convertir de vuelta a probabilidad con sigmoid
    p_adjusted = 1.0 / (1.0 + math.exp(-logit_adj))

    explanation = (
        f"score_diff={score_diff:+d}, poss~{poss:.0f}, "
        f"senal_norm={normalized_signal:+.2f}, delta_logit={delta_logit:+.3f}"
    )

    return p_adjusted, explanation


def _load_ingame_model(period: int):
    """Carga el modelo logistico in-game para un periodo dado.

    Busca en INGAME_MODELS_DIR el archivo mas reciente que matchee
    'logistic_Q{period}_*.pkl'. Cachea en memoria para no recargar.

    Returns:
        (model, conformal) o (None, None) si no existe modelo.
    """
    if period in _model_cache:
        return _model_cache[period], _conformal_cache.get(period)

    try:
        from src.config import INGAME_MODELS_DIR
        import joblib

        if not INGAME_MODELS_DIR.exists():
            return None, None

        # Buscar el modelo mas reciente para este periodo
        pattern = f"logistic_Q{period}_*pct.pkl"
        candidates = sorted(INGAME_MODELS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
        if not candidates:
            return None, None

        model_path = candidates[-1]  # mas reciente
        model = joblib.load(model_path)
        _model_cache[period] = model
        logger.info("Modelo in-game cargado: %s", model_path.name)

        # Cargar conformal si existe
        conformal_path = INGAME_MODELS_DIR / f"logistic_Q{period}_conformal.pkl"
        conformal = None
        if conformal_path.exists():
            conformal = joblib.load(conformal_path)
            _conformal_cache[period] = conformal
            logger.info("Conformal in-game cargado: %s", conformal_path.name)

        return model, conformal

    except Exception as e:
        logger.warning("Error cargando modelo in-game Q%d: %s", period, e)
        return None, None


def multi_signal_adjustment(
    p_pregame: float,
    box_home: dict,
    box_away: dict,
    period: int = 1,
) -> tuple[float, float, str, int]:
    """Ajuste multi-senal usando modelo logistico entrenado.

    Usa 8 features diferenciales (FG%, TOV, REB, AST, FG3%, FT%, score_diff,
    logit_pregame) para ajustar la probabilidad. Si el modelo no existe,
    cae al bayesian_q1_adjustment() original.

    Args:
        p_pregame: probabilidad pre-partido del local (0.0 a 1.0)
        box_home:  dict de stats del local (de get_live_box_score() o similar)
        box_away:  dict de stats del visitante (mismo formato)
        period:    periodo completado (1=Q1, 2=Q2, 3=Q3)

    Returns:
        (p_adjusted, delta, explanation, conformal_set_size)
    """
    # Intentar cargar modelo entrenado
    model, conformal = _load_ingame_model(period)

    if model is None:
        # FALLBACK: formula simple (Stern 1994)
        home_pts = box_home.get("pts", box_home.get("PTS", 0))
        away_pts = box_away.get("pts", box_away.get("PTS", 0))
        score_diff = int(home_pts) - int(away_pts)

        home_poss = box_home.get("possessions", box_home.get("POSS", 25))
        away_poss = box_away.get("possessions", box_away.get("POSS", 25))
        total_poss = (float(home_poss) + float(away_poss)) / 2.0

        p_adj, expl = bayesian_q1_adjustment(p_pregame, score_diff, total_poss)
        delta = p_adj - p_pregame
        return p_adj, delta, f"[simple B=0.45] {expl}", 0

    # --- Modelo entrenado disponible: computar features ---
    from src.sports.nba.features.ingame_features import (
        compute_ingame_differentials,
        box_score_to_stats_dict,
        INGAME_FEATURE_NAMES,
    )
    import numpy as np

    # Convertir formato live API -> formato esperado por compute_ingame_differentials
    home_stats = box_score_to_stats_dict(box_home)
    away_stats = box_score_to_stats_dict(box_away)

    # Computar las 8 features diferenciales
    features = compute_ingame_differentials(home_stats, away_stats, p_pregame)

    # Construir vector de features en el orden exacto
    X = np.array([[features[name] for name in INGAME_FEATURE_NAMES]])

    # Predecir probabilidad
    p_adj = float(model.predict_proba(X)[0, 1])  # P(HOME_WIN=1)
    delta = p_adj - p_pregame

    # Conformal prediction para nivel de confianza
    set_size = 0
    if conformal is not None:
        try:
            proba = model.predict_proba(X)[0]  # [P(0), P(1)]
            set_size, margin = conformal.predict_confidence(proba)
        except Exception:
            set_size = 0

    # Explicacion con top features
    top_features = []
    for name in ["DIFF_FG_PCT", "SCORE_DIFF_NORM", "DIFF_TOV_RATE"]:
        val = features.get(name, 0)
        if abs(val) > 0.01:
            top_features.append(f"{name}={val:+.3f}")

    conf_label = "ALTA" if set_size == 1 else "BAJA" if set_size == 2 else "N/A"
    explanation = (
        f"[logistic Q{period}] P={p_adj:.3f}, delta={delta:+.3f}, "
        f"conf={conf_label}, {', '.join(top_features)}"
    )

    return p_adj, delta, explanation, set_size


def _match_game_to_prediction(game: dict, pregame_predictions: list[dict]) -> dict | None:
    """Encuentra la prediccion pre-partido que corresponde a un juego live."""
    live_home_name = game.get("home_team", "").lower()
    live_away_name = game.get("away_team", "").lower()
    live_home_city = game.get("home_team_city", "").lower()
    live_away_city = game.get("away_team_city", "").lower()

    for pred in pregame_predictions:
        pred_home = pred["home_team"].lower()
        pred_away = pred["away_team"].lower()

        home_match = (
            live_home_name in pred_home or
            pred_home.endswith(live_home_name) or
            live_home_city in pred_home
        )
        away_match = (
            live_away_name in pred_away or
            pred_away.endswith(live_away_name) or
            live_away_city in pred_away
        )

        if home_match and away_match:
            return pred

    return None


def _print_period_update(
    game: dict,
    pred: dict,
    box: dict,
    period: int,
    prev_p: float | None = None,
):
    """Muestra el update de un cuarto finalizado (Q1, Q2 o Q3)."""
    home = pred["home_team"]
    away = pred["away_team"]
    p_pre = pred["p_pregame"]
    p_away_pre = 1.0 - p_pre

    p_ref = prev_p if prev_p is not None else p_pre
    p_away_ref = 1.0 - p_ref

    home_score = game["home_score"]
    away_score = game["away_score"]
    score_diff = home_score - away_score

    # Usar multi_signal_adjustment (con fallback automatico a simple)
    if box is not None:
        p_adj, delta_from_pre, expl, conf_set = multi_signal_adjustment(
            p_pre, box["home"], box["away"], period=period
        )
    else:
        total_poss = 25.0 * period
        p_adj, expl = bayesian_q1_adjustment(p_pre, score_diff, total_poss)
        delta_from_pre = p_adj - p_pre
        expl = f"[sin box score] {expl}"
        conf_set = 0

    p_away_adj = 1.0 - p_adj
    delta_from_prev = p_adj - p_ref

    q_label = f"Q{period}"
    if score_diff > 0:
        leader = f"+{score_diff} {home.split()[-1]}"
        delta_color = Fore.GREEN
    elif score_diff < 0:
        leader = f"+{abs(score_diff)} {away.split()[-1]}"
        delta_color = Fore.RED
    else:
        leader = "EMPATADOS"
        delta_color = Style.RESET_ALL

    conf_label = ""
    if conf_set == 1:
        conf_label = f"  {Fore.GREEN}CONFIANZA ALTA{Style.RESET_ALL}"
    elif conf_set == 2:
        conf_label = f"  {Fore.YELLOW}CONFIANZA BAJA{Style.RESET_ALL}"

    print(f"\n{'=' * 65}")
    print(f"  UPDATE -- {q_label} FINALIZADO: {home} vs {away}{conf_label}")
    print(f"  Score: {home_score}-{away_score}  ({leader})")
    print(f"{'-' * 65}")
    print(f"  PRE-PARTIDO:   {home.split()[-1]} {p_pre:.1%}  <->  {away.split()[-1]} {p_away_pre:.1%}")

    if prev_p is not None and period > 1:
        print(f"  PRE-{q_label}:      {home.split()[-1]} {p_ref:.1%}  <->  {away.split()[-1]} {p_away_ref:.1%}")

    print(f"  LIVE {q_label} ADJ:   {home.split()[-1]} {p_adj:.1%}  <->  {away.split()[-1]} {p_away_adj:.1%}  "
          f"({delta_color}{delta_from_prev:+.1%} {home.split()[-1]}{Style.RESET_ALL})")
    print(f"{'-' * 65}")

    edge_threshold = 0.04
    if abs(delta_from_pre) >= edge_threshold:
        if delta_from_pre > 0:
            print(f"  HINT: Si el mercado aun da {away.split()[-1]} > {p_away_adj:.0%} -> edge UNDER {away.split()[-1]}")
        else:
            print(f"  HINT: Si el mercado aun da {home.split()[-1]} > {p_adj:.0%} -> edge UNDER {home.split()[-1]}")

    print(f"  (detalle: {expl})")
    print(f"{'=' * 65}")

    return p_adj


def run_live_session(pregame_predictions: list[dict]):
    """Loop principal de live betting: monitorea Q1, Q2 y Q3 y ajusta probabilidades.

    Args:
        pregame_predictions: lista de dicts con claves:
          - home_team (str): ej "Denver Nuggets"
          - away_team (str): ej "Boston Celtics"
          - p_pregame (float): probabilidad del local (0.0 a 1.0)
    """
    if not pregame_predictions:
        print("LiveBetting: sin predicciones pre-partido disponibles.")
        return

    init()  # colorama

    print(f"\n{'=' * 65}")
    print(f"  LIVE BETTING -- {datetime.now().strftime('%H:%M:%S')}")
    print(f"  Monitoreando {len(pregame_predictions)} partido(s)")
    print(f"  Cuartos: Q1, Q2, Q3 (ajuste multi-senal con fallback)")
    print(f"  (actualizacion cada {POLL_INTERVAL_SECONDS}s, Ctrl+C para salir)")
    print(f"{'=' * 65}\n")

    last_period = {}
    processed_periods = {}
    last_p_adjusted = {}

    start_time = time.time()
    max_seconds = MAX_POLL_HOURS * 3600
    poll_count = 0

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed > max_seconds:
                print(f"\nLiveBetting: timeout despues de {MAX_POLL_HOURS}h. Saliendo.")
                break

            poll_count += 1

            live_games = get_live_scoreboard()

            if not live_games:
                print(f"  [poll #{poll_count}] Sin datos de la API. Reintentando en {POLL_INTERVAL_SECONDS}s...")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            monitored = []
            all_final = True

            for game in live_games:
                pred = _match_game_to_prediction(game, pregame_predictions)
                if pred is None:
                    continue

                game_id = game["game_id"]
                status = game["status"]
                period = game["period"]

                monitored.append(game)

                if status != 3:
                    all_final = False

                if game_id not in processed_periods:
                    processed_periods[game_id] = set()

                prev_period = last_period.get(game_id, 0)

                for check_period in [1, 2, 3]:
                    if (check_period not in processed_periods[game_id] and
                            period > check_period and
                            status in (2, 3)):

                        box = get_live_box_score(game_id) if status == 2 else None

                        prev_p = last_p_adjusted.get(game_id)
                        p_adj = _print_period_update(
                            game, pred, box,
                            period=check_period,
                            prev_p=prev_p,
                        )
                        processed_periods[game_id].add(check_period)
                        last_p_adjusted[game_id] = p_adj

                last_period[game_id] = period

            if monitored:
                print(f"  [{datetime.now().strftime('%H:%M:%S')} poll #{poll_count}]  ", end="")
                status_parts = []
                for game in monitored:
                    pred = _match_game_to_prediction(game, pregame_predictions)
                    h = pred["home_team"].split()[-1]
                    a = pred["away_team"].split()[-1]
                    s = game["status"]
                    p = game["period"]
                    clk = format_clock(game["clock"])
                    hs = game["home_score"]
                    aws = game["away_score"]

                    if s == 1:
                        status_parts.append(f"{h} vs {a}: programado")
                    elif s == 2:
                        q_label = f"Q{p}" if p <= 4 else f"OT{p-4}"
                        status_parts.append(f"{h} {hs}-{aws} {a} ({q_label} {clk})")
                    else:
                        status_parts.append(f"{h} {hs}-{aws} {a}: FINAL")

                print(" | ".join(status_parts))
            else:
                print(f"  [poll #{poll_count}] Sin juegos monitoreados aun (esperando inicio)...")

            if all_final and monitored and len(monitored) == len(pregame_predictions):
                _print_session_summary(pregame_predictions, last_p_adjusted, processed_periods)
                break

            time.sleep(POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print(f"\n\n  Live betting interrumpido por el usuario.")

    deinit()  # colorama


def _print_session_summary(
    predictions: list[dict],
    last_p: dict[str, float],
    processed: dict[str, set],
):
    """Imprime resumen final de la sesion live con evolucion de probabilidades."""
    print(f"\n{'=' * 65}")
    print(f"  RESUMEN DE SESION LIVE -- {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'-' * 65}")

    for pred in predictions:
        home = pred["home_team"].split()[-1]
        p_pre = pred["p_pregame"]
        print(f"  {home}: Pre={p_pre:.1%}", end="")
        print()

    print(f"{'=' * 65}")
    print(f"  Todos los partidos finalizados. Sesion completada.")


# --- Backward compatibility alias ---
def run_live_q1_session(pregame_predictions: list[dict]):
    """Alias de backward compat. Redirige a run_live_session()."""
    run_live_session(pregame_predictions)
