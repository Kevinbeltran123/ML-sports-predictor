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

from src.sports.nba.features.live_game_state import (
    get_live_scoreboard, get_live_box_score, get_live_play_by_play, format_clock,
)
from src.sports.nba.features.live_pbp_tracker import LivePBPTracker

logger = logging.getLogger(__name__)

# Factor de calibracion Bayesiana per-quarter (Stern 1994, NBA-especifico)
# Mas quarter jugado = score mas predictivo = BETA mas alto
BETA_BY_QUARTER = {1: 0.35, 2: 0.45, 3: 0.60, 4: 0.75}
BETA = 0.45  # default fallback

# Polling: 30s normal, 10s cuando algun juego tiene < 2min restantes en el Q
POLL_INTERVAL_SECONDS = 30
POLL_FAST_SECONDS = 10
FAST_POLL_THRESHOLD = 120  # segundos restantes para activar fast polling

# Tiempo maximo de polling en horas (evita loop infinito si algo falla)
# 5 horas para cubrir OT. Override con env var LIVE_BETTING_TIMEOUT_HOURS
import os
MAX_POLL_HOURS = int(os.environ.get("LIVE_BETTING_TIMEOUT_HOURS", 5))

def _parse_clock_seconds(clock_str: str) -> float:
    """Parse NBA clock 'PT05M30.00S' → 330.0 seconds. Returns -1 if unparseable."""
    import re
    m = re.match(r"PT(\d+)M([\d.]+)S", clock_str)
    return int(m.group(1)) * 60.0 + float(m.group(2)) if m else -1.0


# --- Cache de modelos logisticos in-game (cargados una sola vez) ---
_model_cache: dict[int, object] = {}        # period -> LogisticRegression
_conformal_cache: dict[int, object] = {}    # period -> ConformalClassifier


def bayesian_q1_adjustment(
    p_pregame: float,
    score_diff: int,
    total_possessions: float,
    period: int = 1,
) -> tuple[float, str]:
    """Ajusta la probabilidad de victoria del local usando el score del quarter.

    Formula:
        logit_adj = logit(P_pregame) + B(Q) * (score_diff / sqrt(possessions))
        P_adjusted = sigmoid(logit_adj)

    BETA es adaptativo por quarter: Q1=0.35, Q2=0.45, Q3=0.60, Q4=0.75.
    En quarters mas avanzados, el score es mas predictivo del resultado final.

    Args:
        p_pregame:        probabilidad pre-partido del equipo local (0.0 a 1.0)
        score_diff:       puntos_local - puntos_visitante
        total_possessions: posesiones estimadas del partido hasta el momento
        period:           quarter actual (1-4, default 1)

    Returns:
        (p_adjusted, explanation_string)
    """
    beta = BETA_BY_QUARTER.get(period, BETA)

    p_clamped = max(0.001, min(0.999, p_pregame))
    logit_pre = math.log(p_clamped / (1.0 - p_clamped))

    poss = max(1.0, total_possessions)
    normalized_signal = score_diff / math.sqrt(poss)

    delta_logit = beta * normalized_signal
    logit_adj = logit_pre + delta_logit

    p_adjusted = 1.0 / (1.0 + math.exp(-logit_adj))

    explanation = (
        f"score_diff={score_diff:+d}, poss~{poss:.0f}, "
        f"B={beta:.2f}(Q{period}), senal={normalized_signal:+.2f}, "
        f"delta={delta_logit:+.3f}"
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
    pbp_features: dict | None = None,
) -> tuple[float, float, str, int]:
    """Ajuste multi-senal delegando a ingame_runner.predict_ingame().

    predict_ingame() tiene el cascading correcto:
      XGBoost PBP (18 features) -> Logistic PBP (9 features) -> Simple Bayesian

    Args:
        p_pregame:    probabilidad pre-partido del local (0.0 a 1.0)
        box_home:     dict de stats del local (de get_live_box_score())
        box_away:     dict de stats del visitante
        period:       periodo completado (1=Q1, 2=Q2, 3=Q3)
        pbp_features: dict con 17 PBP features de LivePBPTracker.get_features().
                      Si se provee, XGBoost/Logistic PBP models se usan.
                      Si es None, cae a Bayesian simple.

    Returns:
        (p_adjusted, delta, explanation, conformal_set_size)
    """
    try:
        from src.sports.nba.predict.ingame_runner import predict_ingame

        result = predict_ingame(
            p_pregame=p_pregame,
            box_home=box_home,
            box_away=box_away,
            period=period,
            pbp_features=pbp_features,
        )

        p_adj = result["p_home"]
        delta = result["delta"]
        set_size = result["conformal_set_size"]
        model_used = result["model_used"]

        conf_label = "ALTA" if set_size == 1 else "BAJA" if set_size == 2 else "N/A"
        explanation = (
            f"[{model_used}] P={p_adj:.3f}, delta={delta:+.3f}, conf={conf_label}"
        )

        # Add top features to explanation if available
        features = result.get("features", {})
        top = []
        for name in ["PBP_LEAD_CHANGES", "PBP_LARGEST_LEAD_HOME",
                      "PBP_HOME_RUNS_MAX", "PBP_MOMENTUM",
                      "SCORE_DIFF_NORM", "DIFF_FG_PCT"]:
            val = features.get(name)
            if val is not None and abs(val) > 0.01:
                top.append(f"{name}={val:+.3f}")
        if top:
            explanation += f", {', '.join(top)}"

        return p_adj, delta, explanation, set_size

    except Exception as e:
        logger.warning("predict_ingame failed (Q%d): %s. Using simple Bayesian.", period, e)

        # Ultimate fallback: simple Bayesian
        home_pts = box_home.get("pts", box_home.get("PTS", 0))
        away_pts = box_away.get("pts", box_away.get("PTS", 0))
        score_diff = int(home_pts) - int(away_pts)

        home_poss = box_home.get("possessions", box_home.get("POSS", 25))
        away_poss = box_away.get("possessions", box_away.get("POSS", 25))
        total_poss = (float(home_poss) + float(away_poss)) / 2.0

        p_adj, expl = bayesian_q1_adjustment(p_pregame, score_diff, total_poss, period=period)
        delta = p_adj - p_pregame
        return p_adj, delta, f"[simple] {expl}", 0


def _match_game_to_prediction(game: dict, pregame_predictions: list[dict]) -> dict | None:
    """Encuentra la prediccion pre-partido que corresponde a un juego live.

    Estrategia de matching (en orden de prioridad):
    1. Tricode/abbreviation match (mas robusto)
    2. Substring match: nombre live contenido en nombre prediccion
    3. City match: ciudad live contenida en nombre prediccion
    """
    live_home_tricode = game.get("home_tricode", "").upper()
    live_away_tricode = game.get("away_tricode", "").upper()
    live_home_name = game.get("home_team", "").lower()
    live_away_name = game.get("away_team", "").lower()
    live_home_city = game.get("home_team_city", "").lower()
    live_away_city = game.get("away_team_city", "").lower()

    # Build abbreviation lookup from canonical names
    from src.core.betting.bet_tracker import NBA_API_ABBREV

    for pred in pregame_predictions:
        pred_home = pred["home_team"]
        pred_away = pred["away_team"]
        pred_home_lower = pred_home.lower()
        pred_away_lower = pred_away.lower()

        # 1. Tricode match (strongest signal)
        if live_home_tricode and live_away_tricode:
            pred_home_tri = NBA_API_ABBREV.get(pred_home, "")
            pred_away_tri = NBA_API_ABBREV.get(pred_away, "")
            if (live_home_tricode == pred_home_tri and
                    live_away_tricode == pred_away_tri):
                return pred

        # 2. Substring/endswith match
        home_match = (
            live_home_name in pred_home_lower or
            pred_home_lower.endswith(live_home_name) or
            live_home_city in pred_home_lower
        )
        away_match = (
            live_away_name in pred_away_lower or
            pred_away_lower.endswith(live_away_name) or
            live_away_city in pred_away_lower
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
    pbp_features: dict | None = None,
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
            p_pre, box["home"], box["away"], period=period,
            pbp_features=pbp_features,
        )
    else:
        total_poss = 25.0 * period
        p_adj, expl = bayesian_q1_adjustment(p_pre, score_diff, total_poss, period=period)
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
    if conf_set == 2:
        print(f"  {Fore.YELLOW}SKIP: conformal incierto (set_size=2) — modelo no confia en esta prediccion{Style.RESET_ALL}")
    elif abs(delta_from_pre) >= edge_threshold:
        if delta_from_pre > 0:
            print(f"  HINT: Si el mercado aun da {away.split()[-1]} > {p_away_adj:.0%} -> edge UNDER {away.split()[-1]}")
        else:
            print(f"  HINT: Si el mercado aun da {home.split()[-1]} > {p_adj:.0%} -> edge UNDER {home.split()[-1]}")

    print(f"  (detalle: {expl})")
    print(f"  ({Fore.CYAN}generado {datetime.now().strftime('%H:%M:%S')} — revisar lineas live{Style.RESET_ALL})")
    print(f"{'=' * 65}")

    # --- Persist to CSV tracker ---
    try:
        from src.core.betting.live_tracker import log_adjustment
        method = "logistic" if "[logistic" in expl else "simple"
        # Pick best-side odds for tracking
        pick_home = p_adj >= 0.5
        ml_odds = pred.get("ml_home_odds") if pick_home else pred.get("ml_away_odds")
        kelly = pred.get("kelly_home") if pick_home else pred.get("kelly_away")
        log_adjustment(
            home_team=home,
            away_team=away,
            quarter=period,
            home_score=home_score,
            away_score=away_score,
            p_pregame=p_pre,
            p_adjusted=p_adj,
            conf_set_size=conf_set,
            method=method,
            ml_odds=ml_odds,
            kelly_pct=kelly,
        )
    except Exception as e:
        logger.debug("Live tracker write failed: %s", e)

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
    print(f"  Cascade: XGBoost PBP -> Logistic PBP -> Bayesian simple")
    print(f"  (actualizacion cada {POLL_INTERVAL_SECONDS}s, Ctrl+C para salir)")
    print(f"{'=' * 65}\n")

    last_period = {}
    processed_periods = {}
    last_p_adjusted = {}
    p_history = {}  # game_id -> {Q1: p_adj, Q2: p_adj, ...}
    pbp_trackers = {}  # game_id -> LivePBPTracker

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

                # Create PBP tracker on first sight of a live game
                if game_id not in pbp_trackers and status == 2:
                    home_tri = game.get("home_tricode", "")
                    away_tri = game.get("away_tricode", "")
                    if home_tri and away_tri:
                        pbp_trackers[game_id] = LivePBPTracker(home_tri, away_tri)
                        logger.info("PBP tracker creado: %s vs %s (%s)",
                                    home_tri, away_tri, game_id)

                # Update PBP tracker with latest play-by-play data
                if game_id in pbp_trackers and status == 2:
                    actions = get_live_play_by_play(game_id)
                    if actions:
                        n_before = len(pbp_trackers[game_id]._plays) if hasattr(pbp_trackers[game_id], '_plays') else 0
                        pbp_trackers[game_id].update(actions)
                        n_after = len(pbp_trackers[game_id]._plays) if hasattr(pbp_trackers[game_id], '_plays') else 0
                        if n_after == n_before and n_before > 0:
                            logger.debug("PBP data unchanged for %s (stale feed?)", game_id)

                prev_period = last_period.get(game_id, 0)
                clock_secs = _parse_clock_seconds(game.get("clock", ""))

                for check_period in [1, 2, 3]:
                    if check_period in processed_periods[game_id]:
                        continue
                    # Trigger: period ya avanzó O clock=0:00 en el periodo actual
                    quarter_done = (
                        period > check_period or
                        (period == check_period and clock_secs == 0.0)
                    )
                    if quarter_done and status in (2, 3):

                        box = get_live_box_score(game_id) if status == 2 else None

                        # Get PBP features for the completed period
                        pbp_feats = None
                        tracker = pbp_trackers.get(game_id)
                        if tracker is not None:
                            pbp_feats = tracker.get_features(period_end=check_period)

                        prev_p = last_p_adjusted.get(game_id)
                        p_adj = _print_period_update(
                            game, pred, box,
                            period=check_period,
                            prev_p=prev_p,
                            pbp_features=pbp_feats,
                        )
                        processed_periods[game_id].add(check_period)
                        last_p_adjusted[game_id] = p_adj
                        hist_key = pred["home_team"]
                        p_history.setdefault(hist_key, {})[check_period] = p_adj

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
                _print_session_summary(pregame_predictions, last_p_adjusted, processed_periods, p_history)
                break

            # Fast polling cuando algun juego está cerca de terminar el Q
            any_close = any(
                0 < _parse_clock_seconds(g.get("clock", "")) <= FAST_POLL_THRESHOLD
                for g in monitored
            ) if monitored else False
            sleep_time = POLL_FAST_SECONDS if any_close else POLL_INTERVAL_SECONDS
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f"\n\n  Live betting interrumpido por el usuario.")

    deinit()  # colorama


def _print_session_summary(
    predictions: list[dict],
    last_p: dict[str, float],
    processed: dict[str, set],
    p_history: dict[str, dict[int, float]] | None = None,
):
    """Imprime resumen final de la sesion live con evolucion de probabilidades."""
    p_history = p_history or {}

    print(f"\n{'=' * 65}")
    print(f"  RESUMEN DE SESION LIVE -- {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'=' * 65}")

    # Header
    print(f"  {'Equipo':<14} {'Pre':>7} {'Q1':>7} {'Q2':>7} {'Q3':>7} {'Final':>7}")
    print(f"  {'-'*14} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    for pred in predictions:
        home = pred["home_team"].split()[-1]
        away = pred["away_team"].split()[-1]
        p_pre = pred["p_pregame"]
        hist = p_history.get(pred["home_team"], {})

        q1 = hist.get(1)
        q2 = hist.get(2)
        q3 = hist.get(3)
        final = q3 or q2 or q1 or p_pre  # last available

        q1_s = f"{q1:.1%}" if q1 else "  -  "
        q2_s = f"{q2:.1%}" if q2 else "  -  "
        q3_s = f"{q3:.1%}" if q3 else "  -  "

        a_q1 = f"{1-q1:.1%}" if q1 else "  -  "
        a_q2 = f"{1-q2:.1%}" if q2 else "  -  "
        a_q3 = f"{1-q3:.1%}" if q3 else "  -  "

        print(f"  {home:<14} {p_pre:>6.1%} {q1_s:>7} {q2_s:>7} {q3_s:>7} {final:>6.1%}")
        print(f"  {away:<14} {1-p_pre:>6.1%} {a_q1:>7} {a_q2:>7} {a_q3:>7} {1-final:>6.1%}")
        print()

    print(f"{'=' * 65}")
    print(f"  Todos los partidos finalizados. Sesion completada.")


# --- Backward compatibility alias ---
def run_live_q1_session(pregame_predictions: list[dict]):
    """Alias de backward compat. Redirige a run_live_session()."""
    run_live_session(pregame_predictions)
