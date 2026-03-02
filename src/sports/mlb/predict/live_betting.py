"""MLB Live Betting: ajuste Bayesiano inning-by-inning usando modelo Poisson.

CONCEPTO CLAVE — Poisson run model para beisbol:
==================================================
La distribucion natural de carreras por inning en MLB es Poisson:
  - Promedio ~0.5 carreras/inning/equipo (4.5 runs / 9 innings)
  - P(0 runs) ~ 74%, P(1 run) ~ 18%, P(2+) ~ 8%

A diferencia del NBA live (score_diff / sqrt(possessions)), en MLB:
  1. El score diferencial es menos informativo (1 carrera = mucho)
  2. El pitcher abridor vs bullpen es el factor critico
  3. Los outs restantes decrecen linealmente (no hay posesiones)

Formula:
    logit_adj = logit(P_pregame) + B(inning) * (run_diff / sqrt(outs_played))
    + pitcher_change_adjustment
    P_adjusted = sigmoid(logit_adj)

BETA es adaptativo por inning:
  - Innings 1-3: 0.20 (SP dominan, score poco predictivo)
  - Innings 4-5: 0.35 (SP fatigue, score mas relevante)
  - Innings 6-7: 0.50 (bullpen transition)
  - Innings 8-9: 0.70 (cerrador, score muy predictivo)

Pitcher change penalty:
  - Si el SP sale antes del 5to, delta_logit -= 0.15 (signo de problemas)
  - Si bullpen entra en 6to+, no penalty (normal usage)

Uso:
    from src.sports.mlb.predict.live_betting import run_live_session
    run_live_session(pregame_results)  # monitorea innings 1-9+
"""

import logging
import math
import os
import time
from datetime import datetime

from colorama import Fore, Style, init, deinit

from src.sports.mlb.providers.mlb_stats_api import get_live_scoreboard, get_live_game_feed
from src.sports.mlb.config_mlb import MLB_ABBREV, MLB_ABBREV_TO_NAME

logger = logging.getLogger(__name__)

# Beta por grupo de innings (cuanto mas avanzado, mas predictivo el score)
BETA_BY_INNING = {
    1: 0.20, 2: 0.20, 3: 0.20,   # early innings: SP dominant
    4: 0.35, 5: 0.35,             # mid innings: SP fatigue
    6: 0.50, 7: 0.50,             # late innings: bullpen
    8: 0.70, 9: 0.70,             # endgame: closer
}
BETA_DEFAULT = 0.50

# Poisson run rate per inning per team (league average ~0.5)
POISSON_LAMBDA = 0.50

# Pitcher change adjustment (logit space)
SP_EARLY_EXIT_PENALTY = 0.15  # SP yanked before 5th
BULLPEN_ADVANTAGE = 0.05       # known closer entering in 9th

# Polling intervals (seconds)
POLL_INTERVAL_SECONDS = 180  # 3 minutes (half-innings take ~15-20 min)
POLL_FAST_SECONDS = 60       # 1 minute (late innings, close games)
FAST_POLL_THRESHOLD_INNING = 7  # activate fast polling from 7th inning onward
FAST_POLL_SCORE_DIFF = 2        # fast polling if score diff <= 2 runs

# Max poll time
MAX_POLL_HOURS = int(os.environ.get("MLB_LIVE_TIMEOUT_HOURS", 6))


# ---------------------------------------------------------------------------
# Core Bayesian adjustment
# ---------------------------------------------------------------------------

def bayesian_inning_adjustment(
    p_pregame: float,
    run_diff: int,
    inning: int,
    outs_played: int = None,
    sp_changed_home: bool = False,
    sp_changed_away: bool = False,
    sp_early_exit: bool = False,
) -> tuple[float, str]:
    """Ajusta la probabilidad de victoria del local usando el score y estado del juego.

    Formula:
        logit_adj = logit(P_pregame) + B(inning) * (run_diff / sqrt(outs_played))
                    + pitcher_adjustment
        P_adjusted = sigmoid(logit_adj)

    Args:
        p_pregame:        probabilidad pre-partido del equipo local (0.0 a 1.0)
        run_diff:         home_runs - away_runs
        inning:           inning actual (1-9+)
        outs_played:      total de outs jugados (3 * completed_half_innings).
                          Si None, se estima como inning * 6.
        sp_changed_home:  True si el SP del home ya fue reemplazado
        sp_changed_away:  True si el SP del away ya fue reemplazado
        sp_early_exit:    True si algun SP salio antes del 5to inning

    Returns:
        (p_adjusted, explanation_string)
    """
    beta = BETA_BY_INNING.get(min(inning, 9), BETA_DEFAULT)

    p_clamped = max(0.001, min(0.999, p_pregame))
    logit_pre = math.log(p_clamped / (1.0 - p_clamped))

    # Outs played: cada half-inning = 3 outs, full inning = 6 outs
    if outs_played is None:
        outs_played = max(inning, 1) * 6
    outs_played = max(1.0, float(outs_played))

    # Signal normalizada por outs (analogo a score_diff/sqrt(poss) en NBA)
    normalized_signal = run_diff / math.sqrt(outs_played)

    delta_logit = beta * normalized_signal

    # Pitcher change adjustments
    pitcher_adj = 0.0
    pitcher_note = ""
    if sp_early_exit:
        # SP sacado temprano = mal signo para ese equipo
        # Determinar cual SP salio temprano
        if sp_changed_home and inning <= 5:
            pitcher_adj -= SP_EARLY_EXIT_PENALTY
            pitcher_note = " SP-HOME-exit-early"
        elif sp_changed_away and inning <= 5:
            pitcher_adj += SP_EARLY_EXIT_PENALTY
            pitcher_note = " SP-AWAY-exit-early"

    delta_logit += pitcher_adj

    logit_adj = logit_pre + delta_logit
    p_adjusted = 1.0 / (1.0 + math.exp(-logit_adj))

    explanation = (
        f"run_diff={run_diff:+d}, inn={inning}, outs~{outs_played:.0f}, "
        f"B={beta:.2f}, signal={normalized_signal:+.2f}, "
        f"delta={delta_logit:+.3f}{pitcher_note}"
    )

    return p_adjusted, explanation


# ---------------------------------------------------------------------------
# Game matching
# ---------------------------------------------------------------------------

def _match_game_to_prediction(game: dict, pregame_predictions: list[dict]) -> dict | None:
    """Encuentra la prediccion pre-partido que corresponde a un juego live.

    Matching por nombre de equipo (substring / exact match).
    """
    live_home = (game.get("home_team") or "").lower()
    live_away = (game.get("away_team") or "").lower()

    if not live_home or not live_away:
        return None

    for pred in pregame_predictions:
        pred_home = pred.get("home_team", "").lower()
        pred_away = pred.get("away_team", "").lower()

        # Exact match
        if live_home == pred_home and live_away == pred_away:
            return pred

        # Substring match (e.g., "Red Sox" in "Boston Red Sox")
        home_match = (
            live_home in pred_home or pred_home in live_home
            or live_home.split()[-1] in pred_home
        )
        away_match = (
            live_away in pred_away or pred_away in live_away
            or live_away.split()[-1] in pred_away
        )

        if home_match and away_match:
            return pred

    return None


# ---------------------------------------------------------------------------
# Console output per-inning update
# ---------------------------------------------------------------------------

def _print_inning_update(
    game: dict,
    pred: dict,
    inning: int,
    prev_p: float | None = None,
    sp_changed_home: bool = False,
    sp_changed_away: bool = False,
    sp_early_exit: bool = False,
) -> float:
    """Imprime el update de un inning completado."""
    home = pred["home_team"]
    away = pred["away_team"]
    p_pre = pred["p_pregame"]

    p_ref = prev_p if prev_p is not None else p_pre

    home_score = game.get("home_score", 0) or 0
    away_score = game.get("away_score", 0) or 0
    run_diff = home_score - away_score

    # Estimar outs jugados (top + bottom de cada inning completado)
    outs_played = inning * 6

    p_adj, expl = bayesian_inning_adjustment(
        p_pre, run_diff, inning,
        outs_played=outs_played,
        sp_changed_home=sp_changed_home,
        sp_changed_away=sp_changed_away,
        sp_early_exit=sp_early_exit,
    )

    p_away_pre = 1.0 - p_pre
    p_away_adj = 1.0 - p_adj
    delta_from_prev = p_adj - p_ref

    inn_label = f"INN {inning}"
    if run_diff > 0:
        leader = f"+{run_diff} {home.split()[-1]}"
        delta_color = Fore.GREEN
    elif run_diff < 0:
        leader = f"+{abs(run_diff)} {away.split()[-1]}"
        delta_color = Fore.RED
    else:
        leader = "TIED"
        delta_color = Style.RESET_ALL

    # Pitcher info
    pitcher_info = ""
    if sp_changed_home:
        pitcher_info += f"  {Fore.YELLOW}[{home.split()[-1]} bullpen]{Style.RESET_ALL}"
    if sp_changed_away:
        pitcher_info += f"  {Fore.YELLOW}[{away.split()[-1]} bullpen]{Style.RESET_ALL}"

    print(f"\n{'=' * 65}")
    print(f"  UPDATE -- {inn_label} COMPLETE: {home} vs {away}{pitcher_info}")
    print(f"  Score: {home_score}-{away_score}  ({leader})")
    print(f"{'-' * 65}")
    print(f"  PRE-GAME:    {home.split()[-1]} {p_pre:.1%}  <->  {away.split()[-1]} {p_away_pre:.1%}")

    if prev_p is not None and inning > 1:
        p_away_ref = 1.0 - p_ref
        print(f"  PRE-INN{inning}:   {home.split()[-1]} {p_ref:.1%}  <->  {away.split()[-1]} {p_away_ref:.1%}")

    print(
        f"  LIVE INN{inning}:  {home.split()[-1]} {p_adj:.1%}  <->  {away.split()[-1]} {p_away_adj:.1%}  "
        f"({delta_color}{delta_from_prev:+.1%} {home.split()[-1]}{Style.RESET_ALL})"
    )
    print(f"{'-' * 65}")

    # Edge hint
    edge_threshold = 0.04
    delta_from_pre = p_adj - p_pre
    if abs(delta_from_pre) >= edge_threshold:
        if delta_from_pre > 0:
            print(f"  HINT: Si el mercado aun da {away.split()[-1]} > {p_away_adj:.0%} -> edge UNDER {away.split()[-1]}")
        else:
            print(f"  HINT: Si el mercado aun da {home.split()[-1]} > {p_adj:.0%} -> edge UNDER {home.split()[-1]}")

    print(f"  (detalle: {expl})")
    print(f"  ({Fore.CYAN}generado {datetime.now().strftime('%H:%M:%S')} -- revisar lineas live{Style.RESET_ALL})")
    print(f"{'=' * 65}")

    return p_adj


# ---------------------------------------------------------------------------
# Session summary
# ---------------------------------------------------------------------------

def _print_session_summary(
    predictions: list[dict],
    last_p: dict,
    p_history: dict,
):
    """Imprime resumen final de la sesion live con evolucion por inning."""
    print(f"\n{'=' * 75}")
    print(f"  MLB LIVE SESSION SUMMARY -- {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'=' * 75}")

    header = f"  {'Equipo':<20} {'Pre':>6}"
    for inn in range(1, 10):
        header += f" {'I'+str(inn):>6}"
    header += f" {'Final':>6}"
    print(header)
    print(f"  {'-'*20} {'-'*6}" + f" {'-'*6}" * 9 + f" {'-'*6}")

    for pred in predictions:
        home = pred["home_team"]
        away = pred["away_team"]
        p_pre = pred["p_pregame"]
        hist = p_history.get(home, {})

        row_h = f"  {home.split()[-1]:<20} {p_pre:>5.1%}"
        row_a = f"  {away.split()[-1]:<20} {1-p_pre:>5.1%}"

        final_p = p_pre
        for inn in range(1, 10):
            p = hist.get(inn)
            if p is not None:
                row_h += f" {p:>5.1%}"
                row_a += f" {1-p:>5.1%}"
                final_p = p
            else:
                row_h += f"   {'--':>4}"
                row_a += f"   {'--':>4}"

        row_h += f" {final_p:>5.1%}"
        row_a += f" {1-final_p:>5.1%}"

        print(row_h)
        print(row_a)
        print()

    print(f"{'=' * 75}")
    print(f"  Todos los partidos finalizados. Sesion completada.")


# ---------------------------------------------------------------------------
# Main live session loop
# ---------------------------------------------------------------------------

def run_live_session(pregame_results: list[dict], bot=None):
    """Loop principal de live betting MLB: monitorea innings 1-9+ y ajusta probabilidades.

    Args:
        pregame_results: lista de dicts con claves:
          - home_team (str): ej "Boston Red Sox"
          - away_team (str): ej "New York Yankees"
          - p_pregame (float): probabilidad del local (0.0 a 1.0)
        bot: Telegram bot instance (opcional) para enviar updates.
    """
    if not pregame_results:
        print("MLB LiveBetting: sin predicciones pre-partido disponibles.")
        return

    init()  # colorama

    print(f"\n{'=' * 65}")
    print(f"  MLB LIVE BETTING -- {datetime.now().strftime('%H:%M:%S')}")
    print(f"  Monitoreando {len(pregame_results)} partido(s)")
    print(f"  Modelo: Bayesian Poisson (run_diff / sqrt(outs) + pitcher adj)")
    print(f"  Polling cada {POLL_INTERVAL_SECONDS}s (rapido: {POLL_FAST_SECONDS}s desde inn {FAST_POLL_THRESHOLD_INNING})")
    print(f"  Timeout: {MAX_POLL_HOURS}h (Ctrl+C para salir)")
    print(f"{'=' * 65}\n")

    # State tracking per game
    last_inning = {}                # game_pk -> last seen inning
    processed_innings = {}          # game_pk -> set of completed innings
    last_p_adjusted = {}            # game_pk -> last adjusted probability
    p_history = {}                  # home_team -> {inning: p_adj}
    sp_initial = {}                 # game_pk -> {"home_pitcher": id, "away_pitcher": id}
    sp_changed = {}                 # game_pk -> {"home": bool, "away": bool}

    start_time = time.time()
    max_seconds = MAX_POLL_HOURS * 3600
    poll_count = 0

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed > max_seconds:
                print(f"\nMLB LiveBetting: timeout despues de {MAX_POLL_HOURS}h. Saliendo.")
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
                pred = _match_game_to_prediction(game, pregame_results)
                if pred is None:
                    continue

                game_pk = game["game_pk"]
                status = game.get("status", "")
                inning = game.get("inning") or 0
                inning_half = game.get("inning_half", "")

                monitored.append(game)

                is_final = "Final" in status or "Completed" in status or "Game Over" in status
                if not is_final:
                    all_final = False

                if game_pk not in processed_innings:
                    processed_innings[game_pk] = set()

                if game_pk not in sp_changed:
                    sp_changed[game_pk] = {"home": False, "away": False}

                # Track starting pitcher changes via live feed
                if inning >= 1 and game_pk not in sp_initial:
                    try:
                        feed = get_live_game_feed(game_pk)
                        if feed and feed.get("current_pitcher"):
                            sp_initial[game_pk] = {
                                "current_pitcher_name": feed["current_pitcher"].get("name"),
                                "current_pitcher_id": feed["current_pitcher"].get("id"),
                            }
                    except Exception as e:
                        logger.debug("Could not get initial pitcher for %s: %s", game_pk, e)

                # Check for pitcher changes on subsequent polls
                if game_pk in sp_initial and inning >= 2:
                    try:
                        feed = get_live_game_feed(game_pk)
                        if feed:
                            curr = feed.get("current_pitcher", {})
                            init_name = sp_initial[game_pk].get("current_pitcher_name")
                            if curr.get("name") and curr["name"] != init_name:
                                # Pitcher changed — determine if home or away based on inning_half
                                if inning_half == "Top":
                                    # Top of inning: home team is pitching
                                    if not sp_changed[game_pk]["home"]:
                                        sp_changed[game_pk]["home"] = True
                                        logger.info("SP change detected: %s (home) -> bullpen at INN %d",
                                                     pred["home_team"], inning)
                                else:
                                    # Bottom: away team is pitching
                                    if not sp_changed[game_pk]["away"]:
                                        sp_changed[game_pk]["away"] = True
                                        logger.info("SP change detected: %s (away) -> bullpen at INN %d",
                                                     pred["away_team"], inning)
                    except Exception as e:
                        logger.debug("Pitcher check failed for %s: %s", game_pk, e)

                # Detect completed innings and trigger updates
                for check_inning in range(1, 10):
                    if check_inning in processed_innings[game_pk]:
                        continue

                    # Inning is complete if we're beyond it
                    inning_done = (
                        inning > check_inning
                        or (is_final and check_inning <= inning)
                    )

                    if inning_done:
                        sp_h = sp_changed[game_pk]["home"]
                        sp_a = sp_changed[game_pk]["away"]
                        sp_early = (sp_h and check_inning <= 5) or (sp_a and check_inning <= 5)

                        prev_p = last_p_adjusted.get(game_pk)
                        p_adj = _print_inning_update(
                            game, pred,
                            inning=check_inning,
                            prev_p=prev_p,
                            sp_changed_home=sp_h,
                            sp_changed_away=sp_a,
                            sp_early_exit=sp_early,
                        )
                        processed_innings[game_pk].add(check_inning)
                        last_p_adjusted[game_pk] = p_adj
                        p_history.setdefault(pred["home_team"], {})[check_inning] = p_adj

                        # Send Telegram notification if bot is available
                        if bot is not None:
                            try:
                                _send_telegram_update(bot, pred, game, p_adj, check_inning)
                            except Exception:
                                pass

                last_inning[game_pk] = inning

            # Status line
            if monitored:
                print(f"  [{datetime.now().strftime('%H:%M:%S')} poll #{poll_count}]  ", end="")
                status_parts = []
                for game in monitored:
                    pred = _match_game_to_prediction(game, pregame_results)
                    h = pred["home_team"].split()[-1]
                    a = pred["away_team"].split()[-1]
                    hs = game.get("home_score", 0) or 0
                    aws = game.get("away_score", 0) or 0
                    inn = game.get("inning", 0) or 0
                    half = game.get("inning_half", "")
                    st = game.get("status", "")

                    if "Final" in st or "Completed" in st:
                        status_parts.append(f"{h} {hs}-{aws} {a}: FINAL")
                    elif inn > 0:
                        half_label = "T" if "Top" in half else "B" if "Bot" in half else ""
                        status_parts.append(f"{h} {hs}-{aws} {a} ({half_label}{inn})")
                    else:
                        status_parts.append(f"{h} vs {a}: scheduled")

                print(" | ".join(status_parts))
            else:
                print(f"  [poll #{poll_count}] Sin juegos monitoreados aun (esperando inicio)...")

            # Check if all done
            if all_final and monitored and len(monitored) == len(pregame_results):
                _print_session_summary(pregame_results, last_p_adjusted, p_history)
                break

            # Adaptive polling: fast in late innings with close scores
            any_fast = False
            for game in monitored:
                inn = game.get("inning", 0) or 0
                hs = game.get("home_score", 0) or 0
                aws = game.get("away_score", 0) or 0
                diff = abs(hs - aws)
                if inn >= FAST_POLL_THRESHOLD_INNING and diff <= FAST_POLL_SCORE_DIFF:
                    any_fast = True
                    break

            sleep_time = POLL_FAST_SECONDS if any_fast else POLL_INTERVAL_SECONDS
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f"\n\n  MLB Live betting interrumpido por el usuario.")
        if p_history:
            _print_session_summary(pregame_results, last_p_adjusted, p_history)

    deinit()  # colorama


def _send_telegram_update(bot, pred, game, p_adj, inning):
    """Envia update a Telegram (si bot esta configurado)."""
    home = pred["home_team"]
    away = pred["away_team"]
    hs = game.get("home_score", 0) or 0
    aws = game.get("away_score", 0) or 0
    msg = (
        f"MLB INN{inning}: {home} {hs}-{aws} {away}\n"
        f"P({home.split()[-1]}) = {p_adj:.1%}"
    )
    bot.send_message(msg)
