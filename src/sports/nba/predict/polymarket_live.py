"""Live position management para Polymarket (modo trading).

Monitoreo continuo de precios para capturar TP/SL rapidamente.
Dos loops:
  - Fast loop (cada 20s): check TP/SL en todas las posiciones abiertas
  - Scoreboard (cada 60s): detectar quarter-ends para actualizar model_prob

Flujo:
  1. Subscribe WebSocket para todos los token IDs de posiciones abiertas
  2. Fast loop: evaluar TP/SL en cada posicion via WS o REST midpoint
  3. En cada quarter end: actualizar model_prob via multi_signal_adjustment()
  4. Post-Q3: mismas reglas de TP/SL (trading mode)

Uso:
  run_polymarket_live_session(pregame_preds, tracker, bankroll=50, dry_run=True)
"""

import logging
import time
from datetime import datetime

from colorama import Fore, Style, init, deinit

from src.core.betting.polymarket_kelly import polymarket_ev, polymarket_kelly
from src.core.betting.polymarket_risk import PolymarketRiskManager, TAKE_PROFIT_DELTA, STOP_LOSS_DELTA
from src.core.betting.polymarket_tracker import PolymarketTracker
from src.sports.nba.predict.live_betting import (
    multi_signal_adjustment,
    bayesian_q1_adjustment,
    MAX_POLL_HOURS,
)
from src.sports.nba.features.live_game_state import get_live_scoreboard, get_live_box_score, format_clock

logger = logging.getLogger(__name__)

TRADING_POLL_SECONDS = 20    # Frequency for TP/SL checks
SCOREBOARD_INTERVAL = 60     # Frequency for quarter-end model updates


def run_polymarket_live_session(
    pregame_predictions: list[dict],
    tracker: PolymarketTracker,
    bankroll: float = 50.0,
    dry_run: bool = True,
):
    """Loop principal de gestion de posiciones live en Polymarket.

    Args:
        pregame_predictions: lista de dicts con home_team, away_team, p_pregame
        tracker: PolymarketTracker con posiciones abiertas
        bankroll: bankroll en USDC
        dry_run: si True, solo loguea acciones sin ejecutar
    """
    if not pregame_predictions:
        print("Polymarket Live: sin predicciones pre-partido disponibles.")
        return

    init()  # colorama

    open_positions = tracker.get_open_positions(dry_run=dry_run)
    if not open_positions:
        print("Polymarket Live: sin posiciones abiertas para monitorear.")
        deinit()
        return

    risk_mgr = PolymarketRiskManager(bankroll=bankroll)
    mode = "DRY RUN" if dry_run else "LIVE"

    # Start WebSocket feed
    ws_feed = None
    token_ids = [p["token_id"] for p in open_positions if p.get("token_id")]
    if token_ids:
        try:
            from src.sports.nba.providers.polymarket_ws import PolymarketPriceFeed
            ws_feed = PolymarketPriceFeed()
            ws_feed.subscribe(token_ids)
            ws_feed.run_in_thread()
            time.sleep(2)  # Give WS time to connect
        except Exception as e:
            logger.warning("WebSocket feed unavailable: %s. Using REST fallback.", e)

    print(f"\n{'='*65}")
    print(f"  POLYMARKET TRADING -- {datetime.now().strftime('%H:%M:%S')} | {mode}")
    print(f"  Monitoring {len(open_positions)} position(s)")
    print(f"  TP: +${TAKE_PROFIT_DELTA:.2f} | SL: -${STOP_LOSS_DELTA:.2f} | Poll: {TRADING_POLL_SECONDS}s")
    print(f"  (Ctrl+C to exit)")
    print(f"{'='*65}\n")

    processed_periods: dict[str, set] = {}
    last_p_adjusted: dict[str, float] = {}
    # Store model_prob from pregame for positions (used in TP/SL edge calc)
    for pos in open_positions:
        game_key = pos.get("game_key", "")
        for pred in pregame_predictions:
            if f"{pred['home_team']}:{pred['away_team']}" == game_key:
                last_p_adjusted[game_key] = pred["p_pregame"]
                break

    start_time = time.time()
    max_seconds = MAX_POLL_HOURS * 3600
    poll_count = 0
    last_scoreboard_time = 0.0

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed > max_seconds:
                print(f"\nPolymarket Trading: timeout after {MAX_POLL_HOURS}h.")
                break

            poll_count += 1
            now = time.time()
            all_final = True

            # ── Scoreboard check (every SCOREBOARD_INTERVAL) for model updates ──
            if now - last_scoreboard_time >= SCOREBOARD_INTERVAL:
                last_scoreboard_time = now
                live_games = get_live_scoreboard()

                if live_games:
                    for game in live_games:
                        pred = _match_game(game, pregame_predictions)
                        if pred is None:
                            continue

                        game_id = game["game_id"]
                        game_key = f"{pred['home_team']}:{pred['away_team']}"
                        status = game["status"]
                        period = game["period"]

                        if status != 3:
                            all_final = False

                        if game_id not in processed_periods:
                            processed_periods[game_id] = set()

                        # Process quarter ends — update model prob
                        for check_period in [1, 2, 3]:
                            if (check_period not in processed_periods[game_id] and
                                    period > check_period and status in (2, 3)):

                                box = get_live_box_score(game_id) if status == 2 else None
                                p_pre = pred["p_pregame"]

                                if box is not None:
                                    p_adj, delta, expl, conf_set = multi_signal_adjustment(
                                        p_pre, box["home"], box["away"], period=check_period,
                                    )
                                else:
                                    score_diff = game["home_score"] - game["away_score"]
                                    total_poss = 25.0 * check_period
                                    p_adj, expl = bayesian_q1_adjustment(p_pre, score_diff, total_poss)

                                last_p_adjusted[game_key] = p_adj
                                processed_periods[game_id].add(check_period)
                                print(f"  Q{check_period} model update: {game_key} p_adj={p_adj:.1%}")

            # ── Fast TP/SL check (every poll) ──
            open_positions = tracker.get_open_positions(dry_run=dry_run)
            if not open_positions:
                if all_final:
                    _print_live_summary(tracker, dry_run)
                    break
                time.sleep(TRADING_POLL_SECONDS)
                continue

            exits_this_poll = 0
            for pos in open_positions:
                if pos.get("status") != "OPEN":
                    continue

                current_price = _get_current_price(pos, ws_feed)
                if current_price is None:
                    continue

                entry_price = pos["entry_price"]
                price_delta = current_price - entry_price

                # Quick check: only call full evaluate if near TP/SL
                if price_delta >= TAKE_PROFIT_DELTA or price_delta <= -STOP_LOSS_DELTA:
                    # Compute edge with latest model_prob
                    team = pos["team"]
                    game_key = pos.get("game_key", "")
                    home_team = game_key.split(":")[0] if ":" in game_key else ""
                    model_prob = last_p_adjusted.get(game_key, pos.get("model_prob", 0.5))
                    if team != home_team:
                        model_prob = 1.0 - model_prob

                    current_edge = polymarket_ev(model_prob, current_price)
                    signal = risk_mgr.check_exit_signals(entry_price, current_price, current_edge)

                    market_type = pos.get("market_type", "ML")
                    spread_info = f" ({pos.get('spread_line', 0):+.1f})" if market_type == "AH" else ""
                    action_color = Fore.GREEN if signal.action == "TAKE_PROFIT" else Fore.RED

                    if signal.action in ("TAKE_PROFIT", "STOP_LOSS", "SELL_ALL"):
                        print(f"  [{action_color}{signal.action}{Style.RESET_ALL}] "
                              f"{team} [{market_type}{spread_info}] "
                              f"entry=${entry_price:.3f} -> ${current_price:.3f} "
                              f"({price_delta:+.3f})")
                        print(f"    {signal.reason}")

                        if signal.sell_fraction >= 1.0:
                            _execute_exit(pos, current_price, signal, tracker, dry_run)
                        else:
                            _execute_partial_exit(pos, current_price, signal, tracker, dry_run)
                        exits_this_poll += 1

                elif abs(price_delta) >= 0.01:
                    # Edge-based check (no TP/SL but significant movement)
                    team = pos["team"]
                    game_key = pos.get("game_key", "")
                    home_team = game_key.split(":")[0] if ":" in game_key else ""
                    model_prob = last_p_adjusted.get(game_key, pos.get("model_prob", 0.5))
                    if team != home_team:
                        model_prob = 1.0 - model_prob

                    current_edge = polymarket_ev(model_prob, current_price)
                    signal = risk_mgr.check_exit_signals(entry_price, current_price, current_edge)

                    if signal.action in ("SELL_ALL",):
                        market_type = pos.get("market_type", "ML")
                        print(f"  [{Fore.RED}EDGE_EXIT{Style.RESET_ALL}] "
                              f"{team} [{market_type}] edge={current_edge:+.1%}")
                        _execute_exit(pos, current_price, signal, tracker, dry_run)
                        exits_this_poll += 1

            # Status line (every 5th poll to reduce noise)
            if poll_count % 5 == 0 or exits_this_poll > 0:
                n_open = len(tracker.get_open_positions(dry_run=dry_run))
                exposure = tracker.get_total_exposure(dry_run)
                realized = tracker.get_realized_pnl(dry_run)
                print(f"  [{datetime.now().strftime('%H:%M:%S')} #{poll_count}] "
                      f"{n_open} open | exposure: ${exposure:.2f} | "
                      f"realized: ${realized:+.2f}")

            # Check if all done
            open_positions = tracker.get_open_positions(dry_run=dry_run)
            if all_final and not open_positions:
                _print_live_summary(tracker, dry_run)
                break

            time.sleep(TRADING_POLL_SECONDS)

    except KeyboardInterrupt:
        print(f"\n  Polymarket trading session interrupted.")

    finally:
        if ws_feed:
            ws_feed.stop()
        deinit()


def _match_game(game: dict, predictions: list[dict]) -> dict | None:
    """Match live game to pregame prediction."""
    live_home = game.get("home_team", "").lower()
    live_away = game.get("away_team", "").lower()
    live_home_city = game.get("home_team_city", "").lower()

    for pred in predictions:
        pred_home = pred["home_team"].lower()
        pred_away = pred["away_team"].lower()

        home_match = live_home in pred_home or pred_home.endswith(live_home) or live_home_city in pred_home
        away_match = live_away in pred_away or pred_away.endswith(live_away)

        if home_match and away_match:
            return pred
    return None


def _get_current_price(pos: dict, ws_feed) -> float | None:
    """Obtiene precio actual del token via WS o REST fallback."""
    token_id = pos.get("token_id")
    if not token_id:
        return None

    # Try WebSocket first
    if ws_feed:
        price = ws_feed.get_price(token_id)
        if price is not None:
            return price

    # REST fallback
    try:
        from src.sports.nba.providers.polymarket import PolymarketProvider
        provider = PolymarketProvider()
        return provider.get_midpoint(token_id)
    except Exception:
        return None


def _evaluate_position(
    pos: dict,
    p_adjusted: float,
    period: int,
    ws_feed,
    tracker: PolymarketTracker,
    risk_mgr: PolymarketRiskManager,
    bankroll: float,
    dry_run: bool,
):
    """Evalua senales de salida para una posicion despues de un quarter."""
    current_price = _get_current_price(pos, ws_feed)
    if current_price is None:
        current_price = pos.get("current_price", pos["entry_price"])

    # Determinar edge actual
    team = pos["team"]
    game_key = pos["game_key"]
    home_team = game_key.split(":")[0]

    # Si apostamos al home, usamos p_adjusted directo
    # Si apostamos al away, usamos 1 - p_adjusted
    if team == home_team:
        model_prob = p_adjusted
    else:
        model_prob = 1.0 - p_adjusted

    current_edge = polymarket_ev(model_prob, current_price)
    entry_price = pos["entry_price"]

    signal = risk_mgr.check_exit_signals(entry_price, current_price, current_edge)

    # Print update
    edge_color = Fore.GREEN if current_edge > 0.03 else (Fore.YELLOW if current_edge > 0 else Fore.RED)
    action_color = {
        "TAKE_PROFIT": Fore.GREEN,
        "STOP_LOSS": Fore.RED,
        "SELL_ALL": Fore.RED,
        "BUY_MORE": Fore.GREEN,
        "HOLD": Fore.YELLOW,
    }.get(signal.action, Style.RESET_ALL)

    print(f"\n  Q{period} UPDATE: {team} | "
          f"entry=${entry_price:.2f} -> now=${current_price:.2f} | "
          f"edge={edge_color}{current_edge:+.1%}{Style.RESET_ALL} | "
          f"[{action_color}{signal.action}{Style.RESET_ALL}]")
    print(f"    {signal.reason}")

    # Execute signal
    if signal.action in ("STOP_LOSS", "SELL_ALL"):
        _execute_exit(pos, current_price, signal, tracker, dry_run)
    elif signal.action == "TAKE_PROFIT":
        _execute_partial_exit(pos, current_price, signal, tracker, dry_run)
    elif signal.action == "BUY_MORE":
        # Agregar shares a una posicion existente requiere logica adicional de portfolio.
        # Por ahora: loguear la intencion y mantener la posicion (HOLD efectivo).
        tracker.record_order(
            side="BUY",
            game_date=pos["game_date"],
            game_key=pos["game_key"],
            team=pos["team"],
            token_id=pos.get("token_id"),
            price=current_price,
            shares=0,
            cost_usdc=0.0,
            model_prob=model_prob,
            market_price=current_price,
            edge=current_edge,
            signal_reason="BUY_MORE_HOLD",
            dry_run=dry_run,
        )
        print(f"    HOLD (edge fuerte {current_edge:+.1%} — BUY_MORE registrado, posicion mantenida)")


def _evaluate_post_q3(
    pos: dict,
    p_adjusted: float,
    ws_feed,
    tracker: PolymarketTracker,
    risk_mgr: PolymarketRiskManager,
    dry_run: bool,
):
    """Aplica politica post-Q3: hold winners, exit losers without edge."""
    current_price = _get_current_price(pos, ws_feed)
    if current_price is None:
        return

    team = pos["team"]
    game_key = pos["game_key"]
    home_team = game_key.split(":")[0]
    model_prob = p_adjusted if team == home_team else (1.0 - p_adjusted)
    current_edge = polymarket_ev(model_prob, current_price)

    signal = risk_mgr.check_post_q3_policy(pos["entry_price"], current_price, current_edge)

    if signal.action == "SELL_ALL":
        print(f"  POST-Q3: {team} -> {Fore.RED}EXIT{Style.RESET_ALL}: {signal.reason}")
        _execute_exit(pos, current_price, signal, tracker, dry_run)
    else:
        print(f"  POST-Q3: {team} -> {Fore.GREEN}HOLD{Style.RESET_ALL}: {signal.reason}")


def _execute_exit(pos, price, signal, tracker, dry_run):
    """Cierra posicion completa."""
    if not dry_run and not pos.get("token_id"):
        logger.error(
            "No token_id para la posicion %s %s — no se puede ejecutar la venta en live. "
            "Posicion NO cerrada en tracker para evitar P&L incorrecto.",
            pos["game_key"], pos["team"],
        )
        return

    tracker.close_position(
        game_date=pos["game_date"],
        game_key=pos["game_key"],
        team=pos["team"],
        exit_price=price,
        exit_reason=signal.action,
        dry_run=dry_run,
        market_type=pos.get("market_type", "ML"),
    )

    if not dry_run:
        try:
            from src.sports.nba.providers.polymarket_trader import PolymarketTrader
            trader = PolymarketTrader(dry_run=False)
            trader.sell_shares(pos["token_id"], price, pos["shares"])
        except Exception as e:
            logger.error("Exit execution failed: %s", e)

    tracker.record_order(
        side="SELL",
        game_date=pos["game_date"],
        game_key=pos["game_key"],
        team=pos["team"],
        token_id=pos.get("token_id"),
        price=price,
        shares=pos["shares"],
        cost_usdc=round(pos["shares"] * price, 2),
        signal_reason=signal.action,
        dry_run=dry_run,
    )


def _execute_partial_exit(pos, price, signal, tracker, dry_run):
    """Take profit: sell fraction of position."""
    if signal.sell_fraction >= 1.0:
        _execute_exit(pos, price, signal, tracker, dry_run)
        return

    sell_shares = int(pos["shares"] * signal.sell_fraction)
    if sell_shares <= 0:
        return

    if not dry_run and not pos.get("token_id"):
        logger.error(
            "No token_id para la posicion %s %s — no se puede ejecutar take-profit en live.",
            pos["game_key"], pos["team"],
        )
        return

    print(f"    Taking profit: sell {sell_shares}/{pos['shares']} shares @ ${price:.2f}")

    if not dry_run:
        try:
            from src.sports.nba.providers.polymarket_trader import PolymarketTrader
            trader = PolymarketTrader(dry_run=False)
            trader.sell_shares(pos["token_id"], price, sell_shares)
        except Exception as e:
            logger.error("Partial exit execution failed: %s", e)

    tracker.record_order(
        side="SELL",
        game_date=pos["game_date"],
        game_key=pos["game_key"],
        team=pos["team"],
        token_id=pos.get("token_id"),
        price=price,
        shares=sell_shares,
        cost_usdc=round(sell_shares * price, 2),
        signal_reason=f"TAKE_PROFIT ({signal.sell_fraction:.0%})",
        dry_run=dry_run,
    )


def _print_live_summary(tracker: PolymarketTracker, dry_run: bool):
    """Imprime resumen final de la sesion live."""
    print(f"\n{'='*65}")
    print(f"  POLYMARKET LIVE SESSION COMPLETE -- {datetime.now().strftime('%H:%M:%S')}")
    tracker.print_portfolio_report(dry_run=dry_run)
    print(f"{'='*65}")
