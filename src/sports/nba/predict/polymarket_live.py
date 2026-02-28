"""Live position management para Polymarket.

Integra con live_betting.py para usar probabilidades ajustadas post-quarter
y el WebSocket feed para precios en tiempo real.

Flujo:
  1. Subscribe WebSocket para todos los token IDs de posiciones abiertas
  2. En cada quarter end (polled desde live_betting scoreboard):
     a. Obtener p_adjusted de multi_signal_adjustment()
     b. Obtener current_price del WebSocket
     c. Recomputar edge = p_adjusted - current_price
     d. Aplicar decision tree (take_profit / stop_loss / buy_more / hold / sell)
  3. Post-Q3: hold winners, exit losers without edge

Uso:
  run_polymarket_live_session(pregame_preds, tracker, bankroll=50, dry_run=True)
"""

import logging
import time
from datetime import datetime

from colorama import Fore, Style, init, deinit

from src.core.betting.polymarket_kelly import polymarket_ev, polymarket_kelly
from src.core.betting.polymarket_risk import PolymarketRiskManager
from src.core.betting.polymarket_tracker import PolymarketTracker
from src.sports.nba.predict.live_betting import (
    multi_signal_adjustment,
    bayesian_q1_adjustment,
    POLL_INTERVAL_SECONDS,
    MAX_POLL_HOURS,
)
from src.sports.nba.features.live_game_state import get_live_scoreboard, get_live_box_score, format_clock

logger = logging.getLogger(__name__)


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
    print(f"  POLYMARKET LIVE -- {datetime.now().strftime('%H:%M:%S')} | {mode}")
    print(f"  Monitoring {len(open_positions)} position(s)")
    print(f"  (update every {POLL_INTERVAL_SECONDS}s, Ctrl+C to exit)")
    print(f"{'='*65}\n")

    processed_periods: dict[str, set] = {}
    last_p_adjusted: dict[str, float] = {}
    start_time = time.time()
    max_seconds = MAX_POLL_HOURS * 3600
    poll_count = 0

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed > max_seconds:
                print(f"\nPolymarket Live: timeout after {MAX_POLL_HOURS}h.")
                break

            poll_count += 1

            live_games = get_live_scoreboard()
            if not live_games:
                print(f"  [poll #{poll_count}] No live data. Retrying in {POLL_INTERVAL_SECONDS}s...")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            all_final = True

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

                # Check positions for this game
                game_positions = [
                    p for p in open_positions
                    if p.get("game_key") == game_key and p.get("status") == "OPEN"
                ]
                if not game_positions:
                    continue

                # Process quarter ends
                for check_period in [1, 2, 3]:
                    if (check_period not in processed_periods[game_id] and
                            period > check_period and status in (2, 3)):

                        box = get_live_box_score(game_id) if status == 2 else None

                        if status == 3 and box is None:
                            print(
                                f"  [AVISO] Q{check_period}: partido ya finalizado al iniciar sesion. "
                                f"Usando solo score diff — ajuste bayesiano menos preciso."
                            )

                        p_pre = pred["p_pregame"]
                        if box is not None:
                            p_adj, delta, expl, conf_set = multi_signal_adjustment(
                                p_pre, box["home"], box["away"], period=check_period,
                            )
                        else:
                            score_diff = game["home_score"] - game["away_score"]
                            total_poss = 25.0 * check_period
                            p_adj, expl = bayesian_q1_adjustment(p_pre, score_diff, total_poss)

                        last_p_adjusted[game_id] = p_adj

                        # Evaluate exit signals for each position
                        for pos in game_positions:
                            _evaluate_position(
                                pos, p_adj, check_period,
                                ws_feed, tracker, risk_mgr, bankroll, dry_run,
                            )

                        processed_periods[game_id].add(check_period)

                # Post-Q3 policy
                if period >= 4 or status == 3:
                    p_adj = last_p_adjusted.get(game_id, pred["p_pregame"])
                    for pos in game_positions:
                        _evaluate_post_q3(pos, p_adj, ws_feed, tracker, risk_mgr, dry_run)

            # Status line
            print(f"  [{datetime.now().strftime('%H:%M:%S')} poll #{poll_count}] "
                  f"{len(open_positions)} positions | "
                  f"exposure: ${tracker.get_total_exposure(dry_run):.2f}")

            # Refresh positions
            open_positions = tracker.get_open_positions(dry_run=dry_run)

            if all_final and not open_positions:
                _print_live_summary(tracker, dry_run)
                break

            time.sleep(POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print(f"\n  Polymarket live session interrupted.")

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
