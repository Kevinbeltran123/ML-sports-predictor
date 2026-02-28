"""Polymarket signal generator: edge detection + BUY/SELL/HOLD signals.

Recibe predicciones del ensemble + precios de Polymarket y genera senales
de trading. Se ejecuta despues de la seccion de sportsbook picks.

Flujo:
  ensemble_runner() -> predictions
  PolymarketProvider().get_nba_markets() -> markets
  generate_polymarket_signals(predictions, markets, ...) -> signals
  print_polymarket_output(signals) -> console output
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

from colorama import Fore, Style

from src.core.betting.polymarket_kelly import (
    polymarket_ev,
    polymarket_ev_per_100,
    polymarket_kelly,
    shares_from_kelly,
    cost_basis,
    share_price_to_american_odds,
)
from src.core.betting.polymarket_risk import (
    PolymarketRiskManager,
    MIN_EDGE,
    MIN_LIQUIDITY_USDC,
)
from src.core.betting.polymarket_tracker import PolymarketTracker

logger = logging.getLogger(__name__)


@dataclass
class PolymarketSignal:
    """Senal de trading para un mercado de Polymarket."""
    game_key: str
    home_team: str
    away_team: str
    pick_team: str            # equipo al que apostar
    pick_side: str            # "home" o "away"
    action: str               # BUY / SELL / HOLD / SKIP
    model_prob: float         # P(pick_team gana) del ensemble
    market_price: float       # precio del share en PM
    edge: float               # model_prob - market_price
    ev_per_100: float         # EV normalizado a $100
    kelly_pct: float          # % del bankroll (DRO-Kelly)
    shares: int               # shares a comprar
    cost_usdc: float          # costo total
    liquidity: float          # liquidez del mercado
    sigma: float = 0.05       # epsilon del modelo
    token_id: str = None
    american_odds: int = 0    # equivalente americano del share price
    skip_reason: str = ""


def generate_polymarket_signals(
    predictions: list[dict],
    markets: dict,
    bankroll: float = 50.0,
    sigma_default: float = 0.05,
    dry_run: bool = True,
) -> list[PolymarketSignal]:
    """Genera senales de trading comparando ensemble vs Polymarket prices.

    Args:
        predictions: lista de dicts del ensemble_runner (prob_home, prob_away, etc.)
        markets: dict de PolymarketProvider.match_markets_to_games()
        bankroll: bankroll en USDC
        sigma_default: epsilon default si no hay sigma per-game
        dry_run: si True, lee posiciones dry-run para el risk check; si False, las reales

    Returns:
        lista de PolymarketSignal, rankeada por edge descendente
    """
    risk_mgr = PolymarketRiskManager(bankroll=bankroll)
    tracker = PolymarketTracker()
    open_positions = tracker.get_open_positions(dry_run=dry_run)

    signals = []
    game_date = datetime.now().strftime("%Y-%m-%d")

    for pred in predictions:
        game_key = f"{pred['home_team']}:{pred['away_team']}"

        if game_key not in markets:
            continue

        mkt = markets[game_key]
        sigma = pred.get("sigma", sigma_default)

        # Evaluar home side
        home_signal = _evaluate_side(
            pred, mkt, "home", sigma, bankroll, risk_mgr, open_positions, game_date,
        )
        # Evaluar away side
        away_signal = _evaluate_side(
            pred, mkt, "away", sigma, bankroll, risk_mgr, open_positions, game_date,
        )

        # Elegir el lado con mayor edge
        best = home_signal if home_signal.edge > away_signal.edge else away_signal

        signals.append(best)

    # Rankear por edge descendente
    signals.sort(key=lambda s: s.edge, reverse=True)
    return signals


def _evaluate_side(
    pred: dict,
    mkt: dict,
    side: str,
    sigma: float,
    bankroll: float,
    risk_mgr: PolymarketRiskManager,
    open_positions: list[dict],
    game_date: str,
) -> PolymarketSignal:
    """Evalua un lado (home/away) de un mercado."""
    home_team = pred["home_team"]
    away_team = pred["away_team"]
    game_key = f"{home_team}:{away_team}"

    if side == "home":
        model_prob = pred["prob_home"]
        price = mkt["home_price"]
        team = home_team
        token_id = mkt.get("home_token_id")
    else:
        model_prob = pred["prob_away"]
        price = mkt["away_price"]
        team = away_team
        token_id = mkt.get("away_token_id")

    edge = polymarket_ev(model_prob, price)
    ev_100 = polymarket_ev_per_100(model_prob, price)
    kelly_result = polymarket_kelly(model_prob, price, epsilon=sigma)
    kelly_pct = kelly_result["kelly_pct"]
    shares = shares_from_kelly(kelly_pct, bankroll, price)
    cost = cost_basis(shares, price)
    liquidity = mkt.get("liquidity", 0)
    am_odds = share_price_to_american_odds(price)

    # Determinar action
    # edge y liquidity se validan en validate_new_position (unica fuente de verdad)
    action = "SKIP"
    skip_reason = ""

    if kelly_pct <= 0:
        skip_reason = "no robust edge (Kelly=0)"
    elif shares <= 0:
        skip_reason = "position too small"
    else:
        risk_check = risk_mgr.validate_new_position(
            cost_usdc=cost,
            liquidity=liquidity,
            edge=edge,
            open_positions=open_positions,
            game_date=game_date,
        )
        if risk_check.allowed:
            action = "BUY"
        else:
            skip_reason = risk_check.reason

    return PolymarketSignal(
        game_key=game_key,
        home_team=home_team,
        away_team=away_team,
        pick_team=team,
        pick_side=side,
        action=action,
        model_prob=model_prob,
        market_price=price,
        edge=edge,
        ev_per_100=ev_100,
        kelly_pct=kelly_pct,
        shares=shares,
        cost_usdc=cost,
        liquidity=liquidity,
        sigma=sigma,
        token_id=token_id,
        american_odds=am_odds,
        skip_reason=skip_reason,
    )


def log_signals_to_tracker(
    signals: list[PolymarketSignal],
    tracker: PolymarketTracker,
    dry_run: bool = True,
):
    """Registra ordenes y abre posiciones en el tracker."""
    game_date = datetime.now().strftime("%Y-%m-%d")

    for sig in signals:
        # Log order intent (siempre, incluido SKIPs)
        tracker.record_order(
            side="BUY" if sig.action == "BUY" else "SKIP",
            game_date=game_date,
            game_key=sig.game_key,
            team=sig.pick_team,
            token_id=sig.token_id,
            price=sig.market_price,
            shares=sig.shares if sig.action == "BUY" else 0,
            cost_usdc=sig.cost_usdc if sig.action == "BUY" else 0,
            model_prob=sig.model_prob,
            market_price=sig.market_price,
            edge=sig.edge,
            kelly_pct=sig.kelly_pct,
            signal_reason=sig.skip_reason if sig.action != "BUY" else "edge",
            dry_run=dry_run,
        )

        # Open position for BUY signals
        if sig.action == "BUY":
            tracker.open_position(
                game_date=game_date,
                game_key=sig.game_key,
                team=sig.pick_team,
                token_id=sig.token_id or "",
                entry_price=sig.market_price,
                shares=sig.shares,
                model_prob=sig.model_prob,
                dry_run=dry_run,
            )


def print_polymarket_output(
    signals: list[PolymarketSignal],
    bankroll: float = 50.0,
    dry_run: bool = True,
):
    """Imprime senales de Polymarket en formato compacto.

    Se imprime despues de la seccion de sportsbook picks.
    """
    if not signals:
        return

    n_matched = len(signals)
    n_buy = sum(1 for s in signals if s.action == "BUY")
    total_exposure = sum(s.cost_usdc for s in signals if s.action == "BUY")
    mode = "DRY RUN" if dry_run else "LIVE"

    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"  POLYMARKET SIGNALS ({n_matched} matched, ranked by edge)")
    print(f"{'='*60}{Style.RESET_ALL}")

    for rank, sig in enumerate(signals, 1):
        # Pick line
        prob_pct = sig.model_prob * 100
        loser = sig.away_team if sig.pick_side == "home" else sig.home_team

        if sig.action == "BUY":
            tag = f"{Fore.GREEN}BUY{Style.RESET_ALL}"
        elif sig.action == "HOLD":
            tag = f"{Fore.YELLOW}HOLD{Style.RESET_ALL}"
        elif sig.action == "SELL":
            tag = f"{Fore.RED}SELL{Style.RESET_ALL}"
        else:
            tag = f"{Fore.YELLOW}SKIP{Style.RESET_ALL}"

        print(f"  {Fore.CYAN}#{rank}{Style.RESET_ALL}  "
              f"{sig.pick_team} ({prob_pct:.1f}%) vs {loser}  [{tag}]")

        if sig.action == "BUY":
            edge_color = Fore.GREEN if sig.edge > 0.05 else Fore.YELLOW
            print(f"      PM  {sig.pick_team} YES: ${sig.market_price:.2f} "
                  f"-> edge={edge_color}{sig.edge:+.1%}{Style.RESET_ALL}  "
                  f"EV={sig.ev_per_100:+.1f}  "
                  f"Kelly={sig.kelly_pct:.2f}%")
            print(f"          Buy {sig.shares} shares @ ${sig.market_price:.2f} "
                  f"= ${sig.cost_usdc:.2f}  | Liq: ${sig.liquidity:,.0f}")
        else:
            print(f"      PM  {sig.skip_reason}")

    # Footer
    exposure_pct = (total_exposure / bankroll * 100) if bankroll > 0 else 0
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"  Portfolio: {n_buy} new | "
          f"Exposure: ${total_exposure:.2f} ({exposure_pct:.1f}%) | "
          f"Mode: {mode}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
