"""Risk management para Polymarket NBA trading.

Modo trading: capturar movimiento de precio (CLV), no esperar settlement.
  - Posicion individual max: 10% ($5 en bankroll $50)
  - Exposicion total max: 40% ($20)
  - Exposicion same-day max: 30% ($15)
  - Posiciones abiertas max: 8
  - Take-profit: +$0.05 per share (vender 100%)
  - Stop-loss: -$0.03 per share (vender 100%)
  - Liquidez minima: $200
  - Edge minimo: 3% (model_prob - market_price >= 0.03)
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── Constantes de riesgo ────────────────────────────────────────────

MAX_SINGLE_POSITION_PCT = 10.0    # % del bankroll por posicion
MAX_TOTAL_EXPOSURE_PCT = 40.0     # % del bankroll en todas las posiciones
MAX_SAME_DAY_EXPOSURE_PCT = 30.0  # % del bankroll en posiciones del mismo dia
MAX_OPEN_POSITIONS = 8
TAKE_PROFIT_DELTA = 0.05          # Vender cuando precio sube $0.05 desde entry
STOP_LOSS_DELTA = 0.03            # Vender cuando precio baja $0.03 desde entry
TAKE_PROFIT_SELL_FRACTION = 1.00  # Vender 100% en take-profit (trading mode)
MIN_LIQUIDITY_USDC = 200.0       # Liquidez minima del mercado
MIN_EDGE = 0.03                   # Edge minimo para abrir posicion
MIN_POSITION_USDC = 1.0           # Posicion minima viable (gas fees)


@dataclass
class RiskCheckResult:
    """Resultado de validacion de riesgo."""
    allowed: bool
    reason: str
    max_usdc: float = 0.0  # Maximo que se puede invertir despues de limits


@dataclass
class ExitSignal:
    """Senal de salida para una posicion."""
    action: str       # STOP_LOSS, TAKE_PROFIT, EDGE_GONE, EDGE_REVERSED, HOLD, BUY_MORE
    sell_fraction: float  # 0.0 a 1.0 (que fraccion vender)
    reason: str


class PolymarketRiskManager:
    """Valida limites de riesgo antes de abrir posiciones.

    Uso:
        risk = PolymarketRiskManager(bankroll=50.0)
        result = risk.validate_new_position(
            cost_usdc=5.0,
            liquidity=500.0,
            edge=0.05,
            tracker=tracker,
            game_date="2026-02-27",
        )
        if result.allowed:
            # proceder con la orden
    """

    def __init__(self, bankroll: float = 50.0):
        self.bankroll = bankroll

    def validate_new_position(
        self,
        cost_usdc: float,
        liquidity: float,
        edge: float,
        open_positions: list[dict] = None,
        game_date: str = None,
    ) -> RiskCheckResult:
        """Valida todos los limites antes de abrir una posicion.

        Args:
            cost_usdc: costo de la nueva posicion en USDC
            liquidity: liquidez del mercado en USDC
            edge: model_prob - market_price
            open_positions: lista de posiciones abiertas (dicts con cost_basis, game_date)
            game_date: fecha del partido

        Returns:
            RiskCheckResult con allowed=True/False y razon
        """
        if open_positions is None:
            open_positions = []

        # 1. Edge minimo
        if edge < MIN_EDGE:
            return RiskCheckResult(
                allowed=False,
                reason=f"Edge {edge:.1%} < min {MIN_EDGE:.0%}",
            )

        # 2. Liquidez minima
        if liquidity < MIN_LIQUIDITY_USDC:
            return RiskCheckResult(
                allowed=False,
                reason=f"Liquidity ${liquidity:.0f} < min ${MIN_LIQUIDITY_USDC:.0f}",
            )

        # 3. Posicion individual max
        max_single = self.bankroll * (MAX_SINGLE_POSITION_PCT / 100.0)
        if cost_usdc > max_single:
            return RiskCheckResult(
                allowed=False,
                reason=f"Position ${cost_usdc:.2f} > max ${max_single:.2f} ({MAX_SINGLE_POSITION_PCT}%)",
                max_usdc=max_single,
            )

        # 4. Posicion minima
        if cost_usdc < MIN_POSITION_USDC:
            return RiskCheckResult(
                allowed=False,
                reason=f"Position ${cost_usdc:.2f} < min ${MIN_POSITION_USDC:.2f}",
            )

        # 5. Max posiciones abiertas
        n_open = len([p for p in open_positions if p.get("status") == "OPEN"])
        if n_open >= MAX_OPEN_POSITIONS:
            return RiskCheckResult(
                allowed=False,
                reason=f"Open positions {n_open} >= max {MAX_OPEN_POSITIONS}",
            )

        # 6. Exposicion total
        total_exposure = sum(
            p.get("cost_basis", 0) for p in open_positions if p.get("status") == "OPEN"
        )
        max_total = self.bankroll * (MAX_TOTAL_EXPOSURE_PCT / 100.0)
        if total_exposure + cost_usdc > max_total:
            remaining = max(0, max_total - total_exposure)
            return RiskCheckResult(
                allowed=False,
                reason=f"Total exposure ${total_exposure + cost_usdc:.2f} > max ${max_total:.2f} ({MAX_TOTAL_EXPOSURE_PCT}%)",
                max_usdc=remaining,
            )

        # 7. Exposicion same-day
        if game_date:
            same_day_exposure = sum(
                p.get("cost_basis", 0) for p in open_positions
                if p.get("status") == "OPEN" and p.get("game_date") == game_date
            )
            max_same_day = self.bankroll * (MAX_SAME_DAY_EXPOSURE_PCT / 100.0)
            if same_day_exposure + cost_usdc > max_same_day:
                remaining = max(0, max_same_day - same_day_exposure)
                return RiskCheckResult(
                    allowed=False,
                    reason=f"Same-day exposure ${same_day_exposure + cost_usdc:.2f} > max ${max_same_day:.2f}",
                    max_usdc=remaining,
                )

        # Calcular max USDC permitido
        max_usdc = min(
            max_single,
            max_total - total_exposure,
        )
        if game_date:
            same_day_exposure = sum(
                p.get("cost_basis", 0) for p in open_positions
                if p.get("status") == "OPEN" and p.get("game_date") == game_date
            )
            max_same_day = self.bankroll * (MAX_SAME_DAY_EXPOSURE_PCT / 100.0)
            max_usdc = min(max_usdc, max_same_day - same_day_exposure)

        return RiskCheckResult(
            allowed=True,
            reason="OK",
            max_usdc=max(0, max_usdc),
        )

    def check_exit_signals(
        self,
        entry_price: float,
        current_price: float,
        current_edge: float,
    ) -> ExitSignal:
        """Determina si una posicion debe cerrarse (modo trading).

        Decision tree:
          price >= entry + TP_DELTA?  -> TAKE_PROFIT (sell 100%)
          price <= entry - SL_DELTA?  -> STOP_LOSS (sell 100%)
          edge > 3%?                  -> HOLD (edge fuerte)
          1% < edge < 3%?            -> HOLD (edge marginal)
          edge < -3%?                -> SELL_ALL (edge reversed)
          edge < 1%?                 -> SELL_ALL (edge gone)

        Args:
            entry_price: precio de entrada
            current_price: precio actual del share
            current_edge: p_adjusted - current_price

        Returns:
            ExitSignal con action y sell_fraction
        """
        price_delta = current_price - entry_price

        # Take profit: precio subio TAKE_PROFIT_DELTA desde entry
        if price_delta >= TAKE_PROFIT_DELTA:
            return ExitSignal(
                action="TAKE_PROFIT",
                sell_fraction=TAKE_PROFIT_SELL_FRACTION,
                reason=f"TP: +${price_delta:.3f} >= +${TAKE_PROFIT_DELTA:.2f} (entry ${entry_price:.2f} -> ${current_price:.2f})",
            )

        # Stop loss: precio bajo STOP_LOSS_DELTA desde entry
        if price_delta <= -STOP_LOSS_DELTA:
            return ExitSignal(
                action="STOP_LOSS",
                sell_fraction=1.0,
                reason=f"SL: ${price_delta:.3f} <= -${STOP_LOSS_DELTA:.2f} (entry ${entry_price:.2f} -> ${current_price:.2f})",
            )

        # Edge-based decisions
        if current_edge >= MIN_EDGE:
            return ExitSignal(
                action="HOLD",
                sell_fraction=0.0,
                reason=f"Edge {current_edge:.1%} >= {MIN_EDGE:.0%}, holding",
            )

        if 0.01 <= current_edge < MIN_EDGE:
            return ExitSignal(
                action="HOLD",
                sell_fraction=0.0,
                reason=f"Edge {current_edge:.1%} marginal, holding",
            )

        if current_edge < -0.03:
            return ExitSignal(
                action="SELL_ALL",
                sell_fraction=1.0,
                reason=f"Edge reversed {current_edge:.1%} < -3%",
            )

        # Edge gone (< 1%)
        return ExitSignal(
            action="SELL_ALL",
            sell_fraction=1.0,
            reason=f"Edge gone {current_edge:.1%} < 1%",
        )

    def check_post_q3_policy(
        self,
        entry_price: float,
        current_price: float,
        current_edge: float,
    ) -> ExitSignal:
        """Politica post-Q3: en modo trading, mismas reglas de TP/SL aplican."""
        return self.check_exit_signals(entry_price, current_price, current_edge)
