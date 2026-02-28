"""Risk management para Polymarket NBA trading.

Limites agresivos para bankroll chico ($50 USDC):
  - Posicion individual max: 10% ($5)
  - Exposicion total max: 40% ($20)
  - Exposicion same-day max: 30% ($15)
  - Posiciones abiertas max: 8
  - Stop-loss: precio cae 40% desde entry
  - Take-profit: vender 50% a 2x entry
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
STOP_LOSS_PCT = 0.40              # Vender si precio cae 40% desde entry
TAKE_PROFIT_MULTIPLIER = 2.0      # Vender 50% si precio llega a 2x entry
TAKE_PROFIT_SELL_FRACTION = 0.50  # Vender 50% en take-profit
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
        """Determina si una posicion debe cerrarse.

        Decision tree (del plan):
          price > 2x entry?        -> TAKE_PROFIT (sell 50%)
          price < 60% of entry?    -> STOP_LOSS (sell all)
          new_edge > 3%?           -> BUY_MORE (if risk allows)
          1% < new_edge < 3%?      -> HOLD
          new_edge < 1%?           -> SELL_ALL (edge gone)
          new_edge < -3%?          -> SELL_ALL (edge reversed)

        Args:
            entry_price: precio de entrada
            current_price: precio actual del share
            current_edge: p_adjusted - current_price

        Returns:
            ExitSignal con action y sell_fraction
        """
        # Take profit: precio duplicado
        if current_price >= entry_price * TAKE_PROFIT_MULTIPLIER:
            return ExitSignal(
                action="TAKE_PROFIT",
                sell_fraction=TAKE_PROFIT_SELL_FRACTION,
                reason=f"Price ${current_price:.2f} >= {TAKE_PROFIT_MULTIPLIER}x entry ${entry_price:.2f}",
            )

        # Stop loss: precio cayo 40%
        stop_price = entry_price * (1.0 - STOP_LOSS_PCT)
        if current_price <= stop_price:
            return ExitSignal(
                action="STOP_LOSS",
                sell_fraction=1.0,
                reason=f"Price ${current_price:.2f} <= stop ${stop_price:.2f} (-{STOP_LOSS_PCT:.0%})",
            )

        # Edge-based decisions
        if current_edge >= MIN_EDGE:
            return ExitSignal(
                action="BUY_MORE",
                sell_fraction=0.0,
                reason=f"Edge {current_edge:.1%} >= {MIN_EDGE:.0%}, can add",
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
        """Politica post-Q3: hold winners, exit losers without edge.

        Despues del Q3, si la posicion es profitable la dejamos
        ir a settlement (paga $1.00 si gana). Si esta perdiendo
        y no hay edge, vendemos para recuperar algo.
        """
        is_profitable = current_price > entry_price

        if is_profitable:
            return ExitSignal(
                action="HOLD",
                sell_fraction=0.0,
                reason=f"Post-Q3: profitable (${current_price:.2f} > ${entry_price:.2f}), hold to settlement",
            )

        if current_edge >= 0.01:
            return ExitSignal(
                action="HOLD",
                sell_fraction=0.0,
                reason=f"Post-Q3: losing but edge {current_edge:.1%}, hold",
            )

        return ExitSignal(
            action="SELL_ALL",
            sell_fraction=1.0,
            reason=f"Post-Q3: losing and no edge ({current_edge:.1%}), exit",
        )
