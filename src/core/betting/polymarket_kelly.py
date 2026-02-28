"""Kelly y EV adaptados para Polymarket (share prices 0-1).

Diferencia clave con sportsbooks:
  Sportsbook: odds americanos (-130, +110) -> EV complejo con vig
  Polymarket: share price p in [0,1] -> decimal odds = 1/p, sin vig

  Comprar YES a $0.60 = apostar $0.60 para ganar $1.00 (profit $0.40)
  -> Equivale a odds decimales 1/0.60 = 1.667, o americanos -150

EV simplificado:
  EV = model_prob - price  (por $1 de riesgo maximo)
  Si modelo dice 68% y mercado vende a $0.60 -> edge = +8% -> comprar

Kelly con DRO-CVaR:
  Mismo framework que robust_kelly.py pero con odds = 1/price.
"""

import numpy as np


# ── Constantes ──────────────────────────────────────────────────────

DEFAULT_EPSILON = 0.05
DEFAULT_ALPHA_CVAR = 0.90
DEFAULT_MAX_BET_PCT = 10.0  # Agresivo para $50 bankroll en PM


# ── EV ──────────────────────────────────────────────────────────────

def polymarket_ev(model_prob: float, price: float) -> float:
    """Edge por $1 de riesgo maximo.

    Si compras YES a price=0.60 y el modelo dice prob=0.68:
      Si gana: profit = 1.00 - 0.60 = $0.40
      Si pierde: loss = -$0.60
      EV = 0.68 * 0.40 - 0.32 * 0.60 = +0.080 por share

    Simplificado: EV = model_prob - price
    (porque share paga $1 si gana y $0 si pierde)

    Returns:
        float: edge positivo = comprar, negativo = no comprar
    """
    return model_prob - price


def polymarket_ev_per_100(model_prob: float, price: float) -> float:
    """EV normalizado a $100 de riesgo (compatible con display sportsbook).

    EV por share / price * 100 = EV por $100 invertidos.
    """
    if price <= 0 or price >= 1:
        return 0.0
    ev_per_share = model_prob - price
    return round(ev_per_share / price * 100, 2)


# ── Kelly ───────────────────────────────────────────────────────────

def polymarket_kelly(
    model_prob: float,
    price: float,
    epsilon: float = DEFAULT_EPSILON,
    alpha_cvar: float = DEFAULT_ALPHA_CVAR,
    max_bet_pct: float = DEFAULT_MAX_BET_PCT,
) -> dict:
    """DRO-Kelly adaptado para share prices.

    Convierte price -> decimal odds, aplica CVaR-Wasserstein penalty.

    Flujo:
      1. decimal_odds = (1 - price) / price  (net profit per $1 risked)
      2. p_robust = model_prob - epsilon * (1+alpha)/2
      3. Kelly = (b * p_robust - q_robust) / b
      4. Cap a max_bet_pct

    Args:
        model_prob: P(equipo gana) del ensemble
        price: precio del share en Polymarket (0-1)
        epsilon: radio de incertidumbre (sigma del modelo)
        alpha_cvar: nivel CVaR (0.90 = peor 10%)
        max_bet_pct: cap maximo % del bankroll

    Returns:
        dict con kelly_pct, p_robust, edge, has_edge
    """
    if price <= 0.01 or price >= 0.99:
        return {"kelly_pct": 0.0, "p_robust": 0.0, "edge": 0.0, "has_edge": False}

    # Odds decimales: profit neto por $1 apostado
    # Si compras a $0.60, ganas $0.40 profit -> b = 0.40/0.60 = 0.667
    decimal_odds = (1.0 - price) / price

    # Probabilidad robusta via CVaR-Wasserstein
    cvar_penalty = epsilon * (1.0 + alpha_cvar) / 2.0
    p_robust = np.clip(model_prob - cvar_penalty, 0.01, 0.99)

    # Kelly clasico con p_robust
    q_robust = 1.0 - p_robust
    kelly_full = (decimal_odds * p_robust - q_robust) / decimal_odds
    kelly_pct = kelly_full * 100.0

    kelly_pct = round(max(kelly_pct, 0.0), 2)
    kelly_pct = min(kelly_pct, max_bet_pct)

    edge = model_prob - price

    return {
        "kelly_pct": kelly_pct,
        "p_robust": round(float(p_robust), 4),
        "edge": round(float(edge), 4),
        "has_edge": kelly_pct > 0,
    }


def polymarket_kelly_simple(
    model_prob: float,
    price: float,
    epsilon: float = DEFAULT_EPSILON,
    alpha_cvar: float = DEFAULT_ALPHA_CVAR,
    max_bet_pct: float = DEFAULT_MAX_BET_PCT,
) -> float:
    """Version simplificada: retorna solo el % a apostar."""
    result = polymarket_kelly(model_prob, price, epsilon, alpha_cvar, max_bet_pct)
    return result["kelly_pct"]


# ── Conversiones ────────────────────────────────────────────────────

def shares_from_kelly(kelly_pct: float, bankroll_usdc: float, price: float) -> int:
    """Calcula cantidad de shares a comprar dado Kelly % y bankroll.

    Args:
        kelly_pct: % del bankroll a invertir (ej: 5.0 = 5%)
        bankroll_usdc: bankroll en USDC
        price: precio del share

    Returns:
        int: numero de shares (redondeado hacia abajo)
    """
    if kelly_pct <= 0 or bankroll_usdc <= 0 or price <= 0:
        return 0
    usdc_to_spend = bankroll_usdc * (kelly_pct / 100.0)
    return int(usdc_to_spend / price)


def cost_basis(shares: int, price: float) -> float:
    """Costo total de la posicion en USDC."""
    return round(shares * price, 2)


def share_price_to_american_odds(price: float) -> int:
    """Convierte share price a odds americanos (para display).

    price=0.60 -> -150 (favorito: apuestas $150 para ganar $100)
    price=0.40 -> +150 (underdog: apuestas $100 para ganar $150)
    price=0.50 -> +100 (even)

    Convencion americana:
      p > 0.50: odds = -100 * p / (1-p)  (favorito, odds negativos)
      p < 0.50: odds = +100 * (1-p) / p  (underdog, odds positivos)
      p = 0.50: +100
    """
    if price <= 0.01 or price >= 0.99:
        return 0
    if price > 0.50:
        return int(round(-100.0 * price / (1.0 - price)))
    elif price < 0.50:
        return int(round(100.0 * (1.0 - price) / price))
    else:
        return 100


def american_odds_to_share_price(odds: int) -> float:
    """Convierte odds americanos a share price equivalente.

    -150 -> 0.60
    +150 -> 0.40
    +100 -> 0.50
    """
    if odds < 0:
        return round(abs(odds) / (abs(odds) + 100.0), 4)
    else:
        return round(100.0 / (odds + 100.0), 4)


def unrealized_pnl(shares: int, entry_price: float, current_price: float) -> float:
    """P&L no realizado de una posicion abierta.

    Compro 100 shares a $0.60 (costo $60)
    Precio actual $0.70 -> valor actual $70 -> PnL = +$10
    """
    return round(shares * (current_price - entry_price), 2)
