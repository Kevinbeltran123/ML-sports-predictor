import logging

_logger = logging.getLogger(__name__)

QUARTER_KELLY_FRACTION = 0.25
EIGHTH_KELLY_FRACTION = 0.125
DEFAULT_MAX_BET_PCT = 5.0       # cap para quarter-Kelly
EIGHTH_MAX_BET_PCT = 2.5        # cap mas conservador para eighth-Kelly


def american_to_decimal(american_odds):
    """
    Converts American odds to decimal odds (European odds).
    """
    if american_odds >= 100:
        decimal_odds = (american_odds / 100)
    else:
        decimal_odds = (100 / abs(american_odds))
    return round(decimal_odds, 2)


def calculate_kelly_criterion(american_odds, model_prob):
    """
    Calculates the fraction of the bankroll to be wagered on each bet.
    Retorna Kelly COMPLETO (sin reducir). Usar calculate_quarter_kelly()
    en producción para un sizing más conservador.
    """
    decimal_odds = american_to_decimal(american_odds)
    bankroll_fraction = round((100 * (decimal_odds * model_prob - (1 - model_prob))) / decimal_odds, 2)
    return bankroll_fraction if bankroll_fraction > 0 else 0


def calculate_quarter_kelly(american_odds, model_prob, max_bet_pct=DEFAULT_MAX_BET_PCT):
    """Calcula Quarter-Kelly con cap máximo de seguridad.

    Quarter-Kelly = Kelly completo × 0.25, capeado en max_bet_pct.

    ¿Por qué Quarter-Kelly?
    Kelly completo es matemáticamente óptimo pero asume que el modelo
    es perfectamente preciso — lo cual nunca es verdad en la práctica.
    Usar 1/4 del Kelly completo reduce la volatilidad del bankroll
    sustancialmente (~75% menos) a cambio de crecer más lento.

    ¿Por qué el cap?
    Si el modelo está muy confiado (Kelly_full=25%), Quarter-Kelly=6.25%.
    Pero apostar 6.25% del bankroll en un solo partido es demasiado
    para principiantes. El cap de 5% pone un límite de seguridad
    independiente del edge calculado.

    Ejemplo:
      odds=-130, prob=0.63 → Kelly_full≈12% → Quarter=3.0% (< 5% cap, sin cambio)
      odds=+200, prob=0.55 → Kelly_full≈25% → Quarter=6.25% → capeado a 5.0%
      odds=-110, prob=0.45 → Kelly_full<0  → 0.0% (nunca apostar con EV negativo)

    Args:
        american_odds: odds americanos (e.g., -130, +110)
        model_prob: probabilidad del modelo de ganar (e.g., 0.63)
        max_bet_pct: cap máximo en % del bankroll (default 5.0%)

    Returns:
        float: % del bankroll a apostar (0.0 si no hay edge)
    """
    return calculate_fractional_kelly(american_odds, model_prob,
                                      QUARTER_KELLY_FRACTION, max_bet_pct)


def calculate_fractional_kelly(american_odds, model_prob, fraction=0.125,
                                max_bet_pct=2.5):
    """Kelly fraccional genérico.

    Full Kelly es matemáticamente óptimo pero asume probabilidades EXACTAS.
    En la práctica, las probabilidades tienen error → Kelly sobreapuesta.

    Reducir la fracción reduce la varianza del bankroll:
      - fraction=1.0   → Full Kelly (100% de la volatilidad)
      - fraction=0.25  → Quarter Kelly (reduce ~75%)
      - fraction=0.125 → Eighth Kelly (reduce ~87.5%)

    Paper 2 (Walsh & Joshi 2024): recomienda fracciones conservadoras
    porque incluso modelos bien calibrados tienen incertidumbre.
    """
    kelly_full = calculate_kelly_criterion(american_odds, model_prob)
    if kelly_full <= 0:
        return 0.0
    kelly_frac = round(kelly_full * fraction, 2)
    capped = min(kelly_frac, max_bet_pct)
    if capped < kelly_frac:
        _logger.debug("Kelly capped: %.2f%% → %.2f%% (full=%.2f%%, frac=%.3f)",
                      kelly_frac, capped, kelly_full, fraction)
    return capped


def calculate_eighth_kelly(american_odds, model_prob, max_bet_pct=EIGHTH_MAX_BET_PCT):
    """Eighth-Kelly (1/8) con cap de 2.5%.

    Más conservador que quarter-Kelly:
      - Fracción: 0.125 (vs 0.25)
      - Cap: 2.5% (vs 5.0%)

    Ejemplo:
      odds=-130, prob=0.63 → Kelly_full≈12% → Eighth=1.5% (< 2.5% cap)
      odds=+200, prob=0.55 → Kelly_full≈25% → Eighth=3.12% → capeado a 2.5%

    NOTA: Para player props, se recomienda usar calculate_robust_kelly_simple()
    de src.core.betting.robust_kelly, que adapta la fracción automáticamente
    según el error de calibración del modelo (paper: Sun & Zou 2024, DRO-Kelly).
    """
    return calculate_fractional_kelly(american_odds, model_prob,
                                      EIGHTH_KELLY_FRACTION, max_bet_pct)