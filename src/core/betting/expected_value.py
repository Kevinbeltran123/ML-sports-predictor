def expected_value(Pwin, odds):
    """EV por $100 apostados.

    Retorna la ganancia/perdida esperada si apostaras $100 muchas veces.
    Ejemplo: odds=-130, prob=0.63 → EV = 0.63 * 76.92 - 0.37 * 100 = +11.46
    """
    Ploss = 1 - Pwin
    Mwin = payout(odds)
    return round((Pwin * Mwin) - (Ploss * 100), 2)


def payout(odds):
    """Calcula profit por $100 apostados (no incluye el stake de vuelta).

    Ejemplos:
      -130 → 100/130 * 100 = 76.92 (apuestas $130, ganas $100 → profit $76.92 por cada $100)
      +150 → 150 (apuestas $100, ganas $150)
    """
    if odds > 0:
        return odds
    else:
        return (100 / (-1 * odds)) * 100


def ah_expected_value(ah_probs, odds):
    """Expected Value para Asian Handicap con settlement completo.

    Para full lines (x.0, x.5): igual que expected_value() estándar.
    Para quarter lines (x.25, x.75): settlement con half win/loss.

    Settlement por $100 apostados:
      Full win:  +payout(odds)     (ambas mitades ganan)
      Half win:  +payout(odds)/2   (una gana, otra push → devuelven $50)
      Half loss: -50               (una pierde, otra push → devuelven $50)
      Full loss: -100              (ambas mitades pierden)

    Args:
        ah_probs: dict de ah_probabilities() con p_full_win, p_half_win,
                  p_half_loss, p_full_loss
        odds: odds americanas del spread

    Returns:
        EV por $100 apostados, redondeado a 2 decimales
    """
    # Validar que probabilidades suman ~1
    total = sum(ah_probs.get(k, 0) for k in
                ["p_full_win", "p_half_win", "p_half_loss", "p_full_loss"])
    if abs(total - 1.0) > 0.01:
        import logging
        logging.getLogger(__name__).warning(
            "AH probs don't sum to 1: %.4f (%s)", total, ah_probs)

    win_amount = payout(odds)

    ev = (
        ah_probs["p_full_win"] * win_amount
        + ah_probs["p_half_win"] * (win_amount / 2)
        + ah_probs["p_half_loss"] * (-50)
        + ah_probs["p_full_loss"] * (-100)
    )
    return round(ev, 2)
