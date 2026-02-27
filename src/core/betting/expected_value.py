def expected_value(Pwin, odds):
    Ploss = 1 - Pwin
    Mwin = payout(odds)
    return round((Pwin * Mwin) - (Ploss * 100), 2)


def payout(odds):
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
    win_amount = payout(odds)

    ev = (
        ah_probs["p_full_win"] * win_amount
        + ah_probs["p_half_win"] * (win_amount / 2)
        + ah_probs["p_half_loss"] * (-50)
        + ah_probs["p_full_loss"] * (-100)
    )
    return round(ev, 2)
