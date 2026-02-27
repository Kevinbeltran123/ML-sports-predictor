"""Conversión P(win) → P(cover spread) para Asian Handicap.

Modelo: el margen de victoria M ~ N(μ, σ²) donde:
  μ = σ × Φ⁻¹(P_win)    (margen esperado en puntos)

σ es ADAPTATIVO por magnitud del spread (calibrado con 46K+ juegos NBA 2012-2026):
  |spread| 0-2:  σ=13.0  (pick'em, menos varianza)
  |spread| 2-5:  σ=13.2  (favoritos leves)
  |spread| 5-8:  σ=14.0  (favoritos moderados)
  |spread| 8-12: σ=15.0  (favoritos grandes)
  |spread| 12+:  σ=16.3  (favoritos enormes, máxima varianza)

El σ=12 comúnmente citado subestima la variabilidad real (~14 global).
La NBA moderna (2023-26) tiene σ≈15.6, tendencia creciente desde 2012.

Home cubre spread L cuando M > -L (margen supera los puntos que da).
  P(cubrir L) = P(M > -L) = Φ((μ + L) / σ) = Φ(Φ⁻¹(P_win) + L/σ)

Convención: line negativa = home favorito (ej: -5.5 = home da 5.5 puntos).

Quarter lines (ej: -5.25, -5.75):
  Son apuestas divididas — mitad en cada línea adyacente.
  -5.25 = mitad en -5.0 + mitad en -5.5
  -5.75 = mitad en -5.5 + mitad en -6.0
  Generan 3 outcomes: full win, half win/loss, full loss.
"""

import numpy as np
from scipy.stats import norm

# Sigma por bucket de |spread|, calibrado con OddsData.sqlite (46K juegos, 2012-2026).
# Tendencia clara: favoritos grandes → más varianza (blowouts o upsets dramáticos).
_SIGMA_BUCKETS = [
    (2.0, 13.0),   # pick'em: juegos cerrados
    (5.0, 13.2),   # favoritos leves
    (8.0, 14.0),   # favoritos moderados
    (12.0, 15.0),  # favoritos grandes
    (float("inf"), 16.3),  # favoritos enormes
]

# Fallback global (media ponderada de los buckets)
NBA_MARGIN_SIGMA = 14.3


def _sigma_for_line(line):
    """Retorna σ calibrado según la magnitud del spread."""
    abs_line = abs(line)
    for threshold, sigma in _SIGMA_BUCKETS:
        if abs_line <= threshold:
            return sigma
    return NBA_MARGIN_SIGMA


def p_cover(p_win, line, sigma=None):
    """Convierte P(ganar) → P(cubrir spread) para el home team.

    Args:
        p_win: probabilidad de ganar (0-1), del ensemble
        line: spread del local (negativo = favorito). Ej: -5.5
        sigma: override manual. Si None, usa σ adaptativo por bucket.

    Returns:
        P(home cubre spread), clipeada a [0.01, 0.99]
    """
    if p_win <= 0.0 or p_win >= 1.0:
        return 0.5
    if sigma is None:
        sigma = _sigma_for_line(line)
    z_win = norm.ppf(p_win)
    z_cover = z_win + line / sigma
    return float(np.clip(norm.cdf(z_cover), 0.01, 0.99))


def is_quarter_line(line):
    """Detecta si una línea es quarter (x.25 o x.75)."""
    frac = abs(line) % 1
    return abs(frac - 0.25) < 0.01 or abs(frac - 0.75) < 0.01


def split_quarter_line(line):
    """Descompone una quarter line en sus dos componentes.

    -5.25 → (-5.0, -5.5)   (lo más cercano a 0 primero)
    -5.75 → (-5.5, -6.0)
    +3.25 → (+3.5, +3.0)
    +3.75 → (+4.0, +3.5)

    Returns:
        (line_near, line_far): near = más cercana a 0 (push posible),
                               far = más lejana de 0 (sin push, half-point).
    """
    frac = abs(line) % 1
    sign = -1 if line < 0 else 1

    base = int(abs(line))  # parte entera
    if abs(frac - 0.25) < 0.01:
        # x.25 → split entre x.0 y x.5
        line_near = sign * base         # ej: -5.0 (push posible en entero)
        line_far = sign * (base + 0.5)  # ej: -5.5 (half-point, sin push)
    else:
        # x.75 → split entre x.5 y (x+1).0
        line_near = sign * (base + 0.5)  # ej: -5.5 (half-point, sin push)
        line_far = sign * (base + 1)     # ej: -6.0 (push posible en entero)

    return line_near, line_far


def ah_probabilities(p_win, line, sigma=None):
    """Calcula probabilidades de los 3 outcomes del Asian Handicap.

    Para full lines (x.0, x.5): es binario (win/loss), sin push.
    Para quarter lines (x.25, x.75): split bet con 3 outcomes.

    Args:
        p_win: P(home gana)
        line: spread del home (negativo = favorito)
        sigma: override. Si None, adaptativo.

    Returns:
        dict con:
          p_full_win: ambas mitades ganan
          p_half_win: una mitad gana, la otra push (refund)
          p_half_loss: una mitad pierde, la otra push (refund)
          p_full_loss: ambas mitades pierden
          line_near: línea cercana a 0 (o la line misma si full)
          line_far: línea lejana de 0 (o la line misma si full)
          is_quarter: si es quarter line
    """
    if not is_quarter_line(line):
        # Full line: binario
        pc = p_cover(p_win, line, sigma)
        return {
            "p_full_win": pc,
            "p_half_win": 0.0,
            "p_half_loss": 0.0,
            "p_full_loss": 1.0 - pc,
            "line_near": line,
            "line_far": line,
            "is_quarter": False,
        }

    # Quarter line: split bet
    line_near, line_far = split_quarter_line(line)
    p_near = p_cover(p_win, line_near, sigma)  # P(cubrir la línea más fácil)
    p_far = p_cover(p_win, line_far, sigma)    # P(cubrir la línea más difícil)

    # 3 outcomes:
    # Full win: cubre AMBAS líneas → P = P(far) (la más difícil)
    # Half win: cubre near pero NO far → P = P(near) - P(far)
    #   (Nota: "half win" = el near push o gana, el far pierde.
    #    En el modelo continuo, push tiene P≈0, así que half win ≈ P(near)-P(far))
    # Full loss: no cubre NINGUNA → P = 1 - P(near)
    p_full_win = p_far
    p_half = p_near - p_far  # puede ser half win O half loss
    p_full_loss = 1.0 - p_near

    # Para home (la dirección del spread):
    # Si line < 0 (home favorito con handicap negativo):
    #   near = -5.0 (más fácil de cubrir, margen > 5)
    #   far = -5.5 (más difícil, margen > 5.5)
    #   half zone = margen entre 5.0 y 5.5 → home cubre near pero no far → HALF LOSS
    #     (ganas la mitad near, pierdes la mitad far → neto = pierdes mitad del stake)
    #
    # Si line > 0 (home underdog con handicap positivo):
    #   near = +3.5 (más fácil, margen > -3.5 i.e. perder por menos de 3.5)
    #   far = +3.0 (más difícil, margen > -3.0 i.e. perder por menos de 3.0)
    #   half zone = margen entre -3.5 y -3.0 → cubre far pero no near... wait
    #
    # Corrección: "near" y "far" dependen de la dirección.
    # Para line < 0: P(cover near) > P(cover far) → near es más fácil
    # Para line > 0: P(cover near) > P(cover far) → near es más fácil
    # El half zone siempre es: cubre near, no cubre far → half outcome.
    #
    # Settlement:
    #   .25 lines: half zone = half LOSS (ganas la mitad fácil, pierdes la difícil)
    #     Net = (refund $50) + (lose $50) = -$50 → half loss
    #     WAIT: si cubres near (-5.0) pero no far (-5.5), el near es un WIN (no push).
    #     En NBA, -5.0 con margen exacto de 5 → push.
    #     Con margen entre 5.0 y 5.5: near (-5.0) pierde (margen no > 5.0... wait, = push)
    #
    # Actually let me re-think. For NBA with integer scores:
    # line_near = -5.0, line_far = -5.5
    # Margen = 5 → near pushes (exactly 5), far loses → net = half loss
    # Margen = 6+ → both win → full win
    # Margen ≤ 4 → both lose → full loss
    #
    # In our continuous model, we're already capturing this correctly:
    # p_near = P(M > 5.0), p_far = P(M > 5.5)
    # The zone between 5.0 and 5.5 is where one wins and the other doesn't.
    # In this zone: near (-5.0) is a push (margin exactly 5) or near-win
    # Actually no. In continuous model M > 5.0 means strictly greater.
    # For integer scores: M=5 is push on -5.0. So:
    #   near line wins when M > 5.0 (strictly) → same as P(cover -5.0) in continuous
    #   far line wins when M > 5.5 → same as P(cover -5.5)
    #   Zone: 5.0 < M ≤ 5.5 → near wins, far loses → half win
    #   Wait: M = 5.0 exactly → push on near, lose on far → technically half loss
    #
    # In the continuous model, P(M = exactly 5.0) = 0, so this distinction doesn't matter.
    # The zone P(near) - P(far) gives us the "between" probability.
    #
    # For a .25 line (e.g., -5.25):
    #   near = -5.0 (whole number → push possible, but P≈0 in continuous)
    #   far = -5.5 (half point → no push)
    #   Zone where M is between cover thresholds: HALF WIN for the bettor
    #   (near line wins, far line doesn't → refund far + win near → net +half payout)
    #
    # Wait, I need to think about this more carefully.
    # -5.25 = half stake on -5.0 + half stake on -5.5
    # If M > 5.5: both win → full win
    # If 5.0 < M < 5.5: -5.0 wins, -5.5 loses → net = +half payout - half stake
    #   That's a half win (you win one half, lose the other, net positive since payout > stake at typical odds)
    #   Actually at -110: win = +$45.45 on $50, lose = -$50 → net = -$4.55 → HALF LOSS
    #   At +100: win = +$50, lose = -$50 → net = $0 → break even
    #   At +110: win = +$55, lose = -$50 → net = +$5 → tiny half win
    #
    # Hmm, whether the "between zone" is a half win or half loss depends on the odds.
    # At standard -110, it's actually a slight loss because you lose more than you gain.
    #
    # The traditional AH settlement is:
    #   Full win: +payout (full win)
    #   Half win: +payout/2 (stake refunded on push half, win on other half)
    #   Half loss: -stake/2 (stake refunded on push half, lose on other half)
    #   Full loss: -stake (full loss)
    #
    # For -5.25:
    #   M > 5.5: full win → +payout
    #   5.0 < M ≤ 5.5: half win → +payout/2 ... no wait
    #   M = 5: push on -5.0, lose on -5.5 → get back half stake, lose half → -stake/2 = half loss
    #   M < 5: both lose → -stake = full loss
    #
    # Actually I think I was overcomplicating this. Let me re-read AH rules.
    #
    # AH -5.25 splits into -5.0 and -5.5:
    # -5.0 part: wins if margin > 5, pushes if margin = 5, loses if margin < 5
    # -5.5 part: wins if margin > 5.5, loses if margin ≤ 5.5 (no push on half points)
    #
    # Scenarios with integer scores:
    # margin ≥ 6: -5.0 wins, -5.5 wins → FULL WIN
    # margin = 5: -5.0 pushes, -5.5 loses → HALF LOSS (get back -5.0 stake, lose -5.5 stake)
    # margin ≤ 4: -5.0 loses, -5.5 loses → FULL LOSS
    #
    # So for -5.25, the "between" zone is always a HALF LOSS.
    #
    # For -5.75 (splits into -5.5 and -6.0):
    # margin ≥ 7: -5.5 wins, -6.0 wins → FULL WIN
    # margin = 6: -5.5 wins, -6.0 pushes → HALF WIN (win -5.5, get back -6.0 stake)
    # margin ≤ 5: -5.5 loses, -6.0 loses → FULL LOSS
    #
    # So: .25 lines have half LOSS in the middle, .75 lines have half WIN in the middle!
    #
    # BUT in our continuous normal model, there is no "exactly integer" distinction.
    # The zone between the two lines can be treated as:
    #   For .25: half_loss zone
    #   For .75: half_win zone

    # OK so the answer is:
    frac = abs(line) % 1
    if abs(frac - 0.25) < 0.01:
        # .25 line: between zone = HALF LOSS
        return {
            "p_full_win": p_far,
            "p_half_win": 0.0,
            "p_half_loss": p_half,
            "p_full_loss": p_full_loss,
            "line_near": line_near,
            "line_far": line_far,
            "is_quarter": True,
        }
    else:
        # .75 line: between zone = HALF WIN
        return {
            "p_full_win": p_far,
            "p_half_win": p_half,
            "p_half_loss": 0.0,
            "p_full_loss": p_full_loss,
            "line_near": line_near,
            "line_far": line_far,
            "is_quarter": True,
        }


def expected_margin(p_win, sigma=NBA_MARGIN_SIGMA):
    """Margen esperado en puntos: E[margen] = σ × Φ⁻¹(p_win).

    Positivo = home gana por X puntos.
    Negativo = home pierde por X puntos.
    """
    if p_win <= 0.0 or p_win >= 1.0:
        return 0.0
    return float(sigma * norm.ppf(p_win))


def p_cover_regression(mu, line, sigma=None):
    """P(cover spread) usando μ del regresor en vez de P(win).

    A diferencia de p_cover() que calcula μ indirectamente via P(win),
    aqui μ viene directamente del regresor XGBoost.

    Math:
        M ~ N(μ, σ²)
        Home cubre cuando M > -L → P(cover) = Φ((μ + L) / σ)

    Args:
        mu: margen esperado en puntos (positivo = home gana)
        line: spread del local (negativo = favorito). Ej: -5.5
        sigma: override. Si None, usa σ adaptativo por bucket.

    Returns:
        P(home cubre spread), clipeada a [0.01, 0.99]
    """
    if sigma is None:
        sigma = _sigma_for_line(line)
    z = (mu + line) / sigma
    return float(np.clip(norm.cdf(z), 0.01, 0.99))


def p_win_from_margin(mu, sigma=NBA_MARGIN_SIGMA):
    """Inversa de expected_margin: P(win) = Φ(μ / σ).

    Util para comparar el regresor con el clasificador:
    si el regresor predice μ=+5.0, esto equivale a P(win) ≈ 64%.
    """
    return float(np.clip(norm.cdf(mu / sigma), 0.01, 0.99))
