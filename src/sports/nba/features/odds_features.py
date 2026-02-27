"""Features derivadas de las odds del mercado.

Separa la logica de calculo de odds de create_games.py para mantener
el patron build -> get -> add y facilitar pruebas unitarias.

Patron de uso en create_games.py:
    from src.sports.nba.features.odds_features import compute_vig_magnitude
    # ... despues del loop por partido (lista comprehension):
    vig_magnitudes = [
        compute_vig_magnitude(h, a)
        for h, a in zip(ml_home_values, ml_away_values)
    ]
    frame["VIG_MAGNITUDE"] = np.asarray(vig_magnitudes)
"""

from src.config import get_logger

logger = get_logger(__name__)


def compute_vig_magnitude(ml_home: float, ml_away: float) -> float:
    """Calcula el overround del bookmaker (vig) para un partido.

    El vig es la suma de las probabilidades implicitas menos 1.0.
    Representa el margen de la casa antes de remover el vig.

    Ejemplos tipicos:
        ML_Home=-250, ML_Away=+205 -> p_home=0.714, p_away=0.328 -> vig=0.042 (4.2%)
        ML_Home=-110, ML_Away=-110 -> p_home=0.524, p_away=0.524 -> vig=0.048 (4.8%)

    Interpretacion:
        VIG alto (> 0.08): mercado incierto o bookmaker agresivo
        VIG bajo (< 0.03): mercado eficiente, linea muy definida
        VIG normal NBA: [0.04, 0.07]

    T-1 safety: ml_home y ml_away vienen de OddsData (FanDuel closing lines).
    Las closing lines son el mejor estimado del mercado justo antes del tipoff.
    Usar closing lines como features es seguro -- representan informacion
    pre-partido incorporada por el mercado. MARKET_ML_PROB usa la misma fuente.
    Auditoria de timestamp confirmada en Phase 1 (01-02-PLAN).

    Returns:
        float: overround en escala 0..1 (tipicamente 0.03..0.10 para NBA).
               Valores > 0.15 indican posible error en los datos.
    """
    def _implied(odds: float) -> float:
        odds = float(odds)
        if odds < 0:
            return abs(odds) / (abs(odds) + 100.0)
        return 100.0 / (odds + 100.0)

    p_home = _implied(ml_home)
    p_away = _implied(ml_away)
    vig = p_home + p_away - 1.0  # overround: la suma excede 1.0 por el vig

    if vig > 0.15:
        logger.debug(
            "VIG_MAGNITUDE alto (%.3f) para ML_Home=%.0f ML_Away=%.0f -- "
            "verificar datos de odds",
            vig, ml_home, ml_away,
        )
    return vig
