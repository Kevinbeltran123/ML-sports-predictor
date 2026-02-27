"""Metricas de riesgo basadas en CVaR (Conditional Value-at-Risk).

Basado en: Sun & Zou (2024) — "Data-driven distributionally robust Kelly
portfolio optimization based on coherent Wasserstein metrics"

===============================================================
¿Que es CVaR y por que importa para apuestas?
===============================================================

VaR (Value at Risk) responde: "¿Cual es la peor perdida en el X% mejor de
los escenarios?" Ejemplo: VaR_95% = -3% significa que el 95% de los dias
pierdes menos de 3%.

Pero VaR no dice CUANTO pierdes en el peor 5%. Si ese 5% incluye dias con
-3.1% y -25%, el VaR es igual... pero el riesgo real es MUY distinto.

CVaR (tambien llamado Expected Shortfall) responde: "¿Cuanto pierdo EN
PROMEDIO cuando estoy en el peor X%?" Captura la SEVERIDAD de las perdidas
extremas, no solo el umbral.

Para el bankroll de apuestas, CVaR es la metrica correcta porque:
  - Las rachas perdedoras son lo que DESTRUYE un bankroll
  - Kelly clasico puede generar drawdowns enormes si el modelo esta mal calibrado
  - CVaR nos permite ajustar la agresividad ANTES de que el drawdown sea fatal

Referencia paper: Seccion 2.2, Definition 1 — CVaR-Wasserstein metric.
"""

import numpy as np


def calculate_cvar(returns: np.ndarray, alpha: float = 0.95) -> float:
    """Calcula el Conditional Value-at-Risk de una serie de retornos.

    CVaR_α = promedio de los retornos por debajo del percentil (1-α).

    Ejemplo:
      Retornos diarios: [-5, +3, -8, +2, -1, +4, -12, +1, -2, +6]
      Ordenados: [-12, -8, -5, -2, -1, +1, +2, +3, +4, +6]
      Con α=0.80 → peor 20% = los 2 peores: [-12, -8]
      CVaR_0.80 = promedio(-12, -8) = -10.0

      Interpretacion: "Cuando las cosas van mal (peor 20%), pierdo
      en promedio un 10% del bankroll por dia."

    Args:
        returns: array de retornos (pueden ser en % o decimales)
        alpha: nivel de confianza. α=0.95 → peor 5%, α=0.80 → peor 20%

    Returns:
        float: CVaR (negativo = perdida promedio en peor escenario)
    """
    if len(returns) == 0:
        return 0.0

    # VaR = percentil (1-α)
    var_threshold = np.percentile(returns, (1 - alpha) * 100)

    # CVaR = promedio de todo lo que esta por debajo del VaR
    tail_returns = returns[returns <= var_threshold]

    if len(tail_returns) == 0:
        return float(var_threshold)

    return float(np.mean(tail_returns))


def calculate_max_drawdown(cumulative_returns: np.ndarray) -> float:
    """Calcula el maximo drawdown de una serie de retornos acumulados.

    Drawdown = caida desde el pico maximo hasta el valle mas profundo.
    Es la peor racha perdedora historica.

    Ejemplo:
      Bankroll: [100, 105, 110, 95, 90, 98, 105, 85]
      Pico maximo alcanzado: 110 (dia 3)
      Valle mas profundo desde ese pico: 85 (dia 8)
      Drawdown = (85 - 110) / 110 = -22.7%

    Args:
        cumulative_returns: serie de valores acumulados del bankroll

    Returns:
        float: max drawdown como fraccion negativa (ej: -0.227 para -22.7%)
    """
    if len(cumulative_returns) < 2:
        return 0.0

    peak = cumulative_returns[0]
    max_dd = 0.0

    for val in cumulative_returns:
        if val > peak:
            peak = val
        dd = (val - peak) / peak if peak != 0 else 0.0
        if dd < max_dd:
            max_dd = dd

    return float(max_dd)


def adaptive_risk_multiplier(
    recent_returns: np.ndarray,
    target_cvar: float = -5.0,
    alpha: float = 0.95,
) -> float:
    """Multiplicador de riesgo adaptativo basado en CVaR actual vs objetivo.

    Idea del paper (Seccion 4.1): el radio ε controla cuanto nos desviamos
    de la distribucion empirica. Aqui, en lugar de resolver el problema DRO
    completo, usamos CVaR como señal para ESCALAR el sizing de Kelly.

    Si el CVaR actual (racha reciente) es PEOR que nuestro objetivo,
    reducimos apuestas. Si es MEJOR, permitimos ser mas agresivos.

    Ejemplo:
      target_cvar = -5.0 (maximo que toleramos perder en el peor 5%)

      Caso 1 — racha mala:
        actual_cvar = -8.0  → ratio = -5/-8 = 0.625
        Multiplicador = 0.625 → reducimos apuestas 37.5%
        Si Kelly decia 2.0%, ahora apostamos 2.0 × 0.625 = 1.25%

      Caso 2 — racha buena:
        actual_cvar = -3.0  → ratio = -5/-3 = 1.667, capped a 1.5
        Multiplicador = 1.5 → permitimos apostar 50% mas
        Si Kelly decia 2.0%, ahora apostamos 2.0 × 1.5 = 3.0%

    Args:
        recent_returns: retornos recientes (ultimos 30+ dias)
        target_cvar: CVaR objetivo (negativo, ej: -5.0 para -5%)
        alpha: nivel de CVaR (0.95 = controlar peor 5%)

    Returns:
        float: multiplicador ∈ [0.25, 1.5]
    """
    if len(recent_returns) < 10:
        return 1.0  # sin datos suficientes, no ajustar

    actual_cvar = calculate_cvar(recent_returns, alpha)

    # Si no ha habido perdidas, maxima agresividad permitida
    if actual_cvar >= 0:
        return 1.5

    # Ratio: CVaR_objetivo / CVaR_actual
    # Si actual es peor (mas negativo), ratio < 1 → reduce
    # Si actual es mejor (menos negativo), ratio > 1 → sube
    ratio = target_cvar / actual_cvar

    return float(np.clip(ratio, 0.25, 1.5))


def estimate_epsilon_from_calibration(
    predicted_probs: np.ndarray,
    actual_outcomes: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Estima el radio de incertidumbre ε desde el error de calibracion.

    ε representa cuanto puede desviarse P_modelo de P_real.
    Derivado directamente del concepto de "Wasserstein ball" del paper:
    la bola de radio ε centrada en la distribucion empirica contiene
    la distribucion real con alta probabilidad.

    ¿Por que usar calibracion para estimar ε?
    - ECE mide exactamente la distancia promedio entre P_predicha y P_real
    - Es una aproximacion practica de la distancia Wasserstein
    - Se calcula directamente de datos historicos (data-driven, como pide el paper)

    Ejemplo:
      Modelo con ECE=0.03, max_bin_error=0.06
      → ε = 0.03 + 0.5 × 0.06 = 0.06
      → El modelo puede equivocarse hasta ~6 puntos porcentuales

      Modelo con ECE=0.08, max_bin_error=0.15
      → ε = 0.08 + 0.5 × 0.15 = 0.155
      → Modelo poco confiable, Kelly sera muy conservador

    Args:
        predicted_probs: array de P(OVER) historicas
        actual_outcomes: array de 0/1 (UNDER/OVER reales)
        n_bins: numero de bins para calibracion

    Returns:
        float: ε ∈ [0.02, 0.20]
    """
    if len(predicted_probs) < 20:
        # Muy pocas predicciones → maxima incertidumbre
        return 0.15

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    max_bin_error = 0.0

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (predicted_probs >= bin_edges[i]) & (predicted_probs <= bin_edges[i + 1])
        else:
            mask = (predicted_probs >= bin_edges[i]) & (predicted_probs < bin_edges[i + 1])

        if mask.sum() < 3:
            continue

        bin_pred = predicted_probs[mask].mean()
        bin_actual = actual_outcomes[mask].mean()
        bin_error = abs(bin_pred - bin_actual)
        bin_weight = mask.sum() / len(predicted_probs)

        ece += bin_error * bin_weight
        max_bin_error = max(max_bin_error, bin_error)

    # ε = ECE + margen basado en el peor bin
    epsilon = ece + 0.5 * max_bin_error

    return float(np.clip(epsilon, 0.02, 0.20))
