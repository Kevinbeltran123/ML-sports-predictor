"""Classwise Expected Calibration Error (cw-ECE).

===============================================================
CONCEPTO: Calibración vs Discriminación
===============================================================

Discriminación = ¿puede el modelo separar ganadores de perdedores?
  → Métrica: accuracy, AUC, log_loss
  → Un modelo que predice 90% para TODOS los locales tiene buena accuracy
    si los locales ganan ~55% (predice siempre "home") pero PÉSIMA calibración

Calibración = ¿las probabilidades predichas reflejan la realidad?
  → Métrica: ECE (Expected Calibration Error)
  → Si el modelo dice "70%" para 100 partidos, deberían ganar ~70
  → ESTO es lo que importa para apuestas, porque Kelly Criterion
    usa la probabilidad directamente para calcular cuánto apostar

===============================================================
ECE clásica vs classwise-ECE
===============================================================

ECE clásica (la que ya teníamos en Calibration_Evaluate.py):
  - Solo mira P(home_win) y agrupa en 10 bins
  - Problema: no evalúa la calibración de P(away_win) por separado

classwise-ECE (Paper 2: Walsh & Joshi, ScienceDirect 2024):
  - Evalúa calibración de CADA clase por separado
  - Usa 20 bins (más granular)
  - Promedia sobre clases
  - Constraint: ≥80% de bins no-vacíos (si no, no es confiable)

Resultado del paper: seleccionar modelos por cw-ECE en vez de accuracy
mejoró ROI de -35.17% a +34.69% (swing de 70pp).

===============================================================
Fórmula
===============================================================

cw-ECE = (1/K) × Σ_{k=1}^{K} Σ_{j=1}^{M} (|B_{j,k}|/n) × |freq_k - conf_k|

Donde:
  K = número de clases (2 para binario)
  M = número de bins (20)
  B_{j,k} = predicciones cuya P(clase k) cae en bin j
  freq_k = fracción real de clase k en B_{j,k}
  conf_k = promedio de P(clase k) predicha en B_{j,k}
  n = total de predicciones
"""

import numpy as np
from sklearn.metrics import log_loss as sklearn_log_loss


def compute_classwise_ece(y_true, y_prob, n_bins=20, min_nonempty_ratio=0.8):
    """Calcula classwise-ECE para clasificación binaria.

    Ejemplo intuitivo con n_bins=5 y K=2:
      Bin [0.6, 0.8) para clase "home_win":
        50 predicciones con P(home_win) entre 0.6 y 0.8
        Promedio predicho: 0.68
        Frecuencia real de victoria local: 0.72
        Error del bin: (50/n) × |0.72 - 0.68| = (50/n) × 0.04

    Args:
        y_true: array (n,) con etiquetas reales (0 o 1)
        y_prob: array (n, 2) con [P(away_win), P(home_win)]
        n_bins: bins por clase (20 per Paper 2, más granular que 10)
        min_nonempty_ratio: fracción mínima de bins no-vacíos para
                           considerar la ECE confiable (0.80 per Paper 2)

    Returns:
        (ece_value, is_reliable):
            ece_value: float, classwise-ECE (0=perfecto, 1=terrible)
            is_reliable: bool, True si hay suficientes bins no-vacíos
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    K = y_prob.shape[1]  # 2 para binario
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    total_ece = 0.0
    total_nonempty = 0

    # Iterar sobre cada clase
    for k in range(K):
        # Probabilidades predichas para clase k
        probs_k = y_prob[:, k]
        # Indicador binario: ¿la muestra pertenece a clase k?
        is_class_k = (y_true == k).astype(float)

        nonempty_bins = 0
        class_ece = 0.0

        # Iterar sobre cada bin
        for j in range(n_bins):
            lo = bin_boundaries[j]
            hi = bin_boundaries[j + 1]

            # Último bin incluye el borde derecho [lo, hi]
            if j == n_bins - 1:
                mask = (probs_k >= lo) & (probs_k <= hi)
            else:
                mask = (probs_k >= lo) & (probs_k < hi)

            count = mask.sum()
            if count == 0:
                continue

            nonempty_bins += 1

            # Frecuencia real de clase k en este bin
            freq_k = is_class_k[mask].mean()
            # Confianza promedio predicha para clase k
            avg_conf_k = probs_k[mask].mean()
            # Error ponderado por tamaño del bin
            class_ece += (count / n) * abs(freq_k - avg_conf_k)

        total_ece += class_ece
        total_nonempty += nonempty_bins

    # Promediar sobre clases
    classwise_ece = total_ece / K

    # Verificar confiabilidad: ¿hay suficientes bins no-vacíos?
    total_possible_bins = K * n_bins
    nonempty_ratio = total_nonempty / total_possible_bins if total_possible_bins > 0 else 0
    is_reliable = nonempty_ratio >= min_nonempty_ratio

    return classwise_ece, is_reliable


def compute_classwise_ece_or_fallback(y_true, y_prob, n_bins=20,
                                       min_nonempty_ratio=0.8):
    """Calcula cw-ECE si hay suficientes bins; si no, retorna log_loss.

    ¿Por qué fallback?
    En walk-forward CV, los primeros folds tienen pocos datos de validación.
    Con ~1000 muestras y 20 bins × 2 clases = 40 bins, muchos quedan vacíos.
    En ese caso, cw-ECE no es estadísticamente confiable, así que usamos
    log_loss como proxy (que también penaliza mala calibración, aunque menos).

    Args:
        y_true: array (n,) con etiquetas
        y_prob: array (n, 2) con probabilidades predichas

    Returns:
        (metric_value, metric_name): float y str ("cw-ECE" o "log_loss")
    """
    ece, is_reliable = compute_classwise_ece(
        y_true, y_prob, n_bins, min_nonempty_ratio
    )

    if is_reliable:
        return ece, "cw-ECE"
    else:
        # Fallback a log_loss cuando no hay suficientes bins
        K = y_prob.shape[1]
        ll = sklearn_log_loss(y_true, y_prob, labels=list(range(K)))
        return ll, "log_loss (fallback)"
