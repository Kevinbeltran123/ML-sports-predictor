"""Conformal Prediction para clasificación binaria de Player Props.

===============================================================
¿Qué es Conformal Prediction? (Vovk et al. 2005)
===============================================================

ECE mide calibración GLOBAL: "en promedio, el modelo se equivoca 5%".
Conformal Prediction mide confianza PER-PREDICTION:
  - "Para ESTE jugador, ESTE partido → ¿el modelo puede distinguir OVER de UNDER?"

Método: Split Conformal con scores LAC (Least Ambiguous Criterion).

Paso 1 — Calibración:
    En el set de calibración, para cada ejemplo:
      score_i = 1 - P(clase correcta)
    Si el modelo acierta con P=0.80, score = 0.20 (bajo → bueno).
    Si falla con P=0.45, score = 0.55 (alto → malo).
    Guardamos estos scores ordenados.

Paso 2 — Quantil conformal:
    q̂ = quantil((1-α)(1 + 1/n)) de los scores.
    Con α=0.10 (90% coverage) y 1000 ejemplos: el score percentil 90.1.
    Todo ejemplo con score ≤ q̂ estará en el "set conformal".

Paso 3 — Prediction Set:
    Una clase j está en el set si: P(j) ≥ 1 - q̂.
    - Si solo OVER cumple → set = {OVER} → CONFIADO (set_size=1)
    - Si ambos cumplen → set = {OVER, UNDER} → INCIERTO (set_size=2)
    - Si ninguno cumple → set = {} → MUY INUSUAL (set_size=0)

¿Para qué sirve en apuestas?
    - set_size == 1: apostar con Kelly normal
    - set_size == 2: reducir Kelly o skip
    - conformal_margin alto: margen de seguridad → más confianza

===============================================================
Uso
===============================================================

    # En entrenamiento (props_classifier.py):
    conformal = ConformalClassifier(alpha=0.10)
    conformal.fit(probs_cal, y_cal)    # probs_cal = (n, 2) del calibrador
    joblib.dump(conformal, "model_conformal.pkl")

    # En predicción (props_runner.py):
    conformal = joblib.load("model_conformal.pkl")
    set_size, margin = conformal.predict_confidence(probs_new)
    if set_size == 2:
        kelly = 0  # skip — modelo no puede distinguir
    elif margin > 0.05:
        kelly *= 1.0  # full Kelly
    else:
        kelly *= 0.5  # half Kelly — margen bajo

Paper: Vovk, Gammerman & Shafer (2005) "Algorithmic Learning in a Random World"
       Angelopoulos & Bates (2021) "A Gentle Introduction to Conformal Prediction"
"""

import numpy as np


class ConformalClassifier:
    """Prediction sets conformales para clasificación binaria.

    Garantía teórica: con α=0.10, al menos 90% de las predicciones
    futuras tendrán la clase correcta en su prediction set.

    Atributos:
        alpha: nivel de error (1 - coverage). Default 0.10 = 90%.
        cal_scores_: scores de nonconformidad del set de calibración.
        quantile_: quantil conformal q̂.
        threshold_: probabilidad mínima para inclusión en el set (1 - q̂).
        coverage_: cobertura empírica en el set de calibración.
    """

    def __init__(self, alpha: float = 0.10):
        """
        Args:
            alpha: nivel de error deseado. 0.10 = 90% coverage guarantee.
                   Valores típicos: 0.05 (95%), 0.10 (90%), 0.20 (80%).
        """
        self.alpha = alpha
        self.cal_scores_ = None
        self.quantile_ = None
        self.threshold_ = None
        self.coverage_ = None

    def fit(self, y_probs: np.ndarray, y_true: np.ndarray):
        """Calibra el predictor conformal con el set de calibración.

        Args:
            y_probs: (n, 2) probabilidades [P(UNDER), P(OVER)] del modelo calibrado.
            y_true: (n,) labels reales (0=UNDER, 1=OVER).

        Returns:
            self
        """
        n = len(y_true)
        if n < 30:
            raise ValueError(
                f"Set de calibración muy pequeño ({n}). "
                "Conformal necesita al menos 30 ejemplos para ser útil."
            )

        y_true = np.asarray(y_true, dtype=int)
        y_probs = np.asarray(y_probs, dtype=float)

        # Score de nonconformidad = 1 - P(clase correcta)
        # Mide "qué tan mal" está la predicción para la clase real
        correct_probs = y_probs[np.arange(n), y_true]
        self.cal_scores_ = 1.0 - correct_probs

        # Quantil conformal (Ecuación 3.2, Vovk 2005)
        # Usamos ceil((n+1)(1-α))/n para garantía finita
        quantile_level = min(1.0, (1 - self.alpha) * (1 + 1 / n))
        self.quantile_ = float(np.quantile(self.cal_scores_, quantile_level))

        # Threshold: P(clase) >= threshold para entrar al set
        self.threshold_ = 1.0 - self.quantile_

        # Cobertura empírica en calibración (debería ser >= 1-alpha)
        pred_sets_cal = y_probs >= self.threshold_
        correct_in_set = pred_sets_cal[np.arange(n), y_true]
        self.coverage_ = float(correct_in_set.mean())

        return self

    def predict_sets(self, y_probs: np.ndarray) -> np.ndarray:
        """Retorna prediction sets conformales.

        Args:
            y_probs: (n, 2) probabilidades [P(UNDER), P(OVER)].

        Returns:
            sets: (n, 2) boolean array. sets[i, j] = True si clase j en el set.
        """
        if self.threshold_ is None:
            raise RuntimeError("Llama a fit() primero.")
        return np.asarray(y_probs, dtype=float) >= self.threshold_

    def predict_confidence(self, y_probs: np.ndarray) -> tuple:
        """Retorna métricas de confianza per-prediction.

        Args:
            y_probs: (n, 2) probabilidades [P(UNDER), P(OVER)].

        Returns:
            set_sizes: (n,) int — tamaño del prediction set.
                1 = confiado (solo una clase plausible)
                2 = incierto (ambas clases plausibles)
                0 = vacío (muy inusual, rarísimo)

            conformal_margins: (n,) float — margen de confianza.
                = max(P(OVER), P(UNDER)) - threshold
                Positivo grande → más confianza.
                Cerca de 0 → borderline.
                Negativo → la predicción está fuera del set (set vacío).

        Ejemplo:
            set_size=1, margin=0.12 → confiado, apostar Kelly completo
            set_size=2, margin=0.05 → borderline, reducir Kelly
            set_size=2, margin=0.20 → muy incierto (P cerca de 0.50), skip
        """
        y_probs = np.asarray(y_probs, dtype=float)
        pred_sets = self.predict_sets(y_probs)
        set_sizes = pred_sets.sum(axis=1).astype(int)

        # Margen = max_prob - threshold
        # Mide cuánto "sobra" la clase ganadora por encima del umbral
        max_probs = y_probs.max(axis=1)
        conformal_margins = max_probs - self.threshold_

        return set_sizes, conformal_margins

    def summary(self) -> dict:
        """Resumen del calibrador conformal."""
        if self.cal_scores_ is None:
            return {"fitted": False}
        return {
            "fitted": True,
            "alpha": self.alpha,
            "coverage_guarantee": 1 - self.alpha,
            "coverage_empirical": self.coverage_,
            "quantile": self.quantile_,
            "threshold": self.threshold_,
            "n_calibration": len(self.cal_scores_),
            "cal_score_mean": float(self.cal_scores_.mean()),
            "cal_score_std": float(self.cal_scores_.std()),
        }

    def __repr__(self):
        if self.threshold_ is None:
            return "ConformalClassifier(not fitted)"
        return (
            f"ConformalClassifier(α={self.alpha}, "
            f"coverage={self.coverage_:.1%}, "
            f"threshold={self.threshold_:.3f}, "
            f"n_cal={len(self.cal_scores_)})"
        )
