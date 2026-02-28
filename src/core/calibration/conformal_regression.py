"""Conformal Prediction para regression (intervalos de prediccion).

A diferencia de ConformalClassifier (que produce prediction SETS para clasificacion),
ConformalRegressor produce INTERVALOS de prediccion para valores continuos.

Metodo: Split Conformal con residuos absolutos.

Paso 1 — Calibracion:
    En el set de calibracion, para cada ejemplo:
      residual_i = |y_true_i - y_pred_i|
    Guardamos estos residuos ordenados.

Paso 2 — Quantil conformal:
    q_hat = quantil((1-alpha)(1 + 1/n)) de los residuos.
    Con alpha=0.10: el intervalo [pred - q_hat, pred + q_hat] contiene
    el valor real con probabilidad >= 90%.

Paso 3 — Confidence check:
    Para spreads: si |predicted_margin - spread_line| > q_hat,
    el spread esta fuera del intervalo → alta confianza en la direccion.

Uso en apuestas:
    Si predicted_margin = +5.0 y spread = -3.5, diferencia = 1.5
    Si q_hat = 10.0 → 1.5 < 10.0 → NO confiado (spread dentro del intervalo)
    Si q_hat = 1.0 → 1.5 > 1.0 → CONFIADO (spread fuera del intervalo)

Paper: Lei et al. (2018) "Distribution-Free Predictive Inference For Regression"
"""

import numpy as np


class ConformalRegressor:
    """Intervalos de prediccion conformales para regression.

    Garantia teorica: con alpha=0.10, al menos 90% de las predicciones
    futuras tendran el valor real dentro de [pred - q_hat, pred + q_hat].

    Atributos:
        alpha: nivel de error (1 - coverage). Default 0.10 = 90%.
        cal_residuals_: residuos absolutos del set de calibracion.
        quantile_: quantil conformal q_hat (en unidades del target, e.g. puntos).
        coverage_: cobertura empirica en el set de calibracion.
    """

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.cal_residuals_ = None
        self.quantile_ = None
        self.coverage_ = None

    def fit(self, y_pred: np.ndarray, y_true: np.ndarray):
        """Calibra el predictor conformal con el set de calibracion.

        Args:
            y_pred: (n,) predicciones del modelo.
            y_true: (n,) valores reales.

        Returns:
            self
        """
        y_pred = np.asarray(y_pred, dtype=float)
        y_true = np.asarray(y_true, dtype=float)
        n = len(y_true)

        if n < 30:
            raise ValueError(
                f"Set de calibracion muy pequeno ({n}). "
                "Conformal necesita al menos 30 ejemplos."
            )

        self.cal_residuals_ = np.abs(y_true - y_pred)

        # Quantil conformal (finite-sample correction)
        quantile_level = min(1.0, (1 - self.alpha) * (1 + 1 / n))
        self.quantile_ = float(np.quantile(self.cal_residuals_, quantile_level))

        # Cobertura empirica
        covered = self.cal_residuals_ <= self.quantile_
        self.coverage_ = float(covered.mean())

        return self

    def predict_interval(self, y_pred: np.ndarray):
        """Retorna intervalos de prediccion [pred - q, pred + q].

        Args:
            y_pred: (n,) predicciones del modelo.

        Returns:
            lower: (n,) limite inferior del intervalo.
            upper: (n,) limite superior del intervalo.
        """
        if self.quantile_ is None:
            raise RuntimeError("Llama a fit() primero.")
        y_pred = np.asarray(y_pred, dtype=float)
        return y_pred - self.quantile_, y_pred + self.quantile_

    def is_confident(self, predicted_margin: float, spread_line: float) -> bool:
        """Determina si el spread esta fuera del intervalo conformal.

        Si |predicted_margin - (-spread_line)| > quantile → confiado.

        Nota: spread_line es la linea del home (negativo = favorito).
        El home cubre cuando margin > -line, i.e. margin + line > 0.
        La "distancia" relevante es |predicted_margin + line| vs quantile.

        Args:
            predicted_margin: margin predicho (positivo = home gana por X).
            spread_line: spread del home (e.g. -5.5).

        Returns:
            True si la prediccion esta fuera del intervalo (alta confianza).
        """
        if self.quantile_ is None:
            return False
        distance = abs(predicted_margin + spread_line)
        return distance > self.quantile_

    def is_confident_residual(self, predicted_residual: float) -> bool:
        """For residual model (Camino B): confident when |residual| > q̂.

        Residual = margin + spread. Home covers when residual > 0.
        If |residual| > q̂, the prediction is outside the conformal interval
        around zero, meaning the model is confident about the direction.

        Mathematically equivalent to is_confident(margin, spread) since
        |margin + spread| = |residual|, but cleaner for residual models.
        """
        if self.quantile_ is None:
            return False
        return abs(predicted_residual) > self.quantile_

    def confidence_margin_residual(self, predicted_residual: float) -> float:
        """How much the residual exceeds the conformal interval.

        Positive = confident. Negative = not confident.
        """
        if self.quantile_ is None:
            return 0.0
        return abs(predicted_residual) - self.quantile_

    def confidence_margin(self, predicted_margin: float, spread_line: float) -> float:
        """Retorna cuanto excede la prediccion del intervalo conformal.

        Positivo = confiado (fuera del intervalo).
        Negativo = no confiado (dentro del intervalo).
        """
        if self.quantile_ is None:
            return 0.0
        distance = abs(predicted_margin + spread_line)
        return distance - self.quantile_

    def summary(self) -> dict:
        if self.cal_residuals_ is None:
            return {"fitted": False}
        return {
            "fitted": True,
            "alpha": self.alpha,
            "coverage_guarantee": 1 - self.alpha,
            "coverage_empirical": self.coverage_,
            "quantile": self.quantile_,
            "n_calibration": len(self.cal_residuals_),
            "cal_residual_mean": float(self.cal_residuals_.mean()),
            "cal_residual_std": float(self.cal_residuals_.std()),
            "cal_residual_median": float(np.median(self.cal_residuals_)),
        }

    def __repr__(self):
        if self.quantile_ is None:
            return "ConformalRegressor(not fitted)"
        return (
            f"ConformalRegressor(α={self.alpha}, "
            f"coverage={self.coverage_:.1%}, "
            f"q̂={self.quantile_:.2f}pts, "
            f"n_cal={len(self.cal_residuals_)})"
        )
