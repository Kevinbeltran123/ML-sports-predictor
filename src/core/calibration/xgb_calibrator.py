"""Calibración Platt (sigmoid) para modelos XGBoost.

Clase separada del script de entrenamiento para que joblib pueda
deserializarla desde cualquier contexto (runner, API, tests).

Uso:
    from src.core.calibration.xgb_calibrator import XGBCalibrator
    calibrator = XGBCalibrator(booster, num_classes=2)
    calibrator.fit(X_cal, y_cal)
    proba = calibrator.predict_proba(X_test)  # shape (n, 2)
"""

import numpy as np


class XGBCalibrator:
    """Calibración Platt (sigmoid) para XGBoost in-game.

    Ajusta LogisticRegression 1D sobre P(clase=1) del modelo base.
    El resultado sigue siendo shape (n, 2) compatible con conformal.
    """

    def __init__(self, booster, num_classes: int = 2):
        self.booster = booster
        self.num_classes = num_classes
        self._cal = None

    def fit(self, X, y):
        import xgboost as xgb
        from sklearn.linear_model import LogisticRegression

        raw = self.booster.predict(xgb.DMatrix(X))
        p1 = raw[:, 1].reshape(-1, 1)
        self._cal = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        self._cal.fit(p1, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        import xgboost as xgb

        if isinstance(X, np.ndarray):
            raw = self.booster.predict(xgb.DMatrix(X))
        else:
            raw = self.booster.predict(X)
        p1 = raw[:, 1].reshape(-1, 1)
        return self._cal.predict_proba(p1)  # shape (n, 2)
