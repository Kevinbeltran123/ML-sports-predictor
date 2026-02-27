"""Diagnostico de calibracion de modelos.

CALIBRACION = que tan bien las probabilidades predichas reflejan la realidad.

Si el modelo dice "70% de ganar" para 100 partidos, deberian ganar ~70.
Si ganan 85, el modelo esta MAL calibrado (subconfiado).
Si ganan 55, el modelo esta MAL calibrado (sobreconfiado).

Metricas:
  - ECE (Expected Calibration Error): error promedio ponderado entre
    probabilidad predicha y frecuencia real, agrupado en bins.
    ECE = sum(|bin_count/total| * |avg_predicted - avg_actual|)
    Rango: 0 (perfecto) a 1 (terrible). < 0.05 es bueno.

  - Brier Score: MSE entre probabilidad predicha y resultado real (0 o 1).
    Brier = mean((p_predicted - y_actual)^2)
    Rango: 0 (perfecto) a 1 (terrible). Combina calibracion + discriminacion.

  - Reliability Diagram: grafica de calibracion. Eje X = probabilidad predicha
    (en bins), Eje Y = frecuencia real de acierto. Si esta calibrado, la curva
    sigue la diagonal perfecta y=x.
"""

import argparse
import json
import re
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from src.config import (
    DATASET_DB, XGBOOST_MODELS_DIR as XGB_MODEL_DIR, NBA_ML_MODELS_DIR as NN_MODEL_DIR,
    NBA_SHARED_MODELS_DIR, DROP_COLUMNS_ML, DROP_COLUMNS_UO,
)

DEFAULT_DATASET = "dataset_2012-26"
DATE_COLUMN = "Date"


class BoosterWrapper:
    """Necesario para deserializar calibradores XGBoost guardados con joblib."""
    def __init__(self, booster, num_class):
        self.booster = booster
        self.classes_ = np.arange(num_class)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return self.booster.predict(xgb.DMatrix(X))


def load_dataset(dataset_name):
    with sqlite3.connect(DATASET_DB) as con:
        return pd.read_sql_query(f'SELECT * FROM "{dataset_name}"', con)


def prepare_data_xgb(df, task="ML"):
    """Datos en formato XGBoost (sin normalizar)."""
    data = df.copy()
    if DATE_COLUMN in data.columns:
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
        data = data.sort_values(DATE_COLUMN)
    if task == "ML":
        y = data["Home-Team-Win"].astype(int).to_numpy()
        X = data.drop(columns=DROP_COLUMNS_ML, errors="ignore").astype(float).to_numpy()
    else:
        y = data["OU-Cover"].astype(int).to_numpy()
        # DROP_COLUMNS_UO ya no incluye "OU" (se mantiene como feature)
        X = data.drop(columns=DROP_COLUMNS_UO, errors="ignore").astype(float).to_numpy()
        mask = y != 2
        X, y = X[mask], y[mask]
    return X, y


def prepare_data_nn(df, task="ML"):
    """Datos en formato NN (normalizado L2)."""
    data = df.copy()
    if DATE_COLUMN in data.columns:
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
        data = data.sort_values(DATE_COLUMN)
    if task == "ML":
        y = data["Home-Team-Win"].astype(int).to_numpy()
        X = data.drop(columns=DROP_COLUMNS_ML, errors="ignore")
    else:
        y = data["OU-Cover"].astype(int).to_numpy()
        mask = y != 2
        data = data[mask].reset_index(drop=True)
        y = y[mask]
        # DROP_COLUMNS_UO no incluye "OU", asi que se mantiene como feature
        X = data.drop(columns=DROP_COLUMNS_UO, errors="ignore")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    X = X.to_numpy(dtype=float)
    X = tf.keras.utils.normalize(X, axis=1)
    return X, y


def split_data(X, y, val_size=0.1, test_size=0.1):
    n = len(X)
    val_start = int(n * (1 - val_size - test_size))
    test_start = int(n * (1 - test_size))
    return (
        X[:val_start], y[:val_start],
        X[val_start:test_start], y[val_start:test_start],
        X[test_start:], y[test_start:],
    )


def find_most_recent_model(directory, pattern, extension):
    best_mtime = -1
    best_path = None
    best_acc = -1
    for path in directory.glob(f"*{pattern}*{extension}"):
        mtime = path.stat().st_mtime
        if mtime > best_mtime:
            best_mtime = mtime
            best_path = path
            match = re.search(r"(\d+\.\d+)", path.stem)
            best_acc = float(match.group(1)) if match else -1
    return best_path, best_acc


def load_xgb_model(task="ML"):
    pattern = f"_{'ML' if task == 'ML' else 'UO'}_"
    model_path, acc = find_most_recent_model(XGB_MODEL_DIR, pattern, ".json")
    if model_path is None:
        raise FileNotFoundError(f"No XGBoost {task} model found")
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    calib_path = model_path.with_name(f"{model_path.stem}_calibration.pkl")
    calibrator = joblib.load(calib_path) if calib_path.exists() else None
    return booster, calibrator, model_path.name


def load_nn_model(task="ML"):
    pattern = f"Trained-Model-{'ML' if task == 'ML' else 'OU'}-"
    model_path, acc = find_most_recent_model(NN_MODEL_DIR, pattern, ".keras")
    if model_path is None:
        raise FileNotFoundError(f"No NN {task} model found")
    model = tf.keras.models.load_model(model_path, compile=False)
    return model, model_path.name


def predict_xgb(booster, calibrator, X):
    if calibrator is not None:
        return calibrator.predict_proba(X)
    return booster.predict(xgb.DMatrix(X))


def predict_nn(model, X):
    return model.predict(X, verbose=0)


# ─── Metricas de calibracion ───

def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error.

    Divide las predicciones en 'bins' (grupos) por probabilidad predicha.
    Para cada bin, compara la probabilidad promedio predicha con la
    frecuencia real de acierto. El ECE es el promedio ponderado de
    estas diferencias.

    Ejemplo con n_bins=10:
      Bin [0.6, 0.7): 200 predicciones, probabilidad promedio = 0.65,
                       frecuencia real = 0.72. Error = |0.72 - 0.65| = 0.07
      El peso del bin es 200/total_predicciones.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    bin_data = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if i == n_bins - 1:  # Ultimo bin incluye 1.0
            mask = (y_prob >= lo) & (y_prob <= hi)
        count = mask.sum()
        if count == 0:
            bin_data.append({"range": f"[{lo:.1f},{hi:.1f})", "count": 0,
                             "avg_pred": 0, "avg_actual": 0, "gap": 0})
            continue

        avg_pred = y_prob[mask].mean()
        avg_actual = y_true[mask].mean()
        gap = abs(avg_actual - avg_pred)
        ece += (count / total) * gap

        bin_data.append({
            "range": f"[{lo:.1f},{hi:.1f})",
            "count": int(count),
            "avg_pred": round(avg_pred, 3),
            "avg_actual": round(avg_actual, 3),
            "gap": round(gap, 3),
        })

    return ece, bin_data


def print_reliability_diagram(bin_data, model_name):
    """Imprime diagrama de calibracion en texto (ASCII).

    Cada linea muestra: rango de probabilidad, barra visual, y el gap.
    Si la barra llega a la posicion correcta en la diagonal, esta calibrado.
    """
    print(f"\n  Reliability Diagram: {model_name}")
    print(f"  {'Bin':<12} {'N':>5}  {'Pred':>5} {'Real':>5} {'Gap':>5}  Visual")
    print(f"  {'─'*60}")

    for b in bin_data:
        if b["count"] == 0:
            print(f"  {b['range']:<12} {b['count']:>5}  {'---':>5} {'---':>5} {'---':>5}")
            continue

        # Barra visual: '=' muestra donde esta la prediccion, '*' donde esta la realidad
        bar_len = 30
        pred_pos = int(b["avg_pred"] * bar_len)
        actual_pos = int(b["avg_actual"] * bar_len)

        bar = list("." * (bar_len + 1))
        bar[pred_pos] = "P"  # Prediccion
        bar[actual_pos] = "R"  # Realidad
        if pred_pos == actual_pos:
            bar[pred_pos] = "="  # Coinciden (bien calibrado)

        # Color del gap: bueno (<0.03), ok (0.03-0.07), malo (>0.07)
        gap_indicator = " " if b["gap"] < 0.03 else "!" if b["gap"] < 0.07 else "!!"

        print(
            f"  {b['range']:<12} {b['count']:>5}  "
            f"{b['avg_pred']:>5.3f} {b['avg_actual']:>5.3f} {b['gap']:>5.3f}  "
            f"{''.join(bar)} {gap_indicator}"
        )


def evaluate_calibration(model_name, y_true, y_prob_positive):
    """Evalua calibracion completa de un modelo."""
    ece, bin_data = compute_ece(y_true, y_prob_positive)
    brier = brier_score_loss(y_true, y_prob_positive)

    # Accuracy y log loss
    y_pred = (y_prob_positive >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    # Clamp probabilities para evitar log(0)
    y_prob_clipped = np.clip(y_prob_positive, 1e-7, 1 - 1e-7)
    ll = log_loss(y_true, y_prob_clipped)

    print(f"\n{'─'*55}")
    print(f"  {model_name}")
    print(f"{'─'*55}")
    print(f"  Accuracy:   {acc:.1%}")
    print(f"  Log Loss:   {ll:.4f}")
    print(f"  Brier:      {brier:.4f}  (0=perfecto, 0.25=aleatorio)")
    print(f"  ECE:        {ece:.4f}  (0=perfecto, <0.05=bueno)")

    # Diagnostico
    if ece < 0.03:
        print(f"  Diagnostico: BIEN calibrado")
    elif ece < 0.05:
        print(f"  Diagnostico: Calibracion aceptable")
    elif ece < 0.10:
        print(f"  Diagnostico: Calibracion mejorable")
    else:
        print(f"  Diagnostico: MAL calibrado - necesita correccion")

    print_reliability_diagram(bin_data, model_name)

    return {
        "model": model_name,
        "accuracy": round(acc, 4),
        "log_loss": round(ll, 4),
        "brier": round(brier, 4),
        "ece": round(ece, 4),
    }


def evaluate_task(task, dataset_name):
    """Evalua calibracion de todos los modelos para una tarea."""
    print(f"\n{'='*60}")
    print(f"  CALIBRACION {task}")
    print(f"{'='*60}")

    df = load_dataset(dataset_name)

    # Preparar datos para ambos formatos
    X_xgb, y_xgb = prepare_data_xgb(df, task)
    X_nn, y_nn = prepare_data_nn(df, task)
    assert np.array_equal(y_xgb, y_nn)
    y = y_xgb

    # Split: usamos test set (ultimo 10%) para evaluar calibracion
    _, _, _, _, X_xgb_test, y_test = split_data(X_xgb, y)
    _, _, _, _, X_nn_test, _ = split_data(X_nn, y)

    print(f"Test set: {len(y_test)} samples")

    # --- Cargar modelos ---
    booster, calibrator, xgb_name = load_xgb_model(task)
    nn_model, nn_name = load_nn_model(task)
    print(f"XGBoost: {xgb_name}")
    print(f"NN: {nn_name}")

    # --- Predicciones ---
    probs_xgb = predict_xgb(booster, calibrator, X_xgb_test)
    probs_nn = predict_nn(nn_model, X_nn_test)

    # Probabilidad de la clase positiva (clase 1)
    # Para ML: prob(Home-Team-Win=1)
    # Para UO: prob(Over=1)
    p_xgb = probs_xgb[:, 1]
    p_nn = probs_nn[:, 1]

    # Ensemble
    weights_path = NBA_SHARED_MODELS_DIR / "ensemble_weights.json"
    if weights_path.exists():
        with open(weights_path) as f:
            weights = json.load(f)
        w = weights.get(task, {}).get("w_xgb", 0.5)
    else:
        w = 0.5
    p_ensemble = w * p_xgb + (1 - w) * p_nn

    # --- Evaluar calibracion de cada modelo ---
    results = []

    # XGBoost con calibracion existente (Platt scaling)
    suffix = " (calibrado)" if calibrator else " (sin calibrar)"
    results.append(evaluate_calibration(f"XGBoost{suffix}", y_test, p_xgb))

    # XGBoost SIN calibracion (para comparar efecto del Platt scaling)
    if calibrator is not None:
        p_xgb_raw = booster.predict(xgb.DMatrix(X_xgb_test))[:, 1]
        results.append(evaluate_calibration("XGBoost (sin calibrar)", y_test, p_xgb_raw))

    # Neural Network (softmax crudo)
    results.append(evaluate_calibration("Neural Network", y_test, p_nn))

    # Ensemble
    results.append(evaluate_calibration(f"Ensemble (w={w:.2f})", y_test, p_ensemble))

    # --- Tabla resumen ---
    print(f"\n{'='*60}")
    print(f"  RESUMEN {task}")
    print(f"{'='*60}")
    print(f"{'Modelo':<30} {'Acc':>6} {'LogL':>7} {'Brier':>7} {'ECE':>7}")
    print(f"{'─'*60}")
    for r in results:
        print(f"{r['model']:<30} {r['accuracy']:>5.1%} {r['log_loss']:>7.4f} {r['brier']:>7.4f} {r['ece']:>7.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate model calibration.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--task", default="both", choices=["ML", "UO", "both"])
    args = parser.parse_args()

    tf.config.optimizer.set_experimental_options({"remapping": False})

    if args.task in ("ML", "both"):
        evaluate_task("ML", args.dataset)
    if args.task in ("UO", "both"):
        evaluate_task("UO", args.dataset)


if __name__ == "__main__":
    main()
