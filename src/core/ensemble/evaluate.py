"""Evaluacion de ensemble: combinar XGBoost + Neural Network.

Un ensemble combina las predicciones de multiples modelos para obtener
una prediccion mas robusta. La idea es que cada modelo comete errores
DIFERENTES, asi que al promediar sus probabilidades, los errores se
cancelan parcialmente.

Metodo: promedio ponderado de probabilidades
    p_ensemble = w * p_xgb + (1-w) * p_nn

El peso 'w' se optimiza en el validation set (datos que ningun modelo
uso directamente para entrenar).
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
from sklearn.metrics import accuracy_score, log_loss

from src.config import (
    DATASET_DB, XGBOOST_MODELS_DIR as XGB_MODEL_DIR, NBA_ML_MODELS_DIR as NN_MODEL_DIR,
    NBA_SHARED_MODELS_DIR, DROP_COLUMNS_ML, DROP_COLUMNS_UO,
)

DEFAULT_DATASET = "dataset_2012-26"
DATE_COLUMN = "Date"


def load_dataset(dataset_name):
    with sqlite3.connect(DATASET_DB) as con:
        return pd.read_sql_query(f'SELECT * FROM "{dataset_name}"', con)


def prepare_data_xgb(df, task="ML"):
    """Prepara datos en formato XGBoost (valores crudos, sin normalizar).

    XGBoost trabaja con arboles de decision que solo necesitan comparar
    'valor > umbral', asi que no necesita normalizacion.
    """
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
        # Filtrar pushes (clase 2) — solo UNDER(0) y OVER(1)
        mask = y != 2
        X = X[mask]
        y = y[mask]

    return X, y


def prepare_data_nn(df, task="ML"):
    """Prepara datos en formato Neural Network (normalizado L2).

    Las redes neuronales multiplican cada feature por un peso. Si una
    feature tiene valores grandes (ej: PTS=110) y otra pequenos (ej:
    FG_PCT=0.45), los pesos iniciales tratan ambas igual y la feature
    grande domina. La normalizacion L2 pone todas en la misma escala.

    L2 normalize: divide cada fila por su norma euclidiana
        x_norm = x / sqrt(sum(x_i^2))
    """
    data = df.copy()
    if DATE_COLUMN in data.columns:
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
        data = data.sort_values(DATE_COLUMN)

    if task == "ML":
        y = data["Home-Team-Win"].astype(int).to_numpy()
        X = data.drop(columns=DROP_COLUMNS_ML, errors="ignore")
    else:
        y = data["OU-Cover"].astype(int).to_numpy()
        # Filtrar pushes
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
    """Divide datos cronologicamente en train/val/test.

    No usamos shuffle porque son series de tiempo — el modelo solo
    puede ver el pasado para predecir el futuro.
    """
    n = len(X)
    val_start = int(n * (1 - val_size - test_size))
    test_start = int(n * (1 - test_size))
    return {
        "X_train": X[:val_start],
        "y_train": y[:val_start],
        "X_val": X[val_start:test_start],
        "y_val": y[val_start:test_start],
        "X_test": X[test_start:],
        "y_test": y[test_start:],
    }


class BoosterWrapper:
    """Wrapper para que sklearn CalibratedClassifierCV funcione con XGBoost.

    Necesario para deserializar los calibradores guardados con joblib.
    El calibrador fue entrenado con esta clase en los scripts de XGBoost.
    """
    def __init__(self, booster, num_class):
        self.booster = booster
        self.classes_ = np.arange(num_class)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return self.booster.predict(xgb.DMatrix(X))


def find_most_recent_model(directory, pattern, extension):
    """Encuentra el modelo MAS RECIENTE que coincide con el patron.

    Usamos fecha de modificacion (no accuracy) para asegurar que
    cargamos los modelos del ultimo training run. Esto evita cargar
    modelos viejos entrenados con features diferentes.
    """
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
    """Carga el mejor modelo XGBoost + su calibrador de probabilidades."""
    pattern = f"_{'ML' if task == 'ML' else 'UO'}_"
    model_path, acc = find_most_recent_model(XGB_MODEL_DIR, pattern, ".json")
    if model_path is None:
        raise FileNotFoundError(f"No XGBoost {task} model found in {XGB_MODEL_DIR}")
    print(f"  XGBoost {task}: {model_path.name} ({acc}%)")

    booster = xgb.Booster()
    booster.load_model(str(model_path))

    # Buscar calibrador correspondiente
    calib_path = model_path.with_name(f"{model_path.stem}_calibration.pkl")
    calibrator = None
    if calib_path.exists():
        calibrator = joblib.load(calib_path)
        print(f"  Calibrador: {calib_path.name}")

    return booster, calibrator


def load_nn_model(task="ML"):
    """Carga el mejor modelo de Neural Network."""
    pattern = f"Trained-Model-{'ML' if task == 'ML' else 'OU'}-"
    model_path, acc = find_most_recent_model(NN_MODEL_DIR, pattern, ".keras")
    if model_path is None:
        raise FileNotFoundError(f"No NN {task} model found in {NN_MODEL_DIR}")
    print(f"  NN {task}: {model_path.name} ({acc}%)")

    model = tf.keras.models.load_model(model_path, compile=False)
    return model


def predict_xgb(booster, calibrator, X):
    """Obtiene probabilidades del modelo XGBoost.

    Si tiene calibrador (Platt scaling / sigmoid), usa probabilidades
    calibradas. Si no, usa las probabilidades crudas del booster.
    """
    dmat = xgb.DMatrix(X)
    if calibrator is not None:
        return calibrator.predict_proba(X)
    else:
        return booster.predict(dmat)


def predict_nn(model, X):
    """Obtiene probabilidades de la Neural Network (softmax output)."""
    return model.predict(X, verbose=0)


def optimize_weight(probs_xgb, probs_nn, y_true, num_classes=2):
    """Busca el peso optimo w que minimiza log_loss en el validation set.

    Prueba w desde 0.0 (solo NN) hasta 1.0 (solo XGBoost) en pasos
    de 0.01. El w optimo es donde la mezcla tiene menor log_loss.

    Log loss penaliza predicciones CONFIADAS que estan MAL:
        Si predices 90% y aciertas: penalty baja (-log(0.9) = 0.105)
        Si predices 90% y fallas: penalty alta (-log(0.1) = 2.302)
    """
    labels = list(range(num_classes))
    best_w = 0.5
    best_loss = float("inf")
    results = []

    for w_int in range(0, 101):
        w = w_int / 100.0
        probs_mix = w * probs_xgb + (1 - w) * probs_nn
        loss = log_loss(y_true, probs_mix, labels=labels)
        acc = accuracy_score(y_true, np.argmax(probs_mix, axis=1))
        results.append((w, loss, acc))
        if loss < best_loss:
            best_loss = loss
            best_w = w

    return best_w, best_loss, results


def evaluate_task(task, dataset_name, val_size=0.1, test_size=0.1):
    """Evalua el ensemble para una tarea (ML o UO)."""
    print(f"\n{'='*60}")
    print(f"  ENSEMBLE {task}")
    print(f"{'='*60}")

    # --- Cargar datos ---
    df = load_dataset(dataset_name)
    print(f"\nDataset: {len(df)} filas")

    # Preparar datos en ambos formatos
    X_xgb, y_xgb = prepare_data_xgb(df, task)
    X_nn, y_nn = prepare_data_nn(df, task)

    # Verificar que y es identico (mismas filas, mismo orden)
    assert np.array_equal(y_xgb, y_nn), "Target arrays don't match between XGB and NN!"
    y = y_xgb
    print(f"Samples: {len(y)} (despues de filtrar pushes)" if task == "UO" else f"Samples: {len(y)}")

    # Dividir datos
    split_xgb = split_data(X_xgb, y, val_size, test_size)
    split_nn = split_data(X_nn, y, val_size, test_size)

    print(f"Train: {len(split_xgb['y_train'])}, Val: {len(split_xgb['y_val'])}, Test: {len(split_xgb['y_test'])}")

    # --- Cargar modelos ---
    print("\nCargando modelos:")
    booster, calibrator = load_xgb_model(task)
    nn_model = load_nn_model(task)

    # --- Predicciones en validation ---
    print("\nPrediciendo en validation set...")
    probs_xgb_val = predict_xgb(booster, calibrator, split_xgb["X_val"])
    probs_nn_val = predict_nn(nn_model, split_nn["X_val"])

    # --- Optimizar peso ---
    print("Optimizando peso w...")
    best_w, best_val_loss, weight_results = optimize_weight(
        probs_xgb_val, probs_nn_val, split_xgb["y_val"]
    )
    print(f"Peso optimo: w={best_w:.2f} (XGB={best_w:.0%}, NN={1-best_w:.0%})")
    print(f"Val log loss con w={best_w:.2f}: {best_val_loss:.4f}")

    # --- Evaluar en test ---
    print("\nEvaluando en test set...")
    probs_xgb_test = predict_xgb(booster, calibrator, split_xgb["X_test"])
    probs_nn_test = predict_nn(nn_model, split_nn["X_test"])

    # Ensemble con peso optimo
    probs_ensemble = best_w * probs_xgb_test + (1 - best_w) * probs_nn_test
    y_test = split_xgb["y_test"]

    # Metricas individuales
    acc_xgb = accuracy_score(y_test, np.argmax(probs_xgb_test, axis=1))
    acc_nn = accuracy_score(y_test, np.argmax(probs_nn_test, axis=1))
    acc_ensemble = accuracy_score(y_test, np.argmax(probs_ensemble, axis=1))

    loss_xgb = log_loss(y_test, probs_xgb_test, labels=list(range(2)))
    loss_nn = log_loss(y_test, probs_nn_test, labels=list(range(2)))
    loss_ensemble = log_loss(y_test, probs_ensemble, labels=list(range(2)))

    # --- Resultados ---
    print(f"\n{'─'*50}")
    print(f"  RESULTADOS {task}")
    print(f"{'─'*50}")
    print(f"{'Modelo':<20} {'Accuracy':>10} {'Log Loss':>10}")
    print(f"{'─'*50}")
    print(f"{'XGBoost':<20} {acc_xgb:>9.1%} {loss_xgb:>10.4f}")
    print(f"{'Neural Network':<20} {acc_nn:>9.1%} {loss_nn:>10.4f}")
    ens_label = f"Ensemble (w={best_w:.2f})"
    print(f"{ens_label:<20} {acc_ensemble:>9.1%} {loss_ensemble:>10.4f}")
    print(f"{'─'*50}")

    # Diferencia vs mejor individual
    best_individual = max(acc_xgb, acc_nn)
    delta = acc_ensemble - best_individual
    print(f"Ensemble vs mejor individual: {delta:+.1%}")

    # --- Analisis de acuerdo/desacuerdo ---
    pred_xgb = np.argmax(probs_xgb_test, axis=1)
    pred_nn = np.argmax(probs_nn_test, axis=1)
    pred_ens = np.argmax(probs_ensemble, axis=1)
    agree = pred_xgb == pred_nn
    disagree = ~agree

    print(f"\nAnalisis de diversidad:")
    print(f"  Acuerdo: {agree.sum()}/{len(agree)} ({agree.mean():.1%})")
    print(f"  Desacuerdo: {disagree.sum()}/{len(agree)} ({disagree.mean():.1%})")

    if disagree.sum() > 0:
        # Cuando los modelos no estan de acuerdo, que tan bien decide el ensemble?
        acc_ens_disagree = accuracy_score(y_test[disagree], pred_ens[disagree])
        acc_xgb_disagree = accuracy_score(y_test[disagree], pred_xgb[disagree])
        acc_nn_disagree = accuracy_score(y_test[disagree], pred_nn[disagree])
        print(f"  En desacuerdos:")
        print(f"    XGBoost: {acc_xgb_disagree:.1%}")
        print(f"    NN:      {acc_nn_disagree:.1%}")
        print(f"    Ensemble: {acc_ens_disagree:.1%}")

    # Mostrar curva de peso
    print(f"\nCurva de peso (w -> val_accuracy):")
    for w, loss, acc in weight_results:
        if w % 0.1 < 0.005 or abs(w - best_w) < 0.005:
            marker = " <-- optimo" if abs(w - best_w) < 0.005 else ""
            print(f"  w={w:.2f}: acc={acc:.1%}, loss={loss:.4f}{marker}")

    return {
        "task": task,
        "best_w": best_w,
        "acc_xgb": acc_xgb,
        "acc_nn": acc_nn,
        "acc_ensemble": acc_ensemble,
        "loss_xgb": loss_xgb,
        "loss_nn": loss_nn,
        "loss_ensemble": loss_ensemble,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate XGBoost + NN ensemble.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset table name.")
    parser.add_argument("--task", default="both", choices=["ML", "UO", "both"],
                        help="Which task to evaluate.")
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    args = parser.parse_args()

    tf.config.optimizer.set_experimental_options({"remapping": False})

    results = {}
    if args.task in ("ML", "both"):
        results["ML"] = evaluate_task("ML", args.dataset, args.val_size, args.test_size)
    if args.task in ("UO", "both"):
        results["UO"] = evaluate_task("UO", args.dataset, args.val_size, args.test_size)

    # Guardar pesos optimos para uso en main.py
    weights_path = NBA_SHARED_MODELS_DIR / "ensemble_weights.json"
    weights = {}
    for task, r in results.items():
        weights[task] = {
            "w_xgb": r["best_w"],
            "w_nn": round(1 - r["best_w"], 2),
            "acc_ensemble": round(r["acc_ensemble"], 4),
        }
    with open(weights_path, "w") as f:
        json.dump(weights, f, indent=2)
    print(f"\nPesos guardados en: {weights_path}")


if __name__ == "__main__":
    main()
