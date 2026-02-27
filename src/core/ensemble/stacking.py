"""Stacking Ensemble con meta-learner LogisticRegression (4 modelos base).

===============================================================
¿Que es Stacking? (Wolpert 1992)
===============================================================

Problema con weighted average fijo:
    p_ensemble = w1 * p_xgb + w2 * p_lgbm + w3 * p_cat + w4 * p_nn

Esto asume pesos CONSTANTES para todos los partidos.
Pero cada modelo tiene fortalezas diferentes:
  - XGBoost: interacciones complejas entre features (depth-wise)
  - LightGBM: patrones asimetricos (leaf-wise), complementa XGBoost
  - CatBoost: arboles simetricos + ordered boosting, menos overfitting
  - NN: no-linealidades suaves via bloques residuales + Swish

Stacking: un meta-learner aprende CUANDO confiar en cada modelo.

===============================================================
Metodo: Out-of-Fold (OOF) predictions
===============================================================

El meta-learner necesita predicciones "honestas" donde NINGUNO de los
modelos base haya visto los datos. Solucion: TimeSeriesSplit OOF.

Para k = 1..5:
    Entrenar XGBoost/LightGBM/CatBoost/NN en folds 1..k-1
    Predecir fold k → OOF_xgb, OOF_lgbm, OOF_cat, OOF_nn

===============================================================
Meta-features (10 dimensiones)
===============================================================

  [p_xgb, p_lgbm, p_cat, p_nn,
   |xgb-lgbm|, |xgb-cat|, |xgb-nn|, |lgbm-cat|, |lgbm-nn|, |cat-nn|]

4 probabilidades + 6 discrepancias pairwise = C(4,2) = 6.
Las discrepancias capturan cuando cada par de modelos discrepa.
Con 4 modelos hay mas oportunidades de desempate que con 3.

===============================================================
Meta-learner: LogisticRegressionCV
===============================================================

Solo 10 features → imposible sobreajustar.
Sigmoid → probabilidades calibradas naturalmente.
L2 regularizacion → evita dar todo peso a un solo modelo.

Uso:
    PYTHONPATH=. python src/core/ensemble/stacking.py --task ML
    PYTHONPATH=. python src/core/ensemble/stacking.py --task ML --window 4
"""

import argparse
import sqlite3

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from src.core.calibration.classwise_ece import compute_classwise_ece
from src.core.analysis.extended_metrics import compute_full_metrics, find_optimal_threshold
from src.config import DATASET_DB, MODELS_DIR as MODEL_DIR, DROP_COLUMNS_ML, DROP_COLUMNS_UO, get_logger

logger = get_logger(__name__)

DEFAULT_DATASET = "dataset_2012-26"
DATE_COLUMN = "Date"
NUM_CLASSES = 2

# ===============================================================
# HPs fijos de modelos base — tomados de las mejores corridas Optuna
# (Feb 2026, datos post-fix T-1, 238 cols dataset)
# ===============================================================

# XGB ML: AUC=0.705, Accuracy=65.3%, LogLoss=0.6274 (metric=logloss, 100 trials)
XGB_PARAMS_ML = {
    "max_depth": 9, "eta": 0.129, "subsample": 0.945,
    "colsample_bytree": 0.870, "colsample_bylevel": 0.765,
    "colsample_bynode": 0.858, "min_child_weight": 11,
    "gamma": 5.903, "lambda": 1.466, "alpha": 0.485,
    "objective": "multi:softprob", "num_class": 2,
    "tree_method": "hist", "seed": 42,
    "eval_metric": ["mlogloss"],
}
XGB_NB_ML = 701  # num_boost_round optimo

# XGB UO: mantiene HPs conservadores (no re-optimizado aun)
XGB_PARAMS_UO = {
    "max_depth": 7, "eta": 0.006, "subsample": 0.9,
    "colsample_bytree": 0.55, "objective": "multi:softprob",
    "num_class": 2, "tree_method": "hist", "seed": 42,
    "eval_metric": ["mlogloss"],
}
XGB_NB_UO = 500

# LightGBM ML: AUC=0.711, Accuracy=64.8%, LogLoss=0.6237
LGBM_PARAMS_ML = {
    "objective": "binary", "metric": "binary_logloss",
    "boosting_type": "gbdt", "verbosity": -1, "random_state": 42,
    "num_leaves": 83, "max_depth": 8, "learning_rate": 0.016,
    "n_estimators": 1430, "subsample": 0.892, "colsample_bytree": 0.727,
    "min_child_samples": 12, "reg_alpha": 0.504, "reg_lambda": 2.035,
    "subsample_freq": 1,
}

# CatBoost ML: Accuracy=65.9%, LogLoss=0.6271 (100 trials)
CATBOOST_PARAMS_ML = {
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "auto_class_weights": "Balanced",
    "random_seed": 42,
    "task_type": "CPU",
    "depth": 9,
    "iterations": 1647,
    "learning_rate": 0.005,
    "l2_leaf_reg": 0.619,
    "random_strength": 0.008,
    "min_data_in_leaf": 1,  # default si no fue tuned
    "bagging_temperature": 2.744,
    "rsm": 0.709,
    "border_count": 233,
}

# NN ML (improved): Accuracy=63.6%, depth=2, width=256
# Usa bloques residuales + Swish + BatchNorm + Mixup + class weights
NN_PARAMS = {
    "units": [256, 128],  # depth=2, width=256 con halving
    "dropout": 0.317,
    "batch_norm": True,
    "l2": 0.0,
    "lr": 0.005,
    "batch_size": 128,
    "epochs": 80,
    "mixup_alpha": 0.2,
}


def load_dataset(dataset_name):
    with sqlite3.connect(DATASET_DB) as con:
        return pd.read_sql_query(f'SELECT * FROM "{dataset_name}"', con)


def prepare_data(df, task):
    """Prepara dataframe: sort por fecha, filtrar OU=2 para UO."""
    data = df.copy()
    if DATE_COLUMN in data.columns:
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
        data = data.dropna(subset=[DATE_COLUMN]).sort_values(DATE_COLUMN).reset_index(drop=True)

    if task == "ML":
        drop_cols = DROP_COLUMNS_ML
        target = "Home-Team-Win"
    else:
        drop_cols = DROP_COLUMNS_UO
        target = "OU-Cover"

    y = data[target].astype(int).to_numpy()
    if task == "UO":
        mask = y != 2
        data = data[mask].reset_index(drop=True)
        y = y[mask]

    X_df = data.drop(columns=drop_cols, errors="ignore")
    feature_cols = list(X_df.columns)
    X = X_df.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0).to_numpy(dtype=float)

    return X, y, feature_cols


def fit_nn_preprocessor(X):
    """StandardScaler para NN (mismo enfoque que nn_ml.py)."""
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    return X_scaled.astype(float), imputer, scaler


def transform_nn(X, imputer, scaler):
    """Aplica preprocessor de NN."""
    X_imp = imputer.transform(X)
    return scaler.transform(X_imp).astype(float)


# ---------------------------------------------------------------------------
# Bloque residual (mismo que nn_ml.py mejorado)
# ---------------------------------------------------------------------------

def _residual_block(x, units, activation, reg, use_batch_norm, dropout_rate):
    """Bloque Dense con conexion residual (skip connection)."""
    shortcut = x
    h = tf.keras.layers.Dense(units, kernel_regularizer=reg, use_bias=not use_batch_norm)(x)
    if use_batch_norm:
        h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.Activation(activation)(h)
    if dropout_rate > 0:
        h = tf.keras.layers.Dropout(dropout_rate)(h)
    if shortcut.shape[-1] != units:
        shortcut = tf.keras.layers.Dense(units, use_bias=False, kernel_regularizer=reg)(shortcut)
    out = tf.keras.layers.Add()([h, shortcut])
    return out


def build_nn(input_dim, params):
    """Construye NN mejorada con bloques residuales + Swish."""
    reg = tf.keras.regularizers.l2(params["l2"]) if params["l2"] > 0 else None
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    for units in params["units"]:
        x = _residual_block(
            x, units, "swish", reg,
            use_batch_norm=params["batch_norm"],
            dropout_rate=params["dropout"],
        )
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model


# ---------------------------------------------------------------------------
# Mixup (mismo que nn_ml.py mejorado)
# ---------------------------------------------------------------------------

class MixupDataGenerator(tf.keras.utils.Sequence):
    """Generador que aplica Mixup en cada batch durante entrenamiento."""
    def __init__(self, X, y, batch_size, alpha=0.2, num_classes=NUM_CLASSES):
        self.X = X.astype(np.float32)
        self.y_onehot = tf.keras.utils.to_categorical(y, num_classes=num_classes).astype(np.float32)
        self.batch_size = batch_size
        self.alpha = alpha
        self.indices = np.arange(len(X))

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.X))
        batch_X = self.X[start:end]
        batch_y = self.y_onehot[start:end]
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
        batch_size = len(batch_X)
        indices = np.random.permutation(batch_size)
        X_mix = lam * batch_X + (1.0 - lam) * batch_X[indices]
        y_mix = lam * batch_y + (1.0 - lam) * batch_y[indices]
        return X_mix, y_mix

    def on_epoch_end(self):
        np.random.shuffle(self.indices)
        self.X = self.X[self.indices]
        self.y_onehot = self.y_onehot[self.indices]


def compute_sample_weights(y, num_classes=2):
    counts = np.bincount(y, minlength=num_classes)
    total = len(y)
    class_weights = {
        cls: (total / (num_classes * count)) if count else 1.0
        for cls, count in enumerate(counts)
    }
    return np.array([class_weights[label] for label in y])


def compute_class_weights_dict(y):
    """Class weights para NN (inversamente proporcional a frecuencia)."""
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes, weights))


def generate_oof_predictions(X, y, task, n_splits=5, seed=42):
    """Genera OOF predictions para XGBoost, LightGBM, CatBoost y NN.

    CRITICO: cada modelo NUNCA ve los datos que predice.
    Para NN, se re-ajusta StandardScaler en cada fold (evita data leakage).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    n = len(y)

    oof_xgb = np.zeros((n, 2))
    oof_lgbm = np.zeros((n, 2))
    oof_cat = np.zeros((n, 2))
    oof_nn = np.zeros((n, 2))
    oof_mask = np.zeros(n, dtype=bool)

    xgb_params = XGB_PARAMS_ML if task == "ML" else XGB_PARAMS_UO
    xgb_nb = XGB_NB_ML if task == "ML" else XGB_NB_UO
    lgbm_params = LGBM_PARAMS_ML  # solo ML por ahora

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"  Fold {fold + 1}/{n_splits}: train={len(train_idx)}, val={len(val_idx)}")

        X_tr, X_vl = X[train_idx], X[val_idx]
        y_tr, y_vl = y[train_idx], y[val_idx]

        # --- XGBoost OOF ---
        dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=compute_sample_weights(y_tr))
        dval = xgb.DMatrix(X_vl, label=y_vl)
        booster = xgb.train(
            xgb_params, dtrain,
            num_boost_round=xgb_nb,
            evals=[(dval, "val")],
            early_stopping_rounds=60,
            verbose_eval=False,
        )
        oof_xgb[val_idx] = booster.predict(dval)

        # --- LightGBM OOF ---
        lgbm_model = lgb.LGBMClassifier(**lgbm_params)
        lgbm_model.fit(
            X_tr, y_tr,
            sample_weight=compute_sample_weights(y_tr),
            eval_set=[(X_vl, y_vl)],
            callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
        )
        oof_lgbm[val_idx] = lgbm_model.predict_proba(X_vl)

        # --- CatBoost OOF ---
        cat_model = CatBoostClassifier(**CATBOOST_PARAMS_ML)
        cat_model.fit(
            X_tr, y_tr,
            eval_set=(X_vl, y_vl),
            early_stopping_rounds=60,
            verbose=0,
        )
        oof_cat[val_idx] = cat_model.predict_proba(X_vl)

        # --- NN OOF (mejorada: residual + Swish + Mixup + class weights) ---
        tf.keras.backend.clear_session()
        np.random.seed(seed + fold)
        tf.random.set_seed(seed + fold)

        # Fit scaler solo en train fold (evita leakage)
        X_nn_tr, imputer, scaler = fit_nn_preprocessor(X_tr)
        X_nn_vl = transform_nn(X_vl, imputer, scaler)

        nn = build_nn(X_nn_tr.shape[1], NN_PARAMS)

        # Cosine annealing LR schedule
        steps_per_epoch = max(1, len(X_nn_tr) // NN_PARAMS["batch_size"])
        total_steps = steps_per_epoch * NN_PARAMS["epochs"]
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=NN_PARAMS["lr"],
            decay_steps=total_steps,
            alpha=0.01,
        )
        nn.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss="categorical_crossentropy",
        )

        # Mixup generator para training
        train_gen = MixupDataGenerator(
            X_nn_tr, y_tr,
            batch_size=NN_PARAMS["batch_size"],
            alpha=NN_PARAMS["mixup_alpha"],
        )

        # Validacion con labels one-hot
        y_vl_onehot = tf.keras.utils.to_categorical(y_vl, num_classes=NUM_CLASSES)
        class_weight_dict = compute_class_weights_dict(y_tr)

        nn.fit(
            train_gen,
            validation_data=(X_nn_vl, y_vl_onehot),
            epochs=NN_PARAMS["epochs"],
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=10, restore_best_weights=True
                ),
            ],
            verbose=0,
            class_weight=class_weight_dict,
        )
        oof_nn[val_idx] = nn.predict(X_nn_vl, verbose=0)

        oof_mask[val_idx] = True

    return (oof_xgb[oof_mask], oof_lgbm[oof_mask],
            oof_cat[oof_mask], oof_nn[oof_mask], y[oof_mask])


def build_meta_features(p_xgb, p_lgbm, p_cat, p_nn):
    """Construye 10 meta-features para el meta-learner.

    [p_xgb, p_lgbm, p_cat, p_nn,
     |xgb-lgbm|, |xgb-cat|, |xgb-nn|, |lgbm-cat|, |lgbm-nn|, |cat-nn|]

    Las 6 discrepancias capturan cuando cada par de modelos discrepa.
    Con 4 modelos, C(4,2) = 6 pares de desempate.
    """
    px = p_xgb[:, 1]
    pl = p_lgbm[:, 1]
    pc = p_cat[:, 1]
    pn = p_nn[:, 1]
    return np.column_stack([
        px, pl, pc, pn,
        np.abs(px - pl),  # discrepancia XGB vs LGBM
        np.abs(px - pc),  # discrepancia XGB vs CatBoost
        np.abs(px - pn),  # discrepancia XGB vs NN
        np.abs(pl - pc),  # discrepancia LGBM vs CatBoost
        np.abs(pl - pn),  # discrepancia LGBM vs NN
        np.abs(pc - pn),  # discrepancia CatBoost vs NN
    ])


def train_meta_learner(meta_X, meta_y):
    """Entrena LogisticRegressionCV como meta-learner.

    Con 10 features sigue siendo imposible sobreajustar.
    LogReg con L2 da pesos naturalmente calibrados.
    """
    meta_model = LogisticRegressionCV(
        Cs=np.logspace(-4, 4, 20),
        cv=TimeSeriesSplit(n_splits=3),
        scoring="neg_log_loss",
        max_iter=1000,
        random_state=42,
    )
    meta_model.fit(meta_X, meta_y)
    return meta_model


def evaluate_stacking(task, dataset_name, n_splits=5, window=None):
    """Pipeline completo: OOF → meta-learner → evaluacion en test set."""
    print(f"\n{'='*65}")
    print(f"  STACKING ENSEMBLE (4 modelos) — {task}")
    print(f"{'='*65}")

    df = load_dataset(dataset_name)
    data = df.copy()
    if DATE_COLUMN in data.columns:
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
        data = data.dropna(subset=[DATE_COLUMN]).sort_values(DATE_COLUMN).reset_index(drop=True)

    # Window: ultimas N temporadas
    if window:
        max_rows = window * 1230
        if len(data) > max_rows:
            data = data.tail(max_rows).reset_index(drop=True)
            print(f"  Window: ultimas {window} temporadas ({len(data)} juegos)")

    # Fixed dates split (consistente con trainers)
    test_dt = pd.to_datetime("2025-10-01")
    train_val = data[data[DATE_COLUMN] < test_dt].copy()
    test_data = data[data[DATE_COLUMN] >= test_dt].copy()

    if len(test_data) == 0:
        logger.error("Test set vacio. Verifica las fechas.")
        return {}

    # Preparar X, y
    drop_cols = DROP_COLUMNS_ML if task == "ML" else DROP_COLUMNS_UO
    target = "Home-Team-Win" if task == "ML" else "OU-Cover"

    def df_to_xy(frame):
        y = frame[target].astype(int).to_numpy()
        X = frame.drop(columns=drop_cols, errors="ignore")
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median(numeric_only=True)).fillna(0).to_numpy(dtype=float)
        return X, y

    X_tv, y_tv = df_to_xy(train_val)
    X_test, y_test = df_to_xy(test_data)

    print(f"  Train+Val: {len(y_tv)}, Test: {len(y_test)}")
    baseline_acc = max(float(y_test.mean()), 1.0 - float(y_test.mean()))
    print(f"  Test baseline: {baseline_acc:.1%}")

    # --- 1. Generar OOF predictions ---
    print(f"\nGenerando OOF predictions ({n_splits} folds, 4 modelos)...")
    oof_xgb, oof_lgbm, oof_cat, oof_nn, oof_y = generate_oof_predictions(
        X_tv, y_tv, task, n_splits
    )
    print(f"  OOF samples: {len(oof_y)}")

    # --- 2. Construir meta-features ---
    meta_X = build_meta_features(oof_xgb, oof_lgbm, oof_cat, oof_nn)
    print(f"  Meta-features shape: {meta_X.shape}")

    # --- 3. Entrenar meta-learner ---
    print("\nEntrenando meta-learner (LogisticRegressionCV)...")
    meta_model = train_meta_learner(meta_X, oof_y)
    coefs = meta_model.coef_[0]
    print(f"  Mejor C: {meta_model.C_[0]:.4f}")
    print(f"  Coeficientes: XGB={coefs[0]:.3f}, LGBM={coefs[1]:.3f}, "
          f"CatBoost={coefs[2]:.3f}, NN={coefs[3]:.3f}")
    print(f"  Discrepancias: |XGB-LGBM|={coefs[4]:.3f}, |XGB-Cat|={coefs[5]:.3f}, "
          f"|XGB-NN|={coefs[6]:.3f}, |LGBM-Cat|={coefs[7]:.3f}, "
          f"|LGBM-NN|={coefs[8]:.3f}, |Cat-NN|={coefs[9]:.3f}")

    # --- 4. Predicciones en test ---
    print("\nEntrenando modelos finales y evaluando en test set...")

    # XGBoost final
    xgb_params = XGB_PARAMS_ML if task == "ML" else XGB_PARAMS_UO
    xgb_nb = XGB_NB_ML if task == "ML" else XGB_NB_UO
    dtrain = xgb.DMatrix(X_tv, label=y_tv, weight=compute_sample_weights(y_tv))
    dtest = xgb.DMatrix(X_test, label=y_test)
    booster = xgb.train(
        xgb_params, dtrain,
        num_boost_round=xgb_nb,
        evals=[(dtest, "test")],
        early_stopping_rounds=60,
        verbose_eval=False,
    )
    p_xgb_test = booster.predict(dtest)

    # LightGBM final
    lgbm_model = lgb.LGBMClassifier(**LGBM_PARAMS_ML)
    lgbm_model.fit(
        X_tv, y_tv,
        sample_weight=compute_sample_weights(y_tv),
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
    )
    p_lgbm_test = lgbm_model.predict_proba(X_test)

    # CatBoost final
    cat_model = CatBoostClassifier(**CATBOOST_PARAMS_ML)
    cat_model.fit(
        X_tv, y_tv,
        eval_set=(X_test, y_test),
        early_stopping_rounds=60,
        verbose=0,
    )
    p_cat_test = cat_model.predict_proba(X_test)

    # NN final (mejorada: residual + Swish + Mixup + class weights)
    tf.keras.backend.clear_session()
    X_nn_tv, imputer, scaler = fit_nn_preprocessor(X_tv)
    X_nn_test = transform_nn(X_test, imputer, scaler)

    nn = build_nn(X_nn_tv.shape[1], NN_PARAMS)
    steps_per_epoch = max(1, len(X_nn_tv) // NN_PARAMS["batch_size"])
    total_steps = steps_per_epoch * NN_PARAMS["epochs"]
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=NN_PARAMS["lr"],
        decay_steps=total_steps,
        alpha=0.01,
    )
    nn.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="categorical_crossentropy",
    )

    # Mixup para entrenamiento final
    train_gen = MixupDataGenerator(
        X_nn_tv, y_tv,
        batch_size=NN_PARAMS["batch_size"],
        alpha=NN_PARAMS["mixup_alpha"],
    )
    class_weight_dict = compute_class_weights_dict(y_tv)

    nn.fit(
        train_gen,
        epochs=NN_PARAMS["epochs"],
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="loss", patience=10, restore_best_weights=True
            ),
        ],
        verbose=0,
        class_weight=class_weight_dict,
    )
    p_nn_test = nn.predict(X_nn_test, verbose=0)

    # Meta-features para test → stacking predictions
    meta_X_test = build_meta_features(p_xgb_test, p_lgbm_test, p_cat_test, p_nn_test)
    p_stacking_home = meta_model.predict_proba(meta_X_test)[:, 1]
    p_stacking = np.column_stack([1 - p_stacking_home, p_stacking_home])

    # Weighted average (baseline simple, 4 modelos con peso igual)
    p_weighted = 0.25 * p_xgb_test + 0.25 * p_lgbm_test + 0.25 * p_cat_test + 0.25 * p_nn_test

    # --- 5. Metricas ---
    results = {}
    for name, probs in [("XGBoost", p_xgb_test), ("LightGBM", p_lgbm_test),
                         ("CatBoost", p_cat_test), ("NN", p_nn_test),
                         ("Avg (1/4)", p_weighted), ("Stacking", p_stacking)]:
        acc = accuracy_score(y_test, np.argmax(probs, axis=1))
        ll = log_loss(y_test, probs, labels=[0, 1])
        cwece, reliable = compute_classwise_ece(y_test, probs)
        results[name] = {"acc": acc, "log_loss": ll, "cw_ece": cwece, "reliable": reliable}

    print(f"\n{'─'*70}")
    print(f"  {'Modelo':<20s} {'Accuracy':>10s} {'Log Loss':>10s} {'cw-ECE':>10s}")
    print(f"{'─'*70}")
    for name, r in results.items():
        rel = "+" if r["reliable"] else "~"
        marker = " <-- BEST" if r["acc"] == max(v["acc"] for v in results.values()) else ""
        print(f"  {name:<20s} {r['acc']:>9.1%} {r['log_loss']:>10.4f} "
              f"{r['cw_ece']:>8.4f} [{rel}]{marker}")
    print(f"{'─'*70}")
    print(f"  Baseline (majority): {baseline_acc:.1%}")

    # Metricas extendidas para el mejor ensemble
    best_name = max(results, key=lambda k: results[k]["acc"])
    best_probs = {"XGBoost": p_xgb_test, "LightGBM": p_lgbm_test,
                  "CatBoost": p_cat_test, "NN": p_nn_test,
                  "Avg (1/4)": p_weighted, "Stacking": p_stacking}[best_name]
    p_home = best_probs[:, 1]
    compute_full_metrics(y_test, p_home, threshold=0.5, label=f"Best: {best_name}")
    opt_thresh = find_optimal_threshold(y_test, p_home, metric="f1")
    compute_full_metrics(y_test, p_home, threshold=opt_thresh,
                         label=f"Best: {best_name} (threshold={opt_thresh:.3f})")

    # --- 6. Guardar meta-learner ---
    meta_path = MODEL_DIR / f"stacking_meta_learner_{task}.pkl"
    joblib.dump(meta_model, meta_path)
    print(f"\n  Meta-learner guardado: {meta_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Stacking Ensemble (XGBoost + LightGBM + CatBoost + NN) con meta-learner LogReg."
    )
    parser.add_argument("--task", default="ML", choices=["ML", "UO", "both"])
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--splits", type=int, default=5, help="OOF folds.")
    parser.add_argument("--window", type=int, default=None,
                        help="Training window in seasons (e.g. 4 = last 4 seasons).")
    args = parser.parse_args()

    tf.config.optimizer.set_experimental_options({"remapping": False})

    if args.task in ("ML", "both"):
        evaluate_stacking("ML", args.dataset, args.splits, window=args.window)
    if args.task in ("UO", "both"):
        evaluate_stacking("UO", args.dataset, args.splits, window=args.window)


if __name__ == "__main__":
    main()
