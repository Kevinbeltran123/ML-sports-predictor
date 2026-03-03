"""AH Meta-Learner: reemplaza blend lineal REG/CLF con LogisticRegression stacking.

En vez de p_cover = w*REG + (1-w)*CLF con w estático por bucket de spread,
el meta-learner toma ~12 señales (clf_p_cover, reg_p_cover, spread, sigma,
divergence, ATS rate, etc.) y produce P(cover) calibrado.

Entrenado con walk-forward CV en 17K juegos históricos.
Fallback al blend lineal si el meta-learner no está disponible.

Uso:
    from src.core.betting.ah_meta_learner import predict_p_cover, is_available
    if is_available():
        p = predict_p_cover(features_dict)
"""

import joblib
import numpy as np

from src.config import NBA_MARGIN_MODELS_DIR, get_logger

logger = get_logger(__name__)

_meta_model = None
_feature_names = None
_load_attempted = False

META_LEARNER_PATH = NBA_MARGIN_MODELS_DIR / "ah_meta_learner.pkl"

# Feature order must match training (scripts/train_ah_meta_learner.py)
EXPECTED_FEATURES = [
    "clf_p_cover",
    "reg_p_cover",
    "abs_spread",
    "ah_sigma",
    "divergence",
    "sigma_i",
    "reg_conf_margin",
    "clf_reg_gap",
    "prob_home",
    "ats_rate",
    "ats_streak",
    "reg_margin",
]


def _load():
    """Lazy load del meta-learner (una sola vez)."""
    global _meta_model, _feature_names, _load_attempted
    if _load_attempted:
        return _meta_model is not None
    _load_attempted = True

    if not META_LEARNER_PATH.exists():
        logger.debug("AH meta-learner not found: %s", META_LEARNER_PATH)
        return False

    try:
        artifact = joblib.load(META_LEARNER_PATH)
        if isinstance(artifact, dict):
            _meta_model = artifact["model"]
            _feature_names = artifact.get("feature_names", EXPECTED_FEATURES)
        else:
            _meta_model = artifact
            _feature_names = EXPECTED_FEATURES
        logger.info("AH meta-learner loaded: %s (%d features)",
                     type(_meta_model).__name__, len(_feature_names))
        return True
    except Exception as e:
        logger.warning("Error loading AH meta-learner: %s", e)
        return False


def is_available():
    """True si el meta-learner está cargado y listo."""
    _load()
    return _meta_model is not None


def predict_p_cover(features_dict):
    """Predice P(home cover) usando meta-learner.

    Args:
        features_dict: dict con las 12 señales. Keys deben matchear EXPECTED_FEATURES.
            Valores faltantes se llenan con defaults seguros.

    Returns:
        float: P(cover) calibrado [0.01, 0.99], o None si no hay modelo.
    """
    if not _load():
        return None

    # Construir array en el orden correcto
    x = np.zeros(len(_feature_names))
    defaults = {
        "clf_p_cover": 0.5,
        "reg_p_cover": 0.5,
        "abs_spread": 5.0,
        "ah_sigma": 14.0,
        "divergence": 0.0,
        "sigma_i": 0.05,
        "reg_conf_margin": 0.0,
        "clf_reg_gap": 0.0,
        "prob_home": 0.5,
        "ats_rate": 0.5,
        "ats_streak": 0.0,
        "reg_margin": 0.0,
    }
    for i, name in enumerate(_feature_names):
        val = features_dict.get(name)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            x[i] = defaults.get(name, 0.0)
        else:
            x[i] = float(val)

    try:
        proba = _meta_model.predict_proba(x.reshape(1, -1))[0, 1]
        return float(np.clip(proba, 0.01, 0.99))
    except Exception as e:
        logger.warning("Meta-learner prediction failed: %s", e)
        return None
