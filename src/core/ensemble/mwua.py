"""MWUA Ensemble para Player Props (Multiplicative Weights Update Algorithm).

Basado en Chakrabarti — combina múltiples modelos con pesos adaptativos
que se actualizan automáticamente según el rendimiento reciente.

¿Cómo funciona?
  1. Tenemos N modelos para cada stat (PTS, REB, AST)
  2. Cada modelo tiene un peso w_i (inicialmente 1/N)
  3. La predicción final = promedio ponderado: P = Σ(w_i × P_i)
  4. Después de cada jornada, evaluamos cada modelo vs resultados reales
  5. Actualizamos pesos: w_i(t+1) = w_i(t) × (1 - η × loss_i(t))
  6. Normalizamos para que sumen 1

¿Por qué es útil?
  - Si un modelo funciona bien en temporada regular pero mal en playoffs,
    su peso baja automáticamente al entrar a playoffs
  - Si entrenamos un modelo nuevo (Sprint 4), empieza con peso bajo
    y sube si demuestra ser bueno
  - Protege contra modelos que se degradan con el tiempo (concept drift)

Flujo de uso:
  1. Guardar predicciones diarias:
     mwua = MWUAEnsemble()
     mwua.save_predictions(date, player, stat, model_id, p_over, line)

  2. Cuando llegan resultados (scripts/update_results.py):
     mwua.update_weights(date)

  3. Para predecir con ensemble:
     p_ensemble = mwua.weighted_predict(stat, predictions_by_model)

Base de datos: Data/MWUA.sqlite
  - model_registry: qué modelos hay y sus pesos actuales
  - daily_predictions: P(OVER) de cada modelo para cada jugador/stat/fecha
  - weight_history: historial de pesos (para análisis)
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import MWUA_DB, PLAYER_LOGS_DB as LOGS_DB, NBA_PROPS_MODELS_DIR

# Tasa de aprendizaje para actualización de pesos
# η bajo (0.05) = conservador, cambios lentos
# η alto (0.3) = agresivo, reacciona rápido
# 0.1 es un buen punto medio
DEFAULT_ETA = 0.1

# Peso mínimo para evitar que un modelo llegue a 0 (siempre tiene "voz")
MIN_WEIGHT = 0.05


class MWUAEnsemble:
    """Ensemble de modelos con pesos adaptativos via MWUA."""

    def __init__(self, db_path: Path = MWUA_DB, eta: float = DEFAULT_ETA):
        self.db_path = db_path
        self.eta = eta
        self._init_db()

    def _init_db(self):
        """Crea las tablas si no existen."""
        with sqlite3.connect(self.db_path) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS model_registry (
                    model_id TEXT NOT NULL,
                    stat TEXT NOT NULL,
                    weight REAL NOT NULL DEFAULT 1.0,
                    registered_date TEXT NOT NULL,
                    last_updated TEXT,
                    total_predictions INTEGER DEFAULT 0,
                    PRIMARY KEY (model_id, stat)
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS daily_predictions (
                    prediction_date TEXT NOT NULL,
                    player_name TEXT NOT NULL,
                    stat TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    p_over REAL NOT NULL,
                    line REAL,
                    actual REAL,
                    correct INTEGER,
                    market_prob REAL,
                    PRIMARY KEY (prediction_date, player_name, stat, model_id)
                )
            """)
            # Migraciones: agregar columnas nuevas si no existen
            # (mismo patrón try/except para cada columna)
            for col_def in [
                "market_prob REAL",
                "conformal_set_size INTEGER",
                "conformal_margin REAL",
                "kelly_pct REAL",
                "over_odds INTEGER",
                "under_odds INTEGER",
            ]:
                try:
                    con.execute(f"ALTER TABLE daily_predictions ADD COLUMN {col_def}")
                except sqlite3.OperationalError:
                    pass  # columna ya existe
            con.execute("""
                CREATE TABLE IF NOT EXISTS weight_history (
                    update_date TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    stat TEXT NOT NULL,
                    weight_before REAL,
                    weight_after REAL,
                    day_log_loss REAL,
                    day_accuracy REAL,
                    n_predictions INTEGER,
                    PRIMARY KEY (update_date, model_id, stat)
                )
            """)
            con.commit()

    # ── Registro de modelos ──────────────────────────────────────────────────

    def register_model(self, model_id: str, stat: str, weight: float = 1.0):
        """Registra un modelo nuevo para un stat.

        model_id: nombre del archivo del modelo (sin extensión)
        stat: PTS, REB, o AST
        weight: peso inicial (default 1.0, se normaliza al predecir)
        """
        today = datetime.now().strftime("%Y-%m-%d")
        with sqlite3.connect(self.db_path) as con:
            con.execute("""
                INSERT OR REPLACE INTO model_registry
                (model_id, stat, weight, registered_date, last_updated)
                VALUES (?, ?, ?, ?, ?)
            """, (model_id, stat, weight, today, today))
            con.commit()
        print(f"  [MWUA] Modelo registrado: {model_id} ({stat}) peso={weight:.3f}")

    def get_weights(self, stat: str) -> Dict[str, float]:
        """Retorna pesos normalizados para todos los modelos de un stat.

        Si solo hay 1 modelo, retorna {model_id: 1.0}.
        Si hay N modelos, normaliza para que sumen 1.
        """
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(
                "SELECT model_id, weight FROM model_registry WHERE stat = ?",
                (stat,)
            ).fetchall()

        if not rows:
            return {}

        total = sum(w for _, w in rows)
        if total <= 0:
            total = len(rows)  # fallback: pesos iguales

        return {mid: w / total for mid, w in rows}

    # ── Predicciones ─────────────────────────────────────────────────────────

    def weighted_predict(self, stat: str,
                         predictions: Dict[str, float]) -> float:
        """Calcula P(OVER) como promedio ponderado de múltiples modelos.

        Args:
            stat: PTS, REB, o AST
            predictions: {model_id: p_over} para cada modelo

        Returns:
            float: P(OVER) ponderado
        """
        weights = self.get_weights(stat)

        if not weights or not predictions:
            # Si no hay MWUA setup, retornar promedio simple
            if predictions:
                return sum(predictions.values()) / len(predictions)
            return 0.5

        # Promedio ponderado solo de modelos que tienen predicción
        total_w = 0.0
        weighted_sum = 0.0
        for model_id, p_over in predictions.items():
            w = weights.get(model_id, 0.0)
            if w > 0:
                weighted_sum += w * p_over
                total_w += w

        if total_w > 0:
            return weighted_sum / total_w
        return sum(predictions.values()) / len(predictions)

    def save_predictions(self, prediction_date: str, player_name: str,
                         stat: str, model_id: str, p_over: float,
                         line: Optional[float] = None,
                         market_prob: Optional[float] = None,
                         conformal_set_size: Optional[int] = None,
                         conformal_margin: Optional[float] = None,
                         kelly_pct: Optional[float] = None,
                         over_odds: Optional[int] = None,
                         under_odds: Optional[int] = None):
        """Guarda la prediccion de un modelo para evaluacion posterior.

        Se llama durante Props_Runner para cada jugador/stat/modelo.
        El campo 'actual' se llena despues cuando llegan resultados.

        Args:
            market_prob: P(OVER) implicita del mercado (de-vigged), para DR Shrinkage.
            conformal_set_size: 1=confiado, 2=incierto (conformal prediction).
            conformal_margin: distancia de la probabilidad al threshold conformal.
            kelly_pct: fraccion Kelly en porcentaje (ej: 2.5 = apostar 2.5% del bankroll).
            over_odds: odds americanos para OVER (ej: -130).
            under_odds: odds americanos para UNDER (ej: +110).
        """
        with sqlite3.connect(self.db_path) as con:
            con.execute("""
                INSERT OR REPLACE INTO daily_predictions
                (prediction_date, player_name, stat, model_id, p_over, line,
                 market_prob, conformal_set_size, conformal_margin, kelly_pct,
                 over_odds, under_odds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (prediction_date, player_name, stat, model_id, p_over, line,
                  market_prob, conformal_set_size, conformal_margin, kelly_pct,
                  over_odds, under_odds))
            # Incrementar contador de predicciones del modelo
            con.execute("""
                UPDATE model_registry
                SET total_predictions = total_predictions + 1
                WHERE model_id = ? AND stat = ?
            """, (model_id, stat))
            con.commit()

    # ── Actualización de pesos ───────────────────────────────────────────────

    def fill_actuals(self, prediction_date: str):
        """Rellena los valores reales (actual) desde PlayerGameLogs.

        Se ejecuta después de que los resultados del día están disponibles.
        Compara P(OVER) guardado con el resultado real para cada jugador.
        """
        with sqlite3.connect(self.db_path) as con:
            # Obtener predicciones pendientes de ese día
            preds = con.execute("""
                SELECT player_name, stat, line
                FROM daily_predictions
                WHERE prediction_date = ? AND actual IS NULL
                GROUP BY player_name, stat
            """, (prediction_date,)).fetchall()

        if not preds:
            print(f"  [MWUA] No hay predicciones pendientes para {prediction_date}")
            return 0

        # Determinar temporada desde la fecha
        date_obj = datetime.strptime(prediction_date, "%Y-%m-%d")
        year = date_obj.year
        month = date_obj.month
        if month >= 10:
            season = f"{year}-{str(year + 1)[2:]}"
        else:
            season = f"{year - 1}-{str(year)[2:]}"

        table = f"player_logs_{season}"
        filled = 0

        with sqlite3.connect(LOGS_DB) as logs_con:
            tables = [r[0] for r in logs_con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            if table not in tables:
                print(f"  [MWUA] Tabla {table} no existe en PlayerGameLogs")
                return 0

            # Expresiones SQL para combos (PTS+REB+AST, etc.)
            COMBO_SQL = {
                "PRA": "PTS+REB+AST", "PR": "PTS+REB",
                "PA": "PTS+AST", "RA": "REB+AST", "BS": "BLK+STL",
            }

            for player_name, stat, line in preds:
                # Buscar el resultado real del jugador ese día
                # Para combos usamos expresión SQL (PTS+REB+AST), para individuales la columna directa
                stat_expr = COMBO_SQL.get(stat, stat)
                row = logs_con.execute(f"""
                    SELECT {stat_expr} FROM "{table}"
                    WHERE PLAYER_NAME = ? AND GAME_DATE = ?
                """, (player_name, prediction_date)).fetchone()

                if row is not None and line is not None:
                    actual_val = float(row[0])
                    correct = 1 if actual_val > line else 0

                    with sqlite3.connect(self.db_path) as con:
                        con.execute("""
                            UPDATE daily_predictions
                            SET actual = ?, correct = ?
                            WHERE prediction_date = ?
                              AND player_name = ?
                              AND stat = ?
                        """, (actual_val, correct, prediction_date,
                              player_name, stat))
                        con.commit()
                    filled += 1

        print(f"  [MWUA] Resultados rellenados: {filled}/{len(preds)} para {prediction_date}")
        return filled

    def update_weights(self, prediction_date: str):
        """Actualiza pesos MWUA basado en resultados de un día.

        Fórmula: w_i(t+1) = w_i(t) × (1 - η × loss_i(t))
        Donde loss_i = log_loss del modelo i en ese día.

        Antes de llamar esto, asegúrate de llamar fill_actuals().
        """
        # Primero rellenar resultados si no están
        self.fill_actuals(prediction_date)

        with sqlite3.connect(self.db_path) as con:
            # Obtener todos los modelos con predicciones evaluadas ese día
            stats_models = con.execute("""
                SELECT DISTINCT stat, model_id
                FROM daily_predictions
                WHERE prediction_date = ? AND actual IS NOT NULL
            """, (prediction_date,)).fetchall()

        if not stats_models:
            print(f"  [MWUA] Sin resultados evaluados para {prediction_date}")
            return

        # Agrupar por stat
        stats = set(s for s, _ in stats_models)

        for stat in stats:
            self._update_stat_weights(prediction_date, stat)

    def _update_stat_weights(self, prediction_date: str, stat: str):
        """Actualiza pesos de todos los modelos para un stat específico."""
        with sqlite3.connect(self.db_path) as con:
            models = con.execute("""
                SELECT DISTINCT model_id
                FROM daily_predictions
                WHERE prediction_date = ? AND stat = ? AND actual IS NOT NULL
            """, (prediction_date, stat)).fetchall()

        if not models:
            return

        current_weights = self.get_weights(stat)

        for (model_id,) in models:
            with sqlite3.connect(self.db_path) as con:
                rows = con.execute("""
                    SELECT p_over, correct
                    FROM daily_predictions
                    WHERE prediction_date = ? AND stat = ? AND model_id = ?
                      AND actual IS NOT NULL AND correct IS NOT NULL
                """, (prediction_date, stat, model_id)).fetchall()

            if not rows:
                continue

            # Calcular log loss del día para este modelo
            p_overs = np.array([r[0] for r in rows])
            corrects = np.array([r[1] for r in rows])

            # Clamp para evitar log(0)
            p_overs = np.clip(p_overs, 1e-7, 1 - 1e-7)

            # Log loss: -mean(y*log(p) + (1-y)*log(1-p))
            day_log_loss = -np.mean(
                corrects * np.log(p_overs) +
                (1 - corrects) * np.log(1 - p_overs)
            )
            day_accuracy = np.mean(
                (p_overs >= 0.5).astype(int) == corrects
            )

            # MWUA update: w_new = w_old × (1 - η × loss)
            # loss normalizado: 0 = perfecto, 1 = terrible
            # log_loss de random = ln(2) ≈ 0.693
            # Normalizamos: loss_norm = (log_loss - 0) / ln(2)
            loss_norm = min(day_log_loss / np.log(2), 2.0)  # cap at 2x random
            w_old = current_weights.get(model_id, 1.0 / max(len(models), 1))
            w_new = w_old * (1.0 - self.eta * (loss_norm - 1.0))
            # loss_norm < 1 → modelo mejor que random → peso sube
            # loss_norm > 1 → modelo peor que random → peso baja

            # Aplicar peso mínimo
            w_new = max(w_new, MIN_WEIGHT)

            # Guardar peso actualizado
            with sqlite3.connect(self.db_path) as con:
                con.execute("""
                    UPDATE model_registry
                    SET weight = ?, last_updated = ?
                    WHERE model_id = ? AND stat = ?
                """, (w_new, prediction_date, model_id, stat))

                # Guardar historial
                con.execute("""
                    INSERT OR REPLACE INTO weight_history
                    (update_date, model_id, stat, weight_before, weight_after,
                     day_log_loss, day_accuracy, n_predictions)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (prediction_date, model_id, stat,
                      w_old, w_new, float(day_log_loss),
                      float(day_accuracy), len(rows)))
                con.commit()

            print(f"  [MWUA] {stat} {model_id[:30]}...: "
                  f"loss={day_log_loss:.4f} acc={day_accuracy:.1%} "
                  f"w: {w_old:.3f} → {w_new:.3f}")

    # ── Epsilon adaptativo (DRO / Wasserstein) ────────────────────────────

    def get_model_epsilon(self, model_id: str, stat: str,
                          lookback_days: int = 30) -> float:
        """Calcula el radio de incertidumbre ε para un modelo/stat.

        Basado en: Sun & Zou (2024) — CVaR-Wasserstein Kelly.

        Usa el historial de predicciones evaluadas en MWUA.sqlite para
        medir cuanto se desvia P(OVER) del resultado real. Esto nos da
        una medida DATA-DRIVEN de la incertidumbre del modelo.

        ¿Que mide ε?
          - ε chico (0.02-0.05): el modelo esta bien calibrado, sus P(OVER)
            son cercanas a la realidad → Kelly puede ser mas agresivo
          - ε grande (0.10-0.20): el modelo tiene errores sistematicos,
            sus P(OVER) se desvian mucho → Kelly debe ser conservador

        ¿Por que no un ε fijo?
          - PTS puede estar mejor calibrado que AST
          - Un modelo nuevo tiene mas incertidumbre que uno con 500+ preds
          - La calibracion puede degradarse (concept drift de mitad temporada)

        Calculo:
          1. Toma las ultimas N predicciones evaluadas del modelo
          2. Calcula ECE (Expected Calibration Error) por bins
          3. Agrega margen por peor bin + correccion por tamaño de muestra
          4. Clamp a [0.02, 0.20]

        Args:
            model_id: identificador del modelo
            stat: PTS, REB, AST, etc.
            lookback_days: dias de historial a considerar

        Returns:
            float: ε ∈ [0.02, 0.20]
        """
        from datetime import timedelta

        cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        with sqlite3.connect(self.db_path) as con:
            rows = con.execute("""
                SELECT p_over, correct
                FROM daily_predictions
                WHERE model_id = ? AND stat = ?
                  AND prediction_date >= ?
                  AND actual IS NOT NULL AND correct IS NOT NULL
            """, (model_id, stat, cutoff)).fetchall()

        if len(rows) < 20:
            # Muy pocas predicciones evaluadas → alta incertidumbre
            return 0.15

        p_overs = np.array([r[0] for r in rows])
        corrects = np.array([r[1] for r in rows])

        # Calcular ECE empirico
        n_bins = min(10, len(rows) // 5)
        if n_bins < 3:
            return 0.12

        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        max_error = 0.0

        for i in range(n_bins):
            if i == n_bins - 1:
                mask = (p_overs >= bin_edges[i]) & (p_overs <= bin_edges[i + 1])
            else:
                mask = (p_overs >= bin_edges[i]) & (p_overs < bin_edges[i + 1])

            if mask.sum() < 3:
                continue

            error = abs(p_overs[mask].mean() - corrects[mask].mean())
            ece += error * (mask.sum() / len(p_overs))
            max_error = max(max_error, error)

        # ε = ECE + margen basado en peor bin + correccion por muestra
        # Menos datos → mas incertidumbre (binomial standard error ~1.96/√n)
        sample_correction = 1.96 / np.sqrt(len(rows))
        epsilon = ece + 0.3 * max_error + sample_correction

        return float(np.clip(epsilon, 0.02, 0.20))

    def get_stat_epsilon(self, stat: str, lookback_days: int = 30) -> float:
        """ε promedio ponderado de todos los modelos de un stat.

        Si hay multiples modelos para PTS, calcula ε de cada uno
        y los promedia ponderados por peso MWUA (modelos con mejor
        desempeño tienen mas influencia en el ε conjunto).

        Args:
            stat: PTS, REB, AST, etc.
            lookback_days: dias de historial

        Returns:
            float: ε ponderado ∈ [0.02, 0.20]
        """
        weights = self.get_weights(stat)
        if not weights:
            return 0.10  # default sin modelos

        total_w = 0.0
        weighted_eps = 0.0

        for model_id, w in weights.items():
            eps = self.get_model_epsilon(model_id, stat, lookback_days)
            weighted_eps += w * eps
            total_w += w

        if total_w > 0:
            return float(np.clip(weighted_eps / total_w, 0.02, 0.20))
        return 0.10

    # ── Utilidades ───────────────────────────────────────────────────────────

    def get_summary(self) -> str:
        """Resumen del estado actual del ensemble."""
        lines = ["MWUA Ensemble Status:", "=" * 50]

        for stat in ["PTS", "REB", "AST", "FG3M", "BLK", "STL",
                     "PRA", "PR", "PA", "RA", "BS"]:
            weights = self.get_weights(stat)
            if not weights:
                lines.append(f"  {stat}: Sin modelos registrados")
                continue

            lines.append(f"\n  {stat} ({len(weights)} modelos):")
            for model_id, w in sorted(weights.items(),
                                       key=lambda x: -x[1]):
                short_id = model_id[:40] + "..." if len(model_id) > 40 else model_id
                lines.append(f"    {short_id}: w={w:.3f}")

        with sqlite3.connect(self.db_path) as con:
            total_preds = con.execute(
                "SELECT COUNT(*) FROM daily_predictions"
            ).fetchone()[0]
            evaluated = con.execute(
                "SELECT COUNT(*) FROM daily_predictions WHERE actual IS NOT NULL"
            ).fetchone()[0]

        lines.append(f"\n  Predicciones totales: {total_preds}")
        lines.append(f"  Evaluadas: {evaluated}")

        return "\n".join(lines)

    def auto_register_models(self, model_dir: Path = None):
        """Registra automáticamente todos los modelos .json del directorio.

        Busca archivos XGBoost_Props_Cls_{STAT}_*.json y los registra
        con peso inicial 1.0 si no están ya registrados.
        """
        if model_dir is None:
            model_dir = NBA_PROPS_MODELS_DIR

        import glob
        registered = 0
        for stat in ["PTS", "REB", "AST", "FG3M", "BLK", "STL",
                     "PRA", "PR", "PA", "RA", "BS"]:
            pattern = str(model_dir / f"XGBoost_Props_Cls_{stat}_*.json")
            files = glob.glob(pattern)
            for f in files:
                model_id = Path(f).stem  # nombre sin extensión
                # Verificar si ya está registrado
                with sqlite3.connect(self.db_path) as con:
                    existing = con.execute(
                        "SELECT 1 FROM model_registry WHERE model_id = ? AND stat = ?",
                        (model_id, stat)
                    ).fetchone()
                if not existing:
                    self.register_model(model_id, stat)
                    registered += 1

        if registered > 0:
            print(f"  [MWUA] {registered} modelos nuevos registrados")
        return registered
