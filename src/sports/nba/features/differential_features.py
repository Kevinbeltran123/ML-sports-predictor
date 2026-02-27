"""
Features diferenciales: home - away para stats base.
-----------------------------------------------------
Basado en Paper Ouyang et al. (2024, PLoS ONE):
  "Integration of ML XGBoost and SHAP for NBA game outcome prediction"

POR QUE: El paper demuestra que las features diferenciales (home - away)
capturan mejor la ventaja relativa entre equipos. FG%, DRB, TOV y AST
son top-4 SHAP features cuando se usan como diferenciales.

ANALOGIA NBA: Es como el PLUS_MINUS. No importan los 115 pts del local
y los 110 del visitante por separado — importa el +5.

VENTAJAS (3):
  1. Reduce dimensionalidad: de 2N columnas a N
  2. Fuerza al modelo a aprender la relacion correcta (la diferencia es lo que importa)
  3. Menos ruido: el modelo ve directamente "ventaja/desventaja" en vez de comparar dos numeros

NOTA IMPORTANTE: No eliminamos las columnas originales (FG_PCT, FG_PCT.1, etc.)
Las mantenemos para que el modelo pueda usar AMBAS representaciones.
Si los diffs absorben la informacion, SHAP mostrara que las originales pierden importancia,
y las podemos eliminar despues con evidencia.
"""
import pandas as pd
import numpy as np
from typing import List, Tuple


# Stats base que el paper identifica como mas relevantes via SHAP.
# Formato: (columna_home, columna_away, nombre_diferencial)
# Organizadas por importancia SHAP segun Table 10 del paper.
DIFFERENTIAL_PAIRS: List[Tuple[str, str, str]] = [
    # --- Top SHAP features (siempre en top 5 del paper) ---
    ("FG_PCT",  "FG_PCT.1",  "DIFF_FG_PCT"),     # Rank #1 en los 3 periodos
    ("DREB",    "DREB.1",    "DIFF_DREB"),        # Rank #2-4 consistentemente
    ("TOV",     "TOV.1",     "DIFF_TOV"),         # Rank #3-4 consistentemente
    ("AST",     "AST.1",     "DIFF_AST"),         # Clave en primera mitad

    # --- Features que ganan importancia en 2da mitad (Table 10) ---
    ("OREB",    "OREB.1",    "DIFF_OREB"),        # Sube a rank 5 en H3 y full game
    ("FG3_PCT", "FG3_PCT.1", "DIFF_FG3_PCT"),     # Sube a rank 2 en full game

    # --- Complementarias (significativas segun logistic reg, Tables 3-5) ---
    ("FT_PCT",  "FT_PCT.1",  "DIFF_FT_PCT"),     # p < 0.001 en todos los periodos
    ("STL",     "STL.1",     "DIFF_STL"),
    ("BLK",     "BLK.1",     "DIFF_BLK"),
    ("PF",      "PF.1",      "DIFF_PF"),          # p < 0.001, coef negativo
    ("PTS",     "PTS.1",     "DIFF_PTS"),         # Sanity check: correlacion directa con win
    ("REB",     "REB.1",     "DIFF_REB"),         # Total rebounds = OREB + DREB
]


def add_differential_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega features diferenciales (home - away) para stats base.

    Recibe el DataFrame despues de add_advanced_features() y agrega
    12 nuevas columnas DIFF_*.

    Nota sobre TOV y PF:
      - DIFF_TOV positivo = local comete MAS turnovers = MALO para local
      - DIFF_PF positivo = local comete MAS faltas = MALO para local
      El modelo XGBoost aprende esta relacion automaticamente (SHAP negativo),
      no necesitamos invertir el signo manualmente.

    Returns:
        DataFrame con las columnas diferenciales agregadas.
    """
    new_cols = {}
    added = []

    for col_home, col_away, nombre_diff in DIFFERENTIAL_PAIRS:
        if col_home in df.columns and col_away in df.columns:
            new_cols[nombre_diff] = (
                df[col_home].astype(float) - df[col_away].astype(float)
            )
            added.append(nombre_diff)

    if new_cols:
        new_df = pd.DataFrame(new_cols, index=df.index)
        df = pd.concat([df, new_df], axis=1)

    print(f"  [DIFF] Agregadas {len(added)} features diferenciales")
    return df


def get_differential_columns() -> List[str]:
    """Retorna la lista de nombres de columnas diferenciales."""
    return [nombre for _, _, nombre in DIFFERENTIAL_PAIRS]


# --- Sanity checks: correlaciones esperadas con Home-Team-Win ---
# Usadas por shap_analysis.py para verificar que el modelo esta aprendiendo bien
EXPECTED_POSITIVE_CORR = [
    "DIFF_FG_PCT",   # Mejor FG% → mas probable ganar
    "DIFF_DREB",     # Mas rebotes defensivos → mas posesiones
    "DIFF_AST",      # Mas asistencias → mejor juego en equipo
    "DIFF_OREB",     # Mas rebotes ofensivos → segundas oportunidades
    "DIFF_FG3_PCT",  # Mejor tiro de 3 → mas puntos por posesion
    "DIFF_STL",      # Mas robos → mas posesiones ganadas
    "DIFF_PTS",      # Mas puntos promedio → mas probable ganar
    "DIFF_REB",      # Mas rebotes totales
]

EXPECTED_NEGATIVE_CORR = [
    "DIFF_TOV",      # Mas turnovers → PEOR (pierde posesiones)
    "DIFF_PF",       # Mas faltas → PEOR (da tiros libres al rival)
]
