"""Style features — ratios de estilo de juego derivados del box score.

===============================================================
POR QUE "ESTILO"
===============================================================

Los datos de tracking real (velocidad, distancia recorrida, zonas de tiro)
requieren la API de Second Spectrum (de pago). Sin embargo, podemos derivar
proxies utiles del box score existente:

1. FG3A_RATE = FG3A / FGA
   - Proporcion de intentos de tiro que son triples
   - NO es lo mismo que FG3_PCT (precision) ni FG3A (conteo bruto)
   - Un equipo con FG3A_RATE=0.45 juega MUY distinto a uno con 0.25
   - Ejemplo: Houston Rockets (3-heavy) vs Milwaukee Bucks (paint-dominant)

2. AST_RATIO = AST / FGM
   - Fraccion de canastas que fueron asistidas
   - Mide "juego de sistema" vs "iso-ball"
   - Warriors (alto AST_RATIO = ball movement) vs Mavericks (bajo = Luka iso)

3. PACE_ADJ_DEF = (STL + BLK) / POSS * 100
   - Disrupciones defensivas por 100 posesiones
   - Ajustado por ritmo (raw STL/BLK penaliza equipos lentos)

4. STYLE_CLASH = |FG3A_RATE_home - FG3A_RATE_away|
   - Que tan DIFERENTE es el estilo de ambos equipos
   - Matchups de estilos opuestos generan mas varianza e incertidumbre

===============================================================
POR QUE AYUDAN AL MODELO
===============================================================

XGBoost PUEDE aprender FG3A/FGA internamente, pero:
- Necesita splits complejos (primero particionar por FGA, luego por FG3A)
- Las ratios explicitas reducen la profundidad necesaria del arbol
- Son mas robustas a distribution shift temporal (la NBA evoluciona:
  en 2012 FG3A_RATE promedio era ~0.22, en 2025 es ~0.38)
- El diferencial captura la ventaja relativa directamente

Referencia: Oliver, D. (2004). Basketball on Paper.
"""

import numpy as np
import pandas as pd


def add_style_features(frame):
    """Agrega features de estilo de juego al DataFrame del dataset.

    Recibe un DataFrame donde columnas del local no tienen sufijo
    y las del visitante tienen sufijo '.1'.

    Agrega 9 columnas:
    - FG3A_RATE, FG3A_RATE.1       (tasa de tiros de 3)
    - AST_RATIO, AST_RATIO.1       (tasa de asistencias)
    - PACE_ADJ_DEF, PACE_ADJ_DEF.1 (disrupciones defensivas/100 poss)
    - Diff_FG3A_RATE                (diferencial de estilo de tiro)
    - Diff_AST_RATIO                (diferencial de juego de equipo)
    - STYLE_CLASH                   (magnitud de diferencia de estilos)
    """
    new_cols = {}

    for suffix in ["", ".1"]:
        fga = frame[f"FGA{suffix}"].astype(float)
        fg3a = frame[f"FG3A{suffix}"].astype(float)
        fgm = frame[f"FGM{suffix}"].astype(float)
        ast = frame[f"AST{suffix}"].astype(float)
        stl = frame[f"STL{suffix}"].astype(float)
        blk = frame[f"BLK{suffix}"].astype(float)
        poss = frame[f"POSS{suffix}"].astype(float) if f"POSS{suffix}" in frame.columns else None

        # FG3A_RATE: proporcion de intentos de campo que son triples
        # Captura la filosofia ofensiva (perimetro vs interior)
        new_cols[f"FG3A_RATE{suffix}"] = fg3a / fga.replace(0, np.nan)

        # AST_RATIO: fraccion de canastas que fueron asistidas
        # Mide el nivel de "juego de equipo" (ball movement vs iso)
        new_cols[f"AST_RATIO{suffix}"] = ast / fgm.replace(0, np.nan)

        # PACE_ADJ_DEF: disrupciones defensivas por 100 posesiones
        # (STL + BLK) ajustado por ritmo — compara defensa independientemente del pace
        if poss is not None:
            new_cols[f"PACE_ADJ_DEF{suffix}"] = (stl + blk) / poss.replace(0, np.nan) * 100
        else:
            # Si POSS no existe aun, usar proxy basado en FGA
            proxy_poss = fga + 0.44 * frame[f"FTA{suffix}"].astype(float) + frame[f"TOV{suffix}"].astype(float) - frame[f"OREB{suffix}"].astype(float)
            new_cols[f"PACE_ADJ_DEF{suffix}"] = (stl + blk) / proxy_poss.replace(0, np.nan) * 100

    # --- Diferenciales y features de matchup ---

    # Diff_FG3A_RATE: ventaja en dependencia del triple
    # Positivo = home tira MAS triples (que es bueno o malo depende del matchup)
    new_cols["Diff_FG3A_RATE"] = new_cols["FG3A_RATE"] - new_cols["FG3A_RATE.1"]

    # Diff_AST_RATIO: ventaja en juego de equipo
    # Positivo = home juega mas en equipo
    new_cols["Diff_AST_RATIO"] = new_cols["AST_RATIO"] - new_cols["AST_RATIO.1"]

    # STYLE_CLASH: que tan diferentes son los estilos de ambos equipos
    # Matchups de estilos opuestos (3-point vs paint) generan mas varianza
    # Util para el modelo: puede aprender que "estilos opuestos = mas incertidumbre"
    new_cols["STYLE_CLASH"] = abs(new_cols["FG3A_RATE"] - new_cols["FG3A_RATE.1"])

    # Agregar todas las columnas de golpe
    new_df = pd.DataFrame(new_cols, index=frame.index)
    frame = pd.concat([frame, new_df], axis=1)

    # Reemplazar NaN (de divisiones por 0) con 0
    cols_added = list(new_cols.keys())
    frame[cols_added] = frame[cols_added].fillna(0)

    return frame
