"""
Estadisticas avanzadas y features de matchup para el modelo NBA ML.

Calcula metricas de eficiencia (ORtg, DRtg, TS%, eFG%, etc.) y
features de interaccion entre equipos (mismatches, pace combinado, etc.)
a partir de las stats basicas de la NBA API.

Usado por Create_Games.py (entrenamiento) y main.py (prediccion).
"""

import numpy as np
import pandas as pd


def add_advanced_features(frame):
    """Agrega estadisticas avanzadas y features de matchup al DataFrame.

    Recibe un DataFrame donde las columnas del equipo local no tienen sufijo
    y las del visitante tienen sufijo '.1'. Agrega 29 nuevas columnas:
    - 8 stats avanzadas x 2 equipos = 16
    - 7 features de matchup originales (Diff_Net_Rtg, Diff_eFG, etc.)
    - 3 ORB% (home, away, diferencial) — completa Four Factors de Dean Oliver
    - 1 PACE_MISMATCH (incertidumbre por diferencia de ritmo)
    - 2 diferenciales adicionales (Diff_TOV_PCT, Diff_FT_Rate)
    """
    new_cols = {}

    for suffix in ["", ".1"]:
        fga = frame[f"FGA{suffix}"].astype(float)
        fta = frame[f"FTA{suffix}"].astype(float)
        fgm = frame[f"FGM{suffix}"].astype(float)
        fg3m = frame[f"FG3M{suffix}"].astype(float)
        oreb = frame[f"OREB{suffix}"].astype(float)
        tov = frame[f"TOV{suffix}"].astype(float)
        pts = frame[f"PTS{suffix}"].astype(float)
        ftm = frame[f"FTM{suffix}"].astype(float)
        plus_minus = frame[f"PLUS_MINUS{suffix}"].astype(float)

        # Posesiones estimadas por partido (formula de Dean Oliver)
        # POSS = FGA - OREB + TOV + 0.44 * FTA
        poss = fga - oreb + tov + 0.44 * fta
        poss = poss.replace(0, np.nan)

        # Eficiencia ofensiva: puntos por 100 posesiones
        new_cols[f"ORtg{suffix}"] = pts / poss * 100

        # Eficiencia defensiva: puntos permitidos por 100 posesiones
        # PTS - PLUS_MINUS = puntos del oponente (promedio por partido)
        new_cols[f"DRtg{suffix}"] = (pts - plus_minus) / poss * 100

        # Net Rating: diferencia entre ofensiva y defensiva
        new_cols[f"Net_Rtg{suffix}"] = new_cols[f"ORtg{suffix}"] - new_cols[f"DRtg{suffix}"]

        # Posesiones por partido (proxy de ritmo de juego)
        new_cols[f"POSS{suffix}"] = poss

        # True Shooting %: eficiencia de tiro real (integra 2pts, 3pts y TL)
        new_cols[f"TS_PCT{suffix}"] = pts / (2 * (fga + 0.44 * fta))

        # Effective FG%: ajusta FG% por el valor extra de los triples
        new_cols[f"eFG_PCT{suffix}"] = (fgm + 0.5 * fg3m) / fga

        # Turnover %: perdidas por posesion (Dean Oliver)
        new_cols[f"TOV_PCT{suffix}"] = tov / (fga + 0.44 * fta + tov)

        # Free Throw Rate: capacidad de llegar a la linea de TL
        new_cols[f"FT_Rate{suffix}"] = ftm / fga

    # --- Features de matchup (como interactuan los dos equipos) ---

    # Diferencia de calidad general
    new_cols["Diff_Net_Rtg"] = new_cols["Net_Rtg"] - new_cols["Net_Rtg.1"]

    # Diferencia de eficiencia de tiro
    new_cols["Diff_eFG"] = new_cols["eFG_PCT"] - new_cols["eFG_PCT.1"]

    # Mismatch ofensivo: que tan bien ataca home vs cuanto defiende away
    new_cols["Off_Mismatch"] = new_cols["ORtg"] - new_cols["DRtg.1"]

    # Mismatch defensivo: que tan bien ataca away vs cuanto defiende home
    new_cols["Def_Mismatch"] = new_cols["ORtg.1"] - new_cols["DRtg"]

    # Diferencia de ritmo: quien impone el pace
    new_cols["Diff_Pace"] = new_cols["POSS"] - new_cols["POSS.1"]

    # Pace promedio del matchup: clave para prediccion O/U
    new_cols["Avg_Pace"] = (new_cols["POSS"] + new_cols["POSS.1"]) / 2

    # Diferencia de dias de descanso
    new_cols["Diff_Rest"] = (
        frame["Days-Rest-Home"].astype(float) - frame["Days-Rest-Away"].astype(float)
    )

    # --- ORB% (Porcentaje de Rebote Ofensivo) ---
    # Completa los Four Factors de Dean Oliver (ya tenemos eFG%, TOV%, FT_Rate)
    # ORB% = OREB / (OREB + DREB_oponente)
    # Mide la EFICIENCIA de capturar rebotes ofensivos (tasa por oportunidad),
    # no el conteo bruto como OREB. Un equipo con pocos intentos pero alto ORB%
    # es mas peligroso en segundas oportunidades.

    # Para el equipo local:
    # DREB del oponente = REB_oponente - OREB_oponente
    oreb_home = frame["OREB"].astype(float)
    opp_dreb_for_home = frame["REB.1"].astype(float) - frame["OREB.1"].astype(float)
    new_cols["ORB_PCT"] = oreb_home / (oreb_home + opp_dreb_for_home).replace(0, np.nan)

    # Para el equipo visitante:
    oreb_away = frame["OREB.1"].astype(float)
    opp_dreb_for_away = frame["REB"].astype(float) - frame["OREB"].astype(float)
    new_cols["ORB_PCT.1"] = oreb_away / (oreb_away + opp_dreb_for_away).replace(0, np.nan)

    # Diferencial de ORB% (completa el set de diferenciales de Four Factors)
    new_cols["Diff_ORB_PCT"] = new_cols["ORB_PCT"] - new_cols["ORB_PCT.1"]

    # --- PACE_MISMATCH (Magnitud de diferencia de ritmo) ---
    # Valor absoluto de la diferencia de posesiones entre equipos.
    # Un mismatch alto = mas incertidumbre (un equipo se ve forzado
    # a jugar fuera de su ritmo natural).
    new_cols["PACE_MISMATCH"] = abs(new_cols["POSS"] - new_cols["POSS.1"])

    # --- Diferenciales adicionales de Four Factors ---

    # Diff_TOV_PCT: diferencial de perdidas de balon
    # Valor negativo = home pierde menos balones (bueno para home)
    new_cols["Diff_TOV_PCT"] = new_cols["TOV_PCT"] - new_cols["TOV_PCT.1"]

    # Diff_FT_Rate: diferencial de tasa de tiros libres
    new_cols["Diff_FT_Rate"] = new_cols["FT_Rate"] - new_cols["FT_Rate.1"]

    # Diff_TS_PCT: diferencial de True Shooting %
    # TS% integra 2pts, 3pts y TL en una sola metrica de eficiencia.
    # El diferencial captura ventaja/desventaja relativa de eficiencia de tiro.
    new_cols["Diff_TS_PCT"] = new_cols["TS_PCT"] - new_cols["TS_PCT.1"]

    # Agregar todas las columnas de golpe (evita fragmentacion del DataFrame)
    new_df = pd.DataFrame(new_cols, index=frame.index)
    frame = pd.concat([frame, new_df], axis=1)

    # Reemplazar NaN (de divisiones por 0) con 0
    frame = frame.fillna(0)

    return frame
