"""Features diferenciales para prediccion in-game (post-Q1, Q2, Q3).

SINGLE SOURCE OF TRUTH:
========================
Este modulo computa las MISMAS features tanto en:
  - Training (desde QuarterData.sqlite) via compute_ingame_differentials()
  - Live inference (desde get_live_box_score()) via compute_ingame_differentials()

Esto PREVIENE el bug de feature mismatch que afecto a los props models
(SLIM_FEATURES order != FEATURE_COLS order, Feb 2026).

TRES FAMILIAS DE MODELOS:
=========================
1. Moneyline (ML): ¿quien gana? → target HOME_WIN
2. Spread (ATS): ¿cubre el handicap? → target SPREAD_COVER
3. Total (O/U): ¿se pasa la linea de puntos? → target TOTAL_OVER

Cada familia tiene features especificas:
  - ML: features diferenciales (quien juega mejor?)
  - Spread: features de ML + contexto de linea (SPREAD_PACE, margen vs expectativa)
  - Total: features de ritmo + scoring pace + regresion a la media

POR QUE ESTAS FEATURES:
========================
Ouyang 2024 demostro que los features diferenciales (home - away) son los mas
predictivos via SHAP analysis:
  - FG% siempre es #1 en importancia (eficiencia de tiro)
  - TOV y DRB son top-4 en todos los periodos
  - AST es critico en la primera mitad, pierde importancia despues
  - 3P% y OREB ganan importancia en la segunda mitad

Para SPREAD: la desviacion del score respecto a la expectativa del mercado es
mas informativa que el score raw (Implied Probability Shift).

Para TOTAL: el ritmo de anotacion (scoring pace) proyectado al juego completo
predice el total final (Run Rate, adaptado de cricket/baseball).

Normalizamos por posesiones porque:
  +5 rebotes en 25 posesiones (Q1) es MAS significativo que
  +5 rebotes en 100 posesiones (juego completo).
  La normalizacion hace las features comparables entre periodos.

EJEMPLO DE USO:
===============
    # Training (desde QuarterData DataFrame row)
    home_stats = {"FGM": 10, "FGA": 22, "FG_PCT": 0.455, ...}
    away_stats = {"FGM": 8, "FGA": 20, "FG_PCT": 0.400, ...}
    features = compute_ingame_differentials(home_stats, away_stats, p_pregame=0.68)

    # Spread features (necesitan la linea pregame)
    spread_feats = compute_spread_features(home_stats, away_stats, p_pregame=0.68,
                                           pregame_spread=-5.0, period=1)

    # Total features (necesitan la linea de total)
    total_feats = compute_total_features(home_stats, away_stats,
                                         pregame_total=215.5, period=1)

    # Live inference (desde get_live_box_score())
    box = get_live_box_score(game_id)
    home_stats = box_score_to_stats_dict(box["home"])
    away_stats = box_score_to_stats_dict(box["away"])
    features = compute_ingame_differentials(home_stats, away_stats, p_pregame=0.68)
"""

import math
from collections import OrderedDict

# --- Feature names (ORDEN IMPORTA — debe coincidir entre training e inference) ---

# Features basicas (8) — usadas por el modelo logistico (Nivel 1)
INGAME_FEATURE_NAMES = [
    "LOGIT_PREGAME",    # logit(P_pregame) — prior del modelo en espacio lineal
    "SCORE_DIFF_NORM",  # (home_pts - away_pts) / sqrt(avg_poss) — señal de score normalizada
    "DIFF_FG_PCT",      # FG_PCT_home - FG_PCT_away — eficiencia de tiro
    "DIFF_TOV_RATE",    # (TOV_away - TOV_home) / avg_poss — control de balon (flipped: + = ventaja home)
    "DIFF_REB_RATE",    # (REB_home - REB_away) / avg_poss — dominio del tablero
    "DIFF_AST_RATE",    # (AST_home - AST_away) / avg_poss — calidad de ataque
    "DIFF_FG3_PCT",     # FG3_PCT_home - FG3_PCT_away — tiro de 3 puntos
    "DIFF_FT_PCT",      # FT_PCT_home - FT_PCT_away — tiros libres
]

# Features extendidas (~17) — usadas por el modelo XGBoost (Nivel 2)
INGAME_EXTENDED_FEATURES = INGAME_FEATURE_NAMES + [
    "P_PREGAME",        # probabilidad pre-partido (0-1, sin transformar)
    "MARKET_ML_PROB",   # probabilidad implicita del mercado
    "ELO_DIFF",         # diferencial Elo pre-partido
    "HOME_POSS",        # posesiones totales del local (proxy de ritmo de juego)
    "DIFF_OREB_RATE",   # (OREB_home - OREB_away) / avg_poss — rebotes ofensivos
    "DIFF_STL_RATE",    # (STL_home - STL_away) / avg_poss — robos
    "DIFF_BLK_RATE",    # (BLK_home - BLK_away) / avg_poss — bloqueos
    "HOME_FG3A_RATE",   # FG3A_home / FGA_home — tasa de intentos de 3
    "AWAY_FG3A_RATE",   # FG3A_away / FGA_away
]

# --- Features para SPREAD (handicap) ---
# Incluyen las features de ML extendidas + contexto de la linea pregame.
# Concepto: la desviacion del score respecto a la EXPECTATIVA del mercado
# es mas informativa que el score raw (Implied Probability Shift).
INGAME_SPREAD_FEATURES = INGAME_EXTENDED_FEATURES + [
    "PREGAME_SPREAD",       # linea pregame (neg=home favorito). Ej: -5.0
    "SPREAD_PACE",          # score_diff - expected_diff_at_this_point
    "SPREAD_MARGIN_NORM",   # (score_diff - spread) / sqrt(remaining_poss)
    "PERIOD_FRACTION",      # fraccion del juego transcurrida (0.25, 0.50, 0.75)
    "PTS_NEEDED_TO_COVER",  # puntos que home necesita anotar para cubrir
]

# --- Features para TOTAL (over/under) ---
# Completamente diferentes: se enfocan en ritmo, scoring pace y regresion a la media.
# No usan features diferenciales (no importa QUIEN anota, solo CUANTO).
INGAME_TOTAL_FEATURES = [
    "PREGAME_TOTAL",        # linea O/U pregame. Ej: 215.5
    "COMBINED_PTS",         # home_pts + away_pts hasta ahora
    "SCORING_PACE",         # puntos combinados / posesiones jugadas
    "PROJECTED_TOTAL",      # scoring_pace * posesiones_esperadas_totales (200)
    "PACE_VS_LINE",         # projected_total - pregame_total (+ = on pace for OVER)
    "PERIOD_FRACTION",      # fraccion del juego transcurrida
    "COMBINED_POSS",        # home_poss + away_poss (ritmo real del juego)
    "COMBINED_FG_PCT",      # FG% combinado (eficiencia general de tiro)
    "COMBINED_FG3_RATE",    # tasa de triples combinada (mas 3s = mas puntos por tiro)
    "COMBINED_FT_RATE",     # FTA / FGA combinado (frecuencia de tiros libres)
    "COMBINED_TOV_RATE",    # turnovers / posesiones (pierdas = menos anotaciones)
    "FG_PCT_ZSCORE",        # z-score de FG% vs historico (~45%) → regresion a la media
    "P_PREGAME",            # proxy del ritmo esperado (equipos buenos anotan mas)
    "MARKET_ML_PROB",       # probabilidad implicita del mercado
    "DIFF_EP_RATE",         # expected points por posesion home - away
]

# Posesiones esperadas totales por juego en NBA (~200 por equipo, 100 cada uno)
# Fuente: promedio historico NBA 2020-2025, varia por temporada
EXPECTED_TOTAL_POSS = 200.0  # posesiones combinadas (home + away)
# FG% promedio historico NBA (~45.5%) para calcular z-score
HISTORICAL_FG_PCT = 0.455
HISTORICAL_FG_PCT_STD = 0.045  # desviacion estandar tipica por juego


def compute_ingame_differentials(
    home_stats: dict,
    away_stats: dict,
    p_pregame: float = 0.5,
) -> OrderedDict:
    """Calcula las 8 features diferenciales basicas para prediccion in-game.

    Todas las features son diferenciales (home - away) normalizadas, excepto
    LOGIT_PREGAME que es el prior transformado a espacio log-odds.

    Args:
        home_stats: dict con keys FGM, FGA, FG_PCT, FG3_PCT, FT_PCT,
                    OREB, DREB, REB, AST, STL, BLK, TOV, PTS, POSS
        away_stats: dict con mismas keys
        p_pregame:  probabilidad pre-partido del local (0.0 a 1.0)

    Returns:
        OrderedDict con las 8 features en EXACTAMENTE el orden de INGAME_FEATURE_NAMES.
        Los valores son float.

    Ejemplo:
        >>> compute_ingame_differentials(
        ...     {"PTS": 28, "FGA": 22, "FGM": 10, "FG_PCT": 0.455,
        ...      "FG3_PCT": 0.400, "FT_PCT": 0.800, "REB": 12,
        ...      "AST": 6, "TOV": 3, "POSS": 24.5, ...},
        ...     {"PTS": 22, "FGA": 20, "FGM": 8, "FG_PCT": 0.400,
        ...      "FG3_PCT": 0.333, "FT_PCT": 0.750, "REB": 10,
        ...      "AST": 4, "TOV": 5, "POSS": 23.0, ...},
        ...     p_pregame=0.68
        ... )
        OrderedDict([('LOGIT_PREGAME', 0.753), ('SCORE_DIFF_NORM', 1.23), ...])
    """
    features = OrderedDict()

    # Clamp para evitar log(0) o division por cero
    p = max(0.001, min(0.999, p_pregame))

    # Posesiones promedio entre ambos equipos (evitar div/0)
    home_poss = max(1.0, float(home_stats.get("POSS", 1.0)))
    away_poss = max(1.0, float(away_stats.get("POSS", 1.0)))
    avg_poss = (home_poss + away_poss) / 2.0

    # --- Feature 1: LOGIT_PREGAME ---
    # El prior del modelo pre-partido transformado a espacio log-odds.
    # En este espacio, podemos "sumar" evidencia linealmente (misma idea
    # que la regresion logistica que ya usamos para el modelo de equipo).
    features["LOGIT_PREGAME"] = math.log(p / (1.0 - p))

    # --- Feature 2: SCORE_DIFF_NORM ---
    # Diferencia de score normalizada por sqrt(posesiones).
    # Intuicion: +8 con 25 posesiones es mas significativo que +8 con 100 posesiones.
    # sqrt() viene del modelo de Brownian Motion de Stern 1994.
    home_pts = float(home_stats.get("PTS", 0))
    away_pts = float(away_stats.get("PTS", 0))
    features["SCORE_DIFF_NORM"] = (home_pts - away_pts) / math.sqrt(avg_poss)

    # --- Feature 3: DIFF_FG_PCT ---
    # SHAP rank #1 en Ouyang 2024 en TODOS los periodos.
    # El equipo que tira mejor tiende a ganar.
    features["DIFF_FG_PCT"] = (
        float(home_stats.get("FG_PCT", 0.0)) - float(away_stats.get("FG_PCT", 0.0))
    )

    # --- Feature 4: DIFF_TOV_RATE ---
    # Turnovers normalizados (FLIPPED: positivo = ventaja para home).
    # Mas turnovers = peor. Entonces restamos home_tov (malo) y sumamos away_tov (bueno para home).
    home_tov = float(home_stats.get("TOV", 0))
    away_tov = float(away_stats.get("TOV", 0))
    features["DIFF_TOV_RATE"] = (away_tov - home_tov) / avg_poss

    # --- Feature 5: DIFF_REB_RATE ---
    # Rebotes normalizados. Mas rebotes = mas posesiones = ventaja.
    home_reb = float(home_stats.get("REB", 0))
    away_reb = float(away_stats.get("REB", 0))
    features["DIFF_REB_RATE"] = (home_reb - away_reb) / avg_poss

    # --- Feature 6: DIFF_AST_RATE ---
    # Asistencias normalizadas. Ouyang: critico en Q1-Q2, pierde importancia despues.
    home_ast = float(home_stats.get("AST", 0))
    away_ast = float(away_stats.get("AST", 0))
    features["DIFF_AST_RATE"] = (home_ast - away_ast) / avg_poss

    # --- Feature 7: DIFF_FG3_PCT ---
    # Tiro de 3. Ouyang: gana importancia en la segunda mitad del partido.
    features["DIFF_FG3_PCT"] = (
        float(home_stats.get("FG3_PCT", 0.0)) - float(away_stats.get("FG3_PCT", 0.0))
    )

    # --- Feature 8: DIFF_FT_PCT ---
    # Tiros libres. Indicador de ejecucion en momentos de presion.
    features["DIFF_FT_PCT"] = (
        float(home_stats.get("FT_PCT", 0.0)) - float(away_stats.get("FT_PCT", 0.0))
    )

    return features


def compute_extended_features(
    home_stats: dict,
    away_stats: dict,
    p_pregame: float = 0.5,
    market_ml_prob: float = 0.5,
    elo_diff: float = 0.0,
) -> OrderedDict:
    """Calcula las 17 features extendidas para el modelo XGBoost in-game.

    Incluye las 8 features basicas + 9 adicionales que capturan contexto
    extra (Elo, mercado, ritmo de juego, OREB, steals, blocks, shot selection).

    Args:
        home_stats, away_stats: dicts con stats del box score
        p_pregame:      probabilidad pre-partido del local (0-1)
        market_ml_prob: probabilidad implicita del mercado (0-1)
        elo_diff:       ELO_HOME - ELO_AWAY (pre-partido)

    Returns:
        OrderedDict con las 17 features en orden de INGAME_EXTENDED_FEATURES.
    """
    # Empezar con las 8 basicas
    features = compute_ingame_differentials(home_stats, away_stats, p_pregame)

    # Posesiones
    home_poss = max(1.0, float(home_stats.get("POSS", 1.0)))
    away_poss = max(1.0, float(away_stats.get("POSS", 1.0)))
    avg_poss = (home_poss + away_poss) / 2.0

    # --- Features extendidas ---

    # P_PREGAME sin transformar (XGBoost captura no-linealidad automaticamente)
    features["P_PREGAME"] = float(p_pregame)

    # Probabilidad implicita del mercado (benchmark)
    features["MARKET_ML_PROB"] = float(market_ml_prob)

    # Diferencial Elo pre-partido
    features["ELO_DIFF"] = float(elo_diff)

    # Posesiones del local (proxy del ritmo de juego)
    features["HOME_POSS"] = home_poss

    # Rebotes ofensivos (segundas oportunidades — Ouyang: gana importancia en H2+)
    home_oreb = float(home_stats.get("OREB", 0))
    away_oreb = float(away_stats.get("OREB", 0))
    features["DIFF_OREB_RATE"] = (home_oreb - away_oreb) / avg_poss

    # Robos (crean posesiones extra)
    home_stl = float(home_stats.get("STL", 0))
    away_stl = float(away_stats.get("STL", 0))
    features["DIFF_STL_RATE"] = (home_stl - away_stl) / avg_poss

    # Bloqueos (defensa interior)
    home_blk = float(home_stats.get("BLK", 0))
    away_blk = float(away_stats.get("BLK", 0))
    features["DIFF_BLK_RATE"] = (home_blk - away_blk) / avg_poss

    # Tasa de intentos de 3 (shot selection — indica estilo de juego)
    home_fga = max(1, int(home_stats.get("FGA", 1)))
    away_fga = max(1, int(away_stats.get("FGA", 1)))
    home_fg3a = float(home_stats.get("FG3A", 0))
    away_fg3a = float(away_stats.get("FG3A", 0))
    features["HOME_FG3A_RATE"] = home_fg3a / home_fga
    features["AWAY_FG3A_RATE"] = away_fg3a / away_fga

    return features


def compute_spread_features(
    home_stats: dict,
    away_stats: dict,
    p_pregame: float = 0.5,
    market_ml_prob: float = 0.5,
    elo_diff: float = 0.0,
    pregame_spread: float = 0.0,
    period: int = 1,
) -> OrderedDict:
    """Calcula features para prediccion de spread cover in-game.

    Incluye las 17 features extendidas de ML + 5 features especificas de spread.
    El concepto clave es Implied Probability Shift: la desviacion del score
    respecto a la EXPECTATIVA del mercado.

    Ejemplo: spread=-6, periodo=Q2 (50% del juego), home gana por 1:
      expected_diff = -(-6) * 0.50 = +3 (deberia ganar por 3 a esta altura)
      spread_pace = 1 - 3 = -2 (esta -2 debajo de la expectativa)
      → sugiere que home NO cubrira el spread

    Args:
        home_stats, away_stats: dicts con stats del box score
        p_pregame:      probabilidad pre-partido del local (0-1)
        market_ml_prob: probabilidad implicita del mercado (0-1)
        elo_diff:       ELO_HOME - ELO_AWAY
        pregame_spread: spread del local (negativo = favorito). Ej: -5.0
        period:         periodo completado (1, 2, 3 o 4)

    Returns:
        OrderedDict con las 22 features en orden de INGAME_SPREAD_FEATURES.
    """
    # Empezar con las 17 features extendidas de ML
    features = compute_extended_features(
        home_stats, away_stats, p_pregame, market_ml_prob, elo_diff
    )

    # Posesiones y puntos actuales
    home_pts = float(home_stats.get("PTS", 0))
    away_pts = float(away_stats.get("PTS", 0))
    score_diff = home_pts - away_pts
    home_poss = max(1.0, float(home_stats.get("POSS", 1.0)))
    away_poss = max(1.0, float(away_stats.get("POSS", 1.0)))

    # Fraccion del juego transcurrida (Q1=0.25, Q2=0.50, Q3=0.75, Q4=1.0)
    period_fraction = min(period / 4.0, 1.0)

    # --- Feature 1: PREGAME_SPREAD (contexto crudo) ---
    features["PREGAME_SPREAD"] = float(pregame_spread)

    # --- Feature 2: SPREAD_PACE ---
    # Cuanto se desvia el score actual de lo "esperado" por el mercado.
    # Si spread=-6 y vamos al 50%, el mercado espera que home gane por 3 a este punto.
    # spread_pace > 0 → home rindiendo MEJOR que expectativa del mercado.
    # Nota: spread negativo = home favorito, asi que esperamos score_diff > 0.
    # expected_diff = -spread * period_fraction (spread ya es desde perspectiva home)
    expected_diff_at_this_point = (-pregame_spread) * period_fraction
    features["SPREAD_PACE"] = score_diff - expected_diff_at_this_point

    # --- Feature 3: SPREAD_MARGIN_NORM (Brownian Motion adaptado a spread) ---
    # Cuantas "desviaciones estandar" esta el margen actual del spread.
    # Normalizado por posesiones RESTANTES (no jugadas).
    # Inspirado en Stern 1994: la incertidumbre es sqrt(posesiones restantes).
    remaining_poss = max(1.0, EXPECTED_TOTAL_POSS - (home_poss + away_poss))
    margin_vs_spread = score_diff - (-pregame_spread)  # positivo = cubriendo
    features["SPREAD_MARGIN_NORM"] = margin_vs_spread / math.sqrt(remaining_poss)

    # --- Feature 4: PERIOD_FRACTION ---
    features["PERIOD_FRACTION"] = period_fraction

    # --- Feature 5: PTS_NEEDED_TO_COVER ---
    # Cuantos puntos netos necesita anotar home para cubrir el spread.
    # Si spread=-6 y score_diff=+2 → necesita +4 mas netos.
    # Si spread=-6 y score_diff=+8 → ya cubre por +2 (valor negativo = ya cubre).
    features["PTS_NEEDED_TO_COVER"] = (-pregame_spread) - score_diff

    return features


def compute_total_features(
    home_stats: dict,
    away_stats: dict,
    pregame_total: float = 215.0,
    period: int = 1,
    p_pregame: float = 0.5,
    market_ml_prob: float = 0.5,
) -> OrderedDict:
    """Calcula features para prediccion de total O/U in-game.

    Completamente diferente a ML/spread: se enfoca en RITMO de anotacion.
    El concepto clave es Run Rate (adaptado de cricket): proyectar el ritmo
    actual de scoring al juego completo y comparar con la linea.

    Regresion a la media: FG% extremo en periodos tempranos tiende a revertir.
    Un FG% z-score > +2 en Q1 predice enfriamiento → el total projectado baja.

    Ejemplo: total=215.5, Q1 termina 32-30 (62 pts en ~50 poss combinadas):
      scoring_pace = 62/50 = 1.24 pts/poss
      projected_total = 1.24 * 200 = 248
      pace_vs_line = 248 - 215.5 = +32.5 → on pace for OVER

    Args:
        home_stats, away_stats: dicts con stats del box score
        pregame_total: linea O/U pregame. Ej: 215.5
        period:        periodo completado (1, 2, 3 o 4)
        p_pregame:     probabilidad pre-partido (proxy de calidad de equipos)
        market_ml_prob: probabilidad implicita del mercado

    Returns:
        OrderedDict con las 15 features en orden de INGAME_TOTAL_FEATURES.
    """
    features = OrderedDict()

    # Stats basicas
    home_pts = float(home_stats.get("PTS", 0))
    away_pts = float(away_stats.get("PTS", 0))
    home_poss = max(1.0, float(home_stats.get("POSS", 1.0)))
    away_poss = max(1.0, float(away_stats.get("POSS", 1.0)))
    combined_poss = home_poss + away_poss
    combined_pts = home_pts + away_pts

    # Fraccion del juego
    period_fraction = min(period / 4.0, 1.0)

    # --- Feature 1: PREGAME_TOTAL (contexto de la linea) ---
    features["PREGAME_TOTAL"] = float(pregame_total)

    # --- Feature 2: COMBINED_PTS (puntos anotados hasta ahora) ---
    features["COMBINED_PTS"] = combined_pts

    # --- Feature 3: SCORING_PACE ---
    # Puntos por posesion combinados. Run Rate del juego.
    # Nota: si combined_poss es muy bajo (inicio del juego), el pace es ruidoso.
    features["SCORING_PACE"] = combined_pts / combined_poss

    # --- Feature 4: PROJECTED_TOTAL ---
    # Proyeccion lineal: si mantienen este ritmo, cuantos puntos habra al final.
    # Usamos EXPECTED_TOTAL_POSS (~200) como posesiones esperadas totales.
    projected = (combined_pts / combined_poss) * EXPECTED_TOTAL_POSS
    features["PROJECTED_TOTAL"] = projected

    # --- Feature 5: PACE_VS_LINE ---
    # Diferencia entre proyeccion y linea. Positivo = on pace for OVER.
    features["PACE_VS_LINE"] = projected - pregame_total

    # --- Feature 6: PERIOD_FRACTION ---
    features["PERIOD_FRACTION"] = period_fraction

    # --- Feature 7: COMBINED_POSS (ritmo real) ---
    features["COMBINED_POSS"] = combined_poss

    # --- Feature 8: COMBINED_FG_PCT ---
    # Eficiencia general de tiro. Alto FG% = mas puntos por intento.
    home_fga = max(1, int(home_stats.get("FGA", 1)))
    away_fga = max(1, int(away_stats.get("FGA", 1)))
    home_fgm = float(home_stats.get("FGM", 0))
    away_fgm = float(away_stats.get("FGM", 0))
    features["COMBINED_FG_PCT"] = (home_fgm + away_fgm) / (home_fga + away_fga)

    # --- Feature 9: COMBINED_FG3_RATE ---
    # Tasa de triples combinada. Mas triples = mas puntos por intento convertido.
    home_fg3a = float(home_stats.get("FG3A", 0))
    away_fg3a = float(away_stats.get("FG3A", 0))
    total_fga = max(1, home_fga + away_fga)
    features["COMBINED_FG3_RATE"] = (home_fg3a + away_fg3a) / total_fga

    # --- Feature 10: COMBINED_FT_RATE ---
    # Frecuencia de tiros libres. Mas FTA = mas posesiones usadas en TL.
    home_fta = float(home_stats.get("FTA", 0))
    away_fta = float(away_stats.get("FTA", 0))
    features["COMBINED_FT_RATE"] = (home_fta + away_fta) / total_fga

    # --- Feature 11: COMBINED_TOV_RATE ---
    # Turnovers combinados / posesiones. Menos posesiones → menos puntos.
    home_tov = float(home_stats.get("TOV", 0))
    away_tov = float(away_stats.get("TOV", 0))
    features["COMBINED_TOV_RATE"] = (home_tov + away_tov) / combined_poss

    # --- Feature 12: FG_PCT_ZSCORE ---
    # Regresion a la media: cuanto se desvia el FG% actual del historico.
    # Z > +2 = tiro insosteniblemente bueno, el ritmo va a bajar.
    # Z < -2 = tiro insosteniblemente malo, el ritmo va a subir.
    current_fg_pct = (home_fgm + away_fgm) / (home_fga + away_fga)
    features["FG_PCT_ZSCORE"] = (
        (current_fg_pct - HISTORICAL_FG_PCT) / HISTORICAL_FG_PCT_STD
    )

    # --- Feature 13: P_PREGAME (proxy de calidad) ---
    features["P_PREGAME"] = float(p_pregame)

    # --- Feature 14: MARKET_ML_PROB ---
    features["MARKET_ML_PROB"] = float(market_ml_prob)

    # --- Feature 15: DIFF_EP_RATE ---
    # Expected Points por posesion: (2*FG2M + 3*FG3M + 1*FTM) / POSS
    # Captura la "calidad de tiro" mejor que FG% solo.
    home_fg3m = float(home_stats.get("FG3M", 0))
    away_fg3m = float(away_stats.get("FG3M", 0))
    home_ftm = float(home_stats.get("FTM", 0))
    away_ftm = float(away_stats.get("FTM", 0))
    home_fg2m = home_fgm - home_fg3m
    away_fg2m = away_fgm - away_fg3m

    home_ep = (2 * home_fg2m + 3 * home_fg3m + home_ftm) / home_poss
    away_ep = (2 * away_fg2m + 3 * away_fg3m + away_ftm) / away_poss
    features["DIFF_EP_RATE"] = home_ep - away_ep

    return features


def box_score_to_stats_dict(box_team: dict) -> dict:
    """Convierte el dict de get_live_box_score() al formato esperado.

    get_live_box_score() retorna keys en minuscula:
        {"pts": 28, "fgm": 10, "fga": 22, "fg3m": 4, ...}

    Este modulo espera keys en mayuscula:
        {"PTS": 28, "FGM": 10, "FGA": 22, "FG3M": 4, ...}

    Ademas calcula FG_PCT, FG3_PCT, FT_PCT que no vienen en el live box score.

    Args:
        box_team: dict de get_live_box_score()["home"] o ["away"]

    Returns:
        dict con keys normalizadas para compute_ingame_differentials()
    """
    # Mapeo de keys (live_game_state.py usa minusculas)
    stats = {k.upper(): v for k, v in box_team.items()}

    # Calcular porcentajes (no vienen en el live box score)
    fga = max(1, stats.get("FGA", 1))
    fg3a = max(1, stats.get("FG3A", 1))
    fta = max(1, stats.get("FTA", 1))

    stats["FG_PCT"] = stats.get("FGM", 0) / fga
    stats["FG3_PCT"] = stats.get("FG3M", 0) / fg3a
    stats["FT_PCT"] = stats.get("FTM", 0) / fta

    # OREB y DREB no vienen por separado en live box score
    # Solo viene reb (total). Estimamos: DREB ≈ 70% de REB (promedio historico NBA)
    if "OREB" not in stats and "REB" in stats:
        stats["OREB"] = int(stats["REB"] * 0.3)
        stats["DREB"] = stats["REB"] - stats["OREB"]

    # Renombrar POSSESSIONS → POSS si viene de live_game_state
    if "POSSESSIONS" in stats and "POSS" not in stats:
        stats["POSS"] = stats["POSSESSIONS"]

    return stats


def quarter_data_row_to_stats(row, prefix: str) -> dict:
    """Convierte una fila de QuarterData.sqlite a dict de stats.

    La tabla tiene columnas como HOME_FGM, HOME_FGA, etc.
    Las extraemos y quitamos el prefijo para tener {"FGM": 10, "FGA": 22, ...}.

    Args:
        row: dict-like (de pd.Series o sqlite3.Row)
        prefix: "HOME_" o "AWAY_"

    Returns:
        dict con keys sin prefijo: {"FGM": 10, "FGA": 22, "POSS": 24.5, ...}
    """
    stats = {}
    for col in STAT_COLUMNS + ["POSS"]:
        key = f"{prefix}{col}"
        val = row.get(key, row[key] if hasattr(row, '__getitem__') else 0)
        stats[col] = float(val) if val is not None else 0.0
    return stats


def quarter_data_row_to_pregame_context(row) -> dict:
    """Extrae datos pregame de una fila de QuarterData.sqlite enriquecida.

    Retorna los campos necesarios para compute_spread_features() y
    compute_total_features(). Si los campos no existen (tabla vieja sin
    enriquecer), retorna defaults seguros.

    Args:
        row: dict-like (pd.Series o sqlite3.Row)

    Returns:
        dict con:
          pregame_spread: float (0.0 si no disponible)
          pregame_total: float (215.0 si no disponible)
          pregame_ml_prob: float (0.5 si no disponible)
          spread_cover: int | None (target para spread model)
          total_over: int | None (target para total model)
          final_margin: float
          final_total: float
    """
    def _safe_get(key, default=None):
        try:
            val = row.get(key, default) if hasattr(row, 'get') else row[key]
            return default if val is None or (isinstance(val, float) and math.isnan(val)) else val
        except (KeyError, IndexError):
            return default

    return {
        "pregame_spread": float(_safe_get("PREGAME_SPREAD", 0.0)),
        "pregame_total": float(_safe_get("PREGAME_TOTAL", 215.0)),
        "pregame_ml_prob": float(_safe_get("PREGAME_ML_PROB", 0.5)),
        "spread_cover": _safe_get("SPREAD_COVER"),
        "total_over": _safe_get("TOTAL_OVER"),
        "final_margin": float(_safe_get("FINAL_MARGIN", 0.0)),
        "final_total": float(_safe_get("FINAL_TOTAL", 0.0)),
    }


# Re-exportar STAT_COLUMNS para que collect_quarter_data.py pueda usarlas
STAT_COLUMNS = [
    "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB",
    "AST", "STL", "BLK", "TOV", "PTS",
]
