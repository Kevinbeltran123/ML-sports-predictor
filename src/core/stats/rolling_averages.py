"""
Promedios moviles (rolling averages) para el modelo NBA ML.

En vez de usar el promedio de toda la temporada, calcula promedios
sobre las ultimas N partidas, capturando la forma reciente del equipo.

Truco matematico:
    La NBA API devuelve promedios por partido y GP (games played).
    total_acumulado = promedio_por_partido * GP
    rolling_5 = (total_ahora - total_hace_5_juegos) / 5

Genera 3 tipos de features por equipo:
    - ROLL_5_{stat}  : promedio ultimas 5 partidas
    - ROLL_10_{stat} : promedio ultimas 10 partidas
    - MOM_{stat}     : ROLL_5 - ROLL_10 (momentum / tendencia)

Usado por Create_Games.py (entrenamiento) y main.py (prediccion).
"""

import numpy as np
import pandas as pd

# ── Helpers para features avanzados (NET_RTG_TREND + PACE_CV) ─────────────────


def _extract_recent_game_stats(team_log, current_gp, window=10):
    """Extrae stats individuales de los últimos N partidos.

    team_log tiene totales ACUMULADOS: (gp, {PTS: 850, ...}).
    Para obtener stats de un partido individual, restamos entradas consecutivas:
        game_stats = totals[gp=30] - totals[gp=29]

    Args:
        team_log: list of (gp, {stat: total_acumulado}) ordenado por GP
        current_gp: GP actual del equipo
        window: cuántos partidos hacia atrás mirar

    Returns:
        list of dicts [{stat: valor_de_ese_partido}, ...] ordenados cronológicamente.
        Lista vacía si no hay suficientes datos.
    """
    if not team_log or current_gp < 2:
        return []

    # Encontrar el índice de current_gp en team_log
    idx_current = None
    for i, (gp, _) in enumerate(team_log):
        if gp == current_gp:
            idx_current = i
            break

    if idx_current is None or idx_current < 1:
        return []

    # Caminar hacia atrás, calculando diferencias entre entradas consecutivas
    game_stats = []
    start_idx = max(0, idx_current - window)

    for i in range(start_idx + 1, idx_current + 1):
        prev_gp, prev_totals = team_log[i - 1]
        curr_gp_val, curr_totals = team_log[i]

        # Diferencia = stats de los partidos entre prev_gp y curr_gp_val
        n_games = curr_gp_val - prev_gp
        if n_games <= 0:
            continue

        game = {}
        for stat in curr_totals:
            game[stat] = (curr_totals[stat] - prev_totals[stat]) / n_games
        game_stats.append(game)

    return game_stats


def _compute_trend_slope(team_log, current_gp, window=10):
    """Calcula la pendiente de tendencia lineal del net rating (PLUS_MINUS).

    Ajusta una línea recta (regresión lineal) a los valores de PLUS_MINUS
    de los últimos N partidos. La pendiente indica:
        > 0: equipo mejorando (ej: +0.5 = mejora 0.5 pts/partido)
        < 0: equipo empeorando
        ≈ 0: rendimiento estable

    Por qué es mejor que MOM_PLUS_MINUS:
        MOM usa solo 2 puntos (promedio 5 últimos vs promedio 10 últimos).
        La pendiente usa TODOS los puntos → estimación más robusta.

    Matemáticamente: y = mx + b, donde y=PLUS_MINUS, x=índice del partido.
    Usamos np.polyfit(x, y, grado=1) que minimiza el error cuadrático.
    """
    game_stats = _extract_recent_game_stats(team_log, current_gp, window)

    # Necesitamos al menos 3 puntos para una tendencia significativa
    if len(game_stats) < 3:
        return 0.0

    pm_values = [g.get("PLUS_MINUS", 0.0) for g in game_stats]

    # polyfit devuelve [pendiente, intercepto] para grado 1
    try:
        slope, _ = np.polyfit(np.arange(len(pm_values)), pm_values, 1)
        return round(float(slope), 4)
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def _compute_pace_cv(team_log, current_gp, window=10):
    """Calcula el coeficiente de variación del ritmo (posesiones) por partido.

    POSS por partido = FGA - OREB + TOV + 0.44 × FTA (fórmula Dean Oliver)
    CV = std(POSS) / mean(POSS) — adimensional, comparable entre equipos.

    CV bajo (~0.05) = equipo consistente, siempre juega al mismo ritmo
    CV alto (~0.15) = equipo que varía su ritmo según el rival

    Útil especialmente para O/U: si ambos equipos son consistentes,
    el total de puntos es más predecible.
    """
    game_stats = _extract_recent_game_stats(team_log, current_gp, window)

    if len(game_stats) < 3:
        return 0.0

    # Calcular posesiones por partido usando fórmula Dean Oliver
    poss_values = []
    for g in game_stats:
        fga = g.get("FGA", 0)
        oreb = g.get("OREB", 0)
        tov = g.get("TOV", 0)
        fta = g.get("FTA", 0)
        poss = fga - oreb + tov + 0.44 * fta
        if poss > 0:
            poss_values.append(poss)

    if len(poss_values) < 3:
        return 0.0

    mean_poss = np.mean(poss_values)
    if mean_poss < 1.0:
        return 0.0

    return round(float(np.std(poss_values) / mean_poss), 4)


# Stats base para calcular totales acumulativos
_CUMULATIVE_STATS = [
    "PTS", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
    "OREB", "TOV", "PLUS_MINUS", "REB", "AST",
]

# Columnas a leer de cada tabla (eficiencia: solo lo necesario)
_SELECT_COLS = "TEAM_NAME, GP, " + ", ".join(_CUMULATIVE_STATS)

# Nombres de las features rolling finales
_FEATURE_NAMES = [
    "PTS", "PLUS_MINUS", "TOV", "REB", "AST", "FG_PCT", "FG3_PCT",
]

# Ventanas de rolling (en numero de partidos)
WINDOWS = [5, 10]

# Mapeo de nombres historicos de equipos a nombres actuales
# Necesario porque TeamData usa el nombre de la epoca (ej: "Los Angeles Clippers")
# pero OddsData puede usar el nombre actual (ej: "LA Clippers")
_TEAM_ALIASES = {
    "Los Angeles Clippers": "LA Clippers",
    "Charlotte Bobcats": "Charlotte Hornets",
    "New Orleans Hornets": "New Orleans Pelicans",
    "New Orleans/Oklahoma City Hornets": "New Orleans Pelicans",
}


def build_season_game_logs(teams_con, start_date_str, end_date_str):
    """Construye historial de juegos por equipo para una temporada.

    Recorre todas las tablas de TeamData.sqlite en el rango de fechas.
    Para cada equipo, registra los totales acumulativos cada vez que su GP
    (Games Played) aumenta — es decir, cada vez que jugo un partido.

    Ejemplo: Si Boston Celtics tiene GP=30 el 15 de enero y GP=31 el 17
    de enero, sabemos que jugaron un partido entre esas fechas.

    Args:
        teams_con: conexion SQLite a TeamData.sqlite
        start_date_str: "YYYY-MM-DD" inicio de temporada
        end_date_str: "YYYY-MM-DD" fin de temporada

    Returns:
        dict[str, list[tuple]]: team_name -> [(gp, {stat: total_acumulado}), ...]
        Cada lista esta ordenada por GP ascendente.
    """
    from datetime import datetime

    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d").date()

    # Obtener todas las tablas-fecha en el rango de la temporada
    cursor = teams_con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )
    season_dates = []
    for (name,) in cursor.fetchall():
        try:
            dt = datetime.strptime(name, "%Y-%m-%d").date()
            if start_dt <= dt <= end_dt:
                season_dates.append(name)
        except ValueError:
            continue
    season_dates.sort()

    # Recorrer fechas cronologicamente y registrar cambios en GP
    team_logs = {}      # team_name -> [(gp, {stat: total}), ...]
    team_last_gp = {}   # team_name -> ultimo GP registrado

    for date_str in season_dates:
        try:
            rows = teams_con.execute(
                f'SELECT {_SELECT_COLS} FROM "{date_str}"'
            ).fetchall()
        except Exception:
            continue

        col_names = ["TEAM_NAME", "GP"] + list(_CUMULATIVE_STATS)

        for row in rows:
            row_dict = dict(zip(col_names, row))
            team = row_dict["TEAM_NAME"]
            gp = float(row_dict["GP"])

            if gp == 0:
                continue

            if team not in team_logs:
                team_logs[team] = []
                team_last_gp[team] = 0

            # Solo registrar cuando GP aumenta (nuevo partido jugado)
            if gp > team_last_gp[team]:
                totals = {}
                for stat in _CUMULATIVE_STATS:
                    # avg_por_partido * GP = total acumulado de la temporada
                    totals[stat] = float(row_dict[stat]) * gp
                team_logs[team].append((gp, totals))
                team_last_gp[team] = gp

    # Agregar aliases: si TeamData tiene "Los Angeles Clippers", tambien
    # hacer disponible como "LA Clippers" para que el lookup funcione
    for old_name, new_name in _TEAM_ALIASES.items():
        if old_name in team_logs and new_name not in team_logs:
            team_logs[new_name] = team_logs[old_name]
        elif new_name in team_logs and old_name not in team_logs:
            team_logs[old_name] = team_logs[new_name]

    return team_logs


def _compute_rolling(team_log, current_gp, window):
    """Calcula promedios rolling para un equipo en un punto dado.

    Ejemplo con ventana=5 y GP actual=30:
        total_30 = PTS_avg * 30       (puntos totales en 30 juegos)
        total_25 = PTS_avg_25 * 25    (puntos totales en 25 juegos)
        rolling_5_PTS = (total_30 - total_25) / 5

    Si el equipo tiene < 5 juegos, usa todos los disponibles como fallback.

    Args:
        team_log: list of (gp, {stat: total}) ordenado por GP
        current_gp: GP actual del equipo
        window: ventana de juegos (5 o 10)

    Returns:
        dict con features rolling, o None si no hay datos
    """
    # Buscar la entrada con GP == current_gp
    current_totals = None
    for gp, totals in team_log:
        if gp == current_gp:
            current_totals = totals
            break

    if current_totals is None:
        return None

    target_gp = current_gp - window

    if target_gp <= 0:
        # No hay suficientes juegos — usar todo lo disponible
        actual_window = current_gp
        rolling_totals = {stat: current_totals[stat] for stat in _CUMULATIVE_STATS}
    else:
        # Buscar la entrada con GP mas cercano a target_gp (sin pasarse)
        past_totals = None
        past_gp_found = None
        for gp, totals in team_log:
            if gp <= target_gp:
                past_totals = totals
                past_gp_found = gp
            elif gp > target_gp:
                break

        if past_totals is None:
            # No encontramos datos pasados — usar todo
            actual_window = current_gp
            rolling_totals = {stat: current_totals[stat] for stat in _CUMULATIVE_STATS}
        else:
            actual_window = current_gp - past_gp_found
            rolling_totals = {
                stat: current_totals[stat] - past_totals[stat]
                for stat in _CUMULATIVE_STATS
            }

    # Convertir totales a features
    w = actual_window if actual_window > 0 else 1
    result = {}

    # Stats simples: total_en_ventana / numero_de_juegos = promedio por partido
    result[f"ROLL_{window}_PTS"] = rolling_totals["PTS"] / w
    result[f"ROLL_{window}_PLUS_MINUS"] = rolling_totals["PLUS_MINUS"] / w
    result[f"ROLL_{window}_TOV"] = rolling_totals["TOV"] / w
    result[f"ROLL_{window}_REB"] = rolling_totals["REB"] / w
    result[f"ROLL_{window}_AST"] = rolling_totals["AST"] / w

    # Porcentajes: calculados desde sus componentes (correcto estadisticamente)
    # NO promediamos porcentajes — calculamos: sum(FGM) / sum(FGA)
    # Ejemplo: si un equipo tiro 10/20 un dia y 5/25 otro dia,
    # el promedio correcto es 15/45 = 33.3%, NO (50% + 20%) / 2 = 35%
    fga = rolling_totals["FGA"]
    fg3a = rolling_totals["FG3A"]
    result[f"ROLL_{window}_FG_PCT"] = rolling_totals["FGM"] / fga if fga > 0 else 0.0
    result[f"ROLL_{window}_FG3_PCT"] = rolling_totals["FG3M"] / fg3a if fg3a > 0 else 0.0

    return result


def get_team_rolling_features(team_log, current_gp, windows=None):
    """Obtiene features de momentum para un equipo.

    Solo genera features de MOMENTUM (ROLL_5 - ROLL_10), no los niveles
    rolling individuales. Razon: los niveles rolling (ej: ROLL_10_PTS)
    estan altamente correlacionados con los promedios de temporada (~0.91)
    y agregar features redundantes aumenta dimensionalidad sin aportar
    informacion nueva, perjudicando al modelo.

    El momentum SI es informacion nueva: captura si el equipo esta
    MEJORANDO o EMPEORANDO recientemente, algo que el promedio de
    temporada no puede ver.

    Momentum positivo = equipo en alza (ej: racha ganadora)
    Momentum negativo = equipo en caida (ej: lesiones recientes)

    Args:
        team_log: list of (gp, {stat: total}) del equipo
        current_gp: GP actual
        windows: ventanas a usar (default [5, 10])

    Returns:
        dict con features: MOM_* (7 features por equipo)
    """
    if windows is None:
        windows = WINDOWS

    features = {}
    rolling_results = {}

    for w in windows:
        result = _compute_rolling(team_log, current_gp, w)
        if result is not None:
            rolling_results[w] = result

    # Momentum: ventana corta minus ventana larga
    # Si ROLL_5_PTS > ROLL_10_PTS → equipo anotando mas recientemente
    short_w, long_w = min(windows), max(windows)
    if short_w in rolling_results and long_w in rolling_results:
        for feat in _FEATURE_NAMES:
            short_val = rolling_results[short_w][f"ROLL_{short_w}_{feat}"]
            long_val = rolling_results[long_w][f"ROLL_{long_w}_{feat}"]
            features[f"MOM_{feat}"] = short_val - long_val
    else:
        # Sin datos suficientes — momentum = 0 (no hay tendencia)
        for feat in _FEATURE_NAMES:
            features[f"MOM_{feat}"] = 0.0

    # ── Features avanzados (Mejoras del dataset externo) ─────────────────────

    # NET_RTG_TREND: pendiente lineal de PLUS_MINUS últimos 10 partidos
    # Captura si el equipo está mejorando o empeorando de forma más robusta
    # que MOM_PLUS_MINUS (que solo compara 2 promedios)
    features["NET_RTG_TREND"] = _compute_trend_slope(team_log, current_gp, window=10)

    # PACE_CV: coeficiente de variación del ritmo (posesiones)
    # Mide la consistencia del equipo en ritmo de juego
    features["PACE_CV"] = _compute_pace_cv(team_log, current_gp, window=10)

    return features


def add_rolling_features_to_frame(frame, home_rolling_list, away_rolling_list):
    """Agrega columnas de rolling features al DataFrame de juegos.

    Las features del equipo local van sin sufijo.
    Las del visitante con sufijo .1 (consistente con el pipeline).

    Args:
        frame: DataFrame con los juegos (ya construido)
        home_rolling_list: list of dicts (uno por juego, equipo local)
        away_rolling_list: list of dicts (uno por juego, equipo visitante)

    Returns:
        DataFrame con 18 columnas nuevas (7 MOM + NET_RTG_TREND + PACE_CV) × 2 equipos
    """
    if not home_rolling_list:
        return frame

    home_df = pd.DataFrame(home_rolling_list, index=frame.index)
    away_df = pd.DataFrame(away_rolling_list, index=frame.index)
    away_df.columns = [f"{col}.1" for col in away_df.columns]

    result = pd.concat([frame, home_df, away_df], axis=1)
    result = result.fillna(0)
    return result
