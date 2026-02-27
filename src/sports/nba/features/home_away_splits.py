"""
Home/Away Splits para el modelo NBA ML.

Calcula la DIFERENCIA entre el rendimiento de un equipo en casa vs de visita.
Esto captura la "ventaja de localía" individual de cada equipo.

Ejemplo: Denver con altitud tiene SPLIT_PLUS_MINUS = +8
         (8 puntos mejor en casa que de visita).
         Un equipo sin ventaja de localía tendría SPLIT = 0.

Usamos DIFERENCIAS (home - away), no niveles, por la misma razon que
momentum: los niveles (HOME_AVG_PTS) correlacionan alto con el promedio
de temporada, pero la diferencia es informacion genuinamente nueva.

Datos fuente:
    - TeamData.sqlite: promedios acumulativos por fecha (avg * GP = total)
    - OddsData.sqlite: registros de juegos con Home/Away labels

Truco: cuando GP de un equipo aumenta en TeamData, cruzamos con OddsData
para saber si ese partido fue en casa o de visita. Asi separamos los
totales acumulados en home_totals y away_totals.

Genera 7 features por equipo = 14 columnas totales:
    SPLIT_PTS, SPLIT_PLUS_MINUS, SPLIT_FG_PCT, SPLIT_FG3_PCT,
    SPLIT_REB, SPLIT_AST, SPLIT_TOV

Usado por Create_Games.py (entrenamiento) y main.py (prediccion).
"""

from datetime import datetime

import pandas as pd

# Stats base para calcular totales acumulativos (mismo set que RollingAverages)
_CUMULATIVE_STATS = [
    "PTS", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
    "OREB", "TOV", "PLUS_MINUS", "REB", "AST",
]

# Columnas a leer de TeamData (eficiencia: solo lo necesario)
_SELECT_COLS = "TEAM_NAME, GP, " + ", ".join(_CUMULATIVE_STATS)

# Nombres de las features split finales
_SPLIT_FEATURE_NAMES = [
    "SPLIT_PTS", "SPLIT_PLUS_MINUS", "SPLIT_FG_PCT", "SPLIT_FG3_PCT",
    "SPLIT_REB", "SPLIT_AST", "SPLIT_TOV",
]

# Mapeo de nombres historicos (mismo que RollingAverages)
_TEAM_ALIASES = {
    "Los Angeles Clippers": "LA Clippers",
    "Charlotte Bobcats": "Charlotte Hornets",
    "New Orleans Hornets": "New Orleans Pelicans",
    "New Orleans/Oklahoma City Hornets": "New Orleans Pelicans",
}


def _normalize_team(name):
    """Normaliza nombre de equipo usando aliases conocidos."""
    return _TEAM_ALIASES.get(name, name)


def _build_game_number_map(odds_con, odds_table):
    """Construye un mapa (team_normalizado, game_number) -> 'home' | 'away'.

    En vez de matchear por fecha (fragil por offsets de 1 dia entre
    TeamData y OddsData), matcheamos por NUMERO DE JUEGO.

    Logica:
        - OddsData tiene juegos de cada equipo en orden cronologico
        - TeamData tiene cambios de GP en orden cronologico
        - El juego #N en OddsData = el N-esimo cambio de GP en TeamData
        - Independiente de la fecha exacta → evita errores de offset

    Args:
        odds_con: conexion SQLite a OddsData.sqlite
        odds_table: nombre de la tabla de odds para esta temporada

    Returns:
        dict[(team_normalizado, game_number_int)] -> 'home' | 'away'
    """
    game_map = {}
    try:
        rows = odds_con.execute(
            f'SELECT Date, Home, Away FROM "{odds_table}" ORDER BY Date'
        ).fetchall()
    except Exception:
        return game_map

    # Contador de juegos por equipo (el N-esimo juego de cada equipo)
    team_game_count = {}

    for _date_val, home, away in rows:
        home_norm = _normalize_team(home)
        away_norm = _normalize_team(away)

        # Incrementar contador del equipo local
        team_game_count[home_norm] = team_game_count.get(home_norm, 0) + 1
        game_map[(home_norm, team_game_count[home_norm])] = "home"

        # Incrementar contador del equipo visitante
        team_game_count[away_norm] = team_game_count.get(away_norm, 0) + 1
        game_map[(away_norm, team_game_count[away_norm])] = "away"

    return game_map


def build_season_split_data(teams_con, odds_con, season_key, start_date_str, end_date_str):
    """Construye datos de splits home/away para una temporada completa.

    Similar a build_season_game_logs() de RollingAverages, pero ademas
    cruza con OddsData para saber si cada partido fue en casa o de visita.

    Proceso:
    1. Pre-carga todos los juegos de OddsData para lookup rapido
    2. Recorre TeamData cronologicamente
    3. Cuando GP de un equipo aumenta: extrae stats del partido
    4. Busca en OddsData si fue home o away
    5. Acumula totales separados por ubicacion
    6. Guarda snapshot por GP para lookup durante Create_Games

    Args:
        teams_con: conexion SQLite a TeamData.sqlite
        odds_con: conexion SQLite a OddsData.sqlite
        season_key: clave de temporada (ej: "2024-25")
        start_date_str: "YYYY-MM-DD" inicio de temporada
        end_date_str: "YYYY-MM-DD" fin de temporada

    Returns:
        dict[team_normalizado][gp] -> {
            "home_gp": int,
            "away_gp": int,
            "home_totals": {stat: float},
            "away_totals": {stat: float},
        }
    """
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d").date()

    # Paso 1: Pre-cargar ubicaciones de OddsData indexadas por numero de juego
    # Usamos game_number (no fecha) para evitar errores de offset temporal
    odds_table = _find_odds_table(odds_con, season_key)
    if odds_table is None:
        return {}

    game_map = _build_game_number_map(odds_con, odds_table)

    # Paso 2: Obtener todas las tablas-fecha de TeamData en el rango
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

    # Estado por equipo
    # team -> {"last_gp", "last_totals", "home_gp", "away_gp", "home_totals", "away_totals", "game_count"}
    team_state = {}
    # Resultado: team -> {gp -> snapshot}
    split_data = {}

    # Paso 3: Recorrer fechas cronologicamente
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
            team_raw = row_dict["TEAM_NAME"]
            team = _normalize_team(team_raw)
            gp = float(row_dict["GP"])

            if gp == 0:
                continue

            # Inicializar equipo si es nuevo
            if team not in team_state:
                team_state[team] = {
                    "last_gp": 0,
                    "last_totals": {stat: 0.0 for stat in _CUMULATIVE_STATS},
                    "home_gp": 0,
                    "away_gp": 0,
                    "home_totals": {stat: 0.0 for stat in _CUMULATIVE_STATS},
                    "away_totals": {stat: 0.0 for stat in _CUMULATIVE_STATS},
                    "game_count": 0,  # contador de juegos para match con OddsData
                }
                split_data[team] = {}

            state = team_state[team]

            # Solo procesar cuando GP aumenta (nuevo partido jugado)
            if gp <= state["last_gp"]:
                continue

            # Calcular totales acumulados actuales: avg * GP
            current_totals = {}
            for stat in _CUMULATIVE_STATS:
                current_totals[stat] = float(row_dict[stat]) * gp

            # Extraer stats de este partido: total_ahora - total_antes
            game_stats = {}
            for stat in _CUMULATIVE_STATS:
                game_stats[stat] = current_totals[stat] - state["last_totals"][stat]

            # Buscar si fue home o away usando numero de juego
            # El N-esimo cambio de GP en TeamData = juego #N en OddsData
            state["game_count"] += 1
            location = game_map.get((team, state["game_count"]))

            if location == "home":
                state["home_gp"] += 1
                for stat in _CUMULATIVE_STATS:
                    state["home_totals"][stat] += game_stats[stat]
            elif location == "away":
                state["away_gp"] += 1
                for stat in _CUMULATIVE_STATS:
                    state["away_totals"][stat] += game_stats[stat]
            # Si location es None (no encontrado), ignoramos este partido
            # para los splits (no sabemos si fue home o away)

            # Guardar snapshot para este GP
            split_data[team][gp] = {
                "home_gp": state["home_gp"],
                "away_gp": state["away_gp"],
                "home_totals": dict(state["home_totals"]),
                "away_totals": dict(state["away_totals"]),
            }

            # Actualizar estado previo
            state["last_gp"] = gp
            state["last_totals"] = current_totals

    # Agregar aliases para que el lookup funcione con ambos nombres
    for old_name, new_name in _TEAM_ALIASES.items():
        if old_name in split_data and new_name not in split_data:
            split_data[new_name] = split_data[old_name]
        elif new_name in split_data and old_name not in split_data:
            split_data[old_name] = split_data[new_name]

    return split_data


def _find_odds_table(odds_con, season_key):
    """Busca la tabla de OddsData para una temporada (misma logica que Create_Games)."""
    candidates = [
        f"odds_{season_key}_new",
        f"odds_{season_key}",
        f"{season_key}_new",
        season_key,
    ]
    for table_name in candidates:
        cursor = odds_con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        if cursor.fetchone() is not None:
            return table_name
    return None


def get_team_split_features(split_data, team, current_gp):
    """Obtiene features de split home/away para un equipo en un momento dado.

    Busca el snapshot de splits cuando el equipo tenia exactly `current_gp`
    juegos jugados. Calcula la diferencia entre rendimiento en casa vs visita.

    Si el equipo no tiene suficientes datos (0 juegos home o 0 away),
    retorna todo 0s — semanticamente correcto: "no hay diferencia conocida".

    Ejemplo:
        Si Boston tiene home_avg_PTS=115 y away_avg_PTS=108:
        SPLIT_PTS = 115 - 108 = +7 (7 puntos mejor en casa)

    Para porcentajes, calcula desde componentes:
        SPLIT_FG_PCT = home_FGM/home_FGA - away_FGM/away_FGA

    Args:
        split_data: dict[team][gp] -> snapshot (de build_season_split_data)
        team: nombre del equipo
        current_gp: GP actual (de TeamData)

    Returns:
        dict con 7 features: SPLIT_PTS, SPLIT_PLUS_MINUS, etc.
    """
    # Inicializar todo en 0 (sin diferencia conocida)
    features = {name: 0.0 for name in _SPLIT_FEATURE_NAMES}

    team_norm = _normalize_team(team)

    # Buscar datos del equipo
    team_splits = split_data.get(team_norm)
    if team_splits is None:
        return features

    # Buscar snapshot para este GP
    snapshot = team_splits.get(current_gp)
    if snapshot is None:
        # Intentar con GP cercano (por si hay pequenas discrepancias)
        closest_gp = None
        for gp in sorted(team_splits.keys()):
            if gp <= current_gp:
                closest_gp = gp
        if closest_gp is not None:
            snapshot = team_splits[closest_gp]
        else:
            return features

    home_gp = snapshot["home_gp"]
    away_gp = snapshot["away_gp"]

    # Necesitamos al menos 1 juego en cada ubicacion para calcular diferencia
    if home_gp == 0 or away_gp == 0:
        return features

    ht = snapshot["home_totals"]
    at = snapshot["away_totals"]

    # Stats simples: (home_total / home_gp) - (away_total / away_gp)
    features["SPLIT_PTS"] = ht["PTS"] / home_gp - at["PTS"] / away_gp
    features["SPLIT_PLUS_MINUS"] = ht["PLUS_MINUS"] / home_gp - at["PLUS_MINUS"] / away_gp
    features["SPLIT_REB"] = ht["REB"] / home_gp - at["REB"] / away_gp
    features["SPLIT_AST"] = ht["AST"] / home_gp - at["AST"] / away_gp
    features["SPLIT_TOV"] = ht["TOV"] / home_gp - at["TOV"] / away_gp

    # Porcentajes: calculados desde componentes (correcto estadisticamente)
    # home_FG_PCT = sum(FGM_home) / sum(FGA_home)
    home_fga = ht["FGA"]
    away_fga = at["FGA"]
    if home_fga > 0 and away_fga > 0:
        features["SPLIT_FG_PCT"] = ht["FGM"] / home_fga - at["FGM"] / away_fga

    home_fg3a = ht["FG3A"]
    away_fg3a = at["FG3A"]
    if home_fg3a > 0 and away_fg3a > 0:
        features["SPLIT_FG3_PCT"] = ht["FG3M"] / home_fg3a - at["FG3M"] / away_fg3a

    return features


def add_split_features_to_frame(frame, home_split_list, away_split_list):
    """Agrega columnas de split features al DataFrame de juegos.

    Las features del equipo local van sin sufijo (SPLIT_PTS).
    Las del visitante con sufijo .1 (SPLIT_PTS.1).
    Consistente con el patron del pipeline (home="", away=".1").

    Args:
        frame: DataFrame con los juegos (ya construido)
        home_split_list: list of dicts (uno por juego, equipo local)
        away_split_list: list of dicts (uno por juego, equipo visitante)

    Returns:
        DataFrame con 14 columnas nuevas de split features
    """
    if not home_split_list:
        return frame

    home_df = pd.DataFrame(home_split_list, index=frame.index)
    away_df = pd.DataFrame(away_split_list, index=frame.index)
    away_df.columns = [f"{col}.1" for col in away_df.columns]

    result = pd.concat([frame, home_df, away_df], axis=1)
    result = result.fillna(0)
    return result
