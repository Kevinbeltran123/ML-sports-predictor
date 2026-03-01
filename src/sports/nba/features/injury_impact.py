"""
Weighted Availability Index (AVAIL) para el modelo NBA ML.

Mide que fraccion de los "minutos esperados" de un equipo estan disponibles
para un partido dado. Pesa cada jugador por sus minutos promedio, capturando
que perder a una estrella (35 min) importa mucho mas que perder a un suplente
(8 min).

Ejemplo:
    Denver con Jokic (35 min avg) disponible: AVAIL ~ 1.0
    Denver sin Jokic: AVAIL ~ 0.85 (perdio ~15% de minutos de rotacion)
    Denver sin Jokic + Murray: AVAIL ~ 0.70

Asimetria train/predict:
    - Entrenamiento: usamos datos REALES (box scores) -> informacion perfecta
    - Prediccion: usamos reporte de lesiones (Tank01 API) -> estimacion
    Esto es practica estandar en sports ML.

Features (11 por partido):
    Basicas (4):
    - AVAIL_HOME/AWAY: fraccion de minutos de rotacion disponibles
    - AVAIL_QUALITY_HOME/AWAY: AVAIL ponderado por calidad (PM36)
    Granulares (7):
    - STAR_MISSING_HOME/AWAY: 1 si alguno de los top-2 PM36 no juega
    - N_ROTATION_OUT_HOME/AWAY: cuantos jugadores de rotacion no juegan
    - MISSING_BPM_HOME/AWAY: suma de PM36 de los ausentes (medida absoluta)
    - AVAIL_DIFF: AVAIL_QUALITY_HOME - AVAIL_QUALITY_AWAY (asimetria)

Usado por Create_Games.py (entrenamiento) y main.py (prediccion).
"""

import difflib
import os
import sqlite3
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

from src.config import PLAYER_LOGS_DB

# --- Constantes ---

# Filtro de rotacion: excluye jugadores marginales cuya ausencia no impacta
MIN_GAMES_THRESHOLD = 5       # Minimo partidos jugados para contar en rotacion
MIN_AVG_MINUTES = 10.0        # Minimo minutos promedio para contar en rotacion

# Ventana de recencia: solo contar jugadores que jugaron en los ultimos N
# partidos del equipo. Esto filtra automaticamente:
#   - Jugadores traspasados (ya no juegan para el equipo)
#   - Lesiones de largo plazo (ausencia ya reflejada en stats del equipo)
# 25 partidos ≈ 1 mes de NBA, suficiente para mantener jugadores que
# descansaron algunos juegos pero excluir los que ya no estan
RECENCY_WINDOW = 25

# Temporadas disponibles en PlayerGameLogs.sqlite
SEASONS = [
    "2012-13", "2013-14", "2014-15", "2015-16", "2016-17",
    "2017-18", "2018-19", "2019-20", "2020-21", "2021-22",
    "2022-23", "2023-24", "2024-25",
]

# Mapeo de abreviaciones de nba_api a nombres completos de OddsData/TeamData.
# Necesario porque nba_api usa "BOS", OddsData usa "Boston Celtics", etc.
NBA_ABV_TO_FULL = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "NJN": "New Jersey Nets",
    "CHA": "Charlotte Hornets",
    "CHH": "Charlotte Hornets",
    "CHO": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "LA Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NOH": "New Orleans Pelicans",
    "NOK": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}

# Mapeo inverso: nombre completo -> lista de abreviaciones posibles
FULL_TO_NBA_ABV = defaultdict(list)
for _abv, _full in NBA_ABV_TO_FULL.items():
    FULL_TO_NBA_ABV[_full].append(_abv)

# Aliases historicos (mismo que RollingAverages.py y EloRatings.py)
_TEAM_ALIASES = {
    "Los Angeles Clippers": "LA Clippers",
    "Charlotte Bobcats": "Charlotte Hornets",
    "New Orleans Hornets": "New Orleans Pelicans",
    "New Orleans/Oklahoma City Hornets": "New Orleans Pelicans",
    "New Jersey Nets": "Brooklyn Nets",
}

# Tank01 API: mapeo de nombres de equipos a abreviaciones Tank01
# (diferente de nba_api en algunos casos: SA vs SAS, NO vs NOP, etc.)
TANK01_TEAM_ABV = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GS",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NO",
    "New York Knicks": "NY",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SA",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

# Mapeo de designaciones de lesion a probabilidad de jugar
# Basado en datos historicos de DNP rates por designacion
DESIGNATION_TO_PROB = {
    "Out": 0.0,
    "Doubtful": 0.15,
    "Questionable": 0.50,
    "Day-To-Day": 0.75,
    "Probable": 0.90,
    "Available": 1.0,
    "Healthy": 1.0,
    None: 1.0,
}


def _normalize_team(name):
    """Normaliza nombre de equipo usando aliases historicos."""
    return _TEAM_ALIASES.get(name, name)


def _season_for_date(date_str):
    """Determina la temporada NBA para una fecha dada.

    La temporada NBA va de octubre a junio.
    Ejemplo: 2024-11-15 -> '2024-25', 2025-03-01 -> '2024-25'
    """
    parts = date_str.split("-")
    year = int(parts[0])
    month = int(parts[1])

    # Si es entre enero y junio, la temporada empezo el ano anterior
    if month <= 6:
        start_year = year - 1
    else:
        start_year = year

    end_year_short = str(start_year + 1)[-2:]
    return f"{start_year}-{end_year_short}"


def _compute_season_availability(df):
    """Calcula AVAIL y AVAIL_QUALITY para todos los partidos de una temporada.

    Algoritmo (sin data leakage, con ventana de recencia):
        Para cada fecha F y equipo T:
        1. Tomar los game logs de T en los ultimos RECENCY_WINDOW partidos ANTES de F
        2. Para cada jugador: avg_min = total_min / games_played (en esa ventana)
        3. Filtrar rotacion: >= 5 juegos Y >= 10 min promedio
        4. Ver quienes de la rotacion jugaron en F (MIN > 0)
        5. AVAIL = sum(avg_min de los que jugaron) / sum(avg_min de toda rotacion)
        6. AVAIL_QUALITY = AVAIL ponderado por PM36 (calidad del jugador)

    AVAIL_QUALITY pondera cada jugador por su plus-minus por 36 minutos (PM36).
    Asi, perder a Jokic (PM36=+12) reduce AVAIL_QUALITY mas que perder a un
    suplente (PM36=-2). Para que todos los pesos sean positivos, desplazamos:
        quality = PM36 + |min_PM36| + 1

    Args:
        df: DataFrame con game logs de una temporada
            Columnas: PLAYER_ID, PLAYER_NAME, TEAM_ABBREVIATION,
                      GAME_DATE, MATCHUP, MIN, PLUS_MINUS, IS_HOME

    Returns:
        dict[(date_str, full_team_name)] -> {'AVAIL': float, 'AVAIL_QUALITY': float}
    """
    avail = {}

    if df.empty:
        return avail

    # Verificar si PLUS_MINUS esta disponible en el DataFrame
    has_pm = "PLUS_MINUS" in df.columns

    # Ordenar cronologicamente para procesar en orden
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    # Obtener todas las fechas unicas (dias con partidos)
    game_dates = sorted(df["GAME_DATE"].unique())

    # Por equipo: historial de juegos como lista de (date, {pid: (min, pm)})
    # Guardamos tanto minutos como plus_minus para calcular PM36
    # team_abv -> [(date, {pid: (min, pm)}), ...]
    team_game_history = defaultdict(list)

    for date_str in game_dates:
        # Juegos de HOY
        today_mask = df["GAME_DATE"] == date_str
        today_df = df[today_mask]

        # Equipos que juegan hoy
        teams_today = today_df["TEAM_ABBREVIATION"].unique()

        for team_abv in teams_today:
            full_name = NBA_ABV_TO_FULL.get(team_abv, team_abv)
            full_name = _normalize_team(full_name)

            # Paso 1-3: Construir rotacion usando VENTANA DE RECENCIA
            # Solo los ultimos RECENCY_WINDOW partidos del equipo (NO el de hoy)
            recent_games = team_game_history[team_abv][-RECENCY_WINDOW:]

            # Acumular stats por jugador dentro de la ventana
            player_games = defaultdict(list)   # pid -> [(min, pm), ...]
            for _game_date, game_players in recent_games:
                for pid, (minutes, pm) in game_players.items():
                    player_games[pid].append((minutes, pm))

            # Filtrar rotacion: >= 5 juegos, >= 10 min promedio
            rotation = {}     # player_id -> avg_min
            player_pm36 = {}  # player_id -> PM36
            for pid, stats_list in player_games.items():
                if len(stats_list) >= MIN_GAMES_THRESHOLD:
                    total_min = sum(m for m, _ in stats_list)
                    avg_min = total_min / len(stats_list)
                    if avg_min >= MIN_AVG_MINUTES:
                        rotation[pid] = avg_min
                        # PM36 = plus_minus_total / minutos_total × 36
                        total_pm = sum(p for _, p in stats_list)
                        if total_min > 0:
                            player_pm36[pid] = (total_pm / total_min) * 36.0
                        else:
                            player_pm36[pid] = 0.0

            # Paso 4: Quienes de la rotacion jugaron HOY
            team_today = today_df[today_df["TEAM_ABBREVIATION"] == team_abv]
            played_today = set(
                team_today[team_today["MIN"] > 0]["PLAYER_ID"].tolist()
            )

            # Paso 5-6: Calcular AVAIL y AVAIL_QUALITY
            if not rotation:
                # Sin datos de rotacion (primeros partidos de temporada)
                avail[(date_str, full_name)] = {
                    "AVAIL": 1.0, "AVAIL_QUALITY": 1.0,
                    "STAR_MISSING": 0, "N_ROTATION_OUT": 0, "MISSING_BPM": 0.0,
                }
            else:
                # AVAIL clasico: ponderado solo por minutos
                total_rotation_min = sum(rotation.values())
                available_min = sum(
                    avg_m for pid, avg_m in rotation.items()
                    if pid in played_today
                )
                avail_val = available_min / total_rotation_min if total_rotation_min > 0 else 1.0

                # AVAIL_QUALITY: ponderado por calidad (PM36)
                # Shift PM36 para que todos sean positivos (el peor = 1)
                if player_pm36:
                    min_pm36 = min(player_pm36.values())
                    shift = abs(min_pm36) + 1.0 if min_pm36 < 0 else 1.0
                    quality = {pid: player_pm36[pid] + shift for pid in rotation}
                else:
                    quality = {pid: 1.0 for pid in rotation}

                total_quality = sum(quality[pid] * rotation[pid] for pid in rotation)
                available_quality = sum(
                    quality[pid] * rotation[pid]
                    for pid in rotation if pid in played_today
                )
                avail_quality = available_quality / total_quality if total_quality > 0 else 1.0

                # --- Features granulares de lesiones ---
                # Jugadores de rotacion que NO jugaron hoy
                out_players = [pid for pid in rotation if pid not in played_today]
                n_rotation_out = len(out_players)

                # Suma de PM36 de los ausentes (valor real, sin shift)
                # PM36 puede ser negativo → perder un jugador malo es "positivo"
                missing_bpm = sum(player_pm36.get(pid, 0.0) for pid in out_players)

                # STAR_MISSING: 1 si alguno de los top-2 por PM36 no jugo
                sorted_by_pm36 = sorted(
                    rotation.keys(),
                    key=lambda p: player_pm36.get(p, 0.0),
                    reverse=True,
                )
                top2_pids = set(sorted_by_pm36[:2])
                star_missing = int(bool(top2_pids - played_today))

                avail[(date_str, full_name)] = {
                    "AVAIL": avail_val,
                    "AVAIL_QUALITY": avail_quality,
                    "STAR_MISSING": star_missing,
                    "N_ROTATION_OUT": n_rotation_out,
                    "MISSING_BPM": missing_bpm,
                }

        # DESPUES de calcular AVAIL para hoy, actualizar historial
        # Esto garantiza no-leakage: el partido de hoy no se usa para calcular su propio AVAIL
        for team_abv in teams_today:
            team_today = today_df[today_df["TEAM_ABBREVIATION"] == team_abv]
            game_players = {}
            for _, row in team_today.iterrows():
                if row["MIN"] > 0:
                    pm = float(row["PLUS_MINUS"]) if has_pm else 0.0
                    game_players[row["PLAYER_ID"]] = (row["MIN"], pm)
            if game_players:
                team_game_history[team_abv].append((date_str, game_players))

    return avail


def build_availability_history(logs_db_path=None):
    """Construye lookup de disponibilidad para todo el historial.

    Procesa todas las temporadas en PlayerGameLogs.sqlite y retorna
    un dict grande con AVAIL y AVAIL_QUALITY pre-calculados.

    Args:
        logs_db_path: ruta a PlayerGameLogs.sqlite (default: Data/PlayerGameLogs.sqlite)

    Returns:
        dict[(date_str, full_team_name)] -> {'AVAIL': float, 'AVAIL_QUALITY': float}
    """
    if logs_db_path is None:
        logs_db_path = PLAYER_LOGS_DB

    all_avail = {}

    with sqlite3.connect(logs_db_path) as con:
        # Obtener tablas disponibles
        tables = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()

        for (table_name,) in tables:
            # Solo procesar tablas de game logs (player_logs_YYYY-YY)
            if not table_name.startswith("player_logs_"):
                continue
            # Extraer temporada del nombre de tabla
            season = table_name.replace("player_logs_", "")
            print(f"  AVAIL: procesando {season}...")

            df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', con)
            season_avail = _compute_season_availability(df)
            all_avail.update(season_avail)

            print(f"    {len(season_avail)} entradas (fecha, equipo)")

    # Agregar aliases para que el lookup funcione con ambas versiones del nombre
    aliases_to_add = {}
    for (date_str, team_name), val in all_avail.items():
        for old_name, new_name in _TEAM_ALIASES.items():
            if team_name == new_name and (date_str, old_name) not in all_avail:
                aliases_to_add[(date_str, old_name)] = val
            elif team_name == old_name and (date_str, new_name) not in all_avail:
                aliases_to_add[(date_str, new_name)] = val

    all_avail.update(aliases_to_add)

    return all_avail


def get_game_availability(avail_lookup, date_str, home_team, away_team):
    """Obtiene features de disponibilidad para un partido historico.

    Usado por Create_Games.py durante la construccion del dataset.
    Retorna 11 features: 4 basicas + 6 granulares + 1 derivada.

    Args:
        avail_lookup: dict de build_availability_history()
            Cada valor es {'AVAIL': float, 'AVAIL_QUALITY': float,
                           'STAR_MISSING': int, 'N_ROTATION_OUT': int,
                           'MISSING_BPM': float}
        date_str: fecha del partido 'YYYY-MM-DD'
        home_team: nombre completo del equipo local
        away_team: nombre completo del equipo visitante

    Returns:
        dict con 11 features:
          AVAIL_HOME/AWAY, AVAIL_QUALITY_HOME/AWAY,
          STAR_MISSING_HOME/AWAY, N_ROTATION_OUT_HOME/AWAY,
          MISSING_BPM_HOME/AWAY, AVAIL_DIFF
    """
    home_norm = _normalize_team(home_team)
    away_norm = _normalize_team(away_team)

    default = {
        "AVAIL": 1.0, "AVAIL_QUALITY": 1.0,
        "STAR_MISSING": 0, "N_ROTATION_OUT": 0, "MISSING_BPM": 0.0,
    }
    home_data = avail_lookup.get((date_str, home_norm), default)
    away_data = avail_lookup.get((date_str, away_norm), default)

    # Compatibilidad: si avail_lookup tiene formato viejo (float), convertir
    if isinstance(home_data, (int, float)):
        home_data = {"AVAIL": float(home_data), "AVAIL_QUALITY": 1.0,
                     "STAR_MISSING": 0, "N_ROTATION_OUT": 0, "MISSING_BPM": 0.0}
    if isinstance(away_data, (int, float)):
        away_data = {"AVAIL": float(away_data), "AVAIL_QUALITY": 1.0,
                     "STAR_MISSING": 0, "N_ROTATION_OUT": 0, "MISSING_BPM": 0.0}

    aq_home = home_data.get("AVAIL_QUALITY", 1.0)
    aq_away = away_data.get("AVAIL_QUALITY", 1.0)

    return {
        "AVAIL_HOME": home_data.get("AVAIL", 1.0),
        "AVAIL_AWAY": away_data.get("AVAIL", 1.0),
        "AVAIL_QUALITY_HOME": aq_home,
        "AVAIL_QUALITY_AWAY": aq_away,
        "STAR_MISSING_HOME": home_data.get("STAR_MISSING", 0),
        "STAR_MISSING_AWAY": away_data.get("STAR_MISSING", 0),
        "N_ROTATION_OUT_HOME": home_data.get("N_ROTATION_OUT", 0),
        "N_ROTATION_OUT_AWAY": away_data.get("N_ROTATION_OUT", 0),
        "MISSING_BPM_HOME": home_data.get("MISSING_BPM", 0.0),
        "MISSING_BPM_AWAY": away_data.get("MISSING_BPM", 0.0),
        "AVAIL_DIFF": aq_home - aq_away,
    }


def add_availability_to_frame(frame, avail_features_list):
    """Agrega columnas de disponibilidad al DataFrame.

    Mismo patron que add_elo_features_to_frame() en EloRatings.py.

    Args:
        frame: DataFrame con los juegos (ya construido)
        avail_features_list: list of dicts con AVAIL_HOME, AVAIL_AWAY,
            AVAIL_QUALITY_HOME, AVAIL_QUALITY_AWAY

    Returns:
        DataFrame con 4 columnas nuevas de disponibilidad
    """
    if not avail_features_list:
        return frame

    avail_df = pd.DataFrame(avail_features_list, index=frame.index)
    result = pd.concat([frame, avail_df], axis=1)
    return result


# --- Funciones para prediccion en vivo (main.py) ---


def get_current_rotation(logs_db_path=None, season=None, cutoff_date=None):
    """Construye la rotacion actual de cada equipo para predicciones en vivo.

    Lee los game logs de la temporada actual y calcula los minutos promedio
    y PM36 (plus-minus por 36 min) de cada jugador hasta la fecha de corte.

    Args:
        logs_db_path: ruta a PlayerGameLogs.sqlite
        season: temporada actual (ej: '2024-25')
        cutoff_date: solo usar juegos antes de esta fecha ('YYYY-MM-DD')

    Returns:
        dict[full_team_name][player_name] -> {'avg_min': float, 'pm36': float}
        Solo incluye jugadores de rotacion (>= 5 games, >= 10 min avg)
    """
    if logs_db_path is None:
        logs_db_path = PLAYER_LOGS_DB

    if season is None:
        from datetime import datetime
        season = _season_for_date(datetime.today().strftime("%Y-%m-%d"))

    table_name = f"player_logs_{season}"

    with sqlite3.connect(logs_db_path) as con:
        # Verificar que la tabla existe
        cursor = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        if cursor.fetchone() is None:
            return {}

        df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', con)

    if df.empty:
        return {}

    # Filtrar por fecha de corte si se especifica
    if cutoff_date:
        df = df[df["GAME_DATE"] < cutoff_date]

    if df.empty:
        return {}

    # Solo jugadores que realmente jugaron (MIN > 0)
    df = df[df["MIN"] > 0]

    has_pm = "PLUS_MINUS" in df.columns

    # Aplicar ventana de recencia: solo los ultimos RECENCY_WINDOW partidos por equipo
    # Esto excluye jugadores traspasados o con lesiones de largo plazo
    df = df.sort_values("GAME_DATE")
    recent_df_list = []
    for team_abv, team_df in df.groupby("TEAM_ABBREVIATION"):
        # Obtener las ultimas RECENCY_WINDOW fechas de juego unicas
        team_dates = sorted(team_df["GAME_DATE"].unique())
        recent_dates = set(team_dates[-RECENCY_WINDOW:])
        recent_df_list.append(team_df[team_df["GAME_DATE"].isin(recent_dates)])
    df = pd.concat(recent_df_list, ignore_index=True) if recent_df_list else df

    # Calcular avg_min y PM36 por jugador-equipo (solo dentro de la ventana)
    rotation = defaultdict(dict)

    for (team_abv, player_name), group in df.groupby(["TEAM_ABBREVIATION", "PLAYER_NAME"]):
        games_played = len(group)
        avg_min = group["MIN"].mean()

        # Filtro de rotacion
        if games_played >= MIN_GAMES_THRESHOLD and avg_min >= MIN_AVG_MINUTES:
            full_name = NBA_ABV_TO_FULL.get(team_abv, team_abv)
            full_name = _normalize_team(full_name)
            # PM36 = (total_pm / total_min) × 36
            total_min = group["MIN"].sum()
            total_pm = group["PLUS_MINUS"].sum() if has_pm else 0.0
            pm36 = (total_pm / total_min * 36.0) if total_min > 0 else 0.0
            rotation[full_name][player_name] = {
                "avg_min": avg_min,
                "pm36": pm36,
            }

    # Agregar aliases
    for old_name, new_name in _TEAM_ALIASES.items():
        if new_name in rotation and old_name not in rotation:
            rotation[old_name] = rotation[new_name]
        elif old_name in rotation and new_name not in rotation:
            rotation[new_name] = rotation[old_name]

    return dict(rotation)


# Cache de lesiones: {date_str -> {team_name -> {player -> designation}}}
_injuries_cache: dict = {}


def _fetch_all_injuries_today() -> dict:
    """Fetcha TODAS las lesiones del dia en una sola llamada.

    Usa nba-injury-reports.p.rapidapi.com — retorna lista plana
    [{date, team, player, status, reason, reportTime}].

    Cachea el resultado en memoria para evitar llamadas repetidas.
    Una sola llamada sirve para todos los equipos del dia.

    Returns:
        dict[full_team_name -> {player_name -> designation}]
        Retorna {} si no hay key o falla la API.
    """
    from datetime import datetime

    api_key = os.environ.get("RAPIDAPI_KEY")
    if not api_key:
        return {}

    date_str = datetime.today().strftime("%Y-%m-%d")

    if date_str in _injuries_cache:
        return _injuries_cache[date_str]

    url = f"https://nba-injuries-reports.p.rapidapi.com/injuries/nba/{date_str}"
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "nba-injuries-reports.p.rapidapi.com",
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return {}

    # Construir dict por equipo
    by_team: dict = {}
    records = data if isinstance(data, list) else data.get("body", [])
    for record in records:
        team = record.get("team", "")
        player = record.get("player", "")
        status = record.get("status", "")
        if not team or not player or not status:
            continue
        team_norm = _normalize_team(team)
        if team_norm not in by_team:
            by_team[team_norm] = {}
        if status not in ("", "Healthy", "Available"):
            by_team[team_norm][player] = status

    _injuries_cache[date_str] = by_team
    return by_team


def fetch_injury_report(team_full_name):
    """Obtiene reporte de lesiones de un equipo via nba-injury-reports (RapidAPI).

    Fetcha todas las lesiones del dia en una sola llamada (cacheada) y
    filtra por equipo. Mucho mas eficiente que el endpoint por-equipo anterior.

    La API key se lee del env var RAPIDAPI_KEY.
    Si no hay key configurada, retorna {} (fallback a AVAIL=1.0).

    Args:
        team_full_name: nombre completo (ej: 'Boston Celtics')

    Returns:
        dict[player_name -> designation_string]
        ej: {'Jayson Tatum': 'Questionable', 'Jaylen Brown': 'Out'}
        Jugadores sanos no aparecen en el dict.
    """
    all_injuries = _fetch_all_injuries_today()
    if not all_injuries:
        return {}

    team_norm = _normalize_team(team_full_name)
    return all_injuries.get(team_norm, {})


def estimate_availability(rotation, team_name, injury_report):
    """Estima AVAIL, AVAIL_QUALITY y features granulares para prediccion en vivo.

    Para cada jugador en la rotacion del equipo:
        - Si aparece en el injury_report: prob = DESIGNATION_TO_PROB[designation]
        - Si NO aparece: prob = 1.0 (asumimos sano)

    AVAIL = sum(prob_i * avg_min_i) / sum(avg_min_i)
    AVAIL_QUALITY = sum(prob_i * quality_i * avg_min_i) / sum(quality_i * avg_min_i)

    Features granulares (basadas en prob < 0.5 = "probablemente no juega"):
        STAR_MISSING: 1 si alguno de top-2 PM36 tiene prob < 0.5
        N_ROTATION_OUT: jugadores con prob < 0.5
        MISSING_BPM: suma de PM36 de jugadores con prob < 0.5

    Args:
        rotation: dict[player_name -> {'avg_min': float, 'pm36': float}]
        team_name: nombre del equipo (para lookup)
        injury_report: dict[player_name -> designation] de fetch_injury_report()

    Returns:
        dict {'AVAIL': float, 'AVAIL_QUALITY': float,
              'STAR_MISSING': int, 'N_ROTATION_OUT': int, 'MISSING_BPM': float}
    """
    team_norm = _normalize_team(team_name)
    team_rotation = rotation.get(team_norm, {})

    if not team_rotation:
        return {
            "AVAIL": 1.0, "AVAIL_QUALITY": 1.0,
            "STAR_MISSING": 0, "N_ROTATION_OUT": 0, "MISSING_BPM": 0.0,
        }

    # Compatibilidad: si rotation tiene formato viejo (float), convertir
    first_val = next(iter(team_rotation.values()))
    if isinstance(first_val, (int, float)):
        team_rotation = {k: {"avg_min": v, "pm36": 0.0} for k, v in team_rotation.items()}

    # Pre-computar lista de nombres del reporte para fuzzy matching
    injury_names = list(injury_report.keys())

    # Shift PM36 para que todos sean positivos
    pm36_values = [info["pm36"] for info in team_rotation.values()]
    min_pm36 = min(pm36_values) if pm36_values else 0.0
    shift = abs(min_pm36) + 1.0 if min_pm36 < 0 else 1.0

    total_min = 0.0
    available_min = 0.0
    total_quality = 0.0
    available_quality = 0.0

    # Guardar probabilidad por jugador para features granulares
    player_probs = {}  # player_name -> prob

    for player_name, info in team_rotation.items():
        avg_min = info["avg_min"]
        quality = (info["pm36"] + shift) * avg_min  # peso = calidad × minutos

        total_min += avg_min
        total_quality += quality

        # Buscar si este jugador esta en el reporte de lesiones
        designation = None
        if player_name in injury_report:
            designation = injury_report[player_name]
        elif injury_names:
            matches = difflib.get_close_matches(
                player_name, injury_names, n=1, cutoff=0.85
            )
            if matches:
                designation = injury_report[matches[0]]

        # Convertir designacion a probabilidad de jugar
        prob = DESIGNATION_TO_PROB.get(designation, 1.0)
        player_probs[player_name] = prob
        available_min += prob * avg_min
        available_quality += prob * quality

    avail_val = available_min / total_min if total_min > 0 else 1.0
    avail_quality = available_quality / total_quality if total_quality > 0 else 1.0

    # --- Features granulares ---
    # Umbral: prob < 0.5 = "probablemente no juega" (Out=0, Doubtful=0.15, Questionable=0.50)
    out_threshold = 0.5
    n_rotation_out = sum(1 for p in player_probs.values() if p < out_threshold)
    missing_bpm = sum(
        team_rotation[name]["pm36"]
        for name, p in player_probs.items()
        if p < out_threshold
    )

    # STAR_MISSING: top-2 por PM36 con prob < 0.5
    sorted_by_pm36 = sorted(
        team_rotation.keys(),
        key=lambda n: team_rotation[n]["pm36"],
        reverse=True,
    )
    top2 = sorted_by_pm36[:2]
    star_missing = int(any(player_probs.get(n, 1.0) < out_threshold for n in top2))

    return {
        "AVAIL": avail_val,
        "AVAIL_QUALITY": avail_quality,
        "STAR_MISSING": star_missing,
        "N_ROTATION_OUT": n_rotation_out,
        "MISSING_BPM": missing_bpm,
    }
