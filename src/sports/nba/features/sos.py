"""Strength of Schedule (SOS): calidad de oponentes recientes.

¿Por qué importa?
    Un equipo con ROLL_10_PTS = 115 contra defensas top-5 (DRtg < 108)
    es MUCHO mejor que uno con 115 contra defensas bottom-5 (DRtg > 116).
    Sin SOS, el modelo los trata igual.

¿Qué calculamos?
    SOS_W_PCT_10: W_PCT promedio de los últimos 10 oponentes.
        - SOS = 0.600 → enfrentó equipos ganadores → schedule difícil
        - SOS = 0.400 → enfrentó equipos perdedores → schedule fácil

    Liga promedio = 0.500 (por definición, las victorias suman a derrotas).
    Rango típico: 0.40 - 0.60 para ventanas de 10 juegos.

¿Cómo se construye?
    1. Desde OddsData, rastrear resultados cronológicamente por temporada
    2. Mantener un registro running de W/L por equipo
    3. Para cada partido, buscar los últimos 10 oponentes del equipo
    4. Promediar la W_PCT actual de esos 10 oponentes → SOS

Usado por Create_Games.py (entrenamiento) y main.py (predicción).
"""

from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

# Aliases históricos
_TEAM_ALIASES = {
    "Los Angeles Clippers": "LA Clippers",
    "Charlotte Bobcats": "Charlotte Hornets",
    "New Orleans Hornets": "New Orleans Pelicans",
    "New Orleans/Oklahoma City Hornets": "New Orleans Pelicans",
    "New Jersey Nets": "Brooklyn Nets",
}


def _normalize_team(name):
    return _TEAM_ALIASES.get(name, name)


def build_sos_lookup(odds_con):
    """Construye lookup de SOS para todos los partidos en OddsData.

    Procesa TODOS los partidos cronológicamente, manteniendo:
        - Registro W/L por equipo (running W_PCT)
        - Lista de oponentes por equipo (para buscar últimos 10)

    Para cada partido, calcula SOS_10 ANTES del resultado (sin leakage).

    Args:
        odds_con: conexión SQLite a OddsData.sqlite

    Returns:
        dict[(date_str, home, away)] -> {
            "SOS_W_PCT_10_HOME": float,
            "SOS_W_PCT_10_AWAY": float,
        }
    """
    # Obtener todas las tablas de odds
    cursor = odds_con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = [row[0] for row in cursor.fetchall()]

    # Recolectar TODOS los partidos de todas las temporadas
    all_games = []

    for table_name in tables:
        if not table_name.startswith("odds_") and "-" not in table_name:
            continue

        try:
            rows = odds_con.execute(
                f'SELECT Date, Home, Away, Win_Margin FROM "{table_name}"'
            ).fetchall()
        except Exception:
            continue

        # Extraer el season_key del nombre de tabla
        # "odds_2022-23" → "2022-23"
        season_key = table_name.replace("odds_", "").replace("_new", "")

        for row in rows:
            date_val = row[0]
            home = _normalize_team(row[1])
            away = _normalize_team(row[2])
            win_margin = row[3]

            # Normalizar fecha
            if isinstance(date_val, str):
                try:
                    dt = datetime.strptime(date_val, "%Y-%m-%d").date()
                except ValueError:
                    try:
                        dt = datetime.strptime(date_val[:10], "%Y-%m-%d").date()
                    except ValueError:
                        continue
            elif hasattr(date_val, "date"):
                dt = date_val.date() if callable(getattr(date_val, "date")) else date_val
            else:
                continue

            try:
                wm = float(win_margin)
            except (ValueError, TypeError):
                continue

            all_games.append((dt, season_key, home, away, wm))

    # Ordenar cronológicamente
    all_games.sort(key=lambda x: x[0])

    # Procesar todos los partidos, manteniendo estado por temporada
    # Cada temporada tiene su propio registro W/L (se resetea)
    sos_lookup = {}

    # Estado por temporada
    current_season = None
    team_wins = defaultdict(int)     # team -> wins acumulados
    team_losses = defaultdict(int)   # team -> losses acumulados
    team_opponents = defaultdict(list)  # team -> [(date, opponent), ...] cronológico

    for dt, season_key, home, away, win_margin in all_games:
        # ¿Nueva temporada? → Resetear contadores
        if season_key != current_season:
            current_season = season_key
            team_wins.clear()
            team_losses.clear()
            team_opponents.clear()

        # ANTES del partido: calcular SOS para ambos equipos
        # (sin leakage — usamos datos ANTERIORES a este partido)
        sos_home = _compute_sos(team_opponents, team_wins, team_losses, home)
        sos_away = _compute_sos(team_opponents, team_wins, team_losses, away)

        date_str = dt.isoformat()
        sos_lookup[(date_str, home, away)] = {
            "SOS_W_PCT_10_HOME": sos_home,
            "SOS_W_PCT_10_AWAY": sos_away,
        }

        # DESPUÉS del partido: actualizar registros
        if win_margin > 0:
            # Home ganó
            team_wins[home] += 1
            team_losses[away] += 1
        else:
            # Away ganó (o empate, raro en NBA)
            team_losses[home] += 1
            team_wins[away] += 1

        # Registrar oponentes
        team_opponents[home].append((dt, away))
        team_opponents[away].append((dt, home))

    return sos_lookup


def _compute_sos(team_opponents, team_wins, team_losses, team, window=10):
    """Calcula SOS_W_PCT_10 para un equipo.

    Busca los últimos `window` oponentes y promedia su W_PCT actual.

    Returns:
        float: W_PCT promedio de los últimos 10 oponentes (0.0-1.0)
               0.5 si no hay suficientes datos
    """
    opponents_list = team_opponents.get(team, [])
    if len(opponents_list) < 3:
        return 0.5  # Default: schedule promedio

    # Últimos N oponentes (más recientes al final)
    recent = opponents_list[-window:]

    w_pcts = []
    for _, opp in recent:
        w = team_wins.get(opp, 0)
        l = team_losses.get(opp, 0)
        total = w + l
        if total >= 5:  # Al menos 5 juegos para W_PCT significativo
            w_pcts.append(w / total)

    if not w_pcts:
        return 0.5

    return round(float(np.mean(w_pcts)), 4)


def get_game_sos(sos_lookup, game_date, home_team, away_team):
    """Obtiene features de SOS para un partido.

    Args:
        sos_lookup: de build_sos_lookup()
        game_date: str 'YYYY-MM-DD'
        home_team: nombre del equipo local
        away_team: nombre del equipo visitante

    Returns:
        dict con SOS_W_PCT_10_HOME y SOS_W_PCT_10_AWAY
    """
    home_norm = _normalize_team(home_team)
    away_norm = _normalize_team(away_team)

    result = sos_lookup.get((game_date, home_norm, away_norm))
    if result:
        return result

    # Intentar con nombres originales si normalizados no funcionan
    result = sos_lookup.get((game_date, home_team, away_team))
    if result:
        return result

    return {"SOS_W_PCT_10_HOME": 0.5, "SOS_W_PCT_10_AWAY": 0.5}


def add_sos_to_frame(frame, sos_features_list):
    """Agrega columnas de SOS al DataFrame de juegos.

    Args:
        frame: DataFrame con los juegos
        sos_features_list: list of dicts de get_game_sos()

    Returns:
        DataFrame con 2 columnas nuevas de SOS
    """
    if not sos_features_list:
        return frame

    sos_df = pd.DataFrame(sos_features_list, index=frame.index)
    return pd.concat([frame, sos_df], axis=1)
