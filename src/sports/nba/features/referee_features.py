"""Features de arbitros (officials) para el modelo O/U de NBA.

POR QUE LOS ARBITROS IMPORTAN PARA O/U:
=========================================
Los arbitros NBA tienen tendencias documentadas que afectan directamente el total
de puntos de un partido:

1. **Ritmo de pitos**: Algunos arbitros pitan mas faltas → mas FTs → mas puntos
   → tendency a OVERS. Otros "dejan jugar" → menos interrupciones → UNDERS.

2. **Efecto simetrico**: A diferencia de stats de equipo (que favorecen a un lado),
   las tendencias del arbitro afectan a AMBOS equipos por igual. Esto lo hace
   ideal para O/U prediction (no tanto para moneyline).

3. **Cada juego tiene 3 oficiales**: El "crew effect" es el promedio de los 3.
   Un crew mixto (1 tight + 2 loose) tiende a neutral.

FEATURES (3 nuevas columnas):
==============================
- REF_CREW_TOTAL_TENDENCY: Promedio de (total_puntos - OU_line) sobre los ultimos
  82 juegos de cada ref del crew. Positivo = tiende a overs.

- REF_CREW_OVER_PCT: Promedio de % de juegos que fueron OVER para cada ref del crew.
  > 0.5 indica sesgo a overs.

- REF_CREW_HOME_WIN_PCT: Promedio de % de juegos ganados por el local para cada ref.
  Algunos crews favorecen al local (crowd influence on calls).

DEFAULTS (si arbitro tiene < 20 juegos):
- TOTAL_TENDENCY: 0.0 (neutral)
- OVER_PCT: 0.5 (50/50)
- HOME_WIN_PCT: 0.54 (league average home win rate)

VENTANA: 82 juegos rolling (1 temporada completa), cross-season, estrictamente T-1.

Sigue el patron de 3 funciones de fatigue.py:
  build_*() → lookup tables
  get_game_*() → features para 1 juego
  add_*_to_frame() → agregar columnas al DataFrame
"""

from collections import defaultdict
from datetime import datetime
from typing import Optional

import pandas as pd

from src.config import REFEREE_DB, get_logger

logger = get_logger(__name__)

# Minimo de juegos para considerar las stats de un arbitro como confiables.
# Con menos de 20 juegos, usamos defaults (league average).
MIN_GAMES_THRESHOLD = 20

# Ventana rolling: 82 juegos = ~1 temporada completa de NBA
ROLLING_WINDOW = 82

# Defaults cuando no hay datos suficientes
DEFAULT_TOTAL_TENDENCY = 0.0   # neutral
DEFAULT_OVER_PCT = 0.50        # 50/50
DEFAULT_HOME_WIN_PCT = 0.54    # historical NBA home win rate

# Aliases de equipos para matching con OddsData
_TEAM_ALIASES = {
    "Los Angeles Clippers": "LA Clippers",
    "Charlotte Bobcats": "Charlotte Hornets",
    "New Orleans Hornets": "New Orleans Pelicans",
    "New Orleans/Oklahoma City Hornets": "New Orleans Pelicans",
    "New Jersey Nets": "Brooklyn Nets",
}


def _normalize_team(name: str) -> str:
    """Normaliza nombre de equipo."""
    return _TEAM_ALIASES.get(name, name)


def build_referee_history(referee_db_path=None):
    """Construye historial de arbitros desde RefereeData.sqlite.

    Lee todas las temporadas disponibles y construye dos estructuras:

    1. ref_assignments: dict[(date_str, home, away)] → [ref1_id, ref2_id, ref3_id]
       Permite buscar los arbitros de un juego especifico.

    2. ref_history: dict[ref_id] → sorted list[(game_date, total_points, ou_line, home_win)]
       Historial completo de cada arbitro para calcular rolling stats.

    Args:
        referee_db_path: ruta a RefereeData.sqlite (default: config.REFEREE_DB)

    Returns:
        tuple (ref_assignments, ref_history)
    """
    import sqlite3

    db_path = referee_db_path or REFEREE_DB

    if not db_path.exists():
        logger.warning(
            "RefereeData.sqlite no existe en %s. "
            "Ejecutar: PYTHONPATH=. python scripts/collect_referee_data.py --all",
            db_path,
        )
        return {}, {}

    con = sqlite3.connect(db_path)

    # Encontrar todas las tablas de temporada
    tables = [
        r[0]
        for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'referee_games_%'"
        ).fetchall()
    ]

    if not tables:
        logger.warning("No hay tablas referee_games_* en %s", db_path)
        con.close()
        return {}, {}

    ref_assignments = {}
    ref_history = defaultdict(list)

    total_games = 0

    for table in sorted(tables):
        try:
            rows = con.execute(f'SELECT * FROM "{table}"').fetchall()
            cols = [desc[0] for desc in con.execute(f'SELECT * FROM "{table}" LIMIT 0').description]
        except Exception as e:
            logger.warning("Error leyendo %s: %s", table, e)
            continue

        col_idx = {c: i for i, c in enumerate(cols)}

        for row in rows:
            game_date = row[col_idx["GAME_DATE"]]
            home = _normalize_team(row[col_idx["HOME_TEAM"]])
            away = _normalize_team(row[col_idx["AWAY_TEAM"]])
            total_pts = row[col_idx.get("TOTAL_POINTS", -1)] if "TOTAL_POINTS" in col_idx else None
            ou_line = row[col_idx.get("OU_LINE", -1)] if "OU_LINE" in col_idx else None
            home_win = row[col_idx.get("HOME_WIN", -1)] if "HOME_WIN" in col_idx else None

            # Extraer IDs de los 3 arbitros
            ref_ids = []
            for i in range(1, 4):
                rid = row[col_idx.get(f"REF{i}_ID", -1)] if f"REF{i}_ID" in col_idx else None
                if rid is not None and rid != "":
                    ref_ids.append(int(rid))

            if not ref_ids:
                continue

            # Registrar asignacion
            key = (game_date, home, away)
            ref_assignments[key] = ref_ids

            # Agregar al historial de cada arbitro
            for rid in ref_ids:
                ref_history[rid].append((game_date, total_pts, ou_line, home_win))

            total_games += 1

    con.close()

    # Ordenar historial de cada arbitro por fecha
    for rid in ref_history:
        ref_history[rid].sort(key=lambda x: x[0])

    logger.info(
        "Referee history: %d juegos, %d arbitros unicos, %d temporadas",
        total_games, len(ref_history), len(tables),
    )

    return ref_assignments, dict(ref_history)


def _compute_ref_stats(
    ref_games: list,
    before_date: str,
    window: int = ROLLING_WINDOW,
) -> Optional[dict]:
    """Computa stats rolling para un arbitro antes de una fecha dada.

    Filtra estrictamente juegos ANTES de `before_date` (T-1), toma los
    ultimos `window` juegos, y calcula:
    - total_tendency: mean(total_points - ou_line)
    - over_pct: count(total > ou) / N
    - home_win_pct: count(home_win) / N

    Returns:
        dict con las 3 metricas, o None si < MIN_GAMES_THRESHOLD juegos
    """
    # Filtrar juegos estrictamente antes de la fecha
    eligible = [g for g in ref_games if g[0] < before_date]

    # Tomar los ultimos `window` juegos
    recent = eligible[-window:]

    if len(recent) < MIN_GAMES_THRESHOLD:
        return None

    total_diffs = []
    overs = 0
    home_wins = 0
    n_valid_ou = 0
    n_valid_hw = 0

    for _, total_pts, ou_line, home_win in recent:
        if total_pts is not None and ou_line is not None:
            try:
                diff = float(total_pts) - float(ou_line)
                total_diffs.append(diff)
                if diff > 0:
                    overs += 1
                n_valid_ou += 1
            except (ValueError, TypeError):
                pass

        if home_win is not None:
            try:
                if int(home_win) == 1:
                    home_wins += 1
                n_valid_hw += 1
            except (ValueError, TypeError):
                pass

    if n_valid_ou < MIN_GAMES_THRESHOLD:
        return None

    return {
        "total_tendency": sum(total_diffs) / len(total_diffs),
        "over_pct": overs / n_valid_ou,
        "home_win_pct": home_wins / n_valid_hw if n_valid_hw > 0 else DEFAULT_HOME_WIN_PCT,
    }


def get_game_referee_features(
    ref_assignments: dict,
    ref_history: dict,
    date_str: str,
    home_team: str,
    away_team: str,
) -> dict:
    """Calcula features de arbitros para un juego especifico.

    Busca los 3 arbitros asignados al juego, calcula sus stats rolling
    (estrictamente T-1), y promedia los 3 para obtener el "crew effect".

    Si no hay arbitros asignados o no hay datos suficientes, retorna defaults.

    Args:
        ref_assignments: dict[(date, home, away)] → [ref_ids]
        ref_history: dict[ref_id] → sorted list[(date, total, ou, hw)]
        date_str: fecha del juego "YYYY-MM-DD"
        home_team: nombre del equipo local
        away_team: nombre del equipo visitante

    Returns:
        dict con 3 features:
        - REF_CREW_TOTAL_TENDENCY
        - REF_CREW_OVER_PCT
        - REF_CREW_HOME_WIN_PCT
    """
    defaults = {
        "REF_CREW_TOTAL_TENDENCY": DEFAULT_TOTAL_TENDENCY,
        "REF_CREW_OVER_PCT": DEFAULT_OVER_PCT,
        "REF_CREW_HOME_WIN_PCT": DEFAULT_HOME_WIN_PCT,
    }

    # Buscar asignacion
    home_norm = _normalize_team(home_team)
    away_norm = _normalize_team(away_team)
    key = (date_str, home_norm, away_norm)

    ref_ids = ref_assignments.get(key)
    if not ref_ids:
        return defaults

    # Computar stats para cada arbitro del crew
    tendencies = []
    over_pcts = []
    hw_pcts = []

    for rid in ref_ids:
        games = ref_history.get(rid)
        if not games:
            continue

        stats = _compute_ref_stats(games, date_str)
        if stats is None:
            # Arbitro sin datos suficientes → usar defaults
            tendencies.append(DEFAULT_TOTAL_TENDENCY)
            over_pcts.append(DEFAULT_OVER_PCT)
            hw_pcts.append(DEFAULT_HOME_WIN_PCT)
        else:
            tendencies.append(stats["total_tendency"])
            over_pcts.append(stats["over_pct"])
            hw_pcts.append(stats["home_win_pct"])

    if not tendencies:
        return defaults

    # Promediar los 3 (o 2, si uno falta) arbitros del crew
    return {
        "REF_CREW_TOTAL_TENDENCY": round(sum(tendencies) / len(tendencies), 4),
        "REF_CREW_OVER_PCT": round(sum(over_pcts) / len(over_pcts), 4),
        "REF_CREW_HOME_WIN_PCT": round(sum(hw_pcts) / len(hw_pcts), 4),
    }


def add_referee_features_to_frame(
    frame: pd.DataFrame,
    referee_features_list: list[dict],
) -> pd.DataFrame:
    """Agrega columnas de arbitros al DataFrame.

    Sigue el mismo patron de add_fatigue_to_frame(): recibe el DataFrame
    y una lista de dicts (uno por juego, en el mismo orden) y agrega 3 columnas.

    Args:
        frame: DataFrame con los juegos
        referee_features_list: lista de dicts retornados por get_game_referee_features()

    Returns:
        DataFrame con 3 columnas nuevas: REF_CREW_TOTAL_TENDENCY,
        REF_CREW_OVER_PCT, REF_CREW_HOME_WIN_PCT
    """
    if not referee_features_list:
        logger.warning("Lista de referee features vacia. Usando defaults.")
        frame["REF_CREW_TOTAL_TENDENCY"] = DEFAULT_TOTAL_TENDENCY
        frame["REF_CREW_OVER_PCT"] = DEFAULT_OVER_PCT
        frame["REF_CREW_HOME_WIN_PCT"] = DEFAULT_HOME_WIN_PCT
        return frame

    if len(referee_features_list) != len(frame):
        logger.error(
            "Mismatch: %d referee features vs %d filas en frame",
            len(referee_features_list), len(frame),
        )
        frame["REF_CREW_TOTAL_TENDENCY"] = DEFAULT_TOTAL_TENDENCY
        frame["REF_CREW_OVER_PCT"] = DEFAULT_OVER_PCT
        frame["REF_CREW_HOME_WIN_PCT"] = DEFAULT_HOME_WIN_PCT
        return frame

    ref_df = pd.DataFrame(referee_features_list)
    for col in ["REF_CREW_TOTAL_TENDENCY", "REF_CREW_OVER_PCT", "REF_CREW_HOME_WIN_PCT"]:
        frame[col] = ref_df[col].values

    logger.info(
        "Referee features agregadas: %d juegos, "
        "tendency mean=%.3f, over_pct mean=%.3f, hw_pct mean=%.3f",
        len(frame),
        frame["REF_CREW_TOTAL_TENDENCY"].mean(),
        frame["REF_CREW_OVER_PCT"].mean(),
        frame["REF_CREW_HOME_WIN_PCT"].mean(),
    )

    return frame
