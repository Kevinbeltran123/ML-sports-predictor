"""Features de composicion de lineup y profundidad de banca.

Derivadas de BRefData player_basic (is_starter, gmsc, pts) y archetypes GMM.

LINEUP_DIVERSITY: entropy de distribucion de archetypes en starting 5.
  - Alto (>1.5) = lineup balanceado con jugadores de estilos diversos
  - Bajo (<0.5) = lineup especializado (todos similar estilo)

LINEUP_STAR_FRAC: fraccion maxima de starters en un solo archetype.
  - Alto (>0.6) = equipo con estilo dominante (ej. 3 shooters de 5)
  - Bajo (<0.3) = variedad de roles

BENCH_PPG_GAP: diferencia promedio Game Score starters vs bench.
  - Positivo = starters mucho mejores que bench (roster top-heavy)
  - Cerca de 0 = profundidad equilibrada

BENCH_DEPTH: numero de bench players con >= 10 PTS en el juego.
  - Alto = banco productivo con multiples opciones
  - Bajo = banco limitado

Todos son rolling 10 con T-1 (shift) para evitar leakage.
"""

import sqlite3
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import joblib

from src.config import ARCHETYPES_MODEL_PATH, get_logger
from src.sports.nba.features.team_mappings import TEAM_NAME_TO_BREF
from src.sports.nba.features.player_name_lookup import build_player_name_lookup

logger = get_logger(__name__)

BREF_DB = Path("data/BRefData.sqlite")
PLAYER_LOGS_DB = Path("data/nba/training/PlayerGameLogs.sqlite")
SEASONS = [f"{y}-{str(y+1)[-2:]}" for y in range(2014, 2026)]
# Temporadas con stats completas en PlayerGameLogs (PTS, AST, etc.)
SEASONS_WITH_STATS = [f"{y}-{str(y+1)[-2:]}" for y in range(2022, 2026)]


def _compute_entropy(probs: list[float]) -> float:
    """Shannon entropy de distribucion de probabilidades."""
    probs = [p for p in probs if p > 0]
    if not probs:
        return 0.0
    return -sum(p * math.log2(p) for p in probs)


def _build_archetype_cache() -> dict[int, int]:
    """Construir cache {player_id -> archetype_label} usando el GMM de produccion.

    Replica la derivacion de style features de player_archetypes.py:
    1. Cargar PlayerGameLogs por temporada (solo 2022+ tienen stats)
    2. Agregar por jugador-temporada (mean de cada stat)
    3. Derivar las 9 style features: FG3A_RATE, PTS_PER36, AST_PER36, etc.
    4. scaler.transform() + gmm.predict() para hard labels

    Returns:
        dict: {player_id -> archetype_label (0-7)}
    """
    try:
        artifact = joblib.load(ARCHETYPES_MODEL_PATH)
        gmm = artifact["gmm"]
        scaler = artifact["scaler"]
        feature_names = artifact["feature_names"]
    except Exception as e:
        logger.warning("  No se pudo cargar GMM artifact: %s", e)
        return {}

    eps = 1e-8
    cache = {}

    try:
        logs_con = sqlite3.connect(str(PLAYER_LOGS_DB))
        for season in SEASONS_WITH_STATS:
            try:
                logs_df = pd.read_sql(
                    f'SELECT PLAYER_ID, TEAM_ABBREVIATION, GAME_DATE, MIN, PTS, AST, REB, '
                    f'BLK, STL, FGM, FGA, FG3M, FG3A, FTM, FTA '
                    f'FROM "player_logs_{season}" WHERE MIN > 5',
                    logs_con,
                )
                if logs_df.empty:
                    continue

                # Agregar por jugador-temporada
                agg = logs_df.groupby("PLAYER_ID").agg({
                    "PTS": "mean", "AST": "mean", "REB": "mean",
                    "BLK": "mean", "STL": "mean",
                    "FGM": "mean", "FGA": "mean",
                    "FG3M": "mean", "FG3A": "mean",
                    "FTM": "mean", "FTA": "mean",
                    "MIN": "mean",
                })

                # Calcular team FGA para PLAYER_FGA_SHARE
                team_fga = (
                    logs_df.groupby(["TEAM_ABBREVIATION", "GAME_DATE"])["FGA"]
                    .sum().reset_index()
                )
                team_fga_avg = team_fga.groupby("TEAM_ABBREVIATION")["FGA"].mean()
                player_teams = logs_df.groupby("PLAYER_ID")["TEAM_ABBREVIATION"].first()
                team_fga_for_player = (
                    player_teams.map(team_fga_avg).fillna(80).reindex(agg.index).fillna(80)
                )

                min_safe = agg["MIN"].clip(lower=1)
                fga_safe = agg["FGA"] + eps

                # Derivar las 9 style features (misma logica que player_archetypes.py)
                profiles = pd.DataFrame(index=agg.index)
                profiles["FG3A_RATE_10"] = agg["FG3A"] / fga_safe
                profiles["FT_RATE_10"] = agg["FTA"] / fga_safe
                profiles["PTS_PER36"] = agg["PTS"] / min_safe * 36
                profiles["AST_PER36"] = agg["AST"] / min_safe * 36
                profiles["REB_PER36"] = agg["REB"] / min_safe * 36
                profiles["BLK_STL_PER36"] = (agg["BLK"] + agg["STL"]) / min_safe * 36
                profiles["AST_FGA_RATIO"] = agg["AST"] / fga_safe
                profiles["TS_PCT_10"] = agg["PTS"] / (
                    2 * (agg["FGA"] + 0.44 * agg["FTA"]) + eps
                )
                profiles["PLAYER_FGA_SHARE_10"] = agg["FGA"] / team_fga_for_player

                # Dropear NaN y asignar archetype
                valid = profiles.dropna()
                if valid.empty:
                    continue

                X_scaled = scaler.transform(valid[feature_names])
                labels = gmm.predict(X_scaled)

                for pid, label in zip(valid.index, labels):
                    cache[int(pid)] = int(label)

            except Exception as e:
                logger.debug("  Season %s error en archetype cache: %s", season, e)
                continue

        logs_con.close()
    except Exception as e:
        logger.warning("  No se pudo construir archetype cache: %s", e)

    logger.info("  Archetype cache: %d jugadores con label asignado", len(cache))
    return cache


def build_lineup_history() -> dict:
    """Construir lookup de features de lineup por (date_str, bref_code).

    Para cada juego en BRefData player_basic:
    1. Separar starters vs bench
    2. Calcular archetype distribution de starters (via GMM cache)
    3. Calcular bench depth features (gmsc, pts)
    4. Aplicar rolling 10 T-1

    Returns:
        dict keyed by (date_str, bref_code) -> {
            "LINEUP_DIVERSITY": float,
            "LINEUP_STAR_FRAC": float,
            "BENCH_PPG_GAP": float,
            "BENCH_DEPTH": float,
        }
    """
    # Cargar dependencias
    name_lookup = build_player_name_lookup()
    archetype_cache = _build_archetype_cache()

    if not archetype_cache:
        logger.warning("  Archetype cache vacio. Lineup features no disponibles.")
        return {}

    # Determinar el archetype mas comun como fallback para jugadores desconocidos
    from collections import Counter
    arch_counts = Counter(archetype_cache.values())
    most_common_arch = arch_counts.most_common(1)[0][0] if arch_counts else 0

    bref_con = sqlite3.connect(str(BREF_DB))
    all_game_features = {}  # {(date_str, team_code) -> raw features dict}

    for season in SEASONS:
        try:
            df = pd.read_sql(
                f'SELECT game_id, game_date, team_name, '
                f'player_name, is_starter, pts, gmsc '
                f'FROM "player_basic_{season}" '
                f'WHERE dnp = 0',
                bref_con,
            )
        except Exception:
            continue

        if df.empty:
            continue

        # Limpiar: gmsc puede ser NULL
        df["gmsc"] = pd.to_numeric(df["gmsc"], errors="coerce").fillna(0.0)
        df["pts"] = pd.to_numeric(df["pts"], errors="coerce").fillna(0.0)

        for (game_id, team_name), group in df.groupby(["game_id", "team_name"]):
            game_date = str(group["game_date"].iloc[0])

            starters = group[group["is_starter"] == 1]
            bench = group[group["is_starter"] == 0]

            if len(starters) < 3:
                continue

            # --- Archetype distribution de starters usando GMM real ---
            archetype_counts = np.zeros(8)
            for _, player in starters.iterrows():
                pid = name_lookup.get((player["player_name"], season))
                if pid is not None and pid in archetype_cache:
                    arch_label = archetype_cache[pid]
                    archetype_counts[arch_label] += 1
                else:
                    # Jugador desconocido: asignar al archetype mas comun
                    archetype_counts[most_common_arch] += 1

            # Normalizar a fracciones
            total = archetype_counts.sum()
            if total > 0:
                archetype_fracs = archetype_counts / total
            else:
                archetype_fracs = np.ones(8) / 8

            diversity = _compute_entropy(archetype_fracs.tolist())
            star_frac = float(archetype_fracs.max())

            # --- Bench depth features ---
            starter_gmsc = float(starters["gmsc"].mean()) if not starters.empty else 0.0
            bench_gmsc = float(bench["gmsc"].mean()) if not bench.empty and len(bench) > 0 else 0.0
            ppg_gap = starter_gmsc - bench_gmsc

            bench_depth = int((bench["pts"] >= 10).sum()) if not bench.empty else 0

            key = (game_date, team_name)
            all_game_features[key] = {
                "LINEUP_DIVERSITY": diversity,
                "LINEUP_STAR_FRAC": star_frac,
                "BENCH_PPG_GAP": ppg_gap,
                "BENCH_DEPTH": bench_depth,
            }

    bref_con.close()

    # --- Convertir a rolling 10 con shift(1) para T-1 ---
    team_games = defaultdict(list)
    for (date_str, team), feats in sorted(all_game_features.items()):
        team_games[team].append((date_str, feats))

    rolling_lookup = {}
    for team, games in team_games.items():
        games.sort(key=lambda x: x[0])
        for i, (date_str, _) in enumerate(games):
            # T-1: usar juegos 0..i-1 (excluir el actual)
            window = games[max(0, i - 10):i]
            if not window:
                continue

            avg_feats = {}
            for key in ["LINEUP_DIVERSITY", "LINEUP_STAR_FRAC", "BENCH_PPG_GAP", "BENCH_DEPTH"]:
                vals = [g[1][key] for g in window]
                avg_feats[key] = float(np.mean(vals))

            rolling_lookup[(date_str, team)] = avg_feats

    logger.info("  Lineup composition features: %d entradas construidas", len(rolling_lookup))
    return rolling_lookup


def get_game_lineup_composition_features(
    lookup: dict,
    game_date: str,
    home_team: str,
    away_team: str,
) -> dict:
    """Obtener features de lineup composition para un juego especifico.

    Convierte nombres de equipo (ej. "Boston Celtics") a codigos BRef (ej. "BOS")
    y busca en el lookup pre-construido.

    Args:
        lookup: dict de build_lineup_history()
        game_date: fecha ISO del juego
        home_team: nombre completo del equipo local
        away_team: nombre completo del equipo visitante

    Returns:
        dict con 8 features: {feat}_HOME y {feat}_AWAY
    """
    home_bref = TEAM_NAME_TO_BREF.get(home_team, home_team)
    away_bref = TEAM_NAME_TO_BREF.get(away_team, away_team)

    home_feats = lookup.get((game_date, home_bref), {})
    away_feats = lookup.get((game_date, away_bref), {})

    result = {}
    for key in ["LINEUP_DIVERSITY", "LINEUP_STAR_FRAC", "BENCH_PPG_GAP", "BENCH_DEPTH"]:
        result[f"{key}_HOME"] = home_feats.get(key, float("nan"))
        result[f"{key}_AWAY"] = away_feats.get(key, float("nan"))

    return result


def add_lineup_composition_to_frame(
    df: pd.DataFrame,
    features_list: list[dict],
) -> pd.DataFrame:
    """Concatenar features de lineup composition al DataFrame."""
    if not features_list:
        return df
    feat_df = pd.DataFrame(features_list)
    return pd.concat(
        [df.reset_index(drop=True), feat_df.reset_index(drop=True)],
        axis=1,
    )
