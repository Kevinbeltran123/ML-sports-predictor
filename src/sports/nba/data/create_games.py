import bisect
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import toml

from src.sports.nba.features.advanced_stats import add_advanced_features
from src.sports.nba.features.differential_features import add_differential_features
from src.sports.nba.features.style_features import add_style_features
from src.core.stats.elo_ratings import (
    add_elo_features_to_frame,
    build_elo_history,
    build_srs_history,
    get_game_srs_features,
    add_srs_features_to_frame,
)
from src.sports.nba.features.fatigue import (
    add_fatigue_to_frame,
    add_travel_to_frame,
    add_extended_fatigue_to_frame,
    add_fatigue_combo_to_frame,
    build_team_schedule,
    build_team_travel_schedule,
    get_game_fatigue,
    get_game_travel,
    get_game_extended_fatigue,
)
from src.sports.nba.features.conference_division import (
    add_conference_division_to_frame,
    get_game_conference_division,
)
from src.sports.nba.features.injury_impact import (
    add_availability_to_frame,
    build_availability_history,
    get_game_availability,
)
from src.sports.nba.features.home_away_splits import (
    add_split_features_to_frame,
    build_season_split_data,
    get_team_split_features,
)
from src.core.stats.rolling_averages import (
    add_rolling_features_to_frame,
    build_season_game_logs,
    get_team_rolling_features,
)
from src.sports.nba.features.sos import (
    add_sos_to_frame,
    build_sos_lookup,
    get_game_sos,
)
from src.sports.nba.features.lineup_strength import (
    build_lineup_strength_history,
    get_game_lineup_features,
    add_lineup_features_to_frame,
)
from src.sports.nba.features.referee_features import (
    build_referee_history,
    get_game_referee_features,
    add_referee_features_to_frame,
)
from src.sports.nba.features.bref_game_features import (
    build_four_factors_history,
    get_game_four_factors,
    add_four_factors_to_frame,
)
from src.sports.nba.features.zone_shooting_features import (
    build_zone_shooting_lookup,
    get_game_zone_shooting,
    add_zone_shooting_to_frame,
)
from src.sports.nba.features.shot_chart_features import (
    build_shot_chart_history,
    get_game_shot_chart,
    add_shot_chart_to_frame,
)
from src.sports.nba.features.onoff_features import (
    build_onoff_history,
    get_game_onoff,
    add_onoff_to_frame,
)
from src.sports.nba.features.line_scores_features import (
    build_line_scores_history,
    get_game_line_scores,
    add_line_scores_to_frame,
)
from src.sports.nba.features.player_advanced_features import (
    build_player_advanced_history,
    get_game_player_advanced,
    add_player_advanced_to_frame,
)
from src.sports.nba.features.odds_features import compute_vig_magnitude
from src.sports.nba.features.espn_lines_features import (
    build_espn_consensus_history,
    build_espn_book_disagreement_history,
    get_game_espn_features,
    add_espn_features_to_frame,
)
from src.sports.nba.features.lineup_features import (
    build_lineup_history,
    get_game_lineup_composition_features,
    add_lineup_composition_to_frame,
)
from src.sports.nba.features.dictionaries import (
    team_index_07,
    team_index_08,
    team_index_12,
    team_index_current,
)

from src.config import (
    CONFIG_PATH,
    ODDS_DB as ODDS_DB_PATH,
    TEAMS_DB as TEAMS_DB_PATH,
    DATASET_DB as OUTPUT_DB_PATH,
    PLAYER_LOGS_DB,
    get_logger,
)

logger = get_logger(__name__)
OUTPUT_TABLE = "dataset_2012-26"

TEAM_INDEX_BY_SEASON = {
    "2007-08": team_index_07,
    "2008-09": team_index_08,
    "2009-10": team_index_08,
    "2010-11": team_index_08,
    "2011-12": team_index_08,
    "2012-13": team_index_12,
    "2013-14": team_index_12,
    "2014-15": team_index_12,
    "2015-16": team_index_12,
    "2016-17": team_index_12,
    "2017-18": team_index_12,
    "2018-19": team_index_12,
    "2019-20": team_index_12,
    "2020-21": team_index_12,
    "2021-22": team_index_12,
    "2022-23": team_index_current,
    "2023-24": team_index_current,
    "2024-25": team_index_current,
    "2025-26": team_index_current,
}


def american_odds_to_prob(ml_home, ml_away):
    """Convierte odds americanos a probabilidad implicita del local (sin vig).

    Odds americanos:
        Negativo (favorito): -250 significa apostar $250 para ganar $100
        Positivo (underdog): +205 significa apostar $100 para ganar $205

    Conversion a probabilidad implicita:
        Si odds < 0: prob = |odds| / (|odds| + 100)
        Si odds > 0: prob = 100 / (odds + 100)

    La suma de ambas prob > 1 por el vig (margen de la casa).
    Removemos el vig dividiendo cada una por la suma total.
    """
    def _implied(odds):
        odds = float(odds)
        if odds < 0:
            return abs(odds) / (abs(odds) + 100.0)
        else:
            return 100.0 / (odds + 100.0)

    p_home = _implied(ml_home)
    p_away = _implied(ml_away)
    total = p_home + p_away  # > 1 por el vig
    return p_home / total     # probabilidad limpia (sin vig)


def table_exists(con, table_name):
    cursor = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    return cursor.fetchone() is not None


def normalize_date(value):
    if isinstance(value, datetime):
        return value.date().isoformat()
    if hasattr(value, "date"):
        try:
            return value.date().isoformat()
        except Exception:
            pass
    return str(value)


def get_team_index_map(season_key):
    if season_key in TEAM_INDEX_BY_SEASON:
        return TEAM_INDEX_BY_SEASON[season_key]
    try:
        start_year = int(season_key.split("-")[0])
    except (ValueError, IndexError):
        return team_index_current
    return team_index_current if start_year >= 2022 else team_index_12


def fetch_team_table(teams_con, date_str):
    if not table_exists(teams_con, date_str):
        return None
    return pd.read_sql_query(f'SELECT * FROM "{date_str}"', teams_con)


def get_team_table_dates(teams_con):
    cursor = teams_con.execute("SELECT name FROM sqlite_master WHERE type='table'")
    dates = []
    for (name,) in cursor.fetchall():
        try:
            dates.append(datetime.strptime(name, "%Y-%m-%d").date())
        except ValueError:
            continue
    return sorted(set(dates))


def fetch_team_table_before(teams_con, game_date_str, table_dates, cache):
    """Retorna snapshot T-1 estricto (ultima fecha < game_date)."""
    try:
        game_date = datetime.strptime(str(game_date_str)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None, None

    idx = bisect.bisect_left(table_dates, game_date) - 1
    if idx < 0:
        return None, None

    snapshot_date = table_dates[idx].isoformat()
    if snapshot_date not in cache:
        if not table_exists(teams_con, snapshot_date):
            return None, None
        cache[snapshot_date] = pd.read_sql_query(
            f'SELECT * FROM "{snapshot_date}"',
            teams_con,
        )
    return cache[snapshot_date], snapshot_date


def build_game_features(team_df, home_team, away_team, index_map):
    home_index = index_map.get(home_team)
    away_index = index_map.get(away_team)
    if home_index is None or away_index is None:
        return None
    if len(team_df.index) != 30:
        return None

    home_team_series = team_df.iloc[home_index]
    away_team_series = team_df.iloc[away_index]
    return pd.concat([
        home_team_series,
        away_team_series.rename(index={col: f"{col}.1" for col in team_df.columns.values}),
    ])


def select_odds_table(odds_con, season_key):
    candidates = [
        f"odds_{season_key}_new",
        f"odds_{season_key}",
        f"{season_key}_new",
        f"{season_key}",
    ]
    for table_name in candidates:
        if table_exists(odds_con, table_name):
            return table_name
    return None


def main():
    config = toml.load(CONFIG_PATH)

    scores = []
    win_margin = []
    margin_continuous = []  # Margen continuo: home_score - away_score (target para regresor)
    ou_values = []
    ou_cover = []
    spread_values = []
    ml_home_values = []
    ml_away_values = []
    games = []
    days_rest_away = []
    days_rest_home = []
    home_rolling_list = []
    away_rolling_list = []
    elo_features_list = []
    home_split_list = []
    away_split_list = []
    avail_features_list = []
    fatigue_features_list = []
    travel_features_list = []
    sos_features_list = []
    extended_fatigue_list = []
    srs_features_list = []
    lineup_features_list = []
    referee_features_list = []
    ff_features_list = []
    zone_features_list = []
    sc_features_list = []
    onoff_features_list = []
    ls_features_list = []
    adv_features_list = []
    espn_features_list = []
    lineup_comp_features_list = []
    conf_div_features_list = []
    skipped_no_snapshot = 0
    skipped_zero_labels = 0
    snapshot_audit = []

    with sqlite3.connect(ODDS_DB_PATH) as odds_con, sqlite3.connect(TEAMS_DB_PATH) as teams_con:
        team_table_dates = get_team_table_dates(teams_con)
        team_table_cache = {}
        # Pre-construir historial completo de Elo (desde 2007-08)
        # Procesa TODOS los partidos cronologicamente para que los ratings
        # esten estabilizados cuando empezamos el dataset en 2012-13
        elo_lookup, _ = build_elo_history(odds_con)
        logger.info("Elo: %d partidos procesados", len(elo_lookup))

        # Pre-construir historial de disponibilidad (desde 2012-13)
        # Usa PlayerGameLogs.sqlite para saber quien jugo en cada partido
        avail_lookup = build_availability_history(PLAYER_LOGS_DB)
        logger.info("Availability: %d entradas (fecha, equipo)", len(avail_lookup))

        # Pre-construir calendario de cada equipo para features de fatiga
        # THREE_IN_FOUR y TWO_IN_THREE capturan densidad del calendario
        team_schedule = build_team_schedule(odds_con)
        logger.info("Fatigue: %d equipos con calendario", len(team_schedule))

        # Pre-construir calendario con ubicación para features de viaje
        # TRAVEL_DIST y TZ_CHANGE capturan fatiga por viaje (Fase 5.3)
        travel_schedule = build_team_travel_schedule(odds_con)
        logger.info("Travel: %d equipos con ubicación", len(travel_schedule))

        # Pre-construir lookup de SOS (Strength of Schedule)
        # SOS_W_PCT_10: W_PCT promedio de últimos 10 oponentes (Fase 5.4)
        sos_lookup = build_sos_lookup(odds_con)
        logger.info("SOS: %d partidos con lookup", len(sos_lookup))

        # Pre-construir SRS (Simple Rating System): MOV ajustado por SOS
        # SRS = MOV + SOS resuelto iterativamente. Superior a W_PCT simple
        # porque un equipo que gana por 10 contra buenos equipos > ganar por 10 contra malos
        srs_lookup = build_srs_history(odds_con)
        logger.info("SRS: %d entradas (fecha, equipo)", len(srs_lookup))

        # Pre-construir lineup strength desde PlayerGameLogs
        # TOP5_PM36, DEPTH_SCORE, STAR_POWER, HHI_MINUTES por equipo
        # Solo temporadas 2022+ (logs anteriores no tienen PLUS_MINUS)
        lineup_lookup = build_lineup_strength_history()
        logger.info("Lineup: %d entradas (fecha, equipo)", len(lineup_lookup))

        # Pre-construir historial de arbitros para features O/U
        # REF_CREW_TOTAL_TENDENCY, OVER_PCT, HOME_WIN_PCT
        ref_assignments, ref_history = build_referee_history()
        logger.info("Referee: %d juegos, %d arbitros", len(ref_assignments), len(ref_history))

        # Pre-construir Four Factors rolling (BRefData: pace, ORtg, eFG, TOV%, ORB%)
        ff_lookup = build_four_factors_history()
        logger.info("Four Factors: %d entradas (fecha, equipo)", len(ff_lookup))

        # Pre-construir zone shooting profiles (BRefData: season aggregate, T-1 via prev season)
        zone_lookup = build_zone_shooting_lookup()
        logger.info("Zone Shooting: %d entradas (season, equipo)", len(zone_lookup))

        # Pre-construir shot chart history (BRefData: 2.6M tiros, rolling 10 juegos)
        sc_lookup = build_shot_chart_history()
        logger.info("Shot Chart: %d entradas (fecha, equipo)", len(sc_lookup))

        # Pre-construir on/off court plus/minus (BRefData: rolling 10 juegos)
        onoff_lookup = build_onoff_history()
        logger.info("On/Off: %d entradas (fecha, equipo)", len(onoff_lookup))

        # Pre-construir line scores history (BRefData: quarter scoring patterns, rolling 10)
        ls_lookup = build_line_scores_history()
        logger.info("Line Scores: %d entradas (fecha, equipo)", len(ls_lookup))

        # Pre-construir player advanced history (BRefData: BPM, TS%, USG concentration, rolling 10)
        adv_lookup = build_player_advanced_history()
        logger.info("Player Advanced: %d entradas (fecha, equipo)", len(adv_lookup))

        # Pre-construir ESPN Lines lookup (apertura/cierre de lineas ESPN, 2022-2026)
        # COBERTURA: ~24% de partidos; 76% NaN esperado. XGBoost maneja NaN nativamente.
        espn_consensus_lookup = build_espn_consensus_history()
        espn_disagree_lookup = build_espn_book_disagreement_history()
        logger.info(
            "ESPN Lines: %d entradas consenso, %d entradas book disagreement",
            len(espn_consensus_lookup), len(espn_disagree_lookup),
        )

        # Pre-construir lineup composition features (BRefData player_basic + GMM archetypes)
        # LINEUP_DIVERSITY, LINEUP_STAR_FRAC, BENCH_PPG_GAP, BENCH_DEPTH por equipo (rolling 10 T-1)
        lineup_comp_lookup = build_lineup_history()
        logger.info("Lineup Composition: %d entradas (fecha, equipo)", len(lineup_comp_lookup))

        for season_key in config["create-games"].keys():
            logger.info("Processing season: %s", season_key)
            odds_table = select_odds_table(odds_con, season_key)
            if not odds_table:
                logger.warning("Missing odds tables for %s.", season_key)
                continue

            odds_df = pd.read_sql_query(f'SELECT * FROM "{odds_table}"', odds_con)
            if odds_df.empty:
                logger.warning("No odds data for %s.", season_key)
                continue
            odds_df["Date"] = pd.to_datetime(odds_df["Date"], errors="coerce")
            odds_df = odds_df.dropna(subset=["Date"]).sort_values(
                ["Date", "Home", "Away"]
            ).reset_index(drop=True)

            index_map = get_team_index_map(season_key)

            # Pre-construir game logs para calcular rolling averages
            season_cfg = config["create-games"][season_key]
            game_logs = build_season_game_logs(
                teams_con, season_cfg["start_date"], season_cfg["end_date"]
            )
            logger.debug("Rolling: %d teams tracked", len(game_logs))

            # Pre-construir splits home/away para esta temporada
            split_data = build_season_split_data(
                teams_con, odds_con, season_key,
                season_cfg["start_date"], season_cfg["end_date"]
            )
            logger.debug("Splits: %d teams tracked", len(split_data))

            for row in odds_df.itertuples(index=False):
                date_str = normalize_date(row.Date)
                team_df, snapshot_date = fetch_team_table_before(
                    teams_con=teams_con,
                    game_date_str=date_str,
                    table_dates=team_table_dates,
                    cache=team_table_cache,
                )
                if team_df is None:
                    skipped_no_snapshot += 1
                    continue
                snapshot_audit.append((date_str, snapshot_date))

                # Gate de calidad de labels: excluir juegos sin score final.
                try:
                    points = float(row.Points)
                    margin = float(row.Win_Margin)
                except (TypeError, ValueError):
                    skipped_zero_labels += 1
                    continue
                if points == 0.0 and margin == 0.0:
                    skipped_zero_labels += 1
                    continue

                game = build_game_features(team_df, row.Home, row.Away, index_map)
                if game is None:
                    continue

                scores.append(points)
                ou_values.append(row.OU)
                # "PK" (pick'em) significa spread = 0 (sin favorito)
                raw_spread = row.Spread
                spread_values.append(0.0 if raw_spread == "PK" else float(raw_spread))
                ml_home_values.append(row.ML_Home)
                ml_away_values.append(row.ML_Away)
                days_rest_home.append(row.Days_Rest_Home)
                days_rest_away.append(row.Days_Rest_Away)
                win_margin.append(1 if margin > 0 else 0)
                margin_continuous.append(margin)

                if row.Points < row.OU:
                    ou_cover.append(0)
                elif row.Points > row.OU:
                    ou_cover.append(1)
                else:
                    ou_cover.append(2)

                # Rolling features: promedios ultimas 5/10 partidas + momentum
                home_gp = float(game["GP"])
                away_gp = float(game["GP.1"])
                home_roll = get_team_rolling_features(
                    game_logs.get(row.Home, []), home_gp
                )
                away_roll = get_team_rolling_features(
                    game_logs.get(row.Away, []), away_gp
                )
                home_rolling_list.append(home_roll)
                away_rolling_list.append(away_roll)

                # Elo features: buscar el rating pre-partido en el historial
                elo_key = (date_str, row.Home, row.Away)
                elo_feats = elo_lookup.get(elo_key, {
                    "ELO_HOME": 1500.0, "ELO_AWAY": 1500.0,
                    "ELO_DIFF": 100.0, "ELO_PROB": 0.6,
                })
                elo_features_list.append(elo_feats)

                # Split features: diferencia rendimiento home vs away por equipo
                home_split = get_team_split_features(split_data, row.Home, home_gp)
                away_split = get_team_split_features(split_data, row.Away, away_gp)
                home_split_list.append(home_split)
                away_split_list.append(away_split)

                # Availability features: fraccion de minutos de rotacion disponibles
                avail_feats = get_game_availability(
                    avail_lookup, date_str, row.Home, row.Away
                )
                avail_features_list.append(avail_feats)

                # Fatigue features: densidad del calendario (3-en-4, 2-en-3 dias)
                fatigue_feats = get_game_fatigue(
                    team_schedule, date_str, row.Home, row.Away
                )
                fatigue_features_list.append(fatigue_feats)

                # Travel features: distancia de viaje + cambio de timezone (Fase 5.3)
                travel_feats = get_game_travel(
                    travel_schedule, date_str, row.Home, row.Away
                )
                travel_features_list.append(travel_feats)

                # SOS features: calidad de oponentes recientes (Fase 5.4)
                sos_feats = get_game_sos(sos_lookup, date_str, row.Home, row.Away)
                sos_features_list.append(sos_feats)

                # Extended fatigue: TZ signed, altitude, schedule density (Fase 5.4b)
                ext_fatigue = get_game_extended_fatigue(
                    team_schedule, travel_schedule, date_str, row.Home, row.Away
                )
                extended_fatigue_list.append(ext_fatigue)

                # Conference / Division: rivalidad y familiaridad
                conf_div_features_list.append(
                    get_game_conference_division(row.Home, row.Away)
                )

                # SRS features: rating ajustado por calidad de oponentes (Fase 5.5)
                srs_feats = get_game_srs_features(
                    srs_lookup, date_str, row.Home, row.Away
                )
                srs_features_list.append(srs_feats)

                # Lineup strength: agregacion de stats de jugadores (Fase 5.6)
                # Solo disponible para temporadas 2022+ (logs con PLUS_MINUS)
                lineup_feats = get_game_lineup_features(
                    lineup_lookup, date_str, row.Home, row.Away
                )
                lineup_features_list.append(lineup_feats)

                # Referee features: tendencia del crew a overs/unders (O/U)
                ref_feats = get_game_referee_features(
                    ref_assignments, ref_history, date_str, row.Home, row.Away
                )
                referee_features_list.append(ref_feats)

                # Four Factors: pace, ORtg, eFG%, TOV%, ORB%, FT/FGA (rolling 10, BRef)
                ff_feats = get_game_four_factors(ff_lookup, date_str, row.Home, row.Away)
                ff_features_list.append(ff_feats)

                # Zone Shooting: perfil espacial de tiro por equipo (season agg T-1)
                zone_feats = get_game_zone_shooting(
                    zone_lookup, season_key, date_str, row.Home, row.Away
                )
                zone_features_list.append(zone_feats)

                # Shot Chart: distribución de tiros por zona (rolling 10, BRef)
                sc_feats = get_game_shot_chart(sc_lookup, date_str, row.Home, row.Away)
                sc_features_list.append(sc_feats)

                # On/Off Court: impacto neto top-5 jugadores (rolling 10, BRef)
                onoff_feats = get_game_onoff(onoff_lookup, date_str, row.Home, row.Away)
                onoff_features_list.append(onoff_feats)

                # Line Scores: quarter scoring patterns (clutch, consistency)
                ls_feats = get_game_line_scores(ls_lookup, date_str, row.Home, row.Away)
                ls_features_list.append(ls_feats)

                # Player Advanced: BPM, TS%, USG concentration (BRefData)
                adv_feats = get_game_player_advanced(adv_lookup, date_str, row.Home, row.Away)
                adv_features_list.append(adv_feats)

                # ESPN Lines: movimiento de lineas y desacuerdo entre casas
                # ESPN_LINE_MOVE, ESPN_TOTAL_MOVE, ESPN_OPEN_ML_PROB, ESPN_BOOK_DISAGREEMENT
                # Solo pre-tipoff features (_open_ y spread_movement=close-open)
                espn_feats = get_game_espn_features(
                    espn_consensus_lookup, espn_disagree_lookup,
                    date_str, row.Home, row.Away,
                )
                espn_features_list.append(espn_feats)

                # Lineup composition: archetype diversity + bench depth (BRefData)
                # LINEUP_DIVERSITY, LINEUP_STAR_FRAC, BENCH_PPG_GAP, BENCH_DEPTH
                lineup_comp_feats = get_game_lineup_composition_features(
                    lineup_comp_lookup, date_str, row.Home, row.Away,
                )
                lineup_comp_features_list.append(lineup_comp_feats)

                games.append(game)

    if not games:
        logger.error("No game rows produced. Check odds and team tables.")
        return

    if snapshot_audit:
        violations = sum(1 for game_d, snap_d in snapshot_audit if snap_d >= game_d)
        if violations > 0:
            # Log detalles de las violaciones para debugging
            bad_games = [(g, s) for g, s in snapshot_audit if s >= g]
            for game_d, snap_d in bad_games[:5]:
                logger.warning("Snapshot leakage: game=%s snapshot=%s", game_d, snap_d)
            logger.warning(
                "Snapshot audit: %d/%d juegos con snapshot >= game_date (data leakage risk!)",
                violations, len(snapshot_audit),
            )
            if violations > len(snapshot_audit) * 0.01:  # >1% violaciones
                logger.error(
                    "Too many snapshot violations (%d/%d = %.1f%%). "
                    "Check team table dates for temporal leakage.",
                    violations, len(snapshot_audit),
                    violations / len(snapshot_audit) * 100,
                )
        else:
            logger.info(
                "Snapshot audit: %d juegos | 0 violaciones T-1 ✓",
                len(snapshot_audit),
            )
    logger.info(
        "Filtrado dataset: sin snapshot T-1=%d | labels sospechosos= %d",
        skipped_no_snapshot,
        skipped_zero_labels,
    )

    season = pd.concat(games, ignore_index=True, axis=1).T
    frame = season.drop(columns=["TEAM_ID", "TEAM_ID.1"], errors="ignore")
    frame["Score"] = np.asarray(scores)
    frame["Home-Team-Win"] = np.asarray(win_margin)
    frame["Margin"] = np.asarray(margin_continuous, dtype=float)
    frame["OU"] = np.asarray(ou_values)
    frame["OU-Cover"] = np.asarray(ou_cover)
    frame["Days-Rest-Home"] = np.asarray(days_rest_home)
    frame["Days-Rest-Away"] = np.asarray(days_rest_away)

    # Market features: lineas de apertura del mercado (spread + probabilidad implicita)
    # Clip spread a [-25, 25]: valores fuera de rango son errores de datos
    frame["MARKET_SPREAD"] = np.clip(np.asarray(spread_values, dtype=float), -25.0, 25.0)
    ml_probs = [
        american_odds_to_prob(h, a)
        for h, a in zip(ml_home_values, ml_away_values)
    ]
    frame["MARKET_ML_PROB"] = np.asarray(ml_probs)

    # VIG_MAGNITUDE: overround del bookmaker (suma de prob implicitas - 1.0)
    # Proxy de certeza del mercado: vig alto = partido incierto o book agresivo.
    # T-1 safe: misma fuente que MARKET_ML_PROB (OddsData FanDuel closing lines).
    vig_magnitudes = [
        compute_vig_magnitude(h, a)
        for h, a in zip(ml_home_values, ml_away_values)
    ]
    frame["VIG_MAGNITUDE"] = np.asarray(vig_magnitudes)

    for field in frame.columns.values:
        if "TEAM_" in field or "Date" in field:
            continue
        frame[field] = frame[field].astype(float)

    # Agregar estadisticas avanzadas y features de matchup
    frame = add_advanced_features(frame)

    # Agregar features de estilo de juego (ratios derivados del box score)
    # FG3A_RATE, AST_RATIO, PACE_ADJ_DEF + diferenciales + STYLE_CLASH
    frame = add_style_features(frame)

    # Agregar features diferenciales (home - away) para stats base
    # Paper Ouyang (2024): DIFF_FG_PCT, DIFF_DRB, DIFF_TOV son top SHAP features
    frame = add_differential_features(frame)

    # Agregar rolling averages (ultimas 5/10 partidas) y momentum
    frame = add_rolling_features_to_frame(frame, home_rolling_list, away_rolling_list)

    # Agregar ratings Elo (fuerza dinamica ajustada por calidad del rival)
    frame = add_elo_features_to_frame(frame, elo_features_list)

    # Agregar splits home/away (diferencia rendimiento casa vs visita)
    frame = add_split_features_to_frame(frame, home_split_list, away_split_list)

    # Agregar disponibilidad de rotacion (fraccion de minutos disponibles)
    frame = add_availability_to_frame(frame, avail_features_list)

    # Agregar fatiga por densidad de calendario (3-en-4 y 2-en-3 dias)
    frame = add_fatigue_to_frame(frame, fatigue_features_list)

    # Agregar distancia de viaje y cambio de timezone (Fase 5.3)
    frame = add_travel_to_frame(frame, travel_features_list)

    # Agregar SOS: calidad de oponentes recientes (Fase 5.4)
    frame = add_sos_to_frame(frame, sos_features_list)

    # Agregar fatiga extendida: TZ signed, altitud, densidad (Fase 5.4b)
    frame = add_extended_fatigue_to_frame(frame, extended_fatigue_list)

    # Agregar interaction features: B2B × viaje, fatigue index
    frame = add_fatigue_combo_to_frame(frame)

    # Agregar conference / division rivalry features
    frame = add_conference_division_to_frame(frame, conf_div_features_list)

    # Agregar SRS: Simple Rating System (MOV + SOS iterativo) (Fase 5.5)
    frame = add_srs_features_to_frame(frame, srs_features_list)

    # Agregar lineup strength: agregacion de jugadores (Fase 5.6)
    # TOP5_PM36, DEPTH_SCORE, STAR_POWER, HHI_MINUTES por equipo
    frame = add_lineup_features_to_frame(frame, lineup_features_list)

    # Agregar referee features: tendencia del crew a overs/unders
    # REF_CREW_TOTAL_TENDENCY, OVER_PCT, HOME_WIN_PCT (para O/U)
    frame = add_referee_features_to_frame(frame, referee_features_list)

    # Agregar Four Factors: pace, ORtg, eFG%, TOV%, ORB%, FT/FGA (BRefData rolling 10)
    frame = add_four_factors_to_frame(frame, ff_features_list)

    # Agregar Zone Shooting: perfil espacial de tiro por equipo (BRefData season agg)
    frame = add_zone_shooting_to_frame(frame, zone_features_list)

    # Agregar Shot Chart: distribución de tiros por zona (BRefData rolling 10)
    frame = add_shot_chart_to_frame(frame, sc_features_list)

    # Agregar On/Off Court: impacto neto top-5 y spread (BRefData rolling 10)
    frame = add_onoff_to_frame(frame, onoff_features_list)

    # Agregar Line Scores: quarter scoring patterns (clutch, consistency, BRefData)
    frame = add_line_scores_to_frame(frame, ls_features_list)

    # Agregar Player Advanced: BPM, TS%, USG concentration (BRefData)
    frame = add_player_advanced_to_frame(frame, adv_features_list)

    # Agregar ESPN Lines features: movimiento de lineas y desacuerdo entre casas
    # Cobertura ~24%: 76% NaN esperado. XGBoost maneja NaN via missingness pathways.
    frame = add_espn_features_to_frame(frame, espn_features_list)

    # Agregar Lineup Composition: archetype diversity + bench depth (BRefData, Phase 5)
    # LINEUP_DIVERSITY, LINEUP_STAR_FRAC, BENCH_PPG_GAP, BENCH_DEPTH (x2 home/away)
    frame = add_lineup_composition_to_frame(frame, lineup_comp_features_list)

    # Eliminar duplicados (EDA encontró 2: Lakers 2020-12-27, Suns 2026-02-09)
    n_before = len(frame)
    frame = frame.drop_duplicates(subset=["Date", "TEAM_NAME"], keep="first").reset_index(drop=True)
    n_dupes = n_before - len(frame)
    if n_dupes > 0:
        logger.info("Eliminados %d duplicados (Date + TEAM_NAME)", n_dupes)

    logger.info("Dataset final: %d juegos x %d columnas", frame.shape[0], frame.shape[1])

    with sqlite3.connect(OUTPUT_DB_PATH) as con:
        frame.to_sql(OUTPUT_TABLE, con, if_exists="replace", index=False)


if __name__ == "__main__":
    main()
