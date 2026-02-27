import argparse
import sqlite3
from datetime import datetime, timedelta

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
import pandas as pd
import toml
from colorama import Fore, Style

from src.sports.nba.providers.odds_api import OddsApiProvider, BOOKMAKER_MAP
from src.sports.nba.predict import xgboost_runner as XGBoost_Runner, ensemble_runner as Ensemble_Runner
from src.sports.nba.features.advanced_stats import add_advanced_features
from src.sports.nba.features.style_features import add_style_features
from src.sports.nba.features.differential_features import add_differential_features
from src.sports.nba.features.dictionaries import team_index_current
from src.core.stats.elo_ratings import (
    add_elo_features_to_frame,
    add_srs_features_to_frame,
    build_srs_history,
    get_current_elos,
    get_game_elo_features,
    get_game_srs_features,
)
from src.sports.nba.features.home_away_splits import (
    add_split_features_to_frame,
    build_season_split_data,
    get_team_split_features,
)
from src.sports.nba.features.fatigue import (
    add_fatigue_to_frame,
    add_travel_to_frame,
    add_extended_fatigue_to_frame,
    build_team_schedule,
    build_team_travel_schedule,
    get_game_fatigue,
    get_game_travel,
    get_game_extended_fatigue,
)
from src.sports.nba.features.sos import (
    add_sos_to_frame,
    build_sos_lookup,
    get_game_sos,
)
from src.sports.nba.features.injury_impact import (
    add_availability_to_frame,
    estimate_availability,
    fetch_injury_report,
    get_current_rotation,
)
from src.core.stats.rolling_averages import (
    add_rolling_features_to_frame,
    build_season_game_logs,
    get_team_rolling_features,
)
from src.sports.nba.features.lineup_strength import (
    add_lineup_features_to_frame,
    build_lineup_strength_history,
    get_game_lineup_features,
)
from src.sports.nba.features.referee_features import (
    build_referee_history,
    get_game_referee_features,
    add_referee_features_to_frame,
)
from src.core.tools import (
    create_todays_games_from_odds,
    get_json_data,
    to_data_frame,
    get_todays_games_json,
    create_todays_games,
)
from src.config import (
    CONFIG_PATH,
    TEAMS_DB as TEAMS_DB_PATH,
    ODDS_DB as ODDS_DB_PATH,
    PLAYER_LOGS_DB,
    PROJECT_ROOT,
    SCHEDULES_DIR,
    DROP_COLUMNS_ML,
    get_logger,
)

logger = get_logger(__name__)

TODAYS_GAMES_URL = "https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/2025/scores/00_todays_scores.json"
DATA_URL = "https://stats.nba.com/stats/leaguedashteamstats?Conference=&DateFrom=&DateTo=&Division=&GameScope=&GameSegment=&Height=&ISTRound=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2025-26&SeasonSegment=&SeasonType=Regular%20Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision="

SCHEDULE_PATH = SCHEDULES_DIR / "nba-2025-UTC.csv"

# Columnas a eliminar en prediccion: debe coincidir EXACTAMENTE con
# DROP_COLUMNS_ML del entrenamiento + metadata de prediccion (TEAM_ID).
PREDICTION_DROP = DROP_COLUMNS_ML + [
    'TEAM_ID', 'TEAM_ID.1',
]


def american_odds_to_prob(ml_home, ml_away):
    """Convierte odds americanos a probabilidad implicita del local (sin vig)."""
    def _implied(odds):
        odds = float(odds)
        if odds < 0:
            return abs(odds) / (abs(odds) + 100.0)
        else:
            return 100.0 / (odds + 100.0)

    p_home = _implied(ml_home)
    p_away = _implied(ml_away)
    total = p_home + p_away
    return p_home / total


def build_current_game_logs():
    """Construye game logs de la temporada actual desde TeamData.sqlite."""
    config = toml.load(CONFIG_PATH)
    today = datetime.today().date()
    for season_key, value in config["get-data"].items():
        start_date = datetime.strptime(value["start_date"], "%Y-%m-%d").date()
        end_date = datetime.strptime(value["end_date"], "%Y-%m-%d").date()
        if start_date <= today <= end_date:
            with sqlite3.connect(TEAMS_DB_PATH) as con:
                return build_season_game_logs(con, value["start_date"], value["end_date"])
    return {}


def build_current_split_data():
    """Construye splits home/away de la temporada actual."""
    config = toml.load(CONFIG_PATH)
    today = datetime.today().date()
    for season_key, value in config["get-data"].items():
        start_date = datetime.strptime(value["start_date"], "%Y-%m-%d").date()
        end_date = datetime.strptime(value["end_date"], "%Y-%m-%d").date()
        if start_date <= today <= end_date:
            with sqlite3.connect(TEAMS_DB_PATH) as tcon, \
                 sqlite3.connect(ODDS_DB_PATH) as ocon:
                return build_season_split_data(
                    tcon, ocon, season_key,
                    value["start_date"], value["end_date"]
                )
    return {}


def build_current_availability(games):
    """Construye indice de disponibilidad para los partidos de hoy.

    Si no hay RAPIDAPI_KEY, retorna AVAIL=1.0 para todos (fallback seguro).
    """
    config = toml.load(CONFIG_PATH)
    today = datetime.today().date()
    current_season = None
    for sk, value in config["get-data"].items():
        start_date = datetime.strptime(value["start_date"], "%Y-%m-%d").date()
        end_date = datetime.strptime(value["end_date"], "%Y-%m-%d").date()
        if start_date <= today <= end_date:
            current_season = sk
            break

    if current_season is None:
        return {}

    rotation = get_current_rotation(PLAYER_LOGS_DB, current_season, today.isoformat())

    all_teams = set()
    for home, away in games:
        all_teams.add(home)
        all_teams.add(away)

    availability = {}
    for team in all_teams:
        try:
            injury_report = fetch_injury_report(team)
            avail_val = estimate_availability(rotation, team, injury_report)
            availability[team] = avail_val
            if injury_report:
                logger.info("%s: AVAIL=%.2f Q=%.2f (%d lesionados)",
                            team, avail_val['AVAIL'], avail_val['AVAIL_QUALITY'], len(injury_report))
            else:
                logger.info("%s: AVAIL=%.2f Q=%.2f (sin reporte)",
                            team, avail_val['AVAIL'], avail_val['AVAIL_QUALITY'])
        except Exception as e:
            availability[team] = {"AVAIL": 1.0, "AVAIL_QUALITY": 1.0}
            logger.warning("%s: AVAIL=1.00 (error: %s)", team, e)

    return availability


def _inject_current_srs(srs_lookup, games, today_str):
    """Inyecta SRS actual de cada equipo para los partidos de hoy."""
    latest_srs = {}
    for (date_str, home, away), feats in srs_lookup.items():
        if home not in latest_srs or date_str > latest_srs[home][0]:
            latest_srs[home] = (date_str, feats["SRS_HOME"])
        if away not in latest_srs or date_str > latest_srs[away][0]:
            latest_srs[away] = (date_str, feats["SRS_AWAY"])

    for home, away in games:
        h_srs = latest_srs.get(home, ("", 0.0))[1]
        a_srs = latest_srs.get(away, ("", 0.0))[1]
        srs_lookup[(today_str, home, away)] = {
            "SRS_HOME": h_srs,
            "SRS_AWAY": a_srs,
            "SRS_DIFF": round(h_srs - a_srs, 3),
        }


def _inject_current_lineup(lineup_lookup, games, today_str):
    """Inyecta lineup strength actual para los partidos de hoy."""
    from src.sports.nba.features.lineup_strength import _TEAM_NAME_TO_ABBR

    latest_lineup = {}
    for (date_str, team_abbr), feats in lineup_lookup.items():
        if team_abbr not in latest_lineup or date_str > latest_lineup[team_abbr][0]:
            latest_lineup[team_abbr] = (date_str, feats)

    for home, away in games:
        for team_name in [home, away]:
            abbr = _TEAM_NAME_TO_ABBR.get(team_name)
            if abbr and abbr in latest_lineup:
                lineup_lookup[(today_str, abbr)] = latest_lineup[abbr][1]


def _calculate_days_rest(schedule_df, team, today):
    """Dias desde el ultimo partido del equipo. Default 7 si no hay datos."""
    team_games = schedule_df[
        (schedule_df['Home Team'] == team) | (schedule_df['Away Team'] == team)
    ]
    prev = team_games.loc[
        team_games['Date'] <= today
    ].sort_values('Date', ascending=False).head(1)['Date']
    if len(prev) > 0:
        return (timedelta(days=1) + today - prev.iloc[0]).days
    return 7


def _build_game_base_row(df, home_team, away_team, home_rest, away_rest):
    """Combina stats home + away (.1 suffix) + dias de descanso en una Serie."""
    home_series = df.iloc[team_index_current[home_team]]
    away_series = df.iloc[team_index_current[away_team]]
    away_renamed = away_series.rename(
        index={col: f"{col}.1" for col in df.columns.values}
    )
    stats = pd.concat([home_series, away_renamed])
    stats['Days-Rest-Home'] = home_rest
    stats['Days-Rest-Away'] = away_rest
    return stats


def _add_market_features(games_df, spread_values, home_odds, away_odds):
    """Agrega MARKET_SPREAD y MARKET_ML_PROB al DataFrame."""
    games_df["MARKET_SPREAD"] = np.clip(
        np.asarray(spread_values, dtype=float), -25.0, 25.0
    )
    ml_probs = [
        american_odds_to_prob(h, a)
        for h, a in zip(home_odds, away_odds)
    ]
    games_df["MARKET_ML_PROB"] = np.asarray(ml_probs)
    return games_df


def _prepare_prediction_matrix(games_df):
    """Elimina metadata + features redundantes y convierte a float array."""
    market_info = {}
    if "MARKET_SPREAD" in games_df.columns:
        market_info["MARKET_SPREAD"] = games_df["MARKET_SPREAD"].values.copy()
    if "MARKET_ML_PROB" in games_df.columns:
        market_info["MARKET_ML_PROB"] = games_df["MARKET_ML_PROB"].values.copy()

    frame_ml = games_df.drop(columns=PREDICTION_DROP, errors='ignore')
    return frame_ml.values.astype(float), frame_ml, market_info


def create_todays_games_data(games, df, odds, schedule_df, today, game_logs,
                             elo_ratings, split_data, team_availability,
                             team_schedule, travel_schedule=None,
                             sos_lookup=None, srs_lookup=None,
                             lineup_lookup=None,
                             ref_assignments=None, ref_history=None):
    match_data = []
    todays_games_uo = []
    home_team_odds = []
    away_team_odds = []
    spread_values = []
    spread_home_odds = []
    spread_away_odds = []
    home_rolling_list, away_rolling_list = [], []
    elo_features_list = []
    home_split_list, away_split_list = [], []
    avail_features_list = []
    fatigue_features_list = []
    travel_features_list = []
    sos_features_list = []
    extended_fatigue_list = []
    srs_features_list = []
    lineup_features_list = []
    referee_features_list = []

    today_str = today.strftime("%Y-%m-%d")

    for home_team, away_team in games:
        if home_team not in team_index_current or away_team not in team_index_current:
            continue

        if odds:
            game_key = f"{home_team}:{away_team}"
            game_odds = odds[game_key]
            todays_games_uo.append(game_odds['under_over_odds'])
            home_team_odds.append(game_odds[home_team]['money_line_odds'])
            away_team_odds.append(game_odds[away_team]['money_line_odds'])
            spread_values.append(game_odds.get('spread', 0.0))
            spread_home_odds.append(game_odds.get('spread_home_odds', -110))
            spread_away_odds.append(game_odds.get('spread_away_odds', -110))
        else:
            todays_games_uo.append(input(home_team + ' vs ' + away_team + ' (O/U): '))
            home_team_odds.append(input(home_team + ' ML odds: '))
            away_team_odds.append(input(away_team + ' ML odds: '))
            spread_values.append(float(input(home_team + ' spread: ')))
            spread_home_odds.append(-110)
            spread_away_odds.append(-110)

        home_rest = _calculate_days_rest(schedule_df, home_team, today)
        away_rest = _calculate_days_rest(schedule_df, away_team, today)
        stats = _build_game_base_row(df, home_team, away_team, home_rest, away_rest)

        home_gp = float(df.iloc[team_index_current[home_team]]['GP'])
        away_gp = float(df.iloc[team_index_current[away_team]]['GP'])

        home_rolling_list.append(get_team_rolling_features(game_logs.get(home_team, []), home_gp))
        away_rolling_list.append(get_team_rolling_features(game_logs.get(away_team, []), away_gp))
        elo_features_list.append(get_game_elo_features(elo_ratings, home_team, away_team))
        home_split_list.append(get_team_split_features(split_data, home_team, home_gp))
        away_split_list.append(get_team_split_features(split_data, away_team, away_gp))

        default_avail = {
            "AVAIL": 1.0, "AVAIL_QUALITY": 1.0,
            "STAR_MISSING": 0, "N_ROTATION_OUT": 0, "MISSING_BPM": 0.0,
        }
        home_avail = team_availability.get(home_team, default_avail)
        away_avail = team_availability.get(away_team, default_avail)
        aq_home = home_avail.get("AVAIL_QUALITY", 1.0)
        aq_away = away_avail.get("AVAIL_QUALITY", 1.0)
        avail_features_list.append({
            "AVAIL_HOME": home_avail.get("AVAIL", 1.0),
            "AVAIL_AWAY": away_avail.get("AVAIL", 1.0),
            "AVAIL_QUALITY_HOME": aq_home,
            "AVAIL_QUALITY_AWAY": aq_away,
            "STAR_MISSING_HOME": home_avail.get("STAR_MISSING", 0),
            "STAR_MISSING_AWAY": away_avail.get("STAR_MISSING", 0),
            "N_ROTATION_OUT_HOME": home_avail.get("N_ROTATION_OUT", 0),
            "N_ROTATION_OUT_AWAY": away_avail.get("N_ROTATION_OUT", 0),
            "MISSING_BPM_HOME": home_avail.get("MISSING_BPM", 0.0),
            "MISSING_BPM_AWAY": away_avail.get("MISSING_BPM", 0.0),
            "AVAIL_DIFF": aq_home - aq_away,
        })

        fatigue_features_list.append(
            get_game_fatigue(team_schedule, today_str, home_team, away_team)
        )

        if travel_schedule:
            travel_features_list.append(
                get_game_travel(travel_schedule, today_str, home_team, away_team)
            )
        else:
            travel_features_list.append({
                "TRAVEL_DIST_HOME": 0.0, "TRAVEL_DIST_AWAY": 0.0,
                "TZ_CHANGE_HOME": 0, "TZ_CHANGE_AWAY": 0,
            })

        if travel_schedule:
            extended_fatigue_list.append(
                get_game_extended_fatigue(
                    team_schedule, travel_schedule, today_str,
                    home_team, away_team
                )
            )
        else:
            extended_fatigue_list.append({
                "TZ_CHANGE_SIGNED_HOME": 0, "TZ_CHANGE_SIGNED_AWAY": 0,
                "ALTITUDE_GAME": 0,
                "GAMES_IN_7_HOME": 0, "GAMES_IN_7_AWAY": 0,
                "GAMES_IN_14_HOME": 0, "GAMES_IN_14_AWAY": 0,
                "TRAVEL_7D_HOME": 0.0, "TRAVEL_7D_AWAY": 0.0,
            })

        if sos_lookup:
            sos_features_list.append(
                get_game_sos(sos_lookup, today_str, home_team, away_team)
            )
        else:
            sos_features_list.append({
                "SOS_W_PCT_10_HOME": 0.5, "SOS_W_PCT_10_AWAY": 0.5,
            })

        if srs_lookup:
            srs_features_list.append(
                get_game_srs_features(srs_lookup, today_str, home_team, away_team)
            )
        else:
            srs_features_list.append({
                "SRS_HOME": 0.0, "SRS_AWAY": 0.0, "SRS_DIFF": 0.0,
            })

        if lineup_lookup:
            lineup_features_list.append(
                get_game_lineup_features(lineup_lookup, today_str, home_team, away_team)
            )
        else:
            lineup_features_list.append({
                "TOP5_PM36_HOME": 0.0, "TOP5_PM36_AWAY": 0.0,
                "DEPTH_SCORE_HOME": 1.0, "DEPTH_SCORE_AWAY": 1.0,
                "STAR_POWER_HOME": 0.0, "STAR_POWER_AWAY": 0.0,
                "HHI_MINUTES_HOME": 0.1, "HHI_MINUTES_AWAY": 0.1,
            })

        if ref_assignments and ref_history:
            referee_features_list.append(
                get_game_referee_features(
                    ref_assignments, ref_history, today_str, home_team, away_team
                )
            )
        else:
            referee_features_list.append({
                "REF_CREW_TOTAL_TENDENCY": 0.0,
                "REF_CREW_OVER_PCT": 0.5,
                "REF_CREW_HOME_WIN_PCT": 0.54,
            })

        match_data.append(stats)

    games_data_frame = pd.concat(match_data, ignore_index=True, axis=1).T

    games_data_frame = _add_market_features(
        games_data_frame, spread_values, home_team_odds, away_team_odds
    )

    games_data_frame = add_advanced_features(games_data_frame)
    games_data_frame = add_style_features(games_data_frame)
    games_data_frame = add_differential_features(games_data_frame)
    games_data_frame = add_rolling_features_to_frame(games_data_frame, home_rolling_list, away_rolling_list)
    games_data_frame = add_elo_features_to_frame(games_data_frame, elo_features_list)
    games_data_frame = add_split_features_to_frame(games_data_frame, home_split_list, away_split_list)
    games_data_frame = add_availability_to_frame(games_data_frame, avail_features_list)
    games_data_frame = add_fatigue_to_frame(games_data_frame, fatigue_features_list)
    games_data_frame = add_travel_to_frame(games_data_frame, travel_features_list)
    games_data_frame = add_sos_to_frame(games_data_frame, sos_features_list)
    games_data_frame = add_extended_fatigue_to_frame(games_data_frame, extended_fatigue_list)
    games_data_frame = add_srs_features_to_frame(games_data_frame, srs_features_list)
    games_data_frame = add_lineup_features_to_frame(games_data_frame, lineup_features_list)
    games_data_frame = add_referee_features_to_frame(games_data_frame, referee_features_list)

    data, frame_ml, market_info = _prepare_prediction_matrix(games_data_frame)

    return data, todays_games_uo, frame_ml, home_team_odds, away_team_odds, market_info, spread_home_odds, spread_away_odds


def load_schedule():
    return pd.read_csv(SCHEDULE_PATH, parse_dates=['Date'], date_format='%d/%m/%Y %H:%M')


def resolve_games(odds, sportsbook):
    if odds:
        games = create_todays_games_from_odds(odds, team_index_current)
        if len(games) == 0:
            logger.warning("No games found.")
            return None, None
        game_key = f"{games[0][0]}:{games[0][1]}"
        if game_key not in odds:
            print(game_key)
            print(
                Fore.RED,
                "--------------Games list not up to date for todays games!!!--------------",
            )
            print(Style.RESET_ALL)
            return games, None
        print(f"------------------{sportsbook} odds data------------------")
        for game_key in odds.keys():
            home_team, away_team = game_key.split(":")
            game = odds[game_key]
            line = (
                f"{away_team} ({game[away_team]['money_line_odds']}) @ "
                f"{home_team} ({game[home_team]['money_line_odds']})"
            )
            print(line)

            if 'best_home_ml' in game:
                best_h = game['best_home_ml']
                best_a = game['best_away_ml']
                if best_h.get('book') and best_h['book'] != BOOKMAKER_MAP.get(sportsbook, sportsbook):
                    print(f"  Mejor ML {home_team}: {best_h['odds']} ({best_h['book']})")
                if best_a.get('book') and best_a['book'] != BOOKMAKER_MAP.get(sportsbook, sportsbook):
                    print(f"  Mejor ML {away_team}: {best_a['odds']} ({best_a['book']})")

        return games, odds

    games_json = get_todays_games_json(TODAYS_GAMES_URL)
    return create_todays_games(games_json), None


def run_models(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args,
               odds=None, market_info=None, spread_home_odds=None, spread_away_odds=None):
    predictions = None
    if args.ensemble:
        print("--------------Ensemble Model Predictions---------------")
        predictions = Ensemble_Runner.ensemble_runner(
            data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kelly,
            market_info=market_info,
            spread_home_odds=spread_home_odds,
            spread_away_odds=spread_away_odds,
            sportsbook=args.odds,
        )
        print("-------------------------------------------------------")
    if args.xgb:
        print("---------------XGBoost Model Predictions---------------")
        XGBoost_Runner.xgb_runner(
            data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kelly
        )
        print("-------------------------------------------------------")
    return predictions


def main(args):
    odds = None
    if args.odds:
        provider = OddsApiProvider(sportsbook=args.odds)
        odds = provider.get_odds()

    games, odds = resolve_games(odds, args.odds)
    if games is None:
        return

    stats_json = get_json_data(DATA_URL)
    df = to_data_frame(stats_json)
    schedule_df = load_schedule()
    today = datetime.today()

    game_logs = build_current_game_logs()
    split_data = build_current_split_data()
    logger.info("Splits home/away cargados para %d equipos", len(split_data))

    elo_ratings = get_current_elos(ODDS_DB_PATH)
    logger.info("Elo ratings cargados para %d equipos", len(elo_ratings))

    team_availability = build_current_availability(games)
    logger.info("Availability cargada para %d equipos", len(team_availability))

    with sqlite3.connect(ODDS_DB_PATH) as odds_con:
        team_schedule = build_team_schedule(odds_con)
        travel_schedule = build_team_travel_schedule(odds_con)
        sos_lookup = build_sos_lookup(odds_con)
        srs_lookup = build_srs_history(odds_con)
    logger.info("Fatigue schedule: %d equipos | SOS: %d partidos | SRS: %d partidos",
                len(team_schedule), len(sos_lookup), len(srs_lookup))

    _inject_current_srs(srs_lookup, games, today.strftime("%Y-%m-%d"))

    lineup_lookup = build_lineup_strength_history()
    _inject_current_lineup(lineup_lookup, games, today.strftime("%Y-%m-%d"))

    ref_assignments, ref_history = build_referee_history()

    data, todays_games_uo, frame_ml, home_team_odds, away_team_odds, market_info, spread_home_odds, spread_away_odds = create_todays_games_data(
        games, df, odds, schedule_df, today, game_logs, elo_ratings, split_data, team_availability, team_schedule,
        travel_schedule=travel_schedule, sos_lookup=sos_lookup,
        srs_lookup=srs_lookup, lineup_lookup=lineup_lookup,
        ref_assignments=ref_assignments, ref_history=ref_history,
    )

    predictions = run_models(
        data,
        todays_games_uo,
        frame_ml,
        games,
        home_team_odds,
        away_team_odds,
        args,
        odds=odds,
        market_info=market_info,
        spread_home_odds=spread_home_odds,
        spread_away_odds=spread_away_odds,
    )

    # --- Save predictions + opening lines ---
    if predictions and args.odds:
        today_str = today.strftime("%Y-%m-%d")
        try:
            from src.core.betting.bet_tracker import BetTracker
            tracker = BetTracker()
            tracker.save_predictions(today_str, predictions, args.odds)
        except Exception as e:
            logger.warning("BetTracker save failed: %s", e)

        # Save opening lines for CLV
        try:
            from scripts.collect_closing_lines import collect_lines
            collect_lines(sportsbook=args.odds, line_type="opening")
        except Exception as e:
            logger.debug("Opening lines save skipped: %s", e)

    # --- CLV report ---
    if args.clv:
        try:
            from src.core.betting.clv import print_clv_report
            print_clv_report()
            from src.core.betting.clv_tracking import fill_closing_lines_in_bets, print_clv_report as print_clv_tracking_report
            fill_closing_lines_in_bets()
            print_clv_tracking_report()
        except Exception as e:
            logger.warning("CLV report error: %s", e)

    # --- Live betting session ---
    if args.live and predictions:
        try:
            from src.sports.nba.predict.live_betting import run_live_session
            pregame_preds = [
                {
                    "home_team": p["home_team"],
                    "away_team": p["away_team"],
                    "p_pregame": p["prob_home"],
                }
                for p in predictions
            ]
            run_live_session(pregame_preds)
        except Exception as e:
            logger.warning("Live session error: %s", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NBA W/L Predictor — Ensemble XGBoost 60%% + CatBoost 40%%')
    parser.add_argument('-ensemble', action='store_true', help='Run Ensemble (XGBoost 60%% + CatBoost 40%%)')
    parser.add_argument('-xgb', action='store_true', help='Run XGBoost solo')
    parser.add_argument('-odds', help='Sportsbook: fanduel, draftkings, betmgm, pointsbet, caesars, wynn, bet_rivers_ny')
    parser.add_argument('-kelly', '--kelly', '-kc', dest='kelly', action='store_true', help='Calcula Kelly stake (% de bankroll)')
    parser.add_argument('-clv', action='store_true', help='Imprime reporte CLV (Closing Line Value)')
    parser.add_argument('--live', action='store_true', help='Activa live betting (polling Q1-Q3)')
    args = parser.parse_args()
    main(args)
