"""MLB Win/Loss Predictor — Console CLI.

Punto de entrada para predicciones MLB diarias.
Espeja el patron de predictor.py pero para beisbol.

Uso:
    # Menu interactivo (recomendado)
    PYTHONPATH=. python mlb_predictor.py

    # CLI directo
    PYTHONPATH=. python mlb_predictor.py --ensemble --odds fanduel --kelly
    PYTHONPATH=. python mlb_predictor.py --f5
    PYTHONPATH=. python mlb_predictor.py --totals
    PYTHONPATH=. python mlb_predictor.py --all
    PYTHONPATH=. python mlb_predictor.py --live

Requiere:
    - ODDS_API_KEY en .env o variable de entorno
    - Modelos entrenados en models/mlb/moneyline/ (ver scripts/train_models_mlb.py)
    - Conda env: nba-betting-312 (PYTHONPATH=. desde la raiz del proyecto)
"""
import argparse
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
import pandas as pd
from colorama import Fore, Style, init, deinit

from src.config import get_logger
from src.sports.mlb.config_mlb import (
    MLB_ABBREV, MLB_ABBREV_TO_NAME, MLB_PARKS, MLB_DOMED_PARKS,
    MLB_PLATE_ORIENTATION,
)
from src.sports.mlb.config_paths import (
    MLB_ML_MODELS_DIR, MLB_F5_MODELS_DIR, MLB_TOTALS_MODELS_DIR,
    MLB_TEAMS_DB, MLB_PITCHER_DB, MLB_DATASET_DB,
)

logger = get_logger(__name__)

init()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPORTSBOOKS = [
    "fanduel", "draftkings", "betmgm", "pointsbet", "caesars", "wynn", "bet_rivers_ny"
]

# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------

def _today_et() -> date:
    """Retorna la fecha actual en Eastern Time."""
    et = timezone(timedelta(hours=-5))
    return datetime.now(et).date()


def _get_wind_description(wind_speed_mph: float, wind_dir_deg: float,
                          plate_orientation: float) -> str:
    """Convierte velocidad y direccion de viento en descripcion humana.

    Calcula si el viento va IN, OUT o ACROSS relativo al home plate.

    Args:
        wind_speed_mph:   velocidad en mph
        wind_dir_deg:     direccion de donde viene el viento (meteorologica, grados desde Norte)
        plate_orientation: orientacion del home plate del estadio (grados desde Norte)

    Returns:
        String como "12 mph OUT", "8 mph IN", "5 mph ACROSS"
    """
    if wind_speed_mph < 3:
        return f"{wind_speed_mph:.0f} mph CALM"

    # Angulo relativo entre viento y linea home-plate-pitcher
    # Viento en el campo: de home hacia el outfield = pitcher orientation
    # plate_orientation es el angulo desde donde MIRA el pitcher (desde home)
    relative = (wind_dir_deg - plate_orientation) % 360

    if relative < 45 or relative > 315:
        direction = "IN"    # viento en contra (de OF hacia home)
    elif 135 < relative < 225:
        direction = "OUT"   # viento a favor (de home hacia OF)
    else:
        direction = "ACROSS"

    return f"{wind_speed_mph:.0f} mph {direction}"


def _get_weather_for_game(home_team: str, game_hour_utc: int = 18) -> dict:
    """Obtiene clima para el estadio del equipo local.

    Retorna defaults si el estadio es techado o hay error de API.

    Args:
        home_team:     Nombre completo del equipo local (para buscar coords).
        game_hour_utc: Hora UTC del partido (default 18 = 14:00 ET).

    Returns:
        Dict con: temp_f, wind_speed_mph, wind_dir_deg, humidity_pct,
                  precip_prob, wind_desc (descripcion humana).
    """
    _DEFAULTS = {
        "temp_f": 72.0, "wind_speed_mph": 5.0, "wind_dir_deg": 0.0,
        "humidity_pct": 50.0, "precip_prob": 0.0, "wind_desc": "",
    }

    # Estadio techado: clima no aplica
    if home_team in MLB_DOMED_PARKS:
        return {**_DEFAULTS, "wind_desc": "DOME"}

    coords = MLB_PARKS.get(home_team)
    if coords is None:
        logger.debug("No park coords for %s", home_team)
        return _DEFAULTS

    lat, lon, _ = coords

    try:
        from src.sports.mlb.providers.weather_api import get_game_weather
        weather = get_game_weather(lat, lon, game_hour_local=None)
    except Exception as e:
        logger.debug("Weather API failed for %s: %s", home_team, e)
        return _DEFAULTS

    plate_deg = MLB_PLATE_ORIENTATION.get(home_team, 0.0)
    wind_desc = _get_wind_description(
        weather.get("wind_speed_mph", 5.0),
        weather.get("wind_dir_deg", 0.0),
        plate_deg,
    )

    return {**weather, "wind_desc": wind_desc}


def _get_park_factor(home_team: str) -> int:
    """Retorna el park factor para el estadio (100 = neutral).

    Intenta cargar desde MLBParkFactors.sqlite si existe.
    Fallback: hardcoded valores aproximados para estadios extremos.
    """
    # Valores aproximados (100 = neutral, >100 = hitters park, <100 = pitchers park)
    _PARK_FACTORS_FALLBACK = {
        "Colorado Rockies": 115,    # Coors Field (altitude)
        "Texas Rangers": 107,
        "Boston Red Sox": 105,      # Fenway (Green Monster)
        "Baltimore Orioles": 104,
        "Chicago Cubs": 104,        # Wrigley
        "Cincinnati Reds": 103,
        "Houston Astros": 97,
        "San Francisco Giants": 96,
        "Oakland Athletics": 95,
        "San Diego Padres": 95,
        "Seattle Mariners": 95,
        "New York Mets": 98,
        "Los Angeles Dodgers": 100,
    }

    # Intentar desde SQLite
    try:
        import sqlite3
        if MLB_TEAMS_DB.exists():
            with sqlite3.connect(MLB_TEAMS_DB) as con:
                cur = con.execute(
                    "SELECT park_factor FROM park_factors WHERE team = ? ORDER BY season DESC LIMIT 1",
                    (home_team,)
                )
                row = cur.fetchone()
                if row and row[0]:
                    return int(row[0])
    except Exception:
        pass

    return _PARK_FACTORS_FALLBACK.get(home_team, 100)


def build_today_features(sportsbook: str = "fanduel",
                         include_f5: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    """Construye feature vectors para los juegos MLB de hoy.

    Pasos:
    1. Obtener schedule + probable pitchers desde MLB Stats API
    2. Obtener odds desde Odds API (baseball_mlb)
    3. Cargar lookups de pitcher, batting, park y clima
    4. Construir una fila de features por juego
    5. Retornar (features_df, odds_data)

    Args:
        sportsbook:  Nombre del sportsbook para odds.
        include_f5:  Si True, tambien obtiene F5 odds (usa creditos adicionales de API).

    Returns:
        Tuple de:
          - features_df: DataFrame con una fila por juego + columnas 'home_team', 'away_team'.
                         Retorna DataFrame vacio si no hay juegos.
          - odds_data:   Lista de dicts con odds, una entrada por juego.
    """
    today = _today_et()
    today_str = today.isoformat()

    print(f"\n{Fore.CYAN}[MLB] Obteniendo datos para {today_str}...{Style.RESET_ALL}")

    # --- 1. Schedule + probable pitchers ---
    schedule = []
    try:
        from src.sports.mlb.providers.mlb_stats_api import get_schedule
        schedule = get_schedule(today_str, today_str)
        logger.info("MLB schedule: %d games today", len(schedule))
    except Exception as e:
        logger.warning("MLB Stats API failed: %s", e)

    if not schedule:
        print(f"{Fore.YELLOW}[MLB] No hay partidos programados para hoy ({today_str}).{Style.RESET_ALL}")
        return pd.DataFrame(), []

    # --- 2. Odds desde Odds API ---
    odds_raw = []
    f5_odds_map = {}  # (home, away) -> {f5_ml_home, f5_ml_away}

    try:
        from src.sports.mlb.providers.odds_api_mlb import MLBOddsProvider
        provider = MLBOddsProvider(sportsbook=sportsbook)

        if include_f5:
            odds_raw = provider.get_all_odds_with_f5()
        else:
            odds_raw = provider.get_odds(markets="h2h,spreads,totals")
        logger.info("Odds API: %d MLB games with odds", len(odds_raw))

    except Exception as e:
        logger.warning("Odds API MLB failed: %s", e)
        print(f"{Fore.YELLOW}[MLB] No se pudieron obtener odds: {e}{Style.RESET_ALL}")

    # Construir odds lookup por (home, away)
    odds_by_teams = {}
    for g in odds_raw:
        k = (g.get("home_team", ""), g.get("away_team", ""))
        odds_by_teams[k] = g

    # --- 3. Cargar lookups de features ---
    pitcher_lookup = {}
    batting_lookup = {}

    try:
        from src.sports.mlb.features.pitcher_features import build_pitcher_lookup
        if MLB_PITCHER_DB.exists():
            pitcher_lookup = build_pitcher_lookup(MLB_PITCHER_DB)
            logger.info("Pitcher lookup: %d entries", len(pitcher_lookup))
        else:
            logger.warning("MLBPitcherData.sqlite not found at %s", MLB_PITCHER_DB)
    except Exception as e:
        logger.warning("Pitcher features unavailable: %s", e)

    try:
        from src.sports.mlb.features.batting_features import build_batting_lookup
        if MLB_TEAMS_DB.exists():
            batting_lookup = build_batting_lookup(MLB_TEAMS_DB)
            logger.info("Batting lookup: %d entries", len(batting_lookup))
        else:
            logger.warning("MLBTeamData.sqlite not found at %s", MLB_TEAMS_DB)
    except Exception as e:
        logger.warning("Batting features unavailable: %s", e)

    # --- 4. Construir una fila de features por juego ---
    rows = []
    final_odds = []

    for game in schedule:
        home = game.get("home_team", "")
        away = game.get("away_team", "")

        if not home or not away:
            continue

        sp_home = game.get("home_pitcher") or "TBD"
        sp_away = game.get("away_pitcher") or "TBD"
        game_date = game.get("game_date", today_str)

        # Obtener odds para este juego (match por (home, away) exacto primero,
        # luego intentar match fuzzy si no hay coincidencia exacta)
        game_odds = odds_by_teams.get((home, away), {})
        if not game_odds:
            # Intentar match con nombres normalizados (aliases)
            for (oh, oa), og in odds_by_teams.items():
                if _normalize_team(oh) == _normalize_team(home) and _normalize_team(oa) == _normalize_team(away):
                    game_odds = og
                    break

        # Hora de inicio para clima
        commence_time = game_odds.get("commence_time", "")
        game_hour_utc = 18  # default 14:00 ET
        if commence_time:
            try:
                dt = datetime.fromisoformat(str(commence_time).replace("Z", "+00:00"))
                game_hour_utc = dt.hour
            except Exception:
                pass

        # Merge F5 odds from bulk fetch if available
        game_odds_merged = dict(game_odds)
        if include_f5 and "ml_home_f5" not in game_odds_merged:
            game_odds_merged["ml_home_f5"] = None
            game_odds_merged["ml_away_f5"] = None

        # --- Park factor ---
        park_factor = _get_park_factor(home)
        game_odds_merged["park_factor"] = park_factor

        # --- Clima ---
        weather = _get_weather_for_game(home, game_hour_utc)
        game_odds_merged.update({
            "temp_f":         weather.get("temp_f"),
            "wind_speed_mph": weather.get("wind_speed_mph"),
            "wind_dir_deg":   weather.get("wind_dir_deg"),
            "humidity_pct":   weather.get("humidity_pct"),
            "precip_prob":    weather.get("precip_prob"),
            "wind_desc":      weather.get("wind_desc", ""),
        })

        # --- SP names para display ---
        game_odds_merged["home_pitcher"] = sp_home
        game_odds_merged["away_pitcher"] = sp_away
        game_odds_merged["home_team"] = home
        game_odds_merged["away_team"] = away

        # --- Pitcher features (24 features: 12 home + 12 away) ---
        sp_home_feats = {}
        sp_away_feats = {}
        if pitcher_lookup:
            try:
                from src.sports.mlb.features.pitcher_features import get_game_sp_features
                game_date_obj = datetime.strptime(game_date[:10], "%Y-%m-%d").date()
                # Season key: YYYY (e.g. "2026")
                season = str(game_date_obj.year)
                sp_feats = get_game_sp_features(
                    pitcher_lookup, game_date_obj, season, sp_home, sp_away
                )
                # Retorna dict con SP_HOME_* y SP_AWAY_* prefixes
                for key, val in (sp_feats or {}).items():
                    if key.startswith("SP_HOME_"):
                        sp_home_feats[key] = val
                    elif key.startswith("SP_AWAY_"):
                        sp_away_feats[key] = val
            except Exception as e:
                logger.debug("SP features failed for %s @ %s: %s", away, home, e)

        # --- Batting features (32 features: 16 home + 16 away) ---
        bat_home_feats = {}
        bat_away_feats = {}
        if batting_lookup:
            try:
                from src.sports.mlb.features.batting_features import get_team_batting_features
                game_date_obj = datetime.strptime(game_date[:10], "%Y-%m-%d").date()
                h_bat = get_team_batting_features(batting_lookup, home, game_date_obj)
                a_bat = get_team_batting_features(batting_lookup, away, game_date_obj)
                # Prefijos para distinguir home vs away
                bat_home_feats = {f"HOME_{k}": v for k, v in (h_bat or {}).items()}
                bat_away_feats = {f"AWAY_{k}": v for k, v in (a_bat or {}).items()}
            except Exception as e:
                logger.debug("Batting features failed for %s @ %s: %s", away, home, e)

        # --- Market features ---
        ml_home = game_odds_merged.get("ml_home")
        ml_away = game_odds_merged.get("ml_away")
        run_line = game_odds_merged.get("run_line_home", -1.5)
        total = game_odds_merged.get("total", 8.5)

        market_prob_home = _american_odds_to_prob(ml_home, ml_away) if ml_home and ml_away else 0.5

        vig_magnitude = 0.0
        if ml_home and ml_away:
            def _implied(odds):
                odds = float(odds)
                return abs(odds) / (abs(odds) + 100.0) if odds < 0 else 100.0 / (odds + 100.0)
            vig_magnitude = (_implied(ml_home) + _implied(ml_away)) - 1.0

        # --- Construir fila final ---
        row = {
            "home_team": home,
            "away_team": away,
            # Market features
            "MARKET_ML_PROB": market_prob_home,
            "MARKET_SPREAD": float(run_line) if run_line is not None else -1.5,
            "VIG_MAGNITUDE": vig_magnitude,
            "TOTAL_LINE": float(total) if total is not None else 8.5,
            # Park + weather
            "PARK_FACTOR": park_factor,
            "TEMP_F": weather.get("temp_f", 72.0),
            "WIND_SPEED_MPH": weather.get("wind_speed_mph", 5.0),
            "WIND_DIR_DEG": weather.get("wind_dir_deg", 0.0),
            "HUMIDITY_PCT": weather.get("humidity_pct", 50.0),
            "PRECIP_PROB": weather.get("precip_prob", 0.0),
            # Park is dome
            "IS_DOME": 1.0 if home in MLB_DOMED_PARKS else 0.0,
            **sp_home_feats,
            **sp_away_feats,
            **bat_home_feats,
            **bat_away_feats,
        }

        rows.append(row)
        final_odds.append(game_odds_merged)

    if not rows:
        print(f"{Fore.YELLOW}[MLB] No se pudo construir features para ningun partido.{Style.RESET_ALL}")
        return pd.DataFrame(), []

    features_df = pd.DataFrame(rows)

    n_games = len(rows)
    n_with_odds = sum(1 for g in final_odds if g.get("ml_home") is not None)
    print(f"{Fore.GREEN}[MLB] {n_games} partidos hoy | {n_with_odds} con odds | "
          f"Features: {features_df.shape[1]} columnas{Style.RESET_ALL}")

    return features_df, final_odds


def _normalize_team(name: str) -> str:
    """Normaliza nombre de equipo para matching: lowercase, sin espacios extras."""
    return name.lower().strip() if name else ""


def _american_odds_to_prob(ml_home, ml_away) -> float:
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
    return p_home / total if total > 0 else 0.5


# ---------------------------------------------------------------------------
# Run pipelines
# ---------------------------------------------------------------------------

def run_mlb_moneyline(args) -> list[dict]:
    """Ejecuta predicciones de moneyline full-game."""
    sportsbook = getattr(args, "odds", "fanduel") or "fanduel"

    features_df, odds_data = build_today_features(sportsbook=sportsbook, include_f5=False)

    if features_df.empty:
        return []

    from src.sports.mlb.predict.ensemble_runner import run_mlb_ensemble
    predictions = run_mlb_ensemble(features_df, odds_data, sportsbook=sportsbook)

    # Guardar en BetTracker si kelly
    if getattr(args, "kelly", False) and predictions:
        try:
            from src.core.betting.bet_tracker import BetTracker
            from src.sports.mlb.config_paths import MLB_BETS_DB
            tracker = BetTracker(db_path=str(MLB_BETS_DB))
            today_str = _today_et().isoformat()
            tracker.save_predictions(today_str, predictions, sportsbook)
        except Exception as e:
            logger.debug("BetTracker MLB save failed: %s", e)

    return predictions


def run_mlb_f5(args) -> list[dict]:
    """Ejecuta predicciones de First 5 Innings."""
    sportsbook = getattr(args, "odds", "fanduel") or "fanduel"

    features_df, odds_data = build_today_features(sportsbook=sportsbook, include_f5=True)

    if features_df.empty:
        return []

    from src.sports.mlb.predict.f5_runner import predict_f5, load_f5_models

    f5_models = load_f5_models()
    if f5_models is None:
        print(f"\n{Fore.YELLOW}[F5] Modelos F5 no encontrados en {MLB_F5_MODELS_DIR}")
        print(f"     Entrena primero: PYTHONPATH=. python scripts/train_models_mlb.py --f5{Style.RESET_ALL}\n")
        # Fallback: mostrar solo odds F5 sin modelo
        f5_results = predict_f5(None, odds_data, sportsbook=sportsbook)
    else:
        f5_results = predict_f5(features_df, odds_data, sportsbook=sportsbook)

    if not f5_results:
        print(f"{Fore.YELLOW}[F5] No hay odds de F5 disponibles hoy.{Style.RESET_ALL}")
        return []

    # Print F5-specific output
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"  MLB F5 (First 5 Innings) | {sportsbook.upper()}")
    print(f"{'='*60}{Style.RESET_ALL}\n")

    for r in sorted(f5_results, key=lambda x: abs(x.get("f5_ev", 0)), reverse=True):
        home = r.get("home_team", "")
        away = r.get("away_team", "")
        prob_h = r.get("f5_prob_home", 0.5)
        ev = r.get("f5_ev", 0.0)
        kelly = r.get("f5_kelly", 0.0)
        tag = r.get("f5_tag", "N/A")
        f5_h_odds = r.get("f5_ml_home_odds")
        f5_a_odds = r.get("f5_ml_away_odds")

        # SP matchup from odds data
        game_odds = next((o for o in odds_data if o.get("home_team") == home and o.get("away_team") == away), {})
        sp_home = game_odds.get("home_pitcher", "TBD")
        sp_away = game_odds.get("away_pitcher", "TBD")

        tag_color = Fore.GREEN if tag == "BET" else (Fore.YELLOW if tag == "SKIP" else Fore.RED)
        ev_color = Fore.GREEN if ev > 0 else Fore.RED

        odds_str = ""
        if f5_h_odds and f5_a_odds:
            odds_str = f"  {home} {f5_h_odds:+d} / {away} {f5_a_odds:+d}"

        pitcher_str = ""
        if sp_away != "TBD" and sp_home != "TBD":
            pitcher_str = f"\n    SP: {sp_away} vs {sp_home}"

        print(
            f"  {away} @ {home}{odds_str}\n"
            f"    F5 pick: {home if prob_h >= 0.5 else away} ({max(prob_h, 1-prob_h)*100:.1f}%)  "
            f"EV={ev_color}{ev:+.1f}{Style.RESET_ALL}  Kelly={kelly:.2f}%  "
            f"[{tag_color}{tag}{Style.RESET_ALL}]{pitcher_str}\n"
        )

    return f5_results


def run_mlb_totals(args) -> list[dict]:
    """Ejecuta predicciones de totals (O/U carreras)."""
    sportsbook = getattr(args, "odds", "fanduel") or "fanduel"

    features_df, odds_data = build_today_features(sportsbook=sportsbook, include_f5=False)

    if features_df.empty:
        return []

    from src.sports.mlb.predict.totals_runner import predict_totals, load_totals_model

    model = load_totals_model()
    if model is None:
        print(f"\n{Fore.YELLOW}[Totals] Modelo de regresion no encontrado en {MLB_TOTALS_MODELS_DIR}")
        print(f"         Entrena primero: PYTHONPATH=. python scripts/train_models_mlb.py --totals{Style.RESET_ALL}\n")
        totals_results = predict_totals(None, odds_data)
    else:
        totals_results = predict_totals(features_df, odds_data)

    if not totals_results:
        print(f"{Fore.YELLOW}[Totals] No hay odds de O/U disponibles hoy.{Style.RESET_ALL}")
        return []

    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"  MLB Totals (O/U Carreras) | {sportsbook.upper()}")
    print(f"{'='*60}{Style.RESET_ALL}\n")

    for r in sorted(totals_results, key=lambda x: abs(x.get("ou_ev", 0)), reverse=True):
        home = r.get("home_team", "")
        away = r.get("away_team", "")
        label = r.get("ou_label", "---")
        line = r.get("ou_line", "---")
        prob = r.get("ou_prob", 0.5)
        ev = r.get("ou_ev", 0.0)
        kelly = r.get("ou_kelly", 0.0)
        pred = r.get("predicted_total")

        label_color = Fore.MAGENTA if label == "UNDER" else Fore.BLUE
        ev_color = Fore.GREEN if ev > 0 else Fore.RED
        pred_str = f"  pred={pred:.1f}" if pred is not None else ""

        # Park + weather from odds
        game_odds = next((o for o in odds_data if o.get("home_team") == home and o.get("away_team") == away), {})
        park_str = ""
        pf = game_odds.get("park_factor")
        temp = game_odds.get("temp_f")
        wind = game_odds.get("wind_desc", "")
        if pf and pf != 100:
            park_str += f"  PF={pf}"
        if temp:
            park_str += f"  {temp:.0f}F"
        if wind:
            park_str += f"  {wind}"

        print(
            f"  {away} @ {home}\n"
            f"    O/U: {label_color}{label}{Style.RESET_ALL} {line}  "
            f"P={prob:.1%}  EV={ev_color}{ev:+.1f}{Style.RESET_ALL}  "
            f"Kelly={kelly:.2f}%{pred_str}"
        )
        if park_str:
            print(f"    Park:{park_str}")
        print()

    return totals_results


def run_mlb_live(args, pregame_preds: list[dict] = None):
    """Sesion de live betting MLB — polling por inning."""
    sportsbook = getattr(args, "odds", "fanduel") or "fanduel"

    # Obtener predicciones pregame primero si no fueron proporcionadas
    if not pregame_preds:
        features_df, odds_data = build_today_features(sportsbook=sportsbook, include_f5=False)

        if not features_df.empty:
            try:
                from src.sports.mlb.predict.ensemble_runner import run_mlb_ensemble
                preds = run_mlb_ensemble(features_df, odds_data, sportsbook=sportsbook)
                pregame_preds = [
                    {
                        "home_team": p["home_team"],
                        "away_team": p["away_team"],
                        "p_pregame": p["prob_home"],
                    }
                    for p in preds
                ]
            except Exception as e:
                logger.warning("Ensemble failed for live pregame: %s", e)

        # Fallback: use implied probabilities from odds
        if not pregame_preds and odds_data:
            for game in odds_data:
                ml_h = game.get("ml_home")
                ml_a = game.get("ml_away")
                if ml_h and ml_a:
                    p_home = _american_odds_to_prob(ml_h, ml_a)
                    pregame_preds.append({
                        "home_team": game["home_team"],
                        "away_team": game["away_team"],
                        "p_pregame": p_home,
                    })

    if not pregame_preds:
        print(f"{Fore.YELLOW}[MLB Live] No hay predicciones pregame disponibles.{Style.RESET_ALL}")
        return

    from src.sports.mlb.predict.live_betting import run_live_session
    run_live_session(pregame_preds)


def run_mlb_all(args) -> dict:
    """Ejecuta todos los mercados MLB: ML + F5 + O/U."""
    sportsbook = getattr(args, "odds", "fanduel") or "fanduel"

    # Obtener features y odds una sola vez (incluye F5 odds)
    features_df, odds_data = build_today_features(sportsbook=sportsbook, include_f5=True)

    if features_df.empty:
        return {}

    results = {}

    # --- ML moneyline ---
    try:
        from src.sports.mlb.predict.ensemble_runner import run_mlb_ensemble
        results["moneyline"] = run_mlb_ensemble(features_df, odds_data, sportsbook=sportsbook)
    except Exception as e:
        logger.error("ML ensemble failed: %s", e)
        results["moneyline"] = []

    # --- F5 ---
    try:
        from src.sports.mlb.predict.f5_runner import predict_f5, load_f5_models
        f5_models = load_f5_models()
        f5_feats = features_df if f5_models is not None else None
        results["f5"] = predict_f5(f5_feats, odds_data, sportsbook=sportsbook)

        if results["f5"]:
            print(f"\n{Fore.CYAN}--- F5 Summary ---{Style.RESET_ALL}")
            for r in results["f5"]:
                if r.get("f5_ev", 0) > 0:
                    ev_color = Fore.GREEN
                    print(
                        f"  {r['away_team']} @ {r['home_team']}  "
                        f"F5 EV={ev_color}{r['f5_ev']:+.1f}{Style.RESET_ALL}  "
                        f"Kelly={r['f5_kelly']:.2f}%  [{r.get('f5_tag', 'N/A')}]"
                    )
    except Exception as e:
        logger.error("F5 runner failed: %s", e)
        results["f5"] = []

    # --- Totals ---
    try:
        from src.sports.mlb.predict.totals_runner import predict_totals, load_totals_model
        tot_model = load_totals_model()
        tot_feats = features_df if tot_model is not None else None
        results["totals"] = predict_totals(tot_feats, odds_data)

        if results["totals"]:
            print(f"\n{Fore.CYAN}--- Totals Summary ---{Style.RESET_ALL}")
            for r in results["totals"]:
                if abs(r.get("ou_ev", 0)) > 0.5:
                    label = r.get("ou_label", "---")
                    label_color = Fore.MAGENTA if label == "UNDER" else Fore.BLUE
                    ev_color = Fore.GREEN if r.get("ou_ev", 0) > 0 else Fore.RED
                    pred_str = f" pred={r['predicted_total']:.1f}" if r.get("predicted_total") else ""
                    print(
                        f"  {r['away_team']} @ {r['home_team']}  "
                        f"{label_color}{label}{Style.RESET_ALL} {r.get('ou_line', '---')}  "
                        f"EV={ev_color}{r.get('ou_ev', 0):+.1f}{Style.RESET_ALL}{pred_str}"
                    )
    except Exception as e:
        logger.error("Totals runner failed: %s", e)
        results["totals"] = []

    return results


# ---------------------------------------------------------------------------
# Interactive menu
# ---------------------------------------------------------------------------

def interactive_menu():
    """Menu interactivo — sin flags, solo ejecutar `python mlb_predictor.py`."""
    print(f"\n{Fore.CYAN}{'=' * 52}")
    print(f"  MLB Predictor")
    print(f"{'=' * 52}{Style.RESET_ALL}\n")

    # --- Sportsbook ---
    print(f"  {Fore.YELLOW}Sportsbook:{Style.RESET_ALL}")
    for i, book in enumerate(SPORTSBOOKS, 1):
        default_tag = " (default)" if book == "fanduel" else ""
        print(f"    {i}) {book}{default_tag}")

    book_choice = input(f"\n  Selecciona [1]: ").strip() or "1"
    try:
        sportsbook = SPORTSBOOKS[int(book_choice) - 1]
    except (ValueError, IndexError):
        sportsbook = "fanduel"

    # --- Mercado ---
    print(f"\n  {Fore.YELLOW}Que quieres hacer:{Style.RESET_ALL}")
    print(f"    1) MLB Moneyline")
    print(f"    2) MLB First 5 Innings")
    print(f"    3) MLB Totals (O/U)")
    print(f"    4) MLB Live Betting")
    print(f"    5) Exit")

    market_choice = input(f"\n  Selecciona [1]: ").strip() or "1"
    if market_choice == "5":
        print("  Saliendo.")
        return

    # --- Resumen ---
    market_names = {
        "1": "ML Moneyline",
        "2": "F5 First 5 Innings",
        "3": "Totals O/U",
        "4": "Live Betting",
    }
    market_label = market_names.get(market_choice, "Moneyline")

    print(f"\n{Fore.GREEN}{'_' * 52}")
    print(f"  Configuracion:")
    print(f"    Liga:        MLB")
    print(f"    Mercado:     {market_label}")
    print(f"    Sportsbook:  {sportsbook}")
    print(f"    Kelly:       si")
    print(f"{'_' * 52}{Style.RESET_ALL}\n")

    confirm = input(f"  Ejecutar? (Enter = si, n = cancelar): ").strip()
    if confirm.lower() in ("n", "no"):
        print("  Cancelado.")
        return

    args = argparse.Namespace(
        odds=sportsbook,
        kelly=True,
        ensemble=True,
        f5=(market_choice == "2"),
        totals=(market_choice == "3"),
        live=(market_choice == "4"),
        all_markets=False,
    )

    pregame_results = None

    if market_choice == "1":
        preds = run_mlb_moneyline(args)
        if preds:
            pregame_results = [
                {
                    "home_team": p["home_team"],
                    "away_team": p["away_team"],
                    "p_pregame": p["prob_home"],
                }
                for p in preds
            ]
    elif market_choice == "2":
        run_mlb_f5(args)
    elif market_choice == "3":
        run_mlb_totals(args)
    elif market_choice == "4":
        run_mlb_live(args)
        return

    # Offer live betting after moneyline
    if market_choice == "1" and pregame_results:
        print(f"\n{Fore.YELLOW}Quieres entrar a live betting con estas predicciones?{Style.RESET_ALL}")
        live_choice = input("  (s/n) [n]: ").strip()
        if live_choice.lower() in ("s", "si", "y", "yes"):
            run_mlb_live(args, pregame_preds=pregame_results)


# ---------------------------------------------------------------------------
# CLI main
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MLB W/L Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ensemble", action="store_true", default=True,
        help="Usar ensemble XGB + CatBoost (default)"
    )
    parser.add_argument(
        "--odds", type=str, default="fanduel",
        choices=SPORTSBOOKS,
        help="Sportsbook para odds (default: fanduel)"
    )
    parser.add_argument(
        "--kelly", action="store_true", default=True,
        help="Mostrar Kelly sizing (default: si)"
    )
    parser.add_argument(
        "--f5", action="store_true",
        help="Ejecutar predicciones F5 (First 5 Innings)"
    )
    parser.add_argument(
        "--totals", action="store_true",
        help="Ejecutar predicciones de totals (O/U)"
    )
    parser.add_argument(
        "--all", action="store_true", dest="all_markets",
        help="Ejecutar todos los mercados (ML + F5 + O/U)"
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Modo live betting (polling por inning)"
    )
    return parser


def main(args=None):
    """Punto de entrada para uso con flags CLI."""
    parser = _build_arg_parser()
    args = parser.parse_args(args)

    if args.all_markets:
        run_mlb_all(args)
    elif args.f5:
        run_mlb_f5(args)
    elif args.totals:
        run_mlb_totals(args)
    elif args.live:
        run_mlb_live(args)
    else:
        # Default: moneyline
        run_mlb_moneyline(args)

    deinit()


if __name__ == "__main__":
    # Si hay flags CLI, usar modo tradicional; si no, menu interactivo
    if len(sys.argv) > 1:
        main()
    else:
        try:
            interactive_menu()
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Interrumpido por usuario.{Style.RESET_ALL}")
        finally:
            deinit()
