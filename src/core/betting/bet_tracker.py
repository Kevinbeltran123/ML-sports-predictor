"""BetTracker: persiste y analiza predicciones del ensemble NBA.

Concepto clave — ¿Por qué guardar predicciones automáticamente?
  Sin persistencia, el modelo es una caja negra: genera números y los bota.
  Con persistencia podemos medir:
    1. Accuracy real a lo largo del tiempo (¿el 68.7% se mantiene en producción?)
    2. Edge real vs el mercado (¿nuestro P > market_prob consistentemente?)
    3. CLV (Closing Line Value): ¿apostamos antes de que la línea se mueva en
       nuestra dirección? Si CLV > 0 consistentemente → edge real demostrado.
    4. P&L simulado: si hubiéramos apostado Quarter-Kelly en todas las apuestas
       con EV > 0, ¿cuánto habríamos ganado?

Flujo:
  main.py → ensemble_runner() retorna lista de dicts
           → BetTracker().save_predictions()  → Data/BetsTracking.sqlite

  Al día siguiente:
  scripts/update_results.py → BetTracker().update_results(date)
                             → actualiza ml_correct, ou_correct, actual_scores
"""
import difflib
import sqlite3
from datetime import datetime
from pathlib import Path

from src.config import BETS_DB
from src.core.betting.kelly_criterion import calculate_quarter_kelly

# Nombres de equipos que usa nba_api (TEAM_ABBREVIATION en LeagueGameFinder)
# Mapeamos de nuestros nombres completos a las abreviaciones de nba_api
NBA_API_ABBREV = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

# Mapa inverso: abreviación → nombre completo
ABBREV_TO_FULL = {v: k for k, v in NBA_API_ABBREV.items() if k != "LA Clippers"}


class BetTracker:
    """Guarda, actualiza y analiza predicciones del ensemble.

    Uso básico:
        tracker = BetTracker()
        tracker.save_predictions("2026-02-11", predictions, "fanduel")
        tracker.update_results("2026-02-11")
        tracker.print_report()
    """

    DB_PATH = BETS_DB

    def __init__(self):
        self._init_db()
        self._migrate_conformal_columns()

    def _init_db(self):
        """Crea la base de datos y tabla si no existen.

        Nota técnica — UNIQUE constraint:
          (game_date, home_team, away_team, sportsbook) es la clave única.
          Si corremos main.py dos veces en el mismo día para el mismo sportsbook,
          INSERT OR REPLACE actualiza los datos en lugar de duplicarlos.
          Esto es un "upsert" (update + insert) — patrón muy útil en ETLs.
        """
        self.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.DB_PATH) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    -- Identificadores del partido
                    game_date TEXT NOT NULL,           -- 'YYYY-MM-DD'
                    run_timestamp TEXT NOT NULL,       -- cuándo se corrió el modelo
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    sportsbook TEXT,                   -- 'fanduel', 'draftkings', etc.

                    -- Predicciones del modelo ensemble
                    prob_home REAL,      -- P(home gana), e.g., 0.653
                    prob_away REAL,      -- P(away gana) = 1 - prob_home
                    prob_over REAL,      -- P(total > línea), e.g., 0.521
                    prob_under REAL,     -- P(total < línea)
                    ou_line REAL,        -- línea total del partido, e.g., 215.5

                    -- Mercado al momento de la predicción (apertura)
                    ml_home_odds INTEGER,   -- odds americanos home, e.g., -120
                    ml_away_odds INTEGER,   -- odds americanos away, e.g., +100
                    spread REAL,            -- spread home, e.g., -3.5 (home favorito)
                    market_prob_home REAL,  -- prob implícita del mercado (sin vig)

                    -- Edge: cuánto supera el modelo al mercado
                    -- Edge positivo = modelo cree que home es mejor apuesta que lo que el mercado refleja
                    edge_home REAL,     -- prob_home - market_prob_home
                    edge_away REAL,     -- prob_away - (1 - market_prob_home)

                    -- Valor esperado y Kelly
                    -- EV: si apostamos $100 muchas veces, ¿cuánto ganamos/perdemos en promedio?
                    -- Kelly: fracción óptima del bankroll según la fórmula Kelly Criterion
                    ev_home REAL,       -- expected value de apostar al home
                    ev_away REAL,       -- expected value de apostar al away
                    kelly_home REAL,    -- Quarter-Kelly % bankroll para home
                    kelly_away REAL,    -- Quarter-Kelly % bankroll para away

                    -- Conformal prediction (filtro de confianza)
                    -- set_size=1: solo una clase plausible → apostar
                    -- set_size=2: ambas clases plausibles → skip
                    conformal_set_size INTEGER,  -- 0, 1, o 2
                    conformal_margin REAL,       -- max_prob - threshold

                    -- Varianza per-game (sigma para DRO-Kelly)
                    -- sigma bajo → Kelly agresivo, sigma alto → Kelly conservador
                    sigma REAL,

                    -- CLV (Closing Line Value): métrica de edge a largo plazo
                    -- Si closing_ml_home_odds > ml_home_odds (se volvió más corto),
                    -- significa que el mercado se movió en contra → CLV negativo.
                    -- Si la línea se alarga (casas creen que home tiene menos chance),
                    -- nuestro bet fue en el lado correcto antes que el mercado → CLV positivo.
                    closing_ml_home_odds INTEGER,   -- odds al cierre (justo antes del partido)
                    closing_ml_away_odds INTEGER,
                    clv_home REAL,     -- positivo = apostamos antes de que línea se moviera a nuestro favor
                    clv_away REAL,

                    -- Resultados reales (rellenados por update_results.py)
                    actual_home_score INTEGER,
                    actual_away_score INTEGER,
                    actual_total INTEGER,       -- home_score + away_score
                    actual_winner TEXT,         -- 'HOME' o 'AWAY'
                    actual_ou TEXT,             -- 'OVER' o 'UNDER'
                    ml_correct INTEGER,         -- 1 si el modelo acertó el ganador, 0 si no
                    ou_correct INTEGER,         -- 1 si el modelo acertó OVER/UNDER, 0 si no

                    -- Unicidad: un registro por partido por sportsbook por día
                    UNIQUE(game_date, home_team, away_team, sportsbook)
                )
            """)
            con.commit()

    def _migrate_conformal_columns(self):
        """Agrega columnas conformal, sigma y AH a DBs existentes (no-op si ya existen)."""
        with sqlite3.connect(self.DB_PATH) as con:
            existing = {row[1] for row in con.execute("PRAGMA table_info(predictions)").fetchall()}
            new_cols = [
                ("conformal_set_size", "INTEGER"), ("conformal_margin", "REAL"),
                ("sigma", "REAL"),
                # Asian Handicap columns
                ("ah_spread", "REAL"), ("ah_prob_home_cover", "REAL"),
                ("ah_prob_away_cover", "REAL"), ("ah_home_odds", "INTEGER"),
                ("ah_away_odds", "INTEGER"), ("ah_ev_home", "REAL"),
                ("ah_ev_away", "REAL"), ("ah_kelly_home", "REAL"),
                ("ah_kelly_away", "REAL"), ("ah_expected_margin", "REAL"),
                # Margin regression columns
                ("reg_margin", "REAL"), ("reg_p_cover_home", "REAL"),
                ("reg_sigma", "REAL"),
                # ATS tracking (populated by update_results.py)
                ("ah_actual_cover", "INTEGER"),   # 1=home covered, 0=away covered, NULL=push
                ("ah_residual", "REAL"),           # Margin + MARKET_SPREAD (positive=home covers)
                ("ah_game_sigma", "REAL"),         # game-specific sigma from Q10/Q90
                ("ah_tag", "TEXT"),                # AH-BET / AH-SKIP / AH-PASS
                # League column (NBA/WNBA)
                ("league", "TEXT DEFAULT 'NBA'"),
            ]
            for col, dtype in new_cols:
                if col not in existing:
                    con.execute(f"ALTER TABLE predictions ADD COLUMN {col} {dtype}")
            con.commit()

    def save_predictions(self, game_date: str, predictions: list[dict], sportsbook: str):
        """Guarda las predicciones del ensemble en la base de datos.

        Args:
            game_date: fecha del partido en formato 'YYYY-MM-DD'
            predictions: lista de dicts retornada por ensemble_runner()
                Cada dict tiene: home_team, away_team, prob_home, prob_away,
                prob_over, prob_under, ou_line, ml_home_odds, ml_away_odds,
                spread, market_prob_home, ev_home, ev_away, kelly_home, kelly_away
            sportsbook: nombre del sportsbook ('fanduel', 'draftkings', etc.)

        Nota: usa INSERT OR REPLACE (upsert), así que correr el modelo dos veces
        en el mismo día simplemente actualiza los datos, no duplica.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with sqlite3.connect(self.DB_PATH) as con:
            for p in predictions:
                # Ensemble_Runner ya retorna Quarter-Kelly (calculate_quarter_kelly).
                # Usamos esos valores directamente si están disponibles; de lo contrario
                # los recalculamos desde las odds y probabilidades del modelo.
                if p.get("ml_home_odds") and p.get("prob_home") is not None:
                    kelly_home_qk = p.get("kelly_home") or float(calculate_quarter_kelly(p["ml_home_odds"], p["prob_home"]))
                    kelly_away_qk = p.get("kelly_away") or float(calculate_quarter_kelly(p["ml_away_odds"], p["prob_away"]))
                else:
                    kelly_home_qk = 0.0
                    kelly_away_qk = 0.0

                # Calcular edge: cuánto supera nuestro modelo al mercado
                market_prob = p.get("market_prob_home", 0.5)
                edge_home = round(p["prob_home"] - market_prob, 4)
                edge_away = round(p["prob_away"] - (1 - market_prob), 4)

                con.execute("""
                    INSERT OR REPLACE INTO predictions (
                        game_date, run_timestamp, home_team, away_team, sportsbook,
                        prob_home, prob_away, prob_over, prob_under, ou_line,
                        ml_home_odds, ml_away_odds, spread, market_prob_home,
                        edge_home, edge_away, ev_home, ev_away,
                        kelly_home, kelly_away,
                        conformal_set_size, conformal_margin,
                        sigma,
                        ah_spread, ah_prob_home_cover, ah_prob_away_cover,
                        ah_home_odds, ah_away_odds,
                        ah_ev_home, ah_ev_away,
                        ah_kelly_home, ah_kelly_away,
                        ah_expected_margin,
                        reg_margin, reg_p_cover_home, reg_sigma,
                        ah_game_sigma, ah_tag
                    ) VALUES (
                        ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?,
                        ?, ?,
                        ?,
                        ?, ?, ?,
                        ?, ?,
                        ?, ?,
                        ?, ?,
                        ?,
                        ?, ?, ?,
                        ?, ?
                    )
                """, (
                    game_date, timestamp, p["home_team"], p["away_team"], sportsbook,
                    p["prob_home"], p["prob_away"], p["prob_over"], p["prob_under"], p["ou_line"],
                    p.get("ml_home_odds"), p.get("ml_away_odds"),
                    p.get("spread"), market_prob,
                    edge_home, edge_away,
                    p.get("ev_home"), p.get("ev_away"),
                    kelly_home_qk, kelly_away_qk,
                    p.get("conformal_set_size"), p.get("conformal_margin"),
                    p.get("sigma"),
                    p.get("ah_spread"), p.get("ah_prob_home_cover"), p.get("ah_prob_away_cover"),
                    p.get("ah_home_odds"), p.get("ah_away_odds"),
                    p.get("ah_ev_home"), p.get("ah_ev_away"),
                    p.get("ah_kelly_home"), p.get("ah_kelly_away"),
                    p.get("ah_expected_margin"),
                    p.get("reg_margin"), p.get("reg_p_cover_home"), p.get("reg_sigma"),
                    p.get("ah_game_sigma"), p.get("ah_tag"),
                ))
            con.commit()

        print(f"  [BetTracker] {len(predictions)} predicciones guardadas → {self.DB_PATH}")

    def update_results(self, game_date: str):
        """Descarga los resultados reales del día y actualiza la tabla.

        Usa nba_api.stats.endpoints.LeagueGameFinder para obtener los marcadores
        finales de todos los partidos de una fecha.

        Args:
            game_date: fecha en formato 'YYYY-MM-DD'

        Conceptos:
          - LeagueGameFinder: endpoint de nba_api que retorna una tabla de
            partidos con TEAM_ABBREVIATION, PTS (puntos del equipo), WL (W o L).
          - Cada partido aparece dos veces (una fila por equipo), así que
            agrupamos por GAME_ID para obtener el marcador completo.
        """
        try:
            from nba_api.stats.endpoints import leaguegamefinder
        except ImportError:
            print("  [BetTracker] ERROR: nba_api no instalado. pip install nba_api")
            return

        # Obtener partidos de la DB que no tienen resultado aún
        with sqlite3.connect(self.DB_PATH) as con:
            rows = con.execute("""
                SELECT id, home_team, away_team, ou_line
                FROM predictions
                WHERE game_date = ? AND ml_correct IS NULL
            """, (game_date,)).fetchall()

        if not rows:
            print(f"  [BetTracker] No hay predicciones sin resultado para {game_date}")
            return

        # Descargar resultados — ESPN primero (mas rapido y confiable), nba_api como fallback
        print(f"  [BetTracker] Descargando resultados de {game_date}...")
        results_by_teams = self._fetch_results_espn(game_date)

        if not results_by_teams:
            results_by_teams = self._fetch_results_nba_api(game_date, leaguegamefinder)

        if not results_by_teams:
            print(f"  [BetTracker] Sin resultados disponibles para {game_date}")
            return

        # Actualizar la DB
        updated = 0
        with sqlite3.connect(self.DB_PATH) as con:
            for pred_id, home_team, away_team, ou_line in rows:
                # Buscar el resultado: intentar match directo o fuzzy
                result = self._find_result(home_team, away_team, results_by_teams)
                if result is None:
                    print(f"  [BetTracker] No se encontró resultado para {home_team} vs {away_team}")
                    continue

                # Determinar OVER/UNDER
                actual_ou = "OVER" if result["total"] > ou_line else "UNDER"

                # Recuperar predicción del modelo para calcular ml_correct
                pred_row = con.execute(
                    "SELECT prob_home, prob_over FROM predictions WHERE id = ?",
                    (pred_id,)
                ).fetchone()

                ml_predicted_winner = "HOME" if pred_row[0] >= 0.5 else "AWAY"
                ou_predicted = "OVER" if pred_row[1] >= 0.5 else "UNDER"
                ml_correct = 1 if ml_predicted_winner == result["winner"] else 0
                ou_correct = 1 if ou_predicted == actual_ou else 0

                # AH tracking: compute actual margin residual
                spread_row = con.execute(
                    "SELECT spread FROM predictions WHERE id = ?", (pred_id,)
                ).fetchone()
                ah_margin = result["home_score"] - result["away_score"]
                ah_spread = float(spread_row[0]) if spread_row and spread_row[0] is not None else 0.0
                ah_residual = ah_margin + ah_spread
                ah_actual_cover = 1 if ah_residual > 0 else (None if ah_residual == 0 else 0)

                con.execute("""
                    UPDATE predictions SET
                        actual_home_score = ?,
                        actual_away_score = ?,
                        actual_total = ?,
                        actual_winner = ?,
                        actual_ou = ?,
                        ml_correct = ?,
                        ou_correct = ?,
                        ah_actual_cover = ?,
                        ah_residual = ?
                    WHERE id = ?
                """, (
                    result["home_score"], result["away_score"], result["total"],
                    result["winner"], actual_ou,
                    ml_correct, ou_correct,
                    ah_actual_cover, ah_residual,
                    pred_id,
                ))
                updated += 1

            con.commit()

        print(f"  [BetTracker] {updated}/{len(rows)} resultados actualizados para {game_date}")

        # Auto-calcular CLV si hay closing lines disponibles (Fase 5.1)
        try:
            from src.core.betting.clv import update_clv_for_predictions
            clv_updated = update_clv_for_predictions(game_date)
            if clv_updated > 0:
                print(f"  [BetTracker] CLV calculado para {clv_updated} predicciones")
        except Exception as e:
            print(f"  [BetTracker] CLV no disponible: {e}")

    @staticmethod
    def _fetch_results_espn(game_date: str) -> dict:
        """Descarga scores desde ESPN API (publica, sin auth, rapida).

        ESPN scoreboard endpoint retorna todos los juegos de una fecha
        con home/away teams y scores finales.
        """
        import urllib.request
        import json

        # ESPN format: YYYYMMDD
        dt = game_date.replace("-", "")
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={dt}"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            print(f"  [BetTracker] ESPN API error: {e}")
            return {}

        results = {}
        for event in data.get("events", []):
            competition = event.get("competitions", [{}])[0]
            competitors = competition.get("competitors", [])
            if len(competitors) != 2:
                continue

            # ESPN: competitors[0] is usually home (homeAway field confirms)
            home = away = None
            for c in competitors:
                team_name = c.get("team", {}).get("displayName", "")
                score = int(c.get("score", 0))
                if c.get("homeAway") == "home":
                    home = {"name": team_name, "score": score}
                else:
                    away = {"name": team_name, "score": score}

            if not home or not away:
                continue

            # Solo juegos finalizados
            status = competition.get("status", {}).get("type", {}).get("completed", False)
            if not status:
                continue

            results[(home["name"], away["name"])] = {
                "home_score": home["score"],
                "away_score": away["score"],
                "total": home["score"] + away["score"],
                "winner": "HOME" if home["score"] > away["score"] else "AWAY",
            }

        if results:
            print(f"  [BetTracker] ESPN: {len(results)} juegos finalizados")
        return results

    @staticmethod
    def _fetch_results_nba_api(game_date: str, leaguegamefinder) -> dict:
        """Fallback: descarga scores desde stats.nba.com via nba_api."""
        import time
        try:
            time.sleep(0.6)
            finder = leaguegamefinder.LeagueGameFinder(
                date_from_nullable=game_date,
                date_to_nullable=game_date,
                league_id_nullable="00"
            )
            games_df = finder.get_data_frames()[0]
        except Exception as e:
            print(f"  [BetTracker] nba_api error: {e}")
            return {}

        if games_df.empty:
            return {}

        game_scores = {}
        for _, row in games_df.iterrows():
            gid = row["GAME_ID"]
            if gid not in game_scores:
                game_scores[gid] = {}
            game_scores[gid][row["TEAM_ABBREVIATION"]] = {
                "pts": int(row["PTS"]) if row["PTS"] is not None else 0,
                "wl": row["WL"],
            }

        results = {}
        for gid, teams in game_scores.items():
            if len(teams) != 2:
                continue
            home_abbrev = None
            for _, row in games_df[games_df["GAME_ID"] == gid].iterrows():
                if "vs." in str(row.get("MATCHUP", "")):
                    home_abbrev = row["TEAM_ABBREVIATION"]
                    break
            if home_abbrev is None:
                continue

            away_abbrev = [a for a in list(teams.keys()) if a != home_abbrev][0]
            home_pts = teams[home_abbrev]["pts"]
            away_pts = teams[away_abbrev]["pts"]
            home_full = ABBREV_TO_FULL.get(home_abbrev, home_abbrev)
            away_full = ABBREV_TO_FULL.get(away_abbrev, away_abbrev)

            results[(home_full, away_full)] = {
                "home_score": home_pts,
                "away_score": away_pts,
                "total": home_pts + away_pts,
                "winner": "HOME" if home_pts > away_pts else "AWAY",
            }

        if results:
            print(f"  [BetTracker] nba_api: {len(results)} juegos encontrados")
        return results

    def _find_result(self, home_team: str, away_team: str, results: dict) -> dict | None:
        """Busca el resultado para un partido usando match exacto o fuzzy.

        El fuzzy matching resuelve diferencias menores de nombre:
        "LA Clippers" vs "Los Angeles Clippers", etc.
        """
        # Intento 1: match exacto
        if (home_team, away_team) in results:
            return results[(home_team, away_team)]

        # Intento 2: fuzzy matching sobre los nombres disponibles
        all_home_teams = [k[0] for k in results.keys()]
        all_away_teams = [k[1] for k in results.keys()]

        home_matches = difflib.get_close_matches(home_team, all_home_teams, n=1, cutoff=0.85)
        away_matches = difflib.get_close_matches(away_team, all_away_teams, n=1, cutoff=0.85)

        if home_matches and away_matches:
            key = (home_matches[0], away_matches[0])
            if key in results:
                return results[key]

        return None

    def print_report(self, start_date: str = None, end_date: str = None):
        """Imprime un resumen de accuracy y P&L simulado del modelo.

        P&L simulado: asume que apostamos Quarter-Kelly en todas las
        apuestas donde EV > 0 (edge positivo del modelo).

        Args:
            start_date: 'YYYY-MM-DD', por defecto inicio de todo el historial
            end_date: 'YYYY-MM-DD', por defecto hoy
        """
        with sqlite3.connect(self.DB_PATH) as con:
            # Construir filtro de fechas
            where_clauses = ["ml_correct IS NOT NULL"]
            params = []
            if start_date:
                where_clauses.append("game_date >= ?")
                params.append(start_date)
            if end_date:
                where_clauses.append("game_date <= ?")
                params.append(end_date)
            where = " AND ".join(where_clauses)

            # Estadísticas globales de accuracy
            stats = con.execute(f"""
                SELECT
                    COUNT(*) as n_games,
                    SUM(ml_correct) as ml_wins,
                    SUM(ou_correct) as ou_wins,
                    MIN(game_date) as first_date,
                    MAX(game_date) as last_date
                FROM predictions
                WHERE {where}
            """, params).fetchone()

            if not stats or stats[0] == 0:
                print("  [BetTracker] Sin predicciones con resultados en el rango dado.")
                return

            n_games, ml_wins, ou_wins, first_date, last_date = stats
            ml_acc = ml_wins / n_games * 100
            ou_acc = ou_wins / n_games * 100

            print(f"\n{'='*55}")
            print(f"  REPORTE DE PREDICCIONES: {first_date} → {last_date}")
            print(f"{'='*55}")
            print(f"  Partidos totales:   {n_games}")
            print(f"  ML Accuracy:        {ml_wins}/{n_games} = {ml_acc:.1f}%")
            print(f"  O/U Accuracy:       {ou_wins}/{n_games} = {ou_acc:.1f}%")

            # P&L simulado: solo apuestas con EV > 0 (edge positivo)
            # Asumimos 1 unidad base. Si Kelly dice 3%, apostamos 3 unidades.
            # Ganancia: unidades * (odds/100) si +odds, o unidades * (100/|odds|) si -odds
            bets = con.execute(f"""
                SELECT
                    home_team, away_team, game_date,
                    prob_home, prob_away,
                    ev_home, ev_away,
                    kelly_home, kelly_away,
                    ml_home_odds, ml_away_odds,
                    actual_winner
                FROM predictions
                WHERE {where}
            """, params).fetchall()

            # Simular P&L en apuestas con EV > 0
            # Kelly % se usa como "unidades" (3.0% → 3.0 units)
            # P&L es relativo: si tienes $1000 bankroll, 1 unit = $10
            pnl = 0.0
            n_bets = 0
            n_wins = 0
            for row in bets:
                (home, away, date, ph, pa, ev_h, ev_a, kelly_h, kelly_a,
                 odds_h, odds_a, winner) = row

                # Solo apostar al MEJOR lado (no ambos — ML es binario)
                bet_home = (ev_h or 0) > 0 and (kelly_h or 0) > 0
                bet_away = (ev_a or 0) > 0 and (kelly_a or 0) > 0

                if bet_home and bet_away:
                    # Ambos con EV > 0: apostar solo al de mayor EV
                    if ev_h >= ev_a:
                        bet_away = False
                    else:
                        bet_home = False

                if bet_home:
                    units = kelly_h
                    n_bets += 1
                    if winner == "HOME":
                        pnl += _calc_payout(odds_h, units)
                        n_wins += 1
                    else:
                        pnl -= units

                if bet_away:
                    units = kelly_a
                    n_bets += 1
                    if winner == "AWAY":
                        pnl += _calc_payout(odds_a, units)
                        n_wins += 1
                    else:
                        pnl -= units

            if n_bets > 0:
                print(f"\n  P&L SIMULADO (EV > 0, Quarter-Kelly):")
                print(f"  Apuestas totales:  {n_bets}")
                print(f"  Ganadas:           {n_wins} ({n_wins/n_bets*100:.1f}%)")
                print(f"  P&L total:         {pnl:+.2f} unidades")
                roi = pnl / n_bets * 100
                print(f"  ROI por apuesta:   {roi:+.1f}%")
            else:
                print(f"\n  Sin apuestas con EV > 0 en el rango.")

            print(f"{'='*55}\n")

    def get_predictions_df(self, game_date: str = None):
        """Retorna predicciones como DataFrame de pandas (útil para análisis ad-hoc).

        Args:
            game_date: si se especifica, solo retorna predicciones de esa fecha
        """
        import pandas as pd
        with sqlite3.connect(self.DB_PATH) as con:
            if game_date:
                return pd.read_sql(
                    "SELECT * FROM predictions WHERE game_date = ? ORDER BY id",
                    con, params=(game_date,)
                )
            return pd.read_sql(
                "SELECT * FROM predictions ORDER BY game_date, id",
                con
            )


def _calc_payout(american_odds: int, units: float) -> float:
    """Calcula la ganancia neta en unidades dado un bet ganador.

    Ejemplo:
        odds = -130, units = 3.0 → ganancia = 3.0 * (100/130) = 2.31
        odds = +110, units = 2.0 → ganancia = 2.0 * (110/100) = 2.20
    """
    if american_odds is None:
        return 0.0
    if american_odds > 0:
        return units * (american_odds / 100.0)
    else:
        return units * (100.0 / abs(american_odds))
