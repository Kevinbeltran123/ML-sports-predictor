"""CLV (Closing Line Value): mide si capturas valor antes del mercado.

Concepto clave:
    La linea de cierre (justo antes del tip-off) es el mejor predictor
    conocido del resultado real. Es el consenso del mercado despues de que
    todo el "smart money" ha apostado.

    Si TU apuestas a -135 por la manana y la linea cierra en -150,
    significa que el mercado se movio en tu direccion. Tu capturaste
    valor que el mercado no habia reconocido aun.

    CLV = implied_prob(closing) - implied_prob(opening)
      CLV > 0: capturaste valor (linea se movio a tu favor)
      CLV < 0: el mercado se movio en contra (apostaste en el lado equivocado)
      CLV = 0: la linea no se movio

    CLV > 0 sostenido es la MEJOR evidencia de edge real, mucho mejor
    que el W/L record porque no depende de varianza.

Ejemplo numerico:
    Opening: Lakers ML -135 → implied = 135/(135+100) = 57.4%
    Closing: Lakers ML -150 → implied = 150/(150+100) = 60.0%
    CLV = 60.0% - 57.4% = +2.6% → capturaste 2.6pp de valor

Usado por:
    - BetTracker.update_results() → auto-calcula CLV si hay closing lines
    - Dashboard → muestra CLV acumulado como metrica de edge
"""

import sqlite3
from datetime import datetime

from src.config import HISTORICAL_LINES_DB as LINES_DB, BETS_DB, get_logger

logger = get_logger(__name__)


def american_to_implied(odds: int) -> float:
    """Convierte odds americanos a probabilidad implicita (con vig).

    Odds americanos:
        Negativo (favorito): -150 → apuestas $150 para ganar $100
        Positivo (underdog): +130 → apuestas $100 para ganar $130

    Conversion:
        Si odds < 0: prob = |odds| / (|odds| + 100)
        Si odds > 0: prob = 100 / (odds + 100)

    Nota: esta probabilidad INCLUYE el vig. Para CLV esto esta bien
    porque ambas lineas (opening y closing) tienen vig similar,
    asi que la diferencia cancela el vig.

    Args:
        odds: odds americanos (ej: -150, +130)

    Returns:
        float: probabilidad implicita (0.0 - 1.0)
    """
    if odds is None:
        return None
    odds = int(odds)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    elif odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return 0.5  # Pick'em


def compute_clv(opening_odds: int, closing_odds: int) -> float:
    """Calcula CLV entre linea de apertura y cierre.

    CLV positivo = capturaste valor (la linea se movio a tu favor).
    CLV negativo = el mercado se movio en contra.

    Args:
        opening_odds: odds americanos al abrir (cuando apostaste)
        closing_odds: odds americanos al cerrar (justo antes del partido)

    Returns:
        float: CLV en porcentaje (ej: 0.026 = 2.6%)
               None si faltan datos
    """
    p_open = american_to_implied(opening_odds)
    p_close = american_to_implied(closing_odds)

    if p_open is None or p_close is None:
        return None

    # CLV = prob implicita del cierre - prob implicita de apertura
    # Si apostaste al favorito y el favorito se hizo MAS favorito → CLV > 0
    return round(p_close - p_open, 5)


def get_closing_lines(game_date: str, sportsbook: str = "fanduel") -> dict:
    """Obtiene las lineas de cierre de equipos para una fecha.

    Busca en team_odds_lines con line_type='closing'.
    Si no hay closing, intenta con 'opening' como fallback (no ideal).

    Args:
        game_date: 'YYYY-MM-DD'
        sportsbook: nombre del sportsbook

    Returns:
        dict[(home_team, away_team)] -> {ml_home, ml_away, spread, ou_total}
    """
    try:
        with sqlite3.connect(LINES_DB) as con:
            # Verificar que la tabla existe
            tables = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='team_odds_lines'"
            ).fetchone()
            if not tables:
                return {}

            rows = con.execute("""
                SELECT home_team, away_team, ml_home, ml_away, spread_home, ou_total
                FROM team_odds_lines
                WHERE game_date = ? AND sportsbook = ? AND line_type = 'closing'
            """, (game_date, sportsbook)).fetchall()

            result = {}
            for home, away, ml_h, ml_a, spread, ou in rows:
                result[(home, away)] = {
                    "ml_home": ml_h,
                    "ml_away": ml_a,
                    "spread": spread,
                    "ou_total": ou,
                }
            return result
    except Exception as e:
        logger.warning("Error leyendo closing lines: %s", e)
        return {}


def get_opening_lines(game_date: str, sportsbook: str = "fanduel") -> dict:
    """Obtiene las lineas de apertura de equipos para una fecha.

    Returns:
        dict[(home_team, away_team)] -> {ml_home, ml_away, spread, ou_total}
    """
    try:
        with sqlite3.connect(LINES_DB) as con:
            tables = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='team_odds_lines'"
            ).fetchone()
            if not tables:
                return {}

            rows = con.execute("""
                SELECT home_team, away_team, ml_home, ml_away, spread_home, ou_total
                FROM team_odds_lines
                WHERE game_date = ? AND sportsbook = ? AND line_type = 'opening'
            """, (game_date, sportsbook)).fetchall()

            result = {}
            for home, away, ml_h, ml_a, spread, ou in rows:
                result[(home, away)] = {
                    "ml_home": ml_h,
                    "ml_away": ml_a,
                    "spread": spread,
                    "ou_total": ou,
                }
            return result
    except Exception as e:
        logger.warning("Error leyendo opening lines: %s", e)
        return {}


def compute_line_movement(game_date: str, sportsbook: str = "fanduel") -> dict:
    """Calcula el movimiento de linea para cada partido del dia.

    LINE_MOVE_ML = closing_ml - opening_ml (en probabilidad implicita)
    LINE_MOVE_SPREAD = closing_spread - opening_spread (en puntos)
    LINE_MOVE_OU = closing_ou - opening_ou (en puntos)

    Ejemplo:
        Opening: spread = -3.5, ML = -150
        Closing: spread = -5.0, ML = -180
        LINE_MOVE_SPREAD = -5.0 - (-3.5) = -1.5 (home se hizo mas favorito)
        LINE_MOVE_ML = impl(-180) - impl(-150) = 0.643 - 0.600 = +0.043

    Returns:
        dict[(home, away)] -> {
            LINE_MOVE_ML_HOME: float (prob space),
            LINE_MOVE_SPREAD: float (puntos),
            LINE_MOVE_OU: float (puntos),
        }
    """
    opening = get_opening_lines(game_date, sportsbook)
    closing = get_closing_lines(game_date, sportsbook)

    if not opening or not closing:
        return {}

    result = {}
    for key in opening:
        if key not in closing:
            continue

        o = opening[key]
        c = closing[key]

        # Movimiento de ML en espacio de probabilidad
        ml_move = compute_clv(o["ml_home"], c["ml_home"])

        # Movimiento de spread en puntos
        spread_move = None
        if o["spread"] is not None and c["spread"] is not None:
            spread_move = round(c["spread"] - o["spread"], 1)

        # Movimiento de O/U en puntos
        ou_move = None
        if o["ou_total"] is not None and c["ou_total"] is not None:
            ou_move = round(c["ou_total"] - o["ou_total"], 1)

        result[key] = {
            "LINE_MOVE_ML_HOME": ml_move,
            "LINE_MOVE_SPREAD": spread_move,
            "LINE_MOVE_OU": ou_move,
        }

    return result


def update_clv_for_predictions(game_date: str, sportsbook: str = "fanduel"):
    """Actualiza CLV en BetsTracking.sqlite usando closing lines.

    Busca predicciones del dia en BetsTracking que no tengan CLV calculado,
    luego busca closing lines en HistoricalLines.sqlite y calcula CLV.

    Returns:
        int: numero de predicciones actualizadas
    """
    closing = get_closing_lines(game_date, sportsbook)
    if not closing:
        logger.debug("Sin closing lines para %s — CLV no calculado", game_date)
        return 0

    updated = 0
    with sqlite3.connect(BETS_DB) as con:
        # Verificar que la tabla existe
        tables = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'"
        ).fetchone()
        if not tables:
            return 0

        # Obtener predicciones sin CLV
        rows = con.execute("""
            SELECT id, home_team, away_team, ml_home_odds, ml_away_odds
            FROM predictions
            WHERE game_date = ? AND sportsbook = ? AND clv_home IS NULL
        """, (game_date, sportsbook)).fetchall()

        for pred_id, home, away, ml_open_h, ml_open_a in rows:
            close_data = closing.get((home, away))
            if close_data is None:
                continue

            ml_close_h = close_data["ml_home"]
            ml_close_a = close_data["ml_away"]

            # CLV para apuesta al home: si closing se hizo mas favorito → CLV > 0
            clv_h = compute_clv(ml_open_h, ml_close_h)
            # CLV para apuesta al away: si closing se hizo mas favorito → CLV > 0
            clv_a = compute_clv(ml_open_a, ml_close_a)

            con.execute("""
                UPDATE predictions SET
                    closing_ml_home_odds = ?,
                    closing_ml_away_odds = ?,
                    clv_home = ?,
                    clv_away = ?
                WHERE id = ?
            """, (ml_close_h, ml_close_a, clv_h, clv_a, pred_id))
            updated += 1

        con.commit()

    if updated > 0:
        logger.info("CLV actualizado para %d predicciones (%s)", updated, game_date)
    return updated


def print_clv_report(start_date: str = None, end_date: str = None):
    """Imprime reporte de CLV acumulado.

    CLV promedio > 0 de forma sostenida = evidencia de edge real.
    El benchmark es CLV > 1% para apostadores profesionales.
    """
    with sqlite3.connect(BETS_DB) as con:
        tables = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'"
        ).fetchone()
        if not tables:
            print("  Sin datos de predicciones")
            return

        where = ["clv_home IS NOT NULL"]
        params = []
        if start_date:
            where.append("game_date >= ?")
            params.append(start_date)
        if end_date:
            where.append("game_date <= ?")
            params.append(end_date)

        where_sql = " AND ".join(where)

        stats = con.execute(f"""
            SELECT
                COUNT(*) as n,
                AVG(clv_home) as avg_clv_home,
                AVG(clv_away) as avg_clv_away,
                SUM(CASE WHEN clv_home > 0 THEN 1 ELSE 0 END) as clv_home_pos,
                SUM(CASE WHEN clv_away > 0 THEN 1 ELSE 0 END) as clv_away_pos,
                MIN(game_date) as first_date,
                MAX(game_date) as last_date
            FROM predictions
            WHERE {where_sql}
        """, params).fetchone()

        if not stats or stats[0] == 0:
            print("  Sin datos de CLV. Ejecuta collect_daily_lines.py --type closing")
            return

        n, avg_h, avg_a, pos_h, pos_a, first, last = stats

        print(f"\n{'='*55}")
        print(f"  REPORTE CLV: {first} -> {last}")
        print(f"{'='*55}")
        print(f"  Partidos con CLV:  {n}")
        print(f"  CLV Home promedio: {avg_h*100:+.2f}%")
        print(f"  CLV Away promedio: {avg_a*100:+.2f}%")
        print(f"  CLV Home positivo: {pos_h}/{n} ({pos_h/n*100:.1f}%)")
        print(f"  CLV Away positivo: {pos_a}/{n} ({pos_a/n*100:.1f}%)")
        print()

        # Interpretacion
        avg_combined = (avg_h + avg_a) / 2
        if avg_combined > 0.01:
            print("  -> CLV > 1%: Excelente. Edge real demostrado.")
        elif avg_combined > 0:
            print("  -> CLV > 0%: Positivo. Hay senales de edge.")
        else:
            print("  -> CLV < 0%: El mercado se mueve en contra. Revisar modelo.")
        print(f"{'='*55}\n")
