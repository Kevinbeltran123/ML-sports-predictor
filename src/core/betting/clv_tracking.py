"""CLV (Closing Line Value) Tracking — la metrica #1 de skill en betting.

POR QUE IMPORTA:
================
La closing line es el precio MAS EFICIENTE del mercado porque incorpora toda la
informacion disponible: sharp money, injuries, lineups, public action. Si tus
apuestas consistentemente vencen la closing line (CLV positivo), tienes edge real
independientemente de resultados a corto plazo.

- CLV > 0 de forma consistente = skill genuino (sharp bettor)
- CLV ~ 0 = sin edge (recio justo menos vig)
- CLV < 0 = las casas te estan ganando

COMO SE CALCULA:
================
1. Al momento de apostar, registramos los odds que obtuvimos (opening price)
2. Justo antes de tipoff, la closing line establece el "precio verdadero"
3. CLV = probabilidad_implicita_cierre - probabilidad_implicita_apertura
   Positivo = "compramos" la apuesta a mejor precio que el mercado final

Ejemplo: Si apostamos HOME a -150 (60.0% implicito) y la closing line es -170
(62.96%), nuestro CLV = 62.96% - 60.0% = +2.96pp. Obtuvimos un precio mejor
que el mercado eficiente.

FUENTE:
=======
ESPNLines.sqlite — consolidado de 4 temporadas (2022-23 a 2025-26) con
open/close lines de multiples sportsbooks. Tabla consensus = mediana across
providers (precio mas robusto).
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import BETS_DB, ESPN_LINES_DB, get_logger

logger = get_logger(__name__)


def _american_to_implied(odds: float) -> Optional[float]:
    """Convierte odds americanos a probabilidad implicita (con vig).

    - Negativo (favorito): -150 → 150/(150+100) = 60.0%
    - Positivo (underdog): +130 → 100/(130+100) = 43.5%
    """
    if odds is None or np.isnan(odds):
        return None
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    elif odds > 0:
        return 100.0 / (odds + 100.0)
    return None


def _devig_pair(p_home: float, p_away: float) -> tuple[float, float]:
    """Remueve el vig de un par de probabilidades implicitas.

    El vig (vigorish) es la comision de la casa. Las probabilidades
    implicitas suman > 100% (ej: 52% + 52% = 104%, el 4% es vig).
    Devig = normalizar para que sumen exactamente 100%.

    Usamos el metodo "multiplicativo" (dividir por overround total),
    que es el mas comun y funciona bien para ML de NBA.
    """
    total = p_home + p_away
    if total <= 0:
        return 0.5, 0.5
    return p_home / total, p_away / total


def load_closing_lines(season: str = "2025-26") -> pd.DataFrame:
    """Carga closing lines consensus para una temporada.

    Returns:
        DataFrame con columnas: game_date, home_team, away_team,
        close_spread, close_ml_prob, close_total
    """
    if not ESPN_LINES_DB.exists():
        logger.warning("ESPNLines.sqlite no existe. Ejecutar build_espn_lines_db.py")
        return pd.DataFrame()

    con = sqlite3.connect(ESPN_LINES_DB)
    table = f"espn_consensus_{season}"

    try:
        df = pd.read_sql_query(f'SELECT * FROM "{table}"', con)
    except Exception as e:
        logger.error("Error leyendo tabla %s: %s", table, e)
        return pd.DataFrame()
    finally:
        con.close()

    logger.info("Closing lines %s: %d juegos", season, len(df))
    return df


def load_all_closing_lines() -> pd.DataFrame:
    """Carga closing lines de TODAS las temporadas disponibles."""
    if not ESPN_LINES_DB.exists():
        return pd.DataFrame()

    con = sqlite3.connect(ESPN_LINES_DB)
    tables = [
        r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'espn_consensus_%'"
        ).fetchall()
    ]

    frames = []
    for table in tables:
        season = table.replace("espn_consensus_", "")
        df = pd.read_sql_query(f'SELECT * FROM "{table}"', con)
        df["season"] = season
        frames.append(df)

    con.close()

    if not frames:
        return pd.DataFrame()

    all_df = pd.concat(frames, ignore_index=True)
    logger.info("Closing lines totales: %d juegos de %d temporadas", len(all_df), len(frames))
    return all_df


def fill_closing_lines_in_bets() -> dict:
    """Llena las columnas closing_ml_*_odds y clv_* en BetsTracking.

    Matching: (game_date, home_team) con fallback a date+1 (UTC offset).
    ESPN usa fecha UTC (event_date_utc), pero BetsTracking usa fecha local ET.
    Un juego a las 7:30PM ET del 11 feb es 00:30 UTC del 12 feb.
    Por eso probamos ambas fechas.

    Returns:
        dict con stats del proceso (matched, unmatched, already_filled)
    """
    if not BETS_DB.exists() or not ESPN_LINES_DB.exists():
        logger.error("Falta BetsTracking.sqlite o ESPNLines.sqlite")
        return {"error": "DB not found"}

    # Cargar closing lines
    closing = load_all_closing_lines()
    if closing.empty:
        return {"error": "No closing lines found"}

    # Crear lookup: (date, home) → row
    # Usamos home_team + date como key unica (un equipo no juega 2 veces
    # de local el mismo dia)
    closing_lookup = {}
    for _, row in closing.iterrows():
        key = (row["game_date"], row["home_team"])
        closing_lookup[key] = row

    # Conectar a BetsTracking
    con = sqlite3.connect(BETS_DB)
    predictions = pd.read_sql_query("SELECT * FROM predictions", con)

    stats = {"matched": 0, "unmatched": 0, "already_filled": 0, "total": len(predictions)}

    for idx, pred in predictions.iterrows():
        # Si ya tiene closing lines, saltar
        if pd.notna(pred.get("closing_ml_home_odds")):
            stats["already_filled"] += 1
            continue

        # Intentar match exacto primero, luego date+1 (UTC offset)
        key = (pred["game_date"], pred["home_team"])
        closing_row = closing_lookup.get(key)

        if closing_row is None:
            # Fallback: ESPN usa fecha UTC, BetsTracking usa fecha local ET.
            # Los juegos nocturnos ET aparecen como dia+1 en UTC.
            next_day = (
                datetime.strptime(pred["game_date"], "%Y-%m-%d") + timedelta(days=1)
            ).strftime("%Y-%m-%d")
            key_next = (next_day, pred["home_team"])
            closing_row = closing_lookup.get(key_next)

        if closing_row is None:
            stats["unmatched"] += 1
            continue

        # Calcular closing ML odds desde close_spread (aproximacion)
        # Usamos consensus_close_ml_prob directamente (ya esta devigged)
        close_prob = closing_row.get("consensus_close_ml_prob")
        close_spread = closing_row.get("consensus_close_spread")

        if close_prob is None or np.isnan(close_prob):
            stats["unmatched"] += 1
            continue

        # Convertir probabilidad devigged a odds americanos aproximados
        # (para almacenar en el formato existente de la tabla)
        close_home_odds = _prob_to_american(close_prob)
        close_away_odds = _prob_to_american(1.0 - close_prob)

        # Calcular CLV: P(closing) - P(opening_implied_devigged)
        # P(opening) la calculamos desde los odds que obtuvimos al apostar
        opening_home_implied = _american_to_implied(pred["ml_home_odds"])
        opening_away_implied = _american_to_implied(pred["ml_away_odds"])

        clv_home = None
        clv_away = None

        if opening_home_implied and opening_away_implied:
            # Devig los opening odds del bettor
            open_home_devig, open_away_devig = _devig_pair(opening_home_implied, opening_away_implied)
            # CLV = closing_fair_prob - opening_fair_prob
            # Positivo = compramos a mejor precio que el cierre
            clv_home = round(close_prob - open_home_devig, 6)
            clv_away = round((1.0 - close_prob) - open_away_devig, 6)

        # Update en la BD
        con.execute(
            """UPDATE predictions SET
                closing_ml_home_odds = ?,
                closing_ml_away_odds = ?,
                clv_home = ?,
                clv_away = ?
            WHERE id = ?""",
            (close_home_odds, close_away_odds, clv_home, clv_away, pred["id"]),
        )
        stats["matched"] += 1

    con.commit()
    con.close()

    logger.info(
        "CLV fill: %d matched, %d unmatched, %d already filled (de %d total)",
        stats["matched"], stats["unmatched"], stats["already_filled"], stats["total"],
    )
    return stats


def _prob_to_american(prob: float) -> int:
    """Convierte probabilidad (0-1) a odds americanos.

    - prob > 50% → favorito → odds negativos (ej: 60% → -150)
    - prob < 50% → underdog → odds positivos (ej: 40% → +150)
    """
    if prob >= 1.0:
        return -10000
    if prob <= 0.0:
        return 10000
    if prob >= 0.5:
        return int(round(-100 * prob / (1.0 - prob)))
    else:
        return int(round(100 * (1.0 - prob) / prob))


def compute_clv_summary(min_date: Optional[str] = None) -> dict:
    """Calcula resumen de CLV para todas las predicciones con closing lines.

    El resumen incluye:
    - clv_mean: CLV promedio (esperamos > 0 para skill)
    - clv_median: CLV mediana
    - pct_positive: % de apuestas con CLV > 0
    - n_games: numero de juegos evaluados
    - clv_by_sportsbook: CLV promedio por sportsbook

    Args:
        min_date: filtrar desde esta fecha (YYYY-MM-DD), o None para todo

    Returns:
        dict con metricas CLV
    """
    if not BETS_DB.exists():
        return {"error": "BetsTracking.sqlite not found"}

    con = sqlite3.connect(BETS_DB)

    query = "SELECT * FROM predictions WHERE clv_home IS NOT NULL"
    if min_date:
        query += f" AND game_date >= '{min_date}'"

    df = pd.read_sql_query(query, con)
    con.close()

    if df.empty:
        return {"error": "No predictions with CLV data. Run fill_closing_lines_in_bets() first."}

    # Para cada juego, el CLV relevante es el del lado que apostamos
    # Asumimos que si edge_home > 0, apostamos home (CLV = clv_home)
    # Si edge_away > 0, apostamos away (CLV = clv_away)
    clv_values = []
    bet_sides = []
    for _, row in df.iterrows():
        if row.get("edge_home", 0) > row.get("edge_away", 0):
            clv_values.append(row["clv_home"])
            bet_sides.append("home")
        else:
            clv_values.append(row["clv_away"])
            bet_sides.append("away")

    clv_arr = np.array(clv_values)
    # Filtrar NaN
    valid = ~np.isnan(clv_arr)
    clv_arr = clv_arr[valid]

    if len(clv_arr) == 0:
        return {"error": "No valid CLV values"}

    summary = {
        "n_games": len(clv_arr),
        "clv_mean_pp": round(float(np.mean(clv_arr) * 100), 2),     # en puntos porcentuales
        "clv_median_pp": round(float(np.median(clv_arr) * 100), 2),
        "clv_std_pp": round(float(np.std(clv_arr) * 100), 2),
        "pct_positive_clv": round(float(np.mean(clv_arr > 0) * 100), 1),
        "date_range": f"{df['game_date'].min()} to {df['game_date'].max()}",
    }

    # CLV por sportsbook
    df["clv_bet"] = clv_values
    by_book = df.groupby("sportsbook")["clv_bet"].agg(["mean", "count"])
    summary["by_sportsbook"] = {
        book: {"mean_pp": round(float(row["mean"] * 100), 2), "n": int(row["count"])}
        for book, row in by_book.iterrows()
    }

    # Correlacion CLV con resultado (deberia ser positiva si CLV predice ganancias)
    if "ml_correct" in df.columns:
        valid_correct = df["ml_correct"].notna()
        if valid_correct.sum() > 10:
            from scipy.stats import pearsonr
            r, p = pearsonr(
                df.loc[valid_correct, "clv_bet"],
                df.loc[valid_correct, "ml_correct"],
            )
            summary["clv_result_correlation"] = round(float(r), 3)
            summary["clv_result_pvalue"] = round(float(p), 4)

    return summary


def print_clv_report(min_date: Optional[str] = None):
    """Imprime reporte formateado de CLV."""
    summary = compute_clv_summary(min_date)

    if "error" in summary:
        logger.error(summary["error"])
        return

    print("\n" + "=" * 60)
    print("  CLV (Closing Line Value) Report")
    print("=" * 60)
    print(f"  Periodo: {summary['date_range']}")
    print(f"  Juegos evaluados: {summary['n_games']}")
    print()
    print(f"  CLV promedio:  {summary['clv_mean_pp']:+.2f} pp")
    print(f"  CLV mediana:   {summary['clv_median_pp']:+.2f} pp")
    print(f"  CLV std dev:   {summary['clv_std_pp']:.2f} pp")
    print(f"  % CLV positivo: {summary['pct_positive_clv']:.1f}%")
    print()

    # Interpretacion
    clv = summary["clv_mean_pp"]
    if clv > 1.0:
        verdict = "SHARP — edge significativo sobre el mercado"
    elif clv > 0:
        verdict = "POSITIVO — ligero edge sobre el cierre"
    elif clv > -1.0:
        verdict = "NEUTRAL — sin edge claro"
    else:
        verdict = "NEGATIVO — el mercado cierra en tu contra"
    print(f"  Veredicto: {verdict}")

    if "clv_result_correlation" in summary:
        print(f"\n  Correlacion CLV ↔ resultado: r={summary['clv_result_correlation']}, p={summary['clv_result_pvalue']}")

    if summary.get("by_sportsbook"):
        print("\n  Por sportsbook:")
        for book, stats in summary["by_sportsbook"].items():
            print(f"    {book}: {stats['mean_pp']:+.2f} pp ({stats['n']} juegos)")

    print("=" * 60)


if __name__ == "__main__":
    # 1. Llenar closing lines en BetsTracking
    print("Llenando closing lines en BetsTracking...")
    stats = fill_closing_lines_in_bets()
    print(f"Resultado: {stats}")

    # 2. Generar reporte CLV
    print_clv_report()
