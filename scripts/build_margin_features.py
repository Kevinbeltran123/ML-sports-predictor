"""Construye dataset enriquecido para el modelo de margen.

Parte del dataset base (dataset_2012-26) y agrega features de interaccion
especificas para prediccion de margen:

  - NET_RTG_DIFF    : diferencia de net rating (home - away), predictor directo
  - EFG_NET_HOME    : EFG_HOME - EFG_AWAY (shooting edge)
  - BPM_GAP         : ADV_MAX_BPM_HOME - ADV_MAX_BPM_AWAY (star power gap)
  - PACE_NET_FACTOR : FF_PACE_HOME * Net_Rtg / 100 (pace amplifies net rating)
  - AVAIL_BPM_CROSS : AVAIL_QUALITY_HOME * MISSING_BPM_AWAY (healthy home vs injured away)
  - BENCH_DEPTH_NET : BENCH_DEPTH_HOME - BENCH_DEPTH_AWAY (depth differential)
  - LS_Q4_NET       : LS_Q4_PCT_HOME - LS_Q4_PCT_AWAY (late-game scoring gap)
  - ONOFF_NET_GAP   : ONOFF_NET_TOP5_HOME - ONOFF_NET_TOP5_AWAY (lineup net rating)
  - ESPN_MOVE_ABS   : abs(ESPN_LINE_MOVE) (magnitude of line move, regardless of direction)

Guarda como tabla "dataset_margin_enriched" en el mismo dataset.sqlite.

Uso:
    PYTHONPATH=. python scripts/build_margin_features.py
    PYTHONPATH=. python scripts/build_margin_features.py --dataset dataset_2012-26
"""

import argparse
import sqlite3

import numpy as np
import pandas as pd

from src.config import DATASET_DB, get_logger
from src.sports.nba.features.ats_features import build_ats_lookup, add_ats_to_frame

logger = get_logger(__name__)

DEFAULT_SOURCE = "dataset_2012-26"
DEFAULT_TARGET = "dataset_margin_enriched"


def _safe_diff(df: pd.DataFrame, col_home: str, col_away: str) -> pd.Series:
    """Compute home - away difference if both columns exist, else 0."""
    if col_home in df.columns and col_away in df.columns:
        return df[col_home].fillna(0).astype(float) - df[col_away].fillna(0).astype(float)
    logger.debug("Columns not found for diff: %s / %s", col_home, col_away)
    return pd.Series(0.0, index=df.index)


def add_margin_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega features de interaccion especificas para prediccion de margen.

    No modifica columnas existentes — solo agrega columnas nuevas.
    Columnas existentes se preservan intactas.

    Returns:
        DataFrame con columnas adicionales.
    """
    df = df.copy()
    n_before = len(df.columns)

    # 1. Net rating differential — predictor mas directo de margen
    #    Net_Rtg = ORTG - DRTG del home; Net_Rtg.1 = del away
    if "Net_Rtg" in df.columns and "Net_Rtg.1" in df.columns:
        df["NET_RTG_DIFF"] = df["Net_Rtg"].fillna(0).astype(float) - df["Net_Rtg.1"].fillna(0).astype(float)
    else:
        df["NET_RTG_DIFF"] = 0.0

    # 2. EFG differential — shooting efficiency edge
    if "FF_EFG_HOME" in df.columns and "FF_EFG_AWAY" in df.columns:
        df["EFG_NET_HOME"] = (
            df["FF_EFG_HOME"].fillna(0).astype(float) - df["FF_EFG_AWAY"].fillna(0).astype(float)
        )
    else:
        df["EFG_NET_HOME"] = _safe_diff(df, "eFG_PCT", "eFG_PCT.1")

    # 3. Star power gap — best player BPM difference
    df["BPM_GAP"] = _safe_diff(df, "ADV_MAX_BPM_HOME", "ADV_MAX_BPM_AWAY")

    # 4. Pace × Net Rating — pace amplifies net rating advantage
    #    High pace with positive net rating = larger margins
    if "FF_PACE_HOME" in df.columns and "Net_Rtg" in df.columns:
        pace = df["FF_PACE_HOME"].fillna(100.0).astype(float)
        net = df.get("NET_RTG_DIFF", pd.Series(0.0, index=df.index))
        df["PACE_NET_FACTOR"] = pace * net / 100.0
    else:
        df["PACE_NET_FACTOR"] = 0.0

    # 5. Availability × opponent BPM missing (healthy home vs injured away)
    if "AVAIL_QUALITY_HOME" in df.columns and "MISSING_BPM_AWAY" in df.columns:
        df["AVAIL_BPM_CROSS"] = (
            df["AVAIL_QUALITY_HOME"].fillna(1.0).astype(float)
            * df["MISSING_BPM_AWAY"].fillna(0).astype(float)
        )
    else:
        df["AVAIL_BPM_CROSS"] = 0.0

    # 6. Bench depth differential
    df["BENCH_DEPTH_NET"] = _safe_diff(df, "BENCH_DEPTH_HOME", "BENCH_DEPTH_AWAY")

    # 7. Late-game scoring gap (Q4 % of team's own total)
    df["LS_Q4_NET"] = _safe_diff(df, "LS_Q4_PCT_HOME", "LS_Q4_PCT_AWAY")

    # 8. On/off lineup net rating gap
    df["ONOFF_NET_GAP"] = _safe_diff(df, "ONOFF_NET_TOP5_HOME", "ONOFF_NET_TOP5_AWAY")

    # 9. ESPN line move magnitude — large moves signal news (injury/lineup)
    if "ESPN_LINE_MOVE" in df.columns:
        df["ESPN_MOVE_ABS"] = pd.to_numeric(df["ESPN_LINE_MOVE"], errors="coerce").abs().fillna(0)
    else:
        df["ESPN_MOVE_ABS"] = 0.0

    # 10. Scoring variance gap — high variance team → wider range of margins
    df["SCORING_VAR_NET"] = _safe_diff(df, "LS_SCORING_VAR_HOME", "LS_SCORING_VAR_AWAY")

    n_added = len(df.columns) - n_before
    logger.info("Margin interactions: %d features added", n_added)
    return df


def build_enriched_dataset(source_table: str, target_table: str) -> None:
    """Carga el dataset base, agrega interacciones, guarda como tabla nueva.

    Args:
        source_table: nombre de la tabla fuente en dataset.sqlite
        target_table: nombre de la tabla destino
    """
    print(f"\n{'='*60}")
    print(f"  BUILD MARGIN ENRICHED DATASET")
    print(f"  Source: {source_table} → Target: {target_table}")
    print(f"{'='*60}")

    with sqlite3.connect(DATASET_DB) as con:
        df = pd.read_sql_query(f'SELECT * FROM "{source_table}"', con)

    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Add ATS features if not already present in the base dataset
    if "ATS_RATE_HOME" not in df.columns:
        print("  Adding ATS features (rolling 20-game cover rate + streak)...")
        ats_lookup = build_ats_lookup()
        if ats_lookup:
            df = add_ats_to_frame(df, ats_lookup)
            print(f"  ATS features added: {len(ats_lookup)} teams")
        else:
            print("  ATS: no data found, using defaults")
            for col in ["ATS_RATE_HOME", "ATS_RATE_AWAY"]:
                df[col] = 0.5
            for col in ["ATS_STREAK_HOME", "ATS_STREAK_AWAY"]:
                df[col] = 0.0

    df = add_margin_interactions(df)

    print(f"  After interactions: {len(df.columns)} columns")
    new_cols = ["NET_RTG_DIFF", "EFG_NET_HOME", "BPM_GAP", "PACE_NET_FACTOR",
                "AVAIL_BPM_CROSS", "BENCH_DEPTH_NET", "LS_Q4_NET", "ONOFF_NET_GAP",
                "ESPN_MOVE_ABS", "SCORING_VAR_NET"]
    print(f"  New features: {new_cols}")

    # Sanity check — no all-zero columns
    for col in new_cols:
        if col in df.columns:
            nonzero = (df[col] != 0).sum()
            pct = nonzero / len(df) * 100
            print(f"    {col}: {nonzero:,} non-zero ({pct:.0f}%)")

    with sqlite3.connect(DATASET_DB) as con:
        df.to_sql(target_table, con, if_exists="replace", index=False)
        print(f"\n  Saved '{target_table}' — {len(df):,} rows, {len(df.columns)} columns")

    print(f"\n  Run next:")
    print(f"    PYTHONPATH=. python scripts/tune_margin_optuna.py --n-trials 150")
    print(f"    PYTHONPATH=. python scripts/train_margin_models.py")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Build enriched dataset for margin model")
    parser.add_argument("--dataset", default=DEFAULT_SOURCE, help="Source dataset table name")
    parser.add_argument("--output", default=DEFAULT_TARGET, help="Output table name")
    args = parser.parse_args()
    build_enriched_dataset(args.dataset, args.output)


if __name__ == "__main__":
    main()
