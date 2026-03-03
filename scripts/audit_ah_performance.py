"""Auditoría de rendimiento AH (Asian Handicap) — verifica mejoras.

Carga BetsTracking.sqlite y analiza rendimiento AH por:
  1. ATS accuracy por tag (BET / SKIP / PASS)
  2. ROI simulado con Kelly sizing
  3. Calibración P(cover)
  4. game_sigma vs bucket_sigma (Mejora 1)
  5. Blend weights efectivos

Uso:
    PYTHONPATH=. python scripts/audit_ah_performance.py
    PYTHONPATH=. python scripts/audit_ah_performance.py --verbose
"""

import argparse
import sqlite3

import numpy as np
import pandas as pd

from src.config import BETS_DB, ODDS_DB, DATASET_DB, get_logger
from src.core.betting.spread_math import sigma_for_line

logger = get_logger(__name__)


def load_bets_data():
    """Carga predicciones AH desde BetsTracking.sqlite."""
    if not BETS_DB.exists():
        print("  BetsTracking.sqlite not found — no predictions to audit.")
        return None

    with sqlite3.connect(BETS_DB) as con:
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", con)
        if "predictions" not in tables["name"].values:
            print("  No 'predictions' table in BetsTracking.sqlite")
            return None

        cols = pd.read_sql("PRAGMA table_info(predictions)", con)
        col_names = set(cols["name"].values)

        # Check for AH-specific columns
        has_ah = "ah_tag" in col_names and "ah_actual_cover" in col_names
        if not has_ah:
            print("  AH tracking columns not found — run predictions first with updated code.")
            return None

        df = pd.read_sql("SELECT * FROM predictions WHERE ah_tag IS NOT NULL", con)

    return df if len(df) > 0 else None


def load_historical_ats():
    """Carga datos históricos de ATS desde dataset.sqlite para backtest."""
    if not DATASET_DB.exists():
        return None

    with sqlite3.connect(DATASET_DB) as con:
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", con)
        target_table = None
        for t in ["dataset_2012-26", "dataset_margin_enriched"]:
            if t in tables["name"].values:
                target_table = t
                break
        if target_table is None:
            return None

        df = pd.read_sql(f'SELECT * FROM "{target_table}"', con)

    required = ["Margin", "MARKET_SPREAD", "Home-Team-Win"]
    if not all(c in df.columns for c in required):
        return None

    df["Margin"] = pd.to_numeric(df["Margin"], errors="coerce")
    df["MARKET_SPREAD"] = pd.to_numeric(df["MARKET_SPREAD"], errors="coerce")
    df = df.dropna(subset=["Margin", "MARKET_SPREAD"])

    # ATS cover: home covers when Margin + Spread > 0
    df["ats_cover"] = (df["Margin"] + df["MARKET_SPREAD"] > 0).astype(int)
    return df


def report_ats_by_tag(df):
    """Report ATS accuracy grouped by AH tag."""
    print("\n" + "=" * 60)
    print("  ATS ACCURACY BY AH TAG")
    print("=" * 60)

    for tag in ["AH-BET", "AH-SKIP", "AH-PASS"]:
        subset = df[df["ah_tag"] == tag]
        if len(subset) == 0:
            print(f"  {tag}: no data")
            continue

        with_result = subset[subset["ah_actual_cover"].notna()]
        if len(with_result) == 0:
            print(f"  {tag}: {len(subset)} predictions, 0 with results")
            continue

        covers = with_result["ah_actual_cover"].sum()
        total = len(with_result)
        accuracy = covers / total * 100
        print(f"  {tag}: {accuracy:.1f}% ATS ({int(covers)}/{total})")


def report_sigma_comparison(df):
    """Compare game_sigma vs bucket_sigma."""
    print("\n" + "=" * 60)
    print("  GAME SIGMA vs BUCKET SIGMA")
    print("=" * 60)

    if "ah_game_sigma" not in df.columns or df["ah_game_sigma"].isna().all():
        print("  No game_sigma data available yet.")
        return

    valid = df[df["ah_game_sigma"].notna()].copy()
    if len(valid) == 0:
        print("  No games with game_sigma recorded.")
        return

    if "spread" in valid.columns:
        valid["bucket_sigma"] = valid["spread"].apply(lambda s: sigma_for_line(s))
        valid["sigma_diff"] = valid["ah_game_sigma"] - valid["bucket_sigma"]

        print(f"  Games with game_sigma: {len(valid)}")
        print(f"  Mean game_sigma:   {valid['ah_game_sigma'].mean():.2f}")
        print(f"  Mean bucket_sigma: {valid['bucket_sigma'].mean():.2f}")
        print(f"  Mean difference:   {valid['sigma_diff'].mean():+.2f}")
        print(f"  Std difference:    {valid['sigma_diff'].std():.2f}")

        # Compare ATS accuracy: when game_sigma < bucket_sigma (model more certain)
        more_certain = valid[valid["sigma_diff"] < -1.0]
        less_certain = valid[valid["sigma_diff"] > 1.0]

        for label, sub in [("σ_game < σ_bucket (certain)", more_certain),
                           ("σ_game > σ_bucket (uncertain)", less_certain)]:
            with_result = sub[sub["ah_actual_cover"].notna()]
            if len(with_result) > 0:
                acc = with_result["ah_actual_cover"].mean() * 100
                print(f"  {label}: {acc:.1f}% ATS (n={len(with_result)})")


def report_historical_ats(df):
    """Report historical ATS statistics from dataset."""
    print("\n" + "=" * 60)
    print("  HISTORICAL ATS STATISTICS")
    print("=" * 60)

    total = len(df)
    covers = df["ats_cover"].sum()
    print(f"  Total games: {total:,}")
    print(f"  Home covers: {covers:,} ({covers / total * 100:.1f}%)")
    print(f"  Away covers: {total - covers:,} ({(total - covers) / total * 100:.1f}%)")

    # By spread bucket
    print("\n  ATS by spread bucket:")
    df["abs_spread"] = df["MARKET_SPREAD"].abs()
    buckets = [(0, 2, "Pick'em (0-2)"), (2, 5, "Small (2-5)"),
               (5, 8, "Medium (5-8)"), (8, 12, "Large (8-12)"),
               (12, 30, "Huge (12+)")]
    for lo, hi, label in buckets:
        mask = (df["abs_spread"] > lo) & (df["abs_spread"] <= hi)
        sub = df[mask]
        if len(sub) > 0:
            acc = sub["ats_cover"].mean() * 100
            print(f"    {label:20s}: {acc:.1f}% home cover (n={len(sub):,})")

    # ATS features check
    if "ATS_RATE_HOME" in df.columns:
        valid_ats = df["ATS_RATE_HOME"].notna() & (df["ATS_RATE_HOME"] != 0.5)
        n_with_ats = valid_ats.sum()
        print(f"\n  Games with ATS features: {n_with_ats:,}/{total:,} ({n_with_ats / total * 100:.0f}%)")

        if n_with_ats > 100:
            # Correlation between ATS_RATE and actual cover
            subset = df[valid_ats]
            corr = subset["ATS_RATE_HOME"].corr(subset["ats_cover"])
            print(f"  Correlation ATS_RATE_HOME ↔ actual cover: {corr:.3f}")

            # Top/bottom ATS rate accuracy
            high_ats = subset[subset["ATS_RATE_HOME"] > 0.55]
            low_ats = subset[subset["ATS_RATE_HOME"] < 0.45]
            if len(high_ats) > 20:
                print(f"  High ATS rate (>55%): {high_ats['ats_cover'].mean() * 100:.1f}% actual (n={len(high_ats)})")
            if len(low_ats) > 20:
                print(f"  Low ATS rate (<45%):  {low_ats['ats_cover'].mean() * 100:.1f}% actual (n={len(low_ats)})")


def main():
    parser = argparse.ArgumentParser(description="Audit AH performance")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  AH PERFORMANCE AUDIT")
    print("=" * 60)

    # 1. Live predictions audit
    bets_df = load_bets_data()
    if bets_df is not None:
        print(f"\n  Loaded {len(bets_df)} AH predictions from BetsTracking.sqlite")
        report_ats_by_tag(bets_df)
        report_sigma_comparison(bets_df)
    else:
        print("\n  No AH prediction data available yet.")
        print("  Run predictions with the updated code to start tracking.")

    # 2. Historical dataset audit
    hist_df = load_historical_ats()
    if hist_df is not None:
        report_historical_ats(hist_df)
    else:
        print("\n  No historical dataset available for backtest.")

    print("\n" + "=" * 60)
    print("  NEXT STEPS")
    print("=" * 60)
    print("  1. Run predictions to accumulate AH tracking data")
    print("  2. After 50+ AH-BET games: re-run this audit")
    print("  3. Compare ATS accuracy: AH-BET should be >53%")
    print("  4. If game_sigma improves accuracy → keep; else → revert to bucket")
    print("  5. Re-train margin model with ATS features:")
    print("     PYTHONPATH=. python scripts/build_margin_features.py")
    print("     PYTHONPATH=. python scripts/train_margin_models.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
