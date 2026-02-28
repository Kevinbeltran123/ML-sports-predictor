# NBA W/L Predictor

Standalone NBA moneyline predictor. Ensemble of XGBoost (60%) + CatBoost (40%) achieving **~66.4% accuracy** on 809-game test set (2025-26 season). In-game cascade models update predictions after each quarter using real-time play-by-play data.

## Quick Start

```bash
conda activate nba-betting-312

# Interactive menu (recommended) — guides you through all options
PYTHONPATH=. python predictor.py

# Or with CLI flags directly
PYTHONPATH=. python predictor.py -ensemble -odds fanduel -kelly
```

### Interactive Menu

Running `python predictor.py` without flags launches an interactive menu:

```
==================================================
  NBA W/L Predictor
==================================================

  Liga:
    1) NBA
    2) WNBA

  Sportsbook:
    1) fanduel (default)
    2) draftkings
    3) betmgm
    4) pointsbet
    5) caesars
    6) wynn
    7) bet_rivers_ny

  Modelo:
    1) Ensemble XGB 60% + CatBoost 40% (default)
    2) XGBoost solo

  Features adicionales:
    1) CLV report
    2) Live betting (Q1-Q3)
    3) Polymarket
    4) Ninguno
```

### CLI Flags

| Flag | Description |
|------|-------------|
| `-ensemble` | Run XGB+CatBoost ensemble (recommended) |
| `-xgb` | Run XGBoost solo |
| `-odds BOOK` | Fetch odds: `fanduel`, `draftkings`, `betmgm`, `caesars`, `wynn`, `bet_rivers_ny` |
| `-kelly` | Show Kelly criterion bankroll sizing |
| `-clv` | Print CLV (Closing Line Value) report |
| `--live` | Activate live betting session (Q1-Q3 polling) |
| `--polymarket` | Polymarket trading signals |
| `--polymarket-live` | Live position management on Polymarket |
| `--execute` | Execute real Polymarket orders (requires confirmation) |
| `--bankroll-usdc N` | Polymarket bankroll in USDC |
| `--league nba/wnba` | Select league (default: nba) |
| `--all` | Run all modes (ensemble + kelly + live) |

```bash
# Common usage examples
PYTHONPATH=. python predictor.py -ensemble -odds fanduel -kelly          # daily picks
PYTHONPATH=. python predictor.py -ensemble -odds fanduel -kelly --live   # daily + live Q updates
PYTHONPATH=. python predictor.py --all                                    # everything
PYTHONPATH=. python predictor.py -ensemble -odds draftkings -kelly -clv  # with CLV report
PYTHONPATH=. python predictor.py --league wnba -ensemble -odds fanduel   # WNBA picks
```

## How It Works

1. Fetches today's NBA odds from The Odds API
2. Pulls current team stats from nba.com + historical data from SQLite
3. Engineers 158 features (after ablation pruning from 218)
4. Runs XGBoost + CatBoost ensemble with weighted average
5. Outputs picks ranked by Expected Value with Kelly sizing
6. (Live mode) Polls quarter scores and runs in-game cascade for updated probabilities

## Console Output

Each game is shown as a compact block, ranked by EV (best first):

```
  #1  Celtics (68.2%) vs Hawks  [BET]
       ML  Celtics: EV=+12.3  Kelly=1.85%  [+ XGB:69.1% Cat:66.8%]  σ=0.045
       AH  Celtics (-5.5): P=58.2% EV=+8.1 Kelly=1.20%  margin=+6.2
       O/U OVER 224.5 (61.2%)
       conformal: margin=0.18
```

**Status tags:**
- `BET` — Conformal confident + positive EV. Bet this.
- `SKIP` — Conformal uncertain (set_size=2). Model can't separate the teams.
- `TRAP` — Underdog shows EV+ but prob < 35%. Empirically never hits. Check the AH line of the favorite instead.
- `PASS` — Negative EV on both sides. No edge.

**Key fields:**
- `EV` — Expected value vs sportsbook line. Positive = edge.
- `Kelly` — % of bankroll to wager (DRO-Kelly when sigma available, eighth-Kelly otherwise).
- `[+ XGB:69% Cat:67%]` — `+` means both models agree, `~` means they disagree.
- `σ` — Per-game uncertainty. Green (< 0.07) = predictable, yellow (0.07-0.10) = moderate, red (> 0.10) = uncertain.
- `AH` — Asian Handicap (spread) best side with P(cover), EV, and Kelly.
- `margin` — Expected point margin from classifier.
- `conformal margin` — Distance from threshold. Higher = more confidence.

## Betting Checklist

1. Tag says `BET`
2. Both models agree (`+` not `~`)
3. σ is green (< 0.07)
4. Conformal margin > 0.10
5. If `TRAP`, ignore ML underdog — look at the AH line of the favorite
6. Never bet more than Kelly says

## Models

### Pregame Ensemble

| Model | Type | Accuracy | Location |
|-------|------|----------|----------|
| XGBoost | Moneyline | 64.9% | `models/moneyline/` |
| CatBoost | Moneyline | 66.3% | `models/moneyline/` |
| Ensemble | Weighted avg | ~66.4% | Runtime (60/40) |
| XGB variance | Per-game σ | — | `models/moneyline/ensemble_variance.json` |
| Conformal | Bet filter | — | `models/moneyline/ensemble_conformal.pkl` |

### In-Game Cascade v2 (trained on 8,260 games / 24,780 rows)

Three-level cascade per quarter, falling back gracefully:

| Level | Model | Q1 | Q2 | Q3 | Features |
|-------|-------|----|----|-----|----------|
| 2 | XGBoost + Platt | 65.2% | 71.7% | 81.5% | 21 (PBP + score-context) |
| 1 | Logistic | 65.2% | 71.7% | 79.8% | 12 (PBP subset + score-context) |
| 0 | Bayesian | — | — | — | score_diff + possessions |

**Training data:** 7 NBA seasons (2019-20 through 2025-26), Regular Season + Playoffs.
PBP data sourced from `cdn.nba.com` (public, no auth required).

**PBP Features (20):** Lead changes, largest leads, scoring runs, timeouts, fouls, turnovers, momentum, last-5-min differential, 3PT made, offensive rebounds, plus score-context features (normalized diff, blowout flag, attenuated momentum).

## Features (158 after ablation)

Engineered from `src/sports/nba/features/`:

- **Team stats** — FG%, 3P%, rebounds, assists, turnovers, etc.
- **Advanced** — Offensive/Defensive rating, pace, TS%, ORB%, etc.
- **Style vectors** — Playing style characterization
- **Differentials** — Home vs away deltas
- **Rolling averages** — Recency-weighted team form (last 5/10 games)
- **ELO + SRS** — Strength ratings adjusted for schedule
- **Home/away splits** — Separate performance by venue
- **Fatigue** — Days rest, back-to-back, games in 7/14 days
- **Travel** — Distance, timezone changes, altitude
- **Strength of schedule** — Opponent quality over last 10
- **Injury impact** — AVAIL_QUALITY (weighted availability from injury reports)
- **Lineup strength** — TOP5 plus/minus, depth score, star power, HHI minutes
- **Market** — Devigged ML probability from sportsbook odds

Dropped by ablation: shot chart zones, ESPN line moves, on/off ratings, referee tendencies, granular injury fields, lineup composition metrics.

## Betting Stack

- **Kelly criterion** — Eighth-Kelly (fixed) or DRO-Kelly (sigma-adaptive), 2.5% cap
- **Conformal prediction** — Filters uncertain games (set_size > 1)
- **Variance model** — Per-game sigma for adaptive Kelly sizing
- **Asian Handicap** — Quarter-line settlement math, P(cover), EV
- **Margin regression** — Separate XGBoost regressor for point spread comparison
- **Bet tracker** — SQLite persistence of all predictions
- **CLV tracking** — Closing line value analysis per sportsbook

## Live Betting

```bash
PYTHONPATH=. python predictor.py -ensemble -odds fanduel -kelly --live
```

Polls Q1/Q2/Q3 scores every 30s. Three-level cascade:
- Level 2: XGB in-game models (best accuracy, requires PBP data)
- Level 1: Logistic regression (fallback, requires PBP data)
- Level 0: Bayesian adjustment (β=0.45, always available)

## Training In-Game Models

```bash
# Full pipeline: collect PBP + train (resumable, checkpoints every 100 games)
PYTHONPATH=. python scripts/train_ingame_models.py

# Specific seasons
PYTHONPATH=. python scripts/train_ingame_models.py --seasons 2023-24 2024-25

# Retrain only (skip PBP collection)
PYTHONPATH=. python scripts/train_ingame_models.py --skip-collect
```

Data is fetched from `stats.nba.com` (game schedules) and `cdn.nba.com` (play-by-play).
Both are public endpoints — no API key needed. Rate limited to 0.3s between PBP requests.

## Project Structure

```
predictor.py                  # Entry point (interactive menu)
config.toml                   # Season dates and data config
src/
  config.py                   # Paths, DROP_COLUMNS_ML (158 features)
  core/
    betting/                  # Kelly, EV, spread math, bet tracker, CLV
    calibration/              # Conformal, Platt scaling, ECE
    ensemble/                 # MWUA, stacking, evaluation
    stats/                    # ELO ratings, rolling averages
    odds_cache.py             # Single-fetch cache for OddsApiProvider
  sports/nba/
    features/                 # 20+ feature modules
      live_pbp_tracker.py     # Real-time PBP feature extraction
    predict/                  # Ensemble runner, XGB runner, in-game, live betting, margin
      ingame_runner.py        # In-game cascade: XGBoost → Logistic → Bayesian
      live_betting.py         # Live session polling + cascade trigger
    providers/                # Odds API integration
models/
  moneyline/                  # XGB, CatBoost, conformal, variance models
  ingame/                     # Q1/Q2/Q3 cascade models (XGB, Logistic, calibration, conformal)
  margin/                     # Margin regression models (optional)
  totals/                     # Over/Under models (optional)
data/
  training/                   # TeamData, OddsData, dataset, PBPFeatures SQLite
  nba/predictions/            # BetsTracking.sqlite, live_bets.csv
  nba/research/               # HistoricalLines.sqlite
  nba/schedules/              # Season schedule CSV
scripts/
  train_ingame_models.py      # PBP collection + in-game model training
  train_models.py             # Pregame moneyline model training
  train_margin_models.py      # Margin regression training
  train_totals_models.py      # Over/Under model training
  collect_closing_lines.py    # CLV analysis data collection
```

## Requirements

- Python 3.12 (conda env: `nba-betting-312`)
- `ODDS_API_KEY` in `.env` (The Odds API)
- `RAPIDAPI_KEY` in `.env` (optional, for injury reports)
