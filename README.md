# NBA W/L Predictor

Standalone NBA moneyline predictor. Ensemble of XGBoost (60%) + CatBoost (40%) achieving **66.38% accuracy** on 809-game test set (2025-26 season).

## Quick Start

```bash
conda activate nba-betting-312
PYTHONPATH=. python predictor.py -ensemble -odds fanduel -kelly
```

## How It Works

1. Fetches today's NBA odds from The Odds API
2. Pulls current team stats from nba.com + historical data from SQLite
3. Engineers 158 features (after ablation pruning from 218)
4. Runs XGBoost + CatBoost ensemble with weighted average
5. Outputs picks ranked by Expected Value with Kelly sizing

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

| Model | Type | Accuracy | Location |
|-------|------|----------|----------|
| XGBoost | Moneyline | 65.0% | `models/moneyline/` |
| CatBoost | Moneyline | 65.9% | `models/moneyline/` |
| Ensemble | Weighted avg | 66.38% | Runtime (60/40) |
| XGB variance | Per-game σ | — | `models/moneyline/ensemble_variance.json` |
| Conformal | Bet filter | — | `models/moneyline/ensemble_conformal.pkl` |
| In-game XGB | Q1/Q2/Q3 | 68.9-78.8% | `models/ingame/` |
| In-game Logistic | Q1/Q2/Q3 | 67.2-77.4% | `models/ingame/` |

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
- Level 2: XGB in-game models (best accuracy)
- Level 1: Logistic regression (fallback)
- Level 0: Bayesian adjustment (β=0.45, always available)

## CLI Flags

| Flag | Description |
|------|-------------|
| `-ensemble` | Run XGB+CatBoost ensemble (recommended) |
| `-xgb` | Run XGBoost solo |
| `-odds BOOK` | Fetch odds: `fanduel`, `draftkings`, `betmgm`, `caesars`, `wynn`, `bet_rivers_ny` |
| `-kelly` | Show Kelly criterion bankroll sizing |
| `-clv` | Print CLV (Closing Line Value) report |
| `--live` | Activate live betting session (Q1-Q3 polling) |

## Project Structure

```
predictor.py                  # Entry point
config.toml                   # Season dates and data config
src/
  config.py                   # Paths, DROP_COLUMNS_ML (158 features)
  core/
    betting/                  # Kelly, EV, spread math, bet tracker, CLV
    calibration/              # Conformal, Platt scaling, ECE
    ensemble/                 # MWUA, stacking, evaluation
    stats/                    # ELO ratings, rolling averages
  sports/nba/
    features/                 # 20 feature modules
    predict/                  # Ensemble runner, XGB runner, in-game, live betting, margin
    providers/                # Odds API integration
models/
  moneyline/                  # XGB, CatBoost, conformal, variance models
  ingame/                     # Q1/Q2/Q3 cascade models
data/training/                # TeamData, OddsData, dataset SQLite
scripts/                      # Closing lines collection, conformal fitting
```

## Requirements

- Python 3.12 (conda env: `nba-betting-312`)
- `ODDS_API_KEY` in `.env` (The Odds API)
- `RAPIDAPI_KEY` in `.env` (optional, for injury reports)
