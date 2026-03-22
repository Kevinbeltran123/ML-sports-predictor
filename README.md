# NBA W/L Predictor

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=flat)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat&logo=sqlite&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)

Standalone NBA moneyline predictor. XGBoost + CatBoost ensemble achieving **~65% accuracy** on the 2025-26 season (809-game test set), Optuna-tuned for calibration (ECE=0.031). Includes in-game cascade models that update predictions after each quarter using real-time play-by-play data.

---

## Features

- **Pregame ensemble** — XGBoost + CatBoost weighted average with conformal prediction filter
- **In-game cascade** — Three-level model (XGB → Logistic → Bayesian) updates after Q1/Q2/Q3 using live PBP data
- **First Half model** — Separate ensemble for halftime moneyline (62.3% accuracy)
- **Kelly criterion** — DRO-Kelly (sigma-adaptive) and eighth-Kelly bankroll sizing, 2.5% cap
- **Asian Handicap** — Quarter-line settlement math with P(cover) and EV
- **CLV tracking** — Closing line value analysis per sportsbook
- **188 features** — Team stats, advanced metrics, ELO, fatigue, travel, injury impact, lineup strength, Four Factors, market features

---

## Quick Start

```bash
conda activate nba-betting-312

# Interactive menu (recommended)
PYTHONPATH=. python predictor.py

# CLI flags
PYTHONPATH=. python predictor.py -ensemble -odds fanduel -kelly
PYTHONPATH=. python predictor.py -ensemble -odds fanduel -kelly --live   # with live Q updates
PYTHONPATH=. python predictor.py --league wnba -ensemble -odds fanduel   # WNBA
```

**Requirements:** `ODDS_API_KEY` in `.env` (The Odds API). Python 3.12, conda env `nba-betting-312`.

---

## Model Performance

| Model | Accuracy | Notes |
|-------|----------|-------|
| XGBoost | 65.0% | 809-game test set, 2025-26 season |
| CatBoost | 65.9% | Same test set |
| Ensemble (pregame) | ~65% | Optuna-tuned weights |
| First Half ensemble | 62.3% | 4,052 games, 2019-2026 |
| In-game Q3 (XGB) | 81.5% | 8,260 games, 7 seasons |

---

## Console Output

Each game is displayed as a compact block, ranked by Expected Value:

```
  #1  Celtics (68.2%) vs Hawks  [BET]
       ML  Celtics: EV=+12.3  Kelly=1.85%  [+ XGB:69.1% Cat:66.8%]  σ=0.045
       AH  Celtics (-5.5): P=58.2% EV=+8.1 Kelly=1.20%
       O/U OVER 224.5 (61.2%)
       conformal: margin=0.18
```

| Tag | Meaning |
|-----|---------|
| `BET` | Conformal confident + positive EV |
| `SKIP` | Model uncertainty too high |
| `TRAP` | Underdog shows EV+ but prob < 35% — check the AH line of the favorite |
| `PASS` | Negative EV on both sides |

---

## Project Structure

```
predictor.py          # Entry point
config.toml           # Season config
src/
  core/
    betting/          # Kelly, EV, spread math, bet tracker, CLV
    calibration/      # Conformal prediction, Platt scaling
    ensemble/         # MWUA, stacking, evaluation
  sports/nba/
    features/         # 20+ feature modules
    predict/          # Ensemble runner, in-game cascade, live betting
    providers/        # Odds API integration
models/               # Trained XGBoost, CatBoost, conformal and variance models
data/training/        # TeamData, OddsData, PBPFeatures (SQLite)
scripts/              # Training pipelines
```

---

## CLI Reference

| Flag | Description |
|------|-------------|
| `-ensemble` | XGBoost + CatBoost ensemble (recommended) |
| `-odds BOOK` | Sportsbook: `fanduel`, `draftkings`, `betmgm`, `caesars` |
| `-kelly` | Kelly criterion bankroll sizing |
| `-clv` | Closing Line Value report |
| `--h1` | First Half moneyline predictions |
| `--live` | Live betting session (Q1-Q3 polling) |
| `--league nba/wnba` | Select league (default: nba) |
| `--all` | Run all modes |
