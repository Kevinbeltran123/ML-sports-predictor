"""WNBA-specific configuration constants.

All WNBA league identifiers, team mappings, and model tuning parameters.
Mirrors the structure of the NBA feature dictionaries in
src/sports/nba/features/dictionaries.py but scoped to the 12-team WNBA.
"""

# --------------------------------------------------------------------------- #
# League identifiers
# --------------------------------------------------------------------------- #

WNBA_LEAGUE_ID = "10"
WNBA_ODDS_API_SPORT = "basketball_wnba"

# --------------------------------------------------------------------------- #
# Teams (2024-25 roster — 12 franchises)
# --------------------------------------------------------------------------- #

WNBA_TEAMS = [
    "Atlanta Dream",
    "Chicago Sky",
    "Connecticut Sun",
    "Dallas Wings",
    "Indiana Fever",
    "Las Vegas Aces",
    "Los Angeles Sparks",
    "Minnesota Lynx",
    "New York Liberty",
    "Phoenix Mercury",
    "Seattle Storm",
    "Washington Mystics",
]

# Mapping used by create_games_wnba and Elo: team name -> integer index 0-11
team_index_wnba: dict[str, int] = {team: idx for idx, team in enumerate(WNBA_TEAMS)}

# --------------------------------------------------------------------------- #
# Elo parameters
# --------------------------------------------------------------------------- #

# Higher K than NBA (32 vs ~20) because the WNBA regular season is only ~36 games
# and ratings must converge faster within a single season.
WNBA_ELO_K = 32

# Starting Elo for all teams at the beginning of each season
WNBA_ELO_BASE = 1500.0

# Home-court advantage in Elo points (WNBA arenas are smaller/more intimate)
WNBA_HOME_ADVANTAGE = 70.0

# --------------------------------------------------------------------------- #
# Sigma buckets for robust Kelly
#
# WNBA average combined score ~160-165 vs NBA ~220-230.
# Lower scoring means more game-to-game variance in point differentials,
# but model disagreement (sigma) is historically ~10-12 pp vs NBA ~13-16 pp.
# Buckets are tighter to reflect the compressed range.
# --------------------------------------------------------------------------- #

WNBA_SIGMA_BUCKETS: list[tuple[float, float]] = [
    # (sigma_upper_bound, kelly_fraction_cap)
    (0.06, 0.75),   # very low disagreement — models agree strongly
    (0.09, 0.60),
    (0.12, 0.45),   # typical WNBA sigma midpoint
    (0.15, 0.30),
    (0.20, 0.15),
    (float("inf"), 0.05),  # high disagreement — near-skip
]

# --------------------------------------------------------------------------- #
# Conference / division mapping
# --------------------------------------------------------------------------- #

# WNBA uses a single-conference format with East/West divisions (as of 2024).
WNBA_EASTERN = {
    "Atlanta Dream",
    "Chicago Sky",
    "Connecticut Sun",
    "Indiana Fever",
    "New York Liberty",
    "Washington Mystics",
}

WNBA_WESTERN = {
    "Dallas Wings",
    "Las Vegas Aces",
    "Los Angeles Sparks",
    "Minnesota Lynx",
    "Phoenix Mercury",
    "Seattle Storm",
}

# Convenience lookup: team -> division string
WNBA_DIVISION: dict[str, str] = {}
WNBA_DIVISION.update({t: "East" for t in WNBA_EASTERN})
WNBA_DIVISION.update({t: "West" for t in WNBA_WESTERN})
