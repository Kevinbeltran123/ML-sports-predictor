"""MLB-specific configuration constants.

30-team MLB league: AL East/Central/West, NL East/Central/West.
Elo parameters tuned for 162-game season (slower convergence than NBA/WNBA).
Park coordinates for travel/weather calculations.
"""

# --------------------------------------------------------------------------- #
# League identifiers
# --------------------------------------------------------------------------- #

MLB_ODDS_API_SPORT = "baseball_mlb"
MLB_SEASON_START_MONTH = 3   # Late March (Opening Day)
MLB_SEASON_END_MONTH = 10    # October (World Series)

# --------------------------------------------------------------------------- #
# Teams (30 franchises, 2024-25)
# --------------------------------------------------------------------------- #

MLB_TEAMS = [
    # AL East
    "Baltimore Orioles", "Boston Red Sox", "New York Yankees",
    "Tampa Bay Rays", "Toronto Blue Jays",
    # AL Central
    "Chicago White Sox", "Cleveland Guardians", "Detroit Tigers",
    "Kansas City Royals", "Minnesota Twins",
    # AL West
    "Houston Astros", "Los Angeles Angels", "Oakland Athletics",
    "Seattle Mariners", "Texas Rangers",
    # NL East
    "Atlanta Braves", "Miami Marlins", "New York Mets",
    "Philadelphia Phillies", "Washington Nationals",
    # NL Central
    "Chicago Cubs", "Cincinnati Reds", "Milwaukee Brewers",
    "Pittsburgh Pirates", "St. Louis Cardinals",
    # NL West
    "Arizona Diamondbacks", "Colorado Rockies", "Los Angeles Dodgers",
    "San Diego Padres", "San Francisco Giants",
]

team_index_mlb: dict[str, int] = {team: idx for idx, team in enumerate(MLB_TEAMS)}

# Abbreviations used by MLB Stats API
MLB_ABBREV = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET",
    "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD", "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
}

MLB_ABBREV_TO_NAME = {v: k for k, v in MLB_ABBREV.items()}

# Historical aliases (franchise renames)
MLB_TEAM_ALIASES = {
    "Cleveland Indians": "Cleveland Guardians",
    "Oakland A's": "Oakland Athletics",
    "Anaheim Angels": "Los Angeles Angels",
    "Los Angeles Angels of Anaheim": "Los Angeles Angels",
    "Florida Marlins": "Miami Marlins",
    "Tampa Bay Devil Rays": "Tampa Bay Rays",
    "Montreal Expos": "Washington Nationals",
}

# --------------------------------------------------------------------------- #
# Elo parameters
# --------------------------------------------------------------------------- #

# K=4: 162-game season → much slower convergence than NBA (K=20) or WNBA (K=32)
MLB_ELO_K = 4
MLB_ELO_BASE = 1500.0

# Home field advantage ~53.5% win rate → ~28 Elo points
MLB_HOME_ADVANTAGE = 28.0

# Season carry-over (higher than NBA: baseball teams change less year-to-year)
MLB_SEASON_CARRY = 0.80

# --------------------------------------------------------------------------- #
# Sigma buckets for DRO-Kelly
#
# MLB models are inherently less certain than NBA (more game-to-game variance).
# Typical sigma range: 0.04-0.18.
# --------------------------------------------------------------------------- #

MLB_SIGMA_BUCKETS: list[tuple[float, float]] = [
    (0.05, 0.50),    # very low disagreement
    (0.08, 0.35),
    (0.11, 0.20),    # typical MLB sigma midpoint
    (0.15, 0.10),
    (float("inf"), 0.03),  # high disagreement → near-skip
]

# --------------------------------------------------------------------------- #
# Divisions
# --------------------------------------------------------------------------- #

MLB_AL_EAST = {"Baltimore Orioles", "Boston Red Sox", "New York Yankees",
               "Tampa Bay Rays", "Toronto Blue Jays"}
MLB_AL_CENTRAL = {"Chicago White Sox", "Cleveland Guardians", "Detroit Tigers",
                  "Kansas City Royals", "Minnesota Twins"}
MLB_AL_WEST = {"Houston Astros", "Los Angeles Angels", "Oakland Athletics",
               "Seattle Mariners", "Texas Rangers"}
MLB_NL_EAST = {"Atlanta Braves", "Miami Marlins", "New York Mets",
               "Philadelphia Phillies", "Washington Nationals"}
MLB_NL_CENTRAL = {"Chicago Cubs", "Cincinnati Reds", "Milwaukee Brewers",
                  "Pittsburgh Pirates", "St. Louis Cardinals"}
MLB_NL_WEST = {"Arizona Diamondbacks", "Colorado Rockies", "Los Angeles Dodgers",
               "San Diego Padres", "San Francisco Giants"}

MLB_AL = MLB_AL_EAST | MLB_AL_CENTRAL | MLB_AL_WEST
MLB_NL = MLB_NL_EAST | MLB_NL_CENTRAL | MLB_NL_WEST

MLB_DIVISION: dict[str, str] = {}
MLB_DIVISION.update({t: "AL_East" for t in MLB_AL_EAST})
MLB_DIVISION.update({t: "AL_Central" for t in MLB_AL_CENTRAL})
MLB_DIVISION.update({t: "AL_West" for t in MLB_AL_WEST})
MLB_DIVISION.update({t: "NL_East" for t in MLB_NL_EAST})
MLB_DIVISION.update({t: "NL_Central" for t in MLB_NL_CENTRAL})
MLB_DIVISION.update({t: "NL_West" for t in MLB_NL_WEST})

MLB_LEAGUE: dict[str, str] = {}
MLB_LEAGUE.update({t: "AL" for t in MLB_AL})
MLB_LEAGUE.update({t: "NL" for t in MLB_NL})

# --------------------------------------------------------------------------- #
# Ballpark coordinates (lat, lon, utc_offset_hours)
# utc_offset is standard time (no DST adjustment needed — games are in local TZ)
# --------------------------------------------------------------------------- #

MLB_PARKS: dict[str, tuple[float, float, int]] = {
    "Arizona Diamondbacks":    (33.445, -112.067, -7),   # Chase Field (MST, no DST)
    "Atlanta Braves":          (33.891, -84.468, -5),     # Truist Park
    "Baltimore Orioles":       (39.284, -76.622, -5),     # Camden Yards
    "Boston Red Sox":          (42.346, -71.097, -5),     # Fenway Park
    "Chicago Cubs":            (41.948, -87.656, -6),     # Wrigley Field
    "Chicago White Sox":       (41.830, -87.634, -6),     # Guaranteed Rate Field
    "Cincinnati Reds":         (39.097, -84.507, -5),     # Great American Ball Park
    "Cleveland Guardians":     (41.496, -81.685, -5),     # Progressive Field
    "Colorado Rockies":        (39.756, -104.994, -7),    # Coors Field
    "Detroit Tigers":          (42.339, -83.049, -5),     # Comerica Park
    "Houston Astros":          (29.757, -95.355, -6),     # Minute Maid Park
    "Kansas City Royals":      (39.051, -94.481, -6),     # Kauffman Stadium
    "Los Angeles Angels":      (33.800, -117.883, -8),    # Angel Stadium
    "Los Angeles Dodgers":     (34.074, -118.240, -8),    # Dodger Stadium
    "Miami Marlins":           (25.778, -80.220, -5),     # loanDepot Park
    "Milwaukee Brewers":       (43.028, -87.971, -6),     # American Family Field
    "Minnesota Twins":         (44.982, -93.278, -6),     # Target Field
    "New York Mets":           (40.757, -73.846, -5),     # Citi Field
    "New York Yankees":        (40.829, -73.926, -5),     # Yankee Stadium
    "Oakland Athletics":       (37.752, -122.201, -8),    # Oakland Coliseum
    "Philadelphia Phillies":   (39.906, -75.167, -5),     # Citizens Bank Park
    "Pittsburgh Pirates":      (40.447, -80.006, -5),     # PNC Park
    "San Diego Padres":        (32.707, -117.157, -8),    # Petco Park
    "San Francisco Giants":    (37.778, -122.389, -8),    # Oracle Park
    "Seattle Mariners":        (47.591, -122.332, -8),    # T-Mobile Park
    "St. Louis Cardinals":     (38.623, -90.193, -6),     # Busch Stadium
    "Tampa Bay Rays":          (27.768, -82.653, -5),     # Tropicana Field
    "Texas Rangers":           (32.751, -97.083, -6),     # Globe Life Field
    "Toronto Blue Jays":       (43.641, -79.389, -5),     # Rogers Centre
    "Washington Nationals":    (38.873, -77.007, -5),     # Nationals Park
}

# Parks with retractable roofs or domes (weather features → 0)
MLB_DOMED_PARKS = {
    "Tampa Bay Rays",          # Tropicana Field (fixed dome)
    "Miami Marlins",           # loanDepot Park (retractable)
    "Houston Astros",          # Minute Maid Park (retractable)
    "Texas Rangers",           # Globe Life Field (retractable)
    "Milwaukee Brewers",       # American Family Field (retractable)
    "Toronto Blue Jays",       # Rogers Centre (retractable)
    "Arizona Diamondbacks",    # Chase Field (retractable)
    "Seattle Mariners",        # T-Mobile Park (retractable)
}

# --------------------------------------------------------------------------- #
# Home plate orientation (degrees from North, clockwise)
# Used to compute wind in/out for HR prediction.
# 0° = facing North, 90° = facing East, etc.
# --------------------------------------------------------------------------- #

MLB_PLATE_ORIENTATION: dict[str, float] = {
    "Arizona Diamondbacks": 0,    "Atlanta Braves": 225,
    "Baltimore Orioles": 225,     "Boston Red Sox": 225,
    "Chicago Cubs": 225,          "Chicago White Sox": 180,
    "Cincinnati Reds": 200,       "Cleveland Guardians": 180,
    "Colorado Rockies": 230,      "Detroit Tigers": 180,
    "Houston Astros": 0,          "Kansas City Royals": 180,
    "Los Angeles Angels": 200,    "Los Angeles Dodgers": 225,
    "Miami Marlins": 0,           "Milwaukee Brewers": 0,
    "Minnesota Twins": 180,       "New York Mets": 180,
    "New York Yankees": 225,      "Oakland Athletics": 225,
    "Philadelphia Phillies": 225, "Pittsburgh Pirates": 45,
    "San Diego Padres": 225,      "San Francisco Giants": 225,
    "Seattle Mariners": 0,        "St. Louis Cardinals": 180,
    "Tampa Bay Rays": 0,          "Texas Rangers": 0,
    "Toronto Blue Jays": 0,       "Washington Nationals": 225,
}
