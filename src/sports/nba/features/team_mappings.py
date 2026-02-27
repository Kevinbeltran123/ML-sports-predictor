"""Mapping compartido de nombres de equipos NBA.

Centraliza la conversión entre formatos:
  - OddsData / create_games.py: "Los Angeles Lakers" (nombre completo)
  - PlayerGameLogs (nba_api): "LAL" (3 letras nba_api)
  - BRefData (Basketball Reference): "LAL" (3 letras BRef)

Diferencias clave entre nba_api y BRef:
  - Charlotte: CHA (nba_api) vs CHO (BRef)
  - Phoenix: PHX (nba_api) vs PHO (BRef)
  - Brooklyn: BKN (nba_api) vs BRK (BRef)
  - New Jersey: NJN (ambos, histórico)
"""

# ---------------------------------------------------------------------------
# Nombre completo → abreviatura nba_api (PlayerGameLogs)
# Usado por lineup_strength.py y módulos que consultan PlayerGameLogs
# ---------------------------------------------------------------------------
TEAM_NAME_TO_ABBR = {
    # --- 30 equipos actuales ---
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
    # --- Aliases históricos ---
    "Los Angeles Clippers": "LAC",
    "Charlotte Bobcats": "CHA",
    "New Jersey Nets": "NJN",
    "New Orleans Hornets": "NOH",
    "Seattle SuperSonics": "SEA",
}

# ---------------------------------------------------------------------------
# Nombre completo → abreviatura BRef (Basketball Reference)
# BRef usa códigos ligeramente diferentes: CHO, PHO, BRK
# ---------------------------------------------------------------------------
TEAM_NAME_TO_BREF = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BRK",
    "Charlotte Hornets": "CHO",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
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
    "Phoenix Suns": "PHO",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
    # --- Aliases históricos ---
    "Los Angeles Clippers": "LAC",
    "Charlotte Bobcats": "CHO",
    "New Jersey Nets": "NJN",
    "New Orleans Hornets": "NOH",
    "Seattle SuperSonics": "SEA",
}

# Inverso: BRef code → nombre completo (para joins con schedules)
BREF_TO_TEAM_NAME = {v: k for k, v in TEAM_NAME_TO_BREF.items()
                     if k not in ("Los Angeles Clippers", "Charlotte Bobcats",
                                  "New Jersey Nets", "New Orleans Hornets",
                                  "Seattle SuperSonics")}

# nba_api code → BRef code (para conversiones directas)
ABBR_TO_BREF = {
    "ATL": "ATL", "BOS": "BOS", "BKN": "BRK", "CHA": "CHO",
    "CHI": "CHI", "CLE": "CLE", "DAL": "DAL", "DEN": "DEN",
    "DET": "DET", "GSW": "GSW", "HOU": "HOU", "IND": "IND",
    "LAC": "LAC", "LAL": "LAL", "MEM": "MEM", "MIA": "MIA",
    "MIL": "MIL", "MIN": "MIN", "NOP": "NOP", "NYK": "NYK",
    "OKC": "OKC", "ORL": "ORL", "PHI": "PHI", "PHX": "PHO",
    "POR": "POR", "SAC": "SAC", "SAS": "SAS", "TOR": "TOR",
    "UTA": "UTA", "WAS": "WAS",
    # Históricos
    "NJN": "NJN", "NOH": "NOH", "SEA": "SEA",
}
