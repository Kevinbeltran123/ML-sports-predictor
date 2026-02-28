"""Conference and division rivalry features.

Adds IS_SAME_CONFERENCE and IS_SAME_DIVISION binary features
to capture rivalry/familiarity effects between teams.
"""

import pandas as pd

# Team → (conference, division) mapping — 30 NBA teams (2024-25)
NBA_TEAM_INFO = {
    # --- Atlantic ---
    "Boston Celtics":         ("Eastern", "Atlantic"),
    "Brooklyn Nets":          ("Eastern", "Atlantic"),
    "New York Knicks":        ("Eastern", "Atlantic"),
    "Philadelphia 76ers":     ("Eastern", "Atlantic"),
    "Toronto Raptors":        ("Eastern", "Atlantic"),
    # --- Central ---
    "Chicago Bulls":          ("Eastern", "Central"),
    "Cleveland Cavaliers":    ("Eastern", "Central"),
    "Detroit Pistons":        ("Eastern", "Central"),
    "Indiana Pacers":         ("Eastern", "Central"),
    "Milwaukee Bucks":        ("Eastern", "Central"),
    # --- Southeast ---
    "Atlanta Hawks":          ("Eastern", "Southeast"),
    "Charlotte Hornets":      ("Eastern", "Southeast"),
    "Miami Heat":             ("Eastern", "Southeast"),
    "Orlando Magic":          ("Eastern", "Southeast"),
    "Washington Wizards":     ("Eastern", "Southeast"),
    # --- Northwest ---
    "Denver Nuggets":         ("Western", "Northwest"),
    "Minnesota Timberwolves": ("Western", "Northwest"),
    "Oklahoma City Thunder":  ("Western", "Northwest"),
    "Portland Trail Blazers": ("Western", "Northwest"),
    "Utah Jazz":              ("Western", "Northwest"),
    # --- Pacific ---
    "Golden State Warriors":  ("Western", "Pacific"),
    "LA Clippers":            ("Western", "Pacific"),
    "Los Angeles Lakers":     ("Western", "Pacific"),
    "Phoenix Suns":           ("Western", "Pacific"),
    "Sacramento Kings":       ("Western", "Pacific"),
    # --- Southwest ---
    "Dallas Mavericks":       ("Western", "Southwest"),
    "Houston Rockets":        ("Western", "Southwest"),
    "Memphis Grizzlies":      ("Western", "Southwest"),
    "New Orleans Pelicans":   ("Western", "Southwest"),
    "San Antonio Spurs":      ("Western", "Southwest"),
}


def get_game_conference_division(home_team, away_team):
    """Calcula features de conferencia/division para un partido.

    Args:
        home_team: nombre del equipo local
        away_team: nombre del equipo visitante

    Returns:
        dict con IS_SAME_CONFERENCE (0/1) y IS_SAME_DIVISION (0/1)
    """
    home_info = NBA_TEAM_INFO.get(home_team)
    away_info = NBA_TEAM_INFO.get(away_team)

    if not home_info or not away_info:
        return {"IS_SAME_CONFERENCE": 0, "IS_SAME_DIVISION": 0}

    return {
        "IS_SAME_CONFERENCE": int(home_info[0] == away_info[0]),
        "IS_SAME_DIVISION": int(home_info[1] == away_info[1]),
    }


def add_conference_division_to_frame(frame, features_list):
    """Agrega columnas de conferencia/division al DataFrame.

    Args:
        frame: DataFrame con los juegos
        features_list: list of dicts de get_game_conference_division()

    Returns:
        DataFrame con 2 columnas nuevas
    """
    if not features_list:
        return frame

    cd_df = pd.DataFrame(features_list, index=frame.index)
    return pd.concat([frame, cd_df], axis=1)
