"""LivePBPTracker: computa PBP features en tiempo real desde nba_api.live.PlayByPlay.

Replica la misma lógica de create_pbp_features.py pero adaptada para datos live:
- Input: lista de actions de nba_api.live.PlayByPlay(game_id).get_dict()["game"]["actions"]
- Output: dict con 17 PBP features (mismo formato que ingame_dataset.py espera)

Las features son ACUMULATIVAS hasta period_end (igual que el training set).

Estructura de un action de nba_api.live:
    {
        "period": 1,
        "clock": "PT05M30.00S",
        "teamTricode": "BOS",
        "actionType": "2pt",        # 2pt, 3pt, freethrow, turnover, foul, timeout, rebound, ...
        "subType": "made",           # made, missed, offensive, personal, ...
        "shotResult": "Made",        # Made, Missed (para tiros)
        "scoreHome": "25",
        "scoreAway": "22",
        "description": "J. Tatum makes 3-pt jump shot",
    }
"""

import math
import re
from typing import Optional

from src.config import get_logger

logger = get_logger(__name__)


def _parse_clock_seconds(clock_str: str) -> float:
    """Convierte 'PT05M30.00S' a segundos restantes (330.0).

    Formato ISO 8601 duration de la NBA API:
      PT11M58.00S → 11 min 58 seg → 718.0 segundos
      PT00M00.00S → 0.0 segundos (fin del cuarto)
    """
    if not clock_str:
        return 0.0
    m = re.match(r"PT(\d+)M([\d.]+)S", clock_str)
    if m:
        return int(m.group(1)) * 60.0 + float(m.group(2))
    return 0.0


class LivePBPTracker:
    """Computa 17 PBP features incrementalmente desde play-by-play live.

    Uso:
        tracker = LivePBPTracker("BOS", "LAL")
        tracker.update(actions)  # lista completa de actions del PlayByPlay
        features = tracker.get_features(period_end=1)  # dict con 17 features
    """

    def __init__(self, home_tricode: str, away_tricode: str):
        self._home_tri = home_tricode.upper()
        self._away_tri = away_tricode.upper()
        # Parsed plays: lista de dicts con campos normalizados
        self._plays: list[dict] = []

    def update(self, actions: list[dict]) -> None:
        """Actualiza con la lista COMPLETA de actions (idempotente).

        Se llama con todas las actions cada poll — reemplaza el estado anterior.
        Esto es más robusto que acumular incrementalmente (evita duplicados).
        """
        plays = []
        for act in actions:
            period = act.get("period", 0)
            if period < 1 or period > 6:
                continue  # Permitir hasta 2OT; features OT acumulan sobre Q4

            # Parsear scores (vienen como strings)
            try:
                home_score = int(act.get("scoreHome", 0))
                away_score = int(act.get("scoreAway", 0))
            except (ValueError, TypeError):
                continue

            team_tri = (act.get("teamTricode") or "").upper()
            action_type = (act.get("actionType") or "").lower()
            sub_type = (act.get("subType") or "").lower()
            description = (act.get("description") or "").lower()
            clock = act.get("clock", "")

            is_home = team_tri == self._home_tri
            is_away = team_tri == self._away_tri

            plays.append({
                "period": period,
                "clock": clock,
                "clock_seconds": _parse_clock_seconds(clock),
                "home_score": home_score,
                "away_score": away_score,
                "action_type": action_type,
                "sub_type": sub_type,
                "description": description,
                "is_home": is_home,
                "is_away": is_away,
            })

        self._plays = plays

    def get_features(self, period_end: int) -> dict:
        """Retorna dict con 20 PBP features acumulativas hasta period_end.

        17 PBP originales + 3 score-context features para corregir home bias:
        - SCORE_DIFF_NORMALIZED: score_diff / max(total_points, 1) — contexto de ventaja
        - BLOWOUT_FLAG: 1.0 si abs(score_diff) > 15 — indica juego fuera de alcance
        - SCORE_DIFF_X_MOMENTUM: momentum * sigmoid(score_diff/10) — interaccion que
          reduce el peso del momentum cuando el juego ya esta decidido
        """
        # Filtrar plays hasta el periodo solicitado
        plays = [p for p in self._plays if p["period"] <= period_end]

        if not plays:
            return self._empty_features()

        # --- Lead changes ---
        diffs = [p["home_score"] - p["away_score"] for p in plays]
        lead_changes = self._count_lead_changes(diffs)

        # --- Largest leads ---
        largest_home = max(diffs) if diffs else 0
        largest_away = max(-d for d in diffs) if diffs else 0

        # --- Scoring runs ---
        scores = [(p["home_score"], p["away_score"]) for p in plays]
        home_run_max, away_run_max = self._compute_max_scoring_runs(scores)

        # --- Event counts ---
        timeouts_home = sum(1 for p in plays if p["action_type"] == "timeout" and p["is_home"])
        timeouts_away = sum(1 for p in plays if p["action_type"] == "timeout" and p["is_away"])
        fouls_home = sum(1 for p in plays if p["action_type"] == "foul" and p["is_home"])
        fouls_away = sum(1 for p in plays if p["action_type"] == "foul" and p["is_away"])
        turnovers_home = sum(1 for p in plays if p["action_type"] == "turnover" and p["is_home"])
        turnovers_away = sum(1 for p in plays if p["action_type"] == "turnover" and p["is_away"])

        # 3PT made: actionType="3pt" con shotResult="Made" o subType contiene "made"
        fg3_home = sum(
            1 for p in plays
            if p["action_type"] == "3pt" and "made" in p["sub_type"] and p["is_home"]
        )
        fg3_away = sum(
            1 for p in plays
            if p["action_type"] == "3pt" and "made" in p["sub_type"] and p["is_away"]
        )

        # Offensive rebounds
        oreb_home = sum(
            1 for p in plays
            if p["action_type"] == "rebound" and "offensive" in p["sub_type"] and p["is_home"]
        )
        oreb_away = sum(
            1 for p in plays
            if p["action_type"] == "rebound" and "offensive" in p["sub_type"] and p["is_away"]
        )

        # --- Momentum (últimas 10 plays con scoring vs previas 10) ---
        momentum = self._compute_momentum(diffs)

        # --- Last 5 min diff del cuarto actual ---
        current_q_plays = [p for p in plays if p["period"] == period_end]
        last_5_diff = self._compute_last_5min_diff(current_q_plays)

        # --- Score context features (fix home bias) ---
        last_home = plays[-1]["home_score"]
        last_away = plays[-1]["away_score"]
        score_diff = last_home - last_away
        total_pts = last_home + last_away

        # Normalized score differential: [-1, 1] range
        score_diff_norm = score_diff / max(total_pts, 1)

        # Blowout flag: game is out of reach
        blowout = 1.0 if abs(score_diff) > 15 else 0.0

        # Interaction: momentum attenuated by score context
        # sigmoid(score_diff/10) maps diff to [0,1] — when home leads big,
        # momentum from home is expected (less signal); when home trails,
        # positive momentum is more meaningful
        sigmoid_diff = 1.0 / (1.0 + math.exp(-score_diff / 10.0))
        # When home leads big (sigmoid~1), positive momentum means less
        # When home trails (sigmoid~0), positive momentum means more
        # Flip: multiply momentum by (1 - sigmoid) so momentum matters more when trailing
        score_diff_x_momentum = momentum * (1.0 - sigmoid_diff)

        return {
            "PBP_LEAD_CHANGES": lead_changes,
            "PBP_LARGEST_LEAD_HOME": largest_home,
            "PBP_LARGEST_LEAD_AWAY": largest_away,
            "PBP_HOME_RUNS_MAX": home_run_max,
            "PBP_AWAY_RUNS_MAX": away_run_max,
            "PBP_TIMEOUTS_HOME": timeouts_home,
            "PBP_TIMEOUTS_AWAY": timeouts_away,
            "PBP_FOULS_HOME": fouls_home,
            "PBP_FOULS_AWAY": fouls_away,
            "PBP_TURNOVERS_HOME": turnovers_home,
            "PBP_TURNOVERS_AWAY": turnovers_away,
            "PBP_MOMENTUM": momentum,
            "PBP_LAST_5MIN_DIFF": last_5_diff,
            "PBP_FG3_MADE_HOME": fg3_home,
            "PBP_FG3_MADE_AWAY": fg3_away,
            "PBP_OREB_HOME": oreb_home,
            "PBP_OREB_AWAY": oreb_away,
            "PBP_SCORE_DIFF_NORM": score_diff_norm,
            "PBP_BLOWOUT_FLAG": blowout,
            "PBP_SCORE_DIFF_X_MOMENTUM": score_diff_x_momentum,
        }

    # --- Funciones internas (réplica de create_pbp_features.py) ---

    @staticmethod
    def _count_lead_changes(diffs: list[int]) -> int:
        """Cuenta cambios de liderazgo (signo del diferencial cambia)."""
        signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in diffs]
        nonzero = [s for s in signs if s != 0]
        if len(nonzero) < 2:
            return 0
        changes = sum(1 for i in range(1, len(nonzero)) if nonzero[i] != nonzero[i - 1])
        return changes

    @staticmethod
    def _compute_max_scoring_runs(scores: list[tuple[int, int]]) -> tuple[int, int]:
        """Calcula racha de puntos más larga para cada equipo."""
        if len(scores) < 2:
            return 0, 0

        home_run = 0
        away_run = 0
        home_max = 0
        away_max = 0
        prev_h, prev_a = scores[0]

        for h, a in scores[1:]:
            h_scored = h - prev_h
            a_scored = a - prev_a

            if h_scored > 0:
                home_run += h_scored
                away_run = 0
                home_max = max(home_max, home_run)

            if a_scored > 0:
                away_run += a_scored
                home_run = 0
                away_max = max(away_max, away_run)

            prev_h, prev_a = h, a

        return home_max, away_max

    @staticmethod
    def _compute_momentum(diffs: list[int]) -> float:
        """Momentum: diff de últimas 10 plays vs previas 10."""
        if len(diffs) < 4:
            return 0.0

        if len(diffs) >= 20:
            recent_diff = diffs[-1] - diffs[-11]
            prev_diff = diffs[-11] - diffs[-21] if len(diffs) >= 21 else diffs[-11] - diffs[0]
        else:
            mid = len(diffs) // 2
            recent_diff = diffs[-1] - diffs[mid]
            prev_diff = diffs[mid] - diffs[0]

        return float(recent_diff - prev_diff)

    @staticmethod
    def _compute_last_5min_diff(quarter_plays: list[dict]) -> int:
        """Diferencial de puntos en los últimos 5 min del cuarto."""
        if not quarter_plays:
            return 0

        # Filtrar plays donde clock <= 5:00 (300 segundos)
        last_5 = [p for p in quarter_plays if p["clock_seconds"] <= 300.0]
        if not last_5:
            last_5 = quarter_plays

        start_h = last_5[0]["home_score"]
        start_a = last_5[0]["away_score"]
        end_h = last_5[-1]["home_score"]
        end_a = last_5[-1]["away_score"]

        return int((end_h - start_h) - (end_a - start_a))

    @staticmethod
    def _empty_features() -> dict:
        """Features vacías (todo cero) cuando no hay plays."""
        return {
            "PBP_LEAD_CHANGES": 0,
            "PBP_LARGEST_LEAD_HOME": 0,
            "PBP_LARGEST_LEAD_AWAY": 0,
            "PBP_HOME_RUNS_MAX": 0,
            "PBP_AWAY_RUNS_MAX": 0,
            "PBP_TIMEOUTS_HOME": 0,
            "PBP_TIMEOUTS_AWAY": 0,
            "PBP_FOULS_HOME": 0,
            "PBP_FOULS_AWAY": 0,
            "PBP_TURNOVERS_HOME": 0,
            "PBP_TURNOVERS_AWAY": 0,
            "PBP_MOMENTUM": 0.0,
            "PBP_LAST_5MIN_DIFF": 0,
            "PBP_FG3_MADE_HOME": 0,
            "PBP_FG3_MADE_AWAY": 0,
            "PBP_OREB_HOME": 0,
            "PBP_OREB_AWAY": 0,
            "PBP_SCORE_DIFF_NORM": 0.0,
            "PBP_BLOWOUT_FLAG": 0.0,
            "PBP_SCORE_DIFF_X_MOMENTUM": 0.0,
        }
