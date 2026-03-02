"""
MLB Telegram formatters: prediction dicts -> plain text messages.

Public functions:
  format_mlb_pregame_message(blocks, sportsbook)  -> str
  format_mlb_f5_message(f5_results)               -> str
  format_mlb_inning_update(...)                    -> str
  format_mlb_daily_summary(picks, final_scores)    -> str
"""
from datetime import datetime


def _short(team: str) -> str:
    """Last word: 'New York Yankees' -> 'Yankees'."""
    return team.split()[-1] if team else team


def _tag_emoji(tag: str) -> str:
    return {"BET": "✅", "SKIP": "⏭️", "TRAP": "⚠️", "PASS": "❌"}.get(tag, "•")


# ──────────────────────────────────────────
# Pregame
# ──────────────────────────────────────────

def format_mlb_pregame_message(blocks: list[dict], sportsbook: str = "fanduel") -> str:
    """Format MLB pregame picks for Telegram (plain text)."""
    now = datetime.now().strftime("%I:%M %p")
    date_str = datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"⚾ MLB PICKS — {date_str} {now}",
        f"Book: {sportsbook.upper()} | Ranked by EV",
        "━" * 35,
    ]

    for rank, b in enumerate(blocks, 1):
        tag = b.get("tag", "PASS")
        emoji = _tag_emoji(tag)
        pick = _short(b.get("pick", ""))
        home = _short(b.get("home", ""))
        away = _short(b.get("away", ""))
        prob = b.get("pick_prob", 0.5) * 100
        sp_home = b.get("sp_home", "TBD")
        sp_away = b.get("sp_away", "TBD")

        lines.append("")
        lines.append(f"{emoji} #{rank} [{tag}] {pick} ({prob:.1f}%)")
        lines.append(f"   {away} @ {home}")
        lines.append(f"   SP: {sp_away} vs {sp_home}")

        # ML line
        ev_side = "home" if b.get("ev_home", 0) >= b.get("ev_away", 0) else "away"
        ev_val = b.get(f"ev_{ev_side}", 0)
        kelly_val = b.get(f"kelly_{ev_side}", 0)
        sigma = b.get("sigma", 0)
        agree = "✓" if b.get("xgb_agree", False) else "~"
        lines.append(
            f"   ML  EV={ev_val:+.1f}%  K={kelly_val:.2f}%  "
            f"σ={sigma:.3f}  {agree}XGB:{b.get('xgb_conf', 0)}% Cat:{b.get('cat_conf', 0)}%"
        )

        # F5 line
        f5_ev = b.get("f5_ev")
        if f5_ev is not None:
            f5_prob = b.get("f5_prob", 0) * 100
            f5_kelly = b.get("f5_kelly", 0)
            f5_tag = b.get("f5_tag", "---")
            lines.append(
                f"   F5  {pick} {f5_prob:.1f}%  EV={f5_ev:+.1f}%  K={f5_kelly:.2f}%  [{f5_tag}]"
            )

        # O/U line
        ou_label = b.get("ou_label")
        if ou_label:
            ou_sym = "↑" if ou_label == "OVER" else "↓"
            ou_line = b.get("ou_line", "?")
            ou_conf = b.get("ou_conf", 0)
            ou_ev = b.get("ou_ev", 0)
            pred_total = b.get("predicted_total")
            pred_str = f"  pred={pred_total:.1f}" if pred_total else ""
            lines.append(
                f"   O/U {ou_sym}{ou_label} {ou_line} ({ou_conf}%)  EV={ou_ev:+.1f}%{pred_str}"
            )

        # Run line (AH equivalent)
        if b.get("rl_side"):
            rl_side = _short(b["rl_side"])
            lines.append(
                f"   RL  {rl_side} {b.get('rl_line', '-1.5')}  "
                f"EV={b.get('rl_ev', 0):+.1f}%  K={b.get('rl_kelly', 0):.2f}%"
            )

        # Park + weather
        pf = b.get("park_factor")
        temp = b.get("temp")
        wind = b.get("wind_desc")
        if pf or temp:
            parts = []
            if pf:
                parts.append(f"PF={pf}")
            if temp:
                parts.append(f"{temp:.0f}°F")
            if wind:
                parts.append(wind)
            lines.append(f"   Park: {home}  {' '.join(parts)}")

        # Conformal
        conf_ss = b.get("conf_ss")
        if conf_ss == 1:
            lines.append(f"   ✓ conformal margin={b.get('conf_margin', 0):.2f}")
        elif conf_ss is not None and conf_ss != 1:
            lines.append(f"   ⏭️ skip (conformal set_size={conf_ss})")

    lines.append("")
    lines.append(f"━ {date_str}")
    return "\n".join(lines)


# ──────────────────────────────────────────
# F5 (First 5 Innings)
# ──────────────────────────────────────────

def format_mlb_f5_message(f5_results: list[dict]) -> str:
    """Format F5 predictions for Telegram."""
    date_str = datetime.now().strftime("%Y-%m-%d %I:%M %p")
    lines = [
        f"⚾ F5 MONEYLINE — {date_str}",
        "━" * 35,
    ]

    for r in f5_results:
        home = _short(r.get("home_team", ""))
        away = _short(r.get("away_team", ""))
        tag = r.get("f5_tag", "---")
        emoji = _tag_emoji(tag)

        f5_prob = r.get("f5_prob_home", 0.5)
        pick = home if f5_prob >= 0.5 else away
        pick_prob = f5_prob if f5_prob >= 0.5 else (1 - f5_prob)

        ev = r.get("f5_ev")
        kelly = r.get("f5_kelly", 0)
        sp_home = r.get("sp_home", "TBD")
        sp_away = r.get("sp_away", "TBD")

        ev_str = f"  EV={ev:+.1f}%" if ev is not None else ""
        kelly_str = f"  K={kelly:.2f}%" if kelly else ""

        lines.append(
            f"{emoji} [{tag}] {pick} {pick_prob:.1%} | {sp_away} vs {sp_home}{ev_str}{kelly_str}"
        )

    return "\n".join(lines)


# ──────────────────────────────────────────
# In-Game Inning Update
# ──────────────────────────────────────────

def format_mlb_inning_update(
    home_team: str,
    away_team: str,
    inning: int,
    inning_half: str,
    home_score: int,
    away_score: int,
    p_pregame: float,
    p_adjusted: float,
    delta: float,
    pitcher_change: bool = False,
    current_pitcher: str = None,
) -> str:
    """Format an inning-end in-game update."""
    home = _short(home_team)
    away = _short(away_team)
    half_label = "TOP" if inning_half == "top" else "BOT"
    score_diff = home_score - away_score

    if score_diff > 0:
        leader = f"+{score_diff} {home}"
    elif score_diff < 0:
        leader = f"+{abs(score_diff)} {away}"
    else:
        leader = "TIE"

    delta_emoji = "📈" if delta > 0.03 else ("📉" if delta < -0.03 else "➡️")

    lines = [
        f"⚾ {half_label} {inning} — {away} @ {home}",
        f"Score: {away_score}-{home_score}  ({leader})",
        f"Pre: {p_pregame:.1%} → Live: {p_adjusted:.1%}  {delta_emoji}{delta:+.1%}",
    ]

    if pitcher_change and current_pitcher:
        lines.append(f"🔄 Pitcher change: {current_pitcher}")

    return "\n".join(lines)


# ──────────────────────────────────────────
# Daily Summary
# ──────────────────────────────────────────

def format_mlb_daily_summary(
    picks: list[dict],
    final_scores: list[dict],
) -> str:
    """End-of-day summary with results, record, and P&L."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"📊 MLB SUMMARY — {date_str}",
        "━" * 35,
    ]

    score_map = {}
    for g in final_scores:
        h = _short(g.get("home_team", ""))
        a = _short(g.get("away_team", ""))
        score_map[(h, a)] = (g.get("home_score", 0), g.get("away_score", 0))

    wins = losses = 0
    units_pl = 0.0

    for p in picks:
        home = _short(p.get("home_team", p.get("home", "")))
        away = _short(p.get("away_team", p.get("away", "")))
        prob_home = p.get("prob_home", 0.5)
        pick_home = prob_home >= 0.5
        pick_team = home if pick_home else away
        kelly = p.get("kelly_home", 0) if pick_home else p.get("kelly_away", 0)

        scores = score_map.get((home, away))
        if scores is None:
            lines.append(f"  {away} @ {home}: Pick {pick_team} — no final score")
            continue

        hs, as_ = scores
        home_won = hs > as_
        correct = (pick_home and home_won) or (not pick_home and not home_won)
        result_emoji = "✅" if correct else "❌"

        if correct:
            wins += 1
        else:
            losses += 1

        if kelly > 0:
            if correct:
                units_pl += kelly * 0.91  # approximate -110 profit
            else:
                units_pl -= kelly

        lines.append(
            f"  {result_emoji} {away} {as_} @ {home} {hs}  |  Pick: {pick_team}  K={kelly:.2f}%"
        )

    total = wins + losses
    if total > 0:
        pct = wins / total * 100
        lines.append("")
        lines.append(f"Record: {wins}-{losses} ({pct:.1f}%)")
        if units_pl != 0:
            sign = "+" if units_pl > 0 else ""
            lines.append(f"P&L: {sign}{units_pl:.2f} units")
    else:
        lines.append("\nNo matched games.")

    return "\n".join(lines)
