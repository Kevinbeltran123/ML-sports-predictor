"""
Formatters: prediction dicts -> Telegram message strings.

Public functions:
  format_pregame_message(blocks, sportsbook) -> str
  format_h1_message(h1_results)              -> str
  format_ingame_update(...)                  -> str
  format_daily_summary(picks, final_scores)  -> str

Uses plain text (no MarkdownV2) — avoids escaping hell.
Monospace blocks via triple backtick aren't supported in plain mode,
so we use clean formatting that reads well in Telegram.
"""
from datetime import datetime


# ──────────────────────────────────────────
# Tags
# ──────────────────────────────────────────

_TAG_MAP = {"BET": "BET", "SKIP": "SKIP", "TRAP": "TRAP", "PASS": "PASS"}
_AH_TAG_EMOJI = {"AH-BET": "✅", "AH-SKIP": "⏭️", "AH-PASS": "❌"}
CONF2_KELLY_THRESHOLD = 0.5


def _get_tag(b: dict) -> str:
    """Derive BET/SKIP/TRAP/PASS tag from a block dict."""
    max_kelly = max(b.get("kelly_home", 0), b.get("kelly_away", 0))
    conf_ss = b.get("conf_ss")
    conf_uncertain = conf_ss is not None and conf_ss != 1
    conf_override = conf_uncertain and max_kelly >= CONF2_KELLY_THRESHOLD
    has_trap = b.get("trap_home", False) or b.get("trap_away", False)

    if has_trap:
        return "TRAP"
    if conf_uncertain and not conf_override:
        return "SKIP"
    if b.get("max_ev", 0) > 0:
        return "BET"
    return "PASS"


def _tag_emoji(tag: str) -> str:
    return {"BET": "✅", "SKIP": "⏭️", "TRAP": "⚠️", "PASS": "❌"}.get(tag, "•")


def _short(team: str) -> str:
    """Last word of team name: 'Los Angeles Lakers' -> 'Lakers'."""
    return team.split()[-1] if team else team


# ──────────────────────────────────────────
# Pregame
# ──────────────────────────────────────────

def format_pregame_message(blocks: list[dict], sportsbook: str = "fanduel") -> str:
    """Format pregame blocks from _build_game_blocks() for Telegram.

    Args:
        blocks: list of block dicts, already EV-sorted desc.
        sportsbook: for header display.

    Returns:
        Plain-text formatted string.
    """
    now = datetime.now().strftime("%I:%M %p ET")
    date_str = datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"🏀 NBA PICKS — {date_str} {now}",
        f"Book: {sportsbook.upper()} | Ranked by EV",
        "━" * 35,
    ]

    for rank, b in enumerate(blocks, 1):
        tag = _get_tag(b)
        emoji = _tag_emoji(tag)
        pick = _short(b["pick"])
        away = _short(b["away"])
        home = _short(b["home"])
        prob = b["pick_prob"] * 100

        # Matchup header
        lines.append("")
        lines.append(f"{emoji} #{rank} [{tag}] {pick} ({prob:.1f}%)")
        lines.append(f"   {home} vs {away}")

        # ML line
        ev_side = "home" if b["ev_home"] >= b["ev_away"] else "away"
        ev_val = b[f"ev_{ev_side}"]
        kelly_val = b[f"kelly_{ev_side}"]
        agree = "✓" if b["xgb_agree"] else "~"
        sigma = b["sigma"]
        lines.append(
            f"   ML  EV={ev_val:+.1f}%  K={kelly_val:.2f}%  "
            f"σ={sigma:.3f}  {agree}XGB:{b['xgb_conf']}% Cat:{b['cat_conf']}%"
        )

        # AH spread with tag
        ah_tag = b.get("ah_tag", "AH-PASS")
        ah_emoji = _AH_TAG_EMOJI.get(ah_tag, "•")
        if b.get("ah_side"):
            ah_side = _short(b["ah_side"])
            blend = " (REG)" if b.get("ah_blend") == "REG+CLF" else ""
            lines.append(
                f"   AH {ah_emoji}[{ah_tag}] {ah_side} {b['ah_line']}  "
                f"P={b['ah_p']:.1%}  EV={b['ah_ev']:+.1f}%  K={b['ah_kelly']:.2f}%{blend}"
            )
            if ah_tag == "AH-SKIP" and b.get("ah_skip_reasons"):
                lines.append(f"   ⚠️ {', '.join(b['ah_skip_reasons'])}")

        # O/U
        ou_sym = "↑" if b["ou_label"] == "OVER" else "↓"
        pred_str = f"  pred={int(b['predicted_total'])}" if b.get("predicted_total") else ""
        lines.append(f"   O/U {ou_sym}{b['ou_label']} {b['ou_line']} ({b['ou_conf']}%){pred_str}")

        # Margin regression
        if b.get("reg_margin") is not None:
            reg_m = b["reg_margin"]
            reg_p = b.get("reg_p_cover", 0)
            conf_flag = ""
            if b.get("reg_confident") is True:
                conf_flag = " ✓conf"
            elif b.get("reg_confident") is False:
                conf_flag = " ~conf"
            lines.append(f"   REG margin={reg_m:+.1f}  P(cover)={reg_p:.1%}{conf_flag}")

        # Conformal
        conf_ss = b.get("conf_ss")
        if conf_ss == 1:
            lines.append(f"   ✓ conformal margin={b.get('conf_margin', 0):.2f}")
        elif conf_ss is not None and conf_ss != 1:
            max_k = max(b.get("kelly_home", 0), b.get("kelly_away", 0))
            if max_k >= CONF2_KELLY_THRESHOLD:
                lines.append(f"   ⚡ conformal override (K={max_k:.2f}%)")
            else:
                lines.append(f"   ⏭️ skip (conformal set_size={conf_ss})")

        # Trap warning
        if b.get("trap_home") or b.get("trap_away"):
            lines.append("   ⚠️ underdog trap — use AH favorite")

    lines.append("")
    lines.append(f"━ {date_str}")
    return "\n".join(lines)


# ──────────────────────────────────────────
# First Half (H1)
# ──────────────────────────────────────────

def format_h1_message(h1_results: list[dict]) -> str:
    """Format H1 moneyline results for Telegram."""
    date_str = datetime.now().strftime("%Y-%m-%d %I:%M %p ET")
    lines = [
        f"🏀 1H MONEYLINE — {date_str}",
        "━" * 35,
    ]

    for r in h1_results:
        home = _short(r["home_team"])
        away = _short(r["away_team"])
        cs = r.get("h1_conformal_set_size", 0)
        tag = r.get("h1_tag", "BET" if cs == 1 else ("SKIP" if cs == 2 else "---"))
        emoji = _tag_emoji(tag)

        p_home = r.get("h1_prob_home", 0.5)
        p_away = r.get("h1_prob_away", 0.5)
        pick = r.get("h1_pick", home if p_home >= 0.5 else away)
        pick_prob = max(p_home, p_away)

        ev = r.get("h1_ev")
        kelly = r.get("h1_kelly", 0.0)

        ev_str = f"  EV={ev:+.1f}%" if ev is not None else ""
        kelly_str = f"  K={kelly:.2f}%" if kelly else ""

        h_odds = r.get("h1_ml_home_odds")
        a_odds = r.get("h1_ml_away_odds")
        odds_str = f"  [{h_odds:+d}/{a_odds:+d}]" if h_odds and a_odds else ""

        agree = "✓" if r.get("h1_models_agree", False) else "~"

        lines.append(
            f"{emoji} [{tag}] {pick} {pick_prob:.1%} vs {away}{odds_str}{ev_str}{kelly_str} {agree}"
        )

    return "\n".join(lines)


# ──────────────────────────────────────────
# In-Game Quarter Update
# ──────────────────────────────────────────

def format_ingame_update(
    home_team: str,
    away_team: str,
    period: int,
    home_score: int,
    away_score: int,
    p_pregame: float,
    p_adjusted: float,
    delta: float,
    conformal_set_size: int,
    model_used: str,
) -> str:
    """Format a single quarter-end in-game update."""
    q_label = f"Q{period}"
    home = _short(home_team)
    away = _short(away_team)
    score_diff = home_score - away_score

    conf_label = (
        "🟢HIGH" if conformal_set_size == 1
        else ("🟡LOW" if conformal_set_size == 2
              else "⚪N/A")
    )

    if score_diff > 0:
        leader = f"+{score_diff} {home}"
    elif score_diff < 0:
        leader = f"+{abs(score_diff)} {away}"
    else:
        leader = "TIE"

    delta_emoji = "📈" if delta > 0.03 else ("📉" if delta < -0.03 else "➡️")

    lines = [
        f"🏀 {q_label} END — {home} vs {away}",
        f"Score: {home_score}-{away_score}  ({leader})",
        f"Pre: {p_pregame:.1%} → Live: {p_adjusted:.1%}  {delta_emoji}{delta:+.1%}",
        f"Conf: {conf_label}  Model: {model_used}",
    ]

    return "\n".join(lines)


# ──────────────────────────────────────────
# Daily Summary
# ──────────────────────────────────────────

def format_daily_summary(
    picks: list[dict],
    final_scores: list[dict],
) -> str:
    """Format end-of-day summary with results and record.

    Args:
        picks: pregame predictions (from ensemble_runner), each with
               home_team, away_team, prob_home, kelly_home, kelly_away, ev_home, ev_away.
        final_scores: list of dicts with game_id, home_team, away_team,
                      home_score, away_score from get_live_scoreboard().

    Returns:
        Plain-text summary message.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"📊 DAILY SUMMARY — {date_str}",
        "━" * 35,
    ]

    # Build score lookup: (home_short, away_short) -> (home_score, away_score)
    score_map = {}
    for g in final_scores:
        h = _short(g.get("home_team", g.get("homeTeam", {}).get("teamName", "")))
        a = _short(g.get("away_team", g.get("awayTeam", {}).get("teamName", "")))
        hs = g.get("home_score", 0)
        as_ = g.get("away_score", 0)
        score_map[(h, a)] = (hs, as_)

    wins = 0
    losses = 0
    units_pl = 0.0

    for p in picks:
        home = _short(p["home_team"])
        away = _short(p["away_team"])
        prob_home = p.get("prob_home", 0.5)
        pick_home = prob_home >= 0.5
        pick_team = home if pick_home else away

        ev_home = p.get("ev_home", 0)
        ev_away = p.get("ev_away", 0)
        kelly = p.get("kelly_home", 0) if pick_home else p.get("kelly_away", 0)

        # Find matching score
        scores = score_map.get((home, away))
        if scores is None:
            lines.append(f"  {home} vs {away}: Pick {pick_team} — no final score")
            continue

        hs, as_ = scores
        home_won = hs > as_

        correct = (pick_home and home_won) or (not pick_home and not home_won)
        result_emoji = "✅" if correct else "❌"

        if correct:
            wins += 1
        else:
            losses += 1

        # P&L in units (kelly = % of bankroll)
        if kelly > 0:
            if correct:
                # Simplified: assume -110 odds -> profit = kelly * 0.91
                units_pl += kelly * 0.91
            else:
                units_pl -= kelly

        lines.append(
            f"  {result_emoji} {home} {hs} - {as_} {away}  |  Pick: {pick_team}  K={kelly:.2f}%"
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
