"""Reporting and visualization utilities for TFT League standings and agent evaluations."""

from __future__ import annotations

from typing import Any
from tft_ai_player.rl.league.league_manager import LeagueManager


def generate_league_markdown_report(league: LeagueManager) -> str:
    """Generate a GitHub Flavored Markdown summary report of the current League state."""
    leaderboard = league.get_leaderboard()

    lines: list[str] = []
    lines.append("# 🏆 TFT AI League Standings & Leaderboard")
    lines.append("")
    lines.append(f"**Total Registered Agents:** {len(league.profiles)} | **Total Matches Played:** {len(league.match_history)}")
    lines.append("")
    lines.append("| Rank | Agent Name | Role | Elo Rating | Matches | Top 4 Rate | Win Rate (1st) | Avg Placement |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for entry in leaderboard:
        role_badge = f"`{entry['role']}`"
        lines.append(
            f"| **#{entry['rank']}** | {entry['name']} | {role_badge} | **{entry['rating']:.1f}** | {entry['games']} | {entry['top4_rate']} | {entry['win_rate']} | {entry['avg_placement']} |"
        )

    lines.append("")
    lines.append("### ⚔️ Recent Match History")
    lines.append("")
    if not league.match_history:
        lines.append("*No matches recorded yet.*")
    else:
        recent_matches = league.match_history[-5:]
        for m in reversed(recent_matches):
            sorted_placements = sorted(m.placements.items(), key=lambda item: item[1])
            podium = [f"#{rank} {aid} ({m.elo_deltas.get(aid, 0.0):+0.1f})" for aid, rank in sorted_placements[:4]]
            lines.append(f"- **Match `{m.match_id}`** (Seed {m.seed}, {m.total_rounds} rds): Top 4: " + ", ".join(podium))

    return "\n".join(lines)


def print_league_terminal_summary(league: LeagueManager) -> None:
    """Print clean ASCII summary of the league to stdout."""
    leaderboard = league.get_leaderboard()
    print("=" * 80)
    print(f" {'TFT MULTI-AGENT LEAGUE LEADERBOARD':^76} ")
    print("=" * 80)
    print(f" {'Rank':<5} | {'Agent Name':<24} | {'Role':<14} | {'Elo':<7} | {'Games':<5} | {'Top4%':<7} | {'Avg#':<5}")
    print("-" * 80)
    for entry in leaderboard:
        print(
            f" #{entry['rank']:<4} | {entry['name'][:24]:<24} | {entry['role'][:14]:<14} | {entry['rating']:<7.1f} | {entry['games']:<5} | {entry['top4_rate']:<7} | {entry['avg_placement']:<5}"
        )
    print("=" * 80)
