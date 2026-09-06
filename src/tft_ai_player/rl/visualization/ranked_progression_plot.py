"""Ranked Ladder Progression Plotter (AlphaStar style).

Generates an AlphaStar-style ranked progression plot:
- X-axis: Training Generations (or Total Games)
- Y-axis: Categorical Ranked Ladder Tiers & Divisions (Bronze IV ... Challenger)
- Discrete checkpoints & progression curve mimicking DeepMind's AlphaStar league figure.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

logger = logging.getLogger(__name__)

# Full ordered ranked ladder hierarchy
RANK_TIERS = [
    "IRON 4", "IRON 3", "IRON 2", "IRON 1",
    "BRONZE 4", "BRONZE 3", "BRONZE 2", "BRONZE 1",
    "SILVER 4", "SILVER 3", "SILVER 2", "SILVER 1",
    "GOLD 4", "GOLD 3", "GOLD 2", "GOLD 1",
    "PLATINUM 4", "PLATINUM 3", "PLATINUM 2", "PLATINUM 1",
    "EMERALD 4", "EMERALD 3", "EMERALD 2", "EMERALD 1",
    "DIAMOND 4", "DIAMOND 3", "DIAMOND 2", "DIAMOND 1",
    "MASTER",
    "GRANDMASTER",
    "CHALLENGER",
]

# Absolute LP Milestones (Standard 400 LP per tier from Iron 4 = 0 LP)
# Iron IV: 0 LP | Bronze IV: 400 LP | Silver IV: 800 LP | Gold IV: 1200 LP | Plat IV: 1600 LP
# Emerald IV: 2000 LP | Diamond IV: 2400 LP | Master: 2800 LP | GM: 3050 LP (+250) | Challenger: 3300 LP (+500)
TIER_BASE_ABSOLUTE_LP: dict[str, float] = {
    "IRON": 0.0,
    "BRONZE": 400.0,
    "SILVER": 800.0,
    "GOLD": 1200.0,
    "PLATINUM": 1600.0,
    "EMERALD": 2000.0,
    "DIAMOND": 2400.0,
    "MASTER": 2800.0,
    "GRANDMASTER": 3050.0,
    "CHALLENGER": 3300.0,
}


def rank_to_absolute_lp(tier: str, division: int = 4, lp: int = 0) -> float:
    """Calculate cumulative absolute LP from Iron IV (0 LP) to Challenger."""
    t = tier.upper().strip()
    if t in ("MASTER", "GRANDMASTER", "CHALLENGER"):
        # In Master+, LP is continuous on top of Master threshold (2800 LP)
        return TIER_BASE_ABSOLUTE_LP["MASTER"] + float(lp)

    base = TIER_BASE_ABSOLUTE_LP.get(t, 1200.0)
    # Each division (IV -> III -> II -> I) represents 100 LP within the tier
    division_offset = (4 - max(1, min(4, division))) * 100.0
    return base + division_offset + float(np.clip(lp, 0, 100))


def parse_console_training_log(log_path: str | Path) -> list[dict[str, Any]]:
    """Extract generational progression from the training console log."""
    path = Path(log_path)
    if not path.exists():
        return []

    # Regex matching lines like:
    # [Gen 0001/300] Rank: GOLD 2 (50 LP)   | Reward: +32.204 | Avg Place: 3.19 | Top-4: 75.0%
    # [Gen 0004/300] Rank: MASTER (155 LP)  | Reward: +28.130 | Avg Place: 3.10 | Top-4: 75.9%
    pattern = re.compile(
        r"\[Gen\s+(\d+)/\d+\]\s+Rank:\s+([A-Z]+)(?:\s+(\d))?\s+\((\d+)\s+LP\)\s+\|\s+Reward:\s+([+-]?\d+\.\d+)\s+\|\s+Avg Place:\s+(\d+\.\d+)\s+\|\s+Top-4:\s+(\d+\.\d+)%"
    )

    records = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    for match in pattern.finditer(text):
        gen = int(match.group(1))
        tier = match.group(2)
        div = int(match.group(3)) if match.group(3) is not None else 1
        lp = int(match.group(4))
        rew = float(match.group(5))
        place = float(match.group(6))
        top4 = float(match.group(7))

        y_coord = rank_to_absolute_lp(tier, div, lp)
        records.append({
            "generation": gen,
            "tier": tier,
            "division": div,
            "lp": lp,
            "y_coord": y_coord,
            "reward": rew,
            "avg_place": place,
            "top4": top4,
            "rank_label": f"{tier} {div} ({lp} LP)" if tier not in ("MASTER", "GRANDMASTER", "CHALLENGER") else f"{tier} ({lp} LP)",
        })

    return records


def generate_alphastar_ranked_ladder_plot(
    log_path: str | Path,
    output_png: str | Path = "reports/visualizations/ranked_ladder_progression.png",
    agent_name: str = "AlphaStar v6.2",
) -> Path:
    """Generate the DeepMind AlphaStar-style ranked ladder progression figure."""
    records = parse_console_training_log(log_path)
    out_file = Path(output_png)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Initial starting point at Gen 0: Gold IV (0 LP) -> 1200 Absolute LP
    gens = [0] + [r["generation"] for r in records]
    y_vals = [rank_to_absolute_lp("GOLD", 4, 0)] + [r["y_coord"] for r in records]
    ranks = ["GOLD 4 (0 LP)"] + [r["rank_label"] for r in records]

    # Major League Milestones for left-side tick marks
    major_tiers = [
        ("BRONZE", "Bronze (400 LP)", 400.0, "#cd7f32"),
        ("SILVER", "Silver (800 LP)", 800.0, "#95a5a6"),
        ("GOLD", "Gold (1200 LP)", 1200.0, "#f1c40f"),
        ("PLATINUM", "Platinum (1600 LP)", 1600.0, "#1abc9c"),
        ("EMERALD", "Emerald (2000 LP)", 2000.0, "#2ecc71"),
        ("DIAMOND", "Diamond (2400 LP)", 2400.0, "#3498db"),
        ("MASTER", "Master (2800 LP)", 2800.0, "#9b59b6"),
        ("GRANDMASTER", "Grandmaster (3050 LP)", 3050.0, "#e67e22"),
        ("CHALLENGER", "Challenger (3300 LP)", 3300.0, "#e74c3c"),
    ]

    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
    fig.patch.set_facecolor("#0b0e14")
    ax.set_facecolor("#111622")

    # Draw league horizontal demarcation bands
    for i in range(len(major_tiers)):
        key, name, y_pos, color = major_tiers[i]
        next_y = major_tiers[i+1][2] if i + 1 < len(major_tiers) else y_pos + 600.0
        
        # Subtle horizontal separator
        ax.axhline(y=y_pos, color="#252f44", linestyle="--", linewidth=0.8, alpha=0.7)
        # Subtle tier background tint alternating
        if i % 2 == 1:
            ax.axhspan(y_pos, next_y, color="#ffffff", alpha=0.02)

    # Plot progression line with AlphaStar gradient glow
    ax.plot(gens, y_vals, color="#a855f7", linewidth=3.2, zorder=4, label=f"{agent_name} Cumulative LP Trajectory")
    ax.plot(gens, y_vals, color="#c084fc", linewidth=1.5, zorder=5)

    # Scatter points along generations
    scatter = ax.scatter(
        gens,
        y_vals,
        s=90,
        c=gens,
        cmap="cool",
        edgecolor="#ffffff",
        linewidth=1.5,
        zorder=6,
    )

    # Annotate latest milestone
    if len(gens) > 1:
        latest_gen = gens[-1]
        latest_y = y_vals[-1]
        latest_rank = ranks[-1]
        ax.annotate(
            f"Current: {latest_rank}\nTotal: {int(latest_y)} Absolute LP\n[Gen {latest_gen}]",
            xy=(latest_gen, latest_y),
            xytext=(latest_gen - max(0.5, latest_gen * 0.18), latest_y + 160),
            arrowprops=dict(facecolor="#38bdf8", edgecolor="#ffffff", arrowstyle="->", lw=1.8),
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1e293b", edgecolor="#38bdf8", alpha=0.95),
            color="#ffffff",
            fontweight="bold",
            fontsize=10.5,
            zorder=7,
        )

    # Axis Labels and Styling
    ax.set_xlabel("Training Generations (PPO Rollout Cycles)", color="#e2e8f0", fontsize=13, fontweight="bold", labelpad=12)
    ax.set_ylabel("Ranked Ladder Progression (Cumulative Absolute LP)", color="#e2e8f0", fontsize=13, fontweight="bold", labelpad=12)
    ax.set_title("AlphaStar TFT: Cumulative Ranked Ladder Progression (Absolute LP)", color="#f8fafc", fontsize=16, fontweight="bold", pad=20)

    # Configure X Axis
    max_x = max(10, max(gens) + 2)
    ax.set_xlim(-0.2, max_x)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.tick_params(axis="x", colors="#94a3b8", labelsize=11)

    # Configure Y Axis (Left: Major Tiers with LP threshold)
    y_ticks = [y for _, _, y, _ in major_tiers]
    y_labels = [name for _, name, _, _ in major_tiers]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, color="#f1f5f9", fontsize=10.5, fontweight="bold")
    ax.set_ylim(350, max(y_vals[-1] + 300, 3600))

    # Secondary right Y axis with Division Milestones
    secax = ax.secondary_yaxis("right")
    division_ticks = []
    division_labels = []
    for tier_name in ["BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND"]:
        base = TIER_BASE_ABSOLUTE_LP[tier_name]
        for div in [4, 2]:  # Show div 4 and 2 to keep readable
            division_ticks.append(base + (4 - div) * 100.0)
            division_labels.append(f"{tier_name.title()} {div}")
    division_ticks.extend([2800.0, 3050.0, 3300.0])
    division_labels.extend(["Master", "Grandmaster", "Challenger"])

    secax.set_yticks(division_ticks)
    secax.set_yticklabels(division_labels, color="#64748b", fontsize=9)
    secax.tick_params(colors="#475569")

    # Grid & Spines
    ax.grid(True, which="major", color="#1e293b", linestyle=":", alpha=0.6)
    for spine in ax.spines.values():
        spine.set_color("#334155")
        spine.set_linewidth(1.2)

    # Legend
    leg = ax.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc", loc="upper left", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out_file, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f" [+] AlphaStar Ranked Ladder Plot generated: file:///{out_file.resolve().as_posix()}")
    return out_file
