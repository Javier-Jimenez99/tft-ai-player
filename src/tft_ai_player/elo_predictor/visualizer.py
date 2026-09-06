"""Visualization module for TFT Elo prediction, skill manifold PCA, and RL agent overlay.

Creates interactive (Plotly) and static (Matplotlib) scatter plots of matches across
competitive tiers, visualizing high-dimensional gameplay clusters and projecting
RL agent trajectories onto the human skill manifold.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .dataset import parse_elo_from_record
from .features import FEATURE_NAMES, extract_game_features
from .model import TIER_THRESHOLDS, elo_to_tier_name

logger = logging.getLogger(__name__)

TIER_ORDER = [
    "IRON",
    "BRONZE",
    "SILVER",
    "GOLD",
    "PLATINUM",
    "EMERALD",
    "DIAMOND",
    "MASTER",
    "GRANDMASTER",
    "CHALLENGER",
]

ELO_GROUPS = [
    "Low Elo (Iron - Gold)",
    "Medium Elo (Platinum - Diamond)",
    "High Elo (Master - Challenger)",
]

ELO_GROUP_COLORS = {
    "Low Elo (Iron - Gold)": "#FF4757",         # Coral red (warm, distinct)
    "Medium Elo (Platinum - Diamond)": "#2ED573",     # Vibrant emerald green (crisp, unmistakable)
    "High Elo (Master - Challenger)": "#A55EEA", # Electric violet / royal purple
}


def get_elo_group(tier: str) -> str:
    """Classify a competitive tier into Low, Medium, or High Elo."""
    t = str(tier).upper().strip()
    if t in ["IRON", "BRONZE", "SILVER", "GOLD"]:
        return "Low Elo (Iron - Gold)"
    elif t in ["PLATINUM", "EMERALD", "DIAMOND"]:
        return "Medium Elo (Platinum - Diamond)"
    else:
        return "High Elo (Master - Challenger)"


TIER_COLORS = {
    "IRON": "#6C757D",        # Iron gray
    "BRONZE": "#A05A2C",      # Bronze copper
    "SILVER": "#A8B2B8",      # Silver metallic
    "GOLD": "#E5A93C",        # Gold vibrant
    "PLATINUM": "#25B29E",    # Platinum teal
    "EMERALD": "#00A86B",     # Emerald green
    "DIAMOND": "#4B7BEC",     # Diamond sapphire
    "MASTER": "#9B59B6",      # Master purple
    "GRANDMASTER": "#E74C3C", # Grandmaster crimson
    "CHALLENGER": "#F1C40F",  # Challenger radiant gold
}

AGENT_COLOR = "#00FFFF"       # Neon cyan for RL agent
AGENT_MARKER = "star"


class EloClusterVisualizer:
    """Extracts, reduces, and plots TFT match trajectories with RL agent projection."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2, random_state=42)
        self.is_fitted = False
        self.human_features: np.ndarray = np.empty((0, len(FEATURE_NAMES)))
        self.human_tiers: list[str] = []
        self.human_elos: list[float] = []
        self.human_match_ids: list[str] = []
        self.agent_points: list[dict[str, Any]] = []

    def load_or_build_dataset(
        self,
        data_dir: str | Path = "D:/tft-winner-data/set18/players",
        cache_file: str | Path | None = "models/elo_predictor/elo_manifold_sample.npz",
        samples_per_tier: int = 60,
        min_rounds_per_match: int = 6,
    ) -> EloClusterVisualizer:
        """Load cached sample dataset or extract balanced matches across all competitive tiers."""
        cache_p = Path(cache_file) if cache_file else None
        if cache_p and cache_p.exists():
            print(f" [+] Loading cached manifold dataset from {cache_p}...")
            data = np.load(cache_p, allow_pickle=True)
            self.human_features = data["features"]
            self.human_tiers = list(data["tiers"])
            self.human_elos = list(data["elos"])
            self.human_match_ids = list(data["match_ids"])
            self._fit_projections()
            return self

        print(f" [*] Scanning player files in {data_dir} to build balanced tier sample...")
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        csv_files = list(data_path.glob("*.csv"))
        tier_to_files: dict[str, list[Path]] = {t: [] for t in TIER_ORDER}

        for f in csv_files:
            if all(len(tier_to_files[t]) >= 6 for t in TIER_ORDER):
                break
            try:
                with open(f, "r", encoding="utf-8") as fl:
                    h_line = fl.readline().strip().split(",")
                    d_line = fl.readline().strip().split(",")
                    row = dict(zip(h_line, d_line))
                    raw_t = row.get("focal_tier") or row.get("tier_category") or row.get("avg_match_rating")
                    raw_num = row.get("focal_rating_numeric") or row.get("avg_match_rating_numeric")
                    elo = parse_elo_from_record(raw_t, raw_num)
                    if elo:
                        t = elo_to_tier_name(elo)
                        if t in tier_to_files and len(tier_to_files[t]) < 6:
                            tier_to_files[t].append(f)
            except Exception:
                continue

        # Extract matches per tier
        features_list = []
        tiers_list = []
        elos_list = []
        mids_list = []

        for tier in TIER_ORDER:
            collected = 0
            tier_files = tier_to_files[tier]
            print(f"   -> Processing {tier} ({len(tier_files)} files)...")

            for fpath in tier_files:
                if collected >= samples_per_tier:
                    break
                try:
                    with open(fpath, "r", encoding="utf-8") as fl:
                        reader = csv.DictReader(fl)
                        matches_dict: dict[str, list[dict[str, Any]]] = {}
                        match_elos: dict[str, float] = {}

                        for r in reader:
                            mid = str(r.get("match_id", "")).strip()
                            if not mid:
                                continue
                            state_raw = r.get("input_state_json")
                            focal_board = []
                            if isinstance(state_raw, str) and state_raw.strip():
                                try:
                                    focal_board = json.loads(state_raw).get("focal_board", [])
                                except Exception:
                                    focal_board = []

                            m_elo = parse_elo_from_record(
                                r.get("avg_match_rating"), r.get("avg_match_rating_numeric")
                            ) or parse_elo_from_record(
                                r.get("focal_tier") or r.get("tier_category"), r.get("focal_rating_numeric")
                            )
                            if m_elo and mid not in match_elos:
                                match_elos[mid] = m_elo

                            matches_dict.setdefault(mid, []).append({
                                "round_stage": r.get("round_stage", "2-1"),
                                "focal_health": r.get("focal_health", 100),
                                "focal_level": r.get("focal_level", 1),
                                "focal_gold": r.get("focal_gold", 0),
                                "focal_board": focal_board,
                            })

                        for mid, traj in matches_dict.items():
                            if len(traj) >= min_rounds_per_match and collected < samples_per_tier:
                                feat = extract_game_features(traj)
                                elo = match_elos.get(mid, 1200.0)
                                features_list.append(feat)
                                tiers_list.append(tier)
                                elos_list.append(elo)
                                mids_list.append(mid)
                                collected += 1
                except Exception:
                    continue

        self.human_features = np.array(features_list, dtype=np.float32)
        self.human_tiers = tiers_list
        self.human_elos = elos_list
        self.human_match_ids = mids_list

        print(f" [+] Total extracted balanced matches: {len(self.human_features):,}")
        if cache_p:
            cache_p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_p,
                features=self.human_features,
                tiers=np.array(self.human_tiers),
                elos=np.array(self.human_elos),
                match_ids=np.array(self.human_match_ids),
            )
            print(f" [+] Saved cache artifact to {cache_p}")

        self._fit_projections()
        return self

    def _fit_projections(self) -> None:
        """Fit standard scaler and PCA on human feature distribution."""
        if len(self.human_features) == 0:
            return
        X_scaled = self.scaler.fit_transform(self.human_features)
        self.pca.fit(X_scaled)
        self.is_fitted = True
        var_exp = self.pca.explained_variance_ratio_
        print(f" [+] PCA Fitted: PC1 ({var_exp[0]*100:.1f}%), PC2 ({var_exp[1]*100:.1f}%) | Total 2D: {sum(var_exp)*100:.1f}%")

    def add_agent_match(
        self,
        features: np.ndarray | Sequence[float],
        label: str = "AlphaStar V6",
        gen: int | None = None,
        predicted_elo: float | None = None,
        trust_score: float | None = None,
    ) -> None:
        """Add an RL agent game trajectory vector for overlay onto the manifold."""
        feat_arr = np.array(features, dtype=np.float32).reshape(1, -1)
        if self.is_fitted:
            scaled = self.scaler.transform(feat_arr)
            pca_coords = self.pca.transform(scaled)[0]
        else:
            pca_coords = np.array([0.0, 0.0])

        self.agent_points.append({
            "label": label,
            "gen": gen,
            "features": feat_arr[0],
            "pca_x": float(pca_coords[0]),
            "pca_y": float(pca_coords[1]),
            "avg_gold": float(feat_arr[0, FEATURE_NAMES.index("avg_gold")]),
            "neural_quality": float(feat_arr[0, FEATURE_NAMES.index("neural_quality_mean")]),
            "final_placement": float(feat_arr[0, FEATURE_NAMES.index("final_placement")]),
            "final_level": float(feat_arr[0, FEATURE_NAMES.index("final_level")]),
            "predicted_elo": predicted_elo,
            "trust_score": trust_score,
        })

    def generate_interactive_dashboard(self, output_html: str | Path = "reports/visualizations/elo_clusters.html") -> Path:
        """Generate interactive 2-panel Plotly dashboard with hover tooltips and tier toggle."""
        if not self.is_fitted:
            raise RuntimeError("Visualizer must be fitted with human dataset before plotting.")

        X_scaled = self.scaler.transform(self.human_features)
        pca_coords = self.pca.transform(X_scaled)

        gold_idx = FEATURE_NAMES.index("avg_gold")
        neural_idx = FEATURE_NAMES.index("neural_quality_mean")
        place_idx = FEATURE_NAMES.index("final_placement")
        level_idx = FEATURE_NAMES.index("final_level")

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                f"35D Skill Manifold (PCA Projection: PC1={self.pca.explained_variance_ratio_[0]*100:.1f}%, PC2={self.pca.explained_variance_ratio_[1]*100:.1f}%)",
                "Strategic Domain Space: Economy vs Neural Board Quality",
            ),
            horizontal_spacing=0.08,
        )

        for group in ELO_GROUPS:
            indices = [i for i, t in enumerate(self.human_tiers) if get_elo_group(t) == group]
            if not indices:
                continue

            color = ELO_GROUP_COLORS.get(group, "#FFFFFF")
            t_pca = pca_coords[indices]
            t_gold = self.human_features[indices, gold_idx]
            t_neural = self.human_features[indices, neural_idx]
            t_place = self.human_features[indices, place_idx]
            t_level = self.human_features[indices, level_idx]
            t_elo = [self.human_elos[i] for i in indices]
            t_tiers = [self.human_tiers[i] for i in indices]

            hover_text = [
                f"<b>{group}</b><br>Tier: {tier}<br>Elo: {elo:.0f}<br>Place: {place:.0f}<br>Level: {lvl:.0f}<br>Avg Gold: {g:.1f}<br>Neural Quality: {nq:.2f}"
                for tier, elo, place, lvl, g, nq in zip(t_tiers, t_elo, t_place, t_level, t_gold, t_neural)
            ]

            fig.add_trace(
                go.Scatter(
                    x=t_pca[:, 0],
                    y=t_pca[:, 1],
                    mode="markers",
                    name=group,
                    legendgroup=group,
                    marker=dict(size=7, color=color, opacity=0.60, line=dict(width=0.5, color="#1E1E1E")),
                    text=hover_text,
                    hoverinfo="text",
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=t_gold,
                    y=t_neural,
                    mode="markers",
                    name=group,
                    legendgroup=group,
                    showlegend=False,
                    marker=dict(size=7, color=color, opacity=0.60, line=dict(width=0.5, color="#1E1E1E")),
                    text=hover_text,
                    hoverinfo="text",
                ),
                row=1,
                col=2,
            )

            # Centroid marker for each group
            c_pca_x = [float(np.mean(t_pca[:, 0]))]
            c_pca_y = [float(np.mean(t_pca[:, 1]))]
            c_gold = [float(np.mean(t_gold))]
            c_neural = [float(np.mean(t_neural))]
            c_hover = [f"<b>{group} Centroid</b><br>Mean PC1: {c_pca_x[0]:.2f}<br>Mean PC2: {c_pca_y[0]:.2f}<br>Mean Quality: {c_neural[0]:.2f}<br>Mean Gold: {c_gold[0]:.1f}"]

            fig.add_trace(
                go.Scatter(
                    x=c_pca_x,
                    y=c_pca_y,
                    mode="markers",
                    name=f"◆ {group} Centroid",
                    legendgroup=group,
                    showlegend=False,
                    marker=dict(size=12, color=color, symbol="diamond", line=dict(width=1.5, color="#FFFFFF")),
                    text=c_hover,
                    hoverinfo="text",
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=c_gold,
                    y=c_neural,
                    mode="markers",
                    name=f"◆ {group} Centroid",
                    legendgroup=group,
                    showlegend=False,
                    marker=dict(size=12, color=color, symbol="diamond", line=dict(width=1.5, color="#FFFFFF")),
                    text=c_hover,
                    hoverinfo="text",
                ),
                row=1,
                col=2,
            )

        if self.agent_points:
            agent_pca_x = [p["pca_x"] for p in self.agent_points]
            agent_pca_y = [p["pca_y"] for p in self.agent_points]
            agent_gold = [p["avg_gold"] for p in self.agent_points]
            agent_neural = [p["neural_quality"] for p in self.agent_points]
            agent_labels = [f"Gen {p.get('gen', '?')}" for p in self.agent_points]
            agent_hover = [
                f"<b>{p['label']}</b> (Gen {p.get('gen', '?')})<br>Predicted Elo: {p.get('predicted_elo', 'N/A')}<br>Trust: {p.get('trust_score', 'N/A')}%<br>Place: {p['final_placement']:.1f}<br>Level: {p['final_level']:.1f}<br>Avg Gold: {p['avg_gold']:.1f}<br>Neural Quality: {p['neural_quality']:.2f}"
                for p in self.agent_points
            ]

            # Connect trajectory line if multiple milestones
            if len(self.agent_points) > 1:
                fig.add_trace(
                    go.Scatter(
                        x=agent_pca_x,
                        y=agent_pca_y,
                        mode="lines",
                        name="RL Trajectory",
                        line=dict(color=AGENT_COLOR, width=2, dash="dot"),
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=agent_gold,
                        y=agent_neural,
                        mode="lines",
                        name="RL Trajectory",
                        line=dict(color=AGENT_COLOR, width=2, dash="dot"),
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )

            fig.add_trace(
                go.Scatter(
                    x=agent_pca_x,
                    y=agent_pca_y,
                    mode="markers+text",
                    name="★ AlphaStar V6 Agent",
                    text=agent_labels,
                    textposition="top center",
                    textfont=dict(color=AGENT_COLOR, size=11),
                    marker=dict(
                        size=16,
                        color=AGENT_COLOR,
                        symbol=AGENT_MARKER,
                        line=dict(width=2, color="#FFFFFF"),
                    ),
                    hovertext=agent_hover,
                    hoverinfo="text",
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=agent_gold,
                    y=agent_neural,
                    mode="markers+text",
                    name="★ AlphaStar V6 Agent",
                    showlegend=False,
                    text=agent_labels,
                    textposition="top center",
                    textfont=dict(color=AGENT_COLOR, size=11),
                    marker=dict(
                        size=16,
                        color=AGENT_COLOR,
                        symbol=AGENT_MARKER,
                        line=dict(width=2, color="#FFFFFF"),
                    ),
                    hovertext=agent_hover,
                    hoverinfo="text",
                ),
                row=1,
                col=2,
            )

        fig.update_layout(
            template="plotly_dark",
            title=dict(
                text="<b>TFT Elo Skill Manifold & Agent Strategy Landscape</b>",
                font=dict(size=20, color="#E0E0E0"),
                x=0.03,
            ),
            legend=dict(
                title=dict(text="Skill Tier (Elo Group)", font=dict(size=13)),
                orientation="v",
                yanchor="top",
                y=0.98,
                xanchor="right",
                x=1.12,
                bgcolor="rgba(25, 25, 30, 0.8)",
                bordercolor="rgba(100, 100, 120, 0.4)",
                borderwidth=1,
            ),
            margin=dict(l=50, r=130, t=80, b=50),
            height=650,
        )

        fig.update_xaxes(title_text="Principal Component 1 (Match Tempo & Power)", row=1, col=1)
        fig.update_yaxes(title_text="Principal Component 2 (Economy vs Board Aggression)", row=1, col=1)
        fig.update_xaxes(title_text="Average Gold Maintained (Economy Discipline)", row=1, col=2)
        fig.update_yaxes(title_text="Neural Board Quality Mean (BoardQualityNet)", row=1, col=2)

        out_p = Path(output_html)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_p))
        print(f" [+] Interactive HTML dashboard saved: {out_p}")
        return out_p

    def generate_static_plot(self, output_png: str | Path = "reports/visualizations/elo_clusters.png") -> Path:
        """Generate high-resolution publication-quality PNG plot using Matplotlib."""
        if not self.is_fitted:
            raise RuntimeError("Visualizer must be fitted with human dataset before plotting.")

        X_scaled = self.scaler.transform(self.human_features)
        pca_coords = self.pca.transform(X_scaled)

        gold_idx = FEATURE_NAMES.index("avg_gold")
        neural_idx = FEATURE_NAMES.index("neural_quality_mean")

        plt.style.use("dark_background")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), facecolor="#141419")
        ax1.set_facecolor("#1A1A22")
        ax2.set_facecolor("#1A1A22")

        for group in ELO_GROUPS:
            indices = [i for i, t in enumerate(self.human_tiers) if get_elo_group(t) == group]
            if not indices:
                continue
            color = ELO_GROUP_COLORS.get(group, "#FFFFFF")
            t_pca = pca_coords[indices]
            t_gold = self.human_features[indices, gold_idx]
            t_neural = self.human_features[indices, neural_idx]

            ax1.scatter(t_pca[:, 0], t_pca[:, 1], c=color, label=group, alpha=0.72, s=40, edgecolors="none")
            ax2.scatter(t_gold, t_neural, c=color, label=group, alpha=0.72, s=40, edgecolors="none")

            # Centroids of each group
            c_pca_x = float(np.mean(t_pca[:, 0]))
            c_pca_y = float(np.mean(t_pca[:, 1]))
            c_gold = float(np.mean(t_gold))
            c_neural = float(np.mean(t_neural))

            ax1.scatter(c_pca_x, c_pca_y, c=color, marker="D", s=90, edgecolors="#FFFFFF", linewidth=1.5, zorder=5)
            ax2.scatter(c_gold, c_neural, c=color, marker="D", s=90, edgecolors="#FFFFFF", linewidth=1.5, zorder=5)

        if self.agent_points:
            agent_pca_x = [p["pca_x"] for p in self.agent_points]
            agent_pca_y = [p["pca_y"] for p in self.agent_points]
            agent_gold = [p["avg_gold"] for p in self.agent_points]
            agent_neural = [p["neural_quality"] for p in self.agent_points]

            if len(self.agent_points) > 1:
                ax1.plot(agent_pca_x, agent_pca_y, color=AGENT_COLOR, linestyle="--", linewidth=1.8, alpha=0.85, zorder=9)
                ax2.plot(agent_gold, agent_neural, color=AGENT_COLOR, linestyle="--", linewidth=1.8, alpha=0.85, zorder=9)

            ax1.scatter(
                agent_pca_x, agent_pca_y,
                c=AGENT_COLOR, marker="*", s=280,
                edgecolors="#FFFFFF", linewidth=1.5,
                label="AlphaStar V6 (Agent)", zorder=10
            )
            ax2.scatter(
                agent_gold, agent_neural,
                c=AGENT_COLOR, marker="*", s=280,
                edgecolors="#FFFFFF", linewidth=1.5,
                label="AlphaStar V6 (Agent)", zorder=10
            )

            # Filter labels if many milestones exist to avoid overlapping text
            all_gens = [p.get("gen", 0) for p in self.agent_points]
            latest_gen = max(all_gens) if all_gens else 0
            if len(self.agent_points) <= 8:
                labeled_gens = set(all_gens)
            else:
                labeled_gens = {min(all_gens), 30, 50, 70, 100, 150}
                if latest_gen >= 230:
                    labeled_gens.add(200)
                labeled_gens.add(latest_gen)

            for p in self.agent_points:
                gen = p.get("gen", 0)
                if gen not in labeled_gens:
                    continue
                gen_lbl = f"★ Gen {gen} (Latest)" if (gen == latest_gen and len(self.agent_points) > 5) else f"Gen {gen}"
                ax1.annotate(
                    gen_lbl, (p["pca_x"], p["pca_y"]),
                    textcoords="offset points", xytext=(8, 8),
                    color=AGENT_COLOR, fontsize=10, fontweight="bold", zorder=11,
                )
                ax2.annotate(
                    gen_lbl, (p["avg_gold"], p["neural_quality"]),
                    textcoords="offset points", xytext=(8, 8),
                    color=AGENT_COLOR, fontsize=10, fontweight="bold", zorder=11,
                )

        ax1.set_title(f"35D Skill Manifold (PCA 2D Projection: {sum(self.pca.explained_variance_ratio_)*100:.1f}% Variance)", fontsize=13, pad=12, color="#E0E0E0")
        ax1.set_xlabel(f"Principal Component 1 ({self.pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11, color="#B0B0B0")
        ax1.set_ylabel(f"Principal Component 2 ({self.pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11, color="#B0B0B0")
        ax1.grid(True, linestyle="--", alpha=0.25, color="#555566")

        ax2.set_title("Strategic Domain Space: Economy vs Neural Board Quality", fontsize=13, pad=12, color="#E0E0E0")
        ax2.set_xlabel("Average Gold Maintained", fontsize=11, color="#B0B0B0")
        ax2.set_ylabel("Neural Board Quality Mean (BoardQualityNet)", fontsize=11, color="#B0B0B0")
        ax2.grid(True, linestyle="--", alpha=0.25, color="#555566")

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(
            handles, labels,
            title="Skill Tier (Elo)",
            loc="center right",
            bbox_to_anchor=(0.99, 0.5),
            frameon=True,
            facecolor="#20202A",
            edgecolor="#404055",
            fontsize=10,
        )

        plt.subplots_adjust(left=0.06, right=0.90, top=0.90, bottom=0.10, wspace=0.18)
        fig.suptitle("TFT Competitive Elo Clustering & AI Agent Positioning", fontsize=17, fontweight="bold", color="#FFFFFF", y=0.97)

        out_p = Path(output_png)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        print(f" [+] Static PNG plot saved: {out_p}")
        return out_p


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate TFT Elo Clustering and Skill Manifold Visualization")
    parser.add_argument("--data-dir", type=str, default="D:/tft-winner-data/set18/players")
    parser.add_argument("--cache-file", type=str, default="models/elo_predictor/elo_manifold_sample.npz")
    parser.add_argument("--samples-per-tier", type=int, default=50)
    parser.add_argument("--output-html", type=str, default="reports/visualizations/elo_clusters.html")
    parser.add_argument("--output-png", type=str, default="reports/visualizations/elo_clusters.png")
    args = parser.parse_args()

    viz = EloClusterVisualizer()
    viz.load_or_build_dataset(
        data_dir=args.data_dir,
        cache_file=args.cache_file,
        samples_per_tier=args.samples_per_tier,
    )
    viz.generate_interactive_dashboard(args.output_html)
    viz.generate_static_plot(args.output_png)
    return 0


if __name__ == "__main__":
    main()
