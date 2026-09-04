"""Command-line entry points for MetaTFT discovery and timeline extraction."""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from tqdm import tqdm

from .dataset import (
    GameManifestEntry,
    LobbyEdge,
    PlayerCsvWriter,
    PlayerGraph,
    PlayerNode,
    TimelineValidationError,
    extract_pvp_rounds,
)
from .dataset.models import normalize_tier
from .metatft import LeaderboardPlayer, MetaTftClient, MetaTftRequestError, TrackedTimelineCandidate


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return a process exit code."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command in ("expand-graph", "crawl-graph"):
            return _run_expand_graph(args)
        if args.command in ("download-games", "download-manifest"):
            return _run_download_games(args)
        if args.command == "profile":
            return _collect_profile(args)
        if args.command == "collect":
            return _collect_leaderboard(args)
        if args.command == "timeline":
            return _collect_timeline(args)
        if args.command == "train-round-winner":
            from .round_winner.train import main as train_main
            return train_main()
        if args.command == "simulate":
            return _run_simulation(args)
        if args.command == "rl-train":
            return _run_rl_train(args)
        if args.command == "rl-league":
            return _run_rl_league(args)
        if args.command in ("rl-visualize-progression", "rl-plot-strategy", "rl-progression"):
            return _run_rl_visualize_progression(args)
        if args.command == "pretrain-trunk":
            return _run_pretrain_trunk(args)
        if args.command in ("cluster-compositions", "cluster"):
            return _run_cluster_compositions(args)
        if args.command in ("train-transition", "transition"):
            return _run_train_transition(args)
        if args.command in ("plot-graph", "visualize-graph", "graph-viz"):
            return _run_plot_graph(args)
    except (MetaTftRequestError, TimelineValidationError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    parser.error(f"unsupported command: {args.command}")
    return 2


def get_default_rl_checkpoint_dir(run_name: str | None = None) -> str:
    """Return default checkpoint directory respecting models/rl/checkpoints/<run_name> hierarchy."""
    if Path("D:/tft-winner-data/set18/models/rl/checkpoints").exists() or Path("D:/tft-winner-data/set18").exists():
        base = Path("D:/tft-winner-data/set18/models/rl/checkpoints")
    elif Path("models/rl/checkpoints").exists():
        base = Path("models/rl/checkpoints")
    else:
        base = Path("checkpoints/league")

    if run_name:
        return str(base / run_name)
    return str(base)


def resolve_league_checkpoint_dir(specified_dir: str | None = None, run_name: str | None = None) -> str:
    """Resolve an existing checkpoint directory for evaluation or visualization."""
    if specified_dir:
        return specified_dir
    base = Path(get_default_rl_checkpoint_dir())
    if run_name and (base / run_name).exists():
        return str(base / run_name)
    if base.exists():
        subdirs = [d for d in base.iterdir() if d.is_dir() and not d.name.startswith(".")]
        if subdirs:
            if run_name:
                match = [d for d in subdirs if d.name == run_name]
                if match:
                    return str(match[0])
            subdirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
            return str(subdirs[0])
    return str(base)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tft-ai-player")
    subcommands = parser.add_subparsers(dest="command", required=True)

    profile_parser = subcommands.add_parser(
        "profile",
        help="collect a small number of CSV game files from one player",
    )
    profile_parser.add_argument("--region", required=True, help="Riot platform, for example LA2")
    profile_parser.add_argument("--game-name", required=True, help="Riot game name")
    profile_parser.add_argument("--tag-line", required=True, help="Riot tagline")
    profile_parser.add_argument("--tft-set", default="TFTSet17", help="TFT set to retain")
    profile_parser.add_argument("--games", type=int, default=1, help="maximum games to collect")
    profile_parser.add_argument(
        "-o",
        "--output",
        "--output-dir",
        dest="output",
        type=Path,
        default=Path("data"),
        help="destination directory where games/*.csv will be written (default: data)",
    )
    profile_parser.add_argument(
        "--request-interval",
        type=float,
        default=1.5,
        help="minimum seconds between API requests (default: 1.5)",
    )
    profile_parser.add_argument(
        "--allowed-queues",
        type=int,
        nargs="+",
        default=[1100],
        help="Riot queue IDs to retain (default: [1100] for Ranked TFT)",
    )
    profile_parser.add_argument(
        "--tier-partitioned",
        action="store_true",
        help="partition output CSV storage by tier directory (e.g. data/tiers/gold/players/)",
    )

    collect_parser = subcommands.add_parser(
        "collect",
        help="iterate tracked leaderboard players and write one CSV per selected game",
    )
    collect_parser.add_argument("--players", type=int, default=10, help="number of leaderboard players to sample")
    collect_parser.add_argument(
        "--games-per-player",
        type=int,
        default=None,
        help="optional cap on games contributed by each player (default: unlimited)",
    )
    collect_parser.add_argument(
        "--max-games",
        type=int,
        help="optional cap across all players",
    )
    collect_parser.add_argument(
        "--leaderboard-offset",
        type=int,
        default=0,
        help="leaderboard offset used to choose a different player cohort",
    )
    collect_parser.add_argument("--tft-set", default="TFTSet17", help="TFT set to retain")
    collect_parser.add_argument(
        "-o",
        "--output",
        "--output-dir",
        dest="output",
        type=Path,
        default=Path("data"),
        help="destination directory where games/*.csv will be written (default: data)",
    )
    collect_parser.add_argument(
        "--request-interval",
        type=float,
        default=1.5,
        help="minimum seconds between API requests (default: 1.5)",
    )
    collect_parser.add_argument(
        "--allowed-queues",
        type=int,
        nargs="+",
        default=[1100],
        help="Riot queue IDs to retain (default: [1100] for Ranked TFT)",
    )
    collect_parser.add_argument(
        "--target-tiers",
        nargs="+",
        default=None,
        help="filter and collect only specific tiers, e.g. IRON BRONZE SILVER GOLD PLATINUM EMERALD DIAMOND MASTER GRANDMASTER CHALLENGER (default: all)",
    )
    collect_parser.add_argument(
        "--crawl-depth",
        type=int,
        default=5,
        help="lobby crawl depth: 1 for direct leaderboard/seeds, >=2 to explore other lobby participants across all ranks (default: 5)",
    )
    collect_parser.add_argument(
        "--seed-players",
        nargs="+",
        default=None,
        help="optional seed player Riot IDs to crawl, formatted as REGION/GameName#TagLine or GameName#TagLine (e.g. NA1/Kurumx#FREAK)",
    )
    collect_parser.add_argument(
        "--max-games-per-tier",
        type=int,
        default=None,
        help="optional cap on total games collected per tier category to ensure balanced ELO distribution",
    )
    collect_parser.add_argument(
        "--min-games-per-player",
        type=int,
        default=None,
        help="optional minimum number of tracked app games required to collect a player",
    )
    collect_parser.add_argument(
        "--riot-api-key",
        default=None,
        help="optional Riot Developer API key to directly query players by tier/division from Riot TFT League API",
    )
    collect_parser.add_argument(
        "--tier-partitioned",
        action="store_true",
        help="partition output CSV storage by tier directory (e.g. data/tiers/gold/players/)",
    )

    timeline_parser = subcommands.add_parser(
        "timeline",
        help="fetch a known MetaTFT timeline URL and write one game CSV",
    )
    timeline_parser.add_argument("--timeline-url", required=True, help="known MetaTFT timeline JSON URL")
    timeline_parser.add_argument("--match-id", required=True, help="game ID used for the CSV file name")
    timeline_parser.add_argument("--tft-set", default="TFTSet17", help="TFT set for this match")
    timeline_parser.add_argument("--game-version", default="unknown", help="Riot patch, for example 16.16")
    timeline_parser.add_argument(
        "-o",
        "--output",
        "--output-dir",
        dest="output",
        type=Path,
        default=Path("data"),
        help="destination directory where games/*.csv will be written (default: data)",
    )
    timeline_parser.add_argument(
        "--request-interval",
        type=float,
        default=1.5,
        help="minimum seconds between API requests (default: 1.5)",
    )

    train_parser = subcommands.add_parser(
        "train-round-winner",
        help="train and serialize the round winner probability model",
    )
    train_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(r"D:\tft-winner-data\players"),
        help="path to directory containing player round CSV files",
    )
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/round_winner"),
        help="directory where model bundle and metadata will be saved",
    )
    train_parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="proportion of matches held out for testing",
    )

    simulate_parser = subcommands.add_parser(
        "simulate",
        help="run a full TFT simulation and export an interactive visual dashboard",
    )
    simulate_parser.add_argument(
        "--set",
        "-s",
        type=str,
        default="TFTSet17",
        help="TFT set to simulate (e.g. '17', '18', 'TFTSet17', 'TFTSet18'). Default: 'TFTSet17'.",
    )
    simulate_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for match generation (default: 42)",
    )
    simulate_parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="custom destination path for the HTML dashboard file",
    )
    simulate_parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="dashboards",
        help="output directory when output-path is omitted (default: dashboards)",
    )
    simulate_parser.add_argument(
        "--max-rounds",
        type=int,
        default=60,
        help="maximum number of rounds to simulate before stopping (default: 60)",
    )
    simulate_parser.add_argument(
        "--open-browser",
        action="store_true",
        help="automatically open the exported dashboard in the default browser",
    )
    simulate_parser.add_argument(
        "--rl-model",
        type=str,
        default=None,
        help="path to trained RL PyTorch policy weights (.pt) to control focal player",
    )
    default_rw_model = (
        "D:/tft-winner-data/set18/models/round_winner/round_winner_model.joblib"
        if Path("D:/tft-winner-data/set18/models/round_winner/round_winner_model.joblib").exists()
        else (
            "D:/tft-winner-data/set18/models/round_winner_model.joblib"
            if Path("D:/tft-winner-data/set18/models/round_winner_model.joblib").exists()
            else "models/round_winner/round_winner_model.joblib"
        )
    )
    simulate_parser.add_argument(
        "--round-winner-model",
        type=str,
        default=default_rw_model,
        help="path to trained single round winner ML model for combat resolution",
    )

    rl_train_parser = subcommands.add_parser(
        "rl-train",
        help="train autonomous RL agent using Maskable PPO and AlphaStar League Pipeline",
    )
    rl_train_parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="number of training iterations/generations to run (default: 50)",
    )
    rl_train_parser.add_argument(
        "--rollout-steps",
        type=int,
        default=4096,
        help="number of rollout steps to collect per generation (default: 4096)",
    )
    rl_train_parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="PPO mini-batch size (default: 512)",
    )
    rl_train_parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="PPO epochs per batch (default: 4)",
    )
    rl_train_parser.add_argument(
        "--lr",
        type=float,
        default=2.5e-4,
        help="initial learning rate (default: 2.5e-4)",
    )
    rl_train_parser.add_argument(
        "--eval-every",
        type=int,
        default=25,
        help="evaluate against deterministic benchmark bots every N generations (default: 25)",
    )
    rl_train_parser.add_argument(
        "--snapshot-every",
        type=int,
        default=50,
        help="archive frozen historical snapshot every N generations (default: 50)",
    )
    default_trunk_ckpt = (
        "D:/tft-winner-data/set18/models/trunk/trunk_best.pt"
        if Path("D:/tft-winner-data/set18/models/trunk/trunk_best.pt").exists()
        else ("models/trunk/trunk_best.pt" if Path("models/trunk/trunk_best.pt").exists() else "models/trunk/best_model.pt")
    )
    rl_train_parser.add_argument(
        "--trunk-checkpoint",
        type=str,
        default=default_trunk_ckpt,
        help=f"path to pre-trained frozen MultiModalFusionTrunk checkpoint (default: {default_trunk_ckpt})",
    )
    default_world_model_ckpt = (
        "models/transition_predictor/predictor_best.pt"
        if Path("models/transition_predictor/predictor_best.pt").exists()
        else (
            "D:/tft-winner-data/set18/models/transition/best_model.pt"
            if Path("D:/tft-winner-data/set18/models/transition/best_model.pt").exists()
            else "models/transition_predictor/predictor_final.pt"
        )
    )
    rl_train_parser.add_argument(
        "--world-model-checkpoint",
        type=str,
        default=default_world_model_ckpt,
        help=f"path to pre-trained StateTransitionPredictor (World Model) checkpoint (default: {default_world_model_ckpt})",
    )
    rl_train_parser.add_argument(
        "--z-index-path",
        type=str,
        default="models/clustering/z_index.pt",
        help="path to exported Z-Index centroids artifact (z_index.pt)",
    )
    rl_train_parser.add_argument(
        "--round-winner-model",
        type=str,
        default=None,
        help="path to optional custom combat model checkpoint (.pt for Deep Learning GPU, or .joblib for LightGBM CPU). Defaults to fast GPU Deep Learning resolver.",
    )
    default_rl_base_dir = get_default_rl_checkpoint_dir()

    rl_train_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help=f"directory to persist model weights, checkpoints, and league profiles (default: {default_rl_base_dir}/<run_name>)",
    )
    rl_train_parser.add_argument(
        "--run-name",
        type=str,
        default="ppo_alphastar_v4",
        help="custom experiment run name for WandB tracking and isolated checkpoint directory (default: ppo_alphastar_v4)",
    )
    rl_train_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="hardware execution device (cuda or cpu, default: auto-detect)",
    )
    rl_train_parser.add_argument(
        "--wandb-project",
        type=str,
        default="tft-ai-league",
        help="Weights & Biases project name (default: tft-ai-league)",
    )
    rl_train_parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB username or team entity name",
    )
    rl_train_parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="WandB experiment group",
    )
    rl_train_parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="disable WandB cloud logging and run in pure offline mode",
    )
    rl_train_parser.add_argument(
        "--resume",
        action="store_true",
        help="automatically resume training from the latest generation checkpoint in --checkpoint-dir",
    )
    rl_train_parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="path to specific checkpoint directory to resume from (e.g. checkpoints/league/gen_0032)",
    )

    rl_league_parser = subcommands.add_parser(
        "rl-league",
        help="run or inspect the AlphaStar-style 8-player TFT multi-agent league",
    )
    rl_league_parser.add_argument(
        "--matches",
        type=int,
        default=10,
        help="number of 8-player tournament matches to simulate (default: 10)",
    )
    rl_league_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="directory containing league checkpoints (default: auto-detected latest or models/rl/checkpoints/ppo_alphastar_v3)",
    )
    rl_league_parser.add_argument(
        "--markdown-out",
        type=str,
        default=None,
        help="optional destination file to write markdown leaderboard report",
    )

    viz_parser = subcommands.add_parser(
        "rl-visualize-progression",
        aliases=["rl-plot-strategy", "rl-progression"],
        help="generate AlphaStar-style strategy space 2D progression animation and unit composition dashboard",
    )
    viz_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="directory containing generation checkpoint history (default: auto-detected for --run-name)",
    )
    viz_parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/visualizations",
        help="directory to save exported visualization files (default: reports/visualizations)",
    )
    viz_parser.add_argument(
        "--format",
        type=str,
        choices=["gif", "png", "html", "all"],
        default="all",
        help="export format: 'gif', 'png', 'html', or 'all' (default: all)",
    )
    viz_parser.add_argument(
        "--fps",
        type=int,
        default=12,
        help="frame rate for animated GIF (default: 12)",
    )
    viz_parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="generation sampling step stride (default: 2)",
    )
    viz_parser.add_argument(
        "--agent-name",
        type=str,
        default="AlphaTFT-Main",
        help="display name for the focal agent (default: AlphaTFT-Main)",
    )
    viz_parser.add_argument(
        "--open-browser",
        action="store_true",
        help="automatically open the interactive HTML dashboard in browser",
    )
    viz_parser.add_argument(
        "--wandb",
        action="store_true",
        help="upload generated progression artifacts to Weights & Biases",
    )
    viz_parser.add_argument(
        "--wandb-project",
        type=str,
        default="tft-ai-league",
        help="Weights & Biases project name (default: tft-ai-league)",
    )
    viz_parser.add_argument(
        "--run-name",
        type=str,
        default="ppo_alphastar_v3",
        help="Weights & Biases run name to attach media artifacts to (default: ppo_alphastar_v3)",
    )

    pretrain_parser = subcommands.add_parser(
        "pretrain-trunk",
        help="pretrain the Multi-Modal Fusion Trunk via dual-objective multi-task learning",
    )
    pretrain_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(r"D:\tft-winner-data\set18\players"),
        help="directory containing player CSV files or path to single CSV",
    )
    pretrain_parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("models/trunk"),
        help="destination directory for model checkpoints and vocab",
    )
    pretrain_parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="number of pre-training epochs (default: 5)",
    )
    pretrain_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="mini-batch size of snapshot pairs (default: 64)",
    )
    pretrain_parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="AdamW learning rate (default: 1e-3)",
    )
    pretrain_parser.add_argument(
        "--embed-dim",
        type=int,
        default=32,
        help="Champ2Vec champion embedding dimension (default: 32)",
    )
    pretrain_parser.add_argument(
        "--board-dim",
        type=int,
        default=256,
        help="BoardEncoder spatial CNN output dimension (default: 256)",
    )
    pretrain_parser.add_argument(
        "--fused-dim",
        type=int,
        default=384,
        help="MultiModalFusionTrunk fused latent dimension (default: 384)",
    )
    pretrain_parser.add_argument(
        "--val-weight",
        type=float,
        default=1.0,
        help="weight for Macro Top-4 Cross-Entropy loss (default: 1.0)",
    )
    pretrain_parser.add_argument(
        "--micro-weight",
        type=float,
        default=0.5,
        help="weight for Micro combat round win probability BCE loss (default: 0.5)",
    )
    pretrain_parser.add_argument(
        "--contrast-weight",
        type=float,
        default=0.15,
        help="weight for Flow Time-Contrastive InfoNCE loss (default: 0.15)",
    )
    pretrain_parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="InfoNCE temperature tau (default: 0.07)",
    )
    pretrain_parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="optional cap on total snapshot pairs to load",
    )
    pretrain_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="compute device: 'cpu' or 'cuda' (default: auto)",
    )
    pretrain_parser.add_argument(
        "--synthetic",
        action="store_true",
        help="train on generated synthetic snapshot trajectories for quick testing",
    )
    pretrain_parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="batch frequency for real-time WandB metric logging (default: 10)",
    )
    pretrain_parser.add_argument(
        "--wandb-project",
        type=str,
        default="tft-embeddings",
        help="Weights & Biases project name (default: tft-embeddings)",
    )
    pretrain_parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="custom experiment run name for WandB tracking",
    )
    pretrain_parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB username or team entity name",
    )
    pretrain_parser.add_argument(
        "--wandb-group",
        type=str,
        default="pretrain-phase1",
        help="WandB experiment group",
    )
    pretrain_parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="disable WandB cloud logging and run in pure offline mode",
    )
    pretrain_parser.add_argument(
        "--allowed-tiers",
        nargs="+",
        default=["CHALLENGER"],
        help="Allowed tiers for training dataset (default: CHALLENGER). Pass ALL to use all data.",
    )

    cluster_parser = subcommands.add_parser(
        "cluster-compositions",
        aliases=["cluster"],
        help="extract composition archetypes and Z-Index centroids from curated endgame boards",
    )
    cluster_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(r"D:\tft-winner-data\set18\players"),
        help="directory containing player CSV files or path to single CSV",
    )
    cluster_parser.add_argument(
        "--trunk-checkpoint",
        type=Path,
        default=None,
        help="path to pre-trained MultiModalFusionTrunk checkpoint (.pt)",
    )
    cluster_parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("models/clustering"),
        help="destination directory for Z-Index artifacts and profiles",
    )
    cluster_parser.add_argument(
        "--min-stage",
        type=int,
        default=5,
        help="minimum stage threshold for endgame boards (default: 5)",
    )
    cluster_parser.add_argument(
        "--min-placement",
        type=int,
        default=4,
        help="maximum placement rank to retain (default: 4 for Top 4)",
    )
    cluster_parser.add_argument(
        "--n-clusters",
        "-k",
        type=int,
        default=15,
        help="number of composition archetypes K to extract (default: 15)",
    )
    cluster_parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="mini-batch size for latent extraction (default: 128)",
    )
    cluster_parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="PyTorch device (default: cuda if available else cpu)",
    )
    cluster_parser.add_argument(
        "--wandb-project",
        type=str,
        default="tft-clustering",
        help="Weights & Biases project name (default: tft-clustering)",
    )
    cluster_parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="custom experiment run name for WandB tracking",
    )
    cluster_parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB username or team entity name",
    )
    cluster_parser.add_argument(
        "--wandb-group",
        type=str,
        default="clustering-phase2",
        help="WandB experiment group",
    )
    cluster_parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="disable WandB cloud logging and run in pure offline mode",
    )
    cluster_parser.add_argument(
        "--synthetic",
        action="store_true",
        help="force usage of realistic synthetic endgame dataset for testing",
    )
    cluster_parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="maximum number of curated boards to extract",
    )
    cluster_parser.add_argument(
        "--allowed-tiers",
        nargs="+",
        default=["CHALLENGER"],
        help="Allowed tiers for training dataset (default: CHALLENGER). Pass ALL to use all data.",
    )

    transition_parser = subcommands.add_parser(
        "train-transition",
        aliases=["transition"],
        help="train the State Transition Predictor in frozen 320D latent space",
    )
    transition_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(r"D:\tft-winner-data\set18\players"),
        help="directory containing player CSV files or path to single CSV",
    )
    transition_parser.add_argument(
        "--trunk-checkpoint",
        type=Path,
        default=Path("models/trunk/trunk_best.pt"),
        help="path to pre-trained MultiModalFusionTrunk checkpoint (.pt)",
    )
    transition_parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("models/transition_predictor"),
        help="destination directory for transition predictor checkpoints",
    )
    transition_parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="number of training epochs (default: 15)",
    )
    transition_parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="mini-batch size of transition pairs (default: 512)",
    )
    transition_parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="AdamW learning rate (default: 1e-3)",
    )
    transition_parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="hidden dimension of residual MLP blocks (default: 512)",
    )
    transition_parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="number of residual MLP layers (default: 3)",
    )
    transition_parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="dropout rate (default: 0.1)",
    )
    transition_parser.add_argument(
        "--lambda-cosine",
        type=float,
        default=0.5,
        help="weight for Cosine Embedding directional loss (default: 0.5)",
    )
    transition_parser.add_argument(
        "--huber-beta",
        type=float,
        default=1.0,
        help="beta threshold for Smooth L1 / Huber loss (default: 1.0)",
    )
    transition_parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="fraction of matches to reserve for validation (default: 0.15)",
    )
    transition_parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="optional cap on total transition pairs to load",
    )
    transition_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="compute device: 'cpu' or 'cuda' (default: auto)",
    )
    transition_parser.add_argument(
        "--synthetic",
        action="store_true",
        help="train on synthetic snapshot transitions for quick testing",
    )
    transition_parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
        help="batch frequency for real-time WandB logging (default: 20)",
    )
    transition_parser.add_argument(
        "--wandb-project",
        type=str,
        default="tft-embeddings",
        help="Weights & Biases project name (default: tft-embeddings)",
    )
    transition_parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="custom experiment run name for WandB tracking",
    )
    transition_parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB username or team entity name",
    )
    transition_parser.add_argument(
        "--wandb-group",
        type=str,
        default="transition-predictor",
        help="WandB experiment group",
    )
    transition_parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="disable WandB cloud logging and run in pure offline mode",
    )
    transition_parser.add_argument(
        "--allowed-tiers",
        nargs="+",
        default=["CHALLENGER"],
        help="Allowed tiers for training dataset (default: CHALLENGER). Pass ALL to use all data.",
    )

    expand_parser = subcommands.add_parser(
        "expand-graph",
        aliases=["crawl-graph"],
        help="explore and expand the TFT player lobby navigation graph and manifest without full download",
    )
    expand_parser.add_argument(
        "--seed-players",
        nargs="+",
        default=None,
        help="starting seed player Riot IDs (e.g. NA1/Javi#401)",
    )
    expand_parser.add_argument(
        "--target-tiers",
        "--tiers",
        "--tier",
        "--leagues",
        "--league",
        nargs="+",
        default=None,
        help="filter and search players by league/tier, e.g. IRON BRONZE SILVER GOLD PLATINUM EMERALD DIAMOND MASTER GRANDMASTER CHALLENGER (default: all)",
    )
    expand_parser.add_argument(
        "--max-players",
        type=int,
        default=100,
        help="maximum number of player profiles to scan and catalog (default: 100)",
    )
    expand_parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="maximum lobby hop depth from seeds (default: 5)",
    )
    expand_parser.add_argument(
        "--tft-set",
        default="TFTSet18",
        help="TFT set identifier (default: TFTSet18)",
    )
    expand_parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("data/graph"),
        help="directory to persist players.csv, games.csv, and edges.csv (default: data/graph)",
    )
    expand_parser.add_argument(
        "--max-players-per-tier",
        type=int,
        default=None,
        help="optional cap on total app users collected per tier category to ensure balanced ELO distribution",
    )
    expand_parser.add_argument(
        "--max-games-per-tier",
        type=int,
        default=None,
        help="optional cap on candidate games collected per tier category (e.g. 1000). Tiers reaching this cap will be skipped.",
    )
    expand_parser.add_argument(
        "--riot-api-key",
        default=None,
        help="optional Riot Developer API key to directly query players by tier/division from Riot TFT League API",
    )
    expand_parser.add_argument(
        "--request-interval",
        type=float,
        default=1.2,
        help="minimum seconds between API requests (default: 1.2)",
    )
    expand_parser.add_argument(
        "--enable-leaderboard",
        action="store_true",
        default=False,
        help="allow replenishing high-tier players from MetaTFT leaderboard when graph queue is empty (default: False, pure lobby expansion)",
    )
    expand_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="do not resume from existing graph CSV files in output-dir",
    )

    download_parser = subcommands.add_parser(
        "download-games",
        aliases=["download-manifest"],
        help="download and extract PVP round training data from the games manifest produced by expand-graph",
    )
    download_parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data/graph"),
        help="directory containing games.csv and players.csv (default: data/graph)",
    )
    download_parser.add_argument(
        "--tft-set",
        default="TFTSet18",
        help="TFT set identifier (default: TFTSet18)",
    )
    download_parser.add_argument(
        "--target-tiers",
        nargs="+",
        default=None,
        help="filter and download only specific tiers",
    )
    download_parser.add_argument(
        "--max-games-per-tier",
        type=int,
        default=None,
        help="maximum games to download per tier category",
    )
    download_parser.add_argument(
        "--max-total-games",
        type=int,
        default=None,
        help="maximum total games to download",
    )
    download_parser.add_argument(
        "--tier-partitioned",
        action="store_true",
        help="partition output CSV storage by tier directory (e.g. data/tiers/gold/players/)",
    )
    download_parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("data"),
        help="destination directory for player CSV files (default: data)",
    )
    download_parser.add_argument(
        "--request-interval",
        type=float,
        default=1.2,
        help="minimum seconds between API requests (default: 1.2)",
    )
    download_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed used to shuffle and balance candidate game selection across leagues and players (default: 42)",
    )
    download_parser.add_argument(
        "--no-wait-cooldown",
        dest="auto_wait_cooldown",
        action="store_false",
        default=True,
        help="do not automatically wait for CDN rate limit cooldown (default: auto-waits cooldown and resumes)",
    )
    download_parser.add_argument(
        "--proxy",
        type=str,
        default=os.environ.get("ALL_PROXY") or os.environ.get("SOCKS_PROXY") or os.environ.get("HTTPS_PROXY"),
        help="optional HTTP or SOCKS5 proxy URL (e.g. socks5://127.0.0.1:9050 or socks5://127.0.0.1:40000)",
    )
    download_parser.add_argument(
        "--tor-control-port",
        type=int,
        default=None,
        help="optional Tor ControlPort (e.g. 9051) to instantly rotate IP identity on HTTP 429 rate limit",
    )

    plot_graph_parser = subcommands.add_parser(
        "plot-graph",
        aliases=["visualize-graph", "graph-viz"],
        help="generate interactive HTML and publication-ready multi-panel network plot of the player graph",
    )
    plot_graph_parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data/graph"),
        help="directory containing games.csv, players.csv, and edges.csv (default: data/graph)",
    )
    plot_graph_parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("reports/graph_viz"),
        help="directory to export visualization artifacts (default: reports/graph_viz)",
    )
    plot_graph_parser.add_argument(
        "--format",
        type=str,
        choices=["html", "png", "gif", "all"],
        default="all",
        help="export format: 'html', 'png', 'gif', or 'all' (default: all)",
    )
    plot_graph_parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="number of step-by-step discovery frames in animated progression GIF, or 0 for full graph (default: 0)",
    )
    plot_graph_parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="playback frames per second for animated progression GIF (default: 24)",
    )

    return parser


def _run_expand_graph(args: argparse.Namespace) -> int:
    """Step 1: Discover and map the player graph, cataloging players and games manifest without downloading full match observations."""
    graph = PlayerGraph()
    output_dir = Path(args.output_dir)
    if not getattr(args, "no_resume", False) and output_dir.exists():
        graph.load_csv(output_dir)

    target_tiers: set[str] | None = None
    if getattr(args, "target_tiers", None):
        target_tiers = {t.strip().upper() for t in args.target_tiers if t.strip()}
        if "ALL" in target_tiers:
            target_tiers = None

    # Seed initial nodes (always re-open seeds so their lobbies are explored, without reopening regular players)
    seed_players = getattr(args, "seed_players", None)
    if seed_players:
        for seed in seed_players:
            seed = seed.strip()
            if not seed:
                continue
            if "/" in seed:
                reg, _, riot_id = seed.partition("/")
            else:
                reg, riot_id = "na1", seed
            if "#" in riot_id:
                gname, _, tag = riot_id.partition("#")
                p_rid = f"{gname.strip()}#{tag.strip()}"
                p_node = graph.add_player(
                    riot_id=p_rid,
                    region=reg.lower(),
                    game_name=gname.strip(),
                    tag_line=tag.strip(),
                    depth=0,
                )
                p_node.scanned = False
    else:
        for reg, gname, tag, tier_cat in DEFAULT_MULTI_TIER_SEEDS:
            if target_tiers is None or tier_cat in target_tiers:
                p_rid = f"{gname}#{tag}"
                if p_rid not in graph.nodes:
                    graph.add_player(
                        riot_id=p_rid,
                        region=reg.lower(),
                        game_name=gname,
                        tag_line=tag,
                        tier=tier_cat,
                        depth=0,
                    )

    max_players_per_tier = getattr(args, "max_players_per_tier", None)
    max_games_per_tier = getattr(args, "max_games_per_tier", None)

    # If Riot API key is supplied or in environment, directly query players by tier from Riot League API
    riot_api_key = getattr(args, "riot_api_key", None) or os.environ.get("RIOT_API_KEY")
    if riot_api_key:
        try:
            from .metatft.riot_client import RiotTftClient
            riot_client = RiotTftClient(riot_api_key)
            all_tiers = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND"]
            tqdm.write("Querying Riot League API for multi-tier seed players...")
            for t in all_tiers:
                if target_tiers and t not in target_tiers:
                    continue
                tier_players = riot_client.fetch_tier_players(t, count=20)
                for p in tier_players:
                    graph.add_player(
                        riot_id=p.riot_id,
                        region=p.region.lower(),
                        game_name=p.game_name,
                        tag_line=p.tag_line,
                        tier=t,
                        depth=0,
                    )
        except Exception as e:
            tqdm.write(f"Riot API seeding error: {e}")

    client = MetaTftClient(
        minimum_request_interval_seconds=getattr(args, "request_interval", 1.0),
        retry_count=2,
    )

    allowed_queues = (1100,)
    app_users_scanned = sum(1 for n in graph.nodes.values() if n.scanned and n.is_app_user)
    total_profiles_checked = sum(1 for n in graph.nodes.values() if n.scanned)
    num_to_scan = getattr(args, "max_players", 100)
    if max_games_per_tier:
        target_additional = num_to_scan if num_to_scan != 100 else 10000
        max_to_scan = app_users_scanned + target_additional
    elif max_players_per_tier:
        num_tiers = len(target_tiers) if target_tiers else 9
        max_to_scan = app_users_scanned + max(num_to_scan, num_tiers * max_players_per_tier)
    elif seed_players:
        max_to_scan = app_users_scanned + max(num_to_scan, len(seed_players))
    else:
        max_to_scan = app_users_scanned + num_to_scan
    leaderboard_replenish_offset = 0

    print("\n" + "=" * 75)
    print(f" [TFT PLAYER GRAPH EXPANSION] Set: {args.tft_set} | Target: {max_to_scan} app users with tracked games")
    if target_tiers:
        print(f" Target Tiers: {', '.join(sorted(target_tiers))}")
    if max_players_per_tier:
        print(f" Max Players Per Tier Cap: {max_players_per_tier}")
    if max_games_per_tier:
        print(f" Max Games Per Tier Cap: {max_games_per_tier}")
    print(f" Loaded graph: {len(graph.nodes)} players ({app_users_scanned} app users), {len(graph.games)} games in manifest")
    print("=" * 75 + "\n")

    seen_timeline_extractions: set[str] = set()
    timeline_rate_limited = False
    last_timeline_attempt = 0.0

    interrupted = False
    try:
        with tqdm(total=max_to_scan, initial=app_users_scanned, desc="expanding app users", unit="app_user") as pbar:
            while app_users_scanned < max_to_scan:
                node = graph.get_next_priority_player(
                    target_tiers=target_tiers,
                    max_depth=args.max_depth,
                    max_players_per_tier=max_players_per_tier,
                    max_games_per_tier=max_games_per_tier,
                )
                if node is None:
                    # Only replenish from high-tier leaderboard if explicitly enabled
                    tier_dist = graph.get_tier_distribution()
                    enable_lb = getattr(args, "enable_leaderboard", False)
                    can_add_challenger = (
                        enable_lb
                        and (target_tiers is None or "CHALLENGER" in target_tiers or "GRANDMASTER" in target_tiers or "MASTER" in target_tiers)
                        and (max_players_per_tier is None or tier_dist.get("CHALLENGER", 0) < max_players_per_tier)
                    )
                    if can_add_challenger:
                        try:
                            lb_players = client.fetch_leaderboard_players(
                                count=50,
                                offset=leaderboard_replenish_offset,
                            )
                            leaderboard_replenish_offset += 50
                            added = 0
                            for lp in lb_players:
                                if lp.riot_id not in graph.nodes:
                                    graph.add_player(
                                        riot_id=lp.riot_id,
                                        region=lp.region.lower(),
                                        game_name=lp.game_name,
                                        tag_line=lp.tag_line,
                                        tier="CHALLENGER",
                                        depth=0,
                                    )
                                    added += 1
                            if added > 0:
                                node = graph.get_next_priority_player(
                                    target_tiers=target_tiers,
                                    max_depth=args.max_depth,
                                    max_players_per_tier=max_players_per_tier,
                                    max_games_per_tier=max_games_per_tier,
                                )
                        except Exception as e:
                            tqdm.write(f"Could not fetch leaderboard players: {e}")

                if node is None:
                    # Automatically mine lobbies from existing candidate games of under-represented tiers
                    now = time.monotonic()
                    mined_count = 0
                    if not timeline_rate_limited or (now - last_timeline_attempt > 120.0):
                        games_counts = graph.get_games_tier_distribution()
                        for game in list(graph.games.values()):
                            if max_games_per_tier and games_counts.get(game.tier, 0) >= max_games_per_tier:
                                continue
                            if target_tiers and game.tier not in target_tiers and game.tier != "UNKNOWN":
                                continue
                            if game.match_uuid in seen_timeline_extractions:
                                continue
                            seen_timeline_extractions.add(game.match_uuid)
                            last_timeline_attempt = time.monotonic()
                            try:
                                tl = client.fetch_timeline(game.timeline_url)
                                parts = client.extract_lobby_participants(tl)
                                focal_p = graph.nodes.get(game.focal_player_riot_id)
                                reg = focal_p.region if focal_p else "na1"
                                added_lobby = 0
                                for s_name, s_tag in parts:
                                    p_rid = f"{s_name}#{s_tag}"
                                    if p_rid != game.focal_player_riot_id and p_rid not in graph.nodes:
                                        graph.add_player(
                                            riot_id=p_rid,
                                            region=reg,
                                            game_name=s_name,
                                            tag_line=s_tag,
                                            tier=game.tier,
                                            depth=1,
                                            discovered_from=game.match_uuid,
                                        )
                                        graph.add_edge(game.focal_player_riot_id, p_rid, game.match_uuid)
                                        added_lobby += 1
                                if added_lobby > 0:
                                    mined_count += added_lobby
                                    tqdm.write(f" [Lobby Mining] Discovered +{added_lobby} new {game.tier} players from match {game.match_uuid[:12]}...")
                                    if mined_count >= 15:
                                        break
                            except MetaTftRequestError as e:
                                if "429" in str(e):
                                    timeline_rate_limited = True
                                    tqdm.write(" [!] Match CDN rate-limited (429). Pausing lobby timeline extraction.")
                                    break
                            except Exception:
                                pass

                    if mined_count > 0:
                        node = graph.get_next_priority_player(
                            target_tiers=target_tiers,
                            max_depth=args.max_depth,
                            max_players_per_tier=max_players_per_tier,
                            max_games_per_tier=max_games_per_tier,
                        )

                if node is None:
                    if target_tiers and not any(t in ("CHALLENGER", "GRANDMASTER", "MASTER", "DIAMOND") for t in target_tiers):
                        tqdm.write(
                            "No more unscanned player nodes matching tier criteria in graph queue.\n"
                            "Tip: To discover more low-tier players (Iron/Bronze/Silver/Gold), you can:\n"
                            "  1. Expand target tiers, e.g. --leagues IRON BRONZE SILVER GOLD\n"
                            "  2. Provide additional summoner seeds, e.g. --seed-players 'EUW1/Name#TAG' 'NA1/Name#TAG'\n"
                            "  3. Pass a Riot API key via --riot-api-key RGAPI-... (or RIOT_API_KEY env var)"
                        )
                    else:
                        tqdm.write("No more unscanned player nodes available matching tier criteria in graph queue.")
                    break

                # Update progress bar status before making the network request
                pbar.set_postfix({
                    "evaluating": f"{node.riot_id} ({node.tier})",
                    "scanned": total_profiles_checked,
                    "nodes": len(graph.nodes),
                    "games": len(graph.games),
                })
                pbar.refresh()

                new_lobby_nodes = 0
                def _safe(s: object) -> str:
                    return str(s).encode("ascii", errors="replace").decode("ascii")

                try:
                    prof = client.fetch_profile(
                        region=node.region,
                        game_name=node.game_name,
                        tag_line=node.tag_line,
                        tft_set=args.tft_set,
                    )
                    rating = prof.get("ranked", {}).get("rating_text", "UNRANKED")
                    node.rank_text = rating
                    node.tier = normalize_tier(rating)
                    node.scanned = True

                    candidates = client.tracked_timeline_candidates(
                        prof,
                        tft_set=args.tft_set,
                        allowed_queue_ids=allowed_queues,
                    )
                    node.app_matches = len(candidates)
                    node.is_app_user = len(candidates) > 0

                    # Record candidate matches to games manifest
                    for c in candidates:
                        graph.add_game(
                            match_uuid=c.app_match_uuid,
                            timeline_url=c.timeline_url,
                            focal_player_riot_id=node.riot_id,
                            tier=c.focal_tier or node.tier,
                            avg_rating=c.avg_match_rating,
                            tft_set=args.tft_set,
                        )

                    # Extract other players in the lobby to branch graph within this tier/MMR
                    now = time.monotonic()
                    if node.is_app_user and node.depth < args.max_depth and (not timeline_rate_limited or (now - last_timeline_attempt > 120.0)):
                        tier_g_cnt = graph.get_games_tier_distribution().get(node.tier, 0)
                        sample_count = 5 if (max_games_per_tier and tier_g_cnt < max_games_per_tier) else 2
                        for c in candidates[:sample_count]:
                            if c.app_match_uuid in seen_timeline_extractions:
                                continue
                            seen_timeline_extractions.add(c.app_match_uuid)
                            last_timeline_attempt = time.monotonic()
                            try:
                                timeline = client.fetch_timeline(c.timeline_url)
                                participants = client.extract_lobby_participants(timeline)
                                for s_name, s_tag in participants:
                                    p_rid = f"{s_name}#{s_tag}"
                                    if p_rid != node.riot_id and p_rid not in graph.nodes:
                                        graph.add_player(
                                            riot_id=p_rid,
                                            region=node.region,
                                            game_name=s_name,
                                            tag_line=s_tag,
                                            tier=c.focal_tier or node.tier,
                                            depth=node.depth + 1,
                                            discovered_from=c.app_match_uuid,
                                        )
                                        graph.add_edge(node.riot_id, p_rid, c.app_match_uuid)
                                        new_lobby_nodes += 1
                            except MetaTftRequestError as e:
                                if "429" in str(e):
                                    timeline_rate_limited = True
                                    tqdm.write(" [!] Match CDN rate-limited (429). Pausing lobby timeline extraction.")
                                    break
                            except Exception:
                                pass

                except MetaTftRequestError as error:
                    node.scanned = True
                    tqdm.write(f" [!] Skipped {_safe(node.riot_id)}: {_safe(error)}", file=sys.stderr)

                total_profiles_checked += 1
                if node.is_app_user:
                    app_users_scanned += 1
                    pbar.update(1)
                    lobby_msg = f", +{new_lobby_nodes} lobby players" if new_lobby_nodes > 0 else ""
                    tqdm.write(
                        f" [+] App User: {_safe(node.riot_id)} ({node.tier}) -> {node.app_matches} matches{lobby_msg} (Manifest: {len(graph.games)} games, {len(graph.nodes)} players)"
                    )
                else:
                    tqdm.write(
                        f" [-] Non-App:  {_safe(node.riot_id)} ({node.tier}) -> 0 app games"
                    )

                tier_dist = graph.get_tier_distribution()
                tier_str = " ".join(f"{t[:4]}:{cnt}" for t, cnt in sorted(tier_dist.items(), key=lambda x: -x[1]))
                pbar.set_postfix({
                    "scanned": total_profiles_checked,
                    "app_users": f"{app_users_scanned}/{max_to_scan}",
                    "nodes": len(graph.nodes),
                    "games": len(graph.games),
                    "tiers": tier_str or "scanning",
                    "last": f"{_safe(node.riot_id)} ({node.tier})",
                })
                pbar.refresh()

                # Checkpoint save every 5 players checked
                if total_profiles_checked % 5 == 0:
                    graph.save_csv(output_dir)

    except KeyboardInterrupt:
        interrupted = True
        tqdm.write("\n [!] Interrupted by user (Ctrl+C). Saving current graph state...")
    finally:
        graph.save_csv(output_dir)

    status_title = "[GRAPH EXPANSION PAUSED (SAVED)]" if interrupted else "[GRAPH EXPANSION COMPLETE]"
    total_app_users = sum(1 for n in graph.nodes.values() if n.is_app_user)
    total_scanned = sum(1 for n in graph.nodes.values() if n.scanned)
    total_queued = len(graph.nodes) - total_scanned
    total_games = len(graph.games)
    games_dist = graph.get_games_tier_distribution()

    print("\n" + "=" * 80)
    print(f" {status_title} Saved manifest to: {output_dir.resolve()}")
    print(f" Total Players: {len(graph.nodes)} ({total_scanned} scanned, {total_app_users} app users, {total_queued} queued)")
    print(f" Total Candidate Games: {total_games:,}")
    print(f" Total Lobby Edges: {len(graph.edges):,}")
    print("-" * 80)
    print(f" {'Tier / League':14} | {'Candidate Games':15} | {'Share (%)':9} | {'Scanned':7} | {'App Users':9} | {'In Queue':8}")
    print("-" * 80)
    for t in ("CHALLENGER", "GRANDMASTER", "MASTER", "DIAMOND", "EMERALD", "PLATINUM", "GOLD", "SILVER", "BRONZE", "IRON"):
        g_cnt = games_dist.get(t, 0)
        s_cnt = sum(1 for n in graph.nodes.values() if n.scanned and n.tier == t)
        a_cnt = sum(1 for n in graph.nodes.values() if n.scanned and n.is_app_user and n.tier == t)
        q_cnt = sum(1 for n in graph.nodes.values() if not n.scanned and n.tier == t)
        share = (g_cnt / total_games * 100) if total_games > 0 else 0.0
        if g_cnt > 0 or s_cnt > 0 or q_cnt > 0:
            print(f" {t:14} | {g_cnt:15,d} | {share:8.1f}% | {s_cnt:7d} | {a_cnt:9d} | {q_cnt:8d}")
    print("=" * 80 + "\n")
    return 0


def _rotate_tor_identity(control_host: str = "127.0.0.1", control_port: int = 9051) -> bool:
    """Request a fresh Tor circuit/IP by sending SIGNAL NEWNYM to Tor control port."""
    try:
        import socket
        try:
            import socks
            raw_socket = socks.socksocket(socket.AF_INET, socket.SOCK_STREAM)
            raw_socket.set_proxy()  # Direct localhost connection bypassing proxy monkeypatch
        except Exception:
            raw_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        raw_socket.settimeout(5.0)
        raw_socket.connect((control_host, control_port))
        raw_socket.sendall(b'AUTHENTICATE ""\r\n')
        auth_resp = raw_socket.recv(1024)
        if b"250" not in auth_resp:
            raw_socket.close()
            return False
        raw_socket.sendall(b"SIGNAL NEWNYM\r\n")
        sig_resp = raw_socket.recv(1024)
        raw_socket.close()
        return b"250" in sig_resp
    except Exception:
        return False


def _setup_proxy(proxy_str: str) -> None:
    """Configure global socket or HTTP proxy for requests."""
    from urllib.parse import urlparse
    parsed = urlparse(proxy_str if "://" in proxy_str else f"socks5://{proxy_str}")
    if parsed.scheme.startswith("socks"):
        try:
            import socks
            import socket
            proxy_type = socks.SOCKS5 if "5" in parsed.scheme else socks.SOCKS4
            port = parsed.port or 1080
            socks.set_default_proxy(
                proxy_type,
                parsed.hostname,
                port,
                rdns=True,
                username=parsed.username,
                password=parsed.password,
            )
            socket.socket = socks.socksocket
        except ImportError:
            raise RuntimeError("PySocks is required for SOCKS proxy support. Install with: pip install pysocks")
    elif parsed.scheme.startswith("http"):
        import os
        os.environ["http_proxy"] = proxy_str
        os.environ["https_proxy"] = proxy_str


def _run_download_games(args: argparse.Namespace) -> int:
    """Step 2: Download match timelines and extract PVP observations from the games manifest produced in Step 1."""
    proxy = getattr(args, "proxy", None) or os.environ.get("ALL_PROXY") or os.environ.get("SOCKS_PROXY") or os.environ.get("HTTPS_PROXY")
    if proxy:
        _setup_proxy(proxy)

    graph = PlayerGraph()
    manifest_dir = Path(args.manifest_dir)
    if not manifest_dir.exists():
        print(f"Error: manifest directory {manifest_dir} does not exist. Run expand-graph first.", file=sys.stderr)
        return 1

    graph.load_csv(manifest_dir)
    if len(graph.games) == 0:
        print(f"Error: no games found in {manifest_dir}/games.csv. Run expand-graph first.", file=sys.stderr)
        return 1

    target_tiers: set[str] | None = None
    if getattr(args, "target_tiers", None):
        target_tiers = {t.strip().upper() for t in args.target_tiers if t.strip()}
        if "ALL" in target_tiers:
            target_tiers = None

    client = MetaTftClient(
        minimum_request_interval_seconds=getattr(args, "request_interval", 1.2),
        retry_count=5,
    )
    writer = PlayerCsvWriter(args.output_dir, tier_partitioned=args.tier_partitioned)
    seen_game_ids = _existing_game_ids(args.output_dir)
    existing_by_tier = writer.existing_match_ids_by_tier()

    max_per_tier = getattr(args, "max_games_per_tier", None)
    max_total = getattr(args, "max_total_games", None)

    # 1. Filter out candidate games already on disk and group by tier
    unseen_by_tier: dict[str, list[GameManifestEntry]] = defaultdict(list)
    for game in graph.games.values():
        if target_tiers and game.tier not in target_tiers and game.tier != "UNKNOWN":
            continue
        if game.match_uuid in seen_game_ids:
            continue
        unseen_by_tier[game.tier].append(game)

    # 2. Pre-pick candidate games strictly respecting on-disk quota
    rng = random.Random(getattr(args, "seed", 42))
    selected_games: list[GameManifestEntry] = []
    prepicked_by_tier: dict[str, int] = {}

    all_candidate_tiers = sorted(
        set(unseen_by_tier.keys()) | set(existing_by_tier.keys()) | (target_tiers or set()),
        key=lambda t: ("CHALLENGER", "GRANDMASTER", "MASTER", "DIAMOND", "EMERALD", "PLATINUM", "GOLD", "SILVER", "BRONZE", "IRON").index(t) if t in ("CHALLENGER", "GRANDMASTER", "MASTER", "DIAMOND", "EMERALD", "PLATINUM", "GOLD", "SILVER", "BRONZE", "IRON") else 99
    )

    for tier in all_candidate_tiers:
        if target_tiers and tier not in target_tiers and tier != "UNKNOWN":
            continue
        t_games = unseen_by_tier.get(tier, [])
        on_disk = len(existing_by_tier.get(tier, set()))
        needed = max(0, max_per_tier - on_disk) if max_per_tier is not None else len(t_games)

        if needed == 0:
            prepicked_by_tier[tier] = 0
            continue

        if len(t_games) > needed:
            # Distribute picks evenly across focal players
            player_games: dict[str, list[GameManifestEntry]] = defaultdict(list)
            for g in t_games:
                player_games[g.focal_player_riot_id].append(g)

            players = list(player_games.keys())
            rng.shuffle(players)
            for p in players:
                rng.shuffle(player_games[p])

            t_selected: list[GameManifestEntry] = []
            while len(t_selected) < needed:
                added_any = False
                for p in players:
                    if player_games[p]:
                        t_selected.append(player_games[p].pop())
                        if len(t_selected) >= needed:
                            break
                        added_any = True
                if not added_any:
                    break
            selected_games.extend(t_selected)
            prepicked_by_tier[tier] = len(t_selected)
        else:
            selected_games.extend(t_games)
            prepicked_by_tier[tier] = len(t_games)

    # 3. Randomize across all tiers and players to guarantee even download flow
    rng.shuffle(selected_games)

    if max_total is not None and len(selected_games) > max_total:
        selected_games = selected_games[:max_total]

    # Pre-download distribution report
    cap_str = f"{max_per_tier:,} games/tier" if max_per_tier else "unlimited"
    print("\n" + "=" * 85)
    print(f" [TFT MATCH DOWNLOAD] Quotas & Distribution (Cap: {cap_str})")
    print(f" Destination: {Path(args.output_dir).resolve()} (Tier Partitioned: {args.tier_partitioned})")
    print("-" * 85)
    print(f" {'Tier / League':14} | {'Already on Disk':16} | {'Manifest Candidates':20} | {'To Download Now':16} | {'Cap Target':10}")
    print("-" * 85)
    for t in all_candidate_tiers:
        if target_tiers and t not in target_tiers and t != "UNKNOWN":
            continue
        disk_cnt = len(existing_by_tier.get(t, set()))
        man_cnt = len(unseen_by_tier.get(t, []))
        pick_cnt = prepicked_by_tier.get(t, 0)
        target_str = f"{max_per_tier:,}" if max_per_tier else "N/A"
        print(f" {t:14} | {disk_cnt:16,d} | {man_cnt:20,d} | {pick_cnt:16,d} | {target_str:10}")
    print("=" * 85)
    print(f" Total matches queued to download in this run: {len(selected_games):,}")
    print("=" * 85 + "\n")

    if len(selected_games) == 0:
        print(" [!] All tiers have already reached their requested download quota or have 0 candidates. Nothing to download.")
        return 0

    total_written = 0
    total_skipped = 0
    newly_written_by_tier: dict[str, int] = defaultdict(int)
    on_disk_tracker: dict[str, int] = {t: len(s) for t, s in existing_by_tier.items()}
    tier_str = " ".join(f"{t[:4]}:{on_disk_tracker.get(t, 0)}" for t in sorted(on_disk_tracker.keys()))

    interrupted = False
    rate_limited = False
    try:
        with tqdm(total=len(selected_games), desc="downloading games", unit="game") as pbar:
            for game in selected_games:
                if max_total is not None and total_written >= max_total:
                    break
                if max_per_tier is not None and on_disk_tracker.get(game.tier, 0) >= max_per_tier:
                    total_skipped += 1
                    pbar.update(1)
                    continue

                if game.match_uuid in seen_game_ids:
                    total_skipped += 1
                    pbar.update(1)
                    continue

                try:
                    timeline = client.fetch_timeline(game.timeline_url)
                    observations = extract_pvp_rounds(
                        timeline,
                        match_id=game.match_uuid,
                        tft_set=game.tft_set or args.tft_set,
                        game_version="unknown",
                        focal_tier=game.tier,
                        avg_match_rating=game.avg_rating,
                    )
                except MetaTftRequestError as e:
                    if "429" in str(e):
                        tor_port = getattr(args, "tor_control_port", None)
                        if tor_port is None and proxy and "9050" in proxy:
                            tor_port = 9051

                        tor_success = False
                        if tor_port:
                            tqdm.write(f"\n [!] MetaTFT match CDN is rate-limited (HTTP 429). Rotating Tor circuit via ControlPort {tor_port}...")
                            for rotate_attempt in range(1, 4):
                                if _rotate_tor_identity(control_port=tor_port):
                                    tqdm.write(f" [!] New Tor circuit requested ({rotate_attempt}/3). Waiting 2.5s for route establishment...")
                                    time.sleep(2.5)
                                    try:
                                        timeline = client.fetch_timeline(game.timeline_url)
                                        observations = extract_pvp_rounds(
                                            timeline,
                                            match_id=game.match_uuid,
                                            tft_set=game.tft_set or args.tft_set,
                                            game_version="unknown",
                                            focal_tier=game.tier,
                                            avg_match_rating=game.avg_rating,
                                        )
                                        tor_success = True
                                        break
                                    except MetaTftRequestError as retry_429:
                                        if "429" in str(retry_429):
                                            tqdm.write(f" [!] New exit node also rate-limited. Retrying Tor rotation ({rotate_attempt}/3)...")
                                            continue
                                        raise
                                    except Exception:
                                        raise
                                else:
                                    tqdm.write(f" [!] Failed to contact Tor ControlPort {tor_port}.")
                                    break
                        if tor_success:
                            pass
                        elif getattr(args, "auto_wait_cooldown", True):
                            retry_sec = getattr(e, "retry_after", None) or 300.0
                            tqdm.write(f"\n [!] MetaTFT match CDN is rate-limited (HTTP 429). Cooldown: ~{int(retry_sec)}s ({int(retry_sec)//60}m). Auto-waiting before resuming downloads...")
                            end_time = time.monotonic() + retry_sec + 2.0
                            while time.monotonic() < end_time:
                                rem = int(end_time - time.monotonic())
                                pbar.set_postfix({"cooldown": f"{rem}s remaining", "on_disk": tier_str})
                                time.sleep(min(5.0, max(1.0, rem)))
                            tqdm.write(" [!] Cooldown expired. Resuming download...")
                            try:
                                timeline = client.fetch_timeline(game.timeline_url)
                                observations = extract_pvp_rounds(
                                    timeline,
                                    match_id=game.match_uuid,
                                    tft_set=game.tft_set or args.tft_set,
                                    game_version="unknown",
                                    focal_tier=game.tier,
                                    avg_match_rating=game.avg_rating,
                                    )
                            except Exception as retry_err:
                                tqdm.write(f"skipped game {game.match_uuid}: {retry_err}", file=sys.stderr)
                                seen_game_ids.add(game.match_uuid)
                                continue
                        else:
                            tqdm.write(f"\n [!] MetaTFT match CDN is rate-limited ({e}). Pausing download gracefully to prevent losing quota.")
                            rate_limited = True
                            break
                    else:
                        tqdm.write(f"skipped game {game.match_uuid}: {e}", file=sys.stderr)
                        seen_game_ids.add(game.match_uuid)
                        continue
                except Exception as e:
                    tqdm.write(f"skipped game {game.match_uuid}: {e}", file=sys.stderr)
                    seen_game_ids.add(game.match_uuid)
                    continue

                if not observations:
                    seen_game_ids.add(game.match_uuid)
                    continue

                # Write game observations under the focal player's CSV
                focal_p = graph.nodes.get(game.focal_player_riot_id)
                region = focal_p.region if focal_p else "na1"
                path = writer.write_player_game(
                    observations,
                    collected_from_riot_id=game.focal_player_riot_id,
                    collected_from_region=region,
                    tier=game.tier,
                )
                if path is not None:
                    written_tier = observations[0].tier_category if (observations and observations[0].tier_category) else game.tier
                    newly_written_by_tier[written_tier] += 1
                    on_disk_tracker[written_tier] = on_disk_tracker.get(written_tier, 0) + 1
                    total_written += 1
                    seen_game_ids.add(game.match_uuid)
                    pbar.update(1)

                    tier_str = " ".join(f"{t[:4]}:{on_disk_tracker.get(t, 0)}" for t in sorted(on_disk_tracker.keys()))
                    pbar.set_postfix({
                        "written": total_written,
                        "skipped": total_skipped,
                        "on_disk": tier_str,
                    })
    except KeyboardInterrupt:
        interrupted = True
        tqdm.write("\n [!] Interrupted by user (Ctrl+C). Output files safely flushed.")

    status_title = "[MATCH DOWNLOAD PAUSED (RATE-LIMITED)]" if rate_limited else ("[MATCH DOWNLOAD PAUSED]" if interrupted else "[MATCH DOWNLOAD COMPLETE]")
    print("\n" + "=" * 85)
    print(f" {status_title} Written this session: {total_written}, Skipped: {total_skipped}")
    print("-" * 85)
    print(f" {'Tier / League':14} | {'Before Session':16} | {'Downloaded Now':16} | {'Total on Disk':14} | {'Status'}")
    print("-" * 85)
    for t in all_candidate_tiers:
        if target_tiers and t not in target_tiers and t != "UNKNOWN":
            continue
        before = len(existing_by_tier.get(t, set()))
        now = newly_written_by_tier.get(t, 0)
        tot = before + now
        if max_per_tier:
            status = "CAP REACHED" if tot >= max_per_tier else f"{tot}/{max_per_tier}"
        else:
            status = "OK"
        print(f" {t:14} | {before:16,d} | {now:16,d} | {tot:14,d} | {status}")
    print("=" * 85 + "\n")
    return 0


def _run_plot_graph(args: argparse.Namespace) -> int:
    """Generate interactive HTML, publication multi-panel plot, and animated progression GIF of the player graph."""
    from .dataset.graph_visualizer import export_interactive_html, export_progression_gif, export_static_plot

    manifest_dir = Path(args.manifest_dir)
    if not manifest_dir.exists():
        print(f"Error: manifest directory {manifest_dir} does not exist. Run expand-graph first.", file=sys.stderr)
        return 1

    graph = PlayerGraph()
    graph.load_csv(manifest_dir)
    if len(graph.nodes) == 0:
        print(f"Error: no players found in {manifest_dir}/players.csv. Run expand-graph first.", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = getattr(args, "format", "all")
    print("\n" + "=" * 75)
    print(f" [TFT PLAYER GRAPH VISUALIZATION] Loaded {len(graph.nodes):,} players, {len(graph.edges):,} edges, {len(graph.games):,} games")
    print(f" Output Directory: {out_dir.resolve()}")
    print("=" * 75 + "\n")

    if fmt in ("html", "all"):
        html_path = out_dir / "player_network.html"
        export_interactive_html(graph, html_path)
        print(f" [+] Interactive HTML Graph: {html_path.resolve()}")

    if fmt in ("png", "all"):
        png_path = out_dir / "player_network_dashboard.png"
        export_static_plot(graph, png_path)
        print(f" [+] Publication Dashboard Plot: {png_path.resolve()}")

    if fmt in ("gif", "all"):
        gif_path = out_dir / "graph_progression.gif"
        frames_cnt = getattr(args, "frames", 0)
        fps_val = getattr(args, "fps", 24)
        desc = f"full graph ({len(graph.nodes):,} nodes)" if frames_cnt <= 0 else f"{frames_cnt} steps"
        print(f" [*] Generating high-speed one-by-one discovery progression GIF ({desc} @ {fps_val} FPS)...")
        export_progression_gif(graph, gif_path, num_frames=frames_cnt, fps=fps_val)
        print(f" [+] Animated Progression GIF: {gif_path.resolve()}")

    print("\n" + "=" * 75 + "\n")
    return 0


def _collect_profile(args: argparse.Namespace) -> int:
    client = MetaTftClient(
        minimum_request_interval_seconds=getattr(args, "request_interval", 1.5),
        retry_count=5,
    )
    tier_partitioned = getattr(args, "tier_partitioned", False)
    writer = PlayerCsvWriter(args.output, tier_partitioned=tier_partitioned)
    player = LeaderboardPlayer(
        region=args.region,
        game_name=args.game_name,
        tag_line=args.tag_line,
        player_id=None,
    )
    blacklisted_players = writer.load_player_blacklist()
    if player.riot_id in blacklisted_players:
        print(f"skipping blacklisted player {player.riot_id}", file=sys.stderr)
        return 0

    profile = client.fetch_profile(
        region=player.region,
        game_name=player.game_name,
        tag_line=player.tag_line,
        tft_set=args.tft_set,
    )
    allowed_queues = getattr(args, "allowed_queues", (1100,))
    candidates = client.tracked_timeline_candidates(
        profile,
        tft_set=args.tft_set,
        allowed_queue_ids=allowed_queues,
    )
    seen_game_ids = _existing_game_ids(args.output)
    written, skipped = _download_player_games(
        client=client,
        writer=writer,
        player=player,
        candidates=candidates,
        tft_set=args.tft_set,
        games_per_player=args.games,
        seen_game_ids=seen_game_ids,
        remaining_games_budget=args.games,
        blacklisted_players=blacklisted_players,
    )
    print(f"collected profile {player.riot_id}: {written} new games written, {skipped} existing games skipped")
    return 0


DEFAULT_MULTI_TIER_SEEDS: tuple[tuple[str, str, str, str], ...] = (
    # Iron
    ("euw1", "bombos973", "EUW", "IRON"),
    ("euw1", "jiji123", "EUW", "IRON"),
    ("euw1", "KnutTarDeg", "EUW", "IRON"),
    ("euw1", "Emeloush", "65487", "IRON"),
    ("euw1", "Darknight3", "EUW", "IRON"),
    # Bronze
    ("euw1", "Gin Ichimaru", "BLCH", "BRONZE"),
    ("euw1", "Geep 5", "EUW", "BRONZE"),
    ("euw1", "Chizindikiro", "81100", "BRONZE"),
    ("euw1", "MezMez", "7615", "BRONZE"),
    ("euw1", "shynlah", "1183", "BRONZE"),
    # Silver
    ("euw1", "PrincessPingui", "EUW", "SILVER"),
    ("euw1", "Conso", "Prepu", "SILVER"),
    ("euw1", "tomzer", "6155", "SILVER"),
    ("la1", "KEIN", "Gato", "SILVER"),
    ("euw1", "LaPaf Patrouille", "LEPAF", "SILVER"),
    ("euw1", "roiloooo", "roilo", "SILVER"),
    ("euw1", "I Bims 1 Udo", "EUW", "SILVER"),
    ("la1", "Cizan", "Onion", "SILVER"),
    ("la1", "zilber232", "LAN", "SILVER"),
    ("la1", "Landhark", "GOT", "SILVER"),
    ("la1", "Roaan", "LAN", "SILVER"),
    ("la1", "Klinder05", "LAN", "SILVER"),
    ("la1", "WISDOM", "gabi", "SILVER"),
    ("euw1", "MiguelAFS", "EUW", "SILVER"),
    # Gold
    ("la1", "javi", "cjngg", "GOLD"),
    ("euw1", "xHinkel", "EUW", "GOLD"),
    ("euw1", "SAMY", "LES", "GOLD"),
    ("la1", "ACM1PTSapee", "2409", "GOLD"),
    ("la1", "LISIANTHUS", "Yith", "GOLD"),
    ("la1", "hornytwink", "lcket", "GOLD"),
    ("la1", "l Gio l", "LAN", "GOLD"),
    ("la1", "EstebanCL", "LAN01", "GOLD"),
    ("la1", "Greco4321", "revel", "GOLD"),
    ("euw1", "sauceaigredoucee", "EUW", "GOLD"),
    ("euw1", "ROI DES CAFARDS", "CLOPE", "GOLD"),
    ("euw1", "Calldnathan", "cumin", "GOLD"),
    ("euw1", "seven Bro 7", "BRo", "GOLD"),
    ("euw1", "Sixxpk", "Sixx", "GOLD"),
    ("euw1", "amleee", "EUW", "GOLD"),
    ("euw1", "luciano219", "EUW", "GOLD"),
    ("euw1", "GitanoBlanco", "EUW", "GOLD"),
    ("euw1", "PAN4ELO", "1993", "GOLD"),
    # Platinum
    ("na1", "Javi", "401", "PLATINUM"),
    ("na1", "Protos", "Colin", "PLATINUM"),
    # Emerald
    ("na1", "lettty", "420", "EMERALD"),
    ("na1", "TheMagykal", "NA1", "EMERALD"),
    ("na1", "TJF", "215", "EMERALD"),
    # Master
    ("na1", "Kurumx", "FREAK", "MASTER"),
    ("euw1", "Sologesang", "EUW", "MASTER"),
    # Grandmaster
    ("na1", "prestivent", "NA1", "GRANDMASTER"),
    ("na1", "robin", "007", "GRANDMASTER"),
    # Challenger
    ("na1", "Dishsoap", "NA1", "CHALLENGER"),
    ("kr", "Bebe872", "KR1", "CHALLENGER"),
    ("na1", "k3soju", "NA1", "CHALLENGER"),
    ("na1", "Setsuko", "NA1", "CHALLENGER"),
    ("na1", "Milala", "NA1", "CHALLENGER"),
    ("na1", "Wasianiverson", "NA1", "CHALLENGER"),
    ("euw1", "Double61", "EUW", "CHALLENGER"),
    ("euw1", "Salvyyy", "EUW", "CHALLENGER"),
)


def _collect_leaderboard(args: argparse.Namespace) -> int:
    if args.players <= 0:
        raise ValueError("players must be positive")
    if args.games_per_player is not None and args.games_per_player <= 0:
        raise ValueError("games_per_player must be positive when provided")
    if args.max_games is not None and args.max_games <= 0:
        raise ValueError("max_games must be positive when provided")

    client = MetaTftClient(
        minimum_request_interval_seconds=getattr(args, "request_interval", 1.5),
        retry_count=5,
    )
    tier_partitioned = getattr(args, "tier_partitioned", False)
    writer = PlayerCsvWriter(args.output, tier_partitioned=tier_partitioned)

    target_tiers_raw = getattr(args, "target_tiers", None)
    target_tiers: set[str] | None = None
    if target_tiers_raw:
        target_tiers = {t.strip().upper() for t in target_tiers_raw if t.strip()}
        if "ALL" in target_tiers:
            target_tiers = None

    crawl_depth = max(1, getattr(args, "crawl_depth", 1))
    max_games_per_tier = getattr(args, "max_games_per_tier", None)
    min_games_per_player = getattr(args, "min_games_per_player", None)
    games_by_tier: dict[str, int] = defaultdict(int)

    # Rank-stratified multi-tier queues for fair, balanced exploration
    tier_rotation = [
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
        "UNKNOWN",
    ]
    tier_queues: dict[str, list[tuple[LeaderboardPlayer, int]]] = defaultdict(list)
    tier_rotation_idx = 0
    seen_riot_ids: set[str] = set()

    seed_players = getattr(args, "seed_players", None)
    if seed_players:
        for seed in seed_players:
            seed = seed.strip()
            if not seed:
                continue
            if "/" in seed:
                reg, _, riot_id = seed.partition("/")
            else:
                reg, riot_id = "na1", seed
            if "#" in riot_id:
                gname, _, tag = riot_id.partition("#")
                p = LeaderboardPlayer(region=reg.lower(), game_name=gname.strip(), tag_line=tag.strip(), player_id=None)
                if p.riot_id not in seen_riot_ids:
                    tier_queues["UNKNOWN"].append((p, 1))
                    seen_riot_ids.add(p.riot_id)
    else:
        # Check if Riot Developer API key is available for direct tier querying
        riot_api_key = getattr(args, "riot_api_key", None)
        if not riot_api_key:
            import os
            riot_api_key = os.environ.get("RIOT_API_KEY")

        if riot_api_key:
            from .metatft.riot_client import RiotTftClient
            riot_client = RiotTftClient(riot_api_key)
            tiers_to_query = target_tiers or ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"]
            for t in tiers_to_query:
                try:
                    r_players = riot_client.fetch_tier_players(t, count=5)
                    for p in r_players:
                        if p.riot_id not in seen_riot_ids:
                            tier_queues[t].append((p, 1))
                            seen_riot_ids.add(p.riot_id)
                except Exception as e:
                    tqdm.write(f"Riot API error for tier {t}: {e}")

        # If crawl is enabled, prioritize multi-tier seeds across rank spectrum
        if crawl_depth > 1:
            for reg, gname, tag, tier_cat in DEFAULT_MULTI_TIER_SEEDS:
                p = LeaderboardPlayer(region=reg.lower(), game_name=gname, tag_line=tag, player_id=None)
                if p.riot_id not in seen_riot_ids:
                    tier_queues[tier_cat].append((p, 1))
                    seen_riot_ids.add(p.riot_id)

        # Leaderboard seeds
        needed = min(args.players, 2 if crawl_depth > 1 else args.players)
        lb_players = client.fetch_leaderboard_players(count=needed, offset=args.leaderboard_offset)
        for p in lb_players:
            if p.riot_id not in seen_riot_ids:
                tier_queues["CHALLENGER"].append((p, 1))
                seen_riot_ids.add(p.riot_id)

    seen_game_ids = _existing_game_ids(args.output)
    blacklisted_players = writer.load_player_blacklist()
    allowed_queues = getattr(args, "allowed_queues", (1100,))

    total_written = 0
    total_skipped = 0
    players_processed = 0

    def _pop_next_player() -> tuple[LeaderboardPlayer, int, str] | None:
        nonlocal tier_rotation_idx
        for _ in range(len(tier_rotation)):
            t = tier_rotation[tier_rotation_idx % len(tier_rotation)]
            tier_rotation_idx += 1
            if tier_queues[t]:
                p, d = tier_queues[t].pop(0)
                return p, d, t
        return None

    def _has_queued_players() -> bool:
        return any(bool(q) for q in tier_queues.values())

    with tqdm(total=args.players, desc="collecting players", unit="player") as player_progress:
        while _has_queued_players() and players_processed < args.players:
            if args.max_games is not None and (total_written + total_skipped) >= args.max_games:
                break

            next_item = _pop_next_player()
            if next_item is None:
                break
            current_player, current_depth, current_tier_hint = next_item

            if current_player.riot_id in blacklisted_players:
                tqdm.write(f"skipping blacklisted player {current_player.riot_id}")
                continue

            try:
                profile = client.fetch_profile(
                    region=current_player.region,
                    game_name=current_player.game_name,
                    tag_line=current_player.tag_line,
                    tft_set=args.tft_set,
                )
                candidates = client.tracked_timeline_candidates(
                    profile,
                    tft_set=args.tft_set,
                    allowed_queue_ids=allowed_queues,
                )
            except MetaTftRequestError as error:
                tqdm.write(f"skipped player {current_player.riot_id}: {error}", file=sys.stderr)
                continue

            if min_games_per_player is not None and len(candidates) < min_games_per_player:
                tqdm.write(f"skipping player {current_player.riot_id}: only {len(candidates)} tracked games (< {min_games_per_player})")
                continue

            remaining_budget = (
                (args.max_games - (total_written + total_skipped))
                if args.max_games is not None
                else None
            )

            discovered_lobby_players: list[tuple[str, str, str, str]] = []
            collect_lobby = current_depth < crawl_depth

            written, skipped = _download_player_games(
                client=client,
                writer=writer,
                player=current_player,
                candidates=candidates,
                tft_set=args.tft_set,
                games_per_player=args.games_per_player,
                seen_game_ids=seen_game_ids,
                remaining_games_budget=remaining_budget,
                blacklisted_players=blacklisted_players,
                discovered_lobby_players=discovered_lobby_players if collect_lobby else None,
                target_tiers=target_tiers,
                games_by_tier=games_by_tier,
                max_games_per_tier=max_games_per_tier,
            )
            total_written += written
            total_skipped += skipped
            if candidates:
                players_processed += 1
                player_progress.update(1)

            tier_summary = " ".join(
                f"{t[:4]}:{cnt}"
                for t, cnt in sorted(games_by_tier.items(), key=lambda x: -x[1])
                if cnt > 0
            )
            player_progress.set_postfix({
                "player": current_player.riot_id,
                "new": total_written,
                "tiers": tier_summary or "scanning",
                "cached": total_skipped,
            })

            # Add newly discovered players to their respective tier queue
            if collect_lobby and discovered_lobby_players:
                for reg, gname, tag, p_tier in discovered_lobby_players:
                    rid = f"{gname}#{tag}"
                    if rid not in seen_riot_ids and rid not in blacklisted_players:
                        seen_riot_ids.add(rid)
                        discovered_p = LeaderboardPlayer(
                            region=reg.lower(),
                            game_name=gname,
                            tag_line=tag,
                            player_id=None,
                        )
                        assigned_tier = p_tier if p_tier in tier_queues else "UNKNOWN"
                        tier_queues[assigned_tier].append((discovered_p, current_depth + 1))

    print(f"collection finished: {total_written} new games written, {total_skipped} existing games skipped across player CSV files")
    return 0


def _collect_timeline(args: argparse.Namespace) -> int:
    client = MetaTftClient(
        minimum_request_interval_seconds=getattr(args, "request_interval", 1.5),
        retry_count=5,
    )
    writer = PlayerCsvWriter(args.output)
    timeline = client.fetch_timeline(args.timeline_url)
    observations = extract_pvp_rounds(
        timeline,
        match_id=args.match_id,
        tft_set=args.tft_set,
        game_version=args.game_version,
    )
    path = writer.write_game(observations)
    if path is None:
        writer.add_to_blacklist(args.match_id, reason="no valid PVP rounds")
        print("no valid PVP observations found; match blacklisted and no CSV written", file=sys.stderr)
        return 0
    print(f"wrote {len(observations)} PVP observations to {path}")
    return 0


def _download_player_games(
    *,
    client: MetaTftClient,
    writer: PlayerCsvWriter,
    player: LeaderboardPlayer,
    candidates: Sequence[TrackedTimelineCandidate],
    tft_set: str,
    games_per_player: int | None,
    seen_game_ids: set[str],
    remaining_games_budget: int | None,
    blacklisted_players: set[str] | None = None,
    max_consecutive_no_pvp: int = 5,
    discovered_lobby_players: list[tuple[str, str, str, str]] | None = None,
    target_tiers: set[str] | None = None,
    games_by_tier: dict[str, int] | None = None,
    max_games_per_tier: int | None = None,
) -> tuple[int, int]:
    """Download up to games_per_player unique games for one player, returning (written, skipped)."""

    written = 0
    skipped = 0
    consecutive_no_pvp = 0
    for candidate in candidates:
        if games_per_player is not None and (written + skipped) >= games_per_player:
            break
        if remaining_games_budget is not None and (written + skipped) >= remaining_games_budget:
            break

        game_id = candidate.app_match_uuid
        if game_id in seen_game_ids:
            skipped += 1
            continue

        try:
            timeline = client.fetch_timeline(candidate.timeline_url)
        except MetaTftRequestError as error:
            # Temporary connection or HTTP request error: do NOT blacklist to allow retry in future runs.
            tqdm.write(f"skipped game {game_id} from {player.riot_id} (temporary request error): {error}", file=sys.stderr)
            seen_game_ids.add(game_id)
            continue

        try:
            observations = extract_pvp_rounds(
                timeline,
                match_id=game_id,
                tft_set=tft_set,
                game_version=candidate.game_version or "unknown",
                focal_tier=candidate.focal_tier,
                focal_rating_numeric=candidate.focal_rating_numeric,
                avg_match_rating=candidate.avg_match_rating,
                avg_match_rating_numeric=candidate.avg_match_rating_numeric,
                focal_augments=candidate.focal_augments,
            )
        except (TimelineValidationError, ValueError) as error:
            writer.add_to_blacklist(game_id, reason=f"validation error: {error}")
            seen_game_ids.add(game_id)
            consecutive_no_pvp += 1
            tqdm.write(f"blacklisted game {game_id} from {player.riot_id} (validation error): {error}", file=sys.stderr)
            if consecutive_no_pvp >= max_consecutive_no_pvp:
                writer.add_player_to_blacklist(player.riot_id, reason=f"{consecutive_no_pvp} consecutive games with no valid PVP rounds")
                if blacklisted_players is not None:
                    blacklisted_players.add(player.riot_id)
                tqdm.write(f"blacklisted player {player.riot_id}: {consecutive_no_pvp} consecutive games with no valid PVP rounds; skipping player", file=sys.stderr)
                break
            continue

        if not observations:
            writer.add_to_blacklist(game_id, reason="no valid PVP rounds")
            seen_game_ids.add(game_id)
            consecutive_no_pvp += 1
            tqdm.write(f"blacklisted game {game_id} from {player.riot_id}: no valid PVP rounds")
            if consecutive_no_pvp >= max_consecutive_no_pvp:
                writer.add_player_to_blacklist(player.riot_id, reason=f"{consecutive_no_pvp} consecutive games with no valid PVP rounds")
                if blacklisted_players is not None:
                    blacklisted_players.add(player.riot_id)
                tqdm.write(f"blacklisted player {player.riot_id}: {consecutive_no_pvp} consecutive games with no valid PVP rounds; skipping player", file=sys.stderr)
                break
            continue

        # Check target tiers if specified
        from .dataset.models import normalize_tier
        obs_tier = normalize_tier(
            (observations[0].tier_category if observations else None)
            or candidate.focal_tier
            or candidate.avg_match_rating
        )

        if discovered_lobby_players is not None:
            lobby_participants = MetaTftClient.extract_lobby_participants(timeline)
            for s_name, s_tag in lobby_participants:
                discovered_lobby_players.append((player.region, s_name, s_tag, obs_tier))

        if target_tiers is not None:
            focal_tier_upper = (candidate.focal_tier or "").upper()
            avg_rating_upper = (candidate.avg_match_rating or "").upper()
            matches_target = (
                obs_tier in target_tiers
                or any(t in focal_tier_upper for t in target_tiers)
                or any(t in avg_rating_upper for t in target_tiers)
            )
            if not matches_target:
                seen_game_ids.add(game_id)
                continue

        # Check per-tier quota
        if max_games_per_tier is not None and games_by_tier is not None:
            if games_by_tier.get(obs_tier, 0) >= max_games_per_tier:
                seen_game_ids.add(game_id)
                continue

        path = writer.write_player_game(
            observations,
            collected_from_riot_id=player.riot_id,
            collected_from_region=player.region,
            match_id_ow=candidate.match_id_ow,
        )
        if path is None:
            writer.add_to_blacklist(game_id, reason="no valid PVP rounds")
            seen_game_ids.add(game_id)
            consecutive_no_pvp += 1
            tqdm.write(f"blacklisted game {game_id} from {player.riot_id}: no valid PVP rounds")
            if consecutive_no_pvp >= max_consecutive_no_pvp:
                writer.add_player_to_blacklist(player.riot_id, reason=f"{consecutive_no_pvp} consecutive games with no valid PVP rounds")
                if blacklisted_players is not None:
                    blacklisted_players.add(player.riot_id)
                tqdm.write(f"blacklisted player {player.riot_id}: {consecutive_no_pvp} consecutive games with no valid PVP rounds; skipping player", file=sys.stderr)
                break
            continue

        seen_game_ids.add(game_id)
        written += 1
        consecutive_no_pvp = 0
        if games_by_tier is not None:
            games_by_tier[obs_tier] += 1

    return written, skipped


def _existing_game_ids(root: Path) -> set[str]:
    seen: set[str] = set()
    writer = PlayerCsvWriter(root)
    seen.update(writer.all_existing_match_ids())
    seen.update(writer.load_blacklist())
    return seen


def _run_simulation(args: argparse.Namespace) -> int:
    """Execute a simulated TFT match and export the interactive visual replay HTML dashboard."""
    from .simulation import BaseBot, StandardTempoBot, TFTGame, get_set_data
    from .simulation.combat import CombatResolver, HeuristicCombatResolver, MLCombatResolver
    from .simulation.visualizer import GameRecorder, generate_visual_html

    set_data = get_set_data(args.set)
    set_slug = set_data.set_name.lower()
    seed = args.seed

    print("\n" + "=" * 70)
    print(f" [TFT VISUAL SIMULATION] Running match ({set_data.set_name}, seed={seed})...")

    # 1. Combat resolver
    combat_resolver: CombatResolver = HeuristicCombatResolver()
    if args.round_winner_model and Path(args.round_winner_model).exists():
        combat_resolver = MLCombatResolver(model_pipeline=args.round_winner_model)
        print(f"  Combat Model: ML (LightGBM) from {args.round_winner_model}")
    else:
        print("  Combat Model: Heuristic Combat Resolver")

    # 2. Focal player bot policy (RL or Heuristic)
    focal_bot: BaseBot
    if args.rl_model and Path(args.rl_model).exists():
        import torch
        from tft_ai_player.rl.agent_policy import RLBot
        from tft_ai_player.rl.models.networks import TFTActorCritic

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TFTActorCritic(
            num_champs=len(set_data.champions) + 1,
            num_items=len(set_data.items) + 1,
        )
        weights = torch.load(args.rl_model, map_location=device)
        if isinstance(weights, dict) and "model_state_dict" in weights:
            model.load_state_dict(weights["model_state_dict"])
        else:
            model.load_state_dict(weights)
        model.to(device)
        model.eval()
        focal_bot = RLBot(model=model, set_data=set_data, deterministic=True, device=device)
        print(f"  Focal Agent Policy: RL Neural Policy from {args.rl_model} (Device: {device})")
    else:
        focal_bot = StandardTempoBot()
        print("  Focal Agent Policy: StandardTempoBot (Rule-based Baseline)")

    print("=" * 70)

    game = TFTGame(set_data=set_data, seed=seed, combat_resolver=combat_resolver)
    recorder = GameRecorder(game)
    recorder.capture_snapshot(event_type="GAME_START")

    round_count = 0
    while not game.is_over and round_count < args.max_rounds:
        round_count += 1
        rinfo = game.stage_manager.get_current_round_info()

        # Bot planning actions
        game.execute_bot_turns()

        focal_p = game.get_focal_player()
        if focal_p.alive:
            focal_bot.take_turn(
                player=focal_p,
                pool=game.pool,
                set_data=game.set_data,
                stage=rinfo.stage,
                round_in_stage=rinfo.round_in_stage,
                rng=game.rng,
            )

        combat_results = game.resolve_round_phase()
        recorder.capture_snapshot(
            event_type="ROUND_RESOLVED",
            combat_results=combat_results,
        )
        print(f"  Recorded Stage {rinfo.stage_str} ({rinfo.round_type.value}) - {len(game.players)} Players Monitored")

    print(f"\n[+] Simulation complete! Recorded {len(recorder.frames)} visual snapshots.")

    if args.output_path is not None:
        out_path = Path(args.output_path).resolve()
    else:
        dir_path = Path(args.output_dir).resolve()
        out_path = dir_path / f"tft_simulation_{set_slug}_seed_{seed}.html"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    html_content = generate_visual_html(recorder.frames, title=f"TFT Simulation ({set_data.set_name}, Seed {seed})")
    out_path.write_text(html_content, encoding="utf-8")
    print(f" [+] Visual Dashboard exported to: file:///{out_path.as_posix()}")

    if args.open_browser:
        import webbrowser
        try:
            webbrowser.open(out_path.as_uri())
        except Exception:
            pass

    return 0


def _run_rl_train(args: argparse.Namespace) -> int:
    """Execute RL Maskable PPO training loop with AlphaStar League, WandB tracking, and health telemetry."""
    import torch
    from tft_ai_player.rl.train import LeagueTrainer
    from tft_ai_player.simulation.sets.set18 import get_set18_data

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_wandb = not getattr(args, "no_wandb", False)

    checkpoint_dir = args.checkpoint_dir or get_default_rl_checkpoint_dir(args.run_name)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print(" [TFT RL TRAINING] PPO & AlphaStar League Pipeline (|A|=111, Obs=704D)")
    print(f"  Target Set: Set 18 | Device: {device_name.upper()}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU Hardware: {gpu_name} ({vram_total:.1f} GB VRAM)")
    print(f"  Generations: {args.generations} | Rollout Steps: {args.rollout_steps} | Batch Size: {args.batch_size}")
    print(f"  Trunk Checkpoint: {args.trunk_checkpoint}")
    print(f"  Checkpoint Dir: {checkpoint_dir} | Run Name: {args.run_name}")
    combat_desc = f"LightGBM ({args.round_winner_model})" if args.round_winner_model else "Deep Learning (GPU Neural Trunk combat_head)"
    print(f"  Combat Engine: {combat_desc}")
    print("=" * 75 + "\n")

    trainer = LeagueTrainer(
        set_data=get_set18_data(),
        trunk_checkpoint=args.trunk_checkpoint,
        world_model_checkpoint=args.world_model_checkpoint,
        z_index_path=args.z_index_path,
        round_winner_model_path=args.round_winner_model,
        lr=args.lr,
        total_rollout_steps=args.rollout_steps,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        max_generations=args.generations,
        eval_interval=args.eval_every,
        snapshot_interval=args.snapshot_every,
        device=device_name,
        checkpoint_dir=checkpoint_dir,
        run_name=args.run_name,
        use_wandb=use_wandb,
        wandb_project=getattr(args, "wandb_project", "tft-ai-league"),
        wandb_entity=getattr(args, "wandb_entity", None),
        wandb_group=getattr(args, "wandb_group", None) or args.run_name,
        resume=getattr(args, "resume", False),
        resume_from=getattr(args, "resume_from", None),
    )

    trainer.run_training_loop()
    return 0


def _run_rl_league(args: argparse.Namespace) -> int:
    """Run tournament matches across the TFT multi-agent league and update Elo ratings."""
    from tft_ai_player.rl.evaluation.evaluator import TournamentEvaluator
    from tft_ai_player.rl.evaluation.report import generate_league_markdown_report, print_league_terminal_summary
    from tft_ai_player.rl.league.league_manager import LeagueManager

    print("\n" + "=" * 70)
    print(f" [TFT MULTI-AGENT LEAGUE] Simulating {args.matches} 8-player tournament matches...")
    print("=" * 70)

    ckpt_dir = resolve_league_checkpoint_dir(args.checkpoint_dir, "ppo_alphastar_v3")
    print(f"  Checkpoint Dir: {ckpt_dir}")
    league = LeagueManager(checkpoint_dir=ckpt_dir)
    evaluator = TournamentEvaluator(league)

    agent_pool = list(league.profiles.keys())
    if len(agent_pool) < 8:
        agent_pool = (agent_pool * 8)[:8]

    for match_idx in range(1, args.matches + 1):
        seed = 1000 + match_idx
        seats = [str(s) for s in random.choices(agent_pool, k=8)]
        result = evaluator.run_match(agent_seats=seats, seed=seed)
        podium = [f"#{rank} {aid}" for aid, rank in sorted(result.placements.items(), key=lambda x: x[1])[:3]]
        print(f"  Match {match_idx:02d}/{args.matches:02d} (Seed {seed}): Podium -> {', '.join(podium)}")

    print("\n[+] Tournament series complete!")
    print_league_terminal_summary(league)

    if args.markdown_out:
        out_path = Path(args.markdown_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report_md = generate_league_markdown_report(league)
        out_path.write_text(report_md, encoding="utf-8")
        print(f" [+] Markdown Leaderboard written to: file:///{out_path.as_posix()}")

    return 0


def _run_rl_visualize_progression(args: argparse.Namespace) -> int:
    """Generate AlphaStar-style strategy landscape 2D progression and unit composition plots."""
    from tft_ai_player.rl.visualization.strategy_landscape import generate_alphastar_progression_plot

    run_name = getattr(args, "run_name", "ppo_alphastar_v3")
    ckpt_dir = resolve_league_checkpoint_dir(args.checkpoint_dir, run_name)

    print("\n" + "=" * 80)
    print(" [TFT STRATEGY VISUALIZER] AlphaStar League Progression & Strategy Space")
    print(f"  Checkpoint Dir: {ckpt_dir} | Run: {run_name} | Output Dir: {args.output_dir}")
    print(f"  Format: {args.format.upper()} | FPS: {args.fps} | Stride: {args.stride}")
    print("=" * 80 + "\n")

    formats = ["gif", "png", "html"] if args.format == "all" else [args.format]
    results = generate_alphastar_progression_plot(
        checkpoint_dir=ckpt_dir,
        output_dir=args.output_dir,
        formats=formats,
        fps=args.fps,
        stride=args.stride,
        agent_name=args.agent_name,
    )

    print("\n[+] AlphaStar Strategy Progression generated successfully:")
    for fmt, path in results.items():
        print(f"  - [{fmt.upper()}] file:///{path.resolve().as_posix()}")

    if getattr(args, "wandb", False):
        try:
            from pathlib import Path
            from tft_ai_player.rl.logger import WandBLogger
            wandb_logger = WandBLogger(
                project=getattr(args, "wandb_project", "tft-ai-league"),
                run_name=run_name,
                enabled=True,
            )
            gen_count = len(list(Path(ckpt_dir).glob("gen_*"))) or 1
            wandb_logger.log_strategy_progression(results, step=gen_count)
        except Exception as e:
            print(f" [!] WandB upload skipped: {e}")

    if getattr(args, "open_browser", False) and "html" in results:
        import webbrowser
        try:
            webbrowser.open(results["html"].resolve().as_uri())
        except Exception:
            pass

    return 0


def _run_pretrain_trunk(args: argparse.Namespace) -> int:
    """Execute pre-training of Multi-Modal Fusion Trunk."""
    from tft_ai_player.embeddings import (
        ChampionVocabulary,
        ItemVocabulary,
        TraitVocabulary,
        TFTPretrainDataset,
        TrunkPreTrainer,
        create_synthetic_trajectory_dataset,
    )

    print("\n" + "=" * 75)
    print(" [TFT MULTI-MODAL TRUNK] Phase 1: Dual-Objective Multi-Task Pre-training")
    print("=" * 75)

    vocab = ChampionVocabulary()
    item_vocab = ItemVocabulary()
    trait_vocab = TraitVocabulary()

    if args.synthetic or not Path(args.data_dir).exists():
        if not args.synthetic:
            print(f"[!] Data directory '{args.data_dir}' not found. Falling back to synthetic dataset.")
        print(" [+] Generating synthetic trajectory dataset...")
        dataset = create_synthetic_trajectory_dataset(
            num_matches=50,
            rounds_per_match=16,
            vocab=vocab,
            item_vocab=item_vocab,
            trait_vocab=trait_vocab,
        )
    else:
        print(f" [+] Loading snapshot dataset from: {args.data_dir}")
        allowed_tiers = None if "ALL" in [t.upper() for t in getattr(args, "allowed_tiers", ["CHALLENGER"])] else getattr(args, "allowed_tiers", ["CHALLENGER"])
        dataset = TFTPretrainDataset(
            data=args.data_dir,
            vocab=vocab,
            item_vocab=item_vocab,
            trait_vocab=trait_vocab,
            max_samples=args.max_samples,
            allowed_tiers=allowed_tiers,
        )

    print(
        f" [+] Dataset loaded: {len(dataset)} snapshot pairs | "
        f"Vocabs: {len(vocab)} champs, {len(item_vocab)} items, {len(trait_vocab)} traits"
    )

    if len(dataset) == 0:
        print(" [!] No valid snapshot pairs found in dataset.", file=sys.stderr)
        return 1

    trainer = TrunkPreTrainer(
        vocab=vocab,
        item_vocab=item_vocab,
        trait_vocab=trait_vocab,
        champ_embed_dim=args.embed_dim,
        board_feat_dim=args.board_dim,
        fused_dim=args.fused_dim,
        lr=args.lr,
        value_weight=args.val_weight,
        micro_weight=args.micro_weight,
        contrast_weight=args.contrast_weight,
        temperature=args.temperature,
        log_interval=args.log_interval,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.run_name,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        device=args.device,
    )

    summary = trainer.fit(
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 75)
    print(f" [+] Pre-training completed in {summary['total_time_sec']}s!")
    print(f"     Artifacts saved to: {Path(args.output_dir).resolve()}")
    print("=" * 75 + "\n")
    return 0


def _run_cluster_compositions(args: argparse.Namespace) -> int:
    import torch
    from .embeddings.cluster import run_clustering_pipeline

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    allowed_tiers = None if "ALL" in [t.upper() for t in getattr(args, "allowed_tiers", ["CHALLENGER"])] else getattr(args, "allowed_tiers", ["CHALLENGER"])

    results = run_clustering_pipeline(
        data_dir=args.data_dir,
        trunk_checkpoint=args.trunk_checkpoint,
        min_stage=args.min_stage,
        max_placement=args.min_placement,
        n_clusters=args.n_clusters,
        batch_size=args.batch_size,
        device=device,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        output_dir=args.output_dir,
        synthetic=args.synthetic,
        max_samples=args.max_samples,
        allowed_tiers=allowed_tiers,
    )
    return 0 if "error" not in results else 1


def _run_train_transition(args: argparse.Namespace) -> int:
    import torch
    from tft_ai_player.embeddings import (
        ChampionVocabulary,
        ItemVocabulary,
        MultiModalFusionTrunk,
        StateTransitionPredictor,
        TraitVocabulary,
        TransitionPredictorTrainer,
        TransitionTrajectoryDataset,
        create_synthetic_transition_dataset,
    )

    print("\n" + "=" * 80)
    print(" [TFT TRANSITION PREDICTOR] Offline Behavioral Cloning in 320D Latent Space")
    print("=" * 80)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    vocab = ChampionVocabulary()
    item_vocab = ItemVocabulary()
    trait_vocab = TraitVocabulary()

    # Load frozen trunk
    trunk_ckpt = Path(args.trunk_checkpoint)
    if trunk_ckpt.exists():
        print(f" [+] Loading frozen pre-trained trunk from: {trunk_ckpt.resolve()}")
        trunk = MultiModalFusionTrunk.load_trunk(trunk_ckpt, map_location=device)
    else:
        print(f" [!] Trunk checkpoint not found at '{trunk_ckpt}'. Initializing blank trunk for demonstration.")
        trunk = MultiModalFusionTrunk(fused_dim=320)
    trunk.freeze()
    trunk.eval()

    if args.synthetic or not Path(args.data_dir).exists():
        if not args.synthetic:
            print(f"[!] Data directory '{args.data_dir}' not found. Falling back to synthetic dataset.")
        print(" [+] Generating synthetic transition trajectory dataset...")
        dataset = create_synthetic_transition_dataset(
            num_matches=40,
            rounds_per_match=15,
            vocab=vocab,
            item_vocab=item_vocab,
            trait_vocab=trait_vocab,
        )
    else:
        print(f" [+] Loading transition trajectory dataset from: {args.data_dir}")
        allowed_tiers = None if "ALL" in [t.upper() for t in getattr(args, "allowed_tiers", ["CHALLENGER"])] else getattr(args, "allowed_tiers", ["CHALLENGER"])
        dataset = TransitionTrajectoryDataset(
            data_dir=args.data_dir,
            vocab=vocab,
            item_vocab=item_vocab,
            trait_vocab=trait_vocab,
            max_samples=args.max_samples,
            allowed_tiers=allowed_tiers,
        )

    print(
        f" [+] Dataset loaded: {len(dataset)} PVP transition pairs | "
        f"Vocabs: {len(vocab)} champs, {len(item_vocab)} items, {len(trait_vocab)} traits"
    )

    if len(dataset) == 0:
        print(" [!] No valid transition pairs found in dataset.", file=sys.stderr)
        return 1

    latent_dim = trunk.fused_dim
    predictor = StateTransitionPredictor(
        input_dim=latent_dim,
        hidden_dim=args.hidden_dim,
        output_dim=latent_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_residual_delta=True,
    )

    trainer = TransitionPredictorTrainer(
        predictor=predictor,
        trunk=trunk,
        lr=args.lr,
        lambda_cosine=args.lambda_cosine,
        huber_beta=args.huber_beta,
        log_interval=args.log_interval,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.run_name,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        device=device,
    )

    summary = trainer.fit(
        dataset=dataset,
        val_split=args.val_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 80)
    print(f" [+] Transition Predictor training completed in {summary['total_time_sec']}s!")
    print(f"     Best Val Cosine Similarity: {summary['best_val_cosine_similarity']:.4f}")
    print(f"     Artifacts saved to: {Path(args.output_dir).resolve()}")
    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())



