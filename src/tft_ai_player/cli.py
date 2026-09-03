"""Command-line entry points for MetaTFT discovery and timeline extraction."""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Sequence
from pathlib import Path

from tqdm import tqdm

from .dataset import PlayerCsvWriter, TimelineValidationError, extract_pvp_rounds
from .metatft import LeaderboardPlayer, MetaTftClient, MetaTftRequestError, TrackedTimelineCandidate


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return a process exit code."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
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
    except (MetaTftRequestError, TimelineValidationError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    parser.error(f"unsupported command: {args.command}")
    return 2


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
    simulate_parser.add_argument(
        "--round-winner-model",
        type=str,
        default="D:/tft-winner-data/set18/models/round_winner_model.joblib",
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
    rl_train_parser.add_argument(
        "--trunk-checkpoint",
        type=str,
        default="D:/tft-winner-data/set18/models/trunk/best_model.pt",
        help="path to pre-trained frozen MultiModalFusionTrunk checkpoint",
    )
    rl_train_parser.add_argument(
        "--world-model-checkpoint",
        type=str,
        default="D:/tft-winner-data/set18/models/transition/best_model.pt",
        help="path to pre-trained StateTransitionPredictor (World Model) checkpoint",
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
    default_ckpt_dir = "D:/tft-winner-data/set18/checkpoints" if Path("D:/tft-winner-data/set18").exists() else "checkpoints/league"

    rl_train_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=default_ckpt_dir,
        help=f"directory to persist model weights, checkpoints, and league profiles (default: {default_ckpt_dir})",
    )
    rl_train_parser.add_argument(
        "--run-name",
        type=str,
        default="ppo_alphastar_v1",
        help="custom experiment run name for WandB tracking and grouping (default: ppo_alphastar_v1)",
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
        default=default_ckpt_dir,
        help=f"directory containing league checkpoints (default: {default_ckpt_dir})",
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
        default=default_ckpt_dir,
        help=f"directory containing generation checkpoint history (default: {default_ckpt_dir})",
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

    return parser


def _collect_profile(args: argparse.Namespace) -> int:
    client = MetaTftClient(
        minimum_request_interval_seconds=getattr(args, "request_interval", 1.5),
        retry_count=5,
    )
    writer = PlayerCsvWriter(args.output)
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
    writer = PlayerCsvWriter(args.output)
    players = client.fetch_leaderboard_players(count=args.players, offset=args.leaderboard_offset)
    seen_game_ids = _existing_game_ids(args.output)
    blacklisted_players = writer.load_player_blacklist()
    allowed_queues = getattr(args, "allowed_queues", (1100,))

    total_written = 0
    total_skipped = 0

    with tqdm(players, desc="collecting players", unit="player") as player_progress:
        for player in player_progress:
            player_progress.set_postfix({
                "player": player.riot_id,
                "new": total_written,
                "cached": total_skipped,
            })
            if player.riot_id in blacklisted_players:
                tqdm.write(f"skipping blacklisted player {player.riot_id}")
                continue
            if args.max_games is not None and (total_written + total_skipped) >= args.max_games:
                break

            try:
                profile = client.fetch_profile(
                    region=player.region,
                    game_name=player.game_name,
                    tag_line=player.tag_line,
                    tft_set=args.tft_set,
                )
                candidates = client.tracked_timeline_candidates(
                    profile,
                    tft_set=args.tft_set,
                    allowed_queue_ids=allowed_queues,
                )
            except MetaTftRequestError as error:
                tqdm.write(f"skipped player {player.riot_id}: {error}", file=sys.stderr)
                continue

            remaining_budget = (
                (args.max_games - (total_written + total_skipped))
                if args.max_games is not None
                else None
            )
            written, skipped = _download_player_games(
                client=client,
                writer=writer,
                player=player,
                candidates=candidates,
                tft_set=args.tft_set,
                games_per_player=args.games_per_player,
                seen_game_ids=seen_game_ids,
                remaining_games_budget=remaining_budget,
                blacklisted_players=blacklisted_players,
            )
            total_written += written
            total_skipped += skipped
            player_progress.set_postfix({
                "player": player.riot_id,
                "new": total_written,
                "cached": total_skipped,
            })

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

    return written, skipped


def _existing_game_ids(root: Path) -> set[str]:
    import csv as _csv

    seen: set[str] = set()
    players_dir = root / "players"
    if players_dir.exists():
        for path in players_dir.glob("*.csv"):
            try:
                with path.open(newline="", encoding="utf-8") as file:
                    reader = _csv.DictReader(file)
                    for row in reader:
                        match_id = row.get("match_id")
                        if match_id and match_id.strip():
                            seen.add(match_id.strip())
            except Exception:
                continue

    games_dir = root / "games"
    if games_dir.exists():
        seen.update(path.stem for path in games_dir.glob("*.csv"))

    seen.update(PlayerCsvWriter(root).load_blacklist())
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

    print("\n" + "=" * 75)
    print(" [TFT RL TRAINING] PPO & AlphaStar League Pipeline (|A|=111, Obs=704D)")
    print(f"  Target Set: Set 18 | Device: {device_name.upper()}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU Hardware: {gpu_name} ({vram_total:.1f} GB VRAM)")
    print(f"  Generations: {args.generations} | Rollout Steps: {args.rollout_steps} | Batch Size: {args.batch_size}")
    print(f"  Trunk Checkpoint: {args.trunk_checkpoint}")
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
        checkpoint_dir=args.checkpoint_dir,
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

    league = LeagueManager(checkpoint_dir=args.checkpoint_dir)
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

    print("\n" + "=" * 80)
    print(" [TFT STRATEGY VISUALIZER] AlphaStar League Progression & Strategy Space")
    print(f"  Checkpoint Dir: {args.checkpoint_dir} | Output Dir: {args.output_dir}")
    print(f"  Format: {args.format.upper()} | FPS: {args.fps} | Stride: {args.stride}")
    print("=" * 80 + "\n")

    formats = ["gif", "png", "html"] if args.format == "all" else [args.format]
    results = generate_alphastar_progression_plot(
        checkpoint_dir=args.checkpoint_dir,
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
                run_name=getattr(args, "run_name", "ppo_alphastar_v3"),
                enabled=True,
            )
            gen_count = len(list(Path(args.checkpoint_dir).glob("gen_*"))) or 1
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
        dataset = TFTPretrainDataset(
            data=args.data_dir,
            vocab=vocab,
            item_vocab=item_vocab,
            trait_vocab=trait_vocab,
            max_samples=args.max_samples,
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
        dataset = TransitionTrajectoryDataset(
            data_dir=args.data_dir,
            vocab=vocab,
            item_vocab=item_vocab,
            trait_vocab=trait_vocab,
            max_samples=args.max_samples,
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



