"""Training script for the State Transition Predictor in frozen 320D Latent Space.

Learns offline behavioral cloning for TFT round transitions with Dual-Objective Huber + Cosine Loss.
Logs training metrics and tables to Weights & Biases and serializes model checkpoints.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

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

# Configure UTF-8 stdout
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the TFT State Transition Predictor in frozen 320D Latent Space."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="D:/tft-winner-data/set18/players",
        help="Directory containing match CSV files",
    )
    parser.add_argument(
        "--trunk-checkpoint",
        type=str,
        default="models/trunk/trunk_best.pt",
        help="Path to pre-trained frozen MultiModalFusionTrunk checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="models/transition_predictor",
        help="Output directory for saved models and summaries",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs (default: 15)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for training and latent extraction (default: 512)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="AdamW learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="Hidden dimension of Residual MLP blocks (default: 512)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of Residual MLP layers (default: 3)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate (default: 0.1)",
    )
    parser.add_argument(
        "--lambda-cosine",
        type=float,
        default=0.5,
        help="Loss weight for Cosine Embedding directional loss (default: 0.5)",
    )
    parser.add_argument(
        "--huber-beta",
        type=float,
        default=1.0,
        help="Huber loss threshold beta (default: 1.0)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Fraction of matches reserved for validation (default: 0.15)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of transition pairs to load",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Train on synthetic snapshot transitions for quick validation",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
        help="Batch frequency for real-time WandB step logging (default: 20)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="tft-embeddings",
        help="Weights & Biases project name (default: tft-embeddings)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom run name for WandB tracking",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB username or team entity",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default="transition-predictor",
        help="WandB experiment group",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB cloud logging",
    )

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 80)
    print(" [TFT TRANSITION PREDICTOR] Offline Behavioral Cloning in 320D Latent Space")
    print(f"  Device:          {device}")
    print(f"  Data Source:     {args.data_dir}")
    print(f"  Trunk Checkpoint:{args.trunk_checkpoint}")
    print(f"  Output Dir:      {args.output_dir}")
    print(f"  Epochs:          {args.epochs} | Batch Size: {args.batch_size}")
    print(f"  Lambda Cosine:   {args.lambda_cosine} | Huber Beta: {args.huber_beta} | LR: {args.lr}")
    print("=" * 80)

    # 1. Load Frozen MultiModalFusionTrunk
    trunk_path = Path(args.trunk_checkpoint)
    if trunk_path.exists():
        print(f" [+] Loading frozen pre-trained trunk from: {trunk_path.resolve()}")
        trunk = MultiModalFusionTrunk.load_trunk(trunk_path, map_location=device)
    else:
        print(f" [!] Trunk checkpoint not found at '{trunk_path}'. Initializing blank trunk for demonstration.")
        trunk = MultiModalFusionTrunk(fused_dim=320)
    trunk.freeze()
    trunk.eval()

    vocab = ChampionVocabulary()
    item_vocab = ItemVocabulary()
    trait_vocab = TraitVocabulary()

    # 2. Load Dataset
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
        print(f" [+] Ingesting transition trajectory pairs from: {args.data_dir}")
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

    # 3. Instantiate StateTransitionPredictor
    latent_dim = trunk.fused_dim
    predictor = StateTransitionPredictor(
        input_dim=latent_dim,
        hidden_dim=args.hidden_dim,
        output_dim=latent_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_residual_delta=True,
    )

    # 4. Instantiate Trainer
    trainer = TransitionPredictorTrainer(
        predictor=predictor,
        trunk=trunk,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_cosine=args.lambda_cosine,
        huber_beta=args.huber_beta,
        log_interval=args.log_interval,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.run_name or f"transition-predictor-{int(time.time())}",
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        device=device,
    )

    # 5. Fit model
    summary = trainer.fit(
        dataset=dataset,
        val_split=args.val_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 80)
    print(f" [+] Training Pipeline Succeeded in {summary['total_time_sec']}s!")
    print(f"     Best Val Cosine Similarity: {summary['best_val_cosine_similarity']:.4f}")
    print(f"     Best Val Loss:              {summary['best_val_loss']:.4f}")
    print(f"     Artifacts Saved To:         {Path(args.output_dir).resolve()}")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
