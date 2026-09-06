"""Standalone CLI entry point for Pre-training the Multi-Modal Fusion Trunk."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .dataset import TFTPretrainDataset, create_synthetic_trajectory_dataset
from .trainer import TrunkPreTrainer
from .vocab import ChampionVocabulary, ItemVocabulary, TraitVocabulary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tft-embeddings-train",
        description="Pre-train the Multi-Modal Fusion Trunk, Champ2Vec, and Trait Synergy Encoders",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(r"D:\tft-winner-data\set18\players"),
        help="Path to directory containing player CSV files or single CSV",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("models/trunk"),
        help="Output directory for saved models and vocabulary",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of pretraining epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW (default: 1e-4)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability (default: 0.1)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of Board Transformer layers (default: 2)",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=32,
        help="Champ2Vec champion embedding dimension (default: 32)",
    )
    parser.add_argument(
        "--board-dim",
        type=int,
        default=256,
        help="Board CNN output dimension (default: 256)",
    )
    parser.add_argument(
        "--fused-dim",
        type=int,
        default=384,
        help="Fusion trunk dimension (default: 384)",
    )
    parser.add_argument(
        "--val-weight",
        type=float,
        default=1.0,
        help="Weight for Macro Top-4 CE loss (default: 1.0)",
    )
    parser.add_argument(
        "--micro-weight",
        type=float,
        default=0.5,
        help="Weight for Micro combat round win probability BCE loss (default: 0.5)",
    )
    parser.add_argument(
        "--contrast-weight",
        type=float,
        default=0.15,
        help="Weight for Flow InfoNCE loss (default: 0.15)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="InfoNCE temperature tau (default: 0.07)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum snapshot pairs to load",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device (cuda or cpu)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic dataset for quick testing",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Batch frequency for real-time WandB metric logging (default: 10)",
    )
    # WandB options
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
        help="WandB run display name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB team or entity name",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default="pretrain-phase1",
        help="WandB group name",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("\n" + "=" * 75)
    print(" [TFT MULTI-MODAL TRUNK] Phase 1: Pre-training Embeddings, Synergies & Trunk")
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
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
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


if __name__ == "__main__":
    sys.exit(main())
