"""Weights & Biases Hyperparameter Sweep for TFT Multi-Modal Representation Learning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import wandb

from .dataset import TFTPretrainDataset
from .trainer import TrunkPreTrainer


def get_default_sweep_config(project: str = "tft-embeddings") -> dict[str, Any]:
    """Default Bayesian optimization sweep specification targeting minimal validation loss and maximal retrieval accuracy."""
    return {
        "method": "bayes",
        "metric": {
            "name": "compare/val_total_loss",
            "goal": "minimize",
        },
        "parameters": {
            "lr": {
                "distribution": "log_uniform_values",
                "min": 1e-4,
                "max": 3e-3,
            },
            "weight_decay": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 1e-2,
            },
            "batch_size": {
                "values": [64, 128, 256],
            },
            "dropout": {
                "values": [0.05, 0.1, 0.2, 0.25],
            },
            "num_layers": {
                "values": [2, 3],
            },
            "value_weight": {
                "values": [0.5, 1.0, 1.5],
            },
            "micro_weight": {
                "values": [0.3, 0.5, 0.8],
            },
            "contrast_weight": {
                "values": [0.05, 0.15, 0.30],
            },
            "temperature": {
                "values": [0.05, 0.07, 0.10],
            },
        },
    }


def run_sweep_trial(
    dataset: TFTPretrainDataset,
    output_dir: str | Path = "models/sweep",
    device: str = "cuda",
    epochs: int = 5,
) -> None:
    """Execute a single sweep trial with parameters provided dynamically by WandB agent."""
    run = wandb.init()
    if run is None:
        return

    config = wandb.config

    lr = float(config.get("lr", 1e-3))
    weight_decay = float(config.get("weight_decay", 1e-4))
    batch_size = int(config.get("batch_size", 128))
    dropout = float(config.get("dropout", 0.1))
    num_layers = int(config.get("num_layers", 2))
    value_weight = float(config.get("value_weight", 1.0))
    micro_weight = float(config.get("micro_weight", 0.5))
    contrast_weight = float(config.get("contrast_weight", 0.15))
    temperature = float(config.get("temperature", 0.07))

    trial_name = run.name or f"trial_{run.id}"
    trial_output_dir = Path(output_dir) / trial_name

    trainer = TrunkPreTrainer(
        vocab=dataset.vocab,
        item_vocab=dataset.item_vocab,
        trait_vocab=dataset.trait_vocab,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        value_weight=value_weight,
        micro_weight=micro_weight,
        contrast_weight=contrast_weight,
        temperature=temperature,
        use_wandb=True,
        device=device,
    )
    # Reuse the active wandb run
    trainer._wandb_run = run

    summary = trainer.fit(
        dataset=dataset,
        epochs=epochs,
        batch_size=batch_size,
        val_split=0.15,
        output_dir=trial_output_dir,
        verbose=True,
    )

    print(
        f"\n[+] Finished Sweep Trial {trial_name} | Best Val Loss: {summary.get('best_val_loss', 0.0):.4f} "
        f"| Best Val Top-4 Acc: {summary.get('best_val_top4_acc', 0.0):.1%} "
        f"| Best Val Combat Acc: {summary.get('best_val_combat_acc', 0.0):.1%}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="WandB Hyperparameter Sweep for TFT Multi-Modal Trunk Pre-training")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing player snapshot CSVs")
    parser.add_argument("--sweep-id", type=str, default=None, help="Existing WandB Sweep ID to join (optional)")
    parser.add_argument("--project", type=str, default="tft-embeddings", help="WandB project name")
    parser.add_argument("--entity", type=str, default=None, help="WandB entity/user (optional)")
    parser.add_argument("--count", type=int, default=15, help="Number of sweep trials to execute")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per trial")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional dataset sub-sampling for fast sweep search")
    parser.add_argument("--output-dir", type=str, default="models/sweep", help="Directory to save trial checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")

    args = parser.parse_args()

    print("=" * 75)
    print(" [TFT HYPERPARAMETER SWEEP] WandB Bayesian Architecture & Loss Search")
    print("=" * 75)
    print(f" [+] Preloading dataset from: {args.data_dir}")

    dataset = TFTPretrainDataset(data_dir=args.data_dir, max_samples=args.max_samples)
    print(
        f" [+] Preloaded {len(dataset)} snapshot pairs into memory. "
        f"Vocabs: {len(dataset.vocab)} champs, {len(dataset.item_vocab)} items, {len(dataset.trait_vocab)} traits"
    )

    sweep_id = args.sweep_id
    if not sweep_id:
        sweep_config = get_default_sweep_config(project=args.project)
        sweep_id = wandb.sweep(sweep_config, project=args.project, entity=args.entity)
        print(f"\n[+] Created New WandB Sweep ID: {sweep_id}")
        print(f" [+] View Sweep in WandB: https://wandb.ai/{args.entity or '_'}/{args.project}/sweeps/{sweep_id}\n")

    def trial_fn():
        run_sweep_trial(
            dataset=dataset,
            output_dir=args.output_dir,
            device=args.device,
            epochs=args.epochs,
        )

    print(f"[+] Launching WandB Agent for {args.count} trials on {args.device}...")
    wandb.agent(sweep_id, function=trial_fn, count=args.count, project=args.project, entity=args.entity)


if __name__ == "__main__":
    main()
