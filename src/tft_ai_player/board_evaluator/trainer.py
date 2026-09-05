"""Training pipeline for the Board Quality and Tournament Placement Predictor."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from tft_ai_player.embeddings.model import MultiModalFusionTrunk
from .dataset import BoardPlacementDataset
from .model import BoardQualityNet


class BoardQualityTrainer:
    """Trains BoardQualityNet to predict tournament finish placement and Top-4 rate."""

    def __init__(
        self,
        trunk_checkpoint: str | Path | None = None,
        model: BoardQualityNet | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)

        if model is not None:
            self.model = model
        else:
            trunk = MultiModalFusionTrunk(
                num_champs=500,
                num_items=300,
                num_traits=60,
                fused_dim=384,
            )
            ckpt_path = Path(trunk_checkpoint or "D:/tft-winner-data/set18/models/trunk/trunk_best.pt")
            if ckpt_path.exists():
                s_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                s_dict = s_data.get("trunk_state_dict", s_data.get("model_state_dict", s_data))
                clean_dict = {k.replace("trunk.", ""): v for k, v in s_dict.items() if "head" not in k}
                trunk.load_state_dict(clean_dict, strict=False)

            self.model = BoardQualityNet(trunk=trunk, freeze_trunk=True)

        self.model.to(self.device)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        correct_top4 = 0
        total_samples = 0

        for batch in loader:
            champ_ids = batch["board_champ_ids"].to(self.device)
            star_levels = batch["board_star_levels"].to(self.device)
            item_ids = batch["board_item_ids"].to(self.device)
            traits = batch["board_traits"].to(self.device)
            scalars = batch["state_scalars"].to(self.device)
            y_place = batch["target_placement"].to(self.device)
            y_top4 = batch["target_top4"].to(self.device)

            pred_place, logits_top4 = self.model(
                board_champ_ids=champ_ids,
                board_star_levels=star_levels,
                board_item_ids=item_ids,
                board_traits=traits,
                state_scalars=scalars,
            )

            loss_place = F.huber_loss(pred_place, y_place, delta=1.0)
            loss_top4 = F.cross_entropy(logits_top4, y_top4)
            loss = loss_place + 0.5 * loss_top4

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            bs = y_place.shape[0]
            total_loss += float(loss.item()) * bs
            total_mae += float((pred_place - y_place).abs().sum().item())
            preds_top4 = torch.argmax(logits_top4, dim=-1)
            correct_top4 += int((preds_top4 == y_top4).sum().item())
            total_samples += bs

        n = max(1, total_samples)
        return {
            "loss": total_loss / n,
            "placement_mae": total_mae / n,
            "top4_accuracy": correct_top4 / n,
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        correct_top4 = 0
        total_samples = 0

        for batch in loader:
            champ_ids = batch["board_champ_ids"].to(self.device)
            star_levels = batch["board_star_levels"].to(self.device)
            item_ids = batch["board_item_ids"].to(self.device)
            traits = batch["board_traits"].to(self.device)
            scalars = batch["state_scalars"].to(self.device)
            y_place = batch["target_placement"].to(self.device)
            y_top4 = batch["target_top4"].to(self.device)

            pred_place, logits_top4 = self.model(
                board_champ_ids=champ_ids,
                board_star_levels=star_levels,
                board_item_ids=item_ids,
                board_traits=traits,
                state_scalars=scalars,
            )

            loss_place = F.huber_loss(pred_place, y_place, delta=1.0)
            loss_top4 = F.cross_entropy(logits_top4, y_top4)
            loss = loss_place + 0.5 * loss_top4

            bs = y_place.shape[0]
            total_loss += float(loss.item()) * bs
            total_mae += float((pred_place - y_place).abs().sum().item())
            preds_top4 = torch.argmax(logits_top4, dim=-1)
            correct_top4 += int((preds_top4 == y_top4).sum().item())
            total_samples += bs

        n = max(1, total_samples)
        return {
            "val_loss": total_loss / n,
            "val_placement_mae": total_mae / n,
            "val_top4_accuracy": correct_top4 / n,
        }

    def fit(
        self,
        dataset: BoardPlacementDataset,
        epochs: int = 5,
        batch_size: int = 128,
        val_split: float = 0.2,
        output_dir: str | Path = "models/board_evaluator",
    ) -> dict[str, Any]:
        val_size = max(1, int(len(dataset) * val_split))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        best_val_mae = float("inf")
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        best_ckpt = out_path / "board_quality_best.pt"

        print(f" [+] Training BoardQualityNet ({train_size} train, {val_size} val) on {self.device}")

        for ep in range(1, epochs + 1):
            t0 = time.perf_counter()
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            dt = time.perf_counter() - t0

            print(
                f"  Epoch [{ep:02d}/{epochs:02d}] ({dt:.1f}s) | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train MAE: {train_metrics['placement_mae']:.2f} | "
                f"Val MAE: {val_metrics['val_placement_mae']:.2f} | "
                f"Val Top-4 Acc: {val_metrics['val_top4_accuracy']*100:.1f}%"
            )

            if val_metrics["val_placement_mae"] < best_val_mae:
                best_val_mae = val_metrics["val_placement_mae"]
                self.model.save_checkpoint(best_ckpt)
                print(f"   --> Saved new best model to {best_ckpt} (MAE: {best_val_mae:.2f})")

        return {"best_val_mae": best_val_mae, "checkpoint": str(best_ckpt)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Board Quality & Placement Predictor Network")
    parser.add_argument("--data-dir", type=str, default="D:/tft-winner-data/set18/players")
    parser.add_argument("--trunk-checkpoint", type=str, default="D:/tft-winner-data/set18/models/trunk/trunk_best.pt")
    parser.add_argument("--output-dir", type=str, default="models/board_evaluator")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print(" [TFT BOARD EVALUATOR] Value Oracle & Placement Predictor Training")
    print("=" * 70)

    dataset = BoardPlacementDataset(data_dir=args.data_dir, max_samples=args.max_samples)
    if len(dataset) == 0:
        print("[!] No valid samples loaded from data directory.", file=sys.stderr)
        return 1

    trainer = BoardQualityTrainer(
        trunk_checkpoint=args.trunk_checkpoint,
        device=args.device,
    )
    trainer.fit(
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
