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
from torch.utils.data import DataLoader, random_split, TensorDataset

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
        use_wandb: bool = True,
        wandb_project: str = "tft-board-evaluator",
        wandb_run_name: str | None = None,
        wandb_entity: str | None = None,
        num_threads: int = 4,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cpu" and num_threads:
            torch.set_num_threads(num_threads)

        if model is not None:
            self.model = model
        else:
            trunk = MultiModalFusionTrunk(
                num_champs=500,
                num_items=300,
                num_traits=60,
                fused_dim=384,
            )
            ckpt_path = Path(trunk_checkpoint or "models/trunk/trunk_best.pt")
            if not ckpt_path.exists():
                ckpt_path = Path("D:/tft-winner-data/set18/models/trunk/trunk_best.pt")

            if ckpt_path.exists():
                s_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                s_dict = s_data.get("trunk_state_dict", s_data.get("model_state_dict", s_data))
                clean_dict = {k.replace("trunk.", ""): v for k, v in s_dict.items() if "head" not in k}
                trunk.load_state_dict(clean_dict, strict=False)

            self.model = BoardQualityNet(trunk=trunk, freeze_trunk=True)

        self.model.to(self.device)
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2 and not any(nd in name for nd in ["norm", "bias", "embedding"]):
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(param_groups, lr=lr)

        # WandB Experiment Logger
        self.use_wandb = use_wandb
        self.wandb = None
        if self.use_wandb:
            try:
                import wandb
                self.wandb = wandb
                wandb.init(
                    project=wandb_project,
                    name=wandb_run_name or f"board_quality_v1_{int(time.time())}",
                    entity=wandb_entity,
                    config={
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "device": str(self.device),
                        "model": "BoardQualityNet",
                        "trunk": str(trunk_checkpoint),
                    },
                )
            except Exception as e:
                print(f"[!] Warning: Could not initialize WandB: {e}")
                self.use_wandb = False

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

    @torch.no_grad()
    def precompute_dataset_embeddings(
        self,
        dataset: Any,
        batch_size: int = 256,
    ) -> TensorDataset:
        """Precompute fused state vectors s_t using the frozen trunk once."""
        self.model.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_fused = []
        all_place = []
        all_top4 = []

        total_batches = len(loader)
        for i, batch in enumerate(loader):
            champ_ids = batch["board_champ_ids"].to(self.device)
            star_levels = batch["board_star_levels"].to(self.device)
            item_ids = batch["board_item_ids"].to(self.device)
            traits = batch["board_traits"].to(self.device)
            scalars = batch["state_scalars"].to(self.device)

            fused = self.model.trunk(
                board_champ_ids=champ_ids,
                board_star_levels=star_levels,
                board_item_ids=item_ids,
                board_traits=traits,
                state_scalars=scalars,
            )
            all_fused.append(fused.cpu())
            all_place.append(batch["target_placement"].cpu())
            all_top4.append(batch["target_top4"].cpu())
            if (i + 1) % 25 == 0 or (i + 1) == total_batches:
                print(f"    --> Precomputed [{i + 1}/{total_batches}] batches")

        fused_t = torch.cat(all_fused, dim=0)
        place_t = torch.cat(all_place, dim=0)
        top4_t = torch.cat(all_top4, dim=0)
        return TensorDataset(fused_t, place_t, top4_t)

    def train_epoch_fused(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        correct_top4 = 0
        total_samples = 0

        for fused_state, y_place, y_top4 in loader:
            fused_state = fused_state.to(self.device)
            y_place = y_place.to(self.device)
            y_top4 = y_top4.to(self.device)

            pred_place, logits_top4 = self.model.forward_fused(fused_state)

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
    def evaluate_fused(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        correct_top4 = 0
        total_samples = 0

        for fused_state, y_place, y_top4 in loader:
            fused_state = fused_state.to(self.device)
            y_place = y_place.to(self.device)
            y_top4 = y_top4.to(self.device)

            pred_place, logits_top4 = self.model.forward_fused(fused_state)

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
        epochs: int = 20,
        batch_size: int = 128,
        val_split: float = 0.2,
        output_dir: str | Path = "models/board_evaluator",
        precompute_embeddings: bool = True,
    ) -> dict[str, Any]:
        val_size = max(1, int(len(dataset) * val_split))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        use_precompute = precompute_embeddings and self.model.freeze_trunk
        if use_precompute:
            print(f" [*] Precomputing fused trunk representations ({train_size} train, {val_size} val)...")
            t_pre = time.perf_counter()
            train_fused_ds = self.precompute_dataset_embeddings(train_ds, batch_size=256)
            val_fused_ds = self.precompute_dataset_embeddings(val_ds, batch_size=256)
            print(f" [*] Trunk embeddings precomputed in {time.perf_counter() - t_pre:.1f}s.")

            train_loader = DataLoader(train_fused_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_fused_ds, batch_size=batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        best_val_mae = float("inf")
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        best_ckpt = out_path / "board_quality_best.pt"

        print(f" [+] Training BoardQualityNet ({train_size} train, {val_size} val) on {self.device}")

        for ep in range(1, epochs + 1):
            t0 = time.perf_counter()
            if use_precompute:
                train_metrics = self.train_epoch_fused(train_loader)
                val_metrics = self.evaluate_fused(val_loader)
            else:
                train_metrics = self.train_epoch(train_loader)
                val_metrics = self.evaluate(val_loader)
            scheduler.step()
            dt = time.perf_counter() - t0

            print(
                f"  Epoch [{ep:02d}/{epochs:02d}] ({dt:.2f}s) | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train MAE: {train_metrics['placement_mae']:.2f} | "
                f"Val MAE: {val_metrics['val_placement_mae']:.2f} | "
                f"Val Top-4 Acc: {val_metrics['val_top4_accuracy']*100:.1f}%"
            )

            if self.use_wandb and self.wandb is not None and self.wandb.run is not None:
                self.wandb.log(
                    {
                        "train/loss": train_metrics["loss"],
                        "train/placement_mae": train_metrics["placement_mae"],
                        "train/top4_accuracy": train_metrics["top4_accuracy"],
                        "val/loss": val_metrics["val_loss"],
                        "val/placement_mae": val_metrics["val_placement_mae"],
                        "val/top4_accuracy": val_metrics["val_top4_accuracy"],
                        "lr": scheduler.get_last_lr()[0],
                        "epoch": ep,
                    },
                    step=ep,
                )

            if val_metrics["val_placement_mae"] < best_val_mae:
                best_val_mae = val_metrics["val_placement_mae"]
                self.model.save_checkpoint(best_ckpt)
                print(f"   --> Saved new best model to {best_ckpt} (MAE: {best_val_mae:.2f})")

        if self.use_wandb and self.wandb is not None and self.wandb.run is not None:
            self.wandb.finish()

        return {"best_val_mae": best_val_mae, "checkpoint": str(best_ckpt)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Board Quality & Placement Predictor Network")
    parser.add_argument("--data-dir", type=str, default="D:/tft-winner-data/set18/players")
    parser.add_argument("--files", type=int, default=None, help="Maximum number of player CSV files to load")
    parser.add_argument("--trunk-checkpoint", type=str, default="models/trunk/trunk_best.pt")
    parser.add_argument("--output-dir", type=str, default="models/board_evaluator")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--wandb-project", type=str, default="tft-board-evaluator")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--entity", type=str, default="javier-jimenez99")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print(" [TFT BOARD EVALUATOR] Value Oracle & Placement Predictor Training")
    print(f"  Device: {args.device} (CPU threads: {args.num_threads if args.device == 'cpu' else 'N/A'})")
    print(f"  Data: {args.data_dir} (files limit: {args.files}) | Output: {args.output_dir}")
    print("=" * 70)

    dataset = BoardPlacementDataset(data_dir=args.data_dir, max_samples=args.max_samples, max_files=args.files)
    if len(dataset) == 0:
        print("[!] No valid samples loaded from data directory.", file=sys.stderr)
        return 1

    trainer = BoardQualityTrainer(
        trunk_checkpoint=args.trunk_checkpoint,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        num_threads=args.num_threads,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.run_name,
        wandb_entity=args.entity,
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
