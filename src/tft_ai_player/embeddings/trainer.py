"""Training, evaluation, match-grouped splitting, real-time WandB logging, and serialization engine."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from .dataset import SnapshotPairCollate, TFTPretrainDataset
from .losses import MultiTaskTrunkLoss
from .model import MultiModalFusionTrunk, TrunkPretrainModel
from .vocab import ChampionVocabulary, ItemVocabulary, TraitVocabulary


class TrunkPreTrainer:
    """End-to-End Trainer for Phase 1: Pre-training the Multi-Modal Fusion Trunk with the Tri-Objective 'Holy Trinity' Loss."""

    def __init__(
        self,
        model: TrunkPretrainModel | None = None,
        vocab: ChampionVocabulary | None = None,
        item_vocab: ItemVocabulary | None = None,
        trait_vocab: TraitVocabulary | None = None,
        champ_embed_dim: int = 32,
        board_feat_dim: int = 256,
        state_feat_dim: int = 64,
        fused_dim: int = 320,
        proj_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        value_weight: float = 1.0,
        micro_weight: float = 0.5,
        contrast_weight: float = 0.15,
        temperature: float = 0.07,
        log_interval: int = 10,
        use_wandb: bool = False,
        wandb_project: str = "tft-embeddings",
        wandb_run_name: str | None = None,
        wandb_entity: str | None = None,
        wandb_group: str | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.vocab = vocab or ChampionVocabulary()
        self.item_vocab = item_vocab or ItemVocabulary()
        self.trait_vocab = trait_vocab or TraitVocabulary()

        self.log_interval = max(1, log_interval)
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.wandb_entity = wandb_entity
        self.wandb_group = wandb_group
        self._wandb_run = None
        self.global_step = 0

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if model is not None:
            self.model = model.to(self.device)
        else:
            num_champs = max(len(self.vocab) + 20, 150)
            num_items = max(len(self.item_vocab) + 20, 200)
            num_traits = max(len(self.trait_vocab), 36)
            self.model = TrunkPretrainModel(
                num_champs=num_champs,
                num_items=num_items,
                num_traits=num_traits,
                champ_embed_dim=champ_embed_dim,
                board_feat_dim=board_feat_dim,
                state_feat_dim=state_feat_dim,
                fused_dim=fused_dim,
                proj_dim=proj_dim,
                num_layers=num_layers,
                dropout=dropout,
                num_classes=2,
            ).to(self.device)

        self.criterion = MultiTaskTrunkLoss(
            value_weight=value_weight,
            micro_weight=micro_weight,
            contrast_weight=contrast_weight,
            temperature=temperature,
        ).to(self.device)

        self.lr = lr
        self.weight_decay = weight_decay
        self.value_weight = value_weight
        self.micro_weight = micro_weight
        self.contrast_weight = contrast_weight
        self.temperature = temperature
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.history: list[dict[str, Any]] = []

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int = 1,
        total_epochs: int = 1,
        verbose: bool = True,
    ) -> dict[str, float]:
        """Execute a single training epoch across all snapshot batches with real-time logging."""
        self.model.train()
        accum_metrics: dict[str, float] = {}
        total_batches = len(dataloader)
        current_lr = self.optimizer.param_groups[0]["lr"]

        iterator = dataloader
        pbar = None
        if verbose and tqdm is not None:
            pbar = tqdm(
                dataloader,
                desc=f"Epoch [{epoch:02d}/{total_epochs:02d}]",
                unit="batch",
                leave=False,
            )
            iterator = pbar

        for batch_idx, batch in enumerate(iterator, start=1):
            anchor = {k: v.to(self.device) for k, v in batch["anchor"].items()}
            positive = {k: v.to(self.device) for k, v in batch["positive"].items()}

            # 1. Forward Pass (contrastive projection uses exclusively board_feat)
            _, val_logits, combat_logits, proj_a = self.model.forward_dict(anchor)
            _, _, _, proj_p = self.model.forward_dict(positive)

            # 2. Tri-Objective Loss with PVP Masking & Balanced Gradients
            loss, metrics = self.criterion(
                value_logits=val_logits,
                value_targets=anchor["value_targets"],
                anchor_proj=proj_a,
                positive_proj=proj_p,
                combat_win_logits=combat_logits,
                combat_win_targets=anchor["combat_targets"],
                is_pvp_mask=anchor.get("is_pvp_mask"),
            )

            # 3. Backpropagation & Optimization
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            self.global_step += 1

            for k, v in metrics.items():
                accum_metrics[k] = accum_metrics.get(k, 0.0) + v

            # Real-time WandB batch logging
            if self.use_wandb and (batch_idx % self.log_interval == 0 or batch_idx == total_batches):
                wandb_payload = {
                    "train/step_loss": loss.item(),
                    "train/step_value_ce_loss": metrics.get("value_ce_loss", 0.0),
                    "train/step_value_acc": metrics.get("value_acc", 0.0),
                    "train/step_micro_combat_loss": metrics.get("micro_combat_loss", 0.0),
                    "train/step_combat_acc": metrics.get("combat_acc", 0.0),
                    "train/step_info_nce_loss": metrics.get("info_nce_loss", 0.0),
                    "train/step_pos_sim": metrics.get("pos_sim", 0.0),
                    "train/step_neg_sim": metrics.get("neg_sim", 0.0),
                    "train/lr": current_lr,
                    "train/epoch_progress": round(epoch - 1 + (batch_idx / max(1, total_batches)), 4),
                    "global_step": self.global_step,
                }
                self._log_wandb_raw(wandb_payload, step=self.global_step)

            # Live terminal postfix update
            if pbar is not None:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "top4_acc": f"{metrics.get('value_acc', 0.0)*100:.1f}%",
                    "c_acc": f"{metrics.get('combat_acc', 0.0)*100:.1f}%",
                    "nce": f"{metrics.get('info_nce_loss', 0.0):.3f}",
                    "pos_sim": f"{metrics.get('pos_sim', 0.0):.3f}",
                })
            elif verbose and (batch_idx % (self.log_interval * 5) == 0 or batch_idx == total_batches):
                print(
                    f"  [Epoch {epoch:02d} | Batch {batch_idx:04d}/{total_batches:04d}] "
                    f"Loss: {loss.item():.4f} (Top4-Acc: {metrics.get('value_acc', 0.0)*100:.1f}%, "
                    f"Combat-Acc: {metrics.get('combat_acc', 0.0)*100:.1f}%, "
                    f"InfoNCE: {metrics.get('info_nce_loss', 0.0):.3f})"
                )

        if pbar is not None:
            pbar.close()

        return {k: v / max(1, total_batches) for k, v in accum_metrics.items()}

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate model on validation dataloader."""
        self.model.eval()
        accum_metrics: dict[str, float] = {}
        total_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                anchor = {k: v.to(self.device) for k, v in batch["anchor"].items()}
                positive = {k: v.to(self.device) for k, v in batch["positive"].items()}

                _, val_logits, combat_logits, proj_a = self.model.forward_dict(anchor)
                _, _, _, proj_p = self.model.forward_dict(positive)

                loss, metrics = self.criterion(
                    value_logits=val_logits,
                    value_targets=anchor["value_targets"],
                    anchor_proj=proj_a,
                    positive_proj=proj_p,
                    combat_win_logits=combat_logits,
                    combat_win_targets=anchor["combat_targets"],
                    is_pvp_mask=anchor.get("is_pvp_mask"),
                )

                total_batches += 1
                for k, v in metrics.items():
                    accum_metrics[k] = accum_metrics.get(k, 0.0) + v

        return {f"val_{k}": v / max(1, total_batches) for k, v in accum_metrics.items()}

    def _init_wandb(self, epochs: int, batch_size: int, train_size: int, val_size: int) -> None:
        """Initialize Weights & Biases logging session."""
        if not self.use_wandb:
            return
        try:
            import wandb
            config = {
                "epochs": epochs,
                "batch_size": batch_size,
                "train_samples": train_size,
                "val_samples": val_size,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "value_weight": self.value_weight,
                "micro_weight": self.micro_weight,
                "contrast_weight": self.contrast_weight,
                "temperature": self.temperature,
                "num_champs": self.model.trunk.num_champs,
                "num_items": self.model.trunk.num_items,
                "num_traits": self.model.trunk.num_traits,
                "champ_embed_dim": self.model.trunk.champ_embed_dim,
                "board_feat_dim": self.model.trunk.board_feat_dim,
                "state_feat_dim": self.model.trunk.state_feat_dim,
                "fused_dim": self.model.trunk.fused_dim,
            }
            run_kwargs: dict[str, Any] = {
                "project": self.wandb_project,
                "config": config,
                "reinit": True,
            }
            if self.wandb_run_name:
                run_kwargs["name"] = self.wandb_run_name
            if self.wandb_entity:
                run_kwargs["entity"] = self.wandb_entity
            if self.wandb_group:
                run_kwargs["group"] = self.wandb_group

            self._wandb_run = wandb.init(**run_kwargs)
            wandb.define_metric("epoch/epoch_num")
            wandb.define_metric("epoch/*", step_metric="epoch/epoch_num")
            wandb.define_metric("compare/*", step_metric="epoch/epoch_num")
            wandb.define_metric("gap/*", step_metric="epoch/epoch_num")

            url = getattr(self._wandb_run, "url", None) or getattr(self._wandb_run, "get_url", lambda: None)()
            print(f"[TrunkPreTrainer] Initialized WandB run: {getattr(self._wandb_run, 'name', 'active')}")
            if url:
                print(f" [+] WandB Dashboard URL: {url}")
        except Exception as e:
            print(f"[!] Warning: Failed to initialize WandB ({e}). Continuing with local logging.")
            self._wandb_run = None

    def _log_wandb_raw(self, payload: dict[str, Any], step: int | None = None) -> None:
        """Log raw dictionary payload directly to WandB."""
        if self._wandb_run is None:
            return
        try:
            import wandb
            if step is not None:
                wandb.log(payload, step=step)
            else:
                wandb.log(payload)
        except Exception:
            pass

    def _log_wandb_epoch(self, train_metrics: dict[str, float], val_metrics: dict[str, float], epoch: int) -> None:
        """Log epoch-aggregated metrics and multi-line overlay comparison plots to WandB."""
        if self._wandb_run is None:
            return
        try:
            import wandb

            # 1. Raw Epoch Summary Metrics
            log_payload: dict[str, Any] = {f"epoch/train_{k}": v for k, v in train_metrics.items()}
            log_payload.update({f"epoch/{k}": v for k, v in val_metrics.items()})
            log_payload["epoch/epoch_num"] = epoch

            # 2. Side-by-Side Scalar Comparison
            log_payload.update({
                "compare/train_total_loss": train_metrics.get("total_loss", 0.0),
                "compare/val_total_loss": val_metrics.get("val_total_loss", 0.0),
                "compare/train_top4_acc": train_metrics.get("value_acc", 0.0),
                "compare/val_top4_acc": val_metrics.get("val_value_acc", 0.0),
                "compare/train_combat_acc": train_metrics.get("combat_acc", 0.0),
                "compare/val_combat_acc": val_metrics.get("val_combat_acc", 0.0),
                "compare/train_combat_bce": train_metrics.get("micro_combat_loss", 0.0),
                "compare/val_combat_bce": val_metrics.get("val_micro_combat_loss", 0.0),
                "compare/train_info_nce": train_metrics.get("info_nce_loss", 0.0),
                "compare/val_info_nce": val_metrics.get("val_info_nce_loss", 0.0),
                "compare/train_contrast_retrieval_acc": train_metrics.get("contrast_retrieval_acc", 0.0),
                "compare/val_contrast_retrieval_acc": val_metrics.get("val_contrast_retrieval_acc", 0.0),
                # Generalization Gaps (Val - Train)
                "gap/generalization_loss_gap": val_metrics.get("val_total_loss", 0.0) - train_metrics.get("total_loss", 0.0),
                "gap/top4_acc_gap": val_metrics.get("val_value_acc", 0.0) - train_metrics.get("value_acc", 0.0),
                "gap/combat_acc_gap": val_metrics.get("val_combat_acc", 0.0) - train_metrics.get("combat_acc", 0.0),
                "gap/retrieval_acc_gap": val_metrics.get("val_contrast_retrieval_acc", 0.0) - train_metrics.get("contrast_retrieval_acc", 0.0),
            })

            # 3. Multi-Line Overlay Charts (Both Train and Val curves on the EXACT SAME plot)
            if hasattr(wandb.plot, "line_series") and len(self.history) > 0:
                epoch_list = [h.get("epoch", i + 1) for i, h in enumerate(self.history)]
                train_loss_list = [h.get("total_loss", 0.0) for h in self.history]
                val_loss_list = [h.get("val_total_loss", h.get("total_loss", 0.0)) for h in self.history]

                train_top4_list = [h.get("value_acc", 0.0) for h in self.history]
                val_top4_list = [h.get("val_value_acc", h.get("value_acc", 0.0)) for h in self.history]

                train_c_acc_list = [h.get("combat_acc", 0.0) for h in self.history]
                val_c_acc_list = [h.get("val_combat_acc", h.get("combat_acc", 0.0)) for h in self.history]

                train_c_bce_list = [h.get("micro_combat_loss", 0.0) for h in self.history]
                val_c_bce_list = [h.get("val_micro_combat_loss", h.get("micro_combat_loss", 0.0)) for h in self.history]

                train_nce_list = [h.get("info_nce_loss", 0.0) for h in self.history]
                val_nce_list = [h.get("val_info_nce_loss", h.get("info_nce_loss", 0.0)) for h in self.history]

                train_ret_list = [h.get("contrast_retrieval_acc", 0.0) for h in self.history]
                val_ret_list = [h.get("val_contrast_retrieval_acc", h.get("contrast_retrieval_acc", 0.0)) for h in self.history]

                pos_sim_list = [h.get("pos_sim", 0.0) for h in self.history]
                neg_sim_list = [h.get("neg_sim", 0.0) for h in self.history]

                log_payload["comparison/Total Multi-Task Loss"] = wandb.plot.line_series(
                    xs=epoch_list,
                    ys=[train_loss_list, val_loss_list],
                    keys=["Train Loss", "Val Loss"],
                    title="Total Multi-Task Loss (Train vs Val)",
                    xname="Epoch",
                )
                log_payload["comparison/Macro Top-4 Accuracy"] = wandb.plot.line_series(
                    xs=epoch_list,
                    ys=[train_top4_list, val_top4_list],
                    keys=["Train Top-4 Acc", "Val Top-4 Acc"],
                    title="Macro Top-4 Accuracy (Train vs Val)",
                    xname="Epoch",
                )
                log_payload["comparison/Micro Combat Win Accuracy"] = wandb.plot.line_series(
                    xs=epoch_list,
                    ys=[train_c_acc_list, val_c_acc_list],
                    keys=["Train Combat Acc", "Val Combat Acc"],
                    title="Micro Combat Win Accuracy (Train vs Val)",
                    xname="Epoch",
                )
                log_payload["comparison/Micro Combat BCE Loss"] = wandb.plot.line_series(
                    xs=epoch_list,
                    ys=[train_c_bce_list, val_c_bce_list],
                    keys=["Train Combat BCE", "Val Combat BCE"],
                    title="Micro Combat BCE Loss (Train vs Val)",
                    xname="Epoch",
                )
                log_payload["comparison/Flow Pivot InfoNCE Loss"] = wandb.plot.line_series(
                    xs=epoch_list,
                    ys=[train_nce_list, val_nce_list],
                    keys=["Train InfoNCE", "Val InfoNCE"],
                    title="Flow Pivot Board InfoNCE Loss (Train vs Val)",
                    xname="Epoch",
                )
                log_payload["comparison/Contrastive Board Retrieval Accuracy"] = wandb.plot.line_series(
                    xs=epoch_list,
                    ys=[train_ret_list, val_ret_list],
                    keys=["Train Retrieval Acc", "Val Retrieval Acc"],
                    title="Contrastive Board Retrieval Accuracy (Train vs Val)",
                    xname="Epoch",
                )
                log_payload["comparison/Contrastive Geometry (Alignment vs Uniformity)"] = wandb.plot.line_series(
                    xs=epoch_list,
                    ys=[pos_sim_list, neg_sim_list],
                    keys=["Positive Cosine Sim (Alignment)", "Negative Cosine Sim (Uniformity)"],
                    title="Contrastive Geometry: Positive vs Negative Cosine Similarity",
                    xname="Epoch",
                )

            wandb.log(log_payload, step=self.global_step)
        except Exception:
            pass

    def _finish_wandb(self, summary_dict: dict[str, Any] | None = None) -> None:
        """Close WandB session and update final summary card."""
        if self._wandb_run is not None:
            try:
                import wandb
                if summary_dict:
                    for k, v in summary_dict.items():
                        if isinstance(v, (int, float, str, bool)):
                            wandb.run.summary[k] = v
                wandb.finish()
            except Exception:
                pass
            self._wandb_run = None

    def fit(
        self,
        dataset: TFTPretrainDataset,
        epochs: int = 5,
        batch_size: int = 64,
        val_split: float = 0.15,
        output_dir: str | Path = "models/trunk",
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Fit model across multiple epochs with match-grouped validation and checkpoint serialization."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Enforce match_id grouped splitting to prevent temporal trajectory leakage
        if val_split > 0.0:
            train_ds, val_ds = dataset.split_by_match_id(val_split=val_split, seed=42)
            val_size = len(val_ds)
        else:
            train_ds = dataset
            val_ds = None
            val_size = 0

        train_size = len(train_ds)

        collate_fn = SnapshotPairCollate()
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=(len(train_ds) > batch_size),
        )

        val_loader = (
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            if val_ds is not None and len(val_ds) > 0
            else None
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(1, epochs), eta_min=1e-5)
        best_val_loss = float("inf")
        best_val_top4_acc = 0.0
        best_val_combat_acc = 0.0
        start_time = time.time()

        if verbose:
            print(f"[TrunkPreTrainer] Match-Grouped Split: {train_size} train pairs, {val_size} val pairs on {self.device}")
            print(f"[TrunkPreTrainer] Batches per epoch: {len(train_loader)} (batch_size={batch_size})")

        # Initialize WandB
        self._init_wandb(epochs=epochs, batch_size=batch_size, train_size=train_size, val_size=val_size)

        try:
            for epoch in range(1, epochs + 1):
                ep_start = time.time()
                train_metrics = self.train_epoch(
                    dataloader=train_loader,
                    epoch=epoch,
                    total_epochs=epochs,
                    verbose=verbose,
                )
                scheduler.step()

                if val_loader is not None:
                    val_metrics = self.evaluate(val_loader)
                else:
                    val_metrics = {}

                ep_duration = time.time() - ep_start
                current_lr = self.optimizer.param_groups[0]["lr"]

                combined_metrics = {
                    "epoch": epoch,
                    "duration_sec": round(ep_duration, 2),
                    "lr": current_lr,
                    **train_metrics,
                    **val_metrics,
                }
                self.history.append(combined_metrics)

                # Log epoch summary & side-by-side comparison block to WandB
                self._log_wandb_epoch(train_metrics, val_metrics, epoch=epoch)

                if verbose:
                    msg = (
                        f"\nEpoch [{epoch:02d}/{epochs:02d}] Summary: "
                        f"Loss: {train_metrics.get('total_loss', 0.0):.4f} "
                        f"(Top4-Acc: {train_metrics.get('value_acc', 0.0):.1%}, "
                        f"Combat-Acc: {train_metrics.get('combat_acc', 0.0):.1%}, "
                        f"Ret-Acc: {train_metrics.get('contrast_retrieval_acc', 0.0):.1%}, "
                        f"InfoNCE: {train_metrics.get('info_nce_loss', 0.0):.4f}, "
                        f"PosSim: {train_metrics.get('pos_sim', 0.0):.3f}, "
                        f"NegSim: {train_metrics.get('neg_sim', 0.0):.3f})"
                    )
                    if val_metrics:
                        msg += (
                            f" | ValLoss: {val_metrics.get('val_total_loss', 0.0):.4f} "
                            f"(ValTop4-Acc: {val_metrics.get('val_value_acc', 0.0):.1%}, "
                            f"ValCombat-Acc: {val_metrics.get('val_combat_acc', 0.0):.1%}, "
                            f"ValRet-Acc: {val_metrics.get('val_contrast_retrieval_acc', 0.0):.1%})"
                        )
                    print(msg + "\n")

                # Track best validation metrics
                current_val_loss = val_metrics.get("val_total_loss", train_metrics.get("total_loss", float("inf")))
                if val_metrics.get("val_value_acc", 0.0) > best_val_top4_acc:
                    best_val_top4_acc = val_metrics.get("val_value_acc", 0.0)
                if val_metrics.get("val_combat_acc", 0.0) > best_val_combat_acc:
                    best_val_combat_acc = val_metrics.get("val_combat_acc", 0.0)

                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    self.save_checkpoint(output_path / "trunk_best.pt", is_best=True)

            total_time = time.time() - start_time
            if verbose:
                print(f"[TrunkPreTrainer] Completed {epochs} epochs in {total_time:.2f}s. Saving artifacts to {output_path}")

            # Save final artifacts
            self.save_checkpoint(output_path / "trunk_final.pt", is_best=False)
            self.vocab.save(output_path / "vocab.json")
            self.item_vocab.save(output_path / "item_vocab.json")
            self.trait_vocab.save(output_path / "trait_vocab.json")

            summary = {
                "total_epochs": epochs,
                "train_pairs": train_size,
                "val_pairs": val_size,
                "total_time_sec": round(total_time, 2),
                "best_val_loss": round(best_val_loss, 4),
                "best_val_top4_acc": round(best_val_top4_acc, 4),
                "best_val_combat_acc": round(best_val_combat_acc, 4),
                "history": self.history,
            }
            with open(output_path / "pretrain_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

            return summary

        finally:
            self._finish_wandb(summary_dict={
                "summary/best_val_loss": best_val_loss,
                "summary/best_val_top4_acc": best_val_top4_acc,
                "summary/best_val_combat_acc": best_val_combat_acc,
                "summary/total_time_sec": time.time() - start_time,
            })

    def save_checkpoint(self, path: str | Path, is_best: bool = False) -> None:
        """Serialize model checkpoint and standalone trunk weights."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "trunk_state_dict": self.model.trunk.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "vocab": self.vocab.to_dict(),
            "item_vocab": self.item_vocab.to_dict(),
            "trait_vocab": self.trait_vocab.to_dict(),
            "config": {
                "num_champs": self.model.trunk.num_champs,
                "num_items": self.model.trunk.num_items,
                "num_traits": self.model.trunk.num_traits,
                "champ_embed_dim": self.model.trunk.champ_embed_dim,
                "board_feat_dim": self.model.trunk.board_feat_dim,
                "state_feat_dim": self.model.trunk.state_feat_dim,
                "fused_dim": self.model.trunk.fused_dim,
            },
        }
        torch.save(checkpoint, path)

        # Standalone frozen trunk export
        trunk_path = path.parent / ("trunk_pretrained.pt" if is_best else "trunk_last.pt")
        self.model.trunk.save_trunk(trunk_path)
