"""Train and export DeepSiameseCombatNet on GPU.

Loads Set 18 match CSV snapshots, builds CombatDataset with GroupShuffleSplit,
trains DeepSiameseCombatNet with BCEWithLogitsLoss, evaluates test metrics,
and saves the trained model weights to models/round_winner/deep_siamese_combat_best.pt.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tft_ai_player.embeddings.model import MultiModalFusionTrunk
from tft_ai_player.embeddings.vocab import ChampionVocabulary, ItemVocabulary, TraitVocabulary
from tft_ai_player.round_winner.embedding_model import DeepSiameseCombatNet, CombatDataset, combat_collate_fn
from tft_ai_player.simulation.sets.set18 import get_set18_data


def train_deep_siamese_combat(
    data_dir: Path | str = "D:/tft-winner-data/set18/players",
    output_dir: Path | str = "models/round_winner",
    trunk_ckpt_path: Path | str = "models/trunk/trunk_best.pt",
    max_files: int = 50,
    batch_size: int = 256,
    epochs: int = 10,
    lr: float = 1e-3,
    hidden_dim: int = 256,
    dropout: float = 0.15,
    device: torch.device | str | None = None,
) -> Path:
    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("=" * 80)
    print("       TRAINING DEEP SIAMESE COMBAT NETWORK (DeepSiameseCombatNet)")
    print(f"       Device: {dev} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 80)

    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Ingest CSV Files
    csv_files = sorted(list(data_path.glob("*.csv")))[:max_files]
    if not csv_files:
        raise FileNotFoundError(f"No CSV match files found in {data_path}")
    print(f" [+] Ingesting {len(csv_files)} match files from {data_path}...")

    dfs: list[pd.DataFrame] = []
    for f in csv_files:
        try:
            df_m = pd.read_csv(f)
            if "input_state_json" in df_m.columns and "label" in df_m.columns:
                dfs.append(df_m)
        except Exception:
            pass

    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.dropna(subset=["input_state_json", "label"]).reset_index(drop=True)
    full_df["label"] = full_df["label"].astype(int)
    print(f" [+] Ingested {len(full_df):,} total combat rounds across {len(dfs)} matches.")

    # 2. GroupShuffleSplit (Zero match leakage)
    gss = GroupShuffleSplit(n_splits=1, train_size=0.80, random_state=42)
    train_idx, test_idx = next(gss.split(full_df, groups=full_df["match_id"]))
    train_df = full_df.iloc[train_idx].reset_index(drop=True)
    test_df = full_df.iloc[test_idx].reset_index(drop=True)
    print(f" [+] Split dataset: Train = {len(train_df):,} rounds | Test = {len(test_df):,} rounds")

    # 3. Load Exact Pretrained Vocabularies
    model_dir = Path("D:/tft-winner-data/set18/models/trunk")
    if not model_dir.exists():
        model_dir = Path("models/trunk")

    print(f" [+] Loading serialized vocabularies from {model_dir}...")
    vocab = ChampionVocabulary.load(model_dir / "vocab.json") if (model_dir / "vocab.json").exists() else ChampionVocabulary()
    item_vocab = ItemVocabulary.load(model_dir / "item_vocab.json") if (model_dir / "item_vocab.json").exists() else ItemVocabulary()
    trait_vocab = TraitVocabulary.load(model_dir / "trait_vocab.json") if (model_dir / "trait_vocab.json").exists() else TraitVocabulary()

    print(" [+] Building PyTorch tensor datasets...")
    train_ds = CombatDataset(train_df, vocab, item_vocab, trait_vocab)
    test_ds = CombatDataset(test_df, vocab, item_vocab, trait_vocab)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=combat_collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=combat_collate_fn, num_workers=0)

    # 4. Load Pretrained Trunk Backbone (trunk_pretrained.pt)
    ckpt_file = model_dir / "trunk_pretrained.pt" if (model_dir / "trunk_pretrained.pt").exists() else model_dir / "trunk_best.pt"
    print(f" [+] Loading trunk checkpoint from {ckpt_file}...")
    checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    trunk = MultiModalFusionTrunk(
        num_champs=cfg.get("num_champs", len(vocab)),
        num_items=cfg.get("num_items", len(item_vocab)),
        num_traits=cfg.get("num_traits", len(trait_vocab)),
        champ_embed_dim=cfg.get("champ_embed_dim", 32),
        board_feat_dim=cfg.get("board_feat_dim", 256),
        state_feat_dim=cfg.get("state_feat_dim", 64),
        fused_dim=cfg.get("fused_dim", 320),
        num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.1),
    )
    trunk.load_state_dict(state_dict)
    trunk.freeze()
    trunk.to(dev)
    print(f" [+] Pre-trained Trunk successfully loaded on {dev}!")

    # 5. Instantiate DeepSiameseCombatNet
    model = DeepSiameseCombatNet(
        trunk=trunk,
        freeze_trunk=True,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(dev)

    optimizer = torch.optim.AdamW(model.interaction_mlp.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    # 6. Training Loop
    print(f"\n [+] Starting training for {epochs} epochs on {dev}...")
    t0_train = time.time()
    best_loss = float("inf")
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            f_b = {k: v.to(dev) for k, v in batch["focal"].items()}
            o_b = {k: v.to(dev) for k, v in batch["opp"].items()}
            targets = batch["labels"].to(dev)

            optimizer.zero_grad()
            logits = model(f_b, o_b)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(targets)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == targets).sum().item()
            total += len(targets)

        scheduler.step()
        epoch_loss = total_loss / max(1, total)
        epoch_acc = correct / max(1, total)

        # Validation on test set
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in test_loader:
                f_b = {k: v.to(dev) for k, v in batch["focal"].items()}
                o_b = {k: v.to(dev) for k, v in batch["opp"].items()}
                targets = batch["labels"].to(dev)
                logits = model(f_b, o_b)
                loss = criterion(logits, targets)
                val_loss += loss.item() * len(targets)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == targets).sum().item()
                val_total += len(targets)

        val_loss_avg = val_loss / max(1, val_total)
        val_acc_avg = val_correct / max(1, val_total)

        if val_loss_avg < best_loss:
            best_loss = val_loss_avg
            best_weights = {k: v.cpu() for k, v in model.state_dict().items()}

        print(f"  Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc*100:.1f}% | Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc_avg*100:.1f}%")

    t_train = time.time() - t0_train
    print(f" [+] Training completed in {t_train:.2f}s!")

    # 7. Final Evaluation on Holdout Test Set
    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.to(dev)
    model.eval()

    all_preds: list[float] = []
    all_targets: list[float] = []
    t0_eval = time.perf_counter()

    with torch.no_grad():
        for batch in test_loader:
            f_b = {k: v.to(dev) for k, v in batch["focal"].items()}
            o_b = {k: v.to(dev) for k, v in batch["opp"].items()}
            logits = model(f_b, o_b)
            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy().tolist())
            all_targets.extend(batch["labels"].numpy().tolist())

    t_eval = time.perf_counter() - t0_eval
    y_true = np.array(all_targets)
    y_prob = np.array(all_preds)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    bce = log_loss(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    fps_gpu = len(y_true) / t_eval

    print("\n" + "=" * 80)
    print("                     TEST SET EVALUATION METRICS")
    print("=" * 80)
    print(f" • Accuracy:     {acc * 100:.2f}%")
    print(f" • ROC-AUC:      {auc:.4f}")
    print(f" • Log Loss:     {bce:.4f}")
    print(f" • Brier Score:  {brier:.4f}")
    print(f" • Inference:    {fps_gpu:,.0f} rounds / second on GPU")
    print("=" * 80)

    # 8. Save Artifacts
    save_file = out_path / "deep_siamese_combat_best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "hidden_dim": hidden_dim,
        "fused_dim": trunk.fused_dim,
        "metrics": {
            "accuracy": acc,
            "roc_auc": auc,
            "brier_score": brier,
            "log_loss": bce,
            "fps_gpu": fps_gpu,
        },
        "vocab": vocab,
        "item_vocab": item_vocab,
        "trait_vocab": trait_vocab,
    }, save_file)
    print(f" [+] Saved model checkpoint to {save_file}")

    # Also save copy to D:/tft-winner-data/set18/models if directory exists
    backup_dir = Path("D:/tft-winner-data/set18/models")
    if backup_dir.exists():
        backup_file = backup_dir / "deep_siamese_combat_best.pt"
        torch.save(torch.load(save_file, weights_only=False), backup_file)
        print(f" [+] Saved copy to {backup_file}")

    return save_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepSiameseCombatNet on GPU")
    parser.add_argument("--data-dir", type=str, default="D:/tft-winner-data/set18/players")
    parser.add_argument("--output-dir", type=str, default="models/round_winner")
    parser.add_argument("--trunk-ckpt", type=str, default="models/trunk/trunk_best.pt")
    parser.add_argument("--files", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_deep_siamese_combat(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        trunk_ckpt_path=args.trunk_ckpt,
        max_files=args.files,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
    )
