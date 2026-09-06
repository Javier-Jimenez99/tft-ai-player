"""Comprehensive tests for the State Transition Predictor, Dual-Objective Loss, Strict PvP Filtering, and Trainer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from tft_ai_player.cli import main as cli_main
from tft_ai_player.embeddings import (
    ChampionVocabulary,
    ItemVocabulary,
    MultiModalFusionTrunk,
    ResidualMLPBlock,
    StateTransitionLoss,
    StateTransitionPredictor,
    TraitVocabulary,
    TransitionPredictorTrainer,
    TransitionTrajectoryDataset,
    create_synthetic_transition_dataset,
    extract_latent_transition_pairs,
    is_valid_pvp_transition,
)


def test_residual_mlp_block():
    dim = 512
    block = ResidualMLPBlock(dim=dim, dropout=0.0)
    x = torch.randn(8, dim)
    out = block(x)

    assert out.shape == (8, dim)
    # Output should not be identical to x due to transformations, but should have gradients
    loss = out.sum()
    loss.backward()
    for param in block.parameters():
        assert param.grad is not None


def test_state_transition_predictor_forward():
    model = StateTransitionPredictor(
        input_dim=320,
        hidden_dim=512,
        output_dim=320,
        num_layers=3,
        dropout=0.1,
        use_residual_delta=True,
    )
    s_t = torch.randn(16, 320)
    s_next = model(s_t)

    assert s_next.shape == (16, 320)

    # Test single vector inference
    single_vec = torch.randn(320)
    single_out = model.predict_next_state(single_vec)
    assert single_out.shape == (320,)


def test_state_transition_predictor_serialization(tmp_path: Path):
    model = StateTransitionPredictor(
        input_dim=320,
        hidden_dim=256,
        output_dim=320,
        num_layers=2,
        dropout=0.0,
    )
    model.eval()

    s_t = torch.randn(4, 320)
    orig_out = model(s_t)

    ckpt_path = tmp_path / "test_predictor.pt"
    model.save_predictor(ckpt_path, extra_info={"version": 1.0})

    assert ckpt_path.exists()

    loaded = StateTransitionPredictor.load_predictor(ckpt_path)
    loaded.eval()

    loaded_out = loaded(s_t)
    assert torch.allclose(orig_out, loaded_out, atol=1e-5)


def test_state_transition_loss_identical_and_opposite_vectors():
    loss_fn = StateTransitionLoss(lambda_cosine=1.0, huber_beta=1.0)

    # 1. Identical vectors: Huber loss = 0, Cosine loss = 0 (Cos Sim = 1.0)
    v1 = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    loss_id, metrics_id = loss_fn(v1, v1)

    assert pytest.approx(metrics_id["huber_loss"], abs=1e-5) == 0.0
    assert pytest.approx(metrics_id["cosine_loss"], abs=1e-5) == 0.0
    assert pytest.approx(metrics_id["cosine_similarity"], abs=1e-5) == 1.0
    assert pytest.approx(metrics_id["total_loss"], abs=1e-5) == 0.0

    # 2. Opposite vectors: Huber loss > 0, Cosine loss = 2.0 (Cos Sim = -1.0)
    v2 = -v1
    loss_opp, metrics_opp = loss_fn(v1, v2)
    assert pytest.approx(metrics_opp["cosine_similarity"], abs=1e-5) == -1.0
    assert pytest.approx(metrics_opp["cosine_loss"], abs=1e-5) == 2.0


def test_state_transition_loss_huber_robustness():
    loss_fn = StateTransitionLoss(lambda_cosine=0.5, huber_beta=1.0)

    pred = torch.ones(1, 320, requires_grad=True)
    # Target has a massive outlier dimension (e.g., pivot coordinate shift = 100.0)
    target = torch.ones(1, 320)
    target[0, 0] = 100.0

    loss, metrics = loss_fn(pred, target)
    loss.backward()

    # In Huber loss with beta=1.0, gradient magnitude remains bounded (no explosion)
    grad_val = pred.grad[0, 0].item()
    assert abs(grad_val) <= 2.0



def test_pvp_transition_filtering():
    # 1. Stage 1 rounds should be rejected
    assert not is_valid_pvp_transition((1, 1), "creep", (1, 2), "creep")
    assert not is_valid_pvp_transition((1, 4), "pvp", (2, 1), "pvp")

    # 2. Carousel rounds (x-4) should be rejected
    assert not is_valid_pvp_transition((2, 3), "pvp", (2, 4), "carousel")
    assert not is_valid_pvp_transition((2, 4), "carousel", (2, 5), "pvp")
    assert not is_valid_pvp_transition((3, 3), "pvp", (3, 4), "pvp")

    # 3. Creep / Neutral monster rounds (x-7: Krugs, Wolves, Raptors, Dragon) should be rejected
    assert not is_valid_pvp_transition((2, 6), "pvp", (2, 7), "creep")
    assert not is_valid_pvp_transition((2, 7), "creep", (3, 1), "pvp")
    assert not is_valid_pvp_transition((3, 6), "pvp", (3, 7), "pve")

    # 4. Standard player-driven PvP rounds (2-1 -> 2-2, 2-2 -> 2-3, 2-5 -> 2-6, 3-1 -> 3-2, 5-2 -> 5-3)
    assert is_valid_pvp_transition((2, 1), "pvp", (2, 2), "pvp")
    assert is_valid_pvp_transition((2, 2), "pvp", (2, 3), "pvp")
    assert is_valid_pvp_transition((2, 5), "pvp", (2, 6), "pvp")
    assert is_valid_pvp_transition((3, 1), "pvp", (3, 2), "pvp")
    assert is_valid_pvp_transition((5, 2), "pvp", (5, 3), "pvp")


def test_match_grouped_dataset_splitting():
    dataset = create_synthetic_transition_dataset(num_matches=20, rounds_per_match=10)
    assert len(dataset) > 0

    train_sub, val_sub = dataset.split_by_match_id(val_split=0.25, seed=42)
    assert len(train_sub) > 0
    assert len(val_sub) > 0

    train_matches = set(dataset.match_ids[idx] for idx in train_sub.indices)
    val_matches = set(dataset.match_ids[idx] for idx in val_sub.indices)

    # Zero match ID overlap
    assert len(train_matches.intersection(val_matches)) == 0


def test_latent_extraction_and_caching():
    trunk = MultiModalFusionTrunk(fused_dim=320)
    trunk.freeze()
    trunk.eval()

    dataset = create_synthetic_transition_dataset(num_matches=5, rounds_per_match=8)
    s_t, s_next = extract_latent_transition_pairs(trunk, dataset, batch_size=16, device="cpu", verbose=False)

    assert s_t.shape == (len(dataset), 320)
    assert s_next.shape == (len(dataset), 320)
    assert not torch.isnan(s_t).any()
    assert not torch.isnan(s_next).any()


def test_transition_predictor_trainer_fit(tmp_path: Path):
    trunk = MultiModalFusionTrunk(fused_dim=320)
    trunk.freeze()
    trunk.eval()

    dataset = create_synthetic_transition_dataset(num_matches=8, rounds_per_match=8)

    predictor = StateTransitionPredictor(
        input_dim=320,
        hidden_dim=128,
        output_dim=320,
        num_layers=2,
        dropout=0.0,
    )

    trainer = TransitionPredictorTrainer(
        predictor=predictor,
        trunk=trunk,
        lr=1e-3,
        lambda_cosine=0.5,
        huber_beta=1.0,
        use_wandb=False,
        device="cpu",
    )

    output_dir = tmp_path / "models" / "transition"
    summary = trainer.fit(
        dataset=dataset,
        val_split=0.25,
        epochs=3,
        batch_size=32,
        output_dir=output_dir,
        verbose=False,
    )

    assert summary["total_epochs"] == 3
    assert (output_dir / "predictor_best.pt").exists()
    assert (output_dir / "predictor_final.pt").exists()
    assert (output_dir / "predictor_summary.json").exists()


def test_cli_train_transition_synthetic(tmp_path: Path):
    out_dir = tmp_path / "cli_test"
    ret = cli_main([
        "train-transition",
        "--synthetic",
        "--output-dir",
        str(out_dir),
        "--epochs",
        "2",
        "--batch-size",
        "32",
        "--hidden-dim",
        "128",
        "--num-layers",
        "2",
        "--device",
        "cpu",
        "--no-wandb",
    ])

    assert ret == 0
    assert (out_dir / "predictor_best.pt").exists()
    assert (out_dir / "predictor_summary.json").exists()
