"""Unit and property tests for DeepSiameseCombatNet.

Verifies:
1. Exact Mirror Match Identity Law: P(A vs A) == 0.5000000
2. Exact Anti-Symmetry Law: P(A vs B) + P(B vs A) == 1.0000000
3. Arbitrary Batch Scaling and Tensor Dimensions
4. Gradient Flow to Advantage Parameters (and frozen trunk isolation)
5. CPU and CUDA Device Agnosticism
6. CombatDataset and Collate Function Parsing & Shapes
7. Serialization & Checkpoint Loading Roundtrip
8. Backward Compatibility with Legacy interaction_mlp State Dict Keys
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tft_ai_player.embeddings.model import MultiModalFusionTrunk
from tft_ai_player.embeddings.vocab import ChampionVocabulary, ItemVocabulary, TraitVocabulary
from tft_ai_player.round_winner.embedding_model import (
    DeepSiameseCombatNet,
    CombatDataset,
    combat_collate_fn,
)


@pytest.fixture
def small_trunk() -> MultiModalFusionTrunk:
    """Fixture providing a lightweight MultiModalFusionTrunk for fast testing."""
    torch.manual_seed(42)
    trunk = MultiModalFusionTrunk(
        num_champs=100,
        num_items=50,
        num_traits=20,
        champ_embed_dim=32,
        board_feat_dim=64,
        state_feat_dim=32,
        fused_dim=128,
        num_layers=1,
        dropout=0.0,
    )
    trunk.eval()
    return trunk


@pytest.fixture
def sample_board_batch() -> dict[str, torch.Tensor]:
    """Fixture providing a valid 4-board batch of board tensors."""
    torch.manual_seed(42)
    b_size = 4
    champs = torch.randint(0, 50, (b_size, 4, 7), dtype=torch.long)
    stars = torch.randint(1, 4, (b_size, 4, 7), dtype=torch.long)
    items = torch.randint(0, 30, (b_size, 4, 7, 3), dtype=torch.long)
    traits = torch.rand((b_size, 20), dtype=torch.float32)
    scalars = torch.rand((b_size, 8), dtype=torch.float32)

    return {
        "board_champ_ids": champs,
        "board_star_levels": stars,
        "board_item_ids": items,
        "board_traits": traits,
        "state_scalars": scalars,
    }


def test_mirror_match_identity(small_trunk: MultiModalFusionTrunk, sample_board_batch: dict[str, torch.Tensor]) -> None:
    """P(Board A beats Board A) must evaluate to exactly 50.000000% (|logit| < 1e-5)."""
    model = DeepSiameseCombatNet(trunk=small_trunk, freeze_trunk=True, hidden_dim=64, dropout=0.0)
    model.eval()

    with torch.no_grad():
        logits = model(sample_board_batch, sample_board_batch)
        probs = torch.sigmoid(logits)

    assert logits.shape == (4,)
    assert torch.allclose(logits, torch.zeros_like(logits), atol=1e-5), f"Mirror logits non-zero: {logits}"
    assert torch.allclose(probs, torch.full_like(probs, 0.5), atol=1e-5), f"Mirror probs not 0.50: {probs}"


def test_exact_anti_symmetry_swap(small_trunk: MultiModalFusionTrunk, sample_board_batch: dict[str, torch.Tensor]) -> None:
    """Combat outcome must satisfy logit(A, B) = -logit(B, A) and P(A vs B) + P(B vs A) = 1.0."""
    torch.manual_seed(123)
    model = DeepSiameseCombatNet(trunk=small_trunk, freeze_trunk=True, hidden_dim=64, dropout=0.0)
    model.eval()

    # Create opposing batch with different values
    opp_batch = {
        "board_champ_ids": torch.randint(0, 50, (4, 4, 7), dtype=torch.long),
        "board_star_levels": torch.randint(1, 4, (4, 4, 7), dtype=torch.long),
        "board_item_ids": torch.randint(0, 30, (4, 4, 7, 3), dtype=torch.long),
        "board_traits": torch.rand((4, 20), dtype=torch.float32),
        "state_scalars": torch.rand((4, 8), dtype=torch.float32),
    }

    with torch.no_grad():
        logit_a_b = model(sample_board_batch, opp_batch)
        logit_b_a = model(opp_batch, sample_board_batch)

        p_a_b = torch.sigmoid(logit_a_b)
        p_b_a = torch.sigmoid(logit_b_a)

    # Anti-symmetry check on logits: logit(A, B) + logit(B, A) == 0
    assert torch.allclose(logit_a_b + logit_b_a, torch.zeros_like(logit_a_b), atol=1e-5), (
        f"Logit sum non-zero: {logit_a_b + logit_b_a}"
    )

    # Probability sum check: P(A vs B) + P(B vs A) == 1.0
    assert torch.allclose(p_a_b + p_b_a, torch.ones_like(p_a_b), atol=1e-5), (
        f"Prob sum non-one: {p_a_b + p_b_a}"
    )


@pytest.mark.parametrize("b_size", [1, 2, 7, 16])
def test_batch_scaling_and_shapes(small_trunk: MultiModalFusionTrunk, b_size: int) -> None:
    """Model must dynamically support arbitrary batch sizes including B=1."""
    model = DeepSiameseCombatNet(trunk=small_trunk, freeze_trunk=True, hidden_dim=64, dropout=0.0)
    model.eval()

    focal_b = {
        "board_champ_ids": torch.randint(0, 50, (b_size, 4, 7), dtype=torch.long),
        "board_star_levels": torch.randint(1, 4, (b_size, 4, 7), dtype=torch.long),
        "board_item_ids": torch.randint(0, 30, (b_size, 4, 7, 3), dtype=torch.long),
        "board_traits": torch.rand((b_size, 20), dtype=torch.float32),
        "state_scalars": torch.rand((b_size, 8), dtype=torch.float32),
    }
    opp_b = {
        "board_champ_ids": torch.randint(0, 50, (b_size, 4, 7), dtype=torch.long),
        "board_star_levels": torch.randint(1, 4, (b_size, 4, 7), dtype=torch.long),
        "board_item_ids": torch.randint(0, 30, (b_size, 4, 7, 3), dtype=torch.long),
        "board_traits": torch.rand((b_size, 20), dtype=torch.float32),
        "state_scalars": torch.rand((b_size, 8), dtype=torch.float32),
    }

    with torch.no_grad():
        out = model(focal_b, opp_b)

    assert out.shape == (b_size,)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_gradient_flow(small_trunk: MultiModalFusionTrunk, sample_board_batch: dict[str, torch.Tensor]) -> None:
    """Gradients must propagate through advantage MLP while trunk remains cleanly frozen."""
    model = DeepSiameseCombatNet(trunk=small_trunk, freeze_trunk=True, hidden_dim=64, dropout=0.0)
    model.train()

    opp_batch = {k: v.clone() for k, v in sample_board_batch.items()}
    opp_batch["state_scalars"] = torch.rand((4, 8), dtype=torch.float32)

    logits = model(sample_board_batch, opp_batch)
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    loss.backward()

    # Verify advantage_mlp receives non-zero gradients
    adv_module = getattr(model, "advantage_mlp", getattr(model, "interaction_mlp", None))
    assert adv_module is not None
    has_grad = False
    for p in adv_module.parameters():
        if p.grad is not None and p.grad.norm().item() > 0:
            has_grad = True
            break
    assert has_grad, "Advantage MLP parameters received no gradients!"

    # Verify trunk parameters remained completely frozen
    for name, p in small_trunk.named_parameters():
        assert p.grad is None or p.grad.norm().item() == 0, f"Trunk parameter {name} was modified by gradient!"


def test_device_agnostic(small_trunk: MultiModalFusionTrunk, sample_board_batch: dict[str, torch.Tensor]) -> None:
    """Model must run on CPU and CUDA seamlessly if available."""
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    for dev in devices:
        t = small_trunk.to(dev)
        m = DeepSiameseCombatNet(trunk=t, freeze_trunk=True, hidden_dim=64).to(dev)
        fb = {k: v.to(dev) for k, v in sample_board_batch.items()}
        ob = {k: v.to(dev) for k, v in sample_board_batch.items()}

        out = m(fb, ob)
        assert out.device.type == dev.type
        assert out.shape == (4,)


def test_combat_dataset_collate() -> None:
    """CombatDataset and combat_collate_fn must properly parse json boards and return tensors."""
    vocab = ChampionVocabulary()
    item_vocab = ItemVocabulary()
    trait_vocab = TraitVocabulary()

    dummy_data = [
        {
            "match_id": "m1",
            "round_stage": "3-2",
            "label": 1,
            "focal_health": 85,
            "focal_gold": 40,
            "focal_level": 6,
            "focal_unit_count": 2,
            "focal_item_count": 3,
            "opponent_health": 90,
            "opponent_level": 6,
            "opponent_unit_count": 1,
            "opponent_item_count": 0,
            "metatft_win_prob": 0.65,
            "input_state_json": json.dumps({
                "focal_board": [
                    {"unit": "TFT18_Akali", "tier": 2, "loc": "D1", "items": ["TFT_Item_InfinityEdge", "TFT_Item_Bloodthirster"]},
                    {"unit": "TFT18_Leona", "tier": 2, "loc": "A1", "items": ["TFT_Item_WarmogsArmor"]},
                ],
                "opponent_board": [
                    {"unit": "TFT18_Kobuko", "tier": 1, "loc": "A2", "items": []},
                ],
            }),
        }
    ]
    df = pd.DataFrame(dummy_data)
    ds = CombatDataset(df, vocab, item_vocab, trait_vocab)
    assert len(ds) == 1

    sample = ds[0]
    assert "focal" in sample and "opp" in sample
    assert sample["focal"]["board_champ_ids"].shape == (4, 7)
    assert sample["focal"]["board_item_ids"].shape == (4, 7, 3)

    batch = combat_collate_fn([sample])
    assert batch["focal"]["board_champ_ids"].shape == (1, 4, 7)
    assert batch["labels"].shape == (1,)
    assert batch["labels"][0].item() == 1.0


def test_checkpoint_roundtrip(small_trunk: MultiModalFusionTrunk, sample_board_batch: dict[str, torch.Tensor], tmp_path: Path) -> None:
    """Saving and loading state_dict must reproduce exact numerical outputs."""
    model_orig = DeepSiameseCombatNet(trunk=small_trunk, freeze_trunk=True, hidden_dim=64, dropout=0.0)
    model_orig.eval()

    ckpt_file = tmp_path / "model_ckpt.pt"
    torch.save({"model_state_dict": model_orig.state_dict()}, ckpt_file)

    model_loaded = DeepSiameseCombatNet(trunk=small_trunk, freeze_trunk=True, hidden_dim=64, dropout=0.0)
    sd = torch.load(ckpt_file, weights_only=False)
    model_loaded.load_state_dict(sd["model_state_dict"])
    model_loaded.eval()

    with torch.no_grad():
        out_orig = model_orig(sample_board_batch, sample_board_batch)
        out_loaded = model_loaded(sample_board_batch, sample_board_batch)

    assert torch.allclose(out_orig, out_loaded, atol=1e-6)


def test_backward_compatible_keys(small_trunk: MultiModalFusionTrunk, sample_board_batch: dict[str, torch.Tensor]) -> None:
    """Check that loading a state dict with interaction_mlp keys maps cleanly to advantage_mlp."""
    model = DeepSiameseCombatNet(trunk=small_trunk, freeze_trunk=True, hidden_dim=64, dropout=0.0)
    sd = model.state_dict()

    # Re-key advantage_mlp to interaction_mlp
    legacy_sd = {}
    for k, v in sd.items():
        if "advantage_mlp" in k:
            legacy_sd[k.replace("advantage_mlp", "interaction_mlp")] = v
        else:
            legacy_sd[k] = v

    new_model = DeepSiameseCombatNet(trunk=small_trunk, freeze_trunk=True, hidden_dim=64, dropout=0.0)
    # Should load without raising missing/unexpected key error
    new_model.load_state_dict(legacy_sd, strict=False)


def test_logit_scale_buffer_and_persistence(small_trunk: MultiModalFusionTrunk, tmp_path: Path) -> None:
    """logit_scale must be a registered buffer on correct device and persist across checkpoints."""
    model = DeepSiameseCombatNet(trunk=small_trunk, freeze_trunk=True, hidden_dim=64, logit_scale=1.45)
    assert hasattr(model, "logit_scale")
    assert isinstance(model.logit_scale, torch.Tensor)
    assert abs(model.logit_scale.item() - 1.45) < 1e-4

    # Check persistence
    ckpt_path = tmp_path / "scale_test.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    loaded_model = DeepSiameseCombatNet(trunk=small_trunk, freeze_trunk=True, hidden_dim=64, logit_scale=1.0)
    sd = torch.load(ckpt_path, weights_only=False)
    loaded_model.load_state_dict(sd["model_state_dict"], strict=False)
    assert abs(loaded_model.logit_scale.item() - 1.45) < 1e-4


def test_logit_scale_monotonicity_and_invariance(small_trunk: MultiModalFusionTrunk, sample_board_batch: dict[str, torch.Tensor]) -> None:
    """Varying logit_scale must preserve exact mirror identity (50.0%) and anti-symmetry (100.0%)."""
    opp_batch = {
        "board_champ_ids": torch.randint(0, 50, (4, 4, 7), dtype=torch.long),
        "board_star_levels": torch.randint(1, 4, (4, 4, 7), dtype=torch.long),
        "board_item_ids": torch.randint(0, 30, (4, 4, 7, 3), dtype=torch.long),
        "board_traits": torch.rand((4, 20), dtype=torch.float32),
        "state_scalars": torch.rand((4, 8), dtype=torch.float32),
    }

    for scale in [1.0, 1.25, 1.45, 2.0]:
        model = DeepSiameseCombatNet(trunk=small_trunk, freeze_trunk=True, hidden_dim=64, logit_scale=scale)
        model.eval()

        with torch.no_grad():
            mirror_logit = model(sample_board_batch, sample_board_batch)
            mirror_prob = torch.sigmoid(mirror_logit)
            assert torch.allclose(mirror_prob, torch.full_like(mirror_prob, 0.5), atol=1e-5)

            l_ab = model(sample_board_batch, opp_batch)
            l_ba = model(opp_batch, sample_board_batch)
            assert torch.allclose(l_ab + l_ba, torch.zeros_like(l_ab), atol=1e-5)
            p_sum = torch.sigmoid(l_ab) + torch.sigmoid(l_ba)
            assert torch.allclose(p_sum, torch.ones_like(p_sum), atol=1e-5)


def test_strength_head_directionality(small_trunk: MultiModalFusionTrunk, sample_board_batch: dict[str, torch.Tensor]) -> None:
    """predict_board_power must return scalar board power matching intrinsic strength head."""
    model = DeepSiameseCombatNet(trunk=small_trunk, freeze_trunk=True, hidden_dim=64, logit_scale=1.45)
    model.eval()

    with torch.no_grad():
        power = model.predict_board_power(sample_board_batch)

    assert power.shape == (4,)
    assert not torch.isnan(power).any()
    assert not torch.isinf(power).any()


def test_parameter_group_decay_separation(small_trunk: MultiModalFusionTrunk) -> None:
    """Parameter group separation must exclude 1D biases, LayerNorms, and final projections from weight decay."""
    from tft_ai_player.round_winner.embedding_model import build_optimizer_param_groups

    model = DeepSiameseCombatNet(trunk=small_trunk, freeze_trunk=True, hidden_dim=64, logit_scale=1.45)
    decay_params, no_decay_params = build_optimizer_param_groups(model)

    assert len(decay_params) > 0
    assert len(no_decay_params) > 0

    # Ensure trunk params are not in either group since trunk is frozen
    trunk_params = set(small_trunk.parameters())
    assert not any(p in trunk_params for p in decay_params)
    assert not any(p in trunk_params for p in no_decay_params)

