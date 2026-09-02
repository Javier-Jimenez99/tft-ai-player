"""Unit tests for Phase 1: Pre-training the Multi-Modal Fusion Trunk with Tri-Objective Learning."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from tft_ai_player.embeddings import (
    BoardCNNEncoder,
    BoardHexTransformer,
    Champ2Vec,
    ChampionVocabulary,
    HexDistanceAttention,
    HexTransformerBlock,
    InfoNCELoss,
    ItemVocabulary,
    MultiModalFusionTrunk,
    MultiTaskTrunkLoss,
    PermutationInvariantChamp2Vec,
    SnapshotPairCollate,
    StateMLP,
    TFTAxialBoardEncoder,
    TFTPretrainDataset,
    TraitEncoder,
    TraitVocabulary,
    TrunkPreTrainer,
    TrunkPretrainModel,
    build_hex_geodesic_distance_matrix,
    create_synthetic_trajectory_dataset,
    is_pvp_round,
    parse_loc_to_row_col,
    parse_stage_string,
)


def test_champion_item_and_trait_vocabularies() -> None:
    vocab = ChampionVocabulary()
    item_vocab = ItemVocabulary()
    trait_vocab = TraitVocabulary()

    assert len(vocab) > 1
    assert vocab.encode("<EMPTY>") == 0
    assert vocab.decode(0) == "<EMPTY>"

    idx1 = vocab.add_champion("DA_18_Cassiopeia")
    assert idx1 > 0
    assert vocab.encode("DA_18_Cassiopeia") == idx1
    assert vocab.decode(idx1) == "DA_18_Cassiopeia"

    assert len(item_vocab) > 1
    assert item_vocab.encode("<NO_ITEM>") == 0
    assert item_vocab.decode(0) == "<NO_ITEM>"

    it_idx = item_vocab.add_item("DA_GuinsoosRageblade")
    assert it_idx > 0
    assert item_vocab.encode("DA_GuinsoosRageblade") == it_idx

    assert len(trait_vocab) > 1
    t_idx = trait_vocab.add_trait("Riftbeast")
    assert t_idx >= 0
    t_vec = trait_vocab.compute_trait_vector(["DA_18_Cassiopeia", "DA_18_Camille"])
    assert len(t_vec) == len(trait_vocab)
    assert isinstance(t_vec, np.ndarray)

    with tempfile.TemporaryDirectory() as tmpdir:
        v_path = Path(tmpdir) / "vocab.json"
        vocab.save(v_path)
        loaded = ChampionVocabulary.load(v_path)
        assert len(loaded) == len(vocab)
        assert loaded.encode("DA_18_Cassiopeia") == idx1

        it_path = Path(tmpdir) / "item_vocab.json"
        item_vocab.save(it_path)
        loaded_it = ItemVocabulary.load(it_path)
        assert len(loaded_it) == len(item_vocab)

        tr_path = Path(tmpdir) / "trait_vocab.json"
        trait_vocab.save(tr_path)
        loaded_tr = TraitVocabulary.load(tr_path)
        assert len(loaded_tr) == len(trait_vocab)


def test_champ2vec_permutation_invariance_and_strict_zero_gating() -> None:
    out_dim = 32
    c2v = Champ2Vec(num_champs=100, num_items=50, out_dim=out_dim)

    champ_id = torch.tensor([5, 5, 5], dtype=torch.long)
    star_lvl = torch.tensor([2, 2, 2], dtype=torch.long)

    # 3 items in 3 different permutations: (10, 20, 30), (20, 30, 10), (30, 10, 20)
    items_perm1 = torch.tensor([[10, 20, 30], [20, 30, 10], [30, 10, 20]], dtype=torch.long)

    out = c2v(champ_id, star_lvl, items_perm1)
    assert out.shape == (3, out_dim)

    # Mathematically identical output across all permutations of the same items
    assert torch.allclose(out[0], out[1], atol=1e-6)
    assert torch.allclose(out[1], out[2], atol=1e-6)

    # Empty token index 0 must strictly evaluate to exact 0.0 zero vector (no bias or LayerNorm leakage)
    empty_out = c2v(torch.tensor([0, 0]), torch.tensor([0, 0]), torch.tensor([[0, 0, 0], [0, 0, 0]]))
    assert torch.allclose(empty_out, torch.zeros(2, out_dim), atol=1e-7)

    # Test backward gradient flow
    loss = (out ** 2).sum()
    loss.backward()
    assert c2v.champ_embed.weight.grad is not None
    assert c2v.item_embed.weight.grad is not None
    assert c2v.star_embed.weight.grad is not None


def test_hex_geodesic_distance_matrix_and_transformer_pooling() -> None:
    dist_matrix = build_hex_geodesic_distance_matrix()
    assert dist_matrix.shape == (28, 28)
    assert (torch.diagonal(dist_matrix) == 0.0).all()
    assert dist_matrix.max().item() <= 8.0  # Max distance on 4x7 grid is 8

    # Verify symmetry
    assert torch.allclose(dist_matrix, dist_matrix.T)

    # Test 28-token Hex Transformer with Masked Mean Pooling
    batch_size = 4
    num_traits = 40
    transformer = BoardHexTransformer(
        token_in_dim=32,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        num_traits=num_traits,
        out_dim=256,
    )

    board_tokens = torch.randn(batch_size, 4, 7, 32, requires_grad=True)
    champ_ids = torch.randint(0, 50, (batch_size, 4, 7))
    trait_vec = torch.rand(batch_size, num_traits)

    feat = transformer(board_tokens, board_champ_ids=champ_ids, trait_vec=trait_vec)
    assert feat.shape == (batch_size, 256)

    loss = (feat ** 2).sum()
    loss.backward()
    assert board_tokens.grad is not None
    assert transformer.unit_proj.weight.grad is not None
    assert transformer.pos_embed.grad is not None


def test_state_mlp() -> None:
    batch_size = 4
    state_mlp = StateMLP(in_features=8, out_dim=64)
    state_scalars = torch.randn(batch_size, 8)
    state_feat = state_mlp(state_scalars)
    assert state_feat.shape == (batch_size, 64)


def test_multimodal_fusion_trunk_forward_and_freeze() -> None:
    batch_size = 4
    trunk = MultiModalFusionTrunk(
        num_champs=100,
        num_items=50,
        num_traits=40,
        champ_embed_dim=32,
        board_feat_dim=256,
        state_feat_dim=64,
        fused_dim=320,
    )

    board_ids = torch.randint(0, 50, (batch_size, 4, 7))
    board_stars = torch.randint(1, 4, (batch_size, 4, 7))
    board_items = torch.randint(0, 30, (batch_size, 4, 7, 3))
    board_traits = torch.rand(batch_size, 40)
    state_scalars = torch.rand(batch_size, 8)

    fused = trunk(
        board_champ_ids=board_ids,
        board_star_levels=board_stars,
        board_item_ids=board_items,
        board_traits=board_traits,
        state_scalars=state_scalars,
    )
    assert fused.shape == (batch_size, 320)

    loss = (fused ** 2).sum()
    loss.backward()
    assert trunk.champ2vec.champ_embed.weight.grad is not None
    assert trunk.champ2vec.item_embed.weight.grad is not None
    assert trunk.board_encoder.trait_encoder.net[0].weight.grad is not None

    trunk.freeze()
    for p in trunk.parameters():
        assert not p.requires_grad

    trunk.unfreeze()
    for p in trunk.parameters():
        assert p.requires_grad

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "trunk.pt"
        trunk.save_trunk(save_path)
        loaded = MultiModalFusionTrunk.load_trunk(save_path)
        assert loaded.fused_dim == 320


def test_tri_objective_multitask_loss_with_pvp_and_overlap_masking() -> None:
    batch_size = 8
    proj_dim = 128
    loss_fn = MultiTaskTrunkLoss(
        value_weight=1.0,
        micro_weight=0.5,
        contrast_weight=0.15,
        temperature=0.07,
        max_overlap_threshold=0.70,
    )

    val_logits = torch.randn(batch_size, 2, requires_grad=True)
    val_targets = torch.randint(0, 2, (batch_size,))

    combat_logits = torch.randn(batch_size, 1, requires_grad=True)
    combat_targets = torch.randint(0, 2, (batch_size,)).float()
    pvp_mask = torch.tensor([1, 1, 1, 0, 1, 0, 1, 1], dtype=torch.float32)

    anchor_board_ids = torch.randint(1, 10, (batch_size, 4, 7))
    positive_board_ids = torch.randint(1, 10, (batch_size, 4, 7))

    z_a = torch.randn(batch_size, proj_dim, requires_grad=True)
    z_p = z_a + 0.1 * torch.randn(batch_size, proj_dim)

    total_loss, metrics = loss_fn(
        value_logits=val_logits,
        value_targets=val_targets,
        anchor_proj=z_a,
        positive_proj=z_p,
        combat_win_logits=combat_logits,
        combat_win_targets=combat_targets,
        is_pvp_mask=pvp_mask,
        anchor_board_ids=anchor_board_ids,
        positive_board_ids=positive_board_ids,
    )

    assert total_loss.item() > 0.0
    assert "value_ce_loss" in metrics
    assert "micro_combat_loss" in metrics
    assert "combat_acc" in metrics
    assert "info_nce_loss" in metrics
    assert "pos_sim" in metrics
    assert metrics["pos_sim"] > metrics["neg_sim"]

    total_loss.backward()
    assert z_a.grad is not None
    assert val_logits.grad is not None
    assert combat_logits.grad is not None


def test_trunk_pretrain_model_with_combat_head_and_isolated_board_proj() -> None:
    model = TrunkPretrainModel(
        num_champs=100,
        num_items=50,
        num_traits=40,
        champ_embed_dim=32,
        board_feat_dim=256,
        state_feat_dim=64,
        fused_dim=320,
        proj_dim=128,
        num_classes=2,
    )

    b = 4
    snap = {
        "board_champ_ids": torch.randint(0, 50, (b, 4, 7)),
        "board_star_levels": torch.randint(1, 4, (b, 4, 7)),
        "board_item_ids": torch.randint(0, 30, (b, 4, 7, 3)),
        "board_traits": torch.rand(b, 40),
        "state_scalars": torch.rand(b, 8),
    }

    fused, val_logits, combat_logits, proj = model.forward_dict(snap)
    assert fused.shape == (b, 320)
    assert val_logits.shape == (b, 2)
    assert combat_logits.shape == (b, 1)
    assert proj.shape == (b, 128)


def test_pretrain_dataset_and_match_grouped_split() -> None:
    dataset = create_synthetic_trajectory_dataset(num_matches=10, rounds_per_match=10)
    assert len(dataset) > 0

    item = dataset[0]
    assert "anchor" in item
    assert "positive" in item
    assert item["anchor"]["board_champ_ids"].shape == (4, 7)
    assert item["anchor"]["board_item_ids"].shape == (4, 7, 3)
    assert "board_traits" in item["anchor"]
    assert "combat_label" in item["anchor"]
    assert "is_pvp" in item["anchor"]
    assert item["anchor"]["state_scalars"].shape == (8,)

    # Test match_id grouped splitting
    train_subset, val_subset = dataset.split_by_match_id(val_split=0.2, seed=42)
    assert len(train_subset) > 0
    assert len(val_subset) > 0
    assert len(train_subset) + len(val_subset) == len(dataset)

    # Verify zero match_id overlap between train and val
    train_matches = set(dataset.match_ids[i] for i in train_subset.indices)
    val_matches = set(dataset.match_ids[i] for i in val_subset.indices)
    assert len(train_matches.intersection(val_matches)) == 0

    collate = SnapshotPairCollate()
    batch = collate([dataset[i] for i in range(min(4, len(dataset)))])
    assert batch["anchor"]["board_champ_ids"].shape == (4, 4, 7)
    assert batch["anchor"]["board_item_ids"].shape == (4, 4, 7, 3)
    assert batch["anchor"]["board_traits"].shape[0] == 4
    assert batch["positive"]["board_traits"].shape[0] == 4
    assert batch["anchor"]["value_targets"].shape == (4,)
    assert batch["anchor"]["combat_targets"].shape == (4,)
    assert batch["anchor"]["is_pvp_mask"].shape == (4,)


def test_end_to_end_pretraining_loop() -> None:
    dataset = create_synthetic_trajectory_dataset(num_matches=6, rounds_per_match=8)
    trainer = TrunkPreTrainer(
        champ_embed_dim=16,
        board_feat_dim=64,
        state_feat_dim=32,
        fused_dim=96,
        proj_dim=64,
        value_weight=1.0,
        micro_weight=0.5,
        contrast_weight=0.15,
        lr=1e-3,
        use_wandb=False,
        device="cpu",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        summary = trainer.fit(
            dataset=dataset,
            epochs=2,
            batch_size=8,
            output_dir=tmpdir,
            verbose=False,
        )

        assert summary["total_epochs"] == 2
        assert (Path(tmpdir) / "trunk_best.pt").exists()
        assert (Path(tmpdir) / "trunk_pretrained.pt").exists()
        assert (Path(tmpdir) / "vocab.json").exists()
        assert (Path(tmpdir) / "item_vocab.json").exists()
        assert (Path(tmpdir) / "trait_vocab.json").exists()


def test_parsing_and_pvp_utilities() -> None:
    assert parse_loc_to_row_col("A1") == (0, 0)
    assert parse_loc_to_row_col("d7") == (3, 6)
    assert parse_loc_to_row_col("C_6") == (2, 5)
    assert parse_loc_to_row_col("INVALID") is None

    assert parse_stage_string("2-1") == (2, 1)
    assert parse_stage_string("5-6") == (5, 6)
    assert parse_stage_string("invalid") == (0, 0)

    # PVP round tests
    assert is_pvp_round((1, 2)) == 0  # Stage 1 is minion PvE
    assert is_pvp_round((2, 4)) == 0  # Carousel
    assert is_pvp_round((3, 7)) == 0  # Creeps
    assert is_pvp_round((2, 1)) == 1  # Genuine PVP combat
    assert is_pvp_round((4, 3)) == 1  # Genuine PVP combat


def test_models_package_backward_compatibility() -> None:
    import tft_ai_player.models as models
    assert hasattr(models, "Champ2Vec")
    assert hasattr(models, "PermutationInvariantChamp2Vec")
    assert hasattr(models, "BoardHexTransformer")
    assert hasattr(models, "BoardCNNEncoder")


def test_hyperparameter_sweep_configuration() -> None:
    from tft_ai_player.embeddings.sweep import get_default_sweep_config
    config = get_default_sweep_config(project="test-project")
    assert config["method"] == "bayes"
    assert "metric" in config
    assert "parameters" in config
    assert "lr" in config["parameters"]
    assert "batch_size" in config["parameters"]
    assert "dropout" in config["parameters"]
    assert "num_layers" in config["parameters"]
    assert "contrast_weight" in config["parameters"]
    assert "value_weight" in config["parameters"]
    assert "micro_weight" in config["parameters"]
    assert "temperature" in config["parameters"]
