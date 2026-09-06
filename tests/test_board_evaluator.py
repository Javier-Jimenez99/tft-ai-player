from __future__ import annotations

from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

from tft_ai_player.board_evaluator.dataset import (
    BoardPlacementDataset,
    infer_placement_from_trajectory,
)
from tft_ai_player.board_evaluator.model import BoardQualityNet
from tft_ai_player.board_evaluator.trainer import BoardQualityTrainer
from tft_ai_player.embeddings.model import MultiModalFusionTrunk
from tft_ai_player.embeddings.vocab import ChampionVocabulary, ItemVocabulary, TraitVocabulary


def _create_dummy_trunk(fused_dim: int = 384) -> MultiModalFusionTrunk:
    return MultiModalFusionTrunk(
        num_champs=100,
        num_items=50,
        num_traits=20,
        champ_embed_dim=32,
        board_feat_dim=256,
        state_feat_dim=64,
        fused_dim=fused_dim,
        num_layers=1,
    )


def test_board_quality_net_forward_and_bounds() -> None:
    trunk = _create_dummy_trunk(fused_dim=384)
    net = BoardQualityNet(trunk=trunk, freeze_trunk=True, hidden_dim=64)

    for bs in [1, 4, 16]:
        champs = torch.randint(0, 100, (bs, 28))
        stars = torch.randint(1, 4, (bs, 28))
        items = torch.zeros((bs, 28, 3), dtype=torch.long)
        traits = torch.zeros((bs, 20), dtype=torch.float32)
        scalars = torch.rand((bs, 8), dtype=torch.float32)

        pred_place, logits_top4 = net(
            board_champ_ids=champs,
            board_star_levels=stars,
            board_item_ids=items,
            board_traits=traits,
            state_scalars=scalars,
        )

        assert pred_place.shape == (bs, 1)
        assert logits_top4.shape == (bs, 2)
        # Check placement predictions are strictly bounded in [1.0, 8.0]
        assert (pred_place >= 1.0).all()
        assert (pred_place <= 8.0).all()

        probs = F.softmax(logits_top4, dim=-1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(bs), atol=1e-5)


def test_board_quality_net_forward_fused_equivalence() -> None:
    trunk = _create_dummy_trunk(fused_dim=384)
    net = BoardQualityNet(trunk=trunk, freeze_trunk=True, hidden_dim=64)
    net.eval()

    bs = 3
    champs = torch.randint(0, 100, (bs, 28))
    stars = torch.randint(1, 4, (bs, 28))
    items = torch.zeros((bs, 28, 3), dtype=torch.long)
    traits = torch.zeros((bs, 20), dtype=torch.float32)
    scalars = torch.rand((bs, 8), dtype=torch.float32)

    with torch.no_grad():
        pred_full, logits_full = net(
            board_champ_ids=champs,
            board_star_levels=stars,
            board_item_ids=items,
            board_traits=traits,
            state_scalars=scalars,
        )

        fused = trunk(
            board_champ_ids=champs,
            board_star_levels=stars,
            board_item_ids=items,
            board_traits=traits,
            state_scalars=scalars,
        )
        pred_fused, logits_fused = net.forward_fused(fused)

    assert torch.allclose(pred_full, pred_fused, atol=1e-5)
    assert torch.allclose(logits_full, logits_fused, atol=1e-5)


def test_board_quality_net_freeze_trunk_gradient_flow() -> None:
    trunk = _create_dummy_trunk(fused_dim=384)
    net = BoardQualityNet(trunk=trunk, freeze_trunk=True, hidden_dim=64)
    net.train()

    # Verify trunk weights have requires_grad=False
    for p in net.trunk.parameters():
        assert not p.requires_grad

    # Verify head weights have requires_grad=True
    for p in net.placement_head.parameters():
        assert p.requires_grad
    for p in net.top4_head.parameters():
        assert p.requires_grad

    fused = torch.randn(4, 384, requires_grad=False)
    pred_place, logits_top4 = net.forward_fused(fused)

    y_place = torch.tensor([[2.0], [4.0], [1.0], [7.0]], dtype=torch.float32)
    y_top4 = torch.tensor([1, 1, 1, 0], dtype=torch.long)

    loss = F.huber_loss(pred_place, y_place) + F.cross_entropy(logits_top4, y_top4)
    loss.backward()

    # Head parameters must have non-zero gradients
    assert net.placement_head[0].weight.grad is not None
    assert net.placement_head[0].weight.grad.abs().sum() > 0.0
    assert net.top4_head[0].weight.grad is not None
    assert net.top4_head[0].weight.grad.abs().sum() > 0.0


def test_board_quality_net_variable_fused_dim() -> None:
    for f_dim in [320, 384, 512]:
        trunk = _create_dummy_trunk(fused_dim=f_dim)
        net = BoardQualityNet(trunk=trunk, freeze_trunk=True, hidden_dim=64)
        fused = torch.randn(2, f_dim)
        pred_place, logits_top4 = net.forward_fused(fused)
        assert pred_place.shape == (2, 1)
        assert logits_top4.shape == (2, 2)


def test_board_quality_net_checkpoint_roundtrip(tmp_path: Path) -> None:
    trunk = _create_dummy_trunk(fused_dim=384)
    net = BoardQualityNet(trunk=trunk, freeze_trunk=True, hidden_dim=64)

    ckpt_path = tmp_path / "board_quality_test.pt"
    net.save_checkpoint(ckpt_path)
    assert ckpt_path.exists()

    loaded = BoardQualityNet.load_checkpoint(ckpt_path, trunk=trunk)
    loaded.eval()
    net.eval()

    test_input = torch.randn(3, 384)
    with torch.no_grad():
        p1, l1 = net.forward_fused(test_input)
        p2, l2 = loaded.forward_fused(test_input)

    assert torch.allclose(p1, p2, atol=1e-5)
    assert torch.allclose(l1, l2, atol=1e-5)


def test_board_quality_net_evaluate_tensors_single_and_batched() -> None:
    trunk = _create_dummy_trunk(fused_dim=384)
    net = BoardQualityNet(trunk=trunk, freeze_trunk=True, hidden_dim=64)

    champs_1d = torch.randint(0, 100, (28,))
    stars_1d = torch.randint(1, 4, (28,))
    items_2d = torch.zeros((28, 3), dtype=torch.long)
    scalars_1d = torch.rand((8,), dtype=torch.float32)

    q_score, exp_place, top4_prob = net.evaluate_board_tensors(
        board_champ_ids=champs_1d,
        board_star_levels=stars_1d,
        board_item_ids=items_2d,
        state_scalars=scalars_1d,
    )

    assert 1.0 <= exp_place <= 8.0
    assert 0.0 <= top4_prob <= 1.0
    assert isinstance(q_score, float)
    # Quality score = (8.0 - exp_place) + 2 * top4_prob in [0.0, 9.5]
    assert 0.0 <= q_score <= 9.5


def test_board_quality_net_device_cpu_cuda() -> None:
    trunk = _create_dummy_trunk(fused_dim=384)
    net = BoardQualityNet(trunk=trunk, freeze_trunk=True, hidden_dim=64)
    net.to("cpu")
    fused_cpu = torch.randn(2, 384, device="cpu")
    p_cpu, l_cpu = net.forward_fused(fused_cpu)
    assert p_cpu.device.type == "cpu"

    if torch.cuda.is_available():
        net.to("cuda")
        fused_cuda = torch.randn(2, 384, device="cuda")
        p_cuda, l_cuda = net.forward_fused(fused_cuda)
        assert p_cuda.device.type == "cuda"


def test_infer_placement_from_trajectory_empirical_monotonicity() -> None:
    """Verifies that infer_placement_from_trajectory assigns monotonically realistic placements."""
    # Deep stage victory -> 1st place
    p_win_6 = infer_placement_from_trajectory(last_stage=(6, 5), last_outcome="victory")
    assert p_win_6 == 1.0

    p_win_7 = infer_placement_from_trajectory(last_stage=(7, 1), last_outcome="victory")
    assert p_win_7 == 1.0

    # Defeat progression: later survival round -> strictly lower (better) placement
    p_def_6_5 = infer_placement_from_trajectory(last_stage=(6, 5), last_outcome="defeat")
    p_def_6_2 = infer_placement_from_trajectory(last_stage=(6, 2), last_outcome="defeat")
    p_def_5_5 = infer_placement_from_trajectory(last_stage=(5, 5), last_outcome="defeat")
    p_def_5_2 = infer_placement_from_trajectory(last_stage=(5, 2), last_outcome="defeat")
    p_def_4_5 = infer_placement_from_trajectory(last_stage=(4, 5), last_outcome="defeat")
    p_def_4_1 = infer_placement_from_trajectory(last_stage=(4, 1), last_outcome="defeat")
    p_def_3_5 = infer_placement_from_trajectory(last_stage=(3, 5), last_outcome="defeat")

    assert 1.0 <= p_def_6_5 <= 2.5
    assert p_def_6_5 <= p_def_6_2
    assert p_def_6_2 <= p_def_5_5
    assert p_def_5_5 <= p_def_5_2
    assert p_def_5_2 <= p_def_4_5
    assert p_def_4_5 <= p_def_4_1
    assert p_def_4_1 <= p_def_3_5
    assert p_def_3_5 == 8.0

    # All placements strictly within [1.0, 8.0]
    all_placements = [
        p_win_6,
        p_win_7,
        p_def_6_5,
        p_def_6_2,
        p_def_5_5,
        p_def_5_2,
        p_def_4_5,
        p_def_4_1,
        p_def_3_5,
    ]
    for p in all_placements:
        assert 1.0 <= p <= 8.0


def test_board_placement_dataset_item_format() -> None:
    """Verifies that BoardPlacementDataset __getitem__ returns correctly structured and scaled tensors."""
    vocab = ChampionVocabulary()
    vocab.add_champion("TFT18_Draven")
    item_vocab = ItemVocabulary()
    item_vocab.add_item("TFT_Item_InfinityEdge")
    trait_vocab = TraitVocabulary()
    trait_vocab.add_trait("Set18_Dominator")

    ds = BoardPlacementDataset(data_dir="non_existent_path", vocab=vocab, item_vocab=item_vocab, trait_vocab=trait_vocab)
    # Manually append a sample to test __getitem__
    ds.samples.append({
        "board": [
            {"unit": "TFT18_Draven", "star_level": 2, "location": 14, "items": ["TFT_Item_InfinityEdge"]},
        ],
        "stage_tuple": (4, 2),
        "health": 65.0,
        "level": 7.0,
        "gold": 30.0,
        "unit_count": 7.0,
        "item_count": 5.0,
        "streak": 2.0,
        "target_placement": 3.0,
        "target_top4": 1,
    })

    item = ds[0]
    assert item["board_champ_ids"].shape == (28,)
    assert item["board_star_levels"].shape == (28,)
    assert item["board_item_ids"].shape == (28, 3)
    assert item["board_traits"].shape == (len(trait_vocab),)
    assert item["state_scalars"].shape == (8,)
    assert item["target_placement"].shape == (1,)
    assert item["target_top4"].item() == 1

    # Check scalar scaling
    scalars = item["state_scalars"]
    assert scalars[0] == pytest.approx(0.65)  # hp / 100
    assert scalars[1] == pytest.approx(0.30)  # gold / 100
    assert scalars[2] == pytest.approx(0.70)  # level / 10
    assert scalars[3] == pytest.approx(0.20)  # streak / 10
    assert scalars[4] == pytest.approx(0.40)  # stage 4 / 10
    assert scalars[5] == pytest.approx(0.20)  # round 2 / 10
    assert scalars[6] == pytest.approx(0.70)  # unit count / 10
    assert scalars[7] == pytest.approx(0.50)  # item count / 10.0 (NOT / 30.0!)


def test_board_quality_trainer_precompute_and_step() -> None:
    """Verifies that BoardQualityTrainer precomputes fused embeddings and runs a training step."""
    trunk = _create_dummy_trunk(fused_dim=384)
    net = BoardQualityNet(trunk=trunk, freeze_trunk=True, hidden_dim=64)

    trainer = BoardQualityTrainer(
        model=net,
        lr=1e-3,
        device="cpu",
        use_wandb=False,
    )

    vocab = ChampionVocabulary()
    vocab.add_champion("TFT18_Draven")
    item_vocab = ItemVocabulary()
    item_vocab.add_item("TFT_Item_InfinityEdge")
    trait_vocab = TraitVocabulary()
    trait_vocab.add_trait("Set18_Dominator")

    ds = BoardPlacementDataset(data_dir="non_existent_path", vocab=vocab, item_vocab=item_vocab, trait_vocab=trait_vocab)
    for i in range(6):
        ds.samples.append({
            "board": [{"unit": "TFT18_Draven", "star_level": (i % 3) + 1, "location": i, "items": []}],
            "stage_tuple": (4, 1 + i),
            "health": 50.0 + i * 5,
            "level": 7.0,
            "gold": 20.0 + i * 2,
            "unit_count": 7.0,
            "item_count": 3.0,
            "streak": 0.0,
            "target_placement": float(i + 1),
            "target_top4": 1 if i < 4 else 0,
        })

    fused_ds = trainer.precompute_dataset_embeddings(ds, batch_size=2)
    assert len(fused_ds) == 6

    from torch.utils.data import DataLoader
    loader = DataLoader(fused_ds, batch_size=3, shuffle=False)
    metrics = trainer.train_epoch_fused(loader)

    assert "loss" in metrics
    assert "placement_mae" in metrics
    assert "top4_accuracy" in metrics
    assert not torch.isnan(torch.tensor(metrics["loss"]))
    assert metrics["placement_mae"] > 0.0
