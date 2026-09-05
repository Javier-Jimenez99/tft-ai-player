from __future__ import annotations

import pytest
import torch

from tft_ai_player.board_evaluator.model import BoardQualityNet
from tft_ai_player.embeddings.model import MultiModalFusionTrunk


def test_board_quality_net_forward_and_bounds() -> None:
    trunk = MultiModalFusionTrunk(
        num_champs=100,
        num_items=50,
        num_traits=20,
        fused_dim=320,
        num_layers=1,
    )
    net = BoardQualityNet(trunk=trunk, freeze_trunk=True, hidden_dim=64)

    bs = 4
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
    # Check placement predictions are bounded in [1.0, 8.0]
    assert (pred_place >= 1.0).all()
    assert (pred_place <= 8.0).all()


def test_board_quality_net_evaluate_tensors() -> None:
    trunk = MultiModalFusionTrunk(
        num_champs=100,
        num_items=50,
        num_traits=20,
        fused_dim=320,
        num_layers=1,
    )
    net = BoardQualityNet(trunk=trunk, freeze_trunk=True, hidden_dim=64)

    champs = torch.randint(0, 100, (28,))
    q_score, exp_place, top4_prob = net.evaluate_board_tensors(champs)

    assert 1.0 <= exp_place <= 8.0
    assert 0.0 <= top4_prob <= 1.0
    assert isinstance(q_score, float)
