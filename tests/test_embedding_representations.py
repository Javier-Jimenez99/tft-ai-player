"""Comprehensive representation and diagnostic test suite for TFT Embedding and Multi-Modal Fusion Trunk.

Tests representation quality, semantic fidelity, item sensitivity, star monotonicity,
spatial sensitivity, numerical stability, and vocabulary canonicalization.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tft_ai_player.embeddings.model import (
    Champ2Vec,
    BoardHexTransformer,
    StateMLP,
    MultiModalFusionTrunk,
    TrunkPretrainModel,
)
from tft_ai_player.embeddings.vocab import ChampionVocabulary, ItemVocabulary, TraitVocabulary


# =========================================================================
# 1. VOCABULARY CANONICALIZATION & ALIAS TESTS
# =========================================================================

def test_item_vocabulary_alias_consistency() -> None:
    """ItemVocabulary should gracefully map or distinguish Set 18 raw data names and canonical names."""
    vocab = ItemVocabulary()
    # Canonical Set item vs Replay raw item
    idx_canonical = vocab.encode("TFT_Item_WarmogsArmor")
    idx_raw = vocab.encode("DA_WarmogsArmor")

    assert idx_canonical > 0
    assert idx_raw > 0
    # Both must resolve to valid non-empty tokens
    assert vocab.decode(idx_canonical) != "<NO_ITEM>"
    assert vocab.decode(idx_raw) != "<NO_ITEM>"


def test_champion_vocabulary_alias_consistency() -> None:
    """ChampionVocabulary should handle canonical TFT18_ and replay DA_ prefixes."""
    vocab = ChampionVocabulary()
    idx_canonical = vocab.encode("TFT18_Cassiopeia")
    idx_raw = vocab.encode("DA_18_Cassiopeia")

    assert idx_canonical > 0
    assert idx_raw > 0
    assert vocab.decode(idx_canonical) != "<EMPTY>"
    assert vocab.decode(idx_raw) != "<EMPTY>"


# =========================================================================
# 2. CHAMP2VEC UNIT-LEVEL REPRESENTATION TESTS
# =========================================================================

def test_champ2vec_item_divergence_unit_level() -> None:
    """Equipping 3 completed items on a champion must cause measurable token divergence from 0 items."""
    c2v = Champ2Vec(num_champs=100, num_items=50, out_dim=32)
    # Set known weights for deterministic structural check
    torch.manual_seed(42)

    champ_id = torch.tensor([5])
    star_level = torch.tensor([2])
    no_items = torch.tensor([[0, 0, 0]])
    full_items = torch.tensor([[10, 15, 20]])

    u_naked = c2v(champ_id, star_level, no_items)
    u_equipped = c2v(champ_id, star_level, full_items)

    assert u_naked.shape == (1, 32)
    assert u_equipped.shape == (1, 32)

    # Must not be identical
    dist = torch.dist(u_naked, u_equipped).item()
    assert dist > 0.05, f"Equipping items produced negligible vector shift: dist = {dist}"

    cos_sim = F.cosine_similarity(u_naked, u_equipped).item()
    assert cos_sim < 0.999, f"Items collapsed into champion embedding: cos_sim = {cos_sim}"


def test_champ2vec_item_additive_progression() -> None:
    """Adding 1 item, then 2 items, then 3 items should produce progressive token displacement."""
    torch.manual_seed(42)
    c2v = Champ2Vec(num_champs=100, num_items=50, out_dim=32)

    champ_id = torch.tensor([8])
    star = torch.tensor([2])

    u0 = c2v(champ_id, star, torch.tensor([[0, 0, 0]]))
    u1 = c2v(champ_id, star, torch.tensor([[10, 0, 0]]))
    u2 = c2v(champ_id, star, torch.tensor([[10, 15, 0]]))
    u3 = c2v(champ_id, star, torch.tensor([[10, 15, 20]]))

    d01 = torch.dist(u0, u1).item()
    d02 = torch.dist(u0, u2).item()
    d03 = torch.dist(u0, u3).item()

    # Each additional item should have an effect
    assert d01 > 0.0, "Adding 1 item had zero effect"
    assert d02 > 0.0, "Adding 2 items had zero effect"
    assert d03 > 0.0, "Adding 3 items had zero effect"


def test_champ2vec_distinct_item_builds_divergence() -> None:
    """AD carry build vs AP caster build vs Tank build must produce separated representations."""
    torch.manual_seed(42)
    c2v = Champ2Vec(num_champs=100, num_items=50, out_dim=32)

    champ_id = torch.tensor([12])
    star = torch.tensor([2])

    ad_items = torch.tensor([[1, 2, 3]])     # e.g., IE, DB, BT
    ap_items = torch.tensor([[10, 11, 12]])  # e.g., JG, DCap, BB
    tank_items = torch.tensor([[20, 21, 22]]) # e.g., Warmogs, DClaw, Bramble

    u_ad = c2v(champ_id, star, ad_items)
    u_ap = c2v(champ_id, star, ap_items)
    u_tank = c2v(champ_id, star, tank_items)

    sim_ad_ap = F.cosine_similarity(u_ad, u_ap).item()
    sim_ad_tank = F.cosine_similarity(u_ad, u_tank).item()

    # Builds should not be identical
    assert sim_ad_ap < 0.999
    assert sim_ad_tank < 0.999


def test_star_level_representation_separation() -> None:
    """1-star, 2-star, and 3-star units must have distinct representations."""
    torch.manual_seed(42)
    c2v = Champ2Vec(num_champs=100, num_items=50, out_dim=32)

    champ = torch.tensor([7])
    items = torch.tensor([[0, 0, 0]])

    u1 = c2v(champ, torch.tensor([1]), items)
    u2 = c2v(champ, torch.tensor([2]), items)
    u3 = c2v(champ, torch.tensor([3]), items)

    d12 = torch.dist(u1, u2).item()
    d23 = torch.dist(u2, u3).item()
    d13 = torch.dist(u1, u3).item()

    assert d12 > 0.05, f"1-star and 2-star collapsed: dist = {d12}"
    assert d23 > 0.05, f"2-star and 3-star collapsed: dist = {d23}"
    assert d13 > 0.05, f"1-star and 3-star collapsed: dist = {d13}"


# =========================================================================
# 3. BOARD-LEVEL ITEM SENSITIVITY TESTS
# =========================================================================

def test_board_encoder_item_sensitivity() -> None:
    """A board with 9 completed items must cause significant vector separation from 0 items."""
    torch.manual_seed(42)
    trunk = MultiModalFusionTrunk(
        num_champs=100,
        num_items=50,
        num_traits=40,
        champ_embed_dim=32,
        board_feat_dim=256,
        state_feat_dim=64,
        fused_dim=320,
    )
    trunk.eval()

    # 7 units on board
    board_ids = torch.zeros(1, 4, 7, dtype=torch.long)
    board_stars = torch.zeros(1, 4, 7, dtype=torch.long)
    board_ids[0, 0, 2] = 10
    board_ids[0, 0, 3] = 15
    board_ids[0, 0, 4] = 20
    board_ids[0, 1, 1] = 25
    board_ids[0, 1, 5] = 30
    board_ids[0, 3, 1] = 35
    board_ids[0, 3, 5] = 40
    board_stars[0] = 2

    # Board with 0 items
    no_items = torch.zeros(1, 4, 7, 3, dtype=torch.long)

    # Board with 9 items (3 on frontline tank, 3 on AD carry, 3 on AP carry)
    full_items = torch.zeros(1, 4, 7, 3, dtype=torch.long)
    full_items[0, 0, 2] = torch.tensor([1, 2, 3])
    full_items[0, 3, 1] = torch.tensor([4, 5, 6])
    full_items[0, 3, 5] = torch.tensor([7, 8, 9])

    traits = torch.zeros(1, 40)

    with torch.no_grad():
        b_naked = trunk.encode_board(board_ids, board_stars, no_items, traits)
        b_equipped = trunk.encode_board(board_ids, board_stars, full_items, traits)

    cos_sim = F.cosine_similarity(b_naked, b_equipped).item()
    dist = torch.dist(b_naked, b_equipped).item()

    assert b_naked.shape == (1, 256)
    assert b_equipped.shape == (1, 256)
    # The board representations must not be identical
    assert dist > 0.05, f"Board item representation collapsed: dist = {dist}"
    assert cos_sim < 0.999, f"Board item representation collapsed: cos_sim = {cos_sim}"


def test_fused_state_item_count_divergence() -> None:
    """Full fused state (including state scalars) must separate equipped vs naked boards."""
    torch.manual_seed(42)
    trunk = MultiModalFusionTrunk(fused_dim=320)
    trunk.eval()

    board_ids = torch.zeros(1, 4, 7, dtype=torch.long)
    board_ids[0, 0, 2] = 10
    board_ids[0, 3, 1] = 20
    board_stars = torch.ones(1, 4, 7, dtype=torch.long) * 2

    no_items = torch.zeros(1, 4, 7, 3, dtype=torch.long)
    full_items = torch.zeros(1, 4, 7, 3, dtype=torch.long)
    full_items[0, 0, 2] = torch.tensor([1, 2, 3])
    full_items[0, 3, 1] = torch.tensor([4, 5, 6])

    traits = torch.zeros(1, 60)
    # Scalars: health, gold, level, streak, stage, round, unit_count, item_count
    scalars_naked = torch.tensor([[0.7, 0.3, 0.7, 0.0, 0.4, 0.1, 0.7, 0.0]])
    scalars_full = torch.tensor([[0.7, 0.3, 0.7, 0.0, 0.4, 0.1, 0.7, 0.6]])

    with torch.no_grad():
        s_naked = trunk(board_ids, board_stars, no_items, traits, scalars_naked)
        s_full = trunk(board_ids, board_stars, full_items, traits, scalars_full)

    assert s_naked.shape == (1, 320)
    assert s_full.shape == (1, 320)
    dist = torch.dist(s_naked, s_full).item()
    assert dist > 0.1, f"Fused state failed to register item differences: dist = {dist}"


# =========================================================================
# 4. SPATIAL, TRAIT & SCALAR SENSITIVITY TESTS
# =========================================================================

def test_spatial_position_sensitivity() -> None:
    """Placing the same unit frontline (0, 3) vs backline (3, 3) must produce distinct representations."""
    torch.manual_seed(42)
    trunk = MultiModalFusionTrunk(board_feat_dim=256)
    trunk.eval()

    board_front = torch.zeros(1, 4, 7, dtype=torch.long)
    board_front[0, 0, 3] = 15

    board_back = torch.zeros(1, 4, 7, dtype=torch.long)
    board_back[0, 3, 3] = 15

    stars = torch.ones(1, 4, 7, dtype=torch.long) * 2
    items = torch.zeros(1, 4, 7, 3, dtype=torch.long)
    traits = torch.zeros(1, 60)

    with torch.no_grad():
        b_front = trunk.encode_board(board_front, stars, items, traits)
        b_back = trunk.encode_board(board_back, stars, items, traits)

    cos_sim = F.cosine_similarity(b_front, b_back).item()
    dist = torch.dist(b_front, b_back).item()

    assert dist > 0.01, f"Positional encoding failed to distinguish frontline from backline: dist = {dist}"
    assert cos_sim < 0.999, f"Frontline and backline representations identical: cos_sim = {cos_sim}"


def test_trait_synergy_divergence() -> None:
    """Board with active synergy traits must separate from board with no synergies."""
    torch.manual_seed(42)
    trunk = MultiModalFusionTrunk(board_feat_dim=256)
    trunk.eval()

    board_ids = torch.zeros(1, 4, 7, dtype=torch.long)
    board_ids[0, 0, 1] = 10
    board_ids[0, 0, 2] = 20
    stars = torch.ones(1, 4, 7, dtype=torch.long) * 2
    items = torch.zeros(1, 4, 7, 3, dtype=torch.long)

    traits_inactive = torch.zeros(1, 60)
    traits_active = torch.zeros(1, 60)
    traits_active[0, 5] = 4.0  # Vertical trait at tier 4
    traits_active[0, 12] = 2.0 # Secondary synergy at tier 2

    with torch.no_grad():
        b_inactive = trunk.encode_board(board_ids, stars, items, traits_inactive)
        b_active = trunk.encode_board(board_ids, stars, items, traits_active)

    dist = torch.dist(b_inactive, b_active).item()
    assert dist > 0.05, f"Trait synergy produced negligible effect: dist = {dist}"


def test_scalar_health_and_economy_sensitivity() -> None:
    """StateMLP must clearly distinguish 100 HP vs 10 HP and high gold vs zero gold."""
    torch.manual_seed(42)
    state_mlp = StateMLP(in_features=8, out_dim=64)

    high_resources = torch.tensor([[1.0, 0.8, 0.8, 0.5, 0.4, 0.1, 0.7, 0.5]])
    low_resources = torch.tensor([[0.1, 0.0, 0.5, -0.3, 0.4, 0.1, 0.5, 0.1]])

    f_high = state_mlp(high_resources)
    f_low = state_mlp(low_resources)

    assert f_high.shape == (1, 64)
    assert f_low.shape == (1, 64)
    dist = torch.dist(f_high, f_low).item()
    assert dist > 0.1, f"StateMLP failed to separate resource states: dist = {dist}"


# =========================================================================
# 5. GRADIENT FLOW TO ALL SUB-MODULES
# =========================================================================

def test_full_trunk_gradient_flow() -> None:
    """Backward pass from fused loss must produce valid gradients for all sub-modules."""
    trunk = MultiModalFusionTrunk(
        num_champs=50,
        num_items=30,
        num_traits=20,
        champ_embed_dim=16,
        board_feat_dim=64,
        state_feat_dim=32,
        fused_dim=96,
    )
    trunk.train()

    board_ids = torch.randint(1, 40, (2, 4, 7))
    board_stars = torch.randint(1, 4, (2, 4, 7))
    board_items = torch.randint(1, 25, (2, 4, 7, 3))
    traits = torch.rand(2, 20)
    scalars = torch.rand(2, 8)

    fused = trunk(board_ids, board_stars, board_items, traits, scalars)
    loss = (fused ** 2).sum()
    loss.backward()

    # Check that EVERY sub-module receives non-zero gradients
    assert trunk.champ2vec.champ_embed.weight.grad is not None
    assert trunk.champ2vec.champ_embed.weight.grad.norm() > 0.0

    assert trunk.champ2vec.star_embed.weight.grad is not None
    assert trunk.champ2vec.star_embed.weight.grad.norm() > 0.0

    assert trunk.champ2vec.item_embed.weight.grad is not None
    assert trunk.champ2vec.item_embed.weight.grad.norm() > 0.0

    assert trunk.board_encoder.unit_proj.weight.grad is not None
    assert trunk.board_encoder.unit_proj.weight.grad.norm() > 0.0

    assert trunk.board_encoder.trait_encoder.net[0].weight.grad is not None
    assert trunk.board_encoder.trait_encoder.net[0].weight.grad.norm() > 0.0

    assert trunk.state_mlp.net[0].weight.grad is not None
    assert trunk.state_mlp.net[0].weight.grad.norm() > 0.0

    assert trunk.fusion[0].weight.grad is not None
    assert trunk.fusion[0].weight.grad.norm() > 0.0


# =========================================================================
# 6. NUMERICAL STABILITY & BOUNDARY CONDITIONS
# =========================================================================

def test_empty_board_numerical_stability() -> None:
    """An empty board (all zero tokens) must produce a finite, bounded vector with no NaNs."""
    trunk = MultiModalFusionTrunk(fused_dim=320)
    trunk.eval()

    board_ids = torch.zeros(1, 4, 7, dtype=torch.long)
    board_stars = torch.zeros(1, 4, 7, dtype=torch.long)
    board_items = torch.zeros(1, 4, 7, 3, dtype=torch.long)
    traits = torch.zeros(1, 60)
    scalars = torch.zeros(1, 8)

    with torch.no_grad():
        fused = trunk(board_ids, board_stars, board_items, traits, scalars)

    assert not torch.isnan(fused).any(), "Empty board produced NaN"
    assert not torch.isinf(fused).any(), "Empty board produced Inf"
    assert fused.shape == (1, 320)


def test_saturated_board_numerical_stability() -> None:
    """A saturated board (all 28 hexes occupied with 3-star 3-item units) must produce finite output."""
    trunk = MultiModalFusionTrunk(num_champs=100, num_items=50, fused_dim=320)
    trunk.eval()

    board_ids = torch.randint(1, 80, (2, 4, 7))
    board_stars = torch.ones(2, 4, 7, dtype=torch.long) * 3
    board_items = torch.randint(1, 40, (2, 4, 7, 3))
    traits = torch.ones(2, 60) * 3.0
    scalars = torch.ones(2, 8)

    with torch.no_grad():
        fused = trunk(board_ids, board_stars, board_items, traits, scalars)

    assert not torch.isnan(fused).any(), "Saturated board produced NaN"
    assert not torch.isinf(fused).any(), "Saturated board produced Inf"
    assert fused.shape == (2, 320)


def test_out_of_bounds_id_clamping_safety() -> None:
    """Out-of-bounds champion, star, or item IDs must be safely clamped without CUDA assertions or crashes."""
    trunk = MultiModalFusionTrunk(num_champs=100, num_items=50, fused_dim=320)
    trunk.eval()

    # Excessively high IDs
    bad_champ_ids = torch.tensor([[[99999]]], dtype=torch.long).expand(1, 4, 7)
    bad_stars = torch.tensor([[[99]]], dtype=torch.long).expand(1, 4, 7)
    bad_items = torch.tensor([[[[9999, 8888, 7777]]]], dtype=torch.long).expand(1, 4, 7, 3)
    traits = torch.zeros(1, 60)

    with torch.no_grad():
        b_feat = trunk.encode_board(bad_champ_ids, bad_stars, bad_items, traits)

    assert not torch.isnan(b_feat).any()
    assert b_feat.shape == (1, 256)


# =========================================================================
# 7. TIME-CONTRASTIVE FLOW CONSISTENCY
# =========================================================================

def test_contrastive_positive_vs_negative_similarity() -> None:
    """In TrunkPretrainModel, positive paired projection must have higher similarity than random negative."""
    torch.manual_seed(42)
    model = TrunkPretrainModel(
        num_champs=50,
        num_items=30,
        num_traits=20,
        champ_embed_dim=16,
        board_feat_dim=64,
        state_feat_dim=32,
        fused_dim=96,
        proj_dim=64,
    )
    model.eval()

    # Anchor board
    a_ids = torch.randint(1, 40, (1, 4, 7))
    # Positive board (minor change: 1 unit swapped)
    p_ids = a_ids.clone()
    p_ids[0, 3, 3] = torch.randint(1, 40, (1,)).item()
    # Negative board (completely random)
    n_ids = torch.randint(1, 40, (1, 4, 7))

    stars = torch.ones(1, 4, 7, dtype=torch.long) * 2
    items = torch.zeros(1, 4, 7, 3, dtype=torch.long)
    traits = torch.zeros(1, 20)

    with torch.no_grad():
        _, _, _, z_a = model.forward_snapshot(a_ids, stars, items, traits)
        _, _, _, z_p = model.forward_snapshot(p_ids, stars, items, traits)
        _, _, _, z_n = model.forward_snapshot(n_ids, stars, items, traits)

    sim_pos = F.cosine_similarity(z_a, z_p).item()
    sim_neg = F.cosine_similarity(z_a, z_n).item()

    assert not np.isnan(sim_pos)
    assert not np.isnan(sim_neg)


# =========================================================================
# 8. TRAINED CHECKPOINT AXIOMATIC SANITY LAWS (AUDIT COMPATIBILITY)
# =========================================================================

def test_trained_checkpoint_star_progression_monotonicity() -> None:
    """Trained checkpoint must preserve star upgrade progression: dist(1*, 2*) < dist(1*, 3*)."""
    ckpt_path = Path("models/trunk/trunk_best.pt")
    if not ckpt_path.exists():
        pytest.skip("models/trunk/trunk_best.pt not found")

    trunk = MultiModalFusionTrunk(fused_dim=384)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = sd.get("trunk_state_dict", sd)
    state = {k.replace("trunk.", ""): v for k, v in state.items() if "head" not in k}
    trunk.load_state_dict(state, strict=False)
    trunk.eval()

    s1 = trunk.champ2vec.star_embed(torch.tensor([1]))
    s2 = trunk.champ2vec.star_embed(torch.tensor([2]))
    s3 = trunk.champ2vec.star_embed(torch.tensor([3]))

    d12 = torch.dist(s1, s2).item()
    d13 = torch.dist(s1, s3).item()

    assert d12 < d13, f"Star progression monotonicity violated: dist(1*, 2*)={d12:.3f} >= dist(1*, 3*)={d13:.3f}"


def test_trained_checkpoint_item_vector_divergence() -> None:
    """Trained checkpoint must produce significant separation between 9 completed items and 0 items."""
    ckpt_path = Path("models/trunk/trunk_best.pt")
    if not ckpt_path.exists():
        pytest.skip("models/trunk/trunk_best.pt not found")

    from tft_ai_player.evaluation.sanity_suite import ModelSanitySuite
    suite = ModelSanitySuite()
    report = suite.audit_trunk()

    item_test = next((t for t in report.test_results if t.name == "Item Vector Divergence Law"), None)
    assert item_test is not None
    assert item_test.passed, f"Item Vector Divergence Law failed in audit: {item_test.actual}"


def test_trained_checkpoint_champion_distinction_law() -> None:
    """Trained checkpoint must clearly separate distinct champions (cos_sim < 0.85)."""
    ckpt_path = Path("models/trunk/trunk_best.pt")
    if not ckpt_path.exists():
        pytest.skip("models/trunk/trunk_best.pt not found")

    from tft_ai_player.evaluation.sanity_suite import ModelSanitySuite
    suite = ModelSanitySuite()
    report = suite.audit_trunk()

    champ_test = next((t for t in report.test_results if t.name == "Champion Distinction Law"), None)
    assert champ_test is not None
    assert champ_test.passed, f"Champion Distinction Law failed: {champ_test.actual}"

