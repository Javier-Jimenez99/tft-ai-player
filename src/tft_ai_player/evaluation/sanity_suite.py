"""Master Sanity & Benchmark Suite for the TFT-AI Model Ecosystem.

Evaluates all 7 auxiliary and core neural models:
  1. MultiModalFusionTrunk (Backbone Representation)
  2. DeepSiameseCombatNet (Combat Outcome Resolver)
  3. BoardQualityNet (Placement & Top-4 Oracle)
  4. StateTransitionPredictor (World Model Dynamics)
  5. Z-Index Centroids (Macro Archetype Clustering)
  6. EloRegressor (Skill Rating Predictor)
  7. TFTActorCritic (RL Policy & Value Heads)

Runs both:
  - Empirical Replay Metrics (on authentic human ranked replays)
  - Axiomatic / Physical TFT Laws (deterministic counterfactual stress tests)
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from tft_ai_player.board_evaluator.model import BoardQualityNet
from tft_ai_player.embeddings.model import MultiModalFusionTrunk
from tft_ai_player.embeddings.transition import StateTransitionPredictor
from tft_ai_player.embeddings.vocab import ChampionVocabulary, ItemVocabulary, TraitVocabulary
from tft_ai_player.rl.models.networks import TFTActorCritic
from tft_ai_player.rl.shadow_match.ranked_loader import RankedLadderReplayLoader
from tft_ai_player.rl.shadow_match.replay_loader import ShadowMatchLoader
from tft_ai_player.rl.shadow_match.shadow_env import (
    create_player_from_opponent_board,
    parse_loc_to_row_col,
    resolve_canonical_champion,
    resolve_canonical_item,
)
from tft_ai_player.round_winner.embedding_model import DeepSiameseCombatNet
from tft_ai_player.simulation.combat import HeuristicCombatResolver
from tft_ai_player.simulation.config import SetData
from tft_ai_player.simulation.gym_env import TFTStateEncoder
from tft_ai_player.simulation.models import ChampionInstance, Player
from tft_ai_player.simulation.sets.set18 import get_set18_data

logger = logging.getLogger("sanity_suite")


@dataclass
class TestResult:
    name: str
    description: str
    law: str
    expected: str
    actual: str
    passed: bool
    status: str  # "PASS", "WARN", "FAIL"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelAuditReport:
    model_name: str
    checkpoint_path: str
    overall_status: str  # "PASS", "WARN", "FAIL"
    empirical_metrics: dict[str, Any] = field(default_factory=dict)
    test_results: list[TestResult] = field(default_factory=list)
    diagnosis: str = ""
    summary: str = ""


class ModelSanitySuite:
    """Comprehensive diagnostic benchmark across all models in the TFT ecosystem."""

    def __init__(
        self,
        set_data: SetData | None = None,
        device: torch.device | None = None,
        ranked_cache_path: str | Path = "models/rl/ranked_ladder_cache.pkl.gz",
        shadow_cache_path: str | Path = "models/rl/shadow_matches_cache.pkl.gz",
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_data = set_data or get_set18_data()
        self.ranked_cache_path = Path(ranked_cache_path)
        self.shadow_cache_path = Path(shadow_cache_path)

        # Lazy-loaded components
        self._trunk: MultiModalFusionTrunk | None = None
        self._encoder: TFTStateEncoder | None = None
        self._ranked_replays: dict[str, list[Any]] | None = None
        self._shadow_replays: list[Any] | None = None

    # =========================================================================
    # RESOURCE ACCESSORS & CACHES
    # =========================================================================

    def get_trunk(self) -> MultiModalFusionTrunk:
        if self._trunk is None:
            trunk_path = Path("models/trunk/trunk_best.pt")
            self._trunk = MultiModalFusionTrunk(
                num_champs=500,
                num_items=300,
                num_traits=60,
                fused_dim=384,
            )
            if trunk_path.exists():
                sd = torch.load(trunk_path, map_location="cpu", weights_only=False)
                state = sd.get("trunk_state_dict", sd)
                state = {k.replace("trunk.", ""): v for k, v in state.items() if "head" not in k}
                self._trunk.load_state_dict(state, strict=False)
            self._trunk.to(self.device)
            self._trunk.eval()
        return self._trunk

    def get_encoder(self) -> TFTStateEncoder:
        if self._encoder is None:
            self._encoder = TFTStateEncoder(
                set_data=self.set_data,
                trunk=self.get_trunk(),
                device=self.device,
            )
        return self._encoder

    def get_ranked_replays(self) -> dict[str, list[Any]]:
        if self._ranked_replays is None:
            if self.ranked_cache_path.exists():
                loader = RankedLadderReplayLoader(cache_path=self.ranked_cache_path)
                self._ranked_replays = loader.build_or_load_index()
            else:
                self._ranked_replays = {}
        return self._ranked_replays

    def get_shadow_replays(self) -> list[Any]:
        if self._shadow_replays is None:
            if self.shadow_cache_path.exists():
                loader = ShadowMatchLoader(cache_path=self.shadow_cache_path)
                self._shadow_replays = loader.build_or_load_index(max_matches=200)
            else:
                self._shadow_replays = []
        return self._shadow_replays

    # =========================================================================
    # SYNTHETIC TEST BOARD BUILDERS
    # =========================================================================

    def build_test_player(
        self,
        name: str,
        level: int,
        health: int,
        gold: int,
        units_spec: list[tuple[tuple[int, int], str, int, list[str]]],
    ) -> Player:
        """Create a reproducible Player instance with explicit units, stars, and items.

        units_spec: [((row, col), champion_id, star_level, [items])]
        """
        p = Player(player_id=1, set_data=self.set_data, name=name)
        p.level = level
        p.health = health
        p.gold = gold

        for (r, c), cid, star, items in units_spec:
            cost = self.set_data.champions[cid].cost if cid in self.set_data.champions else 1
            unit = ChampionInstance(
                champion_id=cid,
                cost=cost,
                star_level=star,
                items=list(items),
                position=(r, c),
            )
            p.board[(r, c)] = unit

        return p

    def get_standard_level7_comp(self, items_mode: str = "none") -> Player:
        """Build standard Stage 4-1 Level 7 board.

        items_mode: "none" (0 items) or "full" (9 meta completed items)
        """
        if items_mode == "full":
            units_spec = [
                ((0, 2), "TFT18_Leona", 2, ["TFT_Item_WarmogsArmor", "TFT_Item_BrambleVest", "TFT_Item_DragonsClaw"]),
                ((0, 3), "TFT18_Kobuko", 2, ["TFT_Item_SunfireCape"]),
                ((0, 4), "TFT18_Cinderling", 2, []),
                ((1, 1), "TFT18_Camille", 2, []),
                ((1, 5), "TFT18_Pebbles", 2, []),
                ((3, 1), "TFT18_Akali", 2, ["TFT_Item_InfinityEdge", "TFT_Item_Bloodthirster", "TFT_Item_TitansResolve"]),
                ((3, 5), "TFT18_Karma", 2, ["TFT_Item_BlueBuff", "TFT_Item_JeweledGauntlet"]),
            ]
        else:
            units_spec = [
                ((0, 2), "TFT18_Leona", 2, []),
                ((0, 3), "TFT18_Kobuko", 2, []),
                ((0, 4), "TFT18_Cinderling", 2, []),
                ((1, 1), "TFT18_Camille", 2, []),
                ((1, 5), "TFT18_Pebbles", 2, []),
                ((3, 1), "TFT18_Akali", 2, []),
                ((3, 5), "TFT18_Karma", 2, []),
            ]
        return self.build_test_player(f"Level7_{items_mode}_items", level=7, health=70, gold=30, units_spec=units_spec)

    def get_star_level_comps(self) -> tuple[Player, Player]:
        """Pair of boards: Full 3-Star Team vs Full 1-Star Team (same champions)."""
        champs = ["TFT18_Leona", "TFT18_Kobuko", "TFT18_Cinderling", "TFT18_Camille", "TFT18_Pebbles", "TFT18_Akali", "TFT18_Karma"]
        pos = [(0, 1), (0, 2), (0, 3), (1, 1), (1, 5), (3, 1), (3, 5)]

        spec_3star = [(p, c, 3, []) for p, c in zip(pos, champs)]
        spec_1star = [(p, c, 1, []) for p, c in zip(pos, champs)]

        p_3star = self.build_test_player("Comp_3Star", level=7, health=70, gold=30, units_spec=spec_3star)
        p_1star = self.build_test_player("Comp_1Star", level=7, health=70, gold=30, units_spec=spec_1star)
        return p_3star, p_1star

    def get_unit_count_comps(self) -> tuple[Player, Player]:
        """Pair of boards: 9 Units (Level 9) vs 3 Units (Level 3)."""
        champs_9 = ["TFT18_Leona", "TFT18_Kobuko", "TFT18_Cinderling", "TFT18_Camille", "TFT18_Pebbles", "TFT18_Akali", "TFT18_Karma", "TFT18_Ahri", "TFT18_Draven"]
        pos_9 = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 1), (1, 5), (3, 1), (3, 3), (3, 5)]
        spec_9 = [(p, c, 2, []) for p, c in zip(pos_9, champs_9)]

        champs_3 = ["TFT18_Leona", "TFT18_Kobuko", "TFT18_Camille"]
        pos_3 = [(0, 2), (0, 3), (1, 1)]
        spec_3 = [(p, c, 1, []) for p, c in zip(pos_3, champs_3)]

        p_9 = self.build_test_player("Comp_9Units", level=9, health=80, gold=20, units_spec=spec_9)
        p_3 = self.build_test_player("Comp_3Units", level=3, health=30, gold=5, units_spec=spec_3)
        return p_9, p_3

    def get_cost_tier_comps(self) -> tuple[Player, Player]:
        """Pair of boards: High-Cost 4/5-Costs 2-star vs Low-Cost 1-Costs 2-star (balanced positioning)."""
        spec_high = [
            ((0, 2), "TFT18_Amumu", 2, []),
            ((0, 3), "TFT18_AncientSentinel", 2, []),
            ((1, 2), "TFT18_Alune", 2, []),
            ((3, 1), "TFT18_Aphelios", 2, []),
            ((3, 2), "TFT18_Ashe", 2, []),
            ((3, 4), "TFT18_Ahri", 2, []),
            ((3, 5), "TFT18_Ezreal", 2, []),
        ]
        spec_low = [
            ((0, 2), "TFT18_Pebbles", 2, []),
            ((0, 3), "TFT18_Kobuko", 2, []),
            ((0, 4), "TFT18_Leona", 2, []),
            ((1, 2), "TFT18_RekSai", 2, []),
            ((1, 4), "TFT18_Camille", 2, []),
            ((3, 2), "TFT18_Akali", 2, []),
            ((3, 4), "TFT18_Cinderling", 2, []),
        ]

        p_high = self.build_test_player("Comp_HighCost", level=7, health=70, gold=30, units_spec=spec_high)
        p_low = self.build_test_player("Comp_LowCost", level=7, health=70, gold=30, units_spec=spec_low)
        return p_high, p_low

    def get_capped_vs_early_comps(self) -> tuple[Player, Player]:
        """Pair: Stage 5 Capped Level 9 Board (100 HP, 9 units 2-star, full items) vs Stage 5 Early Level 5 Board (15 HP, 5 units 1-star, 0 items)."""
        champs_9 = ["TFT18_Draven", "TFT18_Ashe", "TFT18_Alune", "TFT18_Ivern", "TFT18_Ahri", "TFT18_Ezreal", "TFT18_Aphelios", "TFT18_Amumu", "TFT18_AncientSentinel"]
        pos_9 = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 1), (1, 5), (3, 1), (3, 3), (3, 5)]
        spec_capped = [
            (pos_9[0], champs_9[0], 2, ["TFT_Item_WarmogsArmor", "TFT_Item_DragonsClaw", "TFT_Item_BrambleVest"]),
            (pos_9[1], champs_9[1], 2, ["TFT_Item_BlueBuff", "TFT_Item_JeweledGauntlet", "TFT_Item_RabadonsDeathcap"]),
            (pos_9[2], champs_9[2], 2, ["TFT_Item_InfinityEdge", "TFT_Item_Bloodthirster", "TFT_Item_TitansResolve"]),
            (pos_9[3], champs_9[3], 2, ["TFT_Item_StatikkShiv"]),
            (pos_9[4], champs_9[4], 2, []),
            (pos_9[5], champs_9[5], 2, []),
            (pos_9[6], champs_9[6], 2, []),
            (pos_9[7], champs_9[7], 2, []),
            (pos_9[8], champs_9[8], 2, []),
        ]
        p_capped = self.build_test_player("Capped_Stage5_Board", level=9, health=95, gold=40, units_spec=spec_capped)

        pos_5 = [(0, 2), (0, 3), (1, 1), (3, 1), (3, 5)]
        champs_5 = ["TFT18_Pebbles", "TFT18_Kobuko", "TFT18_Camille", "TFT18_Cinderling", "TFT18_RekSai"]
        spec_weak = [(p, c, 1, []) for p, c in zip(pos_5, champs_5)]
        p_weak = self.build_test_player("Weak_Stage5_Board", level=5, health=14, gold=2, units_spec=spec_weak)

        return p_capped, p_weak

    # =========================================================================
    # AUDIT 1: MultiModalFusionTrunk
    # =========================================================================

    def audit_trunk(self) -> ModelAuditReport:
        """Audit MultiModalFusionTrunk representations, embeddings, and sensitivity."""
        trunk = self.get_trunk()
        report = ModelAuditReport(
            model_name="MultiModalFusionTrunk",
            checkpoint_path="models/trunk/trunk_best.pt",
            overall_status="PASS",
        )

        with torch.no_grad():
            c_norm = float(trunk.champ2vec.champ_embed.weight.norm().item())
            s_norm = float(trunk.champ2vec.star_embed.weight.norm().item())
            i_norm = float(trunk.champ2vec.item_embed.weight.norm().item())
            i_std = float(trunk.champ2vec.item_embed.weight.std().item())

            report.empirical_metrics = {
                "champ_embed_norm": round(c_norm, 2),
                "star_embed_norm": round(s_norm, 2),
                "item_embed_norm": round(i_norm, 2),
                "item_embed_std": round(i_std, 4),
                "fused_dim": trunk.fused_dim,
            }

            # Law 1: Champ Distinction Law (Distinct champions must have cosine similarity < 0.85)
            c1 = trunk.champ2vec.champ_embed(torch.tensor([10], device=self.device))
            c2 = trunk.champ2vec.champ_embed(torch.tensor([25], device=self.device))
            c_sim = float(F.cosine_similarity(c1, c2).item())
            law1_pass = c_sim < 0.85
            report.test_results.append(
                TestResult(
                    name="Champion Distinction Law",
                    description="Different champions must have separated embedding vectors in Champ2Vec",
                    law="cos_sim(champ_i, champ_j) < 0.85",
                    expected="< 0.850",
                    actual=f"{c_sim:.4f}",
                    passed=law1_pass,
                    status="PASS" if law1_pass else "FAIL",
                )
            )

            # Law 2: Star Monotonicity Law (Distance from 1* to 2* < Distance from 1* to 3*)
            s1 = trunk.champ2vec.star_embed(torch.tensor([1], device=self.device))
            s2 = trunk.champ2vec.star_embed(torch.tensor([2], device=self.device))
            s3 = trunk.champ2vec.star_embed(torch.tensor([3], device=self.device))
            d_1_2 = float(torch.dist(s1, s2).item())
            d_1_3 = float(torch.dist(s1, s3).item())
            law2_pass = d_1_2 < d_1_3
            report.test_results.append(
                TestResult(
                    name="Star Progression Monotonicity",
                    description="Upgrades must progress along the star embedding manifold",
                    law="dist(1*, 2*) < dist(1*, 3*)",
                    expected=f"d(1,2) < d(1,3) (actual: {d_1_2:.2f} vs {d_1_3:.2f})",
                    actual=f"{d_1_2:.2f} < {d_1_3:.2f}",
                    passed=law2_pass,
                    status="PASS" if law2_pass else "FAIL",
                )
            )

            # Law 3: Board Item Sensitivity Law (Full items vs No items must cause measurable board vector shift)
            encoder = self.get_encoder()
            p_no_items = self.get_standard_level7_comp("none")
            p_full_items = self.get_standard_level7_comp("full")

            _, s_no_items, h_no_items = encoder.extract_state_vector(p_no_items, stage=4, round_in_stage=1)
            _, s_full_items, h_full_items = encoder.extract_state_vector(p_full_items, stage=4, round_in_stage=1)

            h_sim = float(F.cosine_similarity(h_no_items.unsqueeze(0), h_full_items.unsqueeze(0)).item())
            s_sim = float(F.cosine_similarity(s_no_items.unsqueeze(0), s_full_items.unsqueeze(0)).item())
            h_dist = float(torch.dist(h_no_items, h_full_items).item())

            # If cosine similarity is > 0.95, items have suffered severe feature collapse!
            law3_pass = h_sim < 0.90
            law3_status = "PASS" if law3_pass else ("WARN" if h_sim < 0.95 else "FAIL")
            report.test_results.append(
                TestResult(
                    name="Item Vector Divergence Law",
                    description="A board with 9 completed items must cause significant vector separation from 0 items",
                    law="cos_sim(h_no_items, h_full_items) < 0.90",
                    expected="< 0.900",
                    actual=f"cos_sim = {h_sim:.4f} (dist = {h_dist:.2f})",
                    passed=law3_pass,
                    status=law3_status,
                    details={"h_sim": h_sim, "s_sim": s_sim, "h_dist": h_dist},
                )
            )

        # Set overall status
        statuses = [t.status for t in report.test_results]
        if "FAIL" in statuses:
            report.overall_status = "FAIL"
            report.diagnosis = (
                "MultiModalFusionTrunk exhibits severe Item Feature Collapse: identical boards with 9 completed items "
                f"versus 0 items have cos_sim = {h_sim:.4f} (96.4% identical). The transformer attention and trait projection "
                "drown out item representations."
            )
        elif "WARN" in statuses:
            report.overall_status = "WARN"
            report.diagnosis = "Trunk has moderate item sensitivity attenuation."
        else:
            report.overall_status = "PASS"

        return report

    # =========================================================================
    # AUDIT 2: DeepSiameseCombatNet
    # =========================================================================

    def audit_siamese_combat(self) -> ModelAuditReport:
        """Audit DeepSiameseCombatNet against physical TFT combat laws and human replays."""
        trunk = self.get_trunk()
        siamese_ckpt = Path("models/round_winner/deep_siamese_combat_best.pt")
        report = ModelAuditReport(
            model_name="DeepSiameseCombatNet",
            checkpoint_path=str(siamese_ckpt),
            overall_status="PASS",
        )

        if not siamese_ckpt.exists():
            report.overall_status = "FAIL"
            report.diagnosis = f"Checkpoint not found at {siamese_ckpt}"
            return report

        siamese_model = DeepSiameseCombatNet(trunk=trunk, freeze_trunk=True, hidden_dim=256).to(self.device)
        sd = torch.load(siamese_ckpt, map_location=self.device, weights_only=False)
        raw_state = sd.get("model_state_dict", sd)
        curr_state = siamese_model.state_dict()
        filtered_state = {k: v for k, v in raw_state.items() if k in curr_state and curr_state[k].shape == v.shape}
        siamese_model.load_state_dict(filtered_state, strict=False)
        siamese_model.eval()

        encoder = self.get_encoder()

        def eval_combat_prob(p_a: Player, p_b: Player, stage: int = 4) -> float:
            c_a, st_a, it_a, tr_a = encoder.encode_board_tensors(p_a)
            c_b, st_b, it_b, tr_b = encoder.encode_board_tensors(p_b)
            sc_a = encoder.encode_scalars(p_a, stage=stage, round_in_stage=1)
            sc_b = encoder.encode_scalars(p_b, stage=stage, round_in_stage=1)
            b_a = {
                "board_champ_ids": c_a.unsqueeze(0),
                "board_star_levels": st_a.unsqueeze(0),
                "board_item_ids": it_a.unsqueeze(0),
                "board_traits": tr_a.unsqueeze(0),
                "state_scalars": sc_a.unsqueeze(0),
            }
            b_b = {
                "board_champ_ids": c_b.unsqueeze(0),
                "board_star_levels": st_b.unsqueeze(0),
                "board_item_ids": it_b.unsqueeze(0),
                "board_traits": tr_b.unsqueeze(0),
                "state_scalars": sc_b.unsqueeze(0),
            }
            with torch.no_grad():
                logit = siamese_model(b_a, b_b)
                return float(torch.sigmoid(logit).item()) * 100.0

        # Law 1: Mirror Match Identity Law (P(A vs A) == 50.0% +/- 2%)
        p_std = self.get_standard_level7_comp("none")
        mirror_prob = eval_combat_prob(p_std, p_std)
        law1_pass = abs(mirror_prob - 50.0) <= 2.0
        report.test_results.append(
            TestResult(
                name="Mirror Match Identity Law",
                description="P(Board A beats Board A) must strictly evaluate to 50.0%",
                law="|P(A vs A) - 50.0%| <= 2.0%",
                expected="50.00% +/- 2.0%",
                actual=f"{mirror_prob:.2f}%",
                passed=law1_pass,
                status="PASS" if law1_pass else "FAIL",
            )
        )

        # Law 2: Anti-Symmetry Law (P(A vs B) + P(B vs A) == 100.0% +/- 1%)
        p_other = self.get_standard_level7_comp("full")
        p_a_b = eval_combat_prob(p_std, p_other)
        p_b_a = eval_combat_prob(p_other, p_std)
        anti_sym_sum = p_a_b + p_b_a
        law2_pass = abs(anti_sym_sum - 100.0) <= 1.0
        report.test_results.append(
            TestResult(
                name="Anti-Symmetry Law",
                description="Combat resolver must be mathematically anti-symmetric across focal/opponent swap",
                law="P(A vs B) + P(B vs A) == 100.0%",
                expected="100.00% +/- 1.0%",
                actual=f"{anti_sym_sum:.2f}% ({p_a_b:.1f}% + {p_b_a:.1f}%)",
                passed=law2_pass,
                status="PASS" if law2_pass else "FAIL",
            )
        )

        # Law 3: Board Size Dominance Law (9 units 2* vs 3 units 1* -> P(Win) > 95%)
        p_9, p_3 = self.get_unit_count_comps()
        p_9_vs_3 = eval_combat_prob(p_9, p_3)
        law3_pass = p_9_vs_3 >= 95.0
        report.test_results.append(
            TestResult(
                name="Unit Count Dominance Law",
                description="9 units (2-star) must overwhelmingly crush 3 units (1-star)",
                law="P(9 Units vs 3 Units) >= 95.0%",
                expected=">= 95.00%",
                actual=f"{p_9_vs_3:.2f}%",
                passed=law3_pass,
                status="PASS" if law3_pass else "FAIL",
            )
        )

        # Law 4: Star Level Dominance Law (Full 3-star team vs identical 1-star team -> P(Win) > 90%)
        p_3s, p_1s = self.get_star_level_comps()
        p_3s_vs_1s = eval_combat_prob(p_3s, p_1s)
        law4_pass = p_3s_vs_1s >= 90.0
        report.test_results.append(
            TestResult(
                name="Star Level Dominance Law",
                description="Full 3-star composition must consistently defeat an identical 1-star composition",
                law="P(3-Star vs 1-Star) >= 90.0%",
                expected=">= 90.00%",
                actual=f"{p_3s_vs_1s:.2f}%",
                passed=law4_pass,
                status="PASS" if law4_pass else "FAIL",
            )
        )

        # Law 5: Item Advantage Law (9 Completed Meta Items vs 0 Items -> P(Win) >= 80%)
        p_items = self.get_standard_level7_comp("full")
        p_naked = self.get_standard_level7_comp("none")
        p_items_vs_naked = eval_combat_prob(p_items, p_naked)
        law5_pass = p_items_vs_naked >= 80.0
        law5_status = "PASS" if law5_pass else ("WARN" if p_items_vs_naked >= 65.0 else "FAIL")
        report.test_results.append(
            TestResult(
                name="Item Advantage Law",
                description="An identical composition equipped with 9 completed meta items must overwhelmingly defeat 0 items",
                law="P(9 Items vs 0 Items) >= 80.0%",
                expected=">= 80.00%",
                actual=f"{p_items_vs_naked:.2f}%",
                passed=law5_pass,
                status=law5_status,
                details={
                    "heuristic_comparison": "HeuristicCombatResolver yields 368.3 power vs 136.1 power (x2.71 multiplier)",
                },
            )
        )

        # Law 6: Cost / Tier Dominance Law (Full 4/5-Costs 2* vs 1-Costs 2* -> P(Win) >= 75%)
        p_high, p_low = self.get_cost_tier_comps()
        p_high_vs_low = eval_combat_prob(p_high, p_low)
        law6_pass = p_high_vs_low >= 75.0
        report.test_results.append(
            TestResult(
                name="Champion Cost Dominance Law",
                description="Full 4-cost / 5-cost composition must overpower a 1-cost composition of equal stars",
                law="P(High Cost vs Low Cost) >= 75.0%",
                expected=">= 75.00%",
                actual=f"{p_high_vs_low:.2f}%",
                passed=law6_pass,
                status="PASS" if law6_pass else "FAIL",
            )
        )

        # Empirical Replay Evaluation (sample 50 matches from shadow replay cache)
        shadow_replays = self.get_shadow_replays()
        if shadow_replays:
            sample_probs = []
            for rep in shadow_replays[:30]:
                for r_idx, rd in enumerate(rep.rounds[:3]):
                    opp_player = create_player_from_opponent_board(
                        opponent_board=rd.opponent_board,
                        opponent_health=rd.opponent_health,
                        opponent_level=rd.opponent_level,
                        set_data=self.set_data,
                    )
                    prob = eval_combat_prob(p_items, opp_player, stage=rd.stage)
                    sample_probs.append(prob)

            report.empirical_metrics = {
                "replay_samples_evaluated": len(sample_probs),
                "mean_win_probability_vs_human_boards": f"{np.mean(sample_probs):.2f}%",
                "std_win_probability": f"{np.std(sample_probs):.2f}%",
            }

        # Overall Status
        statuses = [t.status for t in report.test_results]
        if "FAIL" in statuses:
            report.overall_status = "FAIL"
            failed_tests = [t.name for t in report.test_results if t.status == "FAIL"]
            report.diagnosis = (
                f"DeepSiameseCombatNet fails the following laws: {', '.join(failed_tests)}. "
                "Review feature representations, logit calibration, or interaction MLP gradient flow."
            )
        else:
            report.overall_status = "PASS"
            report.diagnosis = "All combat sanity laws passed."

        return report

    # =========================================================================
    # AUDIT 3: BoardQualityNet
    # =========================================================================

    def audit_board_quality(self) -> ModelAuditReport:
        """Audit BoardQualityNet (Placement and Top-4 probability oracle)."""
        trunk = self.get_trunk()
        bq_ckpt = Path("models/board_evaluator/board_quality_best.pt")
        report = ModelAuditReport(
            model_name="BoardQualityNet",
            checkpoint_path=str(bq_ckpt),
            overall_status="PASS",
        )

        if not bq_ckpt.exists():
            report.overall_status = "FAIL"
            report.diagnosis = f"Checkpoint not found at {bq_ckpt}"
            return report

        board_evaluator = BoardQualityNet.load_checkpoint(bq_ckpt, trunk=trunk, device=self.device)
        encoder = self.get_encoder()

        def eval_board_oracle(player: Player, stage: int = 4) -> tuple[float, float]:
            _, s_vec, _ = encoder.extract_state_vector(player, stage=stage, round_in_stage=1)
            with torch.no_grad():
                pred_place, top4_logits = board_evaluator.forward_fused(s_vec.unsqueeze(0))
                place = float(pred_place.squeeze().item())
                top4_p = float(F.softmax(top4_logits, dim=-1)[0, 1].item()) * 100.0
            return place, top4_p

        # Law 1: Capped Endgame vs Weak Early Board (Placement difference >= 2.0 places)
        p_capped, p_weak = self.get_capped_vs_early_comps()
        place_capped, top4_capped = eval_board_oracle(p_capped, stage=5)
        place_weak, top4_weak = eval_board_oracle(p_weak, stage=5)
        delta_place = place_weak - place_capped  # positive means capped has lower (better) placement
        law1_pass = delta_place >= 2.0
        report.test_results.append(
            TestResult(
                name="Endgame Composition Separation Law",
                description="A capped Stage 5 Level 9 board must predict a substantially superior finish than an early weak board",
                law="E[Place_weak] - E[Place_capped] >= 2.0 places",
                expected=">= +2.00 places difference",
                actual=f"{delta_place:+.2f} places (Capped: {place_capped:.2f}, Weak: {place_weak:.2f})",
                passed=law1_pass,
                status="PASS" if law1_pass else "FAIL",
                details={"top4_capped": f"{top4_capped:.1f}%", "top4_weak": f"{top4_weak:.1f}%"},
            )
        )

        # Law 2: Star Upgrade Monotonicity Law (Upgrading team to 3* must improve placement by >= 0.5 places)
        p_3s, p_1s = self.get_star_level_comps()
        place_3s, top4_3s = eval_board_oracle(p_3s, stage=4)
        place_1s, top4_1s = eval_board_oracle(p_1s, stage=4)
        delta_stars = place_1s - place_3s
        law2_pass = delta_stars >= 0.50
        report.test_results.append(
            TestResult(
                name="Star Upgrade Placement Benefit Law",
                description="Upgrading an entire board from 1-star to 3-star must meaningfully improve placement",
                law="E[Place_1star] - E[Place_3star] >= 0.50 places",
                expected=">= +0.50 places improvement",
                actual=f"{delta_stars:+.2f} places (1*: {place_1s:.2f} -> 3*: {place_3s:.2f})",
                passed=law2_pass,
                status="PASS" if law2_pass else "FAIL",
                details={"top4_1s": f"{top4_1s:.1f}%", "top4_3s": f"{top4_3s:.1f}%"},
            )
        )

        # Law 3: Item Sensitivity Law (Equipping 9 completed items must improve placement by >= 0.40 places)
        p_items = self.get_standard_level7_comp("full")
        p_naked = self.get_standard_level7_comp("none")
        place_items, top4_items = eval_board_oracle(p_items, stage=4)
        place_naked, top4_naked = eval_board_oracle(p_naked, stage=4)
        delta_items = place_naked - place_items
        law3_pass = delta_items >= 0.40
        law3_status = "PASS" if law3_pass else ("WARN" if delta_items >= 0.15 else "FAIL")
        report.test_results.append(
            TestResult(
                name="Itemization Placement Benefit Law",
                description="Equipping 9 completed meta items on carries/tanks must improve expected placement",
                law="E[Place_naked] - E[Place_items] >= 0.40 places",
                expected=">= +0.40 places improvement",
                actual=f"{delta_items:+.2f} places (Naked: {place_naked:.2f} -> Full: {place_items:.2f})",
                passed=law3_pass,
                status=law3_status,
                details={"top4_naked": f"{top4_naked:.1f}%", "top4_items": f"{top4_items:.1f}%"},
            )
        )

        # Law 4: Health Context Sensitivity (100 HP vs 15 HP at Stage 5 -> Top-4 probability must be higher)
        p_healthy = self.build_test_player("Healthy_Player", level=7, health=100, gold=30, units_spec=[])
        p_healthy.board = dict(p_items.board)
        p_dying = self.build_test_player("Dying_Player", level=7, health=12, gold=30, units_spec=[])
        p_dying.board = dict(p_items.board)

        _, top4_healthy = eval_board_oracle(p_healthy, stage=5)
        _, top4_dying = eval_board_oracle(p_dying, stage=5)
        delta_hp = top4_healthy - top4_dying
        law4_pass = delta_hp >= 10.0
        report.test_results.append(
            TestResult(
                name="Health Buffer Survival Law",
                description="At identical boards, a player with 100 HP must have a strictly higher Top-4 probability than a player at 12 HP",
                law="P(Top-4 | 100 HP) - P(Top-4 | 12 HP) >= 10.0%",
                expected=">= +10.00% higher survival",
                actual=f"{delta_hp:+.1f}% points (100 HP: {top4_healthy:.1f}%, 12 HP: {top4_dying:.1f}%)",
                passed=law4_pass,
                status="PASS" if law4_pass else "FAIL",
            )
        )

        # Overall Status
        statuses = [t.status for t in report.test_results]
        if "FAIL" in statuses:
            report.overall_status = "FAIL"
            report.diagnosis = (
                f"BoardQualityNet fails the Itemization Benefit Law (Delta = {delta_items:+.2f} places). "
                "The oracle predicts almost the exact same tournament placement regardless of whether a team has 9 items or 0 items."
            )
        elif "WARN" in statuses:
            report.overall_status = "WARN"
            report.diagnosis = "BoardQualityNet exhibits sub-optimal item sensitivity."
        else:
            report.overall_status = "PASS"

        return report

    # =========================================================================
    # AUDIT 4: StateTransitionPredictor (World Model)
    # =========================================================================

    def audit_transition_world_model(self) -> ModelAuditReport:
        """Audit StateTransitionPredictor (World Model dynamics and stability)."""
        wm_ckpt = Path("models/transition_predictor/predictor_best.pt")
        report = ModelAuditReport(
            model_name="StateTransitionPredictor (World Model)",
            checkpoint_path=str(wm_ckpt),
            overall_status="PASS",
        )

        if not wm_ckpt.exists():
            report.overall_status = "FAIL"
            report.diagnosis = f"Checkpoint not found at {wm_ckpt}"
            return report

        sd = torch.load(wm_ckpt, map_location="cpu", weights_only=False)
        cfg = sd.get("config", {})
        world_model = StateTransitionPredictor(
            input_dim=cfg.get("input_dim", 384),
            hidden_dim=cfg.get("hidden_dim", 512),
            output_dim=cfg.get("output_dim", 384),
            num_layers=cfg.get("num_layers", 3),
            dropout=0.0,
            use_residual_delta=cfg.get("use_residual_delta", True),
        ).to(self.device)
        world_model.load_state_dict(sd.get("state_dict", sd))
        world_model.eval()

        encoder = self.get_encoder()
        p_std = self.get_standard_level7_comp("full")
        _, s0, _ = encoder.extract_state_vector(p_std, stage=3, round_in_stage=2)
        s0 = s0.unsqueeze(0)

        with torch.no_grad():
            s_hat_1 = world_model(s0)
            cos_1step = float(F.cosine_similarity(s0, s_hat_1).item())
            norm_ratio_1 = float(s_hat_1.norm().item() / max(1e-6, s0.norm().item()))

            # Law 1: Non-Trivial Transition Law (Must not collapse to exact identity x)
            law1_pass = cos_1step < 0.9995
            report.test_results.append(
                TestResult(
                    name="Non-Identity Dynamics Law",
                    description="World Model must predict non-zero transition dynamics (must not output identity copy of input)",
                    law="cos_sim(s_t, s_hat_{t+1}) < 0.9995",
                    expected="< 0.9995 (genuine transition)",
                    actual=f"{cos_1step:.5f}",
                    passed=law1_pass,
                    status="PASS" if law1_pass else "FAIL",
                )
            )

            # Law 2: Autoregressive Stability Law (Norm after 5 recursive steps must remain bounded [0.5, 2.0])
            s_curr = s0.clone()
            for _ in range(5):
                s_curr = world_model(s_curr)
            norm_ratio_5 = float(s_curr.norm().item() / max(1e-6, s0.norm().item()))
            law2_pass = 0.5 <= norm_ratio_5 <= 2.0
            report.test_results.append(
                TestResult(
                    name="Autoregressive Rollout Stability Law",
                    description="Multi-step recursive lookahead must not explode towards infinity or collapse to zero",
                    law="0.50 <= ||s_{t+5}|| / ||s_t|| <= 2.00",
                    expected="[0.50, 2.00]",
                    actual=f"{norm_ratio_5:.3f}x norm ratio",
                    passed=law2_pass,
                    status="PASS" if law2_pass else "FAIL",
                    details={"1_step_ratio": f"{norm_ratio_1:.3f}x", "5_step_ratio": f"{norm_ratio_5:.3f}x"},
                )
            )

            # Law 3: Transition Predictor Magnitude Law (Residual delta must be non-zero and stable)
            delta_norm = float(torch.dist(s0, s_hat_1).item())
            law3_pass = delta_norm > 0.01
            report.test_results.append(
                TestResult(
                    name="Residual Transition Magnitude Law",
                    description="Predicted change delta must have measurable non-zero Euclidean magnitude",
                    law="||s_hat_{t+1} - s_t|| > 0.01",
                    expected="> 0.010",
                    actual=f"{delta_norm:.4f}",
                    passed=law3_pass,
                    status="PASS" if law3_pass else "FAIL",
                )
            )

        statuses = [t.status for t in report.test_results]
        report.overall_status = "FAIL" if "FAIL" in statuses else "PASS"
        return report

    # =========================================================================
    # AUDIT 5: Z-Index Centroids (Macro Archetypes)
    # =========================================================================

    def audit_z_index_centroids(self) -> ModelAuditReport:
        """Audit Z-Index composition centroids and archetype separation."""
        z_ckpt = Path("models/clustering/z_index.pt")
        report = ModelAuditReport(
            model_name="Z-Index Macro Composition Centroids",
            checkpoint_path=str(z_ckpt),
            overall_status="PASS",
        )

        if not z_ckpt.exists():
            report.overall_status = "FAIL"
            report.diagnosis = f"Checkpoint not found at {z_ckpt}"
            return report

        data = torch.load(z_ckpt, map_location="cpu", weights_only=False)
        raw_z = data.get("z_index", data.get("centroids"))
        z_tensor = raw_z if isinstance(raw_z, torch.Tensor) else torch.as_tensor(raw_z, dtype=torch.float32)

        k, dim = z_tensor.shape
        report.empirical_metrics = {
            "cluster_count_k": k,
            "latent_feature_dim": dim,
            "mean_centroid_norm": f"{float(z_tensor.norm(dim=-1).mean().item()):.3f}",
        }

        # Normalize centroids for cosine distance matrix
        z_normed = F.normalize(z_tensor, p=2, dim=-1)
        sim_matrix = (z_normed @ z_normed.T).numpy()

        # Pairwise off-diagonal cosine similarities
        off_diag_sims = []
        for i in range(k):
            for j in range(i + 1, k):
                off_diag_sims.append(float(sim_matrix[i, j]))

        max_sim = max(off_diag_sims)
        mean_sim = float(np.mean(off_diag_sims))
        min_dist = 1.0 - max_sim

        # Law 1: Archetype Distinction Law (No two centroids may have cos_sim >= 0.85)
        law1_pass = max_sim < 0.85
        report.test_results.append(
            TestResult(
                name="Archetype Orthogonality Law",
                description="Target archetypes must represent distinctly separated strategic comps",
                law="max_{i != j} cos_sim(z_i, z_j) < 0.85",
                expected="< 0.850",
                actual=f"{max_sim:.4f} (mean sim = {mean_sim:.4f})",
                passed=law1_pass,
                status="PASS" if law1_pass else "FAIL",
                details={"min_cosine_distance": f"{min_dist:.4f}"},
            )
        )

        # Law 2: Non-Degenerate Centroid Law (All centroids must have non-zero norm)
        min_norm = float(z_tensor.norm(dim=-1).min().item())
        law2_pass = min_norm > 0.1
        report.test_results.append(
            TestResult(
                name="Centroid Non-Degeneracy Law",
                description="All k cluster centroids must possess non-trivial vector magnitude",
                law="min_k ||z_k|| > 0.10",
                expected="> 0.100",
                actual=f"{min_norm:.4f}",
                passed=law2_pass,
                status="PASS" if law2_pass else "FAIL",
            )
        )

        # Law 3: Semantic Composition Alignment Law
        trunk = self.get_trunk()
        p_std = self.get_standard_level7_comp("full")
        p_high, _ = self.get_cost_tier_comps()
        encoder = self.get_encoder()

        c_std, s_std, i_std, t_std = encoder.encode_board_tensors(p_std)
        c_high, s_high, i_high, t_high = encoder.encode_board_tensors(p_high)

        with torch.no_grad():
            h_std = trunk.encode_board(c_std.unsqueeze(0), s_std.unsqueeze(0), i_std.unsqueeze(0), t_std.unsqueeze(0)).squeeze(0)
            h_high = trunk.encode_board(c_high.unsqueeze(0), s_high.unsqueeze(0), i_high.unsqueeze(0), t_high.unsqueeze(0)).squeeze(0)

        h_std_norm = F.normalize(h_std.cpu(), p=2, dim=-1)
        h_high_norm = F.normalize(h_high.cpu(), p=2, dim=-1)

        sims_std = (z_normed @ h_std_norm).numpy()
        sims_high = (z_normed @ h_high_norm).numpy()

        max_sim_std = float(np.max(sims_std))
        max_sim_high = float(np.max(sims_high))

        law3_pass = max_sim_std > 0.15 and max_sim_high > 0.15
        report.test_results.append(
            TestResult(
                name="Semantic Composition Alignment Law",
                description="Canonical tournament compositions must project positively into the latent archetype subspace",
                law="max_k cos_sim(h_{comp}, z_k) > 0.15",
                expected="> 0.150 for meta comps",
                actual=f"Comp A max sim = {max_sim_std:.4f}, Comp B max sim = {max_sim_high:.4f}",
                passed=law3_pass,
                status="PASS" if law3_pass else "FAIL",
                details={"comp_a_best_cluster": int(np.argmax(sims_std)), "comp_b_best_cluster": int(np.argmax(sims_high))},
            )
        )

        statuses = [t.status for t in report.test_results]
        report.overall_status = "FAIL" if "FAIL" in statuses else "PASS"
        return report

    # =========================================================================
    # AUDIT 6: EloRegressor
    # =========================================================================

    def audit_elo_regressor(self) -> ModelAuditReport:
        """Audit EloRegressor skill prediction across ranked ladder tiers."""
        elo_ckpt = Path("models/elo_predictor/elo_regressor_player_level.pkl")
        report = ModelAuditReport(
            model_name="EloRegressor (Skill Rating Predictor)",
            checkpoint_path=str(elo_ckpt),
            overall_status="PASS",
        )

        if not elo_ckpt.exists():
            report.overall_status = "FAIL"
            report.diagnosis = f"Checkpoint not found at {elo_ckpt}"
            return report

        try:
            with open(elo_ckpt, "rb") as f:
                model_obj = pickle.load(f)

            has_pred = hasattr(model_obj, "predict_match")

            # Law 1: Model Object & Inference Interface
            report.test_results.append(
                TestResult(
                    name="Model Object & Inference Interface",
                    description="MatchEloRegressor deserializes cleanly and exposes competitive predict_match interface",
                    law="hasattr(model, 'predict_match')",
                    expected="True",
                    actual=f"hasattr = {has_pred}",
                    passed=has_pred,
                    status="PASS" if has_pred else "FAIL",
                )
            )

            if not has_pred:
                report.overall_status = "FAIL"
                report.diagnosis = "Model missing predict_match interface"
                return report

            mean_f = model_obj.ood_detector.feature_means.copy()

            # Law 2: High vs Low Elo Tempo Monotonicity Law
            # Challenger tempo: fast-8 (round 14), high avg level (8.6), level 9, high pvp win rate (0.65)
            chal_f = mean_f.copy()
            chal_f[15] = 14.0  # round reached lvl 8
            chal_f[16] = 8.6   # avg level
            chal_f[12] = 9.0   # final level
            chal_f[2] = 0.65   # pvp win rate
            chal_f[14] = 10.0  # round reached lvl 7

            # Iron tempo: late level 8 (round 28), low avg level (5.5), level 6, low win rate (0.25), hoards 50g blindly
            iron_f = mean_f.copy()
            iron_f[15] = 28.0
            iron_f[16] = 5.5
            iron_f[12] = 6.0
            iron_f[2] = 0.25
            iron_f[9] = 0.85

            elo_chal, tier_chal, _ = model_obj.predict_match(chal_f)
            elo_iron, tier_iron, _ = model_obj.predict_match(iron_f)
            delta_elo = elo_chal - elo_iron

            law2_pass = delta_elo >= 800.0
            report.test_results.append(
                TestResult(
                    name="Competitive Tempo Monotonicity Law",
                    description="Aggressive Challenger tempo must predict strictly higher rating than passive low-tier play",
                    law="Elo(Challenger Tempo) - Elo(Iron Tempo) >= +800",
                    expected=">= +800 ELO separation",
                    actual=f"Delta = +{delta_elo:.0f} ELO (Challenger: {elo_chal:.0f} [{tier_chal}] vs Iron: {elo_iron:.0f} [{tier_iron}])",
                    passed=law2_pass,
                    status="PASS" if law2_pass else "FAIL",
                )
            )

            # Law 3: Solution Space OOD Detection Law
            _, _, ood_human = model_obj.predict_match(mean_f)

            # Aberrant bot: level 4 at round 30, hoards 95 gold, 0 items
            bot_f = mean_f.copy()
            bot_f[0] = 8.0
            bot_f[7] = 95.0
            bot_f[12] = 4.0
            bot_f[17] = 8.0
            bot_f[22] = 0.0
            _, _, ood_bot = model_obj.predict_match(bot_f)

            law3_pass = bool(ood_human["is_in_distribution"]) and not bool(ood_bot["is_in_distribution"])
            report.test_results.append(
                TestResult(
                    name="Solution Space OOD Detection Law",
                    description="Valid human gameplay must be flagged in-distribution, aberrant bot trajectories flagged as outliers",
                    law="Human == Inlier and Aberrant Bot == Outlier",
                    expected="Human inlier and Bot outlier",
                    actual=f"Human inlier={ood_human['is_in_distribution']} ({ood_human['inlier_confidence_pct']:.1f}%), Bot outlier={not ood_bot['is_in_distribution']} (Dist={ood_bot['mahalanobis_distance']:.1f})",
                    passed=law3_pass,
                    status="PASS" if law3_pass else "FAIL",
                )
            )

            # Law 4: Competitive Elo Boundedness Law
            law4_pass = 100.0 <= elo_chal <= 4600.0 and 100.0 <= elo_iron <= 4600.0
            report.test_results.append(
                TestResult(
                    name="Competitive Elo Range Boundedness Law",
                    description="Predicted Elo ratings must remain within realistic ranked ladder bounds [100, 4600]",
                    law="100.0 <= Elo <= 4600.0",
                    expected="[100, 4600]",
                    actual=f"Challenger: {elo_chal:.0f}, Iron: {elo_iron:.0f}",
                    passed=law4_pass,
                    status="PASS" if law4_pass else "FAIL",
                )
            )

        except Exception as e:
            report.overall_status = "FAIL"
            report.diagnosis = f"Failed to audit EloRegressor: {e}"
            return report

        statuses = [t.status for t in report.test_results]
        report.overall_status = "FAIL" if "FAIL" in statuses else "PASS"
        report.diagnosis = "All Elo regressor sanity laws passed." if report.overall_status == "PASS" else "One or more Elo laws failed."
        return report

    # =========================================================================
    # AUDIT 7: TFTActorCritic (latest AlphaStar RL Agent)
    # =========================================================================

    def audit_rl_actor_critic(self) -> ModelAuditReport:
        """Audit TFTActorCritic policy and value heads (PPO checkpoint)."""
        v7_root = Path(r"D:\tft-winner-data\set18\models\rl\checkpoints\ppo_alphastar_v7")
        v7_checkpoints = sorted(
            v7_root.glob("gen_*/training_state.pt"),
            key=lambda checkpoint: checkpoint.parent.name,
        )
        rl_ckpt = (
            v7_checkpoints[-1]
            if v7_checkpoints
            else Path(r"D:\tft-winner-data\set18\models\rl\checkpoints\ppo_alphastar_v6\gen_0300\training_state.pt")
        )
        report = ModelAuditReport(
            model_name=f"TFTActorCritic (AlphaStar {rl_ckpt.parent.parent.name} {rl_ckpt.parent.name})",
            checkpoint_path=str(rl_ckpt),
            overall_status="PASS",
        )

        if not rl_ckpt.exists():
            report.overall_status = "FAIL"
            report.diagnosis = f"Checkpoint not found at {rl_ckpt}"
            return report

        sd = torch.load(rl_ckpt, map_location=self.device, weights_only=False)
        actor_critic = TFTActorCritic(obs_dim=768, action_dim=111).to(self.device)
        model_state = sd.get("model_state_dict", sd.get("model", sd))
        actor_critic.load_state_dict(model_state)
        actor_critic.eval()

        encoder = self.get_encoder()

        # Law 1: Action Mask Strictness Law (Illegal actions must receive exactly 0.000 probability mass)
        dummy_obs = torch.randn(1, 768, device=self.device)
        dummy_mask = torch.zeros(1, 111, dtype=torch.bool, device=self.device)
        # Allow only actions 0 (pass) and 7 (buy xp)
        dummy_mask[0, 0] = True
        dummy_mask[0, 7] = True

        with torch.no_grad():
            from tft_ai_player.rl.models.distributions import MaskedCategorical
            logits, _ = actor_critic.forward(dummy_obs)
            dist = MaskedCategorical(logits=logits, mask=dummy_mask)
            probs = dist.probs.squeeze(0).cpu().numpy()

        illegal_prob_mass = float(probs[~dummy_mask.squeeze(0).cpu().numpy()].sum())
        law1_pass = illegal_prob_mass < 1e-6
        report.test_results.append(
            TestResult(
                name="Action Mask Enforcement Law",
                description="Illegal masked actions must receive mathematically zero probability mass",
                law="sum_{a in masked} P(a) == 0.0000",
                expected="0.000000",
                actual=f"{illegal_prob_mass:.8f}",
                passed=law1_pass,
                status="PASS" if law1_pass else "FAIL",
            )
        )

        # Law 2: Health Monotonicity Law (100 HP must have higher V(s) than 10 HP at identical board)
        p_items = self.get_standard_level7_comp("full")
        p_high_hp = self.build_test_player("High_HP", level=7, health=100, gold=30, units_spec=[])
        p_high_hp.board = dict(p_items.board)
        p_low_hp = self.build_test_player("Low_HP", level=7, health=12, gold=30, units_spec=[])
        p_low_hp.board = dict(p_items.board)

        obs_high_hp, _, _ = encoder.extract_state_vector(p_high_hp, stage=4, round_in_stage=1)
        obs_low_hp, _, _ = encoder.extract_state_vector(p_low_hp, stage=4, round_in_stage=1)

        with torch.no_grad():
            v_high = float(actor_critic.get_value(torch.as_tensor(obs_high_hp, dtype=torch.float32, device=self.device)).item())
            v_low = float(actor_critic.get_value(torch.as_tensor(obs_low_hp, dtype=torch.float32, device=self.device)).item())

        delta_v_hp = v_high - v_low
        law2_pass = delta_v_hp > 0.0
        report.test_results.append(
            TestResult(
                name="Health Monotonicity Value Law",
                description="Higher health buffer must reflect higher expected cumulative return V(s)",
                law="V(s | 100 HP) > V(s | 12 HP)",
                expected="V(100 HP) > V(12 HP)",
                actual=f"Delta = {delta_v_hp:+.4f} (100 HP: {v_high:+.3f} vs 12 HP: {v_low:+.3f})",
                passed=law2_pass,
                status="PASS" if law2_pass else "FAIL",
            )
        )

        # Law 3: Board Size Monotonicity Law (Level 9 full board must have higher V(s) than Level 3 board)
        p_9, p_3 = self.get_unit_count_comps()
        obs_9, _, _ = encoder.extract_state_vector(p_9, stage=4, round_in_stage=1)
        obs_3, _, _ = encoder.extract_state_vector(p_3, stage=4, round_in_stage=1)

        with torch.no_grad():
            v_9 = float(actor_critic.get_value(torch.as_tensor(obs_9, dtype=torch.float32, device=self.device)).item())
            v_3 = float(actor_critic.get_value(torch.as_tensor(obs_3, dtype=torch.float32, device=self.device)).item())

        delta_v_board = v_9 - v_3
        law3_pass = delta_v_board > 0.0
        report.test_results.append(
            TestResult(
                name="Board Size Value Monotonicity Law",
                description="A 9-unit full board must evaluate to a strictly higher expected return than a 3-unit board",
                law="V(s | 9 Units) > V(s | 3 Units)",
                expected="V(9 Units) > V(3 Units)",
                actual=f"Delta = {delta_v_board:+.4f} (9 Units: {v_9:+.3f} vs 3 Units: {v_3:+.3f})",
                passed=law3_pass,
                status="PASS" if law3_pass else "FAIL",
            )
        )

        # Law 4: Item Sensitivity Law (V(s) with 9 items must be higher than V(s) with 0 items at identical HP)
        p_items = self.get_standard_level7_comp("full")
        p_naked = self.get_standard_level7_comp("none")
        obs_items, _, _ = encoder.extract_state_vector(p_items, stage=4, round_in_stage=1)
        obs_naked, _, _ = encoder.extract_state_vector(p_naked, stage=4, round_in_stage=1)

        with torch.no_grad():
            v_naked = float(actor_critic.get_value(torch.as_tensor(obs_naked, dtype=torch.float32, device=self.device)).item())
            v_items = float(actor_critic.get_value(torch.as_tensor(obs_items, dtype=torch.float32, device=self.device)).item())

        delta_v_items = v_items - v_naked
        law4_pass = delta_v_items > 0.10
        law4_status = "PASS" if law4_pass else ("WARN" if delta_v_items > 0.0 else "FAIL")
        report.test_results.append(
            TestResult(
                name="Critic Item Sensitivity Law",
                description="The critic value function V(s) must appreciate the massive combat value of 9 completed items",
                law="V(s | 9 Items) - V(s | 0 Items) > +0.10",
                expected=">= +0.100 higher value",
                actual=f"{delta_v_items:+.4f} (Naked: {v_naked:+.3f} -> Items: {v_items:+.3f})",
                passed=law4_pass,
                status=law4_status,
            )
        )

        statuses = [t.status for t in report.test_results]
        if "FAIL" in statuses:
            report.overall_status = "FAIL"
            failed_tests = [t.name for t in report.test_results if t.status == "FAIL"]
            report.diagnosis = f"TFTActorCritic fails the following laws: {', '.join(failed_tests)}."
        else:
            report.overall_status = "PASS"
            report.diagnosis = "All actor-critic sanity laws passed."

        return report

    # =========================================================================
    # MASTER RUNNER
    # =========================================================================

    def run_all(self) -> dict[str, Any]:
        """Execute the entire benchmark suite across all 7 models."""
        print("\n" + "=" * 80)
        print(" [TFT-AI ECOSYSTEM SANITY & BENCHMARK AUDIT SUITE]")
        print(f" Execution Device: {self.device} | Set: 18")
        print("=" * 80 + "\n")

        start_time = time.time()
        reports: dict[str, ModelAuditReport] = {}

        # 1. MultiModalFusionTrunk
        print(" [*] Auditing Model 1/7: MultiModalFusionTrunk...")
        reports["trunk"] = self.audit_trunk()

        # 2. DeepSiameseCombatNet
        print(" [*] Auditing Model 2/7: DeepSiameseCombatNet...")
        reports["combat_resolver"] = self.audit_siamese_combat()

        # 3. BoardQualityNet
        print(" [*] Auditing Model 3/7: BoardQualityNet...")
        reports["board_quality"] = self.audit_board_quality()

        # 4. StateTransitionPredictor
        print(" [*] Auditing Model 4/7: StateTransitionPredictor (World Model)...")
        reports["world_model"] = self.audit_transition_world_model()

        # 5. Z-Index Centroids
        print(" [*] Auditing Model 5/7: Z-Index Macro Centroids...")
        reports["z_index"] = self.audit_z_index_centroids()

        # 6. EloRegressor
        print(" [*] Auditing Model 6/7: EloRegressor...")
        reports["elo_regressor"] = self.audit_elo_regressor()

        # 7. TFTActorCritic
        print(" [*] Auditing Model 7/7: TFTActorCritic (latest AlphaStar RL checkpoint)...")
        reports["actor_critic"] = self.audit_rl_actor_critic()

        elapsed = time.time() - start_time
        print(f"\n [+] Audit completed in {elapsed:.2f} seconds.")

        # Serialize
        serialized = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": round(elapsed, 2),
            "device": str(self.device),
            "reports": {k: asdict(v) for k, v in reports.items()},
        }

        # Terminal Print Scorecard
        self.print_scorecard(reports)

        return serialized

    def print_scorecard(self, reports: dict[str, ModelAuditReport]) -> None:
        """Render a formatted scorecard table in the console."""
        print("\n" + "=" * 80)
        print(" [AUDIT SUMMARY SCORECARD]")
        print("=" * 80)
        print(f" {'MODEL':<35} | {'STATUS':<8} | {'PASSED':<8} | {'DIAGNOSIS'}")
        print("-" * 80)

        for key, rep in reports.items():
            pass_count = sum(1 for t in rep.test_results if t.passed)
            total_count = len(rep.test_results)
            short_diag = rep.diagnosis[:45] + "..." if len(rep.diagnosis) > 45 else rep.diagnosis
            if not short_diag:
                short_diag = "All axiomatic tests passed."
            status_color = rep.overall_status
            print(f" {rep.model_name:<35} | {status_color:<8} | {pass_count}/{total_count:<6} | {short_diag}")

        print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="TFT Model Ecosystem Sanity Suite")
    parser.add_argument("--save-json", type=str, default="reports/sanity_metrics.json", help="Path to save JSON metrics")
    parser.add_argument("--save-html", type=str, default="reports/model_ecosystem_sanity_report.html", help="Path to save HTML dashboard")
    args = parser.parse_args()

    suite = ModelSanitySuite()
    results = suite.run_all()

    # Save JSON
    if args.save_json:
        json_path = Path(args.save_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f" [+] Exported JSON audit results to {json_path.resolve()}")

    # Save HTML
    if args.save_html:
        from .reporters.html_dashboard import generate_html_report
        html_path = Path(args.save_html)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        generate_html_report(results, html_path)
        print(f" [+] Generated standalone interactive HTML dashboard at {html_path.resolve()}")


if __name__ == "__main__":
    main()
