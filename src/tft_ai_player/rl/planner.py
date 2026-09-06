"""Shop-by-Shop Beam Search Tree Planner for Teamfight Tactics.

Implements deterministic forward-search lookahead over visible shop cards:
  - Explores subsets of purchases (BUY_SHOP_0..4)
  - Manages bench space via intelligent selling of low-synergy units (SELL_BENCH_0..8)
  - Evaluates XP investment towards level breakpoints (BUY_XP)
  - Deploys optimal units to board capacity (DEPLOY_UNIT)
  - Scores candidate states using the World Model (cos(s', ŝ_{t+1})), macro archetype alignment (cos(h', z)),
    gold interest thresholds (10/20/30/40/50g), and board combat strength (2-star/3-star upgrades, active trait tiers).
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F

from tft_ai_player.simulation.actions import (
    TOTAL_DISCRETE_ACTIONS,
    execute_action,
    get_action_mask,
    index_to_hex,
)
from tft_ai_player.simulation.config import SetData
from tft_ai_player.simulation.models import ChampionInstance, ChampionPool, Player

if TYPE_CHECKING:
    from tft_ai_player.simulation.gym_env import TFTStateEncoder

logger = logging.getLogger(__name__)


@dataclass
class PlannerNode:
    """A node in the shop beam search tree representing a simulated player state."""

    player: Player
    pool: ChampionPool
    actions: list[int] = field(default_factory=list)
    heuristic_score: float = 0.0
    neural_score: float = 0.0


class ShopBeamSearchPlanner:
    """Tree search planner that optimizes micro-actions within a visible shop window."""

    def __init__(
        self,
        set_data: SetData,
        encoder: TFTStateEncoder | None = None,
        world_model: torch.nn.Module | None = None,
        board_evaluator: torch.nn.Module | None = None,
        beam_width: int = 8,
        w_world: float = 1.0,
        w_board_quality: float = 1.0,
        w_macro: float = 0.6,
        w_econ: float = 0.3,
        w_stars: float = 0.25,
        w_traits: float = 0.15,
        use_neural_eval: bool = False,
        device: torch.device | None = None,
    ) -> None:
        self.set_data = set_data
        self.encoder = encoder
        self.world_model = world_model
        self.board_evaluator = board_evaluator
        self.beam_width = beam_width
        self.w_world = w_world
        self.w_board_quality = w_board_quality
        self.w_macro = w_macro
        self.w_econ = w_econ
        self.w_stars = w_stars
        self.w_traits = w_traits
        self.use_neural_eval = use_neural_eval
        self.device = device or torch.device("cpu")
        self.action_queue: list[int] = []

    def compute_fast_heuristic(self, player: Player, target_z: np.ndarray | None = None) -> float:
        """Fast scalar heuristic score for candidate pre-screening (runtime < 5 microseconds)."""
        score = 0.0

        # 1. Economy interest preservation (up to 50g)
        interest_gold = min(5, player.gold // 10)
        score += interest_gold * self.w_econ

        # 2. Board unit values and star levels
        for unit in player.board.values():
            score += (unit.star_level - 1) * self.w_stars
            score += unit.cost * 0.02

        # 3. Active trait synergies
        active_tiers = player.get_active_traits()
        score += sum(active_tiers.values()) * self.w_traits

        # 4. Penalty if board is not full while bench has units
        if player.board_unit_count < player.max_board_units and any(u is not None for u in player.bench):
            score -= 0.5

        return score

    @torch.no_grad()
    def evaluate_neural_score(
        self,
        player: Player,
        stage: int,
        round_in_stage: int,
        target_z: np.ndarray | torch.Tensor | None = None,
        s_hat_next: torch.Tensor | None = None,
    ) -> float:
        """Evaluate candidate state using BoardQualityNet or World Model and Trunk Embeddings."""
        if self.encoder is None:
            return self.compute_fast_heuristic(player)

        _, s_cand, h_cand = self.encoder.extract_state_vector(
            player=player,
            stage=stage,
            round_in_stage=round_in_stage,
            target_z=target_z,
        )

        total_score = 0.0

        # V6 Board Quality Oracle: Placement & Top-4 prediction from 320D trunk
        if self.board_evaluator is not None and self.w_board_quality > 0.0:
            pred_place, top4_logits = self.board_evaluator.forward_fused(s_cand.unsqueeze(0))
            # E[Placement] ∈ [1.0, 8.0]. Convert to normalized reward in [0, 1]
            board_quality = float((8.0 - pred_place.squeeze()).clamp(0.0, 7.0).item() / 7.0)
            top4_prob = float(F.softmax(top4_logits, dim=-1)[0, 1].item())
            total_score += self.w_board_quality * (0.6 * board_quality + 0.4 * top4_prob)

        # Legacy / Fallback World Model Alignment: cos(s', ŝ_{t+1})
        elif s_hat_next is not None and self.w_world > 0.0:
            world_sim = float(
                F.cosine_similarity(s_cand.unsqueeze(0), s_hat_next.unsqueeze(0)).item()
            )
            total_score += self.w_world * world_sim

        # Target Archetype Macro Alignment: cos(h_{board}', z)
        if target_z is not None and self.w_macro > 0.0:
            if isinstance(target_z, np.ndarray):
                z_tensor = torch.as_tensor(target_z, dtype=torch.float32, device=self.device)
            else:
                z_tensor = target_z.to(self.device).float()
            if z_tensor.norm() > 1e-6:
                macro_sim = float(
                    F.cosine_similarity(h_cand.unsqueeze(0), z_tensor.unsqueeze(0)).item()
                )
                total_score += self.w_macro * macro_sim

        # Auxiliary heuristic components
        total_score += self.compute_fast_heuristic(player)

        return total_score

    def plan_shop_sequence(
        self,
        player: Player,
        pool: ChampionPool,
        stage: int,
        round_in_stage: int,
        target_z: np.ndarray | None = None,
        s_hat_next: torch.Tensor | None = None,
    ) -> list[int]:
        """Search for the highest-value sequence of actions in the current visible shop.

        Returns:
            list[int]: Sequence of discrete action IDs (e.g. [BUY_SHOP_0, DEPLOY_UNIT_15, ...])
        """
        # Optional neural evaluation projection
        if getattr(self, "use_neural_eval", False):
            if s_hat_next is None and self.world_model is not None and self.encoder is not None:
                with torch.no_grad():
                    _, s_t, _ = self.encoder.extract_state_vector(
                        player=player,
                        stage=stage,
                        round_in_stage=round_in_stage,
                        target_z=target_z,
                    )
                    s_hat_next = self.world_model(s_t.unsqueeze(0)).squeeze(0)

        # Baseline root state (taking 0 actions)
        root_node = PlannerNode(
            player=player.clone(),
            pool=pool.clone(),
            actions=[],
            heuristic_score=self.compute_fast_heuristic(player),
        )
        if getattr(self, "use_neural_eval", False):
            root_score = self.evaluate_neural_score(
                player=player,
                stage=stage,
                round_in_stage=round_in_stage,
                target_z=target_z,
                s_hat_next=s_hat_next,
            )
            root_node.neural_score = root_score

        # Identify purchasable slots
        available_slots = [i for i, card in enumerate(player.shop.slots) if card is not None]

        # Phase 1: Explore subsets of shop purchases
        candidates: list[PlannerNode] = [root_node]

        # Try single and multi-card purchase combinations (up to min(3, len(available_slots)))
        max_combo = min(3, len(available_slots))
        for k in range(1, max_combo + 1):
            for slot_combo in itertools.combinations(available_slots, k):
                p_sim = player.clone()
                pool_sim = pool.clone()
                actions_sim: list[int] = []
                valid_combo = True

                for slot_idx in slot_combo:
                    action_id = 1 + slot_idx  # BUY_SHOP_0..4
                    if p_sim.can_buy_champion(slot_idx):
                        p_sim.buy_shop_slot(slot_idx, pool_sim)
                        actions_sim.append(action_id)
                    elif p_sim.free_bench_slots == 0:
                        c_id = p_sim.shop.slots[slot_idx]
                        c_def = p_sim.set_data.champions.get(c_id, None) if c_id else None
                        c_cost = c_def.cost if c_def else 1
                        if p_sim.gold >= c_cost:
                            # Bench full: Try selling the lowest-tier bench unit to make space
                            bench_units = [(idx, u) for idx, u in enumerate(p_sim.bench) if u is not None]
                            if bench_units:
                                # Prioritize selling 1-star, low cost
                                bench_units.sort(key=lambda item: (item[1].star_level, item[1].cost))
                                sell_slot, _ = bench_units[0]
                                sell_action = 8 + sell_slot
                                p_sim.sell_unit(is_board=False, loc=sell_slot, pool=pool_sim)
                                actions_sim.append(sell_action)
                                if p_sim.can_buy_champion(slot_idx):
                                    p_sim.buy_shop_slot(slot_idx, pool_sim)
                                    actions_sim.append(action_id)
                                else:
                                    valid_combo = False
                                    break
                            else:
                                valid_combo = False
                                break
                        else:
                            valid_combo = False
                            break
                    else:
                        valid_combo = False
                        break

                if valid_combo and actions_sim:
                    h_score = self.compute_fast_heuristic(p_sim)
                    candidates.append(
                        PlannerNode(
                            player=p_sim,
                            pool=pool_sim,
                            actions=actions_sim,
                            heuristic_score=h_score,
                        )
                    )

        # Phase 2: Explore Leveling (BUY_XP = 7)
        xp_candidates: list[PlannerNode] = []
        for cand in candidates:
            # Check if buying XP is feasible and beneficial
            if cand.player.gold >= cand.player.set_data.exp_buy_cost and cand.player.level < cand.player.set_data.max_level:
                p_xp = cand.player.clone()
                pool_xp = cand.pool.clone()
                p_xp.buy_exp()
                xp_acts = list(cand.actions) + [7]
                xp_candidates.append(
                    PlannerNode(
                        player=p_xp,
                        pool=pool_xp,
                        actions=xp_acts,
                        heuristic_score=self.compute_fast_heuristic(p_xp),
                    )
                )
        candidates.extend(xp_candidates)

        # Phase 3: Auto-Deploy to maximize board capacity
        for cand in candidates:
            if cand.player.board_unit_count < cand.player.max_board_units:
                units_deployed = cand.player.auto_fill_board_from_bench(cand.pool)
                if units_deployed > 0:
                    for _ in range(units_deployed):
                        cand.actions.append(45)

        # Phase 3.5: Branch item-to-unit assignments. Player.equip_item also crafts a
        # completed item when the target already holds a compatible component.
        candidates.sort(key=lambda n: n.heuristic_score, reverse=True)
        candidates = candidates[: self.beam_width]

        # Consider a single legacy item action per turn. Its recipient follows the
        # same priority rule as execute_action, so planning and execution agree.
        base_candidates = list(candidates)
        item_candidates: list[PlannerNode] = []
        for cand in base_candidates[:2]:
            for item_slot in range(min(10, len(cand.player.item_bench))):
                p_item = cand.player.clone()
                action_id = 101 + item_slot
                if not execute_action(p_item, cand.pool.clone(), self.set_data, action_id):
                    continue
                item_candidates.append(
                    PlannerNode(
                        player=p_item,
                        pool=cand.pool.clone(),
                        actions=list(cand.actions) + [action_id],
                        heuristic_score=self.compute_fast_heuristic(p_item),
                    )
                )

        # Always retain item leaves in the V shortlist. The fast heuristic is
        # intentionally item-agnostic, so it must not be allowed to erase them.
        if item_candidates:
            item_candidates.sort(key=lambda n: n.heuristic_score, reverse=True)
            item_candidates = item_candidates[:4]
            if getattr(self, "use_neural_eval", False) and self.encoder is not None:
                for cand in item_candidates:
                    cand.neural_score = self.evaluate_neural_score(
                        cand.player, stage, round_in_stage, target_z, s_hat_next
                    )
                item_candidates.sort(key=lambda n: n.neural_score, reverse=True)
            candidates = base_candidates[: self.beam_width - 1] + item_candidates[:1]

        # Phase 4: Compare the best fast candidate against the best item leaf
        # through V. This replaces the previous all-leaf neural evaluation.
        candidates.sort(key=lambda n: n.heuristic_score, reverse=True)
        best_fast_node = candidates[0] if candidates else root_node
        best_node = best_fast_node
        if getattr(self, "use_neural_eval", False) and self.encoder is not None:
            best_node.neural_score = self.evaluate_neural_score(
                best_node.player, stage, round_in_stage, target_z, s_hat_next
            )
            for cand in candidates:
                if cand.actions and cand.actions[-1] >= 101 and cand.neural_score > best_node.neural_score:
                    best_node = cand

        # Phase 6: Reroll Check (Action 6)
        final_actions = list(best_node.actions)

        # Evaluate if rerolling after transactions is beneficial:
        # Conditions: Excess gold >= 50 (or free rerolls) and not late game
        if best_node.player.gold >= 50 or best_node.player.free_rerolls > 0:
            if len(final_actions) < 10:  # within action budget
                final_actions.append(6)  # REROLL_SHOP

        return final_actions
