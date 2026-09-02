"""Gymnasium reinforcement learning environment wrapper for Teamfight Tactics."""

from __future__ import annotations

import random
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from tft_ai_player.simulation.actions import TOTAL_DISCRETE_ACTIONS, get_action_mask
from tft_ai_player.simulation.bots import BaseBot, GreedyBankerBot, RandomBot, StandardTempoBot
from tft_ai_player.simulation.combat import CombatResolver, HeuristicCombatResolver
from tft_ai_player.simulation.config import AgentArchetype, SetData, get_default_set17_data
from tft_ai_player.simulation.game import TFTGame
from tft_ai_player.simulation.observations import ObservationEncoder


class CurriculumBotFactory:
    """Dynamically samples opponent bots using Prioritized Fictitious Self-Play (PFSP) and anchor baselines."""

    def __init__(
        self,
        generation: int = 1,
        league: Any | None = None,
        set_data: SetData | None = None,
        device: torch.device | None = None,
        active_model: Any | None = None,
        focal_role: Any | None = None,
        focal_agent_id: str = "main_agent_v1",
    ) -> None:
        self.generation = generation
        self.league = league
        self.set_data = set_data or get_default_set17_data()
        self.device = device or torch.device("cpu")
        self.active_model = active_model
        self.focal_role = focal_role
        self.focal_agent_id = focal_agent_id
        self._model_cache: dict[str, Any] = {}

    def set_generation(self, gen: int) -> None:
        self.generation = gen

    def set_active_model(self, model: Any) -> None:
        self.active_model = model

    def _resolve_bot_by_id(self, agent_id: str) -> BaseBot:
        """Resolve an agent_id to an executable bot policy."""
        # 1. Check if it's active self-play
        if agent_id == self.focal_agent_id or agent_id == "active_self":
            if self.active_model is not None:
                try:
                    from tft_ai_player.rl.agent_policy import RLBot
                    return RLBot(model=self.active_model, set_data=self.set_data, device=self.device)
                except Exception:
                    pass

        # 2. Check league profile metadata
        if self.league and agent_id in self.league.profiles:
            prof = self.league.profiles[agent_id]
            b_class = prof.metadata.get("bot_class")
            if b_class == "StandardTempoBot":
                return StandardTempoBot()
            if b_class == "GreedyBankerBot":
                return GreedyBankerBot()
            if b_class == "RandomBot":
                return RandomBot()

            # Check if it has a saved checkpoint (Hall of Fame snapshot)
            if prof.checkpoint_path:
                try:
                    from pathlib import Path
                    cp_path = Path(prof.checkpoint_path)
                    if cp_path.exists():
                        if prof.checkpoint_path in self._model_cache:
                            cached_model = self._model_cache[prof.checkpoint_path]
                        else:
                            from tft_ai_player.rl.models.networks import TFTActorCritic
                            cached_model = TFTActorCritic(
                                num_champs=len(self.set_data.champions) + 1,
                                num_items=len(self.set_data.items) + 1,
                                hidden_dim=384,
                            )
                            state_dict = torch.load(prof.checkpoint_path, map_location=self.device, weights_only=True)
                            cached_model.load_state_dict(state_dict)
                            cached_model.to(self.device)
                            cached_model.eval()
                            self._model_cache[prof.checkpoint_path] = cached_model

                        from tft_ai_player.rl.agent_policy import RLBot
                        return RLBot(model=cached_model, set_data=self.set_data, device=self.device)
                except Exception:
                    pass

        # 3. Default fallback anchor bot
        return StandardTempoBot()

    def __call__(self) -> BaseBot:
        if self.league and hasattr(self.league, "sample_role_pfsp_opponents"):
            from tft_ai_player.rl.types import AgentRole
            role = self.focal_role or AgentRole.MAIN
            try:
                opp_ids = self.league.sample_role_pfsp_opponents(
                    focal_role=role,
                    focal_agent_id=self.focal_agent_id,
                    num_opponents=1,
                )
                if opp_ids:
                    return self._resolve_bot_by_id(opp_ids[0])
            except Exception:
                pass

        if self.active_model is not None:
            try:
                from tft_ai_player.rl.agent_policy import RLBot
                return RLBot(model=self.active_model, set_data=self.set_data, device=self.device)
            except Exception:
                pass
        return StandardTempoBot()


class TFTEnv(gym.Env):
    """Reinforcement learning environment for training autonomous TFT agents.

    Compliant with the standard Gymnasium (v1.0+) interface with invalid action masking.
    """

    metadata = {"render_modes": ["human", "ansi", "text"]}

    def __init__(
        self,
        set_data: SetData | None = None,
        combat_resolver: CombatResolver | None = None,
        bot_factory: Any | None = None,
        archetype: AgentArchetype = AgentArchetype.GENERALIST,
        use_flat_obs: bool = False,
        render_mode: str | None = None,
        max_actions_per_round: int = 15,
    ) -> None:
        super().__init__()

        self.set_data = set_data or get_default_set17_data()
        self.combat_resolver = combat_resolver or HeuristicCombatResolver()
        self.bot_factory = bot_factory or CurriculumBotFactory(generation=1)
        self.archetype = archetype
        self.use_flat_obs = use_flat_obs
        self.render_mode = render_mode
        self.max_actions_per_round = max_actions_per_round

        self.encoder = ObservationEncoder(self.set_data)
        self.game = TFTGame(
            set_data=self.set_data,
            combat_resolver=self.combat_resolver,
            bot_factory=self.bot_factory,
        )

        self.actions_in_current_round: int = 0
        self.last_health: int = 100

        # Action Space: 1721 Discrete Actions
        self.action_space = spaces.Discrete(TOTAL_DISCRETE_ACTIONS)

        # Observation Space
        if self.use_flat_obs:
            self.observation_space = spaces.Box(
                low=0.0,
                high=1000.0,
                shape=(self.encoder.flat_observation_dim,),
                dtype=np.float32,
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "player_stats": spaces.Box(low=0.0, high=1000.0, shape=(9,), dtype=np.float32),
                    "board": spaces.Box(low=0.0, high=500.0, shape=(28, 5), dtype=np.float32),
                    "bench": spaces.Box(low=0.0, high=500.0, shape=(9, 5), dtype=np.float32),
                    "item_bench": spaces.Box(low=0.0, high=500.0, shape=(10,), dtype=np.float32),
                    "shop": spaces.Box(low=0.0, high=500.0, shape=(5,), dtype=np.float32),
                    "opponents": spaces.Box(low=0.0, high=1000.0, shape=(7, 8), dtype=np.float32),
                    "opponents_boards": spaces.Box(low=0.0, high=500.0, shape=(7, 28, 5), dtype=np.float32),
                    "opponents_benches": spaces.Box(low=0.0, high=500.0, shape=(7, 9, 5), dtype=np.float32),
                    "pool_counts": spaces.Box(low=0.0, high=100.0, shape=(self.encoder.num_champs,), dtype=np.float32),
                    "action_mask": spaces.Box(low=0, high=1, shape=(TOTAL_DISCRETE_ACTIONS,), dtype=bool),
                }
            )

    def set_curriculum_generation(self, generation: int) -> None:
        """Update opponent difficulty curriculum / self-play sampling generation."""
        if hasattr(self.bot_factory, "set_generation"):
            self.bot_factory.set_generation(generation)
        elif hasattr(self.bot_factory, "generation"):
            self.bot_factory.generation = generation

    def _get_obs(self) -> dict[str, np.ndarray] | np.ndarray:
        focal = self.game.get_focal_player()
        opponents = self.game.get_opponents(focal.player_id)
        if self.use_flat_obs:
            return self.encoder.encode_flat(focal, opponents, self.game.pool, self.game.stage_manager)
        return self.encoder.encode_dict(focal, opponents, self.game.pool, self.game.stage_manager)

    def _get_info(self) -> dict[str, Any]:
        focal = self.game.get_focal_player()
        mask = get_action_mask(focal, self.set_data)
        rinfo = self.game.stage_manager.get_current_round_info()
        return {
            "action_mask": mask,
            "stage": rinfo.stage_str,
            "round_type": rinfo.round_type.value,
            "health": focal.health,
            "gold": focal.gold,
            "level": focal.level,
            "placement": focal.placement,
            "alive": focal.alive,
            "is_over": self.game.is_over,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray] | np.ndarray, dict[str, Any]]:
        """Reset the environment to the beginning of a fresh TFT match."""
        super().reset(seed=seed)
        self.game.reset(seed=seed)
        self.actions_in_current_round = 0
        self.last_health = self.game.get_focal_player().health
        self.prev_stage = self.game.stage_manager.stage
        self.last_eliminated_count = 0

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode in ("human", "ansi", "text"):
            self.render()

        return obs, info

    def _compute_state_potential(self, player: Player) -> float:
        """Potential-Based State Function Phi(s) following Ng, Harada, Russell (1999).

        Guarantees policy invariance while accelerating credit assignment across micro-actions.
        Decomposes state into:
        1. Combat Board Power Potential (active fielded units, star tiers, equipped items, synergies)
        2. Total Net Worth Potential (liquid gold + asset value of all owned champions + item inventory)
        """
        from tft_ai_player.simulation.combat import HeuristicCombatResolver

        # 1. Combat Board Power Potential
        evaluator = HeuristicCombatResolver()
        board_power = evaluator.compute_player_power(player, self.set_data)
        w_combat = 0.30
        if self.archetype == AgentArchetype.HYPER_ROLL:
            w_combat = 0.45
        elif self.archetype == AgentArchetype.FAST8_FLEX:
            w_combat = 0.20
        phi_combat = (board_power / 600.0) * w_combat

        # 2. Total Net Worth Potential
        champ_gold_value = sum(
            (self.set_data.champions[u.champion_id].cost if u.champion_id in self.set_data.champions else 1) * (3 ** (u.star_level - 1))
            for u in player.get_all_units()
        )
        item_gold_value = len(player.item_bench) * 2.0
        total_assets = player.gold + champ_gold_value + item_gold_value
        w_econ = 0.10
        if self.archetype == AgentArchetype.FAST8_FLEX:
            w_econ = 0.20
        elif self.archetype == AgentArchetype.HYPER_ROLL:
            w_econ = 0.05
        phi_econ = (total_assets / 200.0) * w_econ

        return phi_combat + phi_econ

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray] | np.ndarray, float, bool, bool, dict[str, Any]]:
        """Perform a discrete micro-action or advance the round on PASS (0)."""
        focal = self.game.get_focal_player()

        reward_breakdown = {
            # 1. Primary Game Objectives (Zero-Sum Outcomes & Tournament Payoffs)
            "rew_round_win": 0.0,
            "rew_round_loss": 0.0,
            "rew_placement": 0.0,
            "rew_hp_loss": 0.0,
            "rew_elimination_bounty": 0.0,
            "rew_stage_survival": 0.0,
            # 2. Potential-Based State Transition Deltas (Ng et al. 1999)
            "rew_potential_delta": 0.0,
            # 3. Macro Turn Inactivity Constraints
            "rew_penalty_empty_board": 0.0,
            "rew_penalty_hoarding": 0.0,
            # Grouped Blocks for WandB & Analytics
            "block_combat_outcome": 0.0,
            "block_board_power": 0.0,
            "block_constraints_economy": 0.0,
        }

        if not focal.alive or self.game.is_over:
            obs = self._get_obs()
            info = self._get_info()
            info["reward_breakdown"] = reward_breakdown
            return obs, 0.0, True, False, info

        self.actions_in_current_round += 1
        round_advanced = False
        pass_diagnostics = {}

        # Check if action is PASS (0) or exceeded action limit
        if action == 0 or self.actions_in_current_round >= self.max_actions_per_round:
            round_advanced = True

            # Diagnose Pass context
            num_components = sum(1 for it in focal.item_bench if it.is_component)
            has_craftable_pair = False
            if num_components >= 2:
                comp_ids = [it.item_id for it in focal.item_bench if it.is_component]
                for i in range(len(comp_ids)):
                    for j in range(i + 1, len(comp_ids)):
                        if self.set_data.get_recipe_result(comp_ids[i], comp_ids[j]) is not None:
                            has_craftable_pair = True
                            break
                    if has_craftable_pair:
                        break

            has_upgradable_shop = False
            for s_idx, champ_id in enumerate(focal.shop.slots):
                if champ_id and focal.can_buy_champion(s_idx):
                    existing = [u for u in focal.get_all_units() if u.champion_id == champ_id]
                    if len(existing) == 2 and existing[0].star_level == 1:
                        has_upgradable_shop = True
                        break

            pass_diagnostics = {
                "round_actions_taken": self.actions_in_current_round - 1,
                "pass_with_uncombined_items": has_craftable_pair,
                "pass_with_upgradable_shop": has_upgradable_shop,
                "pass_excess_gold": focal.gold > 50,
            }

            # 1. Stage Survival Bonus: Reward surviving deeper into the match
            curr_stage = self.game.stage_manager.stage
            if curr_stage > self.prev_stage:
                reward_breakdown["rew_stage_survival"] = 0.05 * (curr_stage - self.prev_stage)
                self.prev_stage = curr_stage

            # 2. Anti-Empty-Board Penalty in PvP (Fielding 0 units when bench has units)
            rinfo_now = self.game.stage_manager.get_current_round_info()
            if rinfo_now.is_pvp:
                bench_units_count = sum(1 for u in focal.bench if u is not None)
                missing_board_slots = max(0, focal.max_board_units - focal.board_unit_count)
                fieldable_unplaced = min(missing_board_slots, bench_units_count)
                if fieldable_unplaced > 0:
                    reward_breakdown["rew_penalty_empty_board"] -= 0.10 * fieldable_unplaced
                if focal.board_unit_count == 0:
                    reward_breakdown["rew_penalty_empty_board"] -= 0.25

            # 3. Surplus Gold Inactivity (Passing with >50g where interest is capped)
            if focal.gold > 50:
                reward_breakdown["rew_penalty_hoarding"] -= 0.01 * min(5, (focal.gold - 50) // 10)

            # 4. Opponent AI planning phase
            self.game.execute_bot_turns()

            # 5. Resolve round combat & advance
            health_before = focal.health
            combat_results = self.game.resolve_round_phase()
            health_after = focal.health
            delta_hp = health_after - health_before

            # Round outcome combat reward
            focal_combat = next(
                (m for m in combat_results if m.winner_id == focal.player_id or m.loser_id == focal.player_id),
                None,
            )
            if focal_combat is not None:
                if focal_combat.winner_id == focal.player_id:
                    reward_breakdown["rew_round_win"] = 0.40
                elif focal_combat.loser_id == focal.player_id:
                    reward_breakdown["rew_round_loss"] = -0.30

            if delta_hp < 0:
                reward_breakdown["rew_hp_loss"] = (delta_hp / 100.0) * 1.0

            # Opponent Elimination Bounty (Surviving while lobby rivals die)
            eliminated_rivals = sum(1 for p in self.game.players if p.player_id != focal.player_id and not p.alive and p.health <= 0)
            delta_eliminated = max(0, eliminated_rivals - self.last_eliminated_count)
            if delta_eliminated > 0 and focal.alive:
                reward_breakdown["rew_elimination_bounty"] = 0.20 * delta_eliminated
            self.last_eliminated_count = eliminated_rivals

            # Tournament placement reward on game end / elimination
            if not focal.alive or self.game.is_over:
                placement = focal.placement or (1 if focal.alive else 8)
                placement_rewards = {
                    1: 2.0,
                    2: 1.2,
                    3: 0.6,
                    4: 0.2,
                    5: -0.2,
                    6: -0.6,
                    7: -1.2,
                    8: -2.0,
                }
                reward_breakdown["rew_placement"] = placement_rewards.get(placement, 0.0)

            self.actions_in_current_round = 0
        else:
            # Potential-Based Reward Shaping on Micro-Action (Ng et al. 1999)
            phi_before = self._compute_state_potential(focal)
            valid = self.game.step_player_action(focal.player_id, action)
            if valid:
                phi_after = self._compute_state_potential(focal)
                delta_phi = phi_after - phi_before
                reward_breakdown["rew_potential_delta"] = float(np.clip(delta_phi, -0.4, 0.4))

        # Grouped category blocks for WandB & Analytics
        reward_breakdown["block_combat_outcome"] = (
            reward_breakdown["rew_round_win"]
            + reward_breakdown["rew_round_loss"]
            + reward_breakdown["rew_placement"]
            + reward_breakdown["rew_hp_loss"]
            + reward_breakdown["rew_elimination_bounty"]
            + reward_breakdown["rew_stage_survival"]
        )
        reward_breakdown["block_board_power"] = reward_breakdown["rew_potential_delta"]
        reward_breakdown["block_constraints_economy"] = (
            reward_breakdown["rew_penalty_empty_board"]
            + reward_breakdown["rew_penalty_hoarding"]
        )

        total_reward = (
            reward_breakdown["block_combat_outcome"]
            + reward_breakdown["block_board_power"]
            + reward_breakdown["block_constraints_economy"]
        )

        terminated = not focal.alive or self.game.is_over
        truncated = False
        obs = self._get_obs()
        info = self._get_info()
        info["reward_breakdown"] = reward_breakdown
        info["round_advanced"] = round_advanced
        info["pass_diagnostics"] = pass_diagnostics

        if self.render_mode in ("human", "ansi", "text"):
            self.render()

        return obs, total_reward, terminated, truncated, info

    def render(self) -> str | None:
        """Render ASCII representation of current game state."""
        focal = self.game.get_focal_player()
        rinfo = self.game.stage_manager.get_current_round_info()

        lines: list[str] = []
        lines.append("=" * 70)
        lines.append(
            f" [TFT SIMULATION] | Stage: {rinfo.stage_str} ({rinfo.round_type.value}) | Round: {self.game.stage_manager.total_rounds_elapsed}"
        )
        lines.append("=" * 70)
        lines.append(
            f" HP: {focal.health}/100 | Gold: {focal.gold}g | Level: {focal.level} (XP: {focal.exp}/{self.set_data.level_exp.get(focal.level, 'MAX')}) | Streak: {focal.streak:+d}"
        )
        lines.append("-" * 70)

        # Board Grid
        lines.append(" FIELDED BOARD (4x7 Grid):")
        for r in range(self.set_data.board_rows):
            row_str = f"  Row {r}: "
            hex_cells: list[str] = []
            for c in range(self.set_data.board_cols):
                unit = focal.board.get((r, c))
                if unit:
                    star_str = "*" * unit.star_level
                    c_short = unit.champion_id.replace("TFT17_", "")[:7]
                    item_count = len(unit.items)
                    hex_cells.append(f"[{c_short} {star_str}|{item_count}i]")
                else:
                    hex_cells.append("[   .   ]")
            lines.append(row_str + " ".join(hex_cells))

        # Bench
        lines.append("-" * 70)
        bench_cells: list[str] = []
        for idx, slot in enumerate(focal.bench):
            if slot:
                star_str = "*" * slot.star_level
                c_short = slot.champion_id.replace("TFT17_", "")[:6]
                bench_cells.append(f"({idx}:{c_short} {star_str})")
            else:
                bench_cells.append(f"({idx}: - )")
        lines.append(" BENCH: " + " ".join(bench_cells))

        # Items
        item_names = [it.name for it in focal.item_bench]
        lines.append(f" ITEM BENCH ({len(focal.item_bench)}/10): {', '.join(item_names) if item_names else 'Empty'}")

        # Shop
        shop_cards: list[str] = []
        for idx, card in enumerate(focal.shop.slots):
            if card:
                cdef = self.set_data.champions.get(card)
                cost = cdef.cost if cdef else 1
                c_short = card.replace("TFT17_", "")
                shop_cards.append(f"[{idx}] {c_short} ({cost}g)")
            else:
                shop_cards.append(f"[{idx}] (Bought)")
        lock_str = "LOCKED" if focal.shop.locked else "Unlocked"
        lines.append(f" SHOP ({lock_str}): " + " | ".join(shop_cards))

        # Active Traits
        active_traits = focal.get_active_traits()
        active_str = ", ".join(f"{t}: Tier {tier}" for t, tier in active_traits.items() if tier > 0)
        lines.append(f" ACTIVE TRAITS: {active_str if active_str else 'None'}")

        # Opponents Standings
        lines.append("-" * 70)
        lines.append(" LOBBY STANDINGS:")
        for p in self.game.players:
            status = f"Dead (#{p.placement})" if not p.alive else f"HP: {p.health} | Lv:{p.level} | {p.gold}g | Board: {len(p.board)}u"
            is_focal_marker = "-> " if p.player_id == focal.player_id else "   "
            lines.append(f"  {is_focal_marker}Player {p.player_id}: {status}")

        lines.append("=" * 70)

        output = "\n".join(lines)
        if self.render_mode == "human":
            print(output)
        return output
