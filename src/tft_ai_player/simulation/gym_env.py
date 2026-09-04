"""Gymnasium reinforcement learning environment for Teamfight Tactics.

Implements the Gymnasium Decision MDP with:
  - 704D invariant concatenated state observation (320D core + 64D shop + 64D bench + 256D target Z)
  - 111-action factorized discrete action space with strict pre-softmax masking
  - Tabular calibrated LightGBM combat oracle (1,107 dims)
  - Multi-objective reward: R_step = R_env + alpha * R_macro + beta * R_micro
"""

from __future__ import annotations

import random
from typing import Any, Mapping

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn.functional as F

from tft_ai_player.embeddings.model import MultiModalFusionTrunk
from tft_ai_player.embeddings.vocab import ChampionVocabulary, ItemVocabulary, TraitVocabulary
from tft_ai_player.simulation.actions import (
    TOTAL_DISCRETE_ACTIONS,
    execute_action,
    get_action_mask,
)
from tft_ai_player.simulation.bots import BaseBot, BotAlphaFast8, BotBetaHyperroll, BotGammaGreedy, RandomBot
from tft_ai_player.simulation.combat import CombatResolver, HeuristicCombatResolver, MLCombatResolver
from tft_ai_player.simulation.config import AgentArchetype, SetData, get_default_set17_data
from tft_ai_player.simulation.game import TFTGame
from tft_ai_player.simulation.models import ChampionInstance, Player


class TFTStateEncoder:
    """Extracts 704D invariant observation vector using frozen MultiModalFusionTrunk."""

    def __init__(
        self,
        set_data: SetData,
        trunk: MultiModalFusionTrunk | None = None,
        vocab: ChampionVocabulary | None = None,
        item_vocab: ItemVocabulary | None = None,
        trait_vocab: TraitVocabulary | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.set_data = set_data
        self.device = device or torch.device("cpu")
        self.vocab = vocab or ChampionVocabulary()
        self.item_vocab = item_vocab or ItemVocabulary()
        self.trait_vocab = trait_vocab or TraitVocabulary()

        if trunk is None:
            self.trunk = MultiModalFusionTrunk(
                num_champs=len(self.vocab) + 10,
                num_items=len(self.item_vocab) + 10,
                num_traits=len(self.trait_vocab) + 10,
            )
            self.trunk.eval()
        else:
            self.trunk = trunk
        self.trunk.to(self.device)
        self.trunk.eval()

        # Learnable / fixed projection layers for shop (160->64) and bench (64->64)
        self.shop_proj = torch.nn.Sequential(
            torch.nn.Linear(160, 64),
            torch.nn.LayerNorm(64),
            torch.nn.ReLU(),
        ).to(self.device)

        self.bench_proj = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.LayerNorm(64),
            torch.nn.ReLU(),
        ).to(self.device)

    def encode_board_tensors(self, player: Player) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert player board into PyTorch tensors for trunk ingestion."""
        board_champ_ids = torch.zeros((4, 7), dtype=torch.long, device=self.device)
        board_star_levels = torch.zeros((4, 7), dtype=torch.long, device=self.device)
        board_item_ids = torch.zeros((4, 7, 3), dtype=torch.long, device=self.device)
        board_traits = torch.zeros((len(self.trait_vocab) + 10,), dtype=torch.float32, device=self.device)

        for (r, c), unit in player.board.items():
            if 0 <= r < 4 and 0 <= c < 7:
                c_idx = self.vocab.encode(unit.champion_id)
                board_champ_ids[r, c] = c_idx
                board_star_levels[r, c] = unit.star_level
                for it_slot, item_id in enumerate(unit.items[:3]):
                    board_item_ids[r, c, it_slot] = self.item_vocab.encode(item_id)

        active_traits = player.get_active_traits()
        for trait_name, tier in active_traits.items():
            t_idx = self.trait_vocab.encode(trait_name)
            if t_idx < board_traits.shape[0]:
                board_traits[t_idx] = float(tier)

        return board_champ_ids, board_star_levels, board_item_ids, board_traits

    def encode_scalars(self, player: Player, stage: int, round_in_stage: int) -> torch.Tensor:
        """Encode 8 continuous match state scalars."""
        scalars = torch.tensor(
            [
                player.health / 100.0,
                player.gold / 100.0,
                player.level / 10.0,
                player.streak / 10.0,
                stage / 10.0,
                round_in_stage / 10.0,
                player.board_unit_count / 10.0,
                len(player.item_bench) / 10.0,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        return scalars

    @torch.no_grad()
    def extract_state_vector(
        self,
        player: Player,
        stage: int,
        round_in_stage: int,
        target_z: np.ndarray | torch.Tensor | None = None,
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Construct full 704D state vector o_t = [s_t (320), shop_feat (64), bench_feat (64), target_z (256)].

        Returns:
            o_t_np: (704,) np.ndarray float32
            s_t: (320,) torch.Tensor
            h_board: (256,) torch.Tensor
        """
        c_ids, stars, items, traits = self.encode_board_tensors(player)
        scalars = self.encode_scalars(player, stage, round_in_stage)

        # 1. Core State (320D)
        h_board = self.trunk.encode_board(
            board_champ_ids=c_ids.unsqueeze(0),
            board_star_levels=stars.unsqueeze(0),
            board_item_ids=items.unsqueeze(0),
            board_traits=traits.unsqueeze(0),
        ).squeeze(0)  # (256D)

        state_feat = self.trunk.state_mlp(scalars.unsqueeze(0)).squeeze(0)  # (64D)
        s_t = self.trunk.fusion(torch.cat([h_board, state_feat], dim=-1).unsqueeze(0)).squeeze(0)  # (320D)

        # 2. Shop State (64D)
        shop_champ_ids = torch.zeros((5,), dtype=torch.long, device=self.device)
        for i, cid in enumerate(player.shop.slots):
            if cid is not None:
                shop_champ_ids[i] = self.vocab.encode(cid)

        shop_tokens = self.trunk.champ2vec(shop_champ_ids.unsqueeze(0)).view(1, -1)  # (1, 160)
        shop_feat = self.shop_proj(shop_tokens).squeeze(0)  # (64D)

        # 3. Bench State (64D)
        bench_champ_ids = torch.zeros((9,), dtype=torch.long, device=self.device)
        bench_stars = torch.zeros((9,), dtype=torch.long, device=self.device)
        bench_items = torch.zeros((9, 3), dtype=torch.long, device=self.device)

        for i, u in enumerate(player.bench):
            if u is not None:
                bench_champ_ids[i] = self.vocab.encode(u.champion_id)
                bench_stars[i] = u.star_level
                for slot_idx, itm in enumerate(u.items[:3]):
                    bench_items[i, slot_idx] = self.item_vocab.encode(itm)


        bench_tokens = self.trunk.champ2vec(
            champ_ids=bench_champ_ids.unsqueeze(0),
            star_levels=bench_stars.unsqueeze(0),
            item_ids=bench_items.unsqueeze(0),
        ).squeeze(0)  # (9, 32)

        sum_pool = bench_tokens.sum(dim=0)  # (32D)
        mean_pool = bench_tokens.mean(dim=0)  # (32D)
        bench_deepsets = torch.cat([sum_pool, mean_pool], dim=-1).unsqueeze(0)  # (1, 64)
        bench_feat = self.bench_proj(bench_deepsets).squeeze(0)  # (64D)

        # 4. Target Z Conditioning (256D)
        if target_z is not None:
            if isinstance(target_z, np.ndarray):
                z_tensor = torch.as_tensor(target_z, dtype=torch.float32, device=self.device)
            else:
                z_tensor = target_z.to(self.device).float()
            if z_tensor.numel() != 256:
                z_tensor = torch.zeros((256,), dtype=torch.float32, device=self.device)
        else:
            z_tensor = torch.zeros((256,), dtype=torch.float32, device=self.device)

        o_t = torch.cat([s_t, shop_feat, bench_feat, z_tensor], dim=-1)
        o_t_np = o_t.detach().cpu().numpy().astype(np.float32)

        return o_t_np, s_t, h_board


class CurriculumBotFactory:
    """Curriculum opponent sampling from league pool and benchmark baselines."""

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

    def __call__(self) -> BaseBot:
        if self.league and hasattr(self.league, "sample_opponent_bot"):
            return self.league.sample_opponent_bot(self.focal_agent_id)
        # Default fallback: Bot Alpha
        return BotAlphaFast8()


class TFTEnv(gym.Env):
    """Gymnasium reinforcement learning environment for Teamfight Tactics (|A|=111, Obs=704D)."""

    metadata = {"render_modes": ["human", "ansi", "text"]}

    def __init__(
        self,
        set_data: SetData | None = None,
        combat_resolver: CombatResolver | None = None,
        bot_factory: Any | None = None,
        trunk: MultiModalFusionTrunk | None = None,
        world_model: torch.nn.Module | None = None,
        target_z: np.ndarray | None = None,
        z_centroids: np.ndarray | None = None,
        sample_random_z: bool = False,
        alpha: float = 0.0,
        beta: float = 0.3,
        max_actions_per_round: int = 15,
        device: str | torch.device | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.set_data = set_data or get_default_set17_data()
        self.combat_resolver = combat_resolver or HeuristicCombatResolver()
        self.bot_factory = bot_factory or CurriculumBotFactory(generation=1)
        self.initial_alpha = float(alpha)
        self.initial_beta = float(beta)
        self.current_alpha = float(alpha)
        self.current_beta = float(beta)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.max_actions_per_round = max_actions_per_round
        self.render_mode = render_mode
        self.target_z = target_z
        self.z_centroids = z_centroids
        self.sample_random_z = sample_random_z
        self.current_target_z_index: int | None = None

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder = TFTStateEncoder(set_data=self.set_data, trunk=trunk, device=self.device)
        self.world_model = world_model.to(self.device).eval() if world_model is not None else None

        self.game = TFTGame(
            set_data=self.set_data,
            combat_resolver=self.combat_resolver,
            bot_factory=self.bot_factory,
        )

        self.action_space = spaces.Discrete(TOTAL_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(
            low=-100.0,
            high=100.0,
            shape=(704,),
            dtype=np.float32,
        )

        self.actions_in_current_round: int = 0
        self.current_s_t: torch.Tensor | None = None
        self.current_s_hat_next: torch.Tensor | None = None
        self.current_h_board: torch.Tensor | None = None

    def update_reward_weights(self, progress_fraction: float, min_alpha: float = 0.0, min_beta: float = 0.05) -> None:
        """Decay auxiliary representation weights (alpha and beta) linearly down to minimum floor.

        progress_fraction: 0.0 at the beginning of training, 1.0 at max_generations.
        """
        progress = max(0.0, min(1.0, float(progress_fraction)))
        self.current_alpha = max(min_alpha, self.initial_alpha * (1.0 - progress))
        self.current_beta = max(min_beta, self.initial_beta * (1.0 - progress))
        self.alpha = self.current_alpha
        self.beta = self.current_beta

    def set_target_z(self, z_k: np.ndarray | None, alpha: float = 0.8, beta: float = 0.2, z_index: int | None = None) -> None:
        """Assign specialist target archetype Z-Index centroid."""
        self.target_z = z_k
        self.current_target_z_index = z_index
        self.initial_alpha = alpha
        self.initial_beta = beta
        self.current_alpha = alpha
        self.current_beta = beta
        self.alpha = alpha
        self.beta = beta

    def _sample_new_target_z(self) -> None:
        """Sample a new target composition centroid or unforced flex (z=0) with 50/50 probability."""
        if self.z_centroids is not None and len(self.z_centroids) > 0:
            # 50% of games: Train on forcing a specific High-Elo composition (z_k)
            # 50% of games: Train on unforced flexible adaptation (z=None, alpha=0.0)
            if np.random.random() < 0.5:
                k = int(np.random.randint(0, len(self.z_centroids)))
                self.target_z = self.z_centroids[k]
                self.current_target_z_index = k
                self.current_alpha = self.alpha
            else:
                self.target_z = None
                self.current_target_z_index = None
                self.current_alpha = 0.0

    def _get_obs_and_masks(self) -> tuple[np.ndarray, np.ndarray]:
        focal = self.game.get_focal_player()
        stage = self.game.stage_manager.stage
        round_in_stage = self.game.stage_manager.round_in_stage

        obs_vec, s_t, h_board = self.encoder.extract_state_vector(
            player=focal,
            stage=stage,
            round_in_stage=round_in_stage,
            target_z=self.target_z,
        )
        self.current_s_t = s_t
        self.current_h_board = h_board

        # Project strategic oracle ŝ_{t+1} via world model
        if self.world_model is not None and self.current_s_hat_next is None:
            with torch.no_grad():
                self.current_s_hat_next = self.world_model(s_t.unsqueeze(0)).squeeze(0)

        mask = get_action_mask(focal, self.set_data)
        return obs_vec, mask

    def _get_info(self, mask: np.ndarray) -> dict[str, Any]:
        focal = self.game.get_focal_player()
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
            "target_z_index": self.current_target_z_index,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.game.reset(seed=seed)
        self.actions_in_current_round = 0
        self.current_s_hat_next = None

        if options and "target_z" in options:
            self.target_z = options["target_z"]
            self.current_target_z_index = options.get("target_z_index")
        elif self.sample_random_z:
            self._sample_new_target_z()

        obs, mask = self._get_obs_and_masks()
        info = self._get_info(mask)
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        focal = self.game.get_focal_player()
        terminated = False
        truncated = False
        reward = 0.0

        is_pass = (action == 0) or (self.actions_in_current_round >= self.max_actions_per_round)

        if not is_pass:
            # 1. Sequential Decision Execution
            gold_before = focal.gold
            prev_s_t = self.current_s_t
            prev_h_board = self.current_h_board

            success = execute_action(focal, self.game.pool, self.set_data, action)
            self.actions_in_current_round += 1

            if success:
                # Dense potential-based reward shaping per micro-action:
                stage = self.game.stage_manager.stage
                round_in_stage = self.game.stage_manager.round_in_stage
                _, s_next, h_board_next = self.encoder.extract_state_vector(
                    player=focal,
                    stage=stage,
                    round_in_stage=round_in_stage,
                    target_z=self.target_z,
                )

                # Micro alignment delta towards World Model predicted state ŝ_{t+1}:
                r_micro_step = 0.0
                if self.current_s_hat_next is not None and prev_s_t is not None:
                    cos_prev = float(F.cosine_similarity(prev_s_t.unsqueeze(0), self.current_s_hat_next.unsqueeze(0)).item())
                    cos_next = float(F.cosine_similarity(s_next.unsqueeze(0), self.current_s_hat_next.unsqueeze(0)).item())
                    r_micro_step = cos_next - cos_prev

                # Macro alignment delta towards Target Archetype centroid:
                r_macro_step = 0.0
                if self.target_z is not None and prev_h_board is not None:
                    z_target_tensor = torch.as_tensor(self.target_z, dtype=torch.float32, device=self.device)
                    if z_target_tensor.norm() > 1e-6:
                        cos_macro_prev = float(F.cosine_similarity(prev_h_board.unsqueeze(0), z_target_tensor.unsqueeze(0)).item())
                        cos_macro_next = float(F.cosine_similarity(h_board_next.unsqueeze(0), z_target_tensor.unsqueeze(0)).item())
                        r_macro_step = cos_macro_next - cos_macro_prev

                # Economy threshold delta (e.g. selling low-priority units to hit 10g/20g/30g/40g/50g):
                interest_prev = min(5, gold_before // 10)
                interest_next = min(5, focal.gold // 10)
                r_interest_step = float(interest_next - interest_prev) * 0.05

                # Total shaped step reward
                reward = (self.current_beta * r_micro_step) + (self.current_alpha * r_macro_step) + r_interest_step
                self.current_s_t = s_next
                self.current_h_board = h_board_next
                r_step_env = r_interest_step
            else:
                # Small penalty for invalid or no-op action
                reward = -0.02
                r_micro_step = 0.0
                r_macro_step = 0.0
                r_interest_step = 0.0
                r_step_env = -0.02

            reward_breakdown = {
                "r_env": r_step_env,
                "r_combat": 0.0,
                "r_interest": r_interest_step,
                "r_terminal": 0.0,
                "r_macro": r_macro_step,
                "r_micro": r_micro_step,
                "alpha": self.current_alpha,
                "beta": self.current_beta,
            }
        else:
            # 2. Planning phase finalized -> Combat Resolution
            hp_before = focal.health
            round_res = self.game.step_round()
            hp_after = focal.health
            delta_hp = max(0, hp_before - hp_after)
            won_combat = (delta_hp == 0 and focal.alive)

            # R_combat
            if won_combat:
                r_combat = 0.5
            else:
                r_combat = -0.02 * delta_hp

            # R_interest
            r_interest = 0.05 if focal.gold >= 50 else 0.0

            # R_terminal
            r_terminal = 0.0
            if not focal.alive or self.game.is_over:
                terminated = True
                placement = focal.placement or 8
                if placement == 1:
                    r_terminal = 2.0
                elif 2 <= placement <= 4:
                    r_terminal = 1.0
                elif 5 <= placement <= 6:
                    r_terminal = -1.0
                else:
                    r_terminal = -2.0

            r_env = r_combat + r_interest + r_terminal

            # 3. Macro Alignment (R_macro) & Micro Alignment (R_micro) at round boundary
            stage = self.game.stage_manager.stage
            round_in_stage = self.game.stage_manager.round_in_stage
            _, s_next, h_board_next = self.encoder.extract_state_vector(
                player=focal,
                stage=stage,
                round_in_stage=round_in_stage,
                target_z=self.target_z,
            )

            r_micro = 0.0
            if self.current_s_hat_next is not None:
                r_micro = float(
                    F.cosine_similarity(s_next.unsqueeze(0), self.current_s_hat_next.unsqueeze(0)).item()
                )

            r_macro = 0.0
            cluster_match = None
            if self.target_z is not None:
                z_target_tensor = torch.as_tensor(self.target_z, dtype=torch.float32, device=self.device)
                if z_target_tensor.norm() > 1e-6:
                    r_macro = float(
                        F.cosine_similarity(h_board_next.unsqueeze(0), z_target_tensor.unsqueeze(0)).item()
                    )
                if self.z_centroids is not None and self.current_target_z_index is not None and len(self.z_centroids) > 0:
                    z_cents = torch.as_tensor(self.z_centroids, dtype=torch.float32, device=self.device)
                    sims = F.cosine_similarity(h_board_next.unsqueeze(0), z_cents, dim=-1)
                    best_k = int(torch.argmax(sims).item())
                    cluster_match = 1.0 if best_k == self.current_target_z_index else 0.0

            # Round outcome reward
            reward = r_env + (self.current_alpha * r_macro) + (self.current_beta * r_micro)
            reward_breakdown = {
                "r_env": r_env,
                "r_combat": r_combat,
                "r_interest": r_interest,
                "r_terminal": r_terminal,
                "r_macro": r_macro,
                "r_micro": r_micro,
                "cluster_match": cluster_match,
                "alpha": self.current_alpha,
                "beta": self.current_beta,
            }

            # Reset planning counter for next round
            self.actions_in_current_round = 0
            self.current_s_hat_next = None

        obs, mask = self._get_obs_and_masks()
        info = self._get_info(mask)
        info["step_reward"] = reward
        info["reward_breakdown"] = reward_breakdown

        return obs, reward, terminated, truncated, info


