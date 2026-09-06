"""Agent Policy wrapper and RLBot adapter enabling RL models to play in TFT simulations."""

from __future__ import annotations

import random
from typing import Any
import numpy as np
import torch

from tft_ai_player.embeddings.model import MultiModalFusionTrunk
from tft_ai_player.rl.models.networks import TFTActorCritic
from tft_ai_player.rl.planner import ShopBeamSearchPlanner
from tft_ai_player.simulation.actions import execute_action, get_action_mask
from tft_ai_player.simulation.config import SetData
from tft_ai_player.simulation.gym_env import TFTStateEncoder
from tft_ai_player.simulation.models import ChampionPool, Player
from tft_ai_player.simulation.stage_manager import StageManager


class ZMetricsTracker:
    """Multi-agent telemetry accumulator for Z-Archetype alignment and cluster matching."""

    def __init__(self) -> None:
        self.macro_sims: list[float] = []
        self.cluster_matches: list[float] = []

    def record(self, r_macro: float, cluster_match: float | None = None) -> None:
        if r_macro > 1e-6:
            self.macro_sims.append(float(r_macro))
        if cluster_match is not None:
            self.cluster_matches.append(float(cluster_match))

    def get_summary(self) -> tuple[float, float]:
        avg_macro = float(np.mean(self.macro_sims)) if self.macro_sims else 0.0
        avg_match = float(np.mean(self.cluster_matches)) * 100.0 if self.cluster_matches else 0.0
        return avg_macro, avg_match

    def reset(self) -> None:
        self.macro_sims.clear()
        self.cluster_matches.clear()


class RLBot:
    """Adapter that allows a neural RL Actor-Critic policy to act as a bot in TFT games."""

    def __init__(
        self,
        model: TFTActorCritic,
        set_data: SetData,
        trunk: MultiModalFusionTrunk | None = None,
        world_model: torch.nn.Module | None = None,
        board_evaluator: torch.nn.Module | None = None,
        target_z: np.ndarray | None = None,
        z_index: int | None = None,
        z_centroids: np.ndarray | None = None,
        metrics_tracker: ZMetricsTracker | None = None,
        deterministic: bool = False,
        max_micro_actions: int = 15,
        device: torch.device | None = None,
        buffer: Any | None = None,
        record_transitions: bool = False,
        is_historical: bool = False,
        alpha: float = 0.5,
        beta: float = 0.3,
        use_planner: bool = True,
    ) -> None:
        self.model = model
        self.set_data = set_data
        self.world_model = world_model
        self.board_evaluator = board_evaluator
        self.target_z = target_z
        self.z_index = z_index
        self.z_centroids = z_centroids
        self.metrics_tracker = metrics_tracker
        self.deterministic = deterministic
        self.max_micro_actions = max_micro_actions
        self.buffer = buffer
        self.is_historical = is_historical
        self.record_transitions = record_transitions and not is_historical
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.use_planner = use_planner
        if device is not None:
            self.device = device
        elif model is not None and len(list(model.parameters())) > 0:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device("cpu")

        self.encoder = TFTStateEncoder(set_data=set_data, trunk=trunk, device=self.device)
        self.planner = ShopBeamSearchPlanner(
            set_data=set_data,
            encoder=self.encoder,
            world_model=world_model,
            board_evaluator=board_evaluator,
            use_neural_eval=bool(board_evaluator is not None),
            beam_width=8,
            device=self.device,
        )
        self.model.to(self.device)
        self.model.eval()

    def take_turn(
        self,
        player: Player,
        pool: ChampionPool,
        set_data: SetData,
        stage: int,
        round_in_stage: int,
        rng: random.Random | None = None,
        **kwargs: Any,
    ) -> None:
        """Execute bot planning turn by repeatedly predicting and executing micro-actions until PASS (0)."""
        if not player.alive:
            return

        if self.use_planner:
            actions_taken = 0
            while actions_taken < self.max_micro_actions:
                planned_actions = self.planner.plan_shop_sequence(
                    player=player,
                    pool=pool,
                    stage=stage,
                    round_in_stage=round_in_stage,
                    target_z=self.target_z,
                )
                if not planned_actions:
                    break

                reroll_executed = False
                for action_id in planned_actions:
                    if actions_taken >= self.max_micro_actions:
                        break
                    if action_id == 0:
                        break

                    mask = get_action_mask(player, set_data)
                    if not mask[action_id]:
                        continue

                    success = execute_action(player, pool, set_data, action_id, rng=rng)
                    actions_taken += 1

                    if action_id == 6:
                        # Reroll occurred: plan next shop window
                        reroll_executed = True
                        break

                if not reroll_executed:
                    break
            return

        actions_taken = 0
        while actions_taken < self.max_micro_actions:
            # 1. Encode 704D observation
            obs_vec, s_t, h_board = self.encoder.extract_state_vector(
                player=player,
                stage=stage,
                round_in_stage=round_in_stage,
                target_z=self.target_z,
            )
            mask = get_action_mask(player, set_data)

            obs_tensor = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)

            with torch.no_grad():
                action_t, log_prob_t, val_t = self.model.get_action(
                    obs_tensor,
                    mask_tensor,
                    deterministic=self.deterministic,
                )
                action_id = int(action_t.item())
                log_prob = float(log_prob_t.item())
                value = float(val_t.item())

            # Action 0 is PASS_ROUND (end planning turn)
            if action_id == 0:
                break

            # Execute micro action
            success = execute_action(player, pool, set_data, action_id, rng=rng)
            actions_taken += 1

            # Parallel Rollout Buffer Ingestion & Archetype Telemetry
            if self.target_z is not None:
                z_target_tensor = torch.as_tensor(self.target_z, dtype=torch.float32, device=self.device)
                if z_target_tensor.norm() > 1e-6:
                    r_macro = float(
                        torch.nn.functional.cosine_similarity(h_board.unsqueeze(0), z_target_tensor.unsqueeze(0)).item()
                    )
                    cluster_match = None
                    if self.z_centroids is not None and self.z_index is not None and len(self.z_centroids) > 0:
                        z_cents = torch.as_tensor(self.z_centroids, dtype=torch.float32, device=self.device)
                        sims = torch.nn.functional.cosine_similarity(h_board.unsqueeze(0), z_cents, dim=-1)
                        best_k = int(torch.argmax(sims).item())
                        cluster_match = 1.0 if best_k == self.z_index else 0.0

                    if self.metrics_tracker is not None:
                        self.metrics_tracker.record(r_macro, cluster_match)

                    if self.record_transitions and self.buffer is not None and not getattr(self.buffer, "full", False):
                        step_reward = self.alpha * r_macro
                        self.buffer.add(
                            obs=obs_vec,
                            action=action_id,
                            action_mask=mask,
                            reward=step_reward,
                            value=value,
                            log_prob=log_prob,
                            done=not player.alive,
                        )

            elif self.record_transitions and self.buffer is not None and not getattr(self.buffer, "full", False):
                self.buffer.add(
                    obs=obs_vec,
                    action=action_id,
                    action_mask=mask,
                    reward=0.0,
                    value=value,
                    log_prob=log_prob,
                    done=not player.alive,
                )

            if not success:
                break
