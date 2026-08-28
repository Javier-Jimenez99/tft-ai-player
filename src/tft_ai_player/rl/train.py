"""Training loop and League Self-Play orchestrator for TFT RL."""

from __future__ import annotations

import random
from typing import Any
import numpy as np
import torch

from tft_ai_player.rl.algorithms.ppo import MaskablePPO, RolloutBuffer
from tft_ai_player.rl.evaluation.evaluator import TournamentEvaluator
from tft_ai_player.rl.league.league_manager import LeagueManager
from tft_ai_player.rl.models.networks import TFTActorCritic
from tft_ai_player.rl.types import AgentRole
from tft_ai_player.simulation.config import SetData, get_default_set17_data
from tft_ai_player.simulation.gym_env import TFTEnv


class LeagueTrainer:
    """Orchestrates PPO training and League evaluation for TFT."""

    def __init__(
        self,
        set_data: SetData | None = None,
        model: TFTActorCritic | None = None,
        lr: float = 3e-4,
        buffer_size: int = 1024,
        batch_size: int = 64,
        num_epochs: int = 4,
        device: str | None = None,
        checkpoint_dir: str = "checkpoints/league",
    ) -> None:
        self.set_data = set_data or get_default_set17_data()
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        # Initialize neural policy
        self.model = model or TFTActorCritic(
            num_champs=len(self.set_data.champions) + 1,
            num_items=len(self.set_data.items) + 1,
        )
        self.model.to(self.device)

        self.ppo = MaskablePPO(self.model, lr=lr, device=self.device)
        self.league = LeagueManager(checkpoint_dir=checkpoint_dir)
        self.evaluator = TournamentEvaluator(self.league, set_data=self.set_data)

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # Register active learning agent in the league
        self.main_agent_id = "main_agent_v1"
        self.league.register_agent(
            agent_id=self.main_agent_id,
            name="Main PPO Agent",
            role=AgentRole.MAIN,
        )

        # Gymnasium environment
        self.env = TFTEnv(set_data=self.set_data)

    def collect_rollouts(
        self,
        buffer: RolloutBuffer,
        num_steps: int = 512,
        seed: int | None = None,
    ) -> dict[str, float]:
        """Collect on-policy interaction rollouts in the TFT environment."""
        self.model.eval()
        obs, info = self.env.reset(seed=seed)
        step_count = 0
        total_reward = 0.0
        episodes_completed = 0
        placement_history: list[int] = []

        while step_count < num_steps:
            mask = info.get("action_mask", np.ones(1721, dtype=bool))

            # Convert observation dict to batched torch tensors
            tensor_obs: dict[str, torch.Tensor] = {}
            for k, v in obs.items():
                if k != "action_mask":
                    tensor_obs[k] = torch.as_tensor(v, dtype=torch.float32, device=self.device).unsqueeze(0)

            mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)

            with torch.no_grad():
                action_t, log_prob_t, val_t = self.model.get_action(tensor_obs, mask_tensor)
                action = int(action_t.item())
                log_prob = float(log_prob_t.item())
                val = float(val_t.item())

            next_obs, reward, terminated, truncated, next_info = self.env.step(action)
            done = terminated or truncated
            total_reward += reward

            buffer.add(
                obs_dict=obs,
                action=action,
                action_mask=mask,
                reward=reward,
                value=val,
                log_prob=log_prob,
                done=done,
            )

            step_count += 1
            obs = next_obs
            info = next_info

            if done:
                episodes_completed += 1
                focal = self.env.game.get_focal_player()
                placement_history.append(focal.placement or (1 if focal.alive else 8))
                obs, info = self.env.reset()

        # Compute GAE
        last_val = 0.0
        if not done:
            tensor_obs = {
                k: torch.as_tensor(v, dtype=torch.float32, device=self.device).unsqueeze(0)
                for k, v in obs.items()
                if k != "action_mask"
            }
            with torch.no_grad():
                _, _, v_last = self.model.get_action(
                    tensor_obs,
                    torch.as_tensor(info.get("action_mask", np.ones(1721, dtype=bool)), device=self.device).unsqueeze(0),
                )
                last_val = float(v_last.item())

        buffer.compute_returns_and_advantages(last_value=last_val, done=done)

        avg_placement = float(np.mean(placement_history)) if placement_history else 4.5
        return {
            "steps": step_count,
            "episodes": episodes_completed,
            "mean_reward": total_reward / max(step_count, 1),
            "avg_placement": avg_placement,
        }

    def train_iteration(
        self,
        generation: int,
        rollout_steps: int = 512,
        eval_every: int = 5,
        eval_num_seeds: int = 3,
    ) -> dict[str, Any]:
        """Execute one complete training generation: Rollouts -> PPO Update -> Eval -> Checkpointing."""
        # 1. Initialize rollout buffer
        dummy_obs, dummy_info = self.env.reset(seed=100)
        buffer = RolloutBuffer(
            buffer_size=rollout_steps + 64,
            obs_sample=dummy_obs,
            action_dim=1721,
            device=self.device,
        )

        # 2. Collect rollouts
        collect_metrics = self.collect_rollouts(buffer, num_steps=rollout_steps)

        # 3. PPO Update
        ppo_metrics = self.ppo.train_epoch(
            buffer=buffer,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
        )

        metrics: dict[str, Any] = {
            "generation": generation,
            **collect_metrics,
            **ppo_metrics,
        }

        # 4. Periodic Evaluation against League benchmarks
        if generation % eval_every == 0:
            eval_res = self.evaluator.run_paired_benchmark(
                candidate_agent_id=self.main_agent_id,
                opponent_agent_ids=["bot_standard_tempo", "bot_greedy_banker", "bot_random"],
                candidate_model=self.model,
                num_seeds=eval_num_seeds,
            )
            metrics["eval_avg_placement"] = eval_res["avg_placement"]
            metrics["eval_top4_rate"] = eval_res["top4_rate"]
            metrics["eval_win_rate"] = eval_res["win_rate"]
            metrics["league_elo"] = eval_res["current_elo"]

            # Save snapshot to Hall of Fame if high performing
            if generation > 0 and generation % (eval_every * 2) == 0:
                self.league.create_hall_of_fame_snapshot(
                    source_agent_id=self.main_agent_id,
                    generation=generation,
                    weights=self.model.state_dict(),
                )

        return metrics
