"""Trace-Driven Shadow Match Trainer (AlphaStar v6.1).

Orchestrates PPO rollouts and updates directly against chronological match traces
of real Challenger / high-elo TFT players, initialized from the v6 baseline checkpoint.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...simulation.config import SetData
from ...simulation.sets.set18 import get_set18_data
from ..models.networks import TFTActorCritic
from ..train import LeagueTrainer
from .evaluator import ShadowMatchEvaluator
from .replay_loader import ShadowMatchLoader
from .shadow_env import ShadowMatchEnv

logger = logging.getLogger(__name__)


class ShadowTrainer(LeagueTrainer):
    """AlphaStar v6.1 Trainer.
    
    Subclasses LeagueTrainer to execute PPO rollouts over authentic Challenger
    match traces using ShadowMatchEnv, whilst maintaining tree-search lookahead,
    world model latent guidance, entropy annealing, and checkpointing.
    """

    def __init__(
        self,
        cache_file: str | Path = "models/rl/shadow_matches_cache.pkl.gz",
        data_dir: str | Path = r"D:\tft-winner-data\set18\players",
        checkpoint_dir: str | Path = r"D:\tft-winner-data\set18\models\rl\checkpoints\ppo_alphastar_v6_1_shadow",
        pretrained_checkpoint: str | Path | None = None,
        run_name: str = "ppo_alphastar_v6_1_shadow",
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("checkpoint_dir", checkpoint_dir)
        kwargs.setdefault("run_name", run_name)

        super().__init__(**kwargs)

        self.cache_file = Path(cache_file)
        self.data_dir = Path(data_dir)

        # 1. Load Shadow Match Repository
        self.loader = ShadowMatchLoader(
            data_dir=str(self.data_dir),
            cache_path=str(self.cache_file),
        )
        self.repository = self.loader.load_repository()
        logger.info(f"Loaded {len(self.repository.replays)} high-elo shadow replays into ShadowTrainer.")

        # 2. Instantiate Shadow Match Gymnasium Environment for Rollouts
        self.shadow_env = ShadowMatchEnv(
            repository=self.repository,
            set_data=self.set_data,
            device=str(self.device),
        )

        # 3. Evaluator on authentic Challenger replays
        self.shadow_evaluator = ShadowMatchEvaluator(
            repository=self.repository,
            set_data=self.set_data,
            device=str(self.device),
        )

        # 4. Initialize from v6 pretrained weights if supplied
        if pretrained_checkpoint and Path(pretrained_checkpoint).exists():
            self._load_pretrained_weights(pretrained_checkpoint)

    def _load_pretrained_weights(self, ckpt_path: str | Path) -> None:
        """Load weights from a prior run (e.g. v6 Gen 300) into the actor-critic model."""
        target_path = Path(ckpt_path)
        if target_path.is_dir():
            for fname in ("training_state.pt", "weights.pt", "model.pt"):
                if (target_path / fname).exists():
                    target_path = target_path / fname
                    break

        print(f" [+] Initializing v6.1 from pretrained checkpoint: {target_path.resolve()}")
        logger.info(f"Initializing weights from {target_path}")
        state_dict = torch.load(target_path, map_location=self.device)

        if "model_state_dict" in state_dict:
            self.model.load_state_dict(state_dict["model_state_dict"])
            if hasattr(self, "main_exploiter") and "main_exploiter_model_state_dict" in state_dict:
                self.main_exploiter.load_state_dict(state_dict["main_exploiter_model_state_dict"])
            if hasattr(self, "league_exploiter") and "league_exploiter_model_state_dict" in state_dict:
                self.league_exploiter.load_state_dict(state_dict["league_exploiter_model_state_dict"])
        elif "model" in state_dict:
            self.model.load_state_dict(state_dict["model"])
        elif "policy" in state_dict:
            self.model.load_state_dict(state_dict["policy"])
        else:
            self.model.load_state_dict(state_dict)

        self.last_stable_state = copy.deepcopy(self.model.state_dict())
        print(" [+] Pretrained weights successfully loaded into AlphaStar v6.1.")

    def run_training_loop(self) -> None:
        """Execute the v6.1 PPO generations using ShadowMatchEnv rollouts."""
        print("\n" + "=" * 80)
        print(f" [ALPHASTAR v6.1: SHADOW MATCH TRAINING] Starting run: {self.run_name}")
        print(f"  Checkpoint directory: {self.checkpoint_dir.resolve()}")
        print(f"  Rollout steps per gen: {self.total_rollout_steps} | Batch size: {self.batch_size}")
        print(f"  Generations: {self.max_generations} | Device: {self.device}")
        print("=" * 80 + "\n")

        for gen in range(1, self.max_generations + 1):
            self.current_progress = gen / max(1, self.max_generations)

            # 1. Rollout Collection in ShadowMatchEnv
            rollout_metrics = self.collect_rollouts(
                env=self.shadow_env,
                target_steps=self.total_rollout_steps,
                model=self.model,
                buffer=self.buffer,
            )

            # 2. PPO Optimization Step
            ppo_metrics = self.ppo.update(
                self.buffer,
                num_epochs=self.num_epochs,
                batch_size=self.batch_size,
            )

            # 3. Anneal entropy coefficient
            progress = max(0.0, min(1.0, float(gen) / max(1, self.max_generations)))
            current_entropy = self.entropy_coef_min + 0.5 * (self.entropy_coef - self.entropy_coef_min) * (1.0 + np.cos(np.pi * progress))
            self.ppo.set_entropy_coef(float(current_entropy))
            current_lr = max(1e-5, self.initial_lr * (1.0 - progress))
            self.ppo.set_learning_rate(current_lr)

            # Log generation metrics identical to v6 format
            metrics = {
                "generation": gen,
                "learning_rate": current_lr,
                "entropy_coef": float(current_entropy),
                **rollout_metrics,
                **ppo_metrics,
            }

            print(
                f"[Gen {gen:04d}/{self.max_generations}] "
                f"Reward: {metrics['mean_reward']:+0.3f} | "
                f"Avg Place: {metrics['avg_placement']:.2f} | "
                f"Top-4: {metrics['top4_rate']*100:4.1f}% | "
                f"PlanAgr: {metrics.get('planner_agreement_pct', 0.0):4.1f}% | "
                f"Guided: {metrics.get('guided_actions_pct', 0.0):4.1f}% | "
                f"ExpVar: {metrics['explained_variance']:.3f} | "
                f"Loss: {metrics['policy_loss']:+0.3f}"
            )

            # 4. Periodic Challenger Benchmark Evaluation
            if gen % self.eval_interval == 0:
                eval_report = self.shadow_evaluator.evaluate_policy(
                    policy=self.model,
                    num_matches=30,
                    deterministic=True,
                )
                eval_metrics = {
                    "eval_avg_placement": eval_report.mean_estimated_placement,
                    "eval_top4_rate": eval_report.top_4_rate,
                    "eval_win_rate": eval_report.top_1_rate,
                    "eval/rounds_survived": eval_report.mean_rounds_survived,
                    "eval/combat_win_rate": eval_report.mean_combat_win_rate,
                }
                metrics.update(eval_metrics)
                print(
                    f"  --> Benchmark Evaluation [Gen {gen}]: "
                    f"Avg Place: {eval_report.mean_estimated_placement:.2f} | "
                    f"Top-4: {eval_report.top_4_rate*100:.1f}% | "
                    f"Win Rate: {eval_report.top_1_rate*100:.1f}% | "
                    f"Combat WR: {eval_report.mean_combat_win_rate*100:.1f}%"
                )

            # Log unified metrics to WandB
            self.wandb_logger.log(metrics, step=gen)

            # 5. Save Checkpoints & Snapshots
            if gen % self.snapshot_interval == 0 or gen == self.max_generations:
                gen_dir = self.checkpoint_dir / f"gen_{gen:04d}"
                gen_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "generation": gen,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.ppo.optimizer.state_dict(),
                        "entropy_coef": self.entropy_coef,
                    },
                    gen_dir / "training_state.pt",
                )
                torch.save(self.model.state_dict(), self.checkpoint_dir / "rl_model_latest.pt")
                print(f" [+] Checkpoint saved: {gen_dir / 'training_state.pt'}\n")
