"""Trace-Driven Ranked Ladder Trainer (AlphaStar v6.2).

Orchestrates PPO rollouts and updates dynamically adapting to the agent's real ranked Elo/division,
sampling exclusively Top-1 replays from the agent's current bracket (starting at Gold IV),
and initialized strictly from the v6 baseline checkpoint.
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
from .ranked_env import RankedLadderEnv
from .ranked_loader import RankedLadderReplayLoader
from ..visualization.ranked_progression_plot import rank_to_absolute_lp

logger = logging.getLogger(__name__)


class RankedLadderTrainer(LeagueTrainer):
    """AlphaStar v6.2 Trainer with Adaptive Ranked Ladder ELO."""

    def __init__(
        self,
        cache_file: str | Path = "models/rl/ranked_ladder_cache.pkl.gz",
        tiers_dir: str | Path = r"D:\tft-winner-data\tiers",
        checkpoint_dir: str | Path = r"D:\tft-winner-data\set18\models\rl\checkpoints\ppo_alphastar_v6_2_ranked",
        pretrained_checkpoint: str | Path = r"D:\tft-winner-data\set18\models\rl\checkpoints\ppo_alphastar_v6\gen_0300\training_state.pt",
        run_name: str = "ppo_alphastar_v6_2",
        initial_tier: str = "GOLD",
        initial_division: int = 4,
        initial_lp: int = 0,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("checkpoint_dir", checkpoint_dir)
        kwargs.setdefault("run_name", run_name)

        super().__init__(**kwargs)

        self.cache_file = Path(cache_file)
        self.tiers_dir = Path(tiers_dir)

        # 1. Load Ranked Replay Repository
        self.ladder_loader = RankedLadderReplayLoader(
            tiers_dir=self.tiers_dir,
            cache_path=self.cache_file,
        )
        self.replays_by_tier = self.ladder_loader.build_or_load_index()

        # 2. Instantiate Ranked Gymnasium Environment for Rollouts
        self.ranked_env = RankedLadderEnv(
            ladder_loader=self.ladder_loader,
            initial_tier=initial_tier,
            initial_division=initial_division,
            initial_lp=initial_lp,
            set_data=self.set_data,
            device=str(self.device),
        )

        # 3. Initialize strictly from v6 pretrained weights
        if pretrained_checkpoint and Path(pretrained_checkpoint).exists():
            self._load_pretrained_weights(pretrained_checkpoint)
        else:
            raise FileNotFoundError(f"Required v6 baseline checkpoint not found at: {pretrained_checkpoint}")

    def _load_pretrained_weights(self, ckpt_path: str | Path) -> None:
        """Load weights from baseline v6 Gen 300 into the actor-critic model."""
        target_path = Path(ckpt_path)
        if target_path.is_dir():
            for fname in ("training_state.pt", "weights.pt", "model.pt"):
                if (target_path / fname).exists():
                    target_path = target_path / fname
                    break

        print(f" [+] Initializing AlphaStar v6.2 strictly from v6 baseline: {target_path.resolve()}")
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
        print(" [+] Baseline v6 weights successfully loaded into AlphaStar v6.2.")

    def run_training_loop(self) -> None:
        """Execute the v6.2 PPO generations using RankedLadderEnv rollouts."""
        print("\n" + "=" * 80)
        print(f" [ALPHASTAR v6.2: RANKED LADDER TRAINING] Starting run: {self.run_name}")
        print(f"  Checkpoint directory: {self.checkpoint_dir.resolve()}")
        print(f"  Current Rank: {self.ranked_env.ranked_status.display_rank} (MMR: {self.ranked_env.ranked_status.mmr:.0f})")
        print(f"  Rollout steps per gen: {self.total_rollout_steps} | Batch size: {self.batch_size}")
        print(f"  Generations: {self.max_generations} | Device: {self.device}")
        print("=" * 80 + "\n")

        for gen in range(1, self.max_generations + 1):
            self.current_progress = gen / max(1, self.max_generations)

            # 1. Rollout Collection in RankedLadderEnv
            rollout_metrics = self.collect_rollouts(
                env=self.ranked_env,
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

            # 3. Anneal entropy coefficient & learning rate
            progress = max(0.0, min(1.0, float(gen) / max(1, self.max_generations)))
            current_entropy = self.entropy_coef_min + 0.5 * (self.entropy_coef - self.entropy_coef_min) * (1.0 + np.cos(np.pi * progress))
            self.ppo.set_entropy_coef(float(current_entropy))
            current_lr = max(1e-5, self.initial_lr * (1.0 - progress))
            self.ppo.set_learning_rate(current_lr)

            # Ranked Ladder Status
            r_stat = self.ranked_env.ranked_status
            abs_lp = rank_to_absolute_lp(r_stat.tier, r_stat.division, r_stat.lp)
            ranked_metrics = {
                "ranked/tier": r_stat.tier,
                "ranked/division": r_stat.division,
                "ranked/lp": r_stat.lp,
                "ranked/absolute_lp": abs_lp,
                "ranked/mmr": r_stat.mmr,
                "ranked/total_games": r_stat.total_games,
                "ranked/win_rate": (r_stat.total_wins / max(1, r_stat.total_games)),
                "ranked/top4_rate": (r_stat.total_top4 / max(1, r_stat.total_games)),
            }

            metrics = {
                "generation": gen,
                "learning_rate": current_lr,
                "entropy_coef": float(current_entropy),
                **rollout_metrics,
                **ppo_metrics,
                **ranked_metrics,
            }

            print(
                f"[Gen {gen:04d}/{self.max_generations}] "
                f"Rank: {r_stat.display_rank:16s} ({abs_lp:4.0f} LP) | "
                f"Reward: {metrics['mean_reward']:+0.3f} | "
                f"Avg Place: {metrics['avg_placement']:.2f} | "
                f"Top-4: {metrics['top4_rate']*100:4.1f}% | "
                f"BuyXP: {metrics.get('action_buy_xp_pct', 0.0):4.1f}% | "
                f"Loss: {metrics['policy_loss']:+0.3f}"
            )

            # Log unified metrics to WandB
            self.wandb_logger.log(metrics, step=gen)

            # 4. Save Checkpoints & Snapshots
            if gen % self.snapshot_interval == 0 or gen == self.max_generations:
                gen_dir = self.checkpoint_dir / f"gen_{gen:04d}"
                gen_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "generation": gen,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.ppo.optimizer.state_dict(),
                        "entropy_coef": self.entropy_coef,
                        "ranked_status": {
                            "tier": r_stat.tier,
                            "division": r_stat.division,
                            "lp": r_stat.lp,
                            "mmr": r_stat.mmr,
                            "total_games": r_stat.total_games,
                            "total_wins": r_stat.total_wins,
                            "total_top4": r_stat.total_top4,
                        },
                    },
                    gen_dir / "training_state.pt",
                )
                torch.save(self.model.state_dict(), self.checkpoint_dir / "rl_model_latest.pt")
                print(f" [+] Checkpoint saved: {gen_dir / 'training_state.pt'}\n")
