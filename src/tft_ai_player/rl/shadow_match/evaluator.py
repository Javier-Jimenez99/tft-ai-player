"""Shadow Match Evaluator.

Benchmarks an RL policy against authentic high-elo / Challenger 1st-place match traces.
Computes survival rounds, placement estimations, and win rates against real player boards.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...simulation.config import SetData, get_default_set17_data
from ..models.networks import TFTActorCritic
from .replay_loader import ShadowMatchReplay, ShadowMatchRepository
from .shadow_env import ShadowMatchEnv

logger = logging.getLogger(__name__)


@dataclass
class ShadowEvalReport:
    """Summary metrics of policy evaluation on real human match replays."""

    num_matches: int
    mean_rounds_survived: float
    mean_estimated_placement: float
    top_4_rate: float
    top_1_rate: float
    mean_combat_win_rate: float
    match_results: list[dict[str, Any]] = field(default_factory=list)

    def summary_string(self) -> str:
        return (
            f"=== Shadow Match Evaluation ({self.num_matches} matches) ===\n"
            f"  Mean Survival:        {self.mean_rounds_survived:.1f} rounds\n"
            f"  Mean Placement:       #{self.mean_estimated_placement:.2f}\n"
            f"  Top-4 Rate:           {self.top_4_rate * 100:.1f}%\n"
            f"  Win Rate (1st place): {self.top_1_rate * 100:.1f}%\n"
            f"  Combat Win Rate:      {self.mean_combat_win_rate * 100:.1f}%\n"
        )


class ShadowMatchEvaluator:
    """Evaluates policies on chronological Challenger match replays."""

    def __init__(
        self,
        repository: ShadowMatchRepository,
        set_data: SetData | None = None,
        device: str = "cpu",
    ) -> None:
        self.repository = repository
        self.set_data = set_data or get_default_set17_data()
        self.device = torch.device(device)
        self.env = ShadowMatchEnv(
            repository=self.repository,
            set_data=self.set_data,
            device=str(self.device),
        )

    def evaluate_policy(
        self,
        policy: TFTActorCritic,
        num_matches: int = 50,
        deterministic: bool = True,
    ) -> ShadowEvalReport:
        """Run evaluation over sample matches from the repository."""
        policy.eval()
        replays = self.repository.sample_matches(n=num_matches, min_rounds=18)
        if not replays:
            logger.warning("No matches found meeting evaluation criteria.")
            return ShadowEvalReport(0, 0.0, 8.0, 0.0, 0.0, 0.0)

        match_results: list[dict[str, Any]] = []

        for idx, rep in enumerate(replays):
            obs, info = self.env.reset(options={"replay": rep})
            done = False
            total_reward = 0.0
            combats_won = 0
            combats_total = 0

            while not done:
                mask = info.get("action_mask")
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0) if mask is not None else None

                with torch.no_grad():
                    action, _, _ = policy.get_action(obs_t, mask_t, deterministic=deterministic)

                action_int = int(action.item())
                prev_streak = self.env.player.streak
                obs, reward, terminated, truncated, info = self.env.step(action_int)
                total_reward += reward
                done = terminated or truncated

                # Track combat wins
                if self.env.actions_in_current_round == 0:
                    combats_total += 1
                    if self.env.player.streak > 0:
                        combats_won += 1

            est_placement = self.env._estimate_placement(self.env.rounds_survived)
            c_wr = (combats_won / max(1, combats_total))

            match_results.append({
                "match_id": rep.match_id,
                "tier": rep.focal_tier,
                "rounds_survived": self.env.rounds_survived,
                "human_rounds": len(rep.rounds),
                "estimated_placement": est_placement,
                "combat_win_rate": c_wr,
                "reward": total_reward,
            })

        survivals = [r["rounds_survived"] for r in match_results]
        placements = [r["estimated_placement"] for r in match_results]
        c_wrs = [r["combat_win_rate"] for r in match_results]

        top_4 = sum(1 for p in placements if p <= 4) / len(placements)
        top_1 = sum(1 for p in placements if p == 1) / len(placements)

        report = ShadowEvalReport(
            num_matches=len(match_results),
            mean_rounds_survived=float(np.mean(survivals)),
            mean_estimated_placement=float(np.mean(placements)),
            top_4_rate=float(top_4),
            top_1_rate=float(top_1),
            mean_combat_win_rate=float(np.mean(c_wrs)),
            match_results=match_results,
        )
        return report
