"""AlphaStar-inspired League Training Manager and Prioritized Fictitious Self-Play (PFSP) for TFT."""

from __future__ import annotations

import random
from typing import Any, Sequence
import numpy as np

from tft_ai_player.rl.league.checkpoints import CheckpointManager
from tft_ai_player.rl.league.elo import MultilateralEloSystem
from tft_ai_player.rl.types import AgentProfile, AgentRole, EloRating, MatchResult


class LeagueManager:
    """Orchestrates an AlphaStar-style multi-agent league for 8-player TFT lobbies.

    Maintains Main Agents, Exploiters, Hall of Fame checkpoints, and Baseline Bots,
    with Prioritized Fictitious Self-Play (PFSP) matchmaking and 8-player Elo updates.
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints/league",
        k_factor: float = 32.0,
        base_rating: float = 1200.0,
    ) -> None:
        self.elo_system = MultilateralEloSystem(k_factor=k_factor, base_rating=base_rating)
        self.checkpoint_manager = CheckpointManager(base_dir=checkpoint_dir)
        self.profiles: dict[str, AgentProfile] = {}
        self.match_history: list[MatchResult] = []

        # Register standard baseline bots by default
        self._register_default_baselines()

    def _register_default_baselines(self) -> None:
        """Register default rule-based anchor bots in the league."""
        self.register_agent(
            agent_id="bot_standard_tempo",
            name="Standard Tempo Bot",
            role=AgentRole.BASELINE,
            metadata={"bot_class": "StandardTempoBot"},
        )
        self.profiles["bot_standard_tempo"].elo.rating = 1200.0

        self.register_agent(
            agent_id="bot_greedy_banker",
            name="Greedy Banker Bot",
            role=AgentRole.BASELINE,
            metadata={"bot_class": "GreedyBankerBot"},
        )
        self.profiles["bot_greedy_banker"].elo.rating = 1100.0

        self.register_agent(
            agent_id="bot_random",
            name="Random Baseline Bot",
            role=AgentRole.BASELINE,
            metadata={"bot_class": "RandomBot"},
        )
        self.profiles["bot_random"].elo.rating = 900.0

    def register_agent(
        self,
        agent_id: str,
        name: str,
        role: AgentRole,
        checkpoint_path: str | None = None,
        generation: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> AgentProfile:
        """Register a new agent in the league."""
        if agent_id in self.profiles:
            return self.profiles[agent_id]

        profile = AgentProfile(
            agent_id=agent_id,
            name=name,
            role=role,
            checkpoint_path=checkpoint_path,
            generation=generation,
            metadata=metadata or {},
        )
        self.profiles[agent_id] = profile
        return profile

    def create_hall_of_fame_snapshot(
        self,
        source_agent_id: str,
        generation: int,
        weights: Any | None = None,
    ) -> AgentProfile:
        """Freeze a copy of a main learning agent into the Hall of Fame archive."""
        source = self.profiles[source_agent_id]
        hof_id = f"hof_{source_agent_id}_gen{generation}"
        hof_name = f"HoF {source.name} (Gen {generation})"

        hof_profile = AgentProfile(
            agent_id=hof_id,
            name=hof_name,
            role=AgentRole.HALL_OF_FAME,
            elo=EloRating(
                rating=source.elo.rating,
                games_played=0,
                wins=0,
                top4s=0,
                placements=[],
                rating_history=[source.elo.rating],
            ),
            generation=generation,
            metadata={"source_agent": source_agent_id, "snapshot_generation": generation},
        )

        self.checkpoint_manager.save_agent_profile(hof_profile, weights=weights)
        self.profiles[hof_id] = hof_profile
        return hof_profile

    def sample_pfsp_opponents(
        self,
        focal_agent_id: str,
        num_opponents: int = 7,
        temperature: float = 2.0,
        rng: random.Random | None = None,
    ) -> list[str]:
        """Sample opponents for an 8-player lobby using Prioritized Fictitious Self-Play (PFSP).

        Weights candidates based on difficulty (prioritizing opponents the focal agent struggles against).
        """
        r = rng or random
        focal = self.profiles.get(focal_agent_id)
        candidate_ids = [aid for aid in self.profiles.keys() if aid != focal_agent_id]

        if not candidate_ids:
            return [focal_agent_id] * num_opponents

        if len(candidate_ids) <= num_opponents:
            # If not enough distinct candidates, fill with replacement
            return r.choices(candidate_ids, k=num_opponents)

        # Compute PFSP weighting: P(opp) ~ (1 - win_rate_vs_opp)^temperature
        weights: list[float] = []
        for opp_id in candidate_ids:
            if focal:
                win_rate = focal.get_win_rate_vs(opp_id)
            else:
                win_rate = 0.5

            # Challenging opponents (win_rate < 0.5) get higher weights
            weight = (1.0 - min(max(win_rate, 0.05), 0.95)) ** temperature
            weights.append(weight)

        total_weight = sum(weights)
        probs = [w / total_weight for w in weights] if total_weight > 0 else None

        # Sample without replacement if possible
        selected: list[str] = []
        available_indices = list(range(len(candidate_ids)))

        for _ in range(num_opponents):
            if not available_indices:
                selected.append(r.choice(candidate_ids))
                continue

            sub_weights = [weights[idx] for idx in available_indices]
            sub_total = sum(sub_weights)
            sub_probs = [w / sub_total for w in sub_weights] if sub_total > 0 else None

            chosen_local_idx = np.random.choice(len(available_indices), p=sub_probs)
            chosen_global_idx = available_indices.pop(chosen_local_idx)
            selected.append(candidate_ids[chosen_global_idx])

        return selected

    def record_match_outcome(
        self,
        match_id: str,
        seed: int,
        placements: dict[str, int],
        total_rounds: int,
        metadata: dict[str, Any] | None = None,
    ) -> MatchResult:
        """Process 8-player match results, update Elo ratings, and persist history."""
        # Normalize placement scores (1st -> 1.0, 8th -> 0.0)
        scores = {aid: (8 - rank) / 7.0 for aid, rank in placements.items()}

        # Multilateral Elo update
        deltas = self.elo_system.update_lobby_ratings(placements, self.profiles)

        match_res = MatchResult(
            match_id=match_id,
            seed=seed,
            placements=placements,
            scores=scores,
            total_rounds=total_rounds,
            elo_deltas=deltas,
            metadata=metadata or {},
        )
        self.match_history.append(match_res)
        return match_res

    def get_leaderboard(self) -> list[dict[str, Any]]:
        """Get formatted league leaderboard sorted by Elo rating."""
        return self.elo_system.generate_leaderboard(list(self.profiles.values()))
