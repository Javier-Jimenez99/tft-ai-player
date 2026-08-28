"""Multilateral 8-Player Elo Rating System and Performance Analytics for TFT Lobbies."""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np

from tft_ai_player.rl.types import AgentProfile, EloRating, MatchResult


class MultilateralEloSystem:
    """Computes Elo rating updates for N-player (default 8-player) competitive lobbies.

    Decomposes an N-player match into N*(N-1)/2 pairwise head-to-head matches.
    """

    def __init__(self, k_factor: float = 32.0, base_rating: float = 1200.0) -> None:
        self.k_factor = k_factor
        self.base_rating = base_rating

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Compute expected probability of player A beating player B under standard logistic Elo."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update_lobby_ratings(
        self,
        placements: dict[str, int],
        profiles: dict[str, AgentProfile],
    ) -> dict[str, float]:
        """Update Elo ratings and match histories for all agents in an N-player match.

        Parameters
        ----------
        placements : dict[str, int]
            Map of agent_id -> placement rank (1 to 8, where 1 is 1st place).
        profiles : dict[str, AgentProfile]
            Map of agent_id -> agent profile objects to be mutated with updated ratings.

        Returns
        -------
        dict[str, float]
            Map of agent_id -> rating delta (+/- points gained or lost in this match).
        """
        agent_ids = list(placements.keys())
        n = len(agent_ids)
        if n < 2:
            return {aid: 0.0 for aid in agent_ids}

        deltas: dict[str, float] = {aid: 0.0 for aid in agent_ids}

        # Pairwise decomposition
        for i in range(n):
            aid_i = agent_ids[i]
            rank_i = placements[aid_i]
            prof_i = profiles.get(aid_i)
            rating_i = prof_i.elo.rating if prof_i else self.base_rating

            delta_sum = 0.0

            for j in range(n):
                if i == j:
                    continue

                aid_j = agent_ids[j]
                rank_j = placements[aid_j]
                prof_j = profiles.get(aid_j)
                rating_j = prof_j.elo.rating if prof_j else self.base_rating

                # Actual match outcome: lower rank number = better placement
                if rank_i < rank_j:
                    actual_score = 1.0
                    if prof_i:
                        prof_i.record_h2h(aid_j, won=True)
                elif rank_i > rank_j:
                    actual_score = 0.0
                    if prof_i:
                        prof_i.record_h2h(aid_j, won=False)
                else:
                    actual_score = 0.5

                expected = self.expected_score(rating_i, rating_j)
                delta_sum += actual_score - expected

            # Scale by K-factor normalized by (N - 1) opponents
            scaled_delta = (self.k_factor / (n - 1)) * delta_sum
            deltas[aid_i] = round(scaled_delta, 2)

        # Mutate agent profiles with new ratings
        for aid, delta in deltas.items():
            if aid in profiles:
                prof = profiles[aid]
                new_rating = round(prof.elo.rating + delta, 2)
                prof.elo.rating = new_rating
                prof.elo.rating_history.append(new_rating)
                prof.elo.record_placement(placements[aid])

        return deltas

    def generate_leaderboard(
        self,
        profiles: Sequence[AgentProfile],
    ) -> list[dict[str, Any]]:
        """Sort and summarize agent profiles for leaderboard display."""
        sorted_profiles = sorted(profiles, key=lambda p: p.elo.rating, reverse=True)
        leaderboard: list[dict[str, Any]] = []

        for rank, prof in enumerate(sorted_profiles, 1):
            leaderboard.append(
                {
                    "rank": rank,
                    "agent_id": prof.agent_id,
                    "name": prof.name,
                    "role": prof.role.value,
                    "rating": prof.elo.rating,
                    "games": prof.elo.games_played,
                    "win_rate": f"{prof.elo.win_rate * 100:.1f}%",
                    "top4_rate": f"{prof.elo.top4_rate * 100:.1f}%",
                    "avg_placement": f"{prof.elo.avg_placement:.2f}",
                }
            )

        return leaderboard
