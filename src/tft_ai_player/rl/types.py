"""Core types, data structures, and protocols for TFT Reinforcement Learning and League framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol
import numpy as np


from tft_ai_player.simulation.config import AgentArchetype


class AgentRole(str, Enum):
    """Categorical role of an agent within the AlphaStar-style League."""

    MAIN = "main"                     # Continuously learning agent via PPO
    MAIN_EXPLOITER = "main_exploiter" # Specialized to find and punish weaknesses in main agents
    LEAGUE_EXPLOITER = "league_exploiter" # Robust global exploiter
    HALL_OF_FAME = "hall_of_fame"     # Frozen past checkpoints to prevent catastrophic forgetting
    BASELINE = "baseline"             # Deterministic / heuristic reference bots (e.g. StandardTempoBot)


@dataclass
class EloRating:
    """Multilateral 8-player Elo rating profile and performance metrics."""

    rating: float = 1200.0
    games_played: int = 0
    wins: int = 0                     # 1st place finishes
    top4s: int = 0                    # 1st - 4th place finishes
    placements: list[int] = field(default_factory=list)
    rating_history: list[float] = field(default_factory=lambda: [1200.0])

    @property
    def avg_placement(self) -> float:
        """Calculate mean tournament placement (1.0 to 8.0). Lower is better."""
        if not self.placements:
            return 4.5
        return float(np.mean(self.placements))

    @property
    def win_rate(self) -> float:
        """Percentage of 1st place finishes."""
        if self.games_played == 0:
            return 0.0
        return float(self.wins / self.games_played)

    @property
    def top4_rate(self) -> float:
        """Percentage of Top-4 finishes (win condition in competitive TFT)."""
        if self.games_played == 0:
            return 0.0
        return float(self.top4s / self.games_played)

    def record_placement(self, placement: int) -> None:
        """Record a single match placement outcome."""
        self.games_played += 1
        self.placements.append(placement)
        if placement == 1:
            self.wins += 1
        if placement <= 4:
            self.top4s += 1


@dataclass
class AgentProfile:
    """Metadata, statistics, and lineage for a league participant."""

    agent_id: str
    name: str
    role: AgentRole
    archetype: AgentArchetype = AgentArchetype.GENERALIST
    elo: EloRating = field(default_factory=EloRating)
    checkpoint_path: str | None = None
    generation: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    # Head-to-head records: opponent_id -> list of pairwise [won, total]
    h2h_records: dict[str, list[int]] = field(default_factory=dict)

    def get_win_rate_vs(self, opponent_id: str) -> float:
        """Get head-to-head win rate against a specific opponent."""
        record = self.h2h_records.get(opponent_id)
        if not record or record[1] == 0:
            return 0.5  # Prior default
        return record[0] / record[1]

    def record_h2h(self, opponent_id: str, won: bool) -> None:
        """Update pairwise head-to-head record."""
        if opponent_id not in self.h2h_records:
            self.h2h_records[opponent_id] = [0, 0]
        self.h2h_records[opponent_id][1] += 1
        if won:
            self.h2h_records[opponent_id][0] += 1


@dataclass
class MatchResult:
    """Complete summary of an 8-player TFT lobby match."""

    match_id: str
    seed: int
    placements: dict[str, int]        # agent_id -> placement (1..8)
    scores: dict[str, float]          # agent_id -> normalized tournament score
    total_rounds: int
    elo_deltas: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutTransition:
    """Single-step transition data for PPO buffer."""

    obs: np.ndarray
    action: int
    reward: float
    done: bool
    value: float
    log_prob: float
    action_mask: np.ndarray


class AgentPolicyProtocol(Protocol):
    """Protocol for RL agent policy interface."""

    def select_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        """Select action given observation and action mask. Returns (action, log_prob, value)."""
        ...

    def evaluate_actions(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        action_masks: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate log probs, values, and entropy for a batch of transitions."""
        ...
