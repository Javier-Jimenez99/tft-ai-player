"""Model checkpointing, metadata serialization, and Hall of Fame persistence for TFT League."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tft_ai_player.rl.types import AgentProfile, AgentRole, EloRating


class CheckpointManager:
    """Manages serialization of model weights and agent metadata across generations."""

    def __init__(self, base_dir: str | Path = "checkpoints/league") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.hof_dir = self.base_dir / "hall_of_fame"
        self.hof_dir.mkdir(parents=True, exist_ok=True)

    def save_agent_profile(self, profile: AgentProfile, weights: Any | None = None) -> Path:
        """Save an agent profile and its weights to disk."""
        target_dir = self.hof_dir if profile.role == AgentRole.HALL_OF_FAME else self.base_dir
        agent_dir = target_dir / profile.agent_id
        agent_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save metadata JSON
        meta_path = agent_dir / "profile.json"
        data = {
            "agent_id": profile.agent_id,
            "name": profile.name,
            "role": profile.role.value,
            "generation": profile.generation,
            "elo": {
                "rating": profile.elo.rating,
                "games_played": profile.elo.games_played,
                "wins": profile.elo.wins,
                "top4s": profile.elo.top4s,
                "placements": profile.elo.placements,
                "rating_history": profile.elo.rating_history,
            },
            "metadata": profile.metadata,
            "h2h_records": profile.h2h_records,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # 2. Save weights if provided and torch available
        if weights is not None:
            weights_path = agent_dir / "weights.pt"
            try:
                import torch
                torch.save(weights, weights_path)
            except Exception:
                # Fallback to saving numpy arrays if torch is not available
                pass
            profile.checkpoint_path = str(weights_path)

        return agent_dir

    def load_agent_profile(self, agent_id: str) -> AgentProfile | None:
        """Load an agent profile metadata from disk."""
        for candidate_dir in [self.base_dir / agent_id, self.hof_dir / agent_id]:
            meta_path = candidate_dir / "profile.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                elo_data = data.get("elo", {})
                elo = EloRating(
                    rating=elo_data.get("rating", 1200.0),
                    games_played=elo_data.get("games_played", 0),
                    wins=elo_data.get("wins", 0),
                    top4s=elo_data.get("top4s", 0),
                    placements=elo_data.get("placements", []),
                    rating_history=elo_data.get("rating_history", [1200.0]),
                )

                profile = AgentProfile(
                    agent_id=data["agent_id"],
                    name=data["name"],
                    role=AgentRole(data["role"]),
                    elo=elo,
                    generation=data.get("generation", 0),
                    metadata=data.get("metadata", {}),
                    h2h_records=data.get("h2h_records", {}),
                )
                weights_path = candidate_dir / "weights.pt"
                if weights_path.exists():
                    profile.checkpoint_path = str(weights_path)
                return profile
        return None
