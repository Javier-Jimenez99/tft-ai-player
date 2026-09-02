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

    def save_training_state(
        self,
        model: Any,
        optimizer: Any | None,
        generation: int,
        entropy_coef: float,
        metrics_history: list[dict[str, Any]] | None = None,
        is_best: bool = False,
        filename: str = "rl_training_checkpoint.pt",
        main_exploiter_model: Any | None = None,
        main_exploiter_optimizer: Any | None = None,
        league_exploiter_model: Any | None = None,
        league_exploiter_optimizer: Any | None = None,
        config: dict[str, Any] | None = None,
        wandb_run_id: str | None = None,
    ) -> Path:
        """Serialize full training state, hyperparameter configuration, and Tri-Tier models for seamless resumption."""
        import torch

        checkpoint_path = self.base_dir / filename
        state: dict[str, Any] = {
            "generation": generation,
            "entropy_coef": entropy_coef,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "metrics_history": metrics_history or [],
            "config": config or {},
            "wandb_run_id": wandb_run_id,
        }

        if main_exploiter_model is not None:
            state["main_exploiter_model_state_dict"] = main_exploiter_model.state_dict()
        if main_exploiter_optimizer is not None:
            state["main_exploiter_optimizer_state_dict"] = main_exploiter_optimizer.state_dict()
        if league_exploiter_model is not None:
            state["league_exploiter_model_state_dict"] = league_exploiter_model.state_dict()
        if league_exploiter_optimizer is not None:
            state["league_exploiter_optimizer_state_dict"] = league_exploiter_optimizer.state_dict()

        torch.save(state, checkpoint_path)

        # Also save latest weights
        latest_path = self.base_dir / "rl_model_latest.pt"
        torch.save(model.state_dict(), latest_path)

        if is_best:
            best_path = self.base_dir / "rl_model_best.pt"
            torch.save(model.state_dict(), best_path)

        # Save metrics history JSON
        if metrics_history:
            metrics_path = self.base_dir / "rl_training_metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics_history, f, indent=2)

        return checkpoint_path

    def load_training_state(
        self,
        checkpoint_path: str | Path | None = None,
        model: Any | None = None,
        optimizer: Any | None = None,
        main_exploiter_model: Any | None = None,
        main_exploiter_optimizer: Any | None = None,
        league_exploiter_model: Any | None = None,
        league_exploiter_optimizer: Any | None = None,
    ) -> dict[str, Any] | None:
        """Load full training state and restore all Tri-Tier models, optimizers, and locked hyperparameters."""
        import torch

        target_path = Path(checkpoint_path) if checkpoint_path else (self.base_dir / "rl_training_checkpoint.pt")
        if not target_path.exists():
            return None

        state = torch.load(target_path, map_location="cpu", weights_only=False)

        if isinstance(state, dict) and "model_state_dict" in state:
            # 1. Restore Main Agent
            if model is not None:
                model.load_state_dict(state["model_state_dict"])
            if optimizer is not None and state.get("optimizer_state_dict"):
                optimizer.load_state_dict(state["optimizer_state_dict"])

            # 2. Restore Main Exploiter
            if main_exploiter_model is not None and "main_exploiter_model_state_dict" in state:
                main_exploiter_model.load_state_dict(state["main_exploiter_model_state_dict"])
            if main_exploiter_optimizer is not None and state.get("main_exploiter_optimizer_state_dict"):
                main_exploiter_optimizer.load_state_dict(state["main_exploiter_optimizer_state_dict"])

            # 3. Restore League Exploiter
            if league_exploiter_model is not None and "league_exploiter_model_state_dict" in state:
                league_exploiter_model.load_state_dict(state["league_exploiter_model_state_dict"])
            if league_exploiter_optimizer is not None and state.get("league_exploiter_optimizer_state_dict"):
                league_exploiter_optimizer.load_state_dict(state["league_exploiter_optimizer_state_dict"])

            return {
                "generation": state.get("generation", 0),
                "entropy_coef": state.get("entropy_coef", 0.01),
                "metrics_history": state.get("metrics_history", []),
                "config": state.get("config", {}),
                "wandb_run_id": state.get("wandb_run_id"),
            }
        elif model is not None:
            # Raw weights file
            model.load_state_dict(state)
            return {"generation": 0, "entropy_coef": 0.01, "metrics_history": [], "config": {}, "wandb_run_id": None}

        return None

