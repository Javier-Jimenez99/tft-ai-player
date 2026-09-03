"""Model checkpointing, metadata serialization, and League Snapshot persistence for TFT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tft_ai_player.rl.types import AgentProfile, AgentRole, EloRating


class CheckpointManager:
    """Manages serialization of model weights and agent metadata across league generations."""

    def __init__(self, base_dir: str | Path = "checkpoints/league") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir = self.base_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def save_agent_profile(self, profile: AgentProfile, weights: Any | None = None) -> Path:
        """Save an agent profile and its weights to disk."""
        target_dir = self.snapshots_dir if profile.role == AgentRole.HISTORICAL else self.base_dir
        agent_dir = target_dir / profile.agent_id
        agent_dir.mkdir(parents=True, exist_ok=True)

        meta_path = agent_dir / "profile.json"
        data = {
            "agent_id": profile.agent_id,
            "name": profile.name,
            "role": profile.role.value,
            "generation": profile.generation,
            "target_z_index": profile.target_z_index,
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

        if weights is not None:
            weights_path = agent_dir / "weights.pt"
            try:
                import torch
                torch.save(weights, weights_path)
                profile.checkpoint_path = str(weights_path)
            except Exception:
                pass

        return agent_dir

    def load_agent_profile(self, agent_id: str) -> AgentProfile | None:
        """Load an agent profile metadata and weights path from disk."""
        for candidate_dir in [self.base_dir / agent_id, self.snapshots_dir / agent_id]:
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

                role_str = data.get("role", "main")
                try:
                    role = AgentRole(role_str)
                except ValueError:
                    role = AgentRole.HISTORICAL if "historical" in role_str or "hall_of_fame" in role_str else AgentRole.MAIN

                profile = AgentProfile(
                    agent_id=data["agent_id"],
                    name=data["name"],
                    role=role,
                    target_z_index=data.get("target_z_index"),
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
        checkpoint_name: str | None = None,
        main_exploiter_model: Any | None = None,
        main_exploiter_optimizer: Any | None = None,
        league_exploiter_model: Any | None = None,
        league_exploiter_optimizer: Any | None = None,
        config: dict[str, Any] | None = None,
        wandb_run_id: str | None = None,
        is_best: bool = False,
    ) -> Path:
        """Serialize complete training state checkpoint for rollback / resume."""
        name = checkpoint_name or f"gen_{generation:04d}"
        ckpt_dir = self.base_dir / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "generation": generation,
            "entropy_coef": entropy_coef,
            "metrics_history": metrics_history or [],
            "config": config or {},
            "wandb_run_id": wandb_run_id,
        }
        with open(ckpt_dir / "training_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        try:
            import torch

            state = {
                "generation": generation,
                "entropy_coef": entropy_coef,
                "model_state_dict": model.state_dict() if hasattr(model, "state_dict") else None,
                "optimizer_state_dict": optimizer.state_dict() if optimizer and hasattr(optimizer, "state_dict") else None,
                "main_exploiter_model_state_dict": main_exploiter_model.state_dict() if main_exploiter_model and hasattr(main_exploiter_model, "state_dict") else None,
                "main_exploiter_optimizer_state_dict": main_exploiter_optimizer.state_dict() if main_exploiter_optimizer and hasattr(main_exploiter_optimizer, "state_dict") else None,
                "league_exploiter_model_state_dict": league_exploiter_model.state_dict() if league_exploiter_model and hasattr(league_exploiter_model, "state_dict") else None,
                "league_exploiter_optimizer_state_dict": league_exploiter_optimizer.state_dict() if league_exploiter_optimizer and hasattr(league_exploiter_optimizer, "state_dict") else None,
                "metrics_history": metrics_history or [],
                "config": config or {},
                "wandb_run_id": wandb_run_id,
            }
            # 1. Periodic snapshot
            torch.save(state, ckpt_dir / "training_state.pt")

            # 2. Root unified checkpoint
            torch.save(state, self.base_dir / "rl_training_checkpoint.pt")
            if hasattr(model, "state_dict"):
                torch.save(model.state_dict(), self.base_dir / "rl_model_latest.pt")
        except Exception:
            pass

        return ckpt_dir

    def find_latest_training_state(self) -> Path | None:
        """Locate the most recent generation training state checkpoint directory or unified checkpoint."""
        if not self.base_dir.exists():
            return None

        # Check for unified root checkpoint first
        root_ckpt = self.base_dir / "rl_training_checkpoint.pt"
        if root_ckpt.exists():
            return root_ckpt

        # Check gen_XXXX subdirectories sorted by generation number
        gen_dirs = []
        for d in self.base_dir.iterdir():
            if d.is_dir() and d.name.startswith("gen_"):
                try:
                    g_num = int(d.name.replace("gen_", ""))
                    gen_dirs.append((g_num, d))
                except ValueError:
                    pass
        gen_dirs.sort(key=lambda x: x[0], reverse=True)

        for _, g_dir in gen_dirs:
            if (g_dir / "training_state.pt").exists():
                return g_dir
        return None

    def load_training_state(
        self,
        ckpt_dir: str | Path,
        model: Any,
        optimizer: Any | None = None,
        main_exploiter_model: Any | None = None,
        main_exploiter_optimizer: Any | None = None,
        league_exploiter_model: Any | None = None,
        league_exploiter_optimizer: Any | None = None,
    ) -> tuple[int, float]:
        """Restore model weights, optimizer buffers, and entropy coefficient from a training state checkpoint."""
        path = Path(ckpt_dir)
        if path.is_file():
            state_file = path
        elif (path / "rl_training_checkpoint.pt").exists():
            state_file = path / "rl_training_checkpoint.pt"
        elif (path / "training_state.pt").exists():
            state_file = path / "training_state.pt"
        else:
            raise FileNotFoundError(f"Checkpoint file not found in: {ckpt_dir}")

        import torch

        state = torch.load(state_file, map_location="cpu", weights_only=False)
        gen = int(state.get("generation", 0))
        entropy_coef = float(state.get("entropy_coef", 0.01))

        # 1. Main Agent Model & Optimizer
        if "model_state_dict" in state and state["model_state_dict"] is not None and hasattr(model, "load_state_dict"):
            model.load_state_dict(state["model_state_dict"], strict=False)

        if optimizer is not None and "optimizer_state_dict" in state and state["optimizer_state_dict"] is not None:
            try:
                optimizer.load_state_dict(state["optimizer_state_dict"])
            except Exception:
                pass

        # 2. Main Exploiter Model & Optimizer
        if main_exploiter_model is not None and "main_exploiter_model_state_dict" in state and state["main_exploiter_model_state_dict"] is not None:
            try:
                main_exploiter_model.load_state_dict(state["main_exploiter_model_state_dict"], strict=False)
            except Exception:
                pass

        if main_exploiter_optimizer is not None and "main_exploiter_optimizer_state_dict" in state and state["main_exploiter_optimizer_state_dict"] is not None:
            try:
                main_exploiter_optimizer.load_state_dict(state["main_exploiter_optimizer_state_dict"])
            except Exception:
                pass

        # 3. League Exploiter Model & Optimizer
        if league_exploiter_model is not None and "league_exploiter_model_state_dict" in state and state["league_exploiter_model_state_dict"] is not None:
            try:
                league_exploiter_model.load_state_dict(state["league_exploiter_model_state_dict"], strict=False)
            except Exception:
                pass

        if league_exploiter_optimizer is not None and "league_exploiter_optimizer_state_dict" in state and state["league_exploiter_optimizer_state_dict"] is not None:
            try:
                league_exploiter_optimizer.load_state_dict(state["league_exploiter_optimizer_state_dict"])
            except Exception:
                pass

        return gen, entropy_coef
