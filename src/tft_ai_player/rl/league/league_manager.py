"""AlphaStar Multi-Agent League Manager and Prioritized Fictitious Self-Play (PFSP) Matchmaker."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch

from tft_ai_player.rl.league.checkpoints import CheckpointManager
from tft_ai_player.rl.league.elo import MultilateralEloSystem
from tft_ai_player.rl.types import AgentProfile, AgentRole, EloRating, MatchResult
from tft_ai_player.simulation.bots import (
    BaseBot,
    BotAlphaFast8,
    BotBetaHyperroll,
    BotGammaGreedy,
    RandomBot,
)


class LeagueManager:
    """Manages the AlphaStar-style League ecosystem:

    - Main Agent (Generalist)
    - 15 Exploiter Specialists (locked to Z-Index targets z_1 .. z_15)
    - Historical Snapshots (archived every 50 generations)
    - Deterministic Benchmark Bots (Alpha, Beta, Gamma)
    """

    def __init__(
        self,
        checkpoint_dir: str | Path = "checkpoints/league",
        z_index_path: str | Path | None = None,
        num_exploiters: int = 15,
        elo_k_factor: float = 32.0,
        rng_seed: int = 42,
        set_data: Any | None = None,
        device: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self.set_data = set_data
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_manager = CheckpointManager(base_dir=checkpoint_dir)
        self.elo_system = MultilateralEloSystem(k_factor=elo_k_factor)
        self.num_exploiters = num_exploiters
        self.rng = random.Random(rng_seed)
        self.generation = 0

        self.profiles: dict[str, AgentProfile] = {}
        self.match_history: list[MatchResult] = []
        self.trunk: Any | None = None
        self.main_model: Any | None = None
        self.exploiter_model: Any | None = None

        # Load Z-Index centroids if available
        self.z_centroids: np.ndarray | None = None
        if z_index_path and Path(z_index_path).exists():
            try:
                data = torch.load(z_index_path, map_location="cpu", weights_only=False)
                if isinstance(data, dict):
                    if "z_index" in data:
                        raw_z = data["z_index"]
                        self.z_centroids = raw_z.cpu().numpy() if isinstance(raw_z, torch.Tensor) else np.array(raw_z)
                    elif "centroids" in data:
                        raw_z = data["centroids"]
                        self.z_centroids = raw_z.cpu().numpy() if isinstance(raw_z, torch.Tensor) else np.array(raw_z)
                elif isinstance(data, torch.Tensor):
                    self.z_centroids = data.cpu().numpy()
            except Exception:
                pass

        from tft_ai_player.rl.agent_policy import ZMetricsTracker
        self.metrics_tracker = ZMetricsTracker()

        self._initialize_league()

    def _initialize_league(self) -> None:
        """Initialize league members: Main agent, K Exploiters, and Benchmark Bots."""
        # 1. Main Agent
        self.main_agent_id = "main_agent"
        self.profiles[self.main_agent_id] = AgentProfile(
            agent_id=self.main_agent_id,
            name="AlphaTFT-Main",
            role=AgentRole.MAIN,
            generation=0,
            metadata={"description": "Persistent generalist agent optimizing global survival"},
        )

        # 2. 15 Exploiter Specialists (z_1 .. z_15)
        for k in range(self.num_exploiters):
            exp_id = f"exploiter_z{k+1:02d}"
            target_z = self.z_centroids[k] if (self.z_centroids is not None and k < len(self.z_centroids)) else None
            self.profiles[exp_id] = AgentProfile(
                agent_id=exp_id,
                name=f"Specialist-Z{k+1:02d}",
                role=AgentRole.EXPLOITER,
                target_z=target_z,
                target_z_index=k,
                generation=0,
                metadata={"archetype_id": k, "description": f"Specialist agent targeting Z-Index centroid {k+1}"},
            )

        # 3. Deterministic Benchmark Bots
        self.profiles["bot_alpha"] = AgentProfile(
            agent_id="bot_alpha",
            name="Bot-Alpha (Fast-8)",
            role=AgentRole.BASELINE,
            metadata={"bot_type": "BotAlphaFast8"},
        )
        self.profiles["bot_beta"] = AgentProfile(
            agent_id="bot_beta",
            name="Bot-Beta (Hyperroll)",
            role=AgentRole.BASELINE,
            metadata={"bot_type": "BotBetaHyperroll"},
        )
        self.profiles["bot_gamma"] = AgentProfile(
            agent_id="bot_gamma",
            name="Bot-Gamma (Greedy)",
            role=AgentRole.BASELINE,
            metadata={"bot_type": "BotGammaGreedy"},
        )

    def archive_snapshot(self, agent_id: str, generation: int, model_weights: Any | None = None) -> str:
        """Archive a frozen historical snapshot of an agent."""
        if agent_id not in self.profiles:
            return ""

        parent_prof = self.profiles[agent_id]
        snap_id = f"{agent_id}_gen{generation:04d}"

        snap_prof = AgentProfile(
            agent_id=snap_id,
            name=f"{parent_prof.name} (Gen {generation})",
            role=AgentRole.HISTORICAL,
            target_z=parent_prof.target_z,
            target_z_index=parent_prof.target_z_index,
            generation=generation,
            elo=EloRating(rating=parent_prof.elo.rating),
            metadata={"parent_agent_id": agent_id, "snapshot_generation": generation},
        )

        self.checkpoint_manager.save_agent_profile(snap_prof, weights=model_weights)
        self.profiles[snap_id] = snap_prof
        return snap_id

    def sample_pfsp_opponents(
        self,
        focal_agent_id: str = "main_agent",
        num_opponents: int = 7,
    ) -> list[str]:
        """Sample 7 opponents for a match using AlphaStar Prioritized Fictitious Self-Play (PFSP).

        Training Matchmaker Distribution (100% Dynamic Neural Meta):
          - Active Self-Play Mirrors (45%)
          - Meta-Exploiter Pool with random z_k (40%)
          - Historical Hall of Fame Snapshots (15%)
        """
        focal_prof = self.profiles.get(focal_agent_id)
        selected_opponents: list[str] = []

        # Categorize available opponents
        exploiters = [aid for aid, p in self.profiles.items() if p.role == AgentRole.EXPLOITER and aid != focal_agent_id]
        historical = [aid for aid, p in self.profiles.items() if p.role == AgentRole.HISTORICAL and aid != focal_agent_id]

        for _ in range(num_opponents):
            r_cat = self.rng.random()

            # 1. Active Self-Play (45%)
            if r_cat < 0.45:
                selected_opponents.append(focal_agent_id)

            # 2. Meta-Exploiter Pool with random z_k (40%)
            elif r_cat < 0.85 and exploiters:
                chosen = self._sample_pfsp_from_list(focal_prof, exploiters)
                selected_opponents.append(chosen)

            # 3. Historical Hall of Fame Snapshots (15%)
            elif historical:
                chosen = self._sample_pfsp_from_list(focal_prof, historical)
                selected_opponents.append(chosen)

            # Fallback to active self-play / exploiter
            else:
                selected_opponents.append(self.rng.choice(exploiters) if exploiters else focal_agent_id)

        return selected_opponents

    def _sample_pfsp_from_list(self, focal_prof: AgentProfile | None, candidates: list[str]) -> str:
        """Sample from candidate list with PFSP probability P(j) proportional to f(1 - M_{Main, j}), f(x)=x^2."""
        if not candidates:
            return self.main_agent_id
        if focal_prof is None or len(candidates) == 1:
            return self.rng.choice(candidates)

        weights = []
        for opp_id in candidates:
            # Win rate of focal agent against opponent
            win_rate = focal_prof.get_win_rate_vs(opp_id)
            loss_rate = max(0.01, 1.0 - win_rate)
            # f(x) = x^2 weighting prioritized against opponents that beat the agent
            weight = loss_rate ** 2
            weights.append(weight)

        total_w = sum(weights)
        if total_w <= 0:
            return self.rng.choice(candidates)

        probs = [w / total_w for w in weights]
        return self.rng.choices(candidates, weights=probs, k=1)[0]

    def record_match_result(
        self,
        match_id: str,
        placements: dict[str, int],
        scores: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Record match outcome and update multilateral Elo ratings."""
        result = MatchResult(
            match_id=match_id,
            placements=placements,
            scores=scores or {},
        )
        self.match_history.append(result)

        # Update Elo ratings
        deltas = self.elo_system.update_lobby_ratings(placements, self.profiles)
        return deltas

    def register_agent(
        self,
        agent_id: str,
        name: str,
        role: AgentRole,
        target_z: np.ndarray | None = None,
        target_z_index: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentProfile:
        """Register a new agent profile in the league."""
        prof = AgentProfile(
            agent_id=agent_id,
            name=name,
            role=role,
            target_z=target_z,
            target_z_index=target_z_index,
            generation=self.generation,
            metadata=metadata or {},
        )
        self.profiles[agent_id] = prof
        return prof

    def get_leaderboard(self) -> list[dict[str, Any]]:
        """Return sorted leaderboard list across all league profiles."""
        sorted_profs = sorted(
            self.profiles.values(),
            key=lambda p: (p.elo.rating, p.elo.win_rate, p.elo.top4_rate),
            reverse=True,
        )
        entries: list[dict[str, Any]] = []
        for rank, p in enumerate(sorted_profs, start=1):
            entries.append(
                {
                    "rank": rank,
                    "agent_id": p.agent_id,
                    "name": p.name,
                    "role": p.role.value,
                    "rating": p.elo.rating,
                    "games": p.elo.games_played,
                    "wins": p.elo.wins,
                    "top4s": p.elo.top4s,
                    "win_rate": f"{p.elo.win_rate * 100:.1f}%",
                    "top4_rate": f"{p.elo.top4_rate * 100:.1f}%",
                    "avg_placement": f"{p.elo.avg_placement:.2f}",
                }
            )
        return entries

    def set_neural_agents(
        self,
        main_model: Any | None = None,
        exploiter_model: Any | None = None,
        trunk: Any | None = None,
        buffer: Any | None = None,
    ) -> None:
        """Register active neural models and rollout buffer for dynamic parallel self-play sparring."""
        self.main_model = main_model
        self.exploiter_model = exploiter_model
        self.trunk = trunk
        self.buffer = buffer

    def _get_or_load_snapshot_model(self, agent_id: str) -> Any | None:
        """Load and cache frozen weights for historical Hall of Fame snapshots."""
        if not hasattr(self, "_snapshot_models_cache"):
            self._snapshot_models_cache: dict[str, Any] = {}
        if agent_id in self._snapshot_models_cache:
            return self._snapshot_models_cache[agent_id]

        prof = self.profiles.get(agent_id) or self.checkpoint_manager.load_agent_profile(agent_id)
        candidate_path = Path(prof.checkpoint_path) if (prof and prof.checkpoint_path) else (self.checkpoint_manager.snapshots_dir / agent_id / "weights.pt")
        if candidate_path.exists():
            try:
                from tft_ai_player.rl.models.networks import TFTActorCritic
                from tft_ai_player.simulation.actions import TOTAL_DISCRETE_ACTIONS
                obs_dim = 704
                if self.trunk is not None:
                    obs_dim = self.trunk.fused_dim + 384
                snap_model = TFTActorCritic(obs_dim=obs_dim, action_dim=TOTAL_DISCRETE_ACTIONS, hidden_dim=512)
                state_dict = torch.load(candidate_path, map_location=self.device, weights_only=False)
                snap_model.load_state_dict(state_dict)
                snap_model.to(self.device).eval()
                for p in snap_model.parameters():
                    p.requires_grad = False
                self._snapshot_models_cache[agent_id] = snap_model
                return snap_model
            except Exception as e:
                import traceback
                traceback.print_exc()
        return None

    def sample_opponent_bot(self, focal_agent_id: str = "main_agent") -> BaseBot:
        """Instantiate an executable opponent bot for environment simulation."""
        from tft_ai_player.rl.agent_policy import RLBot

        opp_ids = self.sample_pfsp_opponents(focal_agent_id, num_opponents=1)
        opp_id = opp_ids[0] if opp_ids else "bot_alpha"
        neural_model = getattr(self, "exploiter_model", None) or getattr(self, "main_model", None)
        prof = self.profiles.get(opp_id)

        # 1. Historical Hall of Fame Snapshots (Gen 50, Gen 100, Gen 150...) -> FROZEN: No Buffer Recording
        if opp_id.startswith("main_agent_gen") or (prof and prof.role == AgentRole.HISTORICAL):
            snap_model = self._get_or_load_snapshot_model(opp_id)
            if snap_model is not None and self.set_data is not None:
                return RLBot(
                    model=snap_model,
                    set_data=self.set_data,
                    trunk=getattr(self, "trunk", None),
                    device=self.device,
                    record_transitions=False,
                    is_historical=True,
                )

        # 2. Live Neural Opponents with Random Z-Archetype Conditioning -> LIVE: Records Transitions into Buffer!
        if (opp_id.startswith("exploiter") or opp_id == "exploiter_agent") and neural_model is not None:
            target_z = prof.target_z if prof is not None else None
            k = prof.target_z_index if prof is not None else None
            if target_z is None and self.z_centroids is not None and len(self.z_centroids) > 0:
                k = int(np.random.randint(0, len(self.z_centroids)))
                target_z = self.z_centroids[k]
            if self.set_data is not None:
                return RLBot(
                    model=neural_model,
                    set_data=self.set_data,
                    trunk=getattr(self, "trunk", None),
                    target_z=target_z,
                    z_index=k,
                    z_centroids=self.z_centroids,
                    metrics_tracker=self.metrics_tracker,
                    device=self.device,
                    buffer=getattr(self, "buffer", None),
                    record_transitions=True,
                    is_historical=False,
                )

        # 3. Main Agent Self-Play (Mirrors) -> LIVE: Records Transitions into Buffer!
        if (opp_id == "main_agent" or opp_id == "main_agent_v1") and getattr(self, "main_model", None) is not None:
            if self.set_data is not None and focal_agent_id != "main_agent":
                return RLBot(
                    model=self.main_model,
                    set_data=self.set_data,
                    trunk=getattr(self, "trunk", None),
                    device=self.device,
                    buffer=getattr(self, "buffer", None),
                    record_transitions=True,
                    is_historical=False,
                )

        # 4. Deterministic Baseline Bots
        if opp_id == "bot_alpha":
            return BotAlphaFast8()
        if opp_id == "bot_beta":
            return BotBetaHyperroll()
        if opp_id == "bot_gamma":
            return BotGammaGreedy()

        # Fallback default
        return BotAlphaFast8()


