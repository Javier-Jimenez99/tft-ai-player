"""Training loop and League Self-Play orchestrator for TFT RL."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import torch

from tft_ai_player.rl.algorithms.ppo import MaskablePPO, RolloutBuffer
from tft_ai_player.rl.evaluation.evaluator import TournamentEvaluator
from tft_ai_player.rl.league.league_manager import LeagueManager
from tft_ai_player.rl.logger import WandBLogger
from tft_ai_player.rl.models.networks import TFTActorCritic
from tft_ai_player.rl.types import AgentArchetype, AgentRole
from tft_ai_player.simulation.actions import TOTAL_DISCRETE_ACTIONS
from tft_ai_player.simulation.combat import CombatResolver, HeuristicCombatResolver, MLCombatResolver
from tft_ai_player.simulation.config import SetData
from tft_ai_player.simulation.gym_env import TFTEnv
from tft_ai_player.simulation.sets.set18 import get_set18_data


class LeagueTrainer:
    """Orchestrates multi-agent PPO training, Prioritized Fictitious Self-Play,

    dynamic entropy scheduling, tournament evaluation, and checkpointing.
    """

    def __init__(
        self,
        set_data: SetData | None = None,
        model: TFTActorCritic | None = None,
        archetype: AgentArchetype = AgentArchetype.GENERALIST,
        tri_tier: bool = True,
        lr: float = 3e-4,
        buffer_size: int = 2048,
        batch_size: int = 128,
        num_epochs: int = 4,
        hidden_dim: int = 384,
        entropy_coef_start: float = 0.05,
        entropy_coef_min: float = 0.005,
        entropy_decay_rate: float = 0.9985,
        device: str | None = None,
        checkpoint_dir: str | Path = "D:/tft-winner-data/set18/models",
        run_name: str | None = None,
        round_winner_model_path: str | Path | None = "D:/tft-winner-data/set18/models/round_winner_model.joblib",
        use_wandb: bool = True,
        wandb_project: str = "tft-ai-league",
        wandb_entity: str | None = None,
        wandb_group: str | None = None,
        wandb_run_id: str | None = None,
    ) -> None:
        self.set_data = set_data or get_set18_data()
        self.archetype = archetype
        self.tri_tier = tri_tier
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name or "ppo_tri_tier_league_v1"
        self.config: dict[str, Any] = {
            "archetype": archetype.value,
            "tri_tier": tri_tier,
            "lr": lr,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "hidden_dim": hidden_dim,
            "entropy_coef_start": entropy_coef_start,
            "entropy_coef_min": entropy_coef_min,
            "entropy_decay_rate": entropy_decay_rate,
            "device": str(self.device),
        }
        self.wandb_logger = WandBLogger(
            project=wandb_project,
            entity=wandb_entity,
            group=wandb_group or self.run_name,
            run_name=self.run_name,
            run_id=wandb_run_id,
            enabled=use_wandb,
            config=self.config,
        )
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.entropy_coef_start = entropy_coef_start
        self.entropy_coef_min = entropy_coef_min
        self.entropy_decay_rate = entropy_decay_rate

        # 1. Setup ML Combat Resolver with pre-trained non-ensemble round winner model
        self.combat_resolver: CombatResolver
        rw_path = Path(round_winner_model_path) if round_winner_model_path else None
        if rw_path and rw_path.exists():
            try:
                # Test import to ensure C-extensions (scipy, sklearn) load without DLL block
                from tft_ai_player.round_winner.trainer import RoundWinnerPredictor
                self.combat_resolver = MLCombatResolver(model_pipeline=rw_path)
            except (Exception, BaseException):
                self.combat_resolver = HeuristicCombatResolver()
        else:
            self.combat_resolver = HeuristicCombatResolver()

        # 2. League manager and evaluator
        self.league = LeagueManager(checkpoint_dir=str(self.checkpoint_dir))
        self.evaluator = TournamentEvaluator(
            self.league,
            set_data=self.set_data,
            combat_resolver=self.combat_resolver,
        )

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.metrics_history: list[dict[str, Any]] = []
        self.best_elo: float = 1200.0

        from tft_ai_player.simulation.gym_env import CurriculumBotFactory

        # 3. Agent 1: Main Agent (Generalist Mastery)
        self.main_agent_id = "main_agent_v1"
        self.model = model or TFTActorCritic(
            num_champs=len(self.set_data.champions) + 1,
            num_items=len(self.set_data.items) + 1,
            hidden_dim=hidden_dim,
        )
        self.model.to(self.device)
        self.ppo = MaskablePPO(
            self.model,
            lr=lr,
            entropy_coef=entropy_coef_start,
            entropy_coef_start=entropy_coef_start,
            entropy_coef_min=entropy_coef_min,
            entropy_decay_rate=entropy_decay_rate,
            device=self.device,
        )
        if self.main_agent_id not in self.league.profiles:
            self.league.register_agent(
                agent_id=self.main_agent_id,
                name="Main PPO Agent",
                role=AgentRole.MAIN,
            )
        self.league.profiles[self.main_agent_id].archetype = self.archetype

        self.bot_factory = CurriculumBotFactory(
            generation=1,
            league=self.league,
            set_data=self.set_data,
            device=self.device,
            active_model=self.model,
            focal_role=AgentRole.MAIN,
            focal_agent_id=self.main_agent_id,
        )
        self.env = TFTEnv(
            set_data=self.set_data,
            combat_resolver=self.combat_resolver,
            bot_factory=self.bot_factory,
            archetype=self.archetype,
        )

        # 4. Agent 2: Main Exploiter (Specialized to find & punish Main Agent weaknesses)
        self.main_exploiter_id = "main_exploiter_v1"
        self.main_exploiter_model = TFTActorCritic(
            num_champs=len(self.set_data.champions) + 1,
            num_items=len(self.set_data.items) + 1,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.main_exploiter_ppo = MaskablePPO(
            self.main_exploiter_model,
            lr=lr,
            entropy_coef=entropy_coef_start,
            entropy_coef_start=entropy_coef_start,
            entropy_coef_min=entropy_coef_min,
            entropy_decay_rate=entropy_decay_rate,
            device=self.device,
        )
        if self.main_exploiter_id not in self.league.profiles:
            self.league.register_agent(
                agent_id=self.main_exploiter_id,
                name="Main Exploiter Agent",
                role=AgentRole.MAIN_EXPLOITER,
            )
        self.league.profiles[self.main_exploiter_id].archetype = AgentArchetype.HYPER_ROLL
        self.main_exploiter_factory = CurriculumBotFactory(
            generation=1,
            league=self.league,
            set_data=self.set_data,
            device=self.device,
            active_model=self.main_exploiter_model,
            focal_role=AgentRole.MAIN_EXPLOITER,
            focal_agent_id=self.main_exploiter_id,
        )
        self.main_exploiter_env = TFTEnv(
            set_data=self.set_data,
            combat_resolver=self.combat_resolver,
            bot_factory=self.main_exploiter_factory,
            archetype=AgentArchetype.HYPER_ROLL,
        )

        # 5. Agent 3: League Exploiter (Hunts for global cheese strategies across all HoF checkpoints)
        self.league_exploiter_id = "league_exploiter_v1"
        self.league_exploiter_model = TFTActorCritic(
            num_champs=len(self.set_data.champions) + 1,
            num_items=len(self.set_data.items) + 1,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.league_exploiter_ppo = MaskablePPO(
            self.league_exploiter_model,
            lr=lr,
            entropy_coef=entropy_coef_start,
            entropy_coef_start=entropy_coef_start,
            entropy_coef_min=entropy_coef_min,
            entropy_decay_rate=entropy_decay_rate,
            device=self.device,
        )
        if self.league_exploiter_id not in self.league.profiles:
            self.league.register_agent(
                agent_id=self.league_exploiter_id,
                name="League Exploiter Agent",
                role=AgentRole.LEAGUE_EXPLOITER,
            )
        self.league.profiles[self.league_exploiter_id].archetype = AgentArchetype.FAST8_FLEX
        self.league_exploiter_factory = CurriculumBotFactory(
            generation=1,
            league=self.league,
            set_data=self.set_data,
            device=self.device,
            active_model=self.league_exploiter_model,
            focal_role=AgentRole.LEAGUE_EXPLOITER,
            focal_agent_id=self.league_exploiter_id,
        )
        self.league_exploiter_env = TFTEnv(
            set_data=self.set_data,
            combat_resolver=self.combat_resolver,
            bot_factory=self.league_exploiter_factory,
            archetype=AgentArchetype.FAST8_FLEX,
        )

    def load_checkpoint(self, checkpoint_path: str | Path | None = None) -> int:
        """Resume training from a saved checkpoint. Returns resumed generation."""
        resumed = self.league.checkpoints.load_training_state(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.ppo.optimizer,
        )
        if resumed:
            gen = resumed.get("generation", 0)
            self.ppo.entropy_coef = resumed.get("entropy_coef", self.ppo.entropy_coef)
            self.metrics_history = resumed.get("metrics_history", [])

            # Restore continuous Elo rating so there are no restart step-jumps
            if self.metrics_history:
                last_elo = next((e["league_elo"] for e in reversed(self.metrics_history) if "league_elo" in e), None)
                if last_elo is not None and self.main_agent_id in self.league.profiles:
                    self.league.profiles[self.main_agent_id].elo.rating = float(last_elo)

            try:
                event_files = list(self.tb_logger.run_dir.glob("events.out.tfevents.*"))
                if self.metrics_history and (not event_files or sum(f.stat().st_size for f in event_files) < 500000):
                    for idx, entry in enumerate(self.metrics_history):
                        g = entry.get("generation", idx + 1)
                        self.tb_logger.log_generation(g, entry)
                    self.tb_logger.writer.flush()
            except Exception:
                pass
            return gen
        return 0

    def collect_rollouts(
        self,
        buffer: RolloutBuffer,
        num_steps: int = 512,
        model: TFTActorCritic | None = None,
        env: TFTEnv | None = None,
        seed: int | None = None,
    ) -> dict[str, float]:
        """Collect on-policy interaction rollouts carrying recurrent GRU hidden states."""
        target_model = model or self.model
        target_env = env or self.env
        target_model.eval()

        obs, info = target_env.reset(seed=seed)
        step_count = 0
        total_reward = 0.0
        reward_breakdown_sums: dict[str, float] = {}
        episodes_completed = 0
        placement_history: list[int] = []

        # Track action distributions across macro categories
        from tft_ai_player.simulation.actions import ACTION_GROUP_NAMES, get_action_group
        action_group_counts: dict[str, int] = {grp: 0 for grp in ACTION_GROUP_NAMES}

        # Track round efficiency & pass quality metrics
        rounds_advanced = 0
        pass_missed_crafts = 0
        pass_missed_upgrades = 0
        pass_clean_econ = 0

        hidden_state: torch.Tensor | None = None

        while step_count < num_steps:
            mask = info.get("action_mask", np.ones(TOTAL_DISCRETE_ACTIONS, dtype=bool))

            # Convert observation dict to batched torch tensors
            tensor_obs: dict[str, torch.Tensor] = {}
            for k, v in obs.items():
                if k != "action_mask":
                    tensor_obs[k] = torch.as_tensor(v, dtype=torch.float32, device=self.device).unsqueeze(0)

            mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)

            with torch.no_grad():
                action_t, log_prob_t, val_t, next_hidden = target_model.get_action(
                    tensor_obs,
                    mask_tensor,
                    hidden_state=hidden_state,
                    return_hidden=True,
                )
                action = int(action_t.item())
                log_prob = float(log_prob_t.item())
                val = float(val_t.item())

            # Tally action group
            group_name = get_action_group(action)
            action_group_counts[group_name] = action_group_counts.get(group_name, 0) + 1

            next_obs, reward, terminated, truncated, next_info = target_env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Track round advancement & pass context
            if next_info.get("round_advanced", False):
                rounds_advanced += 1
                p_diag = next_info.get("pass_diagnostics", {})
                if p_diag.get("pass_with_uncombined_items", False):
                    pass_missed_crafts += 1
                if p_diag.get("pass_with_upgradable_shop", False):
                    pass_missed_upgrades += 1
                if p_diag.get("pass_excess_gold", False) and not p_diag.get("pass_with_upgradable_shop", False):
                    pass_clean_econ += 1

            # Accumulate fine-grained decomposed rewards
            rb = next_info.get("reward_breakdown", {})
            for k, v in rb.items():
                reward_breakdown_sums[k] = reward_breakdown_sums.get(k, 0.0) + float(v)

            buffer.add(
                obs_dict=obs,
                action=action,
                action_mask=mask,
                reward=reward,
                value=val,
                log_prob=log_prob,
                done=done,
                hidden_state=hidden_state,
            )

            step_count += 1
            obs = next_obs
            info = next_info
            hidden_state = None if done else next_hidden

            if done:
                episodes_completed += 1
                focal = target_env.game.get_focal_player()
                placement_history.append(focal.placement or (1 if focal.alive else 8))
                obs, info = target_env.reset()

        # Compute GAE
        last_val = 0.0
        if not done:
            tensor_obs = {
                k: torch.as_tensor(v, dtype=torch.float32, device=self.device).unsqueeze(0)
                for k, v in obs.items()
                if k != "action_mask"
            }
            with torch.no_grad():
                _, _, v_last = target_model.get_action(
                    tensor_obs,
                    torch.as_tensor(info.get("action_mask", np.ones(TOTAL_DISCRETE_ACTIONS, dtype=bool)), device=self.device).unsqueeze(0),
                    hidden_state=hidden_state,
                )
                last_val = float(v_last.item())

        buffer.compute_returns_and_advantages(last_value=last_val, done=done)

        n_steps = max(step_count, 1)
        avg_placement = float(np.mean(placement_history)) if placement_history else 4.5
        breakdown_averages = {
            k: v / n_steps for k, v in reward_breakdown_sums.items()
        }
        # Compute action distribution proportions
        action_distribution = {
            grp: action_group_counts.get(grp, 0) / n_steps for grp in ACTION_GROUP_NAMES
        }
        # Compute actions per round (APM) and pass efficiency
        n_rounds = max(rounds_advanced, 1)
        non_pass_actions = step_count - action_group_counts.get("Pass", 0)
        actions_per_round = non_pass_actions / n_rounds
        pass_missed_craft_rate = pass_missed_crafts / n_rounds
        pass_missed_upgrade_rate = pass_missed_upgrades / n_rounds
        pass_clean_econ_rate = pass_clean_econ / n_rounds

        return {
            "steps": step_count,
            "episodes": episodes_completed,
            "mean_reward": total_reward / n_steps,
            "avg_placement": avg_placement,
            "action_counts": action_group_counts,
            "action_distribution": action_distribution,
            "rounds_advanced": rounds_advanced,
            "actions_per_round": actions_per_round,
            "pass_missed_craft_rate": pass_missed_craft_rate,
            "pass_missed_upgrade_rate": pass_missed_upgrade_rate,
            "pass_clean_econ_rate": pass_clean_econ_rate,
            **breakdown_averages,
        }

    def train_iteration(
        self,
        generation: int,
        rollout_steps: int = 512,
        eval_every: int = 5,
        eval_num_seeds: int = 3,
    ) -> dict[str, Any]:
        """Execute one complete training generation: Schedule -> Rollouts -> PPO Update -> Eval -> Checkpoint."""
        # 1. Update dynamic entropy schedules
        current_entropy_coef = self.ppo.step_entropy_schedule(generation)
        if hasattr(self.env, "set_curriculum_generation"):
            self.env.set_curriculum_generation(generation)

        # 2. Main Agent: Rollout & PPO Update
        dummy_obs, dummy_info = self.env.reset(seed=100 + generation)
        main_buffer = RolloutBuffer(
            buffer_size=rollout_steps + 64,
            obs_sample=dummy_obs,
            action_dim=TOTAL_DISCRETE_ACTIONS,
            hidden_dim=self.model.hidden_dim,
            device=self.device,
        )
        collect_metrics = self.collect_rollouts(
            main_buffer,
            num_steps=rollout_steps,
            model=self.model,
            env=self.env,
            seed=1000 + generation,
        )
        ppo_metrics = self.ppo.train_epoch(
            buffer=main_buffer,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
        )

        tri_tier_data: dict[str, dict[str, Any]] = {
            "Main_Agent": {
                "elo": self.league.profiles[self.main_agent_id].elo.rating,
                **collect_metrics,
                **ppo_metrics,
            },
        }

        # 3. Optional Co-Training for Main Exploiter & League Exploiter (Tri-Tier Ecosystem)
        if self.tri_tier:
            # Main Exploiter update
            self.main_exploiter_ppo.step_entropy_schedule(generation)
            me_steps = max(128, rollout_steps // 2)
            me_buffer = RolloutBuffer(
                buffer_size=me_steps + 64,
                obs_sample=dummy_obs,
                action_dim=TOTAL_DISCRETE_ACTIONS,
                hidden_dim=self.main_exploiter_model.hidden_dim,
                device=self.device,
            )
            me_collect = self.collect_rollouts(
                me_buffer,
                num_steps=me_steps,
                model=self.main_exploiter_model,
                env=self.main_exploiter_env,
                seed=2000 + generation,
            )
            me_ppo = self.main_exploiter_ppo.train_epoch(
                buffer=me_buffer,
                batch_size=self.batch_size,
                num_epochs=self.num_epochs,
            )
            tri_tier_data["Main_Exploiter"] = {
                "elo": self.league.profiles[self.main_exploiter_id].elo.rating,
                **me_collect,
                **me_ppo,
            }

            # League Exploiter update
            self.league_exploiter_ppo.step_entropy_schedule(generation)
            le_steps = max(128, rollout_steps // 2)
            le_buffer = RolloutBuffer(
                buffer_size=le_steps + 64,
                obs_sample=dummy_obs,
                action_dim=TOTAL_DISCRETE_ACTIONS,
                hidden_dim=self.league_exploiter_model.hidden_dim,
                device=self.device,
            )
            le_collect = self.collect_rollouts(
                le_buffer,
                num_steps=le_steps,
                model=self.league_exploiter_model,
                env=self.league_exploiter_env,
                seed=3000 + generation,
            )
            le_ppo = self.league_exploiter_ppo.train_epoch(
                buffer=le_buffer,
                batch_size=self.batch_size,
                num_epochs=self.num_epochs,
            )
            tri_tier_data["League_Exploiter"] = {
                "elo": self.league.profiles[self.league_exploiter_id].elo.rating,
                **le_collect,
                **le_ppo,
            }

        metrics: dict[str, Any] = {
            "generation": generation,
            "entropy_coef": current_entropy_coef,
            "tri_tier": tri_tier_data,
            **collect_metrics,
            **ppo_metrics,
        }

        # 4. Periodic Evaluation against 4-tier benchmark ladder & Multilateral Tournament
        is_best = False
        if generation % eval_every == 0:
            # A. Static Benchmark Ladder (Frozen Model & Common Random Numbers)
            eval_res = self.evaluator.run_tiered_benchmark(
                candidate_agent_id=self.main_agent_id,
                candidate_model=self.model,
                num_seeds_per_tier=max(1, eval_num_seeds),
                base_seed=1000 + generation,
            )
            metrics.update(eval_res)

            # B. Multi-Agent League Tournament match to update multilateral Elos across all active bots
            tournament_seats = [
                self.main_agent_id,
                self.main_exploiter_id if self.tri_tier else self.main_agent_id,
                self.league_exploiter_id if self.tri_tier else "bot_standard_tempo",
                "bot_standard_tempo",
                "bot_greedy_banker",
                "bot_hyper_roll_exploiter",
                "bot_fast9_econ_exploiter",
                "bot_random",
            ]
            self.evaluator.run_match(agent_seats=tournament_seats, seed=5000 + generation)

            # Update Tri-Tier Elos after tournament
            for aid in [self.main_agent_id, self.main_exploiter_id, self.league_exploiter_id]:
                if aid in self.league.profiles and aid in tri_tier_data:
                    tri_tier_data[aid]["elo"] = self.league.profiles[aid].elo.rating

            if eval_res.get("league_elo", 1200.0) > self.best_elo:
                self.best_elo = eval_res["league_elo"]
                is_best = True

            # Save snapshot to Hall of Fame if milestone reached
            if generation > 0 and generation % (eval_every * 2) == 0:
                self.league.create_hall_of_fame_snapshot(
                    source_agent_id=self.main_agent_id,
                    generation=generation,
                    weights=self.model.state_dict(),
                )
                if self.tri_tier:
                    self.league.create_hall_of_fame_snapshot(
                        source_agent_id=self.main_exploiter_id,
                        generation=generation,
                        weights=self.main_exploiter_model.state_dict(),
                    )
                    self.league.create_hall_of_fame_snapshot(
                        source_agent_id=self.league_exploiter_id,
                        generation=generation,
                        weights=self.league_exploiter_model.state_dict(),
                    )

        # 5. Weights & Biases Experiment Tracking
        self.wandb_logger.log_generation(generation, metrics)

        # 6. Checkpoint persistence with Tri-Tier multi-agent models and WandB state
        self.metrics_history.append(metrics)
        self.league.checkpoints.save_training_state(
            model=self.model,
            optimizer=self.ppo.optimizer,
            generation=generation,
            entropy_coef=current_entropy_coef,
            metrics_history=self.metrics_history,
            is_best=is_best,
            main_exploiter_model=self.main_exploiter_model if self.tri_tier else None,
            main_exploiter_optimizer=self.main_exploiter_ppo.optimizer if self.tri_tier else None,
            league_exploiter_model=self.league_exploiter_model if self.tri_tier else None,
            league_exploiter_optimizer=self.league_exploiter_ppo.optimizer if self.tri_tier else None,
            config=self.config,
            wandb_run_id=self.wandb_logger.run_id,
        )

        return metrics

    def load_checkpoint(self, checkpoint_path: str | Path | None = None) -> int:
        """Restore all 3 models, optimizers, entropy schedule, and WandB run ID from checkpoint.

        Returns:
            The integer generation index that was successfully restored.
        """
        state = self.league.checkpoints.load_training_state(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.ppo.optimizer,
            main_exploiter_model=self.main_exploiter_model if self.tri_tier else None,
            main_exploiter_optimizer=self.main_exploiter_ppo.optimizer if self.tri_tier else None,
            league_exploiter_model=self.league_exploiter_model if self.tri_tier else None,
            league_exploiter_optimizer=self.league_exploiter_ppo.optimizer if self.tri_tier else None,
        )
        if not state:
            return 0

        resumed_gen = int(state.get("generation", 0))
        self.metrics_history = state.get("metrics_history", [])
        entropy_coef = float(state.get("entropy_coef", self.ppo.entropy_coef))
        self.ppo.entropy_coef = entropy_coef
        if self.tri_tier:
            self.main_exploiter_ppo.entropy_coef = entropy_coef
            self.league_exploiter_ppo.entropy_coef = entropy_coef

        # If wandb_run_id was saved and wandb is active, connect back to that exact run
        saved_run_id = state.get("wandb_run_id")
        if saved_run_id and self.wandb_logger.enabled and self.wandb_logger.run is None:
            self.wandb_logger.run_id = saved_run_id
            self.wandb_logger.__init__(
                project=self.wandb_logger.project,
                entity=self.wandb_logger.entity,
                group=self.wandb_logger.group,
                run_name=self.wandb_logger.run_name,
                run_id=saved_run_id,
                resume="allow",
                enabled=True,
            )

        return resumed_gen
