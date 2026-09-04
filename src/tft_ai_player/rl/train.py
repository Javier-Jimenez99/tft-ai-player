"""Training orchestrator for PPO & AlphaStar League Pipeline."""

from __future__ import annotations

import copy
import logging
import math
from pathlib import Path
from typing import Any
import numpy as np
import torch

from tft_ai_player.embeddings.model import MultiModalFusionTrunk
from tft_ai_player.embeddings.transition import StateTransitionPredictor
from tft_ai_player.rl.algorithms.ppo import MaskablePPO, RolloutBuffer
from tft_ai_player.rl.evaluation.evaluator import BenchmarkBotEvaluator, TournamentEvaluator
from tft_ai_player.rl.league.league_manager import LeagueManager
from tft_ai_player.rl.logger import WandBLogger, check_collapse_warnings
from tft_ai_player.rl.models.networks import TFTActorCritic
from tft_ai_player.rl.planner import ShopBeamSearchPlanner
from tft_ai_player.simulation.actions import TOTAL_DISCRETE_ACTIONS
from tft_ai_player.simulation.combat import CombatResolver, HeuristicCombatResolver, MLCombatResolver
from tft_ai_player.simulation.config import SetData
from tft_ai_player.simulation.gym_env import TFTEnv, TFTStateEncoder
from tft_ai_player.simulation.sets.set18 import get_set18_data

logger = logging.getLogger(__name__)


class LeagueTrainer:
    """Orchestrates PPO training, AlphaStar PFSP multi-agent league,

    health telemetry, collapse protection, and deterministic bot evaluation.
    """

    def __init__(
        self,
        set_data: SetData | None = None,
        model: TFTActorCritic | None = None,
        trunk_checkpoint: str | Path | None = None,
        world_model_checkpoint: str | Path | None = None,
        z_index_path: str | Path | None = None,
        round_winner_model_path: str | Path | None = None,
        lr: float = 2.5e-4,
        total_rollout_steps: int = 4096,
        batch_size: int = 512,
        num_epochs: int = 4,
        entropy_coef_start: float = 0.01,
        entropy_coef_min: float = 0.002,
        entropy_decay_rate: float = 0.999,
        max_generations: int = 2000,
        eval_interval: int = 25,
        snapshot_interval: int = 50,
        device: str | None = None,
        checkpoint_dir: str | Path = "checkpoints/league",
        run_name: str | None = None,
        use_wandb: bool = True,
        wandb_project: str = "tft-ai-league",
        wandb_entity: str | None = None,
        wandb_group: str | None = None,
        resume: bool = False,
        resume_from: str | Path | None = None,
    ) -> None:
        self.set_data = set_data or get_set18_data()
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name or "ppo_alphastar_v1"

        self.lr = lr
        self.initial_lr = lr
        self.total_rollout_steps = total_rollout_steps
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.entropy_coef = entropy_coef_start
        self.entropy_coef_min = entropy_coef_min
        self.entropy_decay_rate = entropy_decay_rate
        self.max_generations = max_generations
        self.eval_interval = eval_interval
        self.snapshot_interval = snapshot_interval

        # 1. Load Frozen MultiModalFusionTrunk
        self.trunk = MultiModalFusionTrunk(
            num_champs=len(self.set_data.champions) + 20,
            num_items=len(self.set_data.items) + 20,
            num_traits=len(self.set_data.traits) + 20,
            fused_dim=384,
            num_layers=2,
        )
        if trunk_checkpoint and Path(trunk_checkpoint).exists():
            try:
                state_dict = torch.load(trunk_checkpoint, map_location="cpu", weights_only=False)
                if isinstance(state_dict, dict) and "trunk_state_dict" in state_dict:
                    trunk_state = state_dict["trunk_state_dict"]
                else:
                    trunk_state = {k.replace("trunk.", ""): v for k, v in state_dict.items() if "head" not in k}
                self.trunk.load_state_dict(trunk_state, strict=False)
                logger.info(f"Loaded frozen trunk from {trunk_checkpoint}")
            except Exception as e:
                logger.warning(f"Could not load trunk checkpoint: {e}")
        self.trunk.freeze()
        self.trunk.to(self.device)

        # 2. Load Frozen State Transition Predictor (World Model)
        self.world_model = StateTransitionPredictor(
            input_dim=self.trunk.fused_dim,
            hidden_dim=512,
            output_dim=self.trunk.fused_dim,
            num_layers=3,
        )
        if world_model_checkpoint and Path(world_model_checkpoint).exists():
            try:
                state_dict = torch.load(world_model_checkpoint, map_location="cpu", weights_only=False)
                self.world_model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded World Model from {world_model_checkpoint}")
            except Exception as e:
                logger.warning(f"Could not load World Model checkpoint: {e}")
        self.world_model.eval()
        for p in self.world_model.parameters():
            p.requires_grad = False
        self.world_model.to(self.device)

        # 3. Load Z-Index centroids
        self.z_index_path = z_index_path
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
                if self.z_centroids is not None:
                    print(f" [+] Successfully loaded {len(self.z_centroids)} Z-Index Composition Centroids from {z_index_path}")
            except Exception as e:
                logger.warning(f"Could not load Z-Index centroids: {e}")

        # 4. Calibrated Combat Resolver (Deep Learning on GPU by default for maximum throughput)
        self.combat_resolver: CombatResolver
        rw_path = Path(round_winner_model_path) if round_winner_model_path else None

        if rw_path and rw_path.exists() and str(rw_path).endswith(".joblib"):
            try:
                self.combat_resolver = MLCombatResolver(model_pipeline=rw_path)
                logger.info(f"Loaded LightGBM Combat Resolver from {rw_path}")
            except Exception as e:
                logger.warning(f"Could not load LightGBM Combat Resolver: {e}. Falling back to DL / heuristic.")
                self.combat_resolver = HeuristicCombatResolver()
        else:
            try:
                from tft_ai_player.round_winner.embedding_model import DeepSiameseCombatNet
                from tft_ai_player.embeddings.model import TrunkPretrainModel
                from tft_ai_player.simulation.combat import DLCombatResolver

                siamese_paths = [
                    rw_path if rw_path and "siamese" in str(rw_path).lower() else None,
                    Path("models/round_winner/deep_siamese_combat_best.pt"),
                    Path("D:/tft-winner-data/set18/models/round_winner/deep_siamese_combat_best.pt"),
                    Path("D:/tft-winner-data/set18/models/deep_siamese_combat_best.pt"),
                ]
                siamese_ckpt = next((p for p in siamese_paths if p and Path(p).exists()), None)

                if siamese_ckpt:
                    siamese_model = DeepSiameseCombatNet(trunk=self.trunk, freeze_trunk=True, hidden_dim=256).to(self.device)
                    s_data = torch.load(siamese_ckpt, map_location="cpu", weights_only=False)
                    s_dict = s_data.get("model_state_dict", s_data)
                    siamese_model.load_state_dict(s_dict, strict=False)
                    siamese_model.eval()
                    self.combat_resolver = DLCombatResolver(model=siamese_model, device=self.device)
                    logger.info(f"Loaded DeepSiameseCombatNet from {siamese_ckpt}")
                    print(f" [+] Combat Resolver: DeepSiameseCombatNet (GPU: {siamese_ckpt})")
                else:
                    dl_model = TrunkPretrainModel(
                        num_champs=len(self.set_data.champions) + 20,
                        num_items=len(self.set_data.items) + 20,
                        num_traits=len(self.set_data.traits) + 20,
                        embed_dim=128,
                        fused_dim=384,
                        num_layers=2,
                    ).to(self.device)

                    ckpt_to_load = trunk_checkpoint if trunk_checkpoint and Path(trunk_checkpoint).exists() else "models/trunk/trunk_best.pt"
                    if Path(ckpt_to_load).exists():
                        ckpt = torch.load(ckpt_to_load, map_location="cpu", weights_only=False)
                        s_dict = ckpt.get("model_state_dict", ckpt)
                        dl_model.load_state_dict(s_dict, strict=False)
                    dl_model.eval()
                    self.combat_resolver = DLCombatResolver(model=dl_model, device=self.device)
                    print(f" [+] Combat Resolver: Deep Learning GPU (Trunk: {ckpt_to_load})")
            except Exception as e:
                logger.warning(f"Could not load Deep Learning Combat Resolver: {e}. Falling back to heuristic.")
                self.combat_resolver = HeuristicCombatResolver()
                print(f" [+] Combat Resolver: Heuristic (Fallback)")

        # 5. Unified PPO Actor-Critic Model (Obs Dim = fused_dim + 384)
        obs_dim = self.trunk.fused_dim + 384
        if model is None:
            self.model = TFTActorCritic(obs_dim=obs_dim, action_dim=TOTAL_DISCRETE_ACTIONS, hidden_dim=512)
        else:
            self.model = model
        self.model.to(self.device)

        # 6. Maskable PPO & Rollout Buffer
        self.ppo = MaskablePPO(
            model=self.model,
            lr=self.lr,
            clip_ratio=0.2,
            entropy_coef=self.entropy_coef,
            value_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.05,
            device=self.device,
        )
        self.buffer = RolloutBuffer(
            buffer_size=self.total_rollout_steps,
            obs_dim=obs_dim,
            action_dim=TOTAL_DISCRETE_ACTIONS,
            gamma=0.99,
            gae_lambda=0.95,
            device=self.device,
        )

        # 7. League Manager with registered neural policy (used for opponents with random z_k)
        self.league = LeagueManager(
            checkpoint_dir=self.checkpoint_dir,
            z_index_path=self.z_index_path,
            set_data=self.set_data,
            device=self.device,
        )
        self.league.set_neural_agents(
            main_model=self.model,
            trunk=self.trunk,
            buffer=self.buffer,
        )

        # Check for checkpoint resume
        self.start_generation = 1
        if resume_from:
            try:
                gen_resumed, ent_resumed = self.league.checkpoint_manager.load_training_state(
                    ckpt_dir=resume_from,
                    model=self.model,
                    optimizer=self.ppo.optimizer,
                    main_exploiter_model=getattr(self, "exploiter_model", None),
                    main_exploiter_optimizer=getattr(self, "exploiter_ppo", None).optimizer if hasattr(self, "exploiter_ppo") else None,
                    league_exploiter_model=getattr(self, "league_exploiter_model", None),
                    league_exploiter_optimizer=getattr(self, "league_exploiter_ppo", None).optimizer if hasattr(self, "league_exploiter_ppo") else None,
                )
                self.entropy_coef = ent_resumed
                self.start_generation = gen_resumed + 1
                print(f" [+] Resumed training state from {resume_from} at Generation {gen_resumed} (continuing to Gen {self.max_generations})")
            except Exception as e:
                logger.warning(f"Could not resume from {resume_from}: {e}")
        elif resume:
            latest_ckpt = self.league.checkpoint_manager.find_latest_training_state()
            if latest_ckpt:
                try:
                    gen_resumed, ent_resumed = self.league.checkpoint_manager.load_training_state(
                        ckpt_dir=latest_ckpt,
                        model=self.model,
                        optimizer=self.ppo.optimizer,
                        main_exploiter_model=getattr(self, "exploiter_model", None),
                        main_exploiter_optimizer=getattr(self, "exploiter_ppo", None).optimizer if hasattr(self, "exploiter_ppo") else None,
                        league_exploiter_model=getattr(self, "league_exploiter_model", None),
                        league_exploiter_optimizer=getattr(self, "league_exploiter_ppo", None).optimizer if hasattr(self, "league_exploiter_ppo") else None,
                    )
                    self.entropy_coef = ent_resumed
                    self.start_generation = gen_resumed + 1
                    print(f" [+] Resumed training state from {latest_ckpt} at Generation {gen_resumed} (continuing to Gen {self.max_generations})")
                except Exception as e:
                    logger.warning(f"Could not resume from {latest_ckpt}: {e}")

        # 8. Evaluators & Tree Search Planner
        self.planner = ShopBeamSearchPlanner(
            set_data=self.set_data,
            encoder=TFTStateEncoder(set_data=self.set_data, trunk=self.trunk, device=self.device),
            world_model=self.world_model,
            beam_width=8,
            device=self.device,
        )
        self.bench_evaluator = BenchmarkBotEvaluator(
            set_data=self.set_data,
            combat_resolver=self.combat_resolver,
            trunk=self.trunk,
            world_model=self.world_model,
        )
        self.tournament_evaluator = TournamentEvaluator(
            league=self.league,
            set_data=self.set_data,
            combat_resolver=self.combat_resolver,
            trunk=self.trunk,
        )
        self.current_progress = 0.0

        # 9. WandB Experiment Logger
        config_dict = {
            "obs_dim": 704,
            "action_dim": TOTAL_DISCRETE_ACTIONS,
            "initial_lr": self.lr,
            "rollout_steps": self.total_rollout_steps,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "clip_ratio": 0.2,
            "gae_lambda": 0.95,
            "gamma": 0.99,
            "entropy_coef_start": entropy_coef_start,
            "entropy_coef_min": entropy_coef_min,
            "device": str(self.device),
        }
        self.wandb_logger = WandBLogger(
            project=wandb_project,
            entity=wandb_entity,
            group=wandb_group or self.run_name,
            run_name=self.run_name,
            config=config_dict,
            enabled=use_wandb,
        )

        # 10. Stability & Rollback Protection State
        self.consecutive_negative_exp_var = 0
        self.last_stable_state = copy.deepcopy(self.model.state_dict())
        self.last_stable_generation = 0

    def collect_rollouts(
        self,
        env: TFTEnv,
        target_steps: int,
        model: TFTActorCritic | None = None,
        buffer: RolloutBuffer | None = None,
    ) -> dict[str, float]:
        """Generate rollouts using specified policy in the Gymnasium environment."""
        active_model = model or self.model
        active_buffer = buffer or self.buffer

        active_buffer.reset()
        active_model.eval()

        obs, info = env.reset()
        mask = info["action_mask"]

        episode_rewards: list[float] = []
        episode_placements: list[int] = []
        macro_cosines: list[float] = []
        micro_cosines: list[float] = []
        episode_breakdowns: list[dict[str, float]] = []
        current_ep_breakdown: dict[str, float] = {
            "r_combat": 0.0,
            "r_interest": 0.0,
            "r_terminal": 0.0,
            "r_macro": 0.0,
            "r_micro": 0.0,
            "r_env": 0.0,
        }
        mask_violations = 0
        current_ep_reward = 0.0

        macro_sims: list[float] = []
        cluster_matches: list[float] = []
        planned_queue: list[int] = []
        need_shop_plan = True
        guided_actions_count = 0
        total_planning_actions = 0
        planner_agreements = 0

        while active_buffer.ptr < target_steps:
            focal_player = env.game.get_focal_player()
            rinfo = env.game.stage_manager.get_current_round_info()

            # Plan lookahead actions for current visible shop at round start or after reroll
            if need_shop_plan and focal_player.alive:
                planned_queue = self.planner.plan_shop_sequence(
                    player=focal_player,
                    pool=env.game.pool,
                    stage=rinfo.stage,
                    round_in_stage=rinfo.round_in_stage,
                    target_z=env.target_z,
                )
                need_shop_plan = False

            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)

            with torch.no_grad():
                action_t, _, _ = active_model.get_action(obs_t, mask_t)
                neural_action = int(action_t.item())

            # Decide whether to execute tree-search planned action or neural policy action
            guide_prob = max(0.2, 0.7 * (1.0 - getattr(self, "current_progress", 0.0)))
            selected_action = neural_action

            if planned_queue:
                rec_action = planned_queue.pop(0)
                if mask[rec_action]:
                    if rec_action == neural_action:
                        planner_agreements += 1
                    total_planning_actions += 1

                    if np.random.random() < guide_prob:
                        selected_action = rec_action
                        guided_actions_count += 1

            action = selected_action

            # Evaluate log_prob and value for the executed action
            with torch.no_grad():
                act_t = torch.tensor([action], dtype=torch.long, device=self.device)
                log_prob_t, _, val_t = active_model.evaluate_actions(obs_t, act_t, mask_t)
                log_prob = float(log_prob_t.item())
                value = float(val_t.item())

            # Verify action validity
            if not mask[action]:
                mask_violations += 1

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            next_mask = next_info["action_mask"]
            done = terminated or truncated

            # If reroll (action 6), round pass (action 0), or game over, trigger shop re-plan
            if action == 6 or action == 0 or done:
                planned_queue.clear()
                need_shop_plan = True

            active_buffer.add(
                obs=obs,
                action=action,
                action_mask=mask,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done,
            )

            current_ep_reward += reward
            if "reward_breakdown" in next_info and isinstance(next_info["reward_breakdown"], dict):
                rb = next_info["reward_breakdown"]
                if "cluster_match" in rb and rb["cluster_match"] is not None:
                    cluster_matches.append(float(rb["cluster_match"]))
                if "r_macro" in rb and rb["r_macro"] > 1e-6:
                    macro_sims.append(float(rb["r_macro"]))
                for k in current_ep_breakdown:
                    current_ep_breakdown[k] += rb.get(k, 0.0)

            if done:
                planned_queue.clear()
                episode_rewards.append(current_ep_reward)
                episode_breakdowns.append(dict(current_ep_breakdown))
                current_ep_breakdown = {k: 0.0 for k in current_ep_breakdown}
                if "placement" in next_info and next_info["placement"] is not None:
                    episode_placements.append(next_info["placement"])
                current_ep_reward = 0.0
                obs, info = env.reset()
                mask = info["action_mask"]
            else:
                obs = next_obs
                mask = next_mask

        # Compute GAE advantages
        with torch.no_grad():
            last_obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            last_val = float(active_model.get_value(last_obs_t).item())
        active_buffer.compute_returns_and_advantages(last_value=last_val, done=False)

        mean_rew = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        avg_place = float(np.mean(episode_placements)) if episode_placements else 4.5
        top4_rate = float(np.mean([p <= 4 for p in episode_placements])) if episode_placements else 0.5
        win_rate = float(np.mean([p == 1 for p in episode_placements])) if episode_placements else 0.125
        mean_breakdown = {
            k: float(np.mean([eb[k] for eb in episode_breakdowns])) if episode_breakdowns else 0.0
            for k in ["r_combat", "r_interest", "r_terminal", "r_macro", "r_micro", "r_env"]
        }
        mask_rejection_rate = float(mask_violations / max(1, active_buffer.ptr))

        # Action distribution statistics across the collected buffer
        valid_actions = active_buffer.actions[:active_buffer.ptr].cpu().numpy()
        tot_acts = max(1, len(valid_actions))

        pass_pct = float(np.mean(valid_actions == 0)) * 100.0
        buy_shop_pct = float(np.mean((valid_actions >= 1) & (valid_actions <= 5))) * 100.0
        reroll_pct = float(np.mean(valid_actions == 6)) * 100.0
        buy_xp_pct = float(np.mean(valid_actions == 7)) * 100.0
        sell_bench_pct = float(np.mean((valid_actions >= 8) & (valid_actions <= 16))) * 100.0
        deploy_board_pct = float(np.mean((valid_actions >= 17) & (valid_actions <= 44))) * 100.0
        equip_item_pct = float(np.mean((valid_actions >= 45) & (valid_actions <= 110))) * 100.0

        target_cluster_match_rate = float(np.mean(cluster_matches)) * 100.0 if cluster_matches else 0.0
        macro_alignment_cosine = float(np.mean(macro_sims)) if macro_sims else 0.0

        guided_pct = float(guided_actions_count / max(1, active_buffer.ptr)) * 100.0
        agreement_pct = float(planner_agreements / max(1, total_planning_actions)) * 100.0

        return {
            "mean_reward": mean_rew,
            "avg_placement": avg_place,
            "top4_rate": top4_rate,
            "win_rate": win_rate,
            "action_mask_rejection_rate": mask_rejection_rate,
            "reward_breakdown": mean_breakdown,
            "reward_combat": mean_breakdown["r_combat"],
            "reward_interest": mean_breakdown["r_interest"],
            "reward_terminal": mean_breakdown["r_terminal"],
            "reward_micro_alignment": mean_breakdown["r_micro"],
            "reward_macro_alignment": mean_breakdown["r_macro"],
            "reward_env_total": mean_breakdown["r_env"],
            "target_cluster_match_rate": target_cluster_match_rate,
            "macro_alignment_cosine": macro_alignment_cosine,
            "guided_actions_pct": guided_pct,
            "planner_agreement_pct": agreement_pct,
            "action_buy_xp_pct": buy_xp_pct,
            "action_pass_pct": pass_pct,
            "action_equip_item_pct": equip_item_pct,
            "action_buy_shop_pct": buy_shop_pct,
            "action_deploy_board_pct": deploy_board_pct,
            "action_reroll_pct": reroll_pct,
            "action_sell_bench_pct": sell_bench_pct,
        }

    def train_generation(self, generation: int) -> dict[str, Any]:
        """Execute one full generation of Unified Alpha League self-play training."""
        # 1. Dynamically adapt entropy coefficient and learning rate to total generations schedule
        progress = max(0.0, min(1.0, float(generation) / max(1, self.max_generations)))
        self.current_progress = progress

        # Dynamic Cosine Annealing from entropy_coef_start down to entropy_coef_min:
        current_entropy = self.entropy_coef_min + 0.5 * (self.entropy_coef - self.entropy_coef_min) * (1.0 + math.cos(math.pi * progress))
        self.ppo.set_entropy_coef(current_entropy)

        # Dynamic Linear learning rate decay adapted to max_generations
        current_lr = max(1e-5, self.initial_lr * (1.0 - progress))
        self.ppo.set_learning_rate(current_lr)

        # 2. Build rollout environment with full League Bot Factory (opponents are neural bots with random z_k)
        from tft_ai_player.simulation.gym_env import CurriculumBotFactory
        bot_factory = CurriculumBotFactory(
            generation=generation,
            league=self.league,
            set_data=self.set_data,
            device=self.device,
            active_model=self.model,
            focal_agent_id="main_agent",
        )
        env = TFTEnv(
            set_data=self.set_data,
            combat_resolver=self.combat_resolver,
            bot_factory=bot_factory,
            trunk=self.trunk,
            world_model=self.world_model,
            z_centroids=self.z_centroids,
            sample_random_z=False,  # Focal Agent (Seat 0) is ALWAYS 100% Flex (z=0)
            target_z=None,
            alpha=0.5,
            beta=0.3,
            device=self.device,
        )
        env.update_reward_weights(progress_fraction=progress, min_alpha=0.05, min_beta=0.05)

        # 3. Collect Rollouts
        if hasattr(self.league, "metrics_tracker") and self.league.metrics_tracker is not None:
            self.league.metrics_tracker.reset()

        rollout_metrics = self.collect_rollouts(
            env=env,
            target_steps=self.total_rollout_steps,
            model=self.model,
            buffer=self.buffer,
        )

        # Ingest multi-agent Z-Archetype alignment from sparring bots
        if hasattr(self.league, "metrics_tracker") and self.league.metrics_tracker is not None:
            avg_macro, avg_match = self.league.metrics_tracker.get_summary()
            rollout_metrics["macro_alignment_cosine"] = avg_macro
            rollout_metrics["target_cluster_match_rate"] = avg_match

        # 4. PPO Optimization Step
        opt_metrics = self.ppo.update(self.buffer, num_epochs=self.num_epochs, batch_size=self.batch_size)

        # 5. Merge metrics
        metrics: dict[str, Any] = {
            "generation": generation,
            "learning_rate": current_lr,
            "entropy_coef": current_entropy,
            "current_beta": env.current_beta,
            "current_alpha": env.current_alpha,
            **rollout_metrics,
            **opt_metrics,
        }

        # 6. Collapse & Telemetry Health Checks
        warnings = check_collapse_warnings(metrics)
        for w in warnings:
            logger.warning(f"[Gen {generation}] {w}")

        # Check negative explained variance rollback condition
        if metrics["explained_variance"] < 0.0:
            self.consecutive_negative_exp_var += 1
            if self.consecutive_negative_exp_var >= 5:
                print(f" [!] Policy instability detected! Rolling back weights to Gen {self.last_stable_generation}")
                self.model.load_state_dict(self.last_stable_state)
                self.initial_lr = max(1e-5, self.initial_lr * 0.5)
                self.ppo.set_learning_rate(self.initial_lr)
                self.consecutive_negative_exp_var = 0
        else:
            self.consecutive_negative_exp_var = 0
            if metrics["explained_variance"] > 0.4:
                self.last_stable_state = copy.deepcopy(self.model.state_dict())
                self.last_stable_generation = generation

        # 7. Deterministic Benchmark Bot Evaluation (every eval_interval generations)
        if generation > 0 and generation % self.eval_interval == 0:
            eval_metrics = self.bench_evaluator.evaluate_main_agent(self.model, num_matches=50)
            metrics.update(eval_metrics)
            print(
                f"  --> Benchmark Evaluation [Gen {generation}]: "
                f"Avg Place: {eval_metrics['eval_avg_placement']:.2f} | "
                f"Top-4: {eval_metrics['eval_top4_rate']*100:.1f}% | "
                f"Alpha WR: {eval_metrics['bot_alpha_win_rate']*100:.1f}% | "
                f"Beta WR: {eval_metrics['bot_beta_win_rate']*100:.1f}% | "
                f"Gamma WR: {eval_metrics['bot_gamma_win_rate']*100:.1f}%"
            )

        # 8. Historical Snapshot Archival (every snapshot_interval generations)
        if generation > 0 and generation % self.snapshot_interval == 0:
            snap_id = self.league.archive_snapshot("main_agent", generation, self.model.state_dict())
            print(f"  --> Archived Historical Snapshot: {snap_id}")
            try:
                from tft_ai_player.rl.visualization.strategy_landscape import generate_alphastar_progression_plot
                viz_results = generate_alphastar_progression_plot(
                    checkpoint_dir=self.checkpoint_dir,
                    output_dir="reports/visualizations",
                    formats=["png", "gif", "html"],
                    fps=12,
                    stride=max(1, generation // 70),
                )
                self.wandb_logger.log_strategy_progression(viz_results, step=generation)
            except Exception as e:
                logger.debug(f"Auto-refresh strategy progression plot: {e}")

        # 9. Log to WandB
        self.wandb_logger.log(metrics, step=generation)

        # 10. Save unified training checkpoint
        self.league.checkpoint_manager.save_training_state(
            model=self.model,
            optimizer=self.ppo.optimizer,
            generation=generation,
            entropy_coef=current_entropy,
            metrics_history=[metrics],
        )

        return metrics

    def run_training_loop(self) -> None:
        """Run full league training loop until max_generations."""
        print("\n" + "=" * 80)
        print(f" [+] Starting AlphaStar TFT RL Training Loop ({self.max_generations} Generations)")
        print(f"     Device: {self.device} | Initial LR: {self.lr} | Rollout Steps: {self.total_rollout_steps}")
        print("=" * 80 + "\n")

        for gen in range(self.start_generation, self.max_generations + 1):
            metrics = self.train_generation(gen)
            print(
                f"[Gen {gen:04d}/{self.max_generations}] "
                f"Reward: {metrics['mean_reward']:+0.3f} | "
                f"Avg Place: {metrics['avg_placement']:.2f} | "
                f"Top-4: {metrics['top4_rate']*100:4.1f}% | "
                f"PlanAgr: {metrics.get('planner_agreement_pct', 0.0):4.1f}% | "
                f"Guided: {metrics.get('guided_actions_pct', 0.0):4.1f}% | "
                f"ExpVar: {metrics['explained_variance']:.3f} | "
                f"Loss: {metrics['policy_loss']:+0.3f}"
            )

        self.wandb_logger.close()
        print("\n" + "=" * 80)
        print(" [+] RL Training loop completed successfully!")
        print("=" * 80 + "\n")
