"""End-to-End RL System Profiler: Impact of Combat Model on Full Training Generation Loop.

Measures whether switching between LightGBM, Deep Learning (GPU), and Heuristic combat models
makes a meaningful difference in the overall RL training pipeline, profiling Amdahl's law bottlenecks.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch

from tft_ai_player.embeddings.model import MultiModalFusionTrunk, TrunkPretrainModel
from tft_ai_player.embeddings.vocab import ChampionVocabulary, ItemVocabulary, TraitVocabulary
from tft_ai_player.embeddings.transition import StateTransitionPredictor
from tft_ai_player.round_winner.trainer import RoundWinnerPredictor
from tft_ai_player.rl.algorithms.ppo import MaskablePPO, RolloutBuffer
from tft_ai_player.rl.models.networks import TFTActorCritic
from tft_ai_player.simulation.actions import TOTAL_DISCRETE_ACTIONS
from tft_ai_player.simulation.combat import CombatResolver, DLCombatResolver, HeuristicCombatResolver, MLCombatResolver
from tft_ai_player.simulation.config import SetData
from tft_ai_player.simulation.gym_env import TFTEnv
from tft_ai_player.simulation.sets.set18 import get_set18_data


def profile_rl_rollout_and_training(
    combat_resolver: CombatResolver,
    combat_name: str,
    trunk: MultiModalFusionTrunk,
    world_model: StateTransitionPredictor,
    set_data: SetData,
    device: torch.device,
    rollout_steps: int = 512,
    num_epochs: int = 4,
    batch_size: int = 128,
    num_generations: int = 3,
) -> dict[str, Any]:
    """Run full RL rollout collection + PPO update and measure precise subsystem timing breakdown."""
    obs_dim = trunk.fused_dim + 384
    model = TFTActorCritic(obs_dim=obs_dim, action_dim=TOTAL_DISCRETE_ACTIONS, hidden_dim=512).to(device)
    ppo = MaskablePPO(model=model, lr=2.5e-4, device=device)
    buffer = RolloutBuffer(buffer_size=rollout_steps, obs_dim=obs_dim, action_dim=TOTAL_DISCRETE_ACTIONS, device=device)

    env = TFTEnv(
        set_data=set_data,
        combat_resolver=combat_resolver,
        trunk=trunk,
        world_model=world_model,
        alpha=0.0,
        beta=0.3,
        device=device,
    )

    gen_total_times: list[float] = []
    rollout_times: list[float] = []
    ppo_update_times: list[float] = []

    # Fine-grained timers per step
    t_state_encoding: list[float] = []
    t_policy_inference: list[float] = []
    t_env_step: list[float] = []
    t_combat_only: list[float] = []
    combat_call_count = 0

    # Monkey patch env to measure combat time directly
    original_step_round = env.game.step_round
    round_combat_durations: list[float] = []

    def timed_step_round():
        t_c0 = time.perf_counter()
        res = original_step_round()
        t_c1 = time.perf_counter()
        round_combat_durations.append((t_c1 - t_c0) * 1000.0)  # ms
        return res

    env.game.step_round = timed_step_round

    print(f"\n  Profiling [{combat_name}] across {num_generations} generations ({rollout_steps} steps/gen)...")

    for gen in range(1, num_generations + 1):
        round_combat_durations.clear()
        buffer.reset()
        model.eval()

        t_gen_start = time.perf_counter()
        obs, info = env.reset()
        mask = info["action_mask"]

        t_rollout_start = time.perf_counter()

        for step in range(rollout_steps):
            # 1. Tensor packaging & Policy Inference
            t0 = time.perf_counter()
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
            with torch.no_grad():
                action_t, log_prob_t, val_t = model.get_action(obs_t, mask_t)
                action = int(action_t.item())
                log_prob = float(log_prob_t.item())
                value = float(val_t.item())
                if device.type == "cuda":
                    torch.cuda.synchronize()
            t1 = time.perf_counter()
            t_policy_inference.append((t1 - t0) * 1000.0)

            # 2. Environment Step (Includes action execution, state encoding, combat if pass)
            t2 = time.perf_counter()
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            next_mask = next_info["action_mask"]
            done = terminated or truncated
            t3 = time.perf_counter()
            t_env_step.append((t3 - t2) * 1000.0)

            buffer.add(obs, action, mask, reward, value, log_prob, done)

            if done:
                obs, info = env.reset()
                mask = info["action_mask"]
            else:
                obs = next_obs
                mask = next_mask

        # GAE
        with torch.no_grad():
            last_obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            last_val = float(model.get_value(last_obs_t).item())
        buffer.compute_returns_and_advantages(last_value=last_val, done=False)
        t_rollout_end = time.perf_counter()
        rollout_times.append(t_rollout_end - t_rollout_start)

        # 3. PPO Optimization Step
        t_ppo_start = time.perf_counter()
        opt_metrics = ppo.update(buffer, num_epochs=num_epochs, batch_size=batch_size)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_ppo_end = time.perf_counter()
        ppo_update_times.append(t_ppo_end - t_ppo_start)

        t_gen_end = time.perf_counter()
        gen_total_times.append(t_gen_end - t_gen_start)

        print(f"    Gen {gen}/{num_generations}: Total={gen_total_times[-1]:.3f}s | Rollout={rollout_times[-1]:.3f}s | PPO Update={ppo_update_times[-1]:.3f}s | Combat Total={sum(round_combat_durations)/1000.0:.3f}s ({len(round_combat_durations)} rounds)")

    total_gen_time_s = float(np.mean(gen_total_times))
    rollout_time_s = float(np.mean(rollout_times))
    ppo_update_time_s = float(np.mean(ppo_update_times))
    total_combat_time_s = float(sum(round_combat_durations) / (num_generations * 1000.0))
    combats_per_gen = len(round_combat_durations) / num_generations
    avg_combat_ms = float(np.mean(round_combat_durations)) if round_combat_durations else 0.0

    fps = float(rollout_steps / total_gen_time_s)

    return {
        "combat_name": combat_name,
        "total_gen_time_s": total_gen_time_s,
        "rollout_time_s": rollout_time_s,
        "ppo_update_time_s": ppo_update_time_s,
        "total_combat_time_s": total_combat_time_s,
        "combat_time_pct": (total_combat_time_s / total_gen_time_s) * 100.0,
        "avg_combat_ms": avg_combat_ms,
        "combats_per_gen": combats_per_gen,
        "avg_policy_inf_ms": float(np.mean(t_policy_inference)),
        "avg_env_step_ms": float(np.mean(t_env_step)),
        "fps_steps_per_sec": fps,
    }


def run_system_profile():
    print("=" * 90)
    print("      END-TO-END RL SYSTEM PROFILING: SYSTEM IMPACT OF COMBAT PREDICTOR")
    print("=" * 90)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" [+] Execution Hardware Device: {device}")
    if torch.cuda.is_available():
        print(f"     GPU: {torch.cuda.get_device_name(0)}")

    model_dir = Path("D:/tft-winner-data/set18/models/trunk")
    if not model_dir.exists():
        model_dir = Path("models/trunk")

    set_data = get_set18_data()
    vocab = ChampionVocabulary.load(model_dir / "vocab.json") if (model_dir / "vocab.json").exists() else ChampionVocabulary()
    item_vocab = ItemVocabulary.load(model_dir / "item_vocab.json") if (model_dir / "item_vocab.json").exists() else ItemVocabulary()
    trait_vocab = TraitVocabulary.load(model_dir / "trait_vocab.json") if (model_dir / "trait_vocab.json").exists() else TraitVocabulary()

    # 1. Load Frozen Pretrained Trunk
    ckpt_file = model_dir / "trunk_pretrained.pt" if (model_dir / "trunk_pretrained.pt").exists() else model_dir / "trunk_best.pt"
    checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    trunk = MultiModalFusionTrunk(
        num_champs=cfg.get("num_champs", len(vocab)),
        num_items=cfg.get("num_items", len(item_vocab)),
        num_traits=cfg.get("num_traits", len(trait_vocab)),
        champ_embed_dim=cfg.get("champ_embed_dim", 32),
        board_feat_dim=cfg.get("board_feat_dim", 256),
        state_feat_dim=cfg.get("state_feat_dim", 64),
        fused_dim=cfg.get("fused_dim", 320),
        num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.1),
    )
    trunk.load_state_dict(state_dict)
    trunk.freeze()
    trunk.to(device)

    # 2. Load World Model
    world_model = StateTransitionPredictor(input_dim=trunk.fused_dim, hidden_dim=512, output_dim=trunk.fused_dim, num_layers=3).to(device)
    world_model.eval()

    # 3. Setup Combat Resolvers
    # A: Heuristic Baseline
    heuristic_resolver = HeuristicCombatResolver()

    # B: LightGBM ML Resolver
    print(" [+] Loading LightGBM Resolver from D:/tft-winner-data/set18/models/round_winner_model.joblib...")
    lgb_resolver = MLCombatResolver(model_pipeline="D:/tft-winner-data/set18/models/round_winner_model.joblib")

    # C: DeepSiameseCombatNet (Trained GPU Siamese Model: 72.4% Test Acc)
    print(" [+] Loading DeepSiameseCombatNet from models/round_winner/deep_siamese_combat_best.pt...")
    from tft_ai_player.round_winner.embedding_model import DeepSiameseCombatNet
    siamese_model = DeepSiameseCombatNet(trunk=trunk, freeze_trunk=True, hidden_dim=256).to(device)
    siamese_data = torch.load("models/round_winner/deep_siamese_combat_best.pt", map_location="cpu", weights_only=False)
    siamese_model.load_state_dict(siamese_data["model_state_dict"], strict=False)
    siamese_model.eval()
    siamese_resolver = DLCombatResolver(model=siamese_model, device=device)

    # 4. Profile All Configurations inside the complete RL Loop
    steps_per_gen = 512

    prof_heuristic = profile_rl_rollout_and_training(
        combat_resolver=heuristic_resolver,
        combat_name="Pure Heuristic (Baseline)",
        trunk=trunk,
        world_model=world_model,
        set_data=set_data,
        device=device,
        rollout_steps=steps_per_gen,
        num_epochs=4,
        batch_size=128,
        num_generations=3,
    )

    prof_lgb = profile_rl_rollout_and_training(
        combat_resolver=lgb_resolver,
        combat_name="LightGBM ML (1,107 Features)",
        trunk=trunk,
        world_model=world_model,
        set_data=set_data,
        device=device,
        rollout_steps=steps_per_gen,
        num_epochs=4,
        batch_size=128,
        num_generations=3,
    )

    prof_siamese = profile_rl_rollout_and_training(
        combat_resolver=siamese_resolver,
        combat_name="DeepSiameseCombatNet (GPU: 72.4% Acc)",
        trunk=trunk,
        world_model=world_model,
        set_data=set_data,
        device=device,
        rollout_steps=steps_per_gen,
        num_epochs=4,
        batch_size=128,
        num_generations=3,
    )

    # 5. Display Full Comparative Results Table
    results = [prof_heuristic, prof_lgb, prof_siamese]

    df = pd.DataFrame(results)

    print("\n" + "=" * 90)
    print("                        END-TO-END RL PIPELINE BENCHMARK")
    print("=" * 90)
    display_cols = [
        "combat_name",
        "total_gen_time_s",
        "rollout_time_s",
        "ppo_update_time_s",
        "total_combat_time_s",
        "combat_time_pct",
        "fps_steps_per_sec",
    ]
    formatted_df = df[display_cols].copy()
    formatted_df.columns = [
        "Combat Model",
        "Gen Time (s)",
        "Rollout (s)",
        "PPO Train (s)",
        "Combat Time (s)",
        "Combat %",
        "Throughput (FPS)",
    ]
    print(formatted_df.to_string(index=False))

    print("\n" + "=" * 90)
    print("                             AMDAHL'S LAW ANALYSIS")
    print("=" * 90)
    t_lgb = prof_lgb["total_gen_time_s"]
    t_siamese = prof_siamese["total_gen_time_s"]
    t_heur = prof_heuristic["total_gen_time_s"]
    pct_diff = ((t_lgb - t_siamese) / t_lgb) * 100.0

    print(f" • Generation Time (LightGBM):            {t_lgb:.3f} s ({prof_lgb['fps_steps_per_sec']:.1f} steps/sec)")
    print(f" • Generation Time (DeepSiameseCombatNet): {t_siamese:.3f} s ({prof_siamese['fps_steps_per_sec']:.1f} steps/sec)")
    print(f" • Generation Time (Heuristic):           {t_heur:.3f} s ({prof_heuristic['fps_steps_per_sec']:.1f} steps/sec)")
    print(f" • End-to-End RL Speedup (vs LightGBM):    {t_lgb/t_siamese:.3f}x (Siamese saves {pct_diff:.1f}% total RL generation time)")
    print(f" • Combat Time %:                         {prof_lgb['combat_time_pct']:.1f}% with LightGBM vs {prof_siamese['combat_time_pct']:.1f}% with DeepSiamese")
    print("=" * 90)


if __name__ == "__main__":
    run_system_profile()
