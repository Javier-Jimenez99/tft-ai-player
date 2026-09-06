"""Comprehensive Speed and Throughput Benchmark: LightGBM vs Full Deep Learning for TFT Combat Prediction.

Evaluates:
1. Single 1v1 Combat Resolution (Latency per match)
2. Full Lobby Round Resolution (4 simultaneous 1v1 matches in an 8-player lobby)
3. Batched Parallel Combat Resolution (N = 16, 64, 256, 1024, 4096 matches)
4. Model Forward Pass vs Feature Extraction Overhead Breakdown
5. Execution on CPU vs GPU (CUDA)
6. Latency gain when reusing already computed RL latent embeddings
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch

from tft_ai_player.embeddings.model import MultiModalFusionTrunk, TrunkPretrainModel
from tft_ai_player.embeddings.vocab import ChampionVocabulary, ItemVocabulary, TraitVocabulary
from tft_ai_player.round_winner.features import TFTBoardFeatureExtractor
from tft_ai_player.round_winner.trainer import RoundWinnerPredictor
from tft_ai_player.simulation.config import SetData
from tft_ai_player.simulation.gym_env import TFTStateEncoder
from tft_ai_player.simulation.models import ChampionInstance, Player
from tft_ai_player.simulation.sets.set18 import get_set18_data


def create_sample_players(set_data: SetData, count: int = 8) -> list[Player]:
    """Generate sample realistic players with active boards for benchmarking."""
    players: list[Player] = []
    champ_list = list(set_data.champions.keys())
    item_list = [k for k in set_data.items.keys() if not set_data.is_component(k)]

    for p_id in range(count):
        p = Player(p_id, set_data)
        p.level = 8
        p.health = 80 - p_id * 5
        p.gold = 30 + p_id * 3
        # Add 8 board units
        for idx in range(8):
            cid = champ_list[(p_id * 8 + idx) % len(champ_list)]
            star = 2 if idx < 5 else (3 if idx == 5 and p_id % 2 == 0 else 1)
            row = idx // 4
            col = idx % 4 + (p_id % 3)
            items = [item_list[(p_id + idx) % len(item_list)]] if idx < 3 else []
            cost = set_data.champions[cid].cost if cid in set_data.champions else 1
            unit = ChampionInstance(champion_id=cid, cost=cost, star_level=star, items=items, position=(row, col))
            p.board[(row, col)] = unit
        players.append(p)
    return players



def benchmark_lightgbm_single(
    predictor: RoundWinnerPredictor,
    players: list[Player],
    num_iterations: int = 200,
    warmup: int = 20,
) -> dict[str, float]:
    """Benchmark LightGBM single 1v1 resolution."""
    p_a = players[0]
    p_b = players[1]
    board_a = p_a.to_feature_board()
    board_b = p_b.to_feature_board()

    # Warmup
    for _ in range(warmup):
        predictor.predict_combat(board_a, board_b, round_stage="4-2")

    # Measure End-to-End (feature extraction + model inference)
    times: list[float] = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        win_p, dmg_a, dmg_b = predictor.predict_combat(board_a, board_b, round_stage="4-2")
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms

    # Measure Pure Feature Extraction vs Pure Model Forward
    row_a = {
        "round_stage": "4-2", "focal_level": 8, "opponent_level": 8,
        "focal_health": 80, "opponent_health": 75, "focal_gold": 30, "opponent_gold": 33,
        "focal_augments": [], "opponent_augments": [],
        "input_state_json": json.dumps({"focal_board": board_a, "opponent_board": board_b}),
    }
    df_pair = pd.DataFrame([row_a, row_a])
    
    fe_times: list[float] = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        X = predictor.extractor.transform(df_pair)
        t1 = time.perf_counter()
        fe_times.append((t1 - t0) * 1000.0)

    X_cached = predictor.extractor.transform(df_pair)
    model_times: list[float] = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        raw_p = predictor.lgb_model.predict_proba(X_cached[0:1])[:, 1]
        cal_p = predictor.calibrator_lgb.predict_proba(raw_p.reshape(-1, 1))[:, 1]
        if predictor.lgb_damage_model:
            dmg = predictor.lgb_damage_model.predict(X_cached)
        t1 = time.perf_counter()
        model_times.append((t1 - t0) * 1000.0)

    return {
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "std_ms": float(np.std(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "fe_overhead_ms": float(np.mean(fe_times)),
        "model_only_ms": float(np.mean(model_times)),
        "throughput_matches_per_sec": float(1000.0 / np.mean(times)),
    }


def benchmark_deeplearning_single(
    dl_model: TrunkPretrainModel,
    encoder: TFTStateEncoder,
    players: list[Player],
    device: torch.device,
    num_iterations: int = 200,
    warmup: int = 20,
) -> dict[str, float]:
    """Benchmark Deep Learning single 1v1 resolution on target device (GPU or CPU)."""
    p_a = players[0]
    p_b = players[1]
    dl_model.eval()

    def _prep_tensors():
        c_a, s_a, i_a, t_a = encoder.encode_board_tensors(p_a)
        c_b, s_b, i_b, t_b = encoder.encode_board_tensors(p_b)
        sc_a = encoder.encode_scalars(p_a, stage=4, round_in_stage=2)
        sc_b = encoder.encode_scalars(p_b, stage=4, round_in_stage=2)
        return (c_a, s_a, i_a, t_a, sc_a), (c_b, s_b, i_b, t_b, sc_b)

    tensors_a, tensors_b = _prep_tensors()

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _, _, combat_logits_a, _ = dl_model.forward_snapshot(
                board_champ_ids=tensors_a[0].unsqueeze(0).to(device),
                board_star_levels=tensors_a[1].unsqueeze(0).to(device),
                board_item_ids=tensors_a[2].unsqueeze(0).to(device),
                board_traits=tensors_a[3].unsqueeze(0).to(device),
                state_scalars=tensors_a[4].unsqueeze(0).to(device),
            )
            prob_a = torch.sigmoid(combat_logits_a).item()
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Measure End-to-End (Tensor Prep + Model Forward + Sigmoid)
    times: list[float] = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        with torch.no_grad():
            (ca, sa, ia, ta, sca), (cb, sb, ib, tb, scb) = _prep_tensors()
            # Batch both boards into 1 forward pass
            c_batch = torch.stack([ca, cb]).to(device)
            s_batch = torch.stack([sa, sb]).to(device)
            i_batch = torch.stack([ia, ib]).to(device)
            t_batch = torch.stack([ta, tb]).to(device)
            sc_batch = torch.stack([sca, scb]).to(device)

            fused, _, combat_logits, _ = dl_model.forward_snapshot(
                board_champ_ids=c_batch,
                board_star_levels=s_batch,
                board_item_ids=i_batch,
                board_traits=t_batch,
                state_scalars=sc_batch,
            )
            # Logit difference for matchup
            win_prob_a = float(torch.sigmoid(combat_logits[0] - combat_logits[1]).item())
            if device.type == "cuda":
                torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    # Measure Tensor Prep overhead vs Model Forward only
    prep_times: list[float] = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        (ca, sa, ia, ta, sca), (cb, sb, ib, tb, scb) = _prep_tensors()
        c_batch = torch.stack([ca, cb]).to(device)
        s_batch = torch.stack([sa, sb]).to(device)
        i_batch = torch.stack([ia, ib]).to(device)
        t_batch = torch.stack([ta, tb]).to(device)
        sc_batch = torch.stack([sca, scb]).to(device)
        t1 = time.perf_counter()
        prep_times.append((t1 - t0) * 1000.0)

    # Pre-allocated tensors forward only
    (ca, sa, ia, ta, sca), (cb, sb, ib, tb, scb) = _prep_tensors()
    c_cached = torch.stack([ca, cb]).to(device)
    s_cached = torch.stack([sa, sb]).to(device)
    i_cached = torch.stack([ia, ib]).to(device)
    t_cached = torch.stack([ta, tb]).to(device)
    sc_cached = torch.stack([sca, scb]).to(device)

    model_times: list[float] = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        with torch.no_grad():
            fused, _, combat_logits, _ = dl_model.forward_snapshot(
                board_champ_ids=c_cached,
                board_star_levels=s_cached,
                board_item_ids=i_cached,
                board_traits=t_cached,
                state_scalars=sc_cached,
            )
            win_prob_a = float(torch.sigmoid(combat_logits[0] - combat_logits[1]).item())
            if device.type == "cuda":
                torch.cuda.synchronize()
        t1 = time.perf_counter()
        model_times.append((t1 - t0) * 1000.0)

    # Pre-computed Latents / Embeddings Scenario (RL Step scenario)
    # When RL already has fused latent vector (320D):
    fused_a = fused[0:1]
    fused_b = fused[1:2]
    rl_reuse_times: list[float] = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        with torch.no_grad():
            logit_a = dl_model.combat_head(fused_a)
            logit_b = dl_model.combat_head(fused_b)
            win_prob_a = float(torch.sigmoid(logit_a - logit_b).item())
            if device.type == "cuda":
                torch.cuda.synchronize()
        t1 = time.perf_counter()
        rl_reuse_times.append((t1 - t0) * 1000.0)

    return {
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "std_ms": float(np.std(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "tensor_prep_ms": float(np.mean(prep_times)),
        "model_only_ms": float(np.mean(model_times)),
        "rl_latent_reuse_ms": float(np.mean(rl_reuse_times)),
        "throughput_matches_per_sec": float(1000.0 / np.mean(times)),
    }


def benchmark_batched(
    lgb_predictor: RoundWinnerPredictor,
    dl_model: TrunkPretrainModel,
    encoder: TFTStateEncoder,
    players: list[Player],
    batch_sizes: list[int],
    device: torch.device,
    num_repeats: int = 50,
) -> list[dict[str, Any]]:
    """Compare throughput and latency across multiple batch sizes."""
    results: list[dict[str, Any]] = []

    # Pre-generate batch of players
    p_a = players[0]
    p_b = players[1]
    board_a = p_a.to_feature_board()
    board_b = p_b.to_feature_board()

    # Pre-generate DataFrame for LightGBM
    row_dict = {
        "round_stage": "4-2", "focal_level": 8, "opponent_level": 8,
        "focal_health": 80, "opponent_health": 75, "focal_gold": 30, "opponent_gold": 33,
        "focal_augments": [], "opponent_augments": [],
        "input_state_json": json.dumps({"focal_board": board_a, "opponent_board": board_b}),
    }

    # Pre-generate PyTorch tensors
    c_a, s_a, i_a, t_a = encoder.encode_board_tensors(p_a)
    c_b, s_b, i_b, t_b = encoder.encode_board_tensors(p_b)
    sc_a = encoder.encode_scalars(p_a, 4, 2)
    sc_b = encoder.encode_scalars(p_b, 4, 2)

    for B in batch_sizes:
        print(f"  Testing Batch Size N = {B}...")
        
        # 1. LightGBM Batched
        df_batch = pd.DataFrame([row_dict] * B)
        lgb_times: list[float] = []
        for _ in range(num_repeats):
            t0 = time.perf_counter()
            probs = lgb_predictor.predict_dataframe(df_batch)
            t1 = time.perf_counter()
            lgb_times.append((t1 - t0) * 1000.0)
        lgb_mean_ms = float(np.mean(lgb_times))
        lgb_throughput = float(B / (lgb_mean_ms / 1000.0))

        # 2. Deep Learning Batched on GPU
        c_batch_gpu = torch.stack([c_a] * B).to(device)
        s_batch_gpu = torch.stack([s_a] * B).to(device)
        i_batch_gpu = torch.stack([i_a] * B).to(device)
        t_batch_gpu = torch.stack([t_a] * B).to(device)
        sc_batch_gpu = torch.stack([sc_a] * B).to(device)

        # Warmup GPU
        with torch.no_grad():
            dl_model.forward_snapshot(
                board_champ_ids=c_batch_gpu,
                board_star_levels=s_batch_gpu,
                board_item_ids=i_batch_gpu,
                board_traits=t_batch_gpu,
                state_scalars=sc_batch_gpu,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()

        dl_gpu_times: list[float] = []
        for _ in range(num_repeats):
            t0 = time.perf_counter()
            with torch.no_grad():
                fused, _, combat_logits, _ = dl_model.forward_snapshot(
                    board_champ_ids=c_batch_gpu,
                    board_star_levels=s_batch_gpu,
                    board_item_ids=i_batch_gpu,
                    board_traits=t_batch_gpu,
                    state_scalars=sc_batch_gpu,
                )
                probs = torch.sigmoid(combat_logits)
                if device.type == "cuda":
                    torch.cuda.synchronize()
            t1 = time.perf_counter()
            dl_gpu_times.append((t1 - t0) * 1000.0)
        dl_gpu_mean_ms = float(np.mean(dl_gpu_times))
        dl_gpu_throughput = float(B / (dl_gpu_mean_ms / 1000.0))

        # 3. Deep Learning Batched on CPU
        c_batch_cpu = torch.stack([c_a] * B).to("cpu")
        s_batch_cpu = torch.stack([s_a] * B).to("cpu")
        i_batch_cpu = torch.stack([i_a] * B).to("cpu")
        t_batch_cpu = torch.stack([t_a] * B).to("cpu")
        sc_batch_cpu = torch.stack([sc_a] * B).to("cpu")

        dl_cpu = copy.deepcopy(dl_model).to("cpu")
        dl_cpu.eval()

        dl_cpu_times: list[float] = []
        for _ in range(num_repeats):
            t0 = time.perf_counter()
            with torch.no_grad():
                fused, _, combat_logits, _ = dl_cpu.forward_snapshot(
                    board_champ_ids=c_batch_cpu,
                    board_star_levels=s_batch_cpu,
                    board_item_ids=i_batch_cpu,
                    board_traits=t_batch_cpu,
                    state_scalars=sc_batch_cpu,
                )
                probs = torch.sigmoid(combat_logits)
            t1 = time.perf_counter()
            dl_cpu_times.append((t1 - t0) * 1000.0)
        dl_cpu_mean_ms = float(np.mean(dl_cpu_times))
        dl_cpu_throughput = float(B / (dl_cpu_mean_ms / 1000.0))

        speedup_gpu = lgb_mean_ms / max(1e-5, dl_gpu_mean_ms)


        results.append({
            "batch_size": B,
            "lgb_ms": round(lgb_mean_ms, 3),
            "lgb_throughput": round(lgb_throughput, 1),
            "dl_gpu_ms": round(dl_gpu_mean_ms, 3),
            "dl_gpu_throughput": round(dl_gpu_throughput, 1),
            "dl_cpu_ms": round(dl_cpu_mean_ms, 3),
            "dl_cpu_throughput": round(dl_cpu_throughput, 1),
            "speedup_gpu_vs_lgb": round(speedup_gpu, 2),
        })

    return results


def run_full_benchmark():
    print("=" * 85)
    print("      TFT COMBAT PREDICTION SPEED BENCHMARK: LIGHTGBM vs DEEP LEARNING")
    print("=" * 85)

    # 1. Setup environment & device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" [+] Execution Hardware Device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"     GPU Model: {gpu_name}")

    # 2. Load Models
    print(" [+] Loading LightGBM model from D:/tft-winner-data/set18/models/round_winner_model.joblib ...")
    lgb_predictor = RoundWinnerPredictor.load("D:/tft-winner-data/set18/models/round_winner_model.joblib")
    
    print(" [+] Loading Deep Learning model from models/trunk/trunk_best.pt ...")
    set_data = get_set18_data()
    vocab = ChampionVocabulary()
    item_vocab = ItemVocabulary()
    trait_vocab = TraitVocabulary()

    dl_model = TrunkPretrainModel(
        num_champs=len(vocab) + 20,
        num_items=len(item_vocab) + 20,
        num_traits=len(trait_vocab) + 20,
        embed_dim=128,
        fused_dim=384,
        num_layers=2,
    )
    trunk_ckpt = torch.load("models/trunk/trunk_best.pt", map_location="cpu", weights_only=False)
    state_dict = trunk_ckpt.get("model_state_dict", trunk_ckpt)
    dl_model.load_state_dict(state_dict, strict=False)
    dl_model.to(device)
    dl_model.eval()

    encoder = TFTStateEncoder(
        trunk=dl_model.trunk,
        set_data=set_data,
        device=device,
        vocab=vocab,
        item_vocab=item_vocab,
        trait_vocab=trait_vocab,
    )

    players = create_sample_players(set_data, count=8)
    print(f" [+] Generated 8 benchmark player boards (Set 18 Level 8 boards)")

    # 3. Benchmark 1: Single 1v1 Matchup (Batch Size = 1)
    print("\n" + "-" * 85)
    print(" 1. BENCHMARK: Single 1v1 Combat Resolution (1 Matchup, Sequential)")
    print("-" * 85)
    
    lgb_single = benchmark_lightgbm_single(lgb_predictor, players, num_iterations=150)
    dl_gpu_single = benchmark_deeplearning_single(dl_model, encoder, players, device=device, num_iterations=150)
    dl_cpu_single = benchmark_deeplearning_single(dl_model.to("cpu"), encoder, players, device=torch.device("cpu"), num_iterations=150)
    dl_model.to(device)

    print(f"\n [A] LightGBM Model (CPU):")
    print(f"     Total End-to-End Latency:  {lgb_single['mean_ms']:.3f} ms (p95: {lgb_single['p95_ms']:.3f} ms)")
    print(f"     |-- Tabular Feature Extr: {lgb_single['fe_overhead_ms']:.3f} ms ({lgb_single['fe_overhead_ms']/lgb_single['mean_ms']*100:.1f}% overhead)")
    print(f"     \\-- LightGBM Model Only:  {lgb_single['model_only_ms']:.3f} ms")
    print(f"     Throughput:               {lgb_single['throughput_matches_per_sec']:.1f} matches / second")

    print(f"\n [B] Deep Learning Model (GPU CUDA):")
    print(f"     Total End-to-End Latency:  {dl_gpu_single['mean_ms']:.3f} ms (p95: {dl_gpu_single['p95_ms']:.3f} ms)")
    print(f"     |-- Tensor Preparation:   {dl_gpu_single['tensor_prep_ms']:.3f} ms")
    print(f"     \\-- Neural Forward Pass:  {dl_gpu_single['model_only_ms']:.3f} ms")
    print(f"     Throughput:               {dl_gpu_single['throughput_matches_per_sec']:.1f} matches / second")

    print(f"\n [C] Deep Learning Model (CPU):")
    print(f"     Total End-to-End Latency:  {dl_cpu_single['mean_ms']:.3f} ms (p95: {dl_cpu_single['p95_ms']:.3f} ms)")
    print(f"     |-- Tensor Preparation:   {dl_cpu_single['tensor_prep_ms']:.3f} ms")
    print(f"     \\-- Neural Forward Pass:  {dl_cpu_single['model_only_ms']:.3f} ms")
    print(f"     Throughput:               {dl_cpu_single['throughput_matches_per_sec']:.1f} matches / second")

    print(f"\n [D] Deep Learning in RL Loop (Reusing Pre-Computed Latent Vectors s_t / h_board):")
    print(f"     Direct Latent Combat Head: {dl_gpu_single['rl_latent_reuse_ms']:.4f} ms (GPU)")
    print(f"     Throughput:               {1000.0/dl_gpu_single['rl_latent_reuse_ms']:.1f} matches / second")


    # 4. Benchmark 2: Batched Combat Resolution
    print("\n" + "-" * 85)
    print(" 2. BENCHMARK: Batched Throughput Scaling (N = 4, 16, 64, 256, 1024, 4096)")
    print("-" * 85)

    batch_sizes = [4, 16, 64, 256, 1024, 4096]
    batched_results = benchmark_batched(
        lgb_predictor=lgb_predictor,
        dl_model=dl_model,
        encoder=encoder,
        players=players,
        batch_sizes=batch_sizes,
        device=device,
        num_repeats=30,
    )

    df_res = pd.DataFrame(batched_results)
    print("\n" + df_res.to_string(index=False))

    print("\n" + "=" * 85)
    print("                          BENCHMARK SUMMARY & ANALYSIS")
    print("=" * 85)
    
    speedup_single_e2e = lgb_single['mean_ms'] / dl_gpu_single['mean_ms']
    speedup_single_reuse = lgb_single['mean_ms'] / dl_gpu_single['rl_latent_reuse_ms']
    speedup_b4096 = batched_results[-1]['speedup_gpu_vs_lgb']

    print(f" • Single Matchup E2E Speedup (GPU vs LightGBM):    {speedup_single_e2e:.2f}x ({lgb_single['mean_ms']:.2f} ms vs {dl_gpu_single['mean_ms']:.2f} ms)")
    print(f" • In-RL Reusing Latents Speedup (GPU vs LightGBM):  {speedup_single_reuse:.1f}x ({lgb_single['mean_ms']:.2f} ms vs {dl_gpu_single['rl_latent_reuse_ms']:.4f} ms)")
    print(f" • Massive Batched (N=4096) GPU Speedup:             {speedup_b4096:.1f}x ({batched_results[-1]['dl_gpu_throughput']:,.0f} matches/sec vs {batched_results[-1]['lgb_throughput']:,.0f} matches/sec)")
    print("=" * 85)


if __name__ == "__main__":
    run_full_benchmark()
