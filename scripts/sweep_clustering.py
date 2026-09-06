"""Automated Clustering Optimization Sweep for TFT Z-Index.

Sweeps K and Latent Transformations (Mean-Centering, Standardization, PCA Whitening),
logs all runs with 5-composition tables to WandB, and exports the winning model.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from tft_ai_player.embeddings.cluster import (
    CompositionClusterer,
    CuratedEndgameBoard,
    extract_board_latents,
    load_curated_endgame_snapshots,
    log_clustering_to_wandb,
    profile_clusters,
    save_z_index_artifacts,
)
from tft_ai_player.embeddings.model import MultiModalFusionTrunk
from tft_ai_player.embeddings.vocab import ChampionVocabulary, ItemVocabulary, TraitVocabulary

# Configure UTF-8 stdout
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def apply_latent_transformation(
    latents_np: np.ndarray,
    method: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Transform raw 256D latents to eliminate anisotropy and maximize angular spread."""
    params: dict[str, Any] = {"method": method}

    if method == "mean_centered_sphere":
        mean_vec = np.mean(latents_np, axis=0, keepdims=True)
        centered = latents_np - mean_vec
        norms = np.linalg.norm(centered, axis=1, keepdims=True) + 1e-8
        transformed = centered / norms
        params["mean_vector"] = mean_vec.squeeze(0)

    elif method == "standardized_sphere":
        scaler = StandardScaler()
        std_latents = scaler.fit_transform(latents_np)
        norms = np.linalg.norm(std_latents, axis=1, keepdims=True) + 1e-8
        transformed = std_latents / norms
        params["scaler_mean"] = scaler.mean_
        params["scaler_scale"] = scaler.scale_

    elif method == "pca_whitened_64d":
        pca = PCA(n_components=min(64, latents_np.shape[1]), whiten=True, random_state=42)
        whitened = pca.fit_transform(latents_np)
        norms = np.linalg.norm(whitened, axis=1, keepdims=True) + 1e-8
        transformed = whitened / norms
        params["pca_components"] = pca.components_
        params["pca_mean"] = pca.mean_

    elif method == "raw_sphere":
        norms = np.linalg.norm(latents_np, axis=1, keepdims=True) + 1e-8
        transformed = latents_np / norms

    else:
        transformed = latents_np

    return transformed, params


def evaluate_clustering(
    transformed_latents: np.ndarray,
    k: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Fit KMeans and compute comprehensive separation and quality metrics."""
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(transformed_latents)
    centroids = kmeans.cluster_centers_

    # Spherical normalization on centroids
    cent_tensor = torch.tensor(centroids, dtype=torch.float32)
    cent_norm = F.normalize(cent_tensor, p=2, dim=-1)
    sim_mat = torch.mm(cent_norm, cent_norm.t()).numpy()

    n_c = len(cent_norm)
    off_mask = ~np.eye(n_c, dtype=bool)
    off_sims = sim_mat[off_mask]

    sil = float(silhouette_score(transformed_latents, labels, metric="euclidean"))
    dbi = float(davies_bouldin_score(transformed_latents, labels))
    ch = float(calinski_harabasz_score(transformed_latents, labels))
    inertia = float(kmeans.inertia_)
    off_mean = float(np.mean(off_sims))
    off_max = float(np.max(off_sims))
    off_min = float(np.min(off_sims))

    # Composite quality score (Higher is better: High silhouette, low DBI, low off-diagonal max sim)
    score = sil - (0.3 * dbi) - (0.5 * max(0.0, off_max))

    metrics = {
        "silhouette_score": sil,
        "davies_bouldin_index": dbi,
        "calinski_harabasz_score": ch,
        "inertia": inertia,
        "off_diagonal_mean_cosine_sim": off_mean,
        "off_diagonal_max_cosine_sim": off_max,
        "off_diagonal_min_cosine_sim": off_min,
        "composite_score": score,
    }
    return labels, centroids, metrics


def run_sweep(
    data_dir: str = "D:/tft-winner-data/set18/players",
    trunk_path: str = "models/trunk/trunk_best.pt",
    wandb_project: str = "tft-clustering",
    use_wandb: bool = True,
    max_samples: int = 3000,
    output_dir: str = "models/clustering",
) -> dict[str, Any]:
    """Execute complete systematic sweep over clustering configurations."""
    print("=" * 80)
    print(" [*] TFT Latent Clustering Optimization Sweep & Evaluation")
    print(f"     Data Source:   {data_dir}")
    print(f"     Trunk Checkpoint: {trunk_path}")
    print(f"     Target Project:   {wandb_project}")
    print(f"     Max Samples:      {max_samples}")
    print("=" * 80)

    vocab = ChampionVocabulary()
    item_vocab = ItemVocabulary()
    trait_vocab = TraitVocabulary()

    # 1. Load curated dataset once
    boards, vocab, item_vocab, trait_vocab = load_curated_endgame_snapshots(
        data_dir=data_dir,
        vocab=vocab,
        item_vocab=item_vocab,
        trait_vocab=trait_vocab,
        min_stage=5,
        max_placement=4,
        max_samples=max_samples,
    )
    print(f" [+] Successfully curated {len(boards)} endgame winning boards.")

    # 2. Extract raw 256D latents once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" [+] Loading MultiModalFusionTrunk onto device: {device}")
    trunk = MultiModalFusionTrunk.load_trunk(trunk_path, map_location=device)
    raw_latents_tensor, curated_boards = extract_board_latents(trunk, boards, batch_size=64, device=device)
    raw_latents_np = raw_latents_tensor.detach().cpu().numpy()
    print(f" [+] Extracted raw latents: {raw_latents_np.shape}")

    # 3. Define sweep search space
    k_values = [8, 10, 12, 14, 15, 16, 18, 20]
    transformations = ["mean_centered_sphere", "standardized_sphere"]

    results_leaderboard: list[dict[str, Any]] = []
    best_config: dict[str, Any] | None = None
    best_clusterer: CompositionClusterer | None = None
    best_score = -float("inf")

    sweep_start = time.time()
    total_experiments = len(k_values) * len(transformations)
    exp_idx = 0

    for method in transformations:
        transformed, transform_params = apply_latent_transformation(raw_latents_np, method)

        for k in k_values:
            exp_idx += 1
            run_name = f"{method}_K{k:02d}"
            print(f"\n[Sweep {exp_idx:02d}/{total_experiments:02d}] Evaluating {run_name} ...")

            labels, centroids, metrics = evaluate_clustering(transformed, k=k, random_state=42)

            print(f"     - Silhouette: {metrics['silhouette_score']:.4f}")
            print(f"     - Davies-Bouldin: {metrics['davies_bouldin_index']:.4f}")
            print(f"     - Centroid Sim (Mean/Max): {metrics['off_diagonal_mean_cosine_sim']:.3f} / {metrics['off_diagonal_max_cosine_sim']:.3f}")
            print(f"     - Composite Score: {metrics['composite_score']:.4f}")

            # Instantiate clusterer
            clusterer = CompositionClusterer(
                n_clusters=k,
                normalize_latents=(method in ["mean_centered_sphere", "standardized_sphere", "raw_sphere"]),
                center_latents=(method == "mean_centered_sphere"),
            )
            clusterer.centroids = torch.tensor(centroids, dtype=torch.float32)
            clusterer.labels_ = labels
            clusterer.metrics_ = {
                "n_samples": int(len(boards)),
                "n_clusters": int(k),
                "feature_dim": int(transformed.shape[1]),
                "inertia": metrics["inertia"],
                "silhouette_score": metrics["silhouette_score"],
                "davies_bouldin_index": metrics["davies_bouldin_index"],
                "calinski_harabasz_score": metrics["calinski_harabasz_score"],
            }
            if method == "mean_centered_sphere" and "mean_vector" in transform_params:
                clusterer.mean_vector = torch.tensor(transform_params["mean_vector"], dtype=torch.float32)

            clusterer.is_fitted = True

            # Profile archetypes with 5 representative boards
            profiles = profile_clusters(
                clusterer,
                torch.tensor(raw_latents_np, dtype=torch.float32),
                curated_boards,
            )
            clusterer.profiles_ = profiles

            # Log to WandB
            if use_wandb:
                run_out_dir = Path("models/clustering_runs") / run_name
                log_clustering_to_wandb(
                    clusterer=clusterer,
                    latents=torch.tensor(transformed, dtype=torch.float32),
                    boards=curated_boards,
                    wandb_project=wandb_project,
                    wandb_group="clustering-optimization-sweep",
                    run_name=run_name,
                    output_dir=run_out_dir,
                    use_wandb=True,
                )

            # Record in leaderboard
            entry = {
                "method": method,
                "k": k,
                "run_name": run_name,
                "silhouette": round(metrics["silhouette_score"], 4),
                "davies_bouldin": round(metrics["davies_bouldin_index"], 4),
                "calinski_harabasz": round(metrics["calinski_harabasz_score"], 1),
                "off_diag_mean_sim": round(metrics["off_diagonal_mean_cosine_sim"], 3),
                "off_diag_max_sim": round(metrics["off_diagonal_max_cosine_sim"], 3),
                "composite_score": round(metrics["composite_score"], 4),
            }
            results_leaderboard.append(entry)

            if metrics["composite_score"] > best_score:
                best_score = metrics["composite_score"]
                best_config = entry
                best_clusterer = clusterer

    # 4. Print Summary Leaderboard Table
    print("\n" + "=" * 95)
    print(f" CLUSTERING SWEEP LEADERBOARD ({len(results_leaderboard)} runs completed in {time.time()-sweep_start:.1f}s)")
    print("=" * 95)
    print(f" {'Run Name':<30} | {'K':<3} | {'Silh':<7} | {'DBI':<7} | {'MeanSim':<8} | {'MaxSim':<8} | {'Score':<8}")
    print("-" * 95)
    sorted_board = sorted(results_leaderboard, key=lambda x: x["composite_score"], reverse=True)
    for row in sorted_board:
        print(
            f" {row['run_name']:<30} | {row['k']:<3} | {row['silhouette']:<7.4f} | {row['davies_bouldin']:<7.4f} | "
            f"{row['off_diag_mean_sim']:<8.3f} | {row['off_diag_max_sim']:<8.3f} | {row['composite_score']:<8.4f}"
        )
    print("=" * 95)

    if best_config and best_clusterer:
        print(f"\n [🏆] BEST CONFIGURATION: {best_config['run_name']}")
        print(f"     - K: {best_config['k']}")
        print(f"     - Method: {best_config['method']}")
        print(f"     - Silhouette: {best_config['silhouette']}")
        print(f"     - Davies-Bouldin: {best_config['davies_bouldin']}")
        print(f"     - Mean Centroid Sim: {best_config['off_diag_mean_sim']}")
        print(f"     - Max Centroid Sim:  {best_config['off_diag_max_sim']}")

        # Persist winning model artifacts
        print(f"\n [+] Persisting Best Z-Index Artifacts to: {output_dir}")
        saved_paths = save_z_index_artifacts(
            clusterer=best_clusterer,
            output_dir=output_dir,
            vocab=vocab,
            item_vocab=item_vocab,
            trait_vocab=trait_vocab,
        )
        print(f"     - Z-Index Checkpoint: {saved_paths['z_index_pt']}")
        print(f"     - Archetype Profiles: {saved_paths['profiles_json']}")
        print(f"     - Summary Document:   {saved_paths['profiles_md']}")

    return {
        "leaderboard": results_leaderboard,
        "best_config": best_config,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TFT Z-Index Clustering Optimization Sweep")
    parser.add_argument("--data-dir", type=str, default="D:/tft-winner-data/set18/players")
    parser.add_argument("--trunk-checkpoint", type=str, default="models/trunk/trunk_best.pt")
    parser.add_argument("--wandb-project", type=str, default="tft-clustering")
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--output-dir", type=str, default="models/clustering")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    run_sweep(
        data_dir=args.data_dir,
        trunk_path=args.trunk_checkpoint,
        wandb_project=args.wandb_project,
        use_wandb=not args.no_wandb,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
    )
