"""Unit and integration tests for Composition Archetype Extraction & Latent Clustering (Z-Index)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tft_ai_player.cli import main
from tft_ai_player.embeddings.cluster import (
    CompositionClusterer,
    CuratedEndgameBoard,
    create_synthetic_endgame_dataset,
    extract_board_latents,
    load_curated_endgame_snapshots,
    load_z_index,
    profile_clusters,
    run_clustering_pipeline,
    save_z_index_artifacts,
)
from tft_ai_player.embeddings.model import MultiModalFusionTrunk
from tft_ai_player.embeddings.vocab import ChampionVocabulary, ItemVocabulary, TraitVocabulary


def test_create_synthetic_endgame_dataset() -> None:
    """Verify synthetic dataset generator produces valid curated endgame boards."""
    boards, vocab, item_vocab, trait_vocab = create_synthetic_endgame_dataset(num_samples=30, seed=123)

    assert len(boards) == 30
    assert len(vocab) > 10
    assert len(item_vocab) > 5

    for b in boards:
        assert isinstance(b, CuratedEndgameBoard)
        assert b.stage_tuple[0] >= 5
        assert 1 <= b.placement <= 4
        assert b.board_champ_ids.shape == (4, 7)
        assert b.board_star_levels.shape == (4, 7)
        assert b.board_item_ids.shape == (4, 7, 3)
        assert len(b.unit_names) >= 6
        assert len(b.unit_tiers) == len(b.unit_names)
        assert len(b.unit_items) == len(b.unit_names)


def test_load_curated_endgame_snapshots_filtering() -> None:
    """Verify dataset loader strictly filters to Top-4 placements and Stage >= 5."""
    records = [
        # Match 1, Player A: Early round (Stage 3-2), Top-4 finisher -> should be filtered out due to stage < 5
        {
            "match_id": "M1",
            "focal_player": "P1",
            "round_stage": "3-2",
            "focal_health": 80,
            "placement": 2,
            "input_state_json": json.dumps({"focal_board": [{"unit": "TFT18_Jinx", "tier": 2, "loc": "D7"}]}),
        },
        # Match 1, Player A: Late round (Stage 5-3), Top-4 finisher -> should be KEPT
        {
            "match_id": "M1",
            "focal_player": "P1",
            "round_stage": "5-3",
            "focal_health": 45,
            "placement": 2,
            "input_state_json": json.dumps({"focal_board": [{"unit": "TFT18_Jinx", "tier": 2, "loc": "D7", "items": ["TFT_Item_InfinityEdge"]}]}),
        },
        # Match 2, Player B: Late round (Stage 5-1), Bot-4 finisher (placement 7) -> should be filtered out due to placement > 4
        {
            "match_id": "M2",
            "focal_player": "P2",
            "round_stage": "5-1",
            "focal_health": 10,
            "placement": 7,
            "input_state_json": json.dumps({"focal_board": [{"unit": "TFT18_Karma", "tier": 1, "loc": "D1"}]}),
        },
    ]

    vocab = ChampionVocabulary()
    item_vocab = ItemVocabulary()
    trait_vocab = TraitVocabulary()

    boards, _, _, _ = load_curated_endgame_snapshots(
        data=records,
        vocab=vocab,
        item_vocab=item_vocab,
        trait_vocab=trait_vocab,
        min_stage=5,
        max_placement=4,
    )

    assert len(boards) == 1
    assert boards[0].match_id == "M1"
    assert boards[0].round_stage == "5-3"
    assert boards[0].placement == 2


def test_extract_board_latents() -> None:
    """Verify isolated 256D board feature extraction via frozen trunk."""
    boards, vocab, item_vocab, trait_vocab = create_synthetic_endgame_dataset(num_samples=16, seed=42)

    trunk = MultiModalFusionTrunk(
        num_champs=len(vocab),
        num_items=len(item_vocab),
        num_traits=len(trait_vocab),
        board_feat_dim=256,
    )

    latents, extracted_boards = extract_board_latents(trunk, boards, batch_size=8, device="cpu")

    assert isinstance(latents, torch.Tensor)
    assert latents.shape == (16, 256)
    assert not latents.requires_grad
    assert len(extracted_boards) == 16


def test_composition_clusterer_fit_and_metrics() -> None:
    """Verify K-Means fitting and evaluation metrics computation."""
    boards, vocab, item_vocab, trait_vocab = create_synthetic_endgame_dataset(num_samples=40, seed=42)

    trunk = MultiModalFusionTrunk(
        num_champs=len(vocab),
        num_items=len(item_vocab),
        num_traits=len(trait_vocab),
        board_feat_dim=256,
    )

    latents, _ = extract_board_latents(trunk, boards, device="cpu")

    clusterer = CompositionClusterer(n_clusters=4, random_state=42)
    clusterer.fit(latents, boards)

    assert clusterer.is_fitted
    assert clusterer.centroids is not None
    assert clusterer.centroids.shape == (4, 256)
    assert clusterer.labels_ is not None
    assert len(clusterer.labels_) == 40
    assert "silhouette_score" in clusterer.metrics_
    assert "davies_bouldin_index" in clusterer.metrics_
    assert "inertia" in clusterer.metrics_
    assert len(clusterer.profiles_) == 4


def test_score_similarity_and_predict() -> None:
    """Verify cosine similarity scoring against Z-Index centroids."""
    clusterer = CompositionClusterer(n_clusters=3, random_state=42)
    dummy_latents = torch.randn(30, 256)
    clusterer.fit(dummy_latents)

    # Test single vector (256,)
    test_vec = torch.randn(256)
    sims_single = clusterer.score_similarity(test_vec)
    assert sims_single.shape == (3,)
    assert torch.all(sims_single >= -1.01) and torch.all(sims_single <= 1.01)

    # Test batch vectors (5, 256)
    test_batch = torch.randn(5, 256)
    sims_batch = clusterer.score_similarity(test_batch)
    assert sims_batch.shape == (5, 3)

    # Test predict
    pred_idx = clusterer.predict(test_vec)
    assert 0 <= pred_idx.item() < 3

    batch_preds = clusterer.predict(test_batch)
    assert batch_preds.shape == (5,)
    assert torch.all(batch_preds >= 0) and torch.all(batch_preds < 3)


def test_profile_clusters_structure() -> None:
    """Verify archetype profiling produces rich interpretable summaries."""
    boards, vocab, item_vocab, trait_vocab = create_synthetic_endgame_dataset(num_samples=30, seed=42)
    dummy_latents = torch.randn(30, 256)

    clusterer = CompositionClusterer(n_clusters=3, random_state=42)
    clusterer.fit(dummy_latents, boards)

    profiles = clusterer.profiles_
    assert len(profiles) == 3

    for p in profiles:
        assert isinstance(p.name, str)
        assert p.size >= 0
        assert 0.0 <= p.percentage <= 100.0
        assert isinstance(p.top_units, list)
        assert isinstance(p.dominant_traits, list)
        assert isinstance(p.top_items, list)
        assert isinstance(p.representative_boards, list)


def test_save_and_load_z_index_artifacts(tmp_path: Path) -> None:
    """Verify saving and loading z_index.pt, cluster_profiles.json, and cluster_profiles.md."""
    boards, vocab, item_vocab, trait_vocab = create_synthetic_endgame_dataset(num_samples=25, seed=42)
    dummy_latents = torch.randn(25, 256)

    clusterer = CompositionClusterer(n_clusters=3, random_state=42)
    clusterer.fit(dummy_latents, boards)

    saved_paths = save_z_index_artifacts(
        clusterer=clusterer,
        output_dir=tmp_path,
        vocab=vocab,
        item_vocab=item_vocab,
        trait_vocab=trait_vocab,
    )

    assert saved_paths["z_index_pt"].exists()
    assert saved_paths["profiles_json"].exists()
    assert saved_paths["profiles_md"].exists()

    # Verify loaded checkpoint
    loaded = load_z_index(saved_paths["z_index_pt"])
    assert "z_index" in loaded
    assert loaded["z_index"].shape == (3, 256)
    assert loaded["k"] == 3
    assert loaded["feature_dim"] == 256
    assert "metrics" in loaded


def test_run_clustering_pipeline_end_to_end(tmp_path: Path) -> None:
    """Verify full end-to-end execution of run_clustering_pipeline."""
    output_dir = tmp_path / "clustering_out"

    results = run_clustering_pipeline(
        synthetic=True,
        n_clusters=4,
        batch_size=32,
        device="cpu",
        use_wandb=False,
        output_dir=output_dir,
    )

    assert "error" not in results
    assert "saved_paths" in results
    assert "plot_artifacts" in results
    assert results["saved_paths"]["z_index_pt"].exists()
    assert results["saved_paths"]["profiles_json"].exists()
    assert Path(results["plot_artifacts"]["pca_plot"]).exists()
    assert Path(results["plot_artifacts"]["population_plot"]).exists()


def test_cli_cluster_compositions(tmp_path: Path) -> None:
    """Verify CLI subcommand cluster-compositions runs successfully with synthetic flag."""
    out_dir = str(tmp_path / "cli_clustering")
    exit_code = main([
        "cluster-compositions",
        "--synthetic",
        "--n-clusters", "3",
        "--batch-size", "32",
        "--device", "cpu",
        "--no-wandb",
        "--output-dir", out_dir,
    ])

    assert exit_code == 0
    assert (tmp_path / "cli_clustering" / "z_index.pt").exists()
    assert (tmp_path / "cli_clustering" / "cluster_profiles.json").exists()
    assert (tmp_path / "cli_clustering" / "figures" / "pca_latent_clusters.png").exists()
