"""TFT Multi-Modal Fusion Trunk and Champ2Vec Embeddings Package."""

from .cluster import (
    ArchetypeProfile,
    CompositionClusterer,
    CuratedEndgameBoard,
    create_synthetic_endgame_dataset,
    extract_board_latents,
    load_curated_endgame_snapshots,
    load_z_index,
    log_clustering_to_wandb,
    profile_clusters,
    run_clustering_pipeline,
    save_z_index_artifacts,
)
from .dataset import (
    SnapshotPairCollate,
    TFTPretrainDataset,
    create_synthetic_trajectory_dataset,
    is_pvp_round,
    parse_loc_to_row_col,
    parse_stage_string,
)
from .losses import InfoNCELoss, MultiTaskTrunkLoss
from .model import (
    BoardCNNEncoder,
    BoardHexTransformer,
    Champ2Vec,
    HexDistanceAttention,
    HexTransformerBlock,
    MultiModalFusionTrunk,
    StateMLP,
    TFTAxialBoardEncoder,
    TraitEncoder,
    TrunkPretrainModel,
    build_hex_geodesic_distance_matrix,
)
from .trainer import TrunkPreTrainer
from .transition import (
    ResidualMLPBlock,
    StateTransitionLoss,
    StateTransitionPredictor,
    TransitionPredictorTrainer,
    TransitionTrajectoryDataset,
    create_synthetic_transition_dataset,
    extract_latent_transition_pairs,
    is_valid_pvp_transition,
)
from .vocab import ChampionVocabulary, ItemVocabulary, TraitVocabulary

# Backward compatibility aliases
PermutationInvariantChamp2Vec = Champ2Vec

__all__ = [
    "ChampionVocabulary",
    "ItemVocabulary",
    "TraitVocabulary",
    "Champ2Vec",
    "PermutationInvariantChamp2Vec",
    "BoardHexTransformer",
    "HexDistanceAttention",
    "HexTransformerBlock",
    "BoardCNNEncoder",
    "TFTAxialBoardEncoder",
    "build_hex_geodesic_distance_matrix",
    "TraitEncoder",
    "StateMLP",
    "MultiModalFusionTrunk",
    "TrunkPretrainModel",
    "InfoNCELoss",
    "MultiTaskTrunkLoss",
    "TFTPretrainDataset",
    "SnapshotPairCollate",
    "TrunkPreTrainer",
    "create_synthetic_trajectory_dataset",
    "parse_loc_to_row_col",
    "parse_stage_string",
    "is_pvp_round",
    "CompositionClusterer",
    "CuratedEndgameBoard",
    "ArchetypeProfile",
    "load_curated_endgame_snapshots",
    "create_synthetic_endgame_dataset",
    "extract_board_latents",
    "profile_clusters",
    "log_clustering_to_wandb",
    "save_z_index_artifacts",
    "load_z_index",
    "run_clustering_pipeline",
    "ResidualMLPBlock",
    "StateTransitionPredictor",
    "StateTransitionLoss",
    "TransitionPredictorTrainer",
    "TransitionTrajectoryDataset",
    "create_synthetic_transition_dataset",
    "extract_latent_transition_pairs",
    "is_valid_pvp_transition",
]


