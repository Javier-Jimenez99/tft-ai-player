"""TFT Multi-Modal Fusion Trunk and Champ2Vec Embeddings Package."""

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
]
