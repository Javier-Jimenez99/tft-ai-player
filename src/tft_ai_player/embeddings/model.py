"""Robust and simplified Multi-Modal Fusion Trunk architecture for TFT state embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_hex_geodesic_distance_matrix() -> torch.Tensor:
    """Precompute 28x28 distance matrix for compatibility."""
    coords = []
    for r in range(4):
        for c in range(7):
            x = c - (r // 2) if r % 2 == 0 else c - ((r - 1) // 2)
            z = r
            y = -x - z
            coords.append((x, y, z))

    num_tiles = len(coords)
    dist_matrix = torch.zeros((num_tiles, num_tiles), dtype=torch.float32)
    for i in range(num_tiles):
        for j in range(num_tiles):
            x1, y1, z1 = coords[i]
            x2, y2, z2 = coords[j]
            dist_matrix[i, j] = (abs(x1 - x2) + abs(y1 - y2) + abs(z1 - z2)) / 2.0
    return torch.clamp(dist_matrix, 0.0, 8.0)


class Champ2Vec(nn.Module):
    """Permutation-Invariant Bag-of-Items Champion Token Embedding.

    Embeds champion ID (32D), star level (8D), and sum-pools equipped items (16D).
    Safely clamped to prevent CUDA device-side assertion failures.
    """

    def __init__(
        self,
        num_champs: int = 500,
        num_items: int = 300,
        champ_dim: int = 32,
        star_dim: int = 8,
        item_dim: int = 16,
        out_dim: int = 32,
    ) -> None:
        super().__init__()
        self.num_champs = max(num_champs, 500)
        self.num_items = max(num_items, 300)
        self.out_dim = out_dim

        self.champ_embed = nn.Embedding(self.num_champs + 1, champ_dim, padding_idx=0)
        self.star_embed = nn.Embedding(5, star_dim, padding_idx=0)
        self.item_embed = nn.Embedding(self.num_items + 1, item_dim, padding_idx=0)

        fusion_in = champ_dim + star_dim + item_dim  # 32 + 8 + 16 = 56
        self.net = nn.Sequential(
            nn.Linear(fusion_in, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        champ_ids: torch.Tensor,
        star_levels: torch.Tensor | None = None,
        item_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Strictly clamp IDs to prevent out-of-bounds indexing on CUDA
        safe_champ_ids = torch.clamp(champ_ids.long(), 0, self.num_champs)
        c_vec = self.champ_embed(safe_champ_ids)

        if star_levels is not None:
            safe_stars = torch.clamp(star_levels.long(), 0, 4)
            s_vec = self.star_embed(safe_stars)
        else:
            s_vec = torch.zeros(*champ_ids.shape, self.star_embed.embedding_dim, device=champ_ids.device)

        if item_ids is not None:
            safe_item_ids = torch.clamp(item_ids.long(), 0, self.num_items)
            raw_item_vecs = self.item_embed(safe_item_ids)
            i_sum = raw_item_vecs.sum(dim=-2)
        else:
            i_sum = torch.zeros(*champ_ids.shape, self.item_embed.embedding_dim, device=champ_ids.device)

        combined = torch.cat([c_vec, s_vec, i_sum], dim=-1)
        unit_vec = self.net(combined)

        # Zero-out empty hexes
        is_occupied = (safe_champ_ids > 0).unsqueeze(-1).float()
        return unit_vec * is_occupied


class TraitEncoder(nn.Module):
    """Encodes the active trait synergy density vector into a 64D representation."""

    def __init__(self, num_traits: int = 60, out_dim: int = 64) -> None:
        super().__init__()
        self.num_traits = max(num_traits, 60)
        self.net = nn.Sequential(
            nn.Linear(self.num_traits, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, trait_vec: torch.Tensor) -> torch.Tensor:
        # Align trait vector dimension safely
        if trait_vec.shape[-1] < self.num_traits:
            pad = torch.zeros(
                *trait_vec.shape[:-1],
                self.num_traits - trait_vec.shape[-1],
                device=trait_vec.device,
                dtype=trait_vec.dtype,
            )
            trait_vec = torch.cat([trait_vec, pad], dim=-1)
        elif trait_vec.shape[-1] > self.num_traits:
            trait_vec = trait_vec[..., : self.num_traits]
        return self.net(trait_vec)


class HexDistanceAttention(nn.Module):
    """Compatibility alias for self-attention with spatial projection."""

    def __init__(self, embed_dim: int = 128, num_heads: int = 4, **kwargs) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask)
        return out


class HexTransformerBlock(nn.Module):
    """Compatibility alias for transformer block."""

    def __init__(self, embed_dim: int = 128, num_heads: int = 4, **kwargs) -> None:
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.layer(x, src_key_padding_mask=key_padding_mask)


class BoardHexTransformer(nn.Module):
    """Clean, Stable Transformer Encoder for the 28-Hex TFT Board.

    - Uses standard, highly optimized PyTorch TransformerEncoder layers (zero custom index bugs).
    - Masked Mean Pooling over active units.
    - Fuses active trait synergy activations into a 256D board feature.
    """

    def __init__(
        self,
        token_in_dim: int = 32,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_traits: int = 60,
        out_dim: int = 256,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.unit_proj = nn.Linear(token_in_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 28, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        self.trait_encoder = TraitEncoder(num_traits=num_traits, out_dim=64)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + 64, out_dim),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
        )

    def forward(
        self,
        board_tokens: torch.Tensor,
        board_champ_ids: torch.Tensor | None = None,
        trait_vec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Reshape (Batch, 4, 7, 32) -> (Batch, 28, 32)
        if board_tokens.dim() == 4:
            batch_size = board_tokens.shape[0]
            board_tokens = board_tokens.view(batch_size, 28, -1)
        else:
            batch_size = board_tokens.shape[0]

        if board_champ_ids is not None:
            flat_ids = board_champ_ids.view(batch_size, 28)
            key_padding_mask = (flat_ids == 0)
            all_masked = key_padding_mask.all(dim=-1, keepdim=True)
            if all_masked.any():
                key_padding_mask = key_padding_mask & ~all_masked
            occupancy_mask = (flat_ids > 0).unsqueeze(-1).float()
        else:
            key_padding_mask = None
            occupancy_mask = torch.ones(batch_size, 28, 1, device=board_tokens.device)

        # 1. Project and add positional embeddings
        x = self.unit_proj(board_tokens) + self.pos_embed

        # 2. Standard PyTorch Transformer Encoder
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)


        # 3. Masked Mean Pooling over active units
        masked_tokens = x * occupancy_mask
        unit_counts = occupancy_mask.sum(dim=1).clamp(min=1.0)
        board_spatial = masked_tokens.sum(dim=1) / unit_counts  # (Batch, 128)

        # 4. Trait Synergy Fusion
        if trait_vec is not None:
            trait_feat = self.trait_encoder(trait_vec)
        else:
            trait_feat = torch.zeros(batch_size, 64, device=board_tokens.device)

        combined = torch.cat([board_spatial, trait_feat], dim=-1)
        return self.fusion(combined)


# Backward compatibility aliases
BoardCNNEncoder = BoardHexTransformer
TFTAxialBoardEncoder = BoardHexTransformer


class StateMLP(nn.Module):
    """Encodes the 8 player economy, match stage, unit count, and streak scalars into a 64D vector."""

    def __init__(self, in_features: int = 8, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, state_scalars: torch.Tensor) -> torch.Tensor:
        return self.net(state_scalars)


class MultiModalFusionTrunk(nn.Module):
    """Multi-Modal Fusion Trunk for TFT State Representation.

    Fuses:
    1. Synergy-Informed Hexagonal Board Embedding (256D)
    2. Player Economy & Match State MLP (64D)
    Total Latent State Representation = 320D.
    """

    def __init__(
        self,
        num_champs: int = 500,
        num_items: int = 300,
        num_traits: int = 60,
        champ_embed_dim: int = 32,
        board_feat_dim: int = 256,
        state_feat_dim: int = 64,
        fused_dim: int = 320,
        num_layers: int = 2,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_champs = max(num_champs, 500)
        self.num_items = max(num_items, 300)
        self.num_traits = max(num_traits, 60)
        self.champ_embed_dim = champ_embed_dim
        self.board_feat_dim = board_feat_dim
        self.state_feat_dim = state_feat_dim
        self.fused_dim = fused_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.champ2vec = Champ2Vec(
            num_champs=self.num_champs,
            num_items=self.num_items,
            out_dim=champ_embed_dim,
        )

        self.board_encoder = BoardHexTransformer(
            token_in_dim=champ_embed_dim,
            embed_dim=128,
            num_heads=4,
            num_layers=num_layers,
            dropout=dropout,
            num_traits=self.num_traits,
            out_dim=board_feat_dim,
        )

        self.state_mlp = StateMLP(in_features=8, out_dim=state_feat_dim)

        in_fused = board_feat_dim + state_feat_dim  # 256 + 64 = 320
        self.fusion = nn.Sequential(
            nn.Linear(in_fused, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(fused_dim, fused_dim),
        )

    def encode_board(
        self,
        board_champ_ids: torch.Tensor,
        board_star_levels: torch.Tensor | None = None,
        board_item_ids: torch.Tensor | None = None,
        board_traits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract the isolated 256D synergy-informed board feature."""
        board_tokens = self.champ2vec(
            champ_ids=board_champ_ids,
            star_levels=board_star_levels,
            item_ids=board_item_ids,
        )
        return self.board_encoder(
            board_tokens=board_tokens,
            board_champ_ids=board_champ_ids,
            trait_vec=board_traits,
        )

    def forward(
        self,
        board_champ_ids: torch.Tensor,
        board_star_levels: torch.Tensor | None = None,
        board_item_ids: torch.Tensor | None = None,
        board_traits: torch.Tensor | None = None,
        state_scalars: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        board_feat = self.encode_board(
            board_champ_ids=board_champ_ids,
            board_star_levels=board_star_levels,
            board_item_ids=board_item_ids,
            board_traits=board_traits,
        )

        if state_scalars is not None:
            state_feat = self.state_mlp(state_scalars)
        else:
            state_feat = torch.zeros(board_feat.shape[0], self.state_feat_dim, device=board_feat.device)

        fused_in = torch.cat([board_feat, state_feat], dim=-1)
        return self.fusion(fused_in)

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True
        self.train()

    def save_trunk(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "num_champs": self.num_champs,
                "num_items": self.num_items,
                "num_traits": self.num_traits,
                "champ_embed_dim": self.champ_embed_dim,
                "board_feat_dim": self.board_feat_dim,
                "state_feat_dim": self.state_feat_dim,
                "fused_dim": self.fused_dim,
            },
        }, path)

    @classmethod
    def load_trunk(cls, path: str | Path, map_location: str | torch.device = "cpu") -> MultiModalFusionTrunk:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        model = cls(
            num_champs=config.get("num_champs", 500),
            num_items=config.get("num_items", 300),
            num_traits=config.get("num_traits", 60),
            champ_embed_dim=config.get("champ_embed_dim", 32),
            board_feat_dim=config.get("board_feat_dim", 256),
            state_feat_dim=config.get("state_feat_dim", 64),
            fused_dim=config.get("fused_dim", 320),
        )
        if isinstance(checkpoint, dict):
            if "trunk_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["trunk_state_dict"])
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            elif "model_state_dict" in checkpoint:
                sd = checkpoint["model_state_dict"]
                trunk_sd = {k.replace("trunk.", ""): v for k, v in sd.items() if k.startswith("trunk.")}
                if trunk_sd:
                    model.load_state_dict(trunk_sd)
                else:
                    model.load_state_dict(sd)
            else:
                model.load_state_dict(checkpoint)
        return model


class TrunkPretrainModel(nn.Module):
    """Simplified Tri-Objective Pretraining Model.

    1. Macro: Match Top-4 Placement Classification from fused trunk (320D).
    2. Micro: PVP Combat Round Win Probability Logit from fused trunk (320D).
    3. Flow: Time-Contrastive InfoNCE Projection from ISOLATED board feature (256D).
    """

    def __init__(
        self,
        trunk: MultiModalFusionTrunk | None = None,
        num_champs: int = 500,
        num_items: int = 300,
        num_traits: int = 60,
        champ_embed_dim: int = 32,
        board_feat_dim: int = 256,
        state_feat_dim: int = 64,
        fused_dim: int = 320,
        proj_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = 2,
        **kwargs,
    ) -> None:
        super().__init__()
        if trunk is not None:
            self.trunk = trunk
        else:
            self.trunk = MultiModalFusionTrunk(
                num_champs=num_champs,
                num_items=num_items,
                num_traits=num_traits,
                champ_embed_dim=champ_embed_dim,
                board_feat_dim=board_feat_dim,
                state_feat_dim=state_feat_dim,
                fused_dim=fused_dim,
                num_layers=num_layers,
                dropout=dropout,
            )

        f_dim = self.trunk.fused_dim
        b_dim = self.trunk.board_feat_dim

        # Objective 1: Macro (Top-4 vs Bot-4 Classification)
        self.value_head = nn.Sequential(
            nn.Linear(f_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

        # Objective 2: Micro (PVP Combat Round Win Logit)
        self.combat_head = nn.Sequential(
            nn.Linear(f_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Objective 3: Flow (InfoNCE Projection from ISOLATED board feature)
        self.contrast_head = nn.Sequential(
            nn.Linear(b_dim, 128),
            nn.ReLU(),
            nn.Linear(128, proj_dim),
        )

    def forward_snapshot(
        self,
        board_champ_ids: torch.Tensor,
        board_star_levels: torch.Tensor | None = None,
        board_item_ids: torch.Tensor | None = None,
        board_traits: torch.Tensor | None = None,
        state_scalars: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        board_feat = self.trunk.encode_board(
            board_champ_ids=board_champ_ids,
            board_star_levels=board_star_levels,
            board_item_ids=board_item_ids,
            board_traits=board_traits,
        )

        if state_scalars is not None:
            state_feat = self.trunk.state_mlp(state_scalars)
        else:
            state_feat = torch.zeros(board_feat.shape[0], self.trunk.state_feat_dim, device=board_feat.device)

        fused = self.trunk.fusion(torch.cat([board_feat, state_feat], dim=-1))

        val_logits = self.value_head(fused)
        combat_logits = self.combat_head(fused)
        proj = self.contrast_head(board_feat)

        return fused, val_logits, combat_logits, proj

    def forward_dict(
        self, snapshot_dict: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_snapshot(
            board_champ_ids=snapshot_dict["board_champ_ids"],
            board_star_levels=snapshot_dict.get("board_star_levels"),
            board_item_ids=snapshot_dict.get("board_item_ids"),
            board_traits=snapshot_dict.get("board_traits"),
            state_scalars=snapshot_dict.get("state_scalars"),
        )
