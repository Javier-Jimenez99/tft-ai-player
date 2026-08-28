"""Multi-modal Actor-Critic Neural Network Architecture with Entity Embeddings for TFT."""

from __future__ import annotations

from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tft_ai_player.rl.models.distributions import MaskedCategorical
from tft_ai_player.simulation.actions import TOTAL_DISCRETE_ACTIONS


class TFTActorCritic(nn.Module):
    """Deep Actor-Critic model for TFT decision making.

    Integrates:
    - Entity embeddings for champions and items
    - Board & bench spatial representations
    - Permutation-invariant opponent pooling
    - Tabular player stats MLP
    - Invalid action masked categorical policy head
    - State-value critic head
    """

    def __init__(
        self,
        num_champs: int = 70,
        num_items: int = 150,
        champ_embed_dim: int = 16,
        item_embed_dim: int = 8,
        hidden_dim: int = 256,
        num_actions: int = TOTAL_DISCRETE_ACTIONS,
    ) -> None:
        super().__init__()
        self.num_champs = num_champs
        self.num_items = num_items
        self.champ_embed_dim = champ_embed_dim
        self.item_embed_dim = item_embed_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        # 1. Entity Embeddings
        self.champ_embedding = nn.Embedding(num_champs + 1, champ_embed_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items + 1, item_embed_dim, padding_idx=0)

        # 2. Board Unit Feature Dimension: champ_embed (16) + star (1) + 3 items (3*8=24) = 41
        unit_feat_dim = champ_embed_dim + 1 + (3 * item_embed_dim)

        # 3. Board & Bench Encoders
        self.board_encoder = nn.Sequential(
            nn.Linear(28 * unit_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        self.bench_encoder = nn.Sequential(
            nn.Linear(9 * unit_feat_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
        )

        # 4. Item Bench & Shop Encoders
        self.item_bench_encoder = nn.Sequential(
            nn.Linear(10 * item_embed_dim, hidden_dim // 4),
            nn.ReLU(),
        )

        self.shop_encoder = nn.Sequential(
            nn.Linear(5 * champ_embed_dim, hidden_dim // 4),
            nn.ReLU(),
        )

        # 5. Stats & Opponents Encoders
        # Stats: 9 scalar features
        self.stats_encoder = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Opponent Summary: 7 * 8 features
        self.opponents_summary_encoder = nn.Sequential(
            nn.Linear(7 * 8, 64),
            nn.ReLU(),
        )

        # Opponent boards pooling: each opponent board is (28 * unit_feat_dim)
        self.opp_board_linear = nn.Linear(28 * unit_feat_dim, 64)

        # Total fused representation dimension
        # board (128) + bench (64) + items (64) + shop (64) + stats (64) + opp_summary (64) + opp_boards (64) = 512
        fusion_dim = (hidden_dim // 2) + (hidden_dim // 4) + (hidden_dim // 4) + (hidden_dim // 4) + 64 + 64 + 64

        self.trunk = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor & Critic Heads
        self.actor_head = nn.Linear(hidden_dim, num_actions)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def _embed_units(self, unit_tensor: torch.Tensor) -> torch.Tensor:
        """Embed a unit tensor of shape (batch, num_slots, 5) -> (batch, num_slots, unit_feat_dim).

        Slots structure: [champ_idx, star_level, item1_idx, item2_idx, item3_idx]
        """
        champ_ids = unit_tensor[..., 0].long().clamp(0, self.num_champs)
        star_levels = (unit_tensor[..., 1:2].float() / 3.0)  # Normalize star (1..3)
        item1_ids = unit_tensor[..., 2].long().clamp(0, self.num_items)
        item2_ids = unit_tensor[..., 3].long().clamp(0, self.num_items)
        item3_ids = unit_tensor[..., 4].long().clamp(0, self.num_items)

        c_emb = self.champ_embedding(champ_ids)
        i1_emb = self.item_embedding(item1_ids)
        i2_emb = self.item_embedding(item2_ids)
        i3_emb = self.item_embedding(item3_ids)

        return torch.cat([c_emb, star_levels, i1_emb, i2_emb, i3_emb], dim=-1)

    def _extract_features(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract multi-modal fusion latent representation from observation dict."""
        batch_size = obs_dict["player_stats"].shape[0]

        # 1. Board & Bench Embeddings
        board_raw = obs_dict["board"].view(batch_size, 28, 5)
        board_emb = self._embed_units(board_raw).view(batch_size, -1)
        board_feat = self.board_encoder(board_emb)

        bench_raw = obs_dict["bench"].view(batch_size, 9, 5)
        bench_emb = self._embed_units(bench_raw).view(batch_size, -1)
        bench_feat = self.bench_encoder(bench_emb)

        # 2. Item Bench & Shop Embeddings
        item_bench_raw = obs_dict["item_bench"].long().clamp(0, self.num_items)
        item_bench_emb = self.item_embedding(item_bench_raw).view(batch_size, -1)
        item_bench_feat = self.item_bench_encoder(item_bench_emb)

        shop_raw = obs_dict["shop"].long().clamp(0, self.num_champs)
        shop_emb = self.champ_embedding(shop_raw).view(batch_size, -1)
        shop_feat = self.shop_encoder(shop_emb)

        # 3. Stats & Opponents
        stats_feat = self.stats_encoder(obs_dict["player_stats"].float())
        opp_summary_feat = self.opponents_summary_encoder(obs_dict["opponents"].view(batch_size, -1).float())

        # Opponent boards permutation-invariant pooling
        opp_boards_raw = obs_dict["opponents_boards"].view(batch_size * 7, 28, 5)
        opp_boards_emb = self._embed_units(opp_boards_raw).view(batch_size, 7, -1)
        opp_boards_encoded = F.relu(self.opp_board_linear(opp_boards_emb))
        opp_boards_pool = torch.mean(opp_boards_encoded, dim=1)  # Mean pooling over 7 opponents

        # 4. Fusion Trunk
        fused = torch.cat(
            [board_feat, bench_feat, item_bench_feat, shop_feat, stats_feat, opp_summary_feat, opp_boards_pool],
            dim=-1,
        )
        latent = self.trunk(fused)
        return latent

    def forward(
        self,
        obs_dict: dict[str, torch.Tensor],
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute raw policy action logits and state value.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (logits of shape (batch, 1721), values of shape (batch, 1))
        """
        latent = self._extract_features(obs_dict)
        logits = self.actor_head(latent)
        values = self.critic_head(latent)
        return logits, values

    def get_action(
        self,
        obs_dict: dict[str, torch.Tensor],
        action_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample masked discrete action and value estimate.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (action, log_prob, value)
        """
        logits, value = self.forward(obs_dict)
        dist = MaskedCategorical(logits=logits, mask=action_mask)
        action, log_prob = dist.sample_action(deterministic=deterministic)
        return action, log_prob, value.squeeze(-1)

    def evaluate_actions(
        self,
        obs_dict: dict[str, torch.Tensor],
        actions: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log probs, state values, and entropy for a batch of transitions.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (log_probs, values, entropy)
        """
        logits, values = self.forward(obs_dict)
        dist = MaskedCategorical(logits=logits, mask=action_mask)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, values.squeeze(-1), entropy
