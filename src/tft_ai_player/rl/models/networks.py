"""Multi-modal Actor-Critic Neural Network Architecture with Multi-Head Transformers and Deep LSTM for TFT."""

from __future__ import annotations

from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tft_ai_player.rl.models.distributions import (
    ACTION_TYPE_RANGES,
    NUM_ACTION_TYPES,
    HierarchicalMaskedCategorical,
    MaskedCategorical,
)
from tft_ai_player.simulation.actions import TOTAL_DISCRETE_ACTIONS


class UnitSynergyTransformer(nn.Module):
    """Lightweight Multi-Head Self-Attention Transformer block over champion and item embeddings."""

    def __init__(self, embed_dim: int = 32, num_heads: int = 4, hidden_dim: int = 64) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        # Token type embeddings: 0 = champion, 1 = item1, 2 = item2, 3 = item3
        self.token_type_embed = nn.Parameter(torch.randn(1, 4, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.0,
            batch_first=True,
            activation="relu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=False)

    def forward(self, unit_tokens: torch.Tensor) -> torch.Tensor:
        """Process unit tokens (..., 4, embed_dim) through self-attention across items and champion."""
        orig_shape = unit_tokens.shape
        flat_tokens = unit_tokens.view(-1, 4, self.embed_dim) + self.token_type_embed
        attended = self.transformer(flat_tokens)
        return attended.view(orig_shape)


class BoardSpatialTransformer(nn.Module):
    """Spatial Multi-Head Attention Transformer over 28 board hexes.

    Models frontline positioning, backline carry protection, and trait aura adjacency.
    """

    def __init__(self, unit_dim: int, num_heads: int = 4, hidden_dim: int = 192) -> None:
        super().__init__()
        self.unit_proj = nn.Linear(unit_dim, hidden_dim)
        # Learnable 2D positional embeddings for 4 rows x 7 cols = 28 hexes
        self.pos_embed = nn.Parameter(torch.randn(1, 28, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            batch_first=True,
            activation="relu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, board_units: torch.Tensor) -> torch.Tensor:
        """Process (batch_size, 28, unit_dim) -> returns (batch_size, hidden_dim)."""
        projected = self.unit_proj(board_units) + self.pos_embed
        attended = self.transformer(projected)
        board_feat = self.out_norm(torch.mean(attended, dim=1))
        return board_feat


class OpponentScoutingTransformer(nn.Module):
    """Multi-Head Cross-Attention Transformer across all 7 opponents in the lobby.

    Extracts high-level lobby dynamics, leader threat levels, and contested champion compositions.
    """

    def __init__(self, opp_dim: int = 128, num_heads: int = 4, hidden_dim: int = 256) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, opp_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=opp_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.0,
            batch_first=True,
            activation="relu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(opp_dim)

    def forward(self, opp_tokens: torch.Tensor) -> torch.Tensor:
        """Process (batch_size, 7, opp_dim) -> returns (batch_size, opp_dim) [SCOUT_CLS] vector."""
        batch_size = opp_tokens.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, opp_tokens], dim=1)  # (batch_size, 8, opp_dim)
        attended = self.transformer(tokens)
        cls_out = self.out_norm(attended[:, 0, :])  # Extract [SCOUT_CLS]
        return cls_out


class LatentStrategyManager:
    """AlphaStar-style Latent Strategy Manager for Set 18 Archetype Conditioning."""

    SET_18_ARCHETYPES = [
        "FAST_8_FLEX",
        "DUELIST_REROLL",
        "BRUISER_TANK",
        "ARCANIST_MAGE",
        "ASSASSIN_INFILTRATOR",
        "ECON_GREED_LEVEL_9",
    ]

    @classmethod
    def num_strategies(cls) -> int:
        return len(cls.SET_18_ARCHETYPES)


class TFTActorCritic(nn.Module):
    """Deep Multi-Modal Transformer + 2-Layer LSTM Actor-Critic model for TFT decision making.

    Integrates:
    - Entity embeddings for champions and items
    - UnitSynergyTransformer for combinatorial item-champion synergies
    - BoardSpatialTransformer with 2D positional embeddings across 28 hexes
    - OpponentScoutingTransformer across all 7 opponents in the lobby
    - Latent Strategy Conditioning (z-vector) across Set 18 archetypes
    - LayerNorm-stabilized bench, shop, and stats encoders
    - 2-layer Recurrent LSTM memory (h, c) for long-horizon POMDP credit assignment
    - Hierarchical & Autoregressive Action Space (13 Action Types + Conditioned Argument Heads)
    - Asymmetric Privileged Critic evaluating global lobby state
    - Auxiliary Multi-Task Contrastive Heads (Suphx-style combat win & placement forecasting)
    """

    def __init__(
        self,
        num_champs: int = 70,
        num_items: int = 150,
        champ_embed_dim: int = 32,
        item_embed_dim: int = 32,
        hidden_dim: int = 384,
        num_lstm_layers: int = 2,
        num_actions: int = TOTAL_DISCRETE_ACTIONS,
        num_action_types: int = NUM_ACTION_TYPES,
    ) -> None:
        super().__init__()
        self.num_champs = num_champs
        self.num_items = num_items
        self.champ_embed_dim = champ_embed_dim
        self.item_embed_dim = item_embed_dim
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.num_actions = num_actions
        self.num_action_types = num_action_types

        # 1. Entity Embeddings & Unit Synergy Transformer
        self.champ_embedding = nn.Embedding(num_champs + 1, champ_embed_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items + 1, item_embed_dim, padding_idx=0)
        self.synergy_embed_dim = 32
        self.champ_proj = nn.Linear(champ_embed_dim, self.synergy_embed_dim)
        self.item_proj = nn.Linear(item_embed_dim, self.synergy_embed_dim)
        self.synergy_transformer = UnitSynergyTransformer(embed_dim=self.synergy_embed_dim, num_heads=4, hidden_dim=64)

        # 2. Board Unit Feature Dimension: 4 attended tokens (4 * 32 = 128) + star level (1) = 129
        self.unit_feat_dim = (4 * self.synergy_embed_dim) + 1

        # 3. Board Spatial Transformer & Bench Spatial Encoder
        self.board_transformer = BoardSpatialTransformer(
            unit_dim=self.unit_feat_dim,
            num_heads=4,
            hidden_dim=hidden_dim // 2,
        )

        self.bench_encoder = nn.Sequential(
            nn.Linear(9 * self.unit_feat_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
        )

        # 4. Item Bench & Shop Encoders with LayerNorm
        self.item_bench_encoder = nn.Sequential(
            nn.Linear(10 * item_embed_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
        )

        self.shop_encoder = nn.Sequential(
            nn.Linear(5 * champ_embed_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
        )

        # 5. Stats & Opponents Multi-Head Scouting Transformer
        self.stats_encoder = nn.Sequential(
            nn.Linear(9, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

        self.opp_single_stat_encoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        self.opp_single_board_encoder = nn.Sequential(
            nn.Linear(28 * self.unit_feat_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

        # Opponent Scouting Transformer: processes 7 opponents x 128 features with [SCOUT_CLS]
        self.scouting_transformer = OpponentScoutingTransformer(opp_dim=128, num_heads=4, hidden_dim=256)

        # 6. Latent Strategy Conditioning Embedding (z-vector)
        self.strategy_embed = nn.Embedding(LatentStrategyManager.num_strategies() + 1, 32)

        # 7. Multi-Modal Fusion Trunk
        # board (hidden_dim//2) + bench (hidden_dim//4) + items (hidden_dim//4) + shop (hidden_dim//4) + stats (64) + opp_scout (128) + strategy (32)
        fusion_dim = (hidden_dim // 2) + (hidden_dim // 4) + (hidden_dim // 4) + (hidden_dim // 4) + 64 + 128 + 32

        self.trunk = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # 8. 2-Layer Recurrent Memory (LSTM) with Cell State for POMDP game tracking
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_lstm_layers, batch_first=True)

        # 9. Hierarchical Policy Heads:
        self.type_head = nn.Linear(hidden_dim, num_action_types)
        self.arg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

        # 10. State-Value Critic Head
        self.critic_head = nn.Linear(hidden_dim, 1)

        # 11. Asymmetric Privileged Critic Network (OpenAI Five / AlphaStar)
        self.privileged_critic = nn.Sequential(
            nn.Linear(hidden_dim + 128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # 12. Auxiliary Multi-Task Contrastive Prediction Heads (Suphx Style)
        self.aux_win_head = nn.Linear(hidden_dim, 1)  # Win probability of next combat
        self.aux_placement_head = nn.Linear(hidden_dim, 8)  # Expected lobby placement (1st - 8th)

    def _embed_units(self, unit_tensor: torch.Tensor) -> torch.Tensor:
        """Embed unit tensor [champ_idx, star_level, item1, item2, item3] through self-attention transformer.

        Returns tensor of shape (..., num_slots, unit_feat_dim).
        """
        champ_ids = unit_tensor[..., 0].long().clamp(0, self.num_champs)
        star_levels = unit_tensor[..., 1:2].float() / 3.0  # Normalize star (1..3)
        item1_ids = unit_tensor[..., 2].long().clamp(0, self.num_items)
        item2_ids = unit_tensor[..., 3].long().clamp(0, self.num_items)
        item3_ids = unit_tensor[..., 4].long().clamp(0, self.num_items)

        c_emb = self.champ_embedding(champ_ids)
        i1_emb = self.item_embedding(item1_ids)
        i2_emb = self.item_embedding(item2_ids)
        i3_emb = self.item_embedding(item3_ids)

        c_tok = self.champ_proj(c_emb)
        i1_tok = self.item_proj(i1_emb)
        i2_tok = self.item_proj(i2_emb)
        i3_tok = self.item_proj(i3_emb)
        tokens = torch.stack([c_tok, i1_tok, i2_tok, i3_tok], dim=-2)
        attended_tokens = self.synergy_transformer(tokens)

        unit_vec = torch.cat([attended_tokens.flatten(start_dim=-2), star_levels], dim=-1)
        return unit_vec

    def _extract_features(
        self,
        obs_dict: dict[str, torch.Tensor],
        hidden_state: tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Extract multi-modal fusion latent representation and apply recurrent LSTM step."""
        batch_size = obs_dict["player_stats"].shape[0]
        device = next(self.parameters()).device
        obs_dict = {
            k: v.to(device) if isinstance(v, torch.Tensor) and v.device != device else v
            for k, v in obs_dict.items()
        }

        # 1. Board Spatial Transformer & Bench Encoders
        board_raw = obs_dict["board"].view(batch_size, 28, 5)
        board_units = self._embed_units(board_raw)  # (batch_size, 28, unit_feat_dim)
        board_feat = self.board_transformer(board_units)  # (batch_size, hidden_dim // 2)

        bench_raw = obs_dict["bench"].view(batch_size, 9, 5)
        bench_emb = self._embed_units(bench_raw).view(batch_size, -1)
        bench_feat = self.bench_encoder(bench_emb)

        # 2. Item Bench & Shop Encoders
        item_bench_raw = obs_dict["item_bench"].long().clamp(0, self.num_items)
        item_bench_emb = self.item_embedding(item_bench_raw).view(batch_size, -1)
        item_bench_feat = self.item_bench_encoder(item_bench_emb)

        shop_raw = obs_dict["shop"].long().clamp(0, self.num_champs)
        shop_emb = self.champ_embedding(shop_raw).view(batch_size, -1)
        shop_feat = self.shop_encoder(shop_emb)

        # 3. Player Stats & Opponents Multi-Head Scouting Transformer
        stats_feat = self.stats_encoder(obs_dict["player_stats"].float())

        # Encode each of the 7 opponents (stats + board) -> (batch_size, 7, 128)
        opp_stats_raw = obs_dict["opponents"].view(batch_size, 7, 8).float()
        opp_stats_encoded = self.opp_single_stat_encoder(opp_stats_raw)  # (batch_size, 7, 64)

        opp_boards_raw = obs_dict["opponents_boards"].view(batch_size * 7, 28, 5)
        opp_boards_units = self._embed_units(opp_boards_raw).view(batch_size, 7, -1)
        opp_boards_encoded = self.opp_single_board_encoder(opp_boards_units)  # (batch_size, 7, 64)

        opp_tokens = torch.cat([opp_stats_encoded, opp_boards_encoded], dim=-1)  # (batch_size, 7, 128)
        opp_scout_feat = self.scouting_transformer(opp_tokens)  # (batch_size, 128)

        # 4. Latent Strategy Conditioning
        if "strategy" in obs_dict:
            strat_raw = obs_dict["strategy"].long().clamp(0, LatentStrategyManager.num_strategies())
            strat_emb = self.strategy_embed(strat_raw).view(batch_size, -1)
        else:
            strat_emb = torch.zeros(batch_size, 32, device=stats_feat.device)

        # 5. Fusion Trunk
        fused = torch.cat(
            [board_feat, bench_feat, item_bench_feat, shop_feat, stats_feat, opp_scout_feat, strat_emb],
            dim=-1,
        )
        trunk_out = self.trunk(fused)

        # 6. Recurrent 2-Layer LSTM memory update (h, c)
        if hidden_state is None:
            h = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_dim, device=trunk_out.device)
            c = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_dim, device=trunk_out.device)
        elif isinstance(hidden_state, tuple):
            h, c = hidden_state
            if h.shape[0] != self.num_lstm_layers:
                h = h.repeat(self.num_lstm_layers // h.shape[0], 1, 1)
                c = c.repeat(self.num_lstm_layers // c.shape[0], 1, 1)
        elif isinstance(hidden_state, torch.Tensor):
            if hidden_state.dim() == 2:
                h = hidden_state.unsqueeze(0).repeat(self.num_lstm_layers, 1, 1)
            else:
                h = hidden_state
            c = torch.zeros_like(h)
        else:
            h = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_dim, device=trunk_out.device)
            c = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_dim, device=trunk_out.device)

        lstm_out, (next_h, next_c) = self.lstm(trunk_out.unsqueeze(1), (h, c))
        latent = lstm_out.squeeze(1)

        return latent, (next_h, next_c)

    def forward(
        self,
        obs_dict: dict[str, torch.Tensor],
        hidden_state: tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Compute action logits and state value.

        If return_hidden is True:
            returns (type_logits, arg_logits, values, (next_h, next_c))
        Else:
            returns (arg_logits, values)
        """
        latent, (next_h, next_c) = self._extract_features(obs_dict, hidden_state)
        type_logits = self.type_head(latent)
        arg_logits = self.arg_head(latent)
        values = self.critic_head(latent)

        if return_hidden:
            return type_logits, arg_logits, values, (next_h, next_c)
        return arg_logits, values

    def get_action(
        self,
        obs_dict: dict[str, torch.Tensor],
        action_mask: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None = None,
        deterministic: bool = False,
        return_hidden: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Sample discrete action via hierarchical policy and value estimate."""
        latent, (next_h, next_c) = self._extract_features(obs_dict, hidden_state)
        type_logits = self.type_head(latent)
        arg_logits = self.arg_head(latent)
        values = self.critic_head(latent).squeeze(-1)

        device = latent.device
        action_mask = action_mask.to(device)

        # 1. Derive high-level Action Type Mask (13 types)
        type_mask = torch.zeros(action_mask.shape[0], self.num_action_types, dtype=torch.bool, device=device)
        for t_idx, (start_idx, end_idx) in enumerate(ACTION_TYPE_RANGES):
            type_mask[:, t_idx] = action_mask[:, start_idx:end_idx].any(dim=-1)

        # Fallback to PASS (type 0) if all masked
        all_false = ~type_mask.any(dim=-1)
        if all_false.any():
            type_mask[all_false, 0] = True

        type_dist = MaskedCategorical(logits=type_logits, mask=type_mask)
        if deterministic:
            type_action = torch.argmax(type_dist.probs, dim=-1)
        else:
            type_action = type_dist.sample()

        type_log_prob = type_dist.log_prob(type_action)

        # 2. Derive conditioned discrete action within chosen macro type
        conditioned_mask = torch.zeros_like(action_mask, dtype=torch.bool)
        for b in range(action_mask.shape[0]):
            t = int(type_action[b].item())
            start_idx, end_idx = ACTION_TYPE_RANGES[t]
            conditioned_mask[b, start_idx:end_idx] = action_mask[b, start_idx:end_idx]

        cond_all_false = ~conditioned_mask.any(dim=-1)
        if cond_all_false.any():
            conditioned_mask[cond_all_false, 0] = True

        arg_dist = MaskedCategorical(logits=arg_logits, mask=conditioned_mask)
        if deterministic:
            action = torch.argmax(arg_dist.probs, dim=-1)
        else:
            action = arg_dist.sample()

        arg_log_prob = arg_dist.log_prob(action)
        total_log_prob = type_log_prob + arg_log_prob

        if return_hidden:
            return action, total_log_prob, values, (next_h, next_c)
        return action, total_log_prob, values

    def get_value(
        self,
        obs_dict: dict[str, torch.Tensor],
        hidden_state: tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None = None,
        global_obs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute state value estimate.

        If privileged global_obs is provided (training time), applies Asymmetric Privileged Critic.
        """
        latent, _ = self._extract_features(obs_dict, hidden_state)
        if global_obs is not None:
            device = latent.device
            g_obs = global_obs.to(device).float()
            if g_obs.dim() == 1:
                g_obs = g_obs.unsqueeze(0)
            if g_obs.shape[-1] != 128:
                if g_obs.shape[-1] < 128:
                    pad = torch.zeros(g_obs.shape[0], 128 - g_obs.shape[-1], device=device)
                    g_obs = torch.cat([g_obs, pad], dim=-1)
                else:
                    g_obs = g_obs[:, :128]
            priv_in = torch.cat([latent, g_obs], dim=-1)
            return self.privileged_critic(priv_in).squeeze(-1)
        return self.critic_head(latent).squeeze(-1)

    def forward_auxiliary(
        self,
        obs_dict: dict[str, torch.Tensor],
        hidden_state: tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute self-supervised auxiliary predictions: (pred_win_prob, pred_placement_logits)."""
        latent, _ = self._extract_features(obs_dict, hidden_state)
        pred_win_prob = torch.sigmoid(self.aux_win_head(latent)).squeeze(-1)
        pred_placement_logits = self.aux_placement_head(latent)
        return pred_win_prob, pred_placement_logits

    def evaluate_actions(
        self,
        obs_dict: dict[str, torch.Tensor],
        actions: torch.Tensor,
        action_masks: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None = None,
        global_obs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute log probabilities, state values, policy entropy, and auxiliary predictions for PPO updates."""
        latent, _ = self._extract_features(obs_dict, hidden_state)
        type_logits = self.type_head(latent)
        arg_logits = self.arg_head(latent)

        if global_obs is not None:
            device = latent.device
            g_obs = global_obs.to(device).float()
            if g_obs.dim() == 1:
                g_obs = g_obs.unsqueeze(0)
            if g_obs.shape[-1] != 128:
                if g_obs.shape[-1] < 128:
                    pad = torch.zeros(g_obs.shape[0], 128 - g_obs.shape[-1], device=device)
                    g_obs = torch.cat([g_obs, pad], dim=-1)
                else:
                    g_obs = g_obs[:, :128]
            priv_in = torch.cat([latent, g_obs], dim=-1)
            values = self.privileged_critic(priv_in).squeeze(-1)
        else:
            values = self.critic_head(latent).squeeze(-1)

        device = latent.device
        action_masks = action_masks.to(device)
        actions = actions.to(device)

        # 1. Action Type distribution evaluation
        type_mask = torch.zeros(action_masks.shape[0], self.num_action_types, dtype=torch.bool, device=device)
        for t_idx, (start_idx, end_idx) in enumerate(ACTION_TYPE_RANGES):
            type_mask[:, t_idx] = action_masks[:, start_idx:end_idx].any(dim=-1)

        all_false = ~type_mask.any(dim=-1)
        if all_false.any():
            type_mask[all_false, 0] = True

        type_dist = MaskedCategorical(logits=type_logits, mask=type_mask)

        # Determine target macro types from chosen discrete actions
        target_types = torch.zeros_like(actions)
        for t_idx, (start_idx, end_idx) in enumerate(ACTION_TYPE_RANGES):
            in_range = (actions >= start_idx) & (actions < end_idx)
            target_types[in_range] = t_idx

        type_log_probs = type_dist.log_prob(target_types)
        type_entropy = type_dist.entropy()

        # 2. Conditioned Argument distribution evaluation
        conditioned_mask = torch.zeros_like(action_masks, dtype=torch.bool)
        for b in range(action_masks.shape[0]):
            t = int(target_types[b].item())
            start_idx, end_idx = ACTION_TYPE_RANGES[t]
            conditioned_mask[b, start_idx:end_idx] = action_masks[b, start_idx:end_idx]

        cond_all_false = ~conditioned_mask.any(dim=-1)
        if cond_all_false.any():
            conditioned_mask[cond_all_false, 0] = True

        arg_dist = MaskedCategorical(logits=arg_logits, mask=conditioned_mask)
        arg_log_probs = arg_dist.log_prob(actions)
        arg_entropy = arg_dist.entropy()

        total_log_probs = type_log_probs + arg_log_probs
        total_entropy = type_entropy + arg_entropy

        # 3. Auxiliary multi-task predictions
        aux_win = torch.sigmoid(self.aux_win_head(latent)).squeeze(-1)
        aux_place = self.aux_placement_head(latent)

        return total_log_probs, values, total_entropy, aux_win, aux_place
