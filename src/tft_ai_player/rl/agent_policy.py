"""Agent Policy wrapper and RLBot adapter enabling RL models to play in TFT simulations."""

from __future__ import annotations

import random
from typing import Any
import numpy as np
import torch

from tft_ai_player.rl.models.networks import TFTActorCritic
from tft_ai_player.simulation.actions import execute_action, get_action_mask
from tft_ai_player.simulation.config import SetData
from tft_ai_player.simulation.models import ChampionPool, Player
from tft_ai_player.simulation.observations import ObservationEncoder
from tft_ai_player.simulation.stage_manager import StageManager


class RLBot:
    """Adapter that allows a neural RL Actor-Critic policy to act as a bot in TFT games."""

    def __init__(
        self,
        model: TFTActorCritic,
        set_data: SetData,
        deterministic: bool = False,
        max_micro_actions: int = 30,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.set_data = set_data
        self.deterministic = deterministic
        self.max_micro_actions = max_micro_actions
        self.device = device or torch.device("cpu")
        self.encoder = ObservationEncoder(set_data)
        self.model.eval()

    def take_turn(
        self,
        player: Player,
        pool: ChampionPool,
        set_data: SetData,
        stage: int,
        round_in_stage: int,
        rng: random.Random | None = None,
        all_players: list[Player] | None = None,
        stage_manager: StageManager | None = None,
    ) -> None:
        """Execute bot planning turn by repeatedly predicting and executing micro-actions until PASS."""
        if not player.alive:
            return

        opponents = [p for p in (all_players or [player]) if p.player_id != player.player_id]
        if not opponents:
            opponents = [player]

        sm = stage_manager or StageManager()

        actions_taken = 0
        while actions_taken < self.max_micro_actions:
            # 1. Encode observation
            obs_dict = self.encoder.encode_dict(player, opponents, pool, sm)
            mask = obs_dict["action_mask"]

            # Prepare tensor batch of size 1
            tensor_obs: dict[str, torch.Tensor] = {}
            for k, v in obs_dict.items():
                if k != "action_mask":
                    tensor_obs[k] = torch.as_tensor(v, dtype=torch.float32, device=self.device).unsqueeze(0)

            mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)

            with torch.no_grad():
                action_t, _, _ = self.model.get_action(
                    tensor_obs, mask_tensor, deterministic=self.deterministic
                )
                action_id = int(action_t.item())

            # Action 0 is PASS (end planning turn)
            if action_id == 0:
                break

            # Execute micro action
            success = execute_action(player, pool, set_data, action_id, rng=rng)
            actions_taken += 1

            if not success:
                # If action failed for any reason, avoid infinite loops
                break
