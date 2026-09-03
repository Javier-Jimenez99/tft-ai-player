"""Surrogate combat engines: ML model-based resolver and domain heuristic fallback."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from tft_ai_player.simulation.config import SetData, UnitRole
from tft_ai_player.simulation.models import Player


@dataclass(frozen=True, slots=True)
class CombatResult:
    """Outcome of a single 1v1 round matchup."""

    winner_id: int
    loser_id: int
    damage_dealt: int
    win_prob_a: float
    surviving_units: int
    is_ghost_b: bool = False


class CombatResolver(Protocol):
    """Protocol for resolving 1v1 combat outcomes between two players."""

    def resolve(
        self,
        player_a: Player,
        player_b: Player,
        is_ghost_b: bool,
        stage: int,
        stage_str: str,
        set_data: SetData,
        rng: random.Random | None = None,
    ) -> CombatResult:
        """Evaluate combat and return structured result."""
        ...


class HeuristicCombatResolver:
    """Fast domain heuristic combat evaluator based on gold value, star levels, items, and traits."""

    def __init__(self, stochastic: bool = True) -> None:
        self.stochastic = stochastic

    def compute_player_power(self, player: Player, set_data: SetData) -> float:
        """Calculate effective combat rating for a player's board."""
        if not player.board:
            return 0.0

        total_power = 0.0

        for unit in player.board.values():
            cdef = set_data.champions.get(unit.champion_id)
            cost = cdef.cost if cdef else unit.cost
            role = cdef.role if cdef else UnitRole.UTILITY

            # Star multiplier: 1-star (1.0), 2-star (1.8), 3-star (3.6)
            star_multiplier = 1.8 ** (unit.star_level - 1)
            unit_base_power = cost * 10.0 * star_multiplier

            # Item power & synergies
            item_power = 0.0
            for item_id in unit.items:
                is_comp = set_data.is_component(item_id)
                if is_comp:
                    item_power += 5.0
                else:
                    item_power += 15.0
                    # Role synergy bonuses
                    if role == UnitRole.AP_CARRY and ("Rabadon" in item_id or "Jeweled" in item_id or "Archangel" in item_id or "Gunblade" in item_id or "Blue" in item_id or "Shojin" in item_id):
                        item_power += 10.0
                    elif role == UnitRole.AD_CARRY and ("Infinity" in item_id or "Deathblade" in item_id or "LastWhisper" in item_id or "Rageblade" in item_id or "Bloodthirster" in item_id):
                        item_power += 10.0
                    elif role == UnitRole.TANK and ("Warmog" in item_id or "Bramble" in item_id or "Dragon" in item_id or "Gargoyle" in item_id or "Sunfire" in item_id or "Steadfast" in item_id):
                        item_power += 10.0

            total_power += unit_base_power + item_power

        # Trait synergy multipliers
        active_traits = player.get_active_traits()
        active_tier_sum = sum(active_traits.values())
        synergy_multiplier = 1.0 + (active_tier_sum * 0.08)

        return total_power * synergy_multiplier

    def resolve(
        self,
        player_a: Player,
        player_b: Player,
        is_ghost_b: bool,
        stage: int,
        stage_str: str,
        set_data: SetData,
        rng: random.Random | None = None,
    ) -> CombatResult:
        """Resolve combat between player A and player B."""
        r = rng or random

        power_a = self.compute_player_power(player_a, set_data)
        power_b = self.compute_player_power(player_b, set_data)

        # Handle empty boards edge cases
        if power_a == 0 and power_b == 0:
            win_prob_a = 0.5
        elif power_a == 0:
            win_prob_a = 0.0
        elif power_b == 0:
            win_prob_a = 1.0
        else:
            # Logistic sigmoid over power difference
            power_diff = power_a - power_b
            # Scale difference so that ~50 power diff gives ~90% win rate
            win_prob_a = 1.0 / (1.0 + math.exp(-power_diff / 25.0))
            win_prob_a = max(0.01, min(0.99, win_prob_a))

        if self.stochastic:
            a_won = r.random() < win_prob_a
        else:
            a_won = win_prob_a >= 0.5

        base_damage = set_data.stage_base_damage.get(stage, set_data.stage_base_damage.get(7, 17))

        if a_won:
            winner_id = player_a.player_id
            loser_id = player_b.player_id
            winner_board_len = len(player_a.board) or 1
            # Estimate surviving units based on win margin
            margin = max(0.0, power_a - power_b)
            surviving = min(winner_board_len, max(1, int(round(margin / 30.0)) + 1))
        else:
            winner_id = player_b.player_id
            loser_id = player_a.player_id
            winner_board_len = len(player_b.board) or 1
            margin = max(0.0, power_b - power_a)
            surviving = min(winner_board_len, max(1, int(round(margin / 30.0)) + 1))

        damage = base_damage + surviving

        return CombatResult(
            winner_id=winner_id,
            loser_id=loser_id,
            damage_dealt=damage,
            win_prob_a=win_prob_a,
            surviving_units=surviving,
            is_ghost_b=is_ghost_b,
        )


class MLCombatResolver:
    """Surrogate combat resolver utilizing the trained round winner ML model."""

    def __init__(
        self,
        model_pipeline: Any | None = None,
        stochastic: bool = True,
        fallback_resolver: CombatResolver | None = None,
    ) -> None:
        self.model_pipeline = model_pipeline
        self.stochastic = stochastic
        self.fallback = fallback_resolver or HeuristicCombatResolver(stochastic=stochastic)

    def resolve(
        self,
        player_a: Player,
        player_b: Player,
        is_ghost_b: bool,
        stage: int,
        stage_str: str,
        set_data: SetData,
        rng: random.Random | None = None,
    ) -> CombatResult:
        """Predict win probability via ML model and resolve combat."""
        if self.model_pipeline is None:
            return self.fallback.resolve(
                player_a,
                player_b,
                is_ghost_b,
                stage,
                stage_str,
                set_data,
                rng=rng,
            )

        r = rng or random

        try:
            if isinstance(self.model_pipeline, (str, Path)):
                from tft_ai_player.round_winner.trainer import RoundWinnerPredictor
                self.model_pipeline = RoundWinnerPredictor.load(self.model_pipeline)
        except (Exception, BaseException):
            return self.fallback.resolve(
                player_a,
                player_b,
                is_ghost_b,
                stage,
                stage_str,
                set_data,
                rng=rng,
            )

        predicted_damage_loss: int | None = None

        try:
            from tft_ai_player.round_winner.trainer import RoundWinnerPredictor
            if isinstance(self.model_pipeline, RoundWinnerPredictor):
                win_prob_a, dmg_a_loss, dmg_b_loss = self.model_pipeline.predict_combat(
                    player_a.to_feature_board(),
                    player_b.to_feature_board(),
                    round_stage=stage_str,
                    focal_level=player_a.level,
                    opponent_level=player_b.level,
                    focal_health=player_a.health,
                    opponent_health=player_b.health,
                    focal_gold=player_a.gold,
                    opponent_gold=player_b.gold,
                )
            else:
                row_dict = {
                    "round_stage": stage_str,
                    "focal_level": player_a.level,
                    "focal_health": player_a.health,
                    "focal_gold": player_a.gold,
                    "focal_augments": "",
                    "opponent_level": player_b.level,
                    "opponent_health": player_b.health,
                    "opponent_augments": "",
                    "input_state_json": {
                        "focal_board": player_a.to_feature_board(),
                        "opponent_board": player_b.to_feature_board(),
                    },
                }
                df = pd.DataFrame([row_dict])
                probs = self.model_pipeline.predict_proba(df)
                win_prob_a = float(probs[0, 1])
                dmg_a_loss = None
                dmg_b_loss = None
            win_prob_a = max(0.01, min(0.99, float(win_prob_a)))
        except (Exception, BaseException):
            # Fallback to heuristic on unexpected feature error or missing DLL
            return self.fallback.resolve(
                player_a,
                player_b,
                is_ghost_b,
                stage,
                stage_str,
                set_data,
                rng=rng,
            )

        if self.stochastic:
            a_won = r.random() < win_prob_a
        else:
            a_won = win_prob_a >= 0.5

        base_damage = set_data.stage_base_damage.get(stage, set_data.stage_base_damage.get(7, 17))

        if a_won:
            winner_id = player_a.player_id
            loser_id = player_b.player_id
            winner_board_len = len(player_a.board) or 1
            surviving = min(winner_board_len, max(1, int(round(win_prob_a * winner_board_len * 0.7))))
            if dmg_b_loss is not None:
                damage = dmg_b_loss
            else:
                damage = base_damage + surviving
        else:
            winner_id = player_b.player_id
            loser_id = player_a.player_id
            winner_board_len = len(player_b.board) or 1
            surviving = min(winner_board_len, max(1, int(round((1.0 - win_prob_a) * winner_board_len * 0.7))))
            if dmg_a_loss is not None:
                damage = dmg_a_loss
            else:
                damage = base_damage + surviving

        return CombatResult(
            winner_id=winner_id,
            loser_id=loser_id,
            damage_dealt=damage,
            win_prob_a=win_prob_a,
            surviving_units=surviving,
            is_ghost_b=is_ghost_b,
        )


class DLCombatResolver:
    """Surrogate combat resolver utilizing the pre-trained neural Trunk combat head."""

    def __init__(
        self,
        model: Any,
        encoder: Any | None = None,
        stochastic: bool = True,
        device: Any | None = None,
    ) -> None:
        import torch
        self.model = model
        self.encoder = encoder
        self.stochastic = stochastic
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fallback = HeuristicCombatResolver(stochastic=stochastic)

    def resolve(
        self,
        player_a: Player,
        player_b: Player,
        is_ghost_b: bool,
        stage: int,
        stage_str: str,
        set_data: SetData,
        rng: random.Random | None = None,
    ) -> CombatResult:
        """Resolve combat using Deep Learning board embeddings and combat head."""
        import torch

        if not player_a.board or not player_b.board:
            return self.fallback.resolve(
                player_a, player_b, is_ghost_b, stage, stage_str, set_data, rng=rng
            )

        r = rng or random

        try:
            if self.encoder is None:
                from tft_ai_player.simulation.gym_env import TFTStateEncoder
                trunk_obj = getattr(self.model, "trunk", self.model)
                self.encoder = TFTStateEncoder(trunk=trunk_obj, set_data=set_data, device=self.device)

            ca, sa, ia, ta = self.encoder.encode_board_tensors(player_a)
            cb, sb, ib, tb = self.encoder.encode_board_tensors(player_b)
            sca = self.encoder.encode_scalars(player_a, stage=stage, round_in_stage=1)
            scb = self.encoder.encode_scalars(player_b, stage=stage, round_in_stage=1)

            if hasattr(self.model, "interaction_mlp"):
                f_batch = {
                    "board_champ_ids": ca.unsqueeze(0).to(self.device),
                    "board_star_levels": sa.unsqueeze(0).to(self.device),
                    "board_item_ids": ia.unsqueeze(0).to(self.device),
                    "board_traits": ta.unsqueeze(0).to(self.device),
                    "state_scalars": sca.unsqueeze(0).to(self.device),
                }
                o_batch = {
                    "board_champ_ids": cb.unsqueeze(0).to(self.device),
                    "board_star_levels": sb.unsqueeze(0).to(self.device),
                    "board_item_ids": ib.unsqueeze(0).to(self.device),
                    "board_traits": tb.unsqueeze(0).to(self.device),
                    "state_scalars": scb.unsqueeze(0).to(self.device),
                }
                with torch.no_grad():
                    logit = self.model(f_batch, o_batch)
                    win_prob_a = float(torch.sigmoid(logit).item())
            else:
                c_batch = torch.stack([ca, cb]).to(self.device)
                s_batch = torch.stack([sa, sb]).to(self.device)
                i_batch = torch.stack([ia, ib]).to(self.device)
                t_batch = torch.stack([ta, tb]).to(self.device)
                sc_batch = torch.stack([sca, scb]).to(self.device)

                with torch.no_grad():
                    fused, _, combat_logits, _ = self.model.forward_snapshot(
                        board_champ_ids=c_batch,
                        board_star_levels=s_batch,
                        board_item_ids=i_batch,
                        board_traits=t_batch,
                        state_scalars=sc_batch,
                    )
                    diff = (combat_logits[0] - combat_logits[1]).item()
                    win_prob_a = float(torch.sigmoid(torch.tensor(diff)).item())
            win_prob_a = max(0.01, min(0.99, win_prob_a))
        except Exception:
            return self.fallback.resolve(
                player_a, player_b, is_ghost_b, stage, stage_str, set_data, rng=rng
            )

        if self.stochastic:
            a_won = r.random() < win_prob_a
        else:
            a_won = win_prob_a >= 0.5

        base_damage = set_data.stage_base_damage.get(stage, set_data.stage_base_damage.get(7, 17))
        if a_won:
            winner_id = player_a.player_id
            loser_id = player_b.player_id
            winner_board_len = len(player_a.board) or 1
            surviving = min(winner_board_len, max(1, int(round(win_prob_a * winner_board_len * 0.7))))
        else:
            winner_id = player_b.player_id
            loser_id = player_a.player_id
            winner_board_len = len(player_b.board) or 1
            surviving = min(winner_board_len, max(1, int(round((1.0 - win_prob_a) * winner_board_len * 0.7))))

        damage = base_damage + surviving

        return CombatResult(
            winner_id=winner_id,
            loser_id=loser_id,
            damage_dealt=damage,
            win_prob_a=win_prob_a,
            surviving_units=surviving,
            is_ghost_b=is_ghost_b,
        )

