"""Tournament Evaluator and Deterministic Benchmark Bot Validation for TFT League."""

from __future__ import annotations

import random
from typing import Any, Sequence
import numpy as np

from tft_ai_player.embeddings.model import MultiModalFusionTrunk
from tft_ai_player.rl.agent_policy import RLBot
from tft_ai_player.rl.league.league_manager import LeagueManager
from tft_ai_player.rl.models.networks import TFTActorCritic
from tft_ai_player.rl.types import AgentProfile, AgentRole, MatchResult
from tft_ai_player.simulation.bots import (
    BaseBot,
    BotAlphaFast8,
    BotBetaHyperroll,
    BotGammaGreedy,
    RandomBot,
)
from tft_ai_player.simulation.config import SetData, get_default_set17_data
from tft_ai_player.simulation.game import TFTGame


class BenchmarkBotEvaluator:
    """Evaluates the Main Agent against fixed deterministic benchmark bots:

    - Bot Alpha (Standard 4-Cost Fast-8)
    - Bot Beta (Hyperroll 1-Cost)
    - Bot Gamma (Greedy Economy Open-Fort)
    """

    def __init__(
        self,
        set_data: SetData | None = None,
        combat_resolver: Any | None = None,
        trunk: MultiModalFusionTrunk | None = None,
    ) -> None:
        self.set_data = set_data or get_default_set17_data()
        self.combat_resolver = combat_resolver
        self.trunk = trunk

    def evaluate_main_agent(
        self,
        main_model: TFTActorCritic,
        num_matches: int = 100,
        base_seed: int = 10000,
    ) -> dict[str, float]:
        """Run evaluation matches where Main Agent plays in 8-player lobbies against benchmark bots.

        Lobby composition: 1 Main Agent + 3 Bot Alpha + 2 Bot Beta + 2 Bot Gamma.
        """
        main_model.eval()
        main_bot = RLBot(model=main_model, set_data=self.set_data, trunk=self.trunk, deterministic=True)

        placements: list[int] = []
        wins_vs_alpha = 0
        wins_vs_beta = 0
        wins_vs_gamma = 0
        total_encounters_alpha = 0
        total_encounters_beta = 0
        total_encounters_gamma = 0

        for m_idx in range(num_matches):
            game = TFTGame(set_data=self.set_data, combat_resolver=self.combat_resolver)
            game.reset(seed=base_seed + m_idx)

            # Assign seats: Seat 0 = Main Agent, Seats 1..3 = Alpha, Seats 4..5 = Beta, Seats 6..7 = Gamma
            bots: list[Any] = [
                main_bot,
                BotAlphaFast8(),
                BotAlphaFast8(),
                BotAlphaFast8(),
                BotBetaHyperroll(),
                BotBetaHyperroll(),
                BotGammaGreedy(),
                BotGammaGreedy(),
            ]

            while not game.is_over:
                for idx, player in enumerate(game.players):
                    if player.alive:
                        rinfo = game.stage_manager.get_current_round_info()
                        bots[idx].take_turn(
                            player=player,
                            pool=game.pool,
                            set_data=self.set_data,
                            stage=rinfo.stage,
                            round_in_stage=rinfo.round_in_stage,
                        )
                game.resolve_round_phase()

            main_player = game.players[0]
            main_place = main_player.placement or 8
            placements.append(main_place)

            # Track relative wins vs benchmark bot archetypes
            for opp_idx in range(1, 8):
                opp_place = game.players[opp_idx].placement or 8
                main_beat_opp = (main_place < opp_place)
                if 1 <= opp_idx <= 3:
                    total_encounters_alpha += 1
                    if main_beat_opp:
                        wins_vs_alpha += 1
                elif 4 <= opp_idx <= 5:
                    total_encounters_beta += 1
                    if main_beat_opp:
                        wins_vs_beta += 1
                else:
                    total_encounters_gamma += 1
                    if main_beat_opp:
                        wins_vs_gamma += 1

        avg_placement = float(np.mean(placements))
        win_rate = float(np.mean([p == 1 for p in placements]))
        top4_rate = float(np.mean([p <= 4 for p in placements]))
        alpha_wr = float(wins_vs_alpha / max(1, total_encounters_alpha))
        beta_wr = float(wins_vs_beta / max(1, total_encounters_beta))
        gamma_wr = float(wins_vs_gamma / max(1, total_encounters_gamma))

        return {
            "eval_avg_placement": avg_placement,
            "eval_win_rate": win_rate,
            "eval_top4_rate": top4_rate,
            "bot_alpha_win_rate": alpha_wr,
            "bot_beta_win_rate": beta_wr,
            "bot_gamma_win_rate": gamma_wr,
            "eval_matches": float(num_matches),
        }


class TournamentEvaluator:
    """Evaluates multiple agents in 8-player TFT lobbies with multilateral Elo tracking."""

    def __init__(
        self,
        league: LeagueManager,
        set_data: SetData | None = None,
        combat_resolver: Any | None = None,
        trunk: MultiModalFusionTrunk | None = None,
    ) -> None:
        self.league = league
        self.set_data = set_data or get_default_set17_data()
        self.combat_resolver = combat_resolver
        self.trunk = trunk

    def run_match(
        self,
        agent_seats: list[str],
        seed: int,
        agent_models: dict[str, TFTActorCritic] | None = None,
    ) -> MatchResult:
        """Run a single 8-player lobby match with given seed and assigned agent IDs."""
        models = agent_models or {}
        assert len(agent_seats) == 8, f"TFT lobby requires 8 players, got {len(agent_seats)}"

        game = TFTGame(set_data=self.set_data, combat_resolver=self.combat_resolver)
        game.reset(seed=seed)

        seat_bots: list[Any] = []
        for aid in agent_seats:
            prof = self.league.profiles.get(aid)
            if prof and prof.role == AgentRole.BASELINE:
                bot_type = prof.metadata.get("bot_type")
                if bot_type == "BotBetaHyperroll":
                    seat_bots.append(BotBetaHyperroll())
                elif bot_type == "BotGammaGreedy":
                    seat_bots.append(BotGammaGreedy())
                else:
                    seat_bots.append(BotAlphaFast8())
            elif aid in models:
                seat_bots.append(
                    RLBot(model=models[aid], set_data=self.set_data, trunk=self.trunk, target_z=prof.target_z if prof else None)
                )
            else:
                seat_bots.append(BotAlphaFast8())

        while not game.is_over:
            for idx, player in enumerate(game.players):
                if player.alive:
                    rinfo = game.stage_manager.get_current_round_info()
                    seat_bots[idx].take_turn(
                        player=player,
                        pool=game.pool,
                        set_data=self.set_data,
                        stage=rinfo.stage,
                        round_in_stage=rinfo.round_in_stage,
                    )
            game.resolve_round_phase()

        placements: dict[str, int] = {}
        for idx, player in enumerate(game.players):
            aid = agent_seats[idx]
            placements[aid] = player.placement or 8

        return MatchResult(match_id=f"eval_match_{seed}", placements=placements)
