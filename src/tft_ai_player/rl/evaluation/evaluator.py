"""Tournament Evaluator and Paired Seed (CRN) Luck-Mitigation Benchmark for TFT."""

from __future__ import annotations

import random
from typing import Any, Sequence

from tft_ai_player.rl.agent_policy import RLBot
from tft_ai_player.rl.league.league_manager import LeagueManager
from tft_ai_player.rl.types import AgentProfile, AgentRole, MatchResult
from tft_ai_player.simulation.bots import GreedyBankerBot, RandomBot, StandardTempoBot
from tft_ai_player.simulation.config import SetData, get_default_set17_data
from tft_ai_player.simulation.game import TFTGame


class TournamentEvaluator:
    """Evaluates multiple agents in 8-player TFT lobbies with paired-seed variance reduction."""

    def __init__(
        self,
        league: LeagueManager,
        set_data: SetData | None = None,
    ) -> None:
        self.league = league
        self.set_data = set_data or get_default_set17_data()

    def _resolve_bot_for_profile(self, profile: AgentProfile, model: Any | None = None) -> Any:
        """Resolve executable bot policy from agent profile."""
        if profile.metadata.get("bot_class") == "StandardTempoBot":
            return StandardTempoBot()
        if profile.metadata.get("bot_class") == "GreedyBankerBot":
            return GreedyBankerBot()
        if profile.metadata.get("bot_class") == "RandomBot":
            return RandomBot()

        # If it's an RL policy
        if model is not None:
            return RLBot(model=model, set_data=self.set_data)

        # Default fallback to standard tempo bot
        return StandardTempoBot()

    def run_match(
        self,
        agent_seats: list[str],
        seed: int,
        agent_models: dict[str, Any] | None = None,
    ) -> MatchResult:
        """Run a single 8-player lobby match with given seed and assigned agent IDs."""
        models = agent_models or {}
        num_players = len(agent_seats)
        assert num_players == 8, f"TFT lobby requires 8 players, got {num_players}"

        # Initialize game
        game = TFTGame(set_data=self.set_data)
        game.reset(seed=seed)

        # Build bot policies for each seat
        seat_bots: list[Any] = []
        for aid in agent_seats:
            prof = self.league.profiles.get(aid)
            if not prof:
                prof = self.league.register_agent(aid, aid, AgentRole.BASELINE)
            seat_bots.append(self._resolve_bot_for_profile(prof, models.get(aid)))

        # Simulate game rounds
        while not game.is_over:
            # 1. Planning phase for all alive players
            for idx, player in enumerate(game.players):
                if player.alive:
                    bot = seat_bots[idx]
                    rinfo = game.stage_manager.get_current_round_info()
                    if isinstance(bot, RLBot):
                        bot.take_turn(
                            player=player,
                            pool=game.pool,
                            set_data=self.set_data,
                            stage=rinfo.stage,
                            round_in_stage=rinfo.round_in_stage,
                            all_players=game.players,
                            stage_manager=game.stage_manager,
                        )
                    else:
                        bot.take_turn(
                            player=player,
                            pool=game.pool,
                            set_data=self.set_data,
                            stage=rinfo.stage,
                            round_in_stage=rinfo.round_in_stage,
                        )

            # 2. Combat phase
            game.resolve_round_phase()

        # Compute final placements
        placements: dict[str, int] = {}
        for idx, player in enumerate(game.players):
            aid = agent_seats[idx]
            placements[aid] = player.placement or (1 if player.alive else 8)

        match_id = f"match_s{seed}_{random.randint(1000, 9999)}"
        result = self.league.record_match_outcome(
            match_id=match_id,
            seed=seed,
            placements=placements,
            total_rounds=game.stage_manager.total_rounds_elapsed,
        )
        return result

    def run_paired_benchmark(
        self,
        candidate_agent_id: str,
        opponent_agent_ids: list[str],
        candidate_model: Any | None = None,
        opponent_models: dict[str, Any] | None = None,
        num_seeds: int = 10,
        base_seed: int = 42,
    ) -> dict[str, Any]:
        """Run Common Random Numbers (CRN) paired seed benchmark to isolate strategy from luck.

        Each seed runs the exact same shop cards and item drops to eliminate RNG noise.
        """
        all_models = {candidate_agent_id: candidate_model}
        if opponent_models:
            all_models.update(opponent_models)

        candidate_placements: list[int] = []
        candidate_top4_count = 0
        candidate_win_count = 0

        # Construct 8-player lobby seats
        # Candidate occupies seat 0; opponents fill remaining 7 seats
        assert len(opponent_agent_ids) >= 1, "Must provide at least 1 opponent agent ID"
        if len(opponent_agent_ids) < 7:
            # Fill remaining seats by repeating opponents
            extended_opps = (opponent_agent_ids * 7)[:7]
        else:
            extended_opps = opponent_agent_ids[:7]

        lobby_seats = [candidate_agent_id] + extended_opps

        for seed_offset in range(num_seeds):
            current_seed = base_seed + seed_offset
            match_res = self.run_match(
                agent_seats=lobby_seats,
                seed=current_seed,
                agent_models=all_models,
            )
            cand_rank = match_res.placements[candidate_agent_id]
            candidate_placements.append(cand_rank)
            if cand_rank <= 4:
                candidate_top4_count += 1
            if cand_rank == 1:
                candidate_win_count += 1

        avg_placement = float(sum(candidate_placements) / len(candidate_placements))
        top4_rate = float(candidate_top4_count / num_seeds)
        win_rate = float(candidate_win_count / num_seeds)

        return {
            "candidate_id": candidate_agent_id,
            "num_matches": num_seeds,
            "placements": candidate_placements,
            "avg_placement": avg_placement,
            "top4_rate": top4_rate,
            "win_rate": win_rate,
            "current_elo": self.league.profiles[candidate_agent_id].elo.rating,
        }
