"""Multi-agent 8-player TFT game simulation engine."""

from __future__ import annotations

import random
from typing import Sequence

from tft_ai_player.simulation.actions import execute_action
from tft_ai_player.simulation.bots import BaseBot, StandardTempoBot
from tft_ai_player.simulation.combat import (
    CombatResolver,
    CombatResult,
    HeuristicCombatResolver,
)
from tft_ai_player.simulation.config import SetData, get_default_set17_data
from tft_ai_player.simulation.matchmaking import MatchmakingEngine
from tft_ai_player.simulation.models import ChampionPool, Player
from tft_ai_player.simulation.stage_manager import StageManager


class TFTGame:
    """Orchestrates an entire 8-player TFT match from Stage 1-1 to final victory."""

    def __init__(
        self,
        set_data: SetData | None = None,
        combat_resolver: CombatResolver | None = None,
        bot_factory: type[BaseBot] = StandardTempoBot,
        seed: int | None = None,
    ) -> None:
        self.set_data = set_data or get_default_set17_data()
        self.combat_resolver = combat_resolver or HeuristicCombatResolver()
        self.matchmaking = MatchmakingEngine()
        self.bot_factory = bot_factory
        self.rng = random.Random(seed)

        self.pool = ChampionPool(self.set_data)
        self.stage_manager = StageManager(self.set_data)
        self.players: list[Player] = [
            Player(i, self.set_data, name=f"Player {i}") for i in range(8)
        ]
        self.bots: dict[int, BaseBot] = {
            i: self.bot_factory() for i in range(1, 8)
        }

        self.is_over: bool = False
        self.next_placement_to_assign: int = 8
        self.round_combat_results: list[CombatResult] = []

        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> None:
        """Reset game to initial starting state (Stage 1-1)."""
        if seed is not None:
            self.rng = random.Random(seed)

        self.pool = ChampionPool(self.set_data)
        self.stage_manager = StageManager(self.set_data)
        self.players = [
            Player(i, self.set_data, name=f"Player {i}") for i in range(8)
        ]
        self.bots = {
            i: self.bot_factory() for i in range(1, 8)
        }
        self.is_over = False
        self.next_placement_to_assign = 8
        self.round_combat_results = []

        # Stage 1-1 Initial Setup: First Carousel & Starting Gold
        self.stage_manager.handle_carousel(self.players, pool=self.pool, rng=self.rng)
        for player in self.players:
            player.gold = 2  # Stage 1-1 starting gold

        # Start first round
        self.stage_manager.execute_round_start(self.players, self.pool, rng=self.rng)

    def step_player_action(self, player_id: int, action_id: int) -> bool:
        """Execute a planning action for a specific player."""
        if self.is_over or not (0 <= player_id < len(self.players)):
            return False

        player = self.players[player_id]
        if not player.alive:
            return False

        return execute_action(player, self.pool, self.set_data, action_id, rng=self.rng)

    def execute_bot_turns(self) -> None:
        """Run planning phase policies for all alive AI bots."""
        rinfo = self.stage_manager.get_current_round_info()
        for p_id, bot in self.bots.items():
            player = self.players[p_id]
            if player.alive:
                bot.take_turn(
                    player=player,
                    pool=self.pool,
                    set_data=self.set_data,
                    stage=rinfo.stage,
                    round_in_stage=rinfo.round_in_stage,
                    rng=self.rng,
                )

    def resolve_round_phase(self) -> list[CombatResult]:
        """Resolve current round (PvP battles, PvE drops, or Carousels), process damage, and advance stage."""
        if self.is_over:
            return []

        rinfo = self.stage_manager.get_current_round_info()
        results: list[CombatResult] = []

        # 0. Enforce strict board capacity limits for all alive players
        for p in self.players:
            if p.alive:
                p.enforce_board_capacity(self.pool)

        # 1. Resolve Round Type Mechanics
        if rinfo.is_carousel:
            self.stage_manager.handle_carousel(self.players, pool=self.pool, rng=self.rng)
        elif rinfo.is_pve:
            self.stage_manager.handle_pve_loot(self.players, rng=self.rng)
            # Players maintain neutral streak through PvE
        elif rinfo.is_pvp:
            # Generate 1v1 matchups
            matchups = self.matchmaking.generate_pairings(self.players, rng=self.rng)

            for m in matchups:
                res = self.combat_resolver.resolve(
                    player_a=m.player_a,
                    player_b=m.player_b,
                    is_ghost_b=m.is_ghost_b,
                    stage=rinfo.stage,
                    stage_str=rinfo.stage_str,
                    set_data=self.set_data,
                    rng=self.rng,
                )
                results.append(res)

                # Process Winner
                winner = self.players[res.winner_id]
                winner.add_gold(1)  # +1 PvP Win bonus gold
                if winner.streak < 0:
                    winner.streak = 1
                else:
                    winner.streak += 1

                # Process Loser (ghost boards do not take damage)
                if not (m.is_ghost_b and res.loser_id == m.player_b.player_id):
                    loser = self.players[res.loser_id]
                    loser.health -= res.damage_dealt
                    if loser.streak > 0:
                        loser.streak = -1
                    else:
                        loser.streak -= 1

        self.round_combat_results = results

        # 2. Check for Eliminations
        self._check_and_process_eliminations()

        # 3. Advance to next round if game not over
        if not self.is_over:
            self.stage_manager.advance_round()
            self.stage_manager.execute_round_start(self.players, self.pool, rng=self.rng)

        return results

    def _check_and_process_eliminations(self) -> None:
        """Eliminate dead players, assign placements, and return units to pool."""
        # Find newly dead players
        dead_players = [p for p in self.players if p.alive and p.health <= 0]
        if not dead_players:
            return

        # Sort dead players by health ascending (most negative health gets lower placement)
        dead_players.sort(key=lambda p: (p.health, p.player_id))

        for p in dead_players:
            p.alive = False
            p.placement = self.next_placement_to_assign
            self.next_placement_to_assign -= 1
            # Return units to shared pool
            self.pool.return_player_units(p)

        # Check remaining alive count
        alive_players = [p for p in self.players if p.alive]
        if len(alive_players) <= 1:
            self.is_over = True
            if alive_players:
                winner = alive_players[0]
                winner.placement = 1

    def get_rankings(self) -> list[tuple[int, Player]]:
        """Return all players sorted by placement (1st to 8th)."""
        # Separate alive and dead players
        alive_players = sorted([p for p in self.players if p.alive], key=lambda p: (p.health, p.get_board_value()), reverse=True)
        dead_players = sorted([p for p in self.players if not p.alive], key=lambda p: (p.placement if p.placement is not None else 8))

        rankings: list[tuple[int, Player]] = []
        for rank_idx, p in enumerate(alive_players, start=1):
            rankings.append((rank_idx if not self.is_over else (p.placement or rank_idx), p))
        for p in dead_players:
            rankings.append((p.placement or (len(rankings) + 1), p))

        return rankings

    def get_focal_player(self) -> Player:
        """Return reference to Player 0."""
        return self.players[0]

    def get_opponents(self, focal_id: int = 0) -> list[Player]:
        """Return opponents of designated player."""
        return [p for p in self.players if p.player_id != focal_id]
