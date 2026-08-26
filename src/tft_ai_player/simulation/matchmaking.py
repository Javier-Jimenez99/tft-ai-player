"""Matchmaking engine for 8-player TFT lobbies with opponent exclusion pool and ghost boards."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from tft_ai_player.simulation.models import Player


@dataclass(frozen=True, slots=True)
class Matchup:
    """A pair of players scheduled to battle in a PvP round."""

    player_a: Player
    player_b: Player
    is_ghost_b: bool = False


class MatchmakingEngine:
    """Computes round pairings with anti-repeat opponent exclusion pools and odd-player ghost boards."""

    def __init__(self) -> None:
        pass

    def _get_exclusion_size(self, num_alive: int) -> int:
        """Determine how many past opponents should be excluded from candidate pool."""
        if num_alive >= 7:
            return 4
        if num_alive >= 5:
            return 3
        if num_alive >= 3:
            return 2
        return 0

    def generate_pairings(
        self,
        players: Sequence[Player],
        rng: random.Random | None = None,
    ) -> list[Matchup]:
        """Generate 1v1 matchups for all living players."""
        r = rng or random
        alive_players = [p for p in players if p.alive]
        num_alive = len(alive_players)

        if num_alive < 2:
            return []

        # If exactly 2 players alive: direct match
        if num_alive == 2:
            p1, p2 = alive_players[0], alive_players[1]
            p1.last_opponents.append(p2.player_id)
            p2.last_opponents.append(p1.player_id)
            return [Matchup(player_a=p1, player_b=p2, is_ghost_b=False)]

        exclusion_size = self._get_exclusion_size(num_alive)
        unmatched = list(alive_players)
        r.shuffle(unmatched)

        pairings: list[Matchup] = []

        # If odd number of players: designate one player to fight a ghost board
        ghost_matchup: Matchup | None = None
        if len(unmatched) % 2 == 1:
            ghost_player = unmatched.pop()
            # Select another alive player to act as the ghost army
            possible_ghosts = [p for p in alive_players if p.player_id != ghost_player.player_id]
            ghost_source = r.choice(possible_ghosts)
            ghost_matchup = Matchup(player_a=ghost_player, player_b=ghost_source, is_ghost_b=True)
            ghost_player.last_opponents.append(ghost_source.player_id)

        # Pair remaining even number of players
        while len(unmatched) >= 2:
            p1 = unmatched.pop(0)
            excluded_ids = set(p1.last_opponents[-exclusion_size:]) if exclusion_size > 0 else set()

            # Filter candidates not in exclusion list
            candidates = [p for p in unmatched if p.player_id not in excluded_ids]
            if not candidates:
                # If all available candidates are in exclusion, relax constraint
                candidates = unmatched

            p2 = r.choice(candidates)
            unmatched.remove(p2)

            p1.last_opponents.append(p2.player_id)
            p2.last_opponents.append(p1.player_id)
            pairings.append(Matchup(player_a=p1, player_b=p2, is_ghost_b=False))

        if ghost_matchup is not None:
            pairings.append(ghost_matchup)

        return pairings
