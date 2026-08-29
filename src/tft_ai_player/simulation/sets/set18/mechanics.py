"""Set 18 'Enchanted Wilds' Specific Trait Mechanics, Economy Drops, and Shop Modifiers."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Sequence

from tft_ai_player.simulation.config import STANDARD_COMPONENTS

if TYPE_CHECKING:
    from tft_ai_player.simulation.config import SetData
    from tft_ai_player.simulation.models import ChampionPool, Player


class Set18MechanicsHandler:
    """Dispatches Set 18 specific economy trait mechanics, shop overruns, and combat loot."""

    @staticmethod
    def apply_round_start(
        player: Player,
        set_data: SetData,
        pool: ChampionPool,
        rng: random.Random | None = None,
    ) -> list[str] | None:
        """Apply Set 18 round-start trait bonuses (Blossom rebate, Riftbeast overrun candidates)."""
        active_traits = player.get_active_traits()

        # 1. Blossom Trait: Wisps grant bonus gold rebate on stage progression
        if active_traits.get("Blossom", 0) >= 3:
            # 3+ Blossom: Wisp purchase rebate bonus (+1g)
            player.add_gold(1)

        # 2. Riftbeast Overrun Shop Candidates
        riftbeast_ids = [
            cid for cid, cdef in set_data.champions.items()
            if "Riftbeast" in cdef.traits
        ]
        candidates_pool: list[str] | None = None
        if active_traits.get("Riftbeast", 0) >= 2 and player.combats_since_overrun >= 4:
            if riftbeast_ids:
                candidates_pool = riftbeast_ids
                player.combats_since_overrun = 0

        return candidates_pool

    @staticmethod
    def apply_combat_loot_and_traits(
        players: list[Player],
        combat_results: Sequence[Any],
        pool: ChampionPool,
        set_data: SetData,
        rng: random.Random | None = None,
    ) -> None:
        """Resolve Set 18 Coven essence, Draven bounties, Fae pixies, Rengar takedowns, and Inferno ignitions."""
        r = rng or random
        component_ids = [cid for cid, _ in STANDARD_COMPONENTS if cid != "TFT_Item_Spatula"]

        for res in combat_results:
            winner = players[res.winner_id]
            loser = players[res.loser_id]

            if not winner.alive:
                continue

            # Average takedowns estimation for surrogate ML / heuristic resolver
            # Winner typically eliminates most enemy units (6-8), loser gets 2-4 takedowns
            winner_kills = max(3, len(loser.board) - max(0, res.surviving_units - 1))
            loser_kills = max(1, len(winner.board) - res.surviving_units)

            # -------------------------------------------------------------
            # Winner Trait & Item Resolution
            # -------------------------------------------------------------
            winner_traits = winner.get_active_traits()

            # Tactician's Crown win gold proc (10% chance for 1g)
            for it in winner.item_bench:
                if "Tactician" in it.name or "Crown" in it.item_id:
                    if r.random() < 0.10:
                        winner.add_gold(1)
            for u in winner.board.values():
                if any("Tactician" in i or "Crown" in i for i in u.items):
                    if r.random() < 0.10:
                        winner.add_gold(1)

            # Coven (Winner: Essence per kill)
            if winner_traits.get("Coven", 0) >= 1:
                winner.coven_essence += winner_kills
                if winner.coven_essence >= 40:
                    cashout_gold = 4 + (winner.coven_essence // 10)
                    winner.add_gold(cashout_gold)
                    if winner.coven_essence >= 80 and component_ids:
                        winner.add_item(r.choice(component_ids))
                    winner.coven_essence = 0

            # Bounty Seeker (Draven takedowns)
            has_draven = any("Draven" in u.champion_id for u in winner.board.values())
            if has_draven:
                winner.draven_bounty_progress += winner_kills
                if winner.draven_bounty_progress >= 12:
                    winner.add_gold(8)
                    if component_ids:
                        winner.add_item(r.choice(component_ids))
                    winner.draven_bounty_progress = 0

            # Fae (Golden Pixies after 7 pixies)
            if winner_traits.get("Fae", 0) >= 1:
                winner.fae_pixies += 1
                if winner_traits.get("Fae", 0) >= 2 and winner.fae_pixies >= 7:
                    winner.add_gold(2)

            # Rival (Rengar takedowns)
            has_rengar = any("Rengar" in u.champion_id for u in winner.board.values())
            if has_rengar:
                winner.rengar_takedowns += winner_kills
                if winner.rengar_takedowns >= 10:
                    winner.add_gold(5)
                    winner.rengar_takedowns = 0

            # Inferno (Shop Ignition for next round: roll 1 tier higher)
            if winner_traits.get("Inferno", 0) >= 2:
                winner.ignited_shop_slots = [0, 1]

            # Riftbeast Combats Counter
            if winner_traits.get("Riftbeast", 0) >= 2:
                winner.combats_since_overrun += 1

            # -------------------------------------------------------------
            # Loser Trait & Item Resolution
            # -------------------------------------------------------------
            if loser.alive:
                loser_traits = loser.get_active_traits()

                # Coven (Loser: Essence per kill + bonus essence per loss)
                if loser_traits.get("Coven", 0) >= 1:
                    loser.coven_essence += loser_kills + 2
                    if loser.coven_essence >= 40:
                        cashout_gold = 4 + (loser.coven_essence // 10)
                        loser.add_gold(cashout_gold)
                        if loser.coven_essence >= 80 and component_ids:
                            loser.add_item(r.choice(component_ids))
                        loser.coven_essence = 0

                # Bounty Seeker (Draven on loser board)
                has_draven_loser = any("Draven" in u.champion_id for u in loser.board.values())
                if has_draven_loser:
                    loser.draven_bounty_progress += loser_kills
                    if loser.draven_bounty_progress >= 12:
                        loser.add_gold(8)
                        if component_ids:
                            loser.add_item(r.choice(component_ids))
                        loser.draven_bounty_progress = 0

                # Fae on loser board
                if loser_traits.get("Fae", 0) >= 1:
                    loser.fae_pixies += 1
                    if loser_traits.get("Fae", 0) >= 2 and loser.fae_pixies >= 7:
                        loser.add_gold(2)

                # Rival on loser board
                has_rengar_loser = any("Rengar" in u.champion_id for u in loser.board.values())
                if has_rengar_loser:
                    loser.rengar_takedowns += loser_kills
                    if loser.rengar_takedowns >= 10:
                        loser.add_gold(5)
                        loser.rengar_takedowns = 0

                # Inferno on loser board
                if loser_traits.get("Inferno", 0) >= 2:
                    loser.ignited_shop_slots = [0, 1]

                # Riftbeast Combats Counter
                if loser_traits.get("Riftbeast", 0) >= 2:
                    loser.combats_since_overrun += 1
