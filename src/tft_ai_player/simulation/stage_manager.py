"""Stage timeline, round progression, economy schedules, and PvE/Carousel events."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import StrEnum

from tft_ai_player.simulation.augments import AugmentDef, AugmentManager
from tft_ai_player.simulation.config import STANDARD_COMPONENTS, SetData
from tft_ai_player.simulation.models import ChampionInstance, ChampionPool, Player
from tft_ai_player.simulation.sets.set18.mechanics import Set18MechanicsHandler


class RoundType(StrEnum):
    """Classification of TFT round phases."""

    PVP = "PVP"
    PVE = "PVE"
    CAROUSEL = "CAROUSEL"


# Stage-dependent carousel cost distribution weights
CAROUSEL_STAGE_COST_WEIGHTS: dict[int, dict[int, float]] = {
    1: {1: 0.65, 2: 0.35},                    # Stage 1-1: 1-cost and 2-cost only
    2: {1: 0.20, 2: 0.50, 3: 0.30},          # Stage 2-4: 1-cost, 2-cost, 3-cost
    3: {2: 0.25, 3: 0.55, 4: 0.20},          # Stage 3-4: 2-cost, 3-cost, 4-cost
    4: {3: 0.25, 4: 0.60, 5: 0.15},          # Stage 4-4: 3-cost, 4-cost, 5-cost
    5: {4: 0.50, 5: 0.50},                    # Stage 5-4: 4-cost, 5-cost
    6: {4: 0.30, 5: 0.70},                    # Stage 6-4+: 4-cost, 5-cost
}


@dataclass(frozen=True, slots=True)
class RoundInfo:
    """Detailed metadata for a specific game round."""

    stage: int
    round_in_stage: int
    round_type: RoundType
    stage_str: str

    @property
    def is_pvp(self) -> bool:
        """True if the round is a PvP battle."""
        return self.round_type == RoundType.PVP

    @property
    def is_pve(self) -> bool:
        """True if the round is a PvE neutral monster round."""
        return self.round_type == RoundType.PVE

    @property
    def is_carousel(self) -> bool:
        """True if the round is a shared draft carousel."""
        return self.round_type == RoundType.CAROUSEL


class StageManager:
    """Orchestrates TFT round transitions, income, PvE drops, carousels, and augment/loot events."""

    def __init__(self, set_data: SetData, augment_manager: AugmentManager | None = None) -> None:
        self.set_data = set_data
        self.augment_manager: AugmentManager = augment_manager or AugmentManager()
        self.stage: int = 1
        self.round_in_stage: int = 1
        self.total_rounds_elapsed: int = 0

    def get_round_type(self, stage: int, round_in_stage: int) -> RoundType:
        """Determine round classification by stage timeline."""
        if stage == 1:
            if round_in_stage == 1:
                return RoundType.CAROUSEL
            return RoundType.PVE
        if round_in_stage == 4:
            return RoundType.CAROUSEL
        if round_in_stage == 7:
            return RoundType.PVE
        return RoundType.PVP

    @property
    def stage_str(self) -> str:
        """String representation of current round stage (e.g. '2-1')."""
        return f"{self.stage}-{self.round_in_stage}"

    def get_current_round_info(self) -> RoundInfo:
        """Return metadata for current stage and round."""
        rtype = self.get_round_type(self.stage, self.round_in_stage)
        return RoundInfo(
            stage=self.stage,
            round_in_stage=self.round_in_stage,
            round_type=rtype,
            stage_str=self.stage_str,
        )

    def advance_round(self) -> RoundInfo:
        """Advance game clock to the next round."""
        self.total_rounds_elapsed += 1

        if self.stage == 1:
            if self.round_in_stage < 4:
                self.round_in_stage += 1
            else:
                self.stage = 2
                self.round_in_stage = 1
        else:
            if self.round_in_stage < 7:
                self.round_in_stage += 1
            else:
                self.stage += 1
                self.round_in_stage = 1

        return self.get_current_round_info()

    def get_passive_gold(self, stage_str: str) -> int:
        """Determine base passive income for the current round."""
        sched = self.set_data.passive_gold_schedule
        if stage_str in sched:
            return sched[stage_str]
        return sched.get("default", 5)

    def get_base_stage_damage(self, stage: int) -> int:
        """Look up standard player base damage for losses in the current stage."""
        dmg_map = self.set_data.stage_base_damage
        return dmg_map.get(stage, dmg_map.get(7, 17))

    def execute_round_start(
        self,
        players: list[Player],
        pool: ChampionPool,
        rng: random.Random | None = None,
    ) -> None:
        """Apply augment offerings, passive XP/income, trait mechanics, and shop refreshes."""
        r = rng or random
        rinfo = self.get_current_round_info()
        base_gold = self.get_passive_gold(rinfo.stage_str)

        # 0. Augment Selection Round (2-1, 3-2, 4-2)
        if self.augment_manager.is_augment_round(rinfo.stage_str):
            lobby_tier = self.augment_manager.get_round_augment_tier(rinfo.stage_str, rng=r)
            for player in players:
                if not player.alive:
                    continue
                choices = self.augment_manager.generate_augment_choices(
                    player, rinfo.stage_str, rng=r, target_tier=lobby_tier
                )
                if choices:
                    active_traits = set(player.get_active_traits().keys())
                    best_choice = choices[0]
                    for c in choices:
                        if any(t in active_traits for t in c.associated_traits):
                            best_choice = c
                            break
                        if c.instant_gold > best_choice.instant_gold:
                            best_choice = c
                    self.augment_manager.apply_augment(
                        player, best_choice.augment_id, pool=pool, set_data=self.set_data, rng=r
                    )

        for player in players:
            if not player.alive:
                continue

            # 1. Passive XP (+2 XP per round, except stage 1-1 / 1-2)
            if not (rinfo.stage == 1 and rinfo.round_in_stage <= 2):
                player.exp += 2
                while player.level < self.set_data.max_level:
                    req_exp = self.set_data.level_exp.get(player.level, 0)
                    if req_exp > 0 and player.exp >= req_exp:
                        player.exp -= req_exp
                        player.level += 1
                    else:
                        break

            # 2. Augment Passive Income & Per-Round XP
            self.augment_manager.execute_round_start_augments(player, self.set_data)

            # 3. Set-Specific Trait Round-Start Mechanics
            candidates_pool: list[str] | None = None
            if "18" in self.set_data.set_name:
                candidates_pool = Set18MechanicsHandler.apply_round_start(
                    player=player,
                    set_data=self.set_data,
                    pool=pool,
                    rng=r,
                )

            # 4. Standard Gold Income (Base + Interest + Streak)
            interest_gold = player.calculate_interest_gold()
            streak_gold = player.calculate_streak_gold()
            total_income = base_gold + interest_gold + streak_gold
            player.add_gold(total_income)

            # 5. Shop Refresh
            player.shop.refresh(
                level=player.level,
                pool=pool,
                set_data=self.set_data,
                rng=r,
                ignited_slots=player.ignited_shop_slots,
                candidates_pool=candidates_pool,
            )
            player.ignited_shop_slots = []  # Consumed

    def handle_combat_loot_and_traits(
        self,
        players: list[Player],
        combat_results: list[Any],
        pool: ChampionPool,
        rng: random.Random | None = None,
    ) -> None:
        """Resolve set-specific combat loot, takedown bounties, and essence."""
        r = rng or random

        if "18" in self.set_data.set_name:
            Set18MechanicsHandler.apply_combat_loot_and_traits(
                players=players,
                combat_results=combat_results,
                pool=pool,
                set_data=self.set_data,
                rng=r,
            )

    def handle_pve_loot(
        self,
        players: list[Player],
        rng: random.Random | None = None,
    ) -> None:
        """Distribute item components and bonus gold on neutral rounds in accordance with official TFT drops."""
        r = rng or random
        rinfo = self.get_current_round_info()
        if not rinfo.is_pve:
            return

        component_ids = [cid for cid, _ in STANDARD_COMPONENTS if cid != "TFT_Item_Spatula"]
        completed_items = list(self.set_data.recipes.values())

        for player in players:
            if not player.alive:
                continue

            if rinfo.stage == 1:
                # Stage 1 Minions (1-2, 1-3, 1-4)
                if rinfo.round_in_stage == 2:
                    # 1-2 (2 Minions): 80% 1 component, 20% 2 gold
                    if r.random() < 0.80:
                        player.add_item(r.choice(component_ids))
                    else:
                        player.add_gold(2)
                elif rinfo.round_in_stage == 3:
                    # 1-3 (3 Minions): 70% 1 component, 30% 2 gold
                    if r.random() < 0.70:
                        player.add_item(r.choice(component_ids))
                    else:
                        player.add_gold(2)
                elif rinfo.round_in_stage == 4:
                    # 1-4 (4 Minions): 50% 1 component + 1g, 50% 3 gold
                    if r.random() < 0.50:
                        player.add_item(r.choice(component_ids))
                        player.add_gold(1)
                    else:
                        player.add_gold(3)

            elif rinfo.round_in_stage == 7:
                # Neutral monster stage boss rounds
                if rinfo.stage == 2:
                    # 2-7 Krugs: 1 component + 2 gold
                    player.add_item(r.choice(component_ids))
                    player.add_gold(2)
                elif rinfo.stage == 3:
                    # 3-7 Wolves: 1-2 components + 2 gold
                    player.add_item(r.choice(component_ids))
                    if r.random() < 0.50:
                        player.add_item(r.choice(component_ids))
                    player.add_gold(2)
                elif rinfo.stage == 4:
                    # 4-7 Raptors: 2 components + 2 gold
                    player.add_item(r.choice(component_ids))
                    player.add_item(r.choice(component_ids))
                    player.add_gold(2)
                elif rinfo.stage == 5:
                    # 5-7 Dragon / Rift Boss: 1 completed item + 3 gold
                    if completed_items:
                        player.add_item(r.choice(completed_items))
                    else:
                        player.add_item(r.choice(component_ids))
                    player.add_gold(3)
                else:
                    # 6-7+ Elder Dragon / Boss: 1 completed item + 5 gold
                    if completed_items:
                        player.add_item(r.choice(completed_items))
                    else:
                        player.add_item(r.choice(component_ids))
                    player.add_gold(5)

    def handle_carousel(
        self,
        players: list[Player],
        pool: ChampionPool,
        rng: random.Random | None = None,
    ) -> list[dict[str, Any]]:
        """Simulate shared draft carousel: generates (champion, item) pairs and drafts to players in HP order."""
        r = rng or random
        rinfo = self.get_current_round_info()
        if not rinfo.is_carousel:
            return []

        alive_players = [p for p in players if p.alive]
        if not alive_players:
            return []

        # 1. Determine stage cost distribution
        stage_key = min(rinfo.stage, 6)
        cost_weights = CAROUSEL_STAGE_COST_WEIGHTS.get(stage_key, {4: 0.5, 5: 0.5})

        # 2. Determine available items pool for this stage
        basic_components = [cid for cid, _ in STANDARD_COMPONENTS if cid != "TFT_Item_Spatula"]
        completed_items = list(self.set_data.recipes.values())

        # 3. Generate Carousel Ring Offerings (N + 1 offerings)
        num_offerings = len(alive_players) + 1
        offerings: list[tuple[str, int, str]] = []

        for _ in range(num_offerings):
            # Sample cost tier based on stage weights
            costs = list(cost_weights.keys())
            weights = list(cost_weights.values())
            chosen_cost = r.choices(costs, weights=weights, k=1)[0]

            # Find matching champions in this cost tier
            champ_candidates = [
                cid for cid, cdef in self.set_data.champions.items()
                if cdef.cost == chosen_cost
            ]
            if not champ_candidates:
                champ_candidates = list(self.set_data.champions.keys())

            chosen_champ = r.choice(champ_candidates)

            # Sample item (Stage 1-4: components; Stage 5+: components or completed items)
            if rinfo.stage <= 4 or not completed_items:
                if r.random() < 0.05:
                    chosen_item = "TFT_Item_Spatula"
                else:
                    chosen_item = r.choice(basic_components)
            else:
                if r.random() < 0.50:
                    chosen_item = r.choice(completed_items)
                else:
                    chosen_item = r.choice(basic_components)

            # Decrement from pool if available
            if pool.counts.get(chosen_champ, 0) > 0:
                pool.counts[chosen_champ] -= 1

            offerings.append((chosen_champ, chosen_cost, chosen_item))

        # 4. Draft Order: Lowest HP picks first (In stage 1-1, random order)
        if rinfo.stage == 1 and rinfo.round_in_stage == 1:
            draft_order = list(alive_players)
            r.shuffle(draft_order)
        else:
            draft_order = sorted(alive_players, key=lambda p: (p.health, p.player_id))

        draft_events: list[dict[str, Any]] = []

        # 5. Assign champion + item to each player
        for player in draft_order:
            if not offerings:
                break
            champ_id, cost, item_id = offerings.pop(0)

            # Create champion instance holding the item
            champ_instance = ChampionInstance(
                champion_id=champ_id,
                cost=cost,
                star_level=1,
                items=[item_id],
            )

            # Add to player bench/board with star-up check
            player.add_champion_to_bench(champ_instance, pool, allow_board_overflow=True)

            draft_events.append({
                "player_id": player.player_id,
                "champion_id": champ_id,
                "cost": cost,
                "item_id": item_id,
            })

        return draft_events
