"""High-resolution combat density and domain feature extraction for TFT round winner prediction."""

from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# =============================================================================
# SET 17 DOMAIN KNOWLEDGE: COSTS, TRAITS, ROLES, ITEMS
# =============================================================================
CHAMP_BASE_COSTS: dict[str, int] = {
    # 1-Cost
    "TFT17_Aatrox": 1, "TFT17_Pantheon": 1, "TFT17_Milio": 1, "TFT17_Jax": 1,
    "TFT17_Caitlyn": 1, "TFT17_Chogath": 1, "TFT17_Blitzcrank": 1, "TFT17_Poppy": 1,
    "TFT17_Zoe": 1, "TFT17_Talon": 1, "TFT17_Lulu": 1, "TFT17_Sona": 1,
    # 2-Cost
    "TFT17_Maokai": 2, "TFT17_Lissandra": 2, "TFT17_Briar": 2, "TFT17_RekSai": 2,
    "TFT17_Nasus": 2, "TFT17_Leona": 2, "TFT17_TwistedFate": 2, "TFT17_Rhaast": 2,
    "TFT17_Shen": 2, "TFT17_Kindred": 2, "TFT17_Gragas": 2, "TFT17_Teemo": 2,
    "TFT17_Graves": 2, "TFT17_Pyke": 2,
    # 3-Cost
    "TFT17_TahmKench": 3, "TFT17_Akali": 3, "TFT17_Illaoi": 3, "TFT17_Nunu": 3,
    "TFT17_Gwen": 3, "TFT17_Ezreal": 3, "TFT17_Riven": 3, "TFT17_Urgot": 3,
    "TFT17_Rammus": 3, "TFT17_Gnar": 3, "TFT17_Veigar": 3, "TFT17_Bard": 3,
    "TFT17_Fizz": 3, "TFT17_Vex": 3, "TFT17_Xayah": 3,
    # 4-Cost
    "TFT17_Mordekaiser": 4, "TFT17_Belveth": 4, "TFT17_Ornn": 4, "TFT17_Morgana": 4,
    "TFT17_Karma": 4, "TFT17_Jinx": 4, "TFT17_Samira": 4, "TFT17_Fiora": 4,
    "TFT17_Kaisa": 4, "TFT17_Jhin": 4, "TFT17_Corki": 4, "TFT17_MissFortune": 4,
    "TFT17_Diana": 4,
    # 5-Cost Legendaries
    "TFT17_Galio": 5, "TFT17_AurelionSol": 5, "TFT17_Viktor": 5, "TFT17_Aurora": 5,
    "TFT17_Leblanc": 5, "TFT17_MasterYi": 5, "TFT17_Zed": 5,
}

# Role Categorization
AP_CARRIES: set[str] = {
    "TFT17_Karma", "TFT17_Lissandra", "TFT17_Viktor", "TFT17_Veigar", "TFT17_Teemo",
    "TFT17_Zoe", "TFT17_Aurora", "TFT17_Vex", "TFT17_TwistedFate", "TFT17_Morgana",
    "TFT17_AurelionSol", "TFT17_Leblanc", "TFT17_Diana",
}

AD_CARRIES: set[str] = {
    "TFT17_Jinx", "TFT17_Belveth", "TFT17_Caitlyn", "TFT17_Ezreal", "TFT17_Samira",
    "TFT17_Fiora", "TFT17_Kaisa", "TFT17_Jhin", "TFT17_Corki", "TFT17_MasterYi",
    "TFT17_MissFortune", "TFT17_Kindred", "TFT17_Graves", "TFT17_Zed",
}

MAIN_TANKS: set[str] = {
    "TFT17_Nasus", "TFT17_Ornn", "TFT17_Maokai", "TFT17_Pantheon", "TFT17_Illaoi",
    "TFT17_Galio", "TFT17_Jax", "TFT17_Poppy", "TFT17_Chogath", "TFT17_Blitzcrank",
    "TFT17_TahmKench", "TFT17_Shen", "TFT17_Leona", "TFT17_Rammus", "TFT17_Gragas",
}

AP_ITEMS: set[str] = {
    "TFT_Item_JeweledGauntlet", "TFT_Item_RabadonsDeathcap", "TFT_Item_ArchangelsStaff",
    "TFT_Item_HextechGunblade", "TFT_Item_Crownguard", "TFT_Item_StatikkShiv",
    "TFT_Item_Morellonomicon", "TFT_Item_Leviathan",
}

AD_ITEMS: set[str] = {
    "TFT_Item_InfinityEdge", "TFT_Item_Deathblade", "TFT_Item_LastWhisper",
    "TFT_Item_KrakenSlayer", "TFT_Item_SteraksGage", "TFT_Item_Bloodthirster",
    "TFT_Item_GuinsoosRageblade", "TFT_Item_RapidFireCannon", "TFT_Item_MadredsBloodrazor",
}

TANK_ITEMS: set[str] = {
    "TFT_Item_GargoyleStoneplate", "TFT_Item_WarmogsArmor", "TFT_Item_FrozenHeart",
    "TFT_Item_DragonsClaw", "TFT_Item_BrambleVest", "TFT_Item_RedBuff",
    "TFT_Item_NightHarvester", "TFT_Item_SpiritVisage",
}

ANTI_HEAL_ITEMS: set[str] = {
    "TFT_Item_Morellonomicon", "TFT_Item_RedBuff", "TFT_Item_SunfireCape",
}

SHRED_ITEMS: set[str] = {
    "TFT_Item_LastWhisper", "TFT_Item_StatikkShiv", "TFT_Item_IonicSpark", "TFT_Item_Evenshroud",
}

MANA_ITEMS: set[str] = {
    "TFT_Item_SpearOfShojin", "TFT_Item_BlueBuff", "TFT_Item_AdaptiveHelm",
}

# Set 17 Traits & Synergies
SET17_TRAIT_CHAMPIONS: dict[str, list[str]] = {
    "Bastion": ["TFT17_Jax", "TFT17_Maokai", "TFT17_Poppy", "TFT17_Shen", "TFT17_Diana", "TFT17_Illaoi"],
    "Brawler": ["TFT17_Aatrox", "TFT17_Briar", "TFT17_RekSai", "TFT17_Nunu", "TFT17_Gragas", "TFT17_Urgot", "TFT17_Chogath"],
    "Vanguard": ["TFT17_Mordekaiser", "TFT17_Nasus", "TFT17_Blitzcrank", "TFT17_Leona", "TFT17_Poppy", "TFT17_Galio"],
    "Sniper": ["TFT17_Caitlyn", "TFT17_Kindred", "TFT17_Jinx", "TFT17_Xayah", "TFT17_Jhin", "TFT17_Corki"],
    "Stargazer": ["TFT17_Diana", "TFT17_TwistedFate", "TFT17_Leona", "TFT17_Teemo", "TFT17_Karma", "TFT17_Nami"],
    "SpaceGroove": ["TFT17_Blitzcrank", "TFT17_Lissandra", "TFT17_Nunu", "TFT17_Samira", "TFT17_Teemo", "TFT17_Ornn", "TFT17_Gragas"],
    "DarkStar": ["TFT17_Mordekaiser", "TFT17_Karma", "TFT17_Jhin", "TFT17_Chogath", "TFT17_DarkStar_FakeUnit"],
    "Rogue": ["TFT17_Akali", "TFT17_Talon", "TFT17_Fizz", "TFT17_Pyke", "TFT17_Zed", "TFT17_Leblanc"],
    "PsyOps": ["TFT17_Sona", "TFT17_Pyke", "TFT17_Viktor", "TFT17_Samira", "TFT17_Shen", "TFT17_Ezreal"],
    "AnimaTech": ["TFT17_Riven", "TFT17_Vex", "TFT17_Jinx", "TFT17_MissFortune", "TFT17_Aurora"],
    "Marauder": ["TFT17_Pantheon", "TFT17_Aatrox", "TFT17_Graves", "TFT17_Rhaast"],
    "Sheperd": ["TFT17_Milio", "TFT17_Kindred", "TFT17_Bard", "TFT17_IvernMinion"],
    "Primordian": ["TFT17_Belveth", "TFT17_Briar", "TFT17_RekSai", "TFT17_Aatrox", "TFT17_Morgana"],
    "Arbiter": ["TFT17_Morgana", "TFT17_Galio", "TFT17_Leona"],
    "Voyager": ["TFT17_Ezreal", "TFT17_Corki", "TFT17_Bard", "TFT17_Aurora"],
    "Challenger": ["TFT17_Belveth", "TFT17_Fiora", "TFT17_Kaisa", "TFT17_MasterYi", "TFT17_Samira"],
    "Timebreaker": ["TFT17_Diana", "TFT17_Viktor"],
    "Meeple": ["TFT17_Poppy", "TFT17_Teemo", "TFT17_Veigar", "TFT17_Lulu", "TFT17_Gnar", "TFT17_Fizz", "TFT17_Vex"],
    "NOVA": ["TFT17_Ornn", "TFT17_AurelionSol", "TFT17_Viktor", "TFT17_Shen", "TFT17_Zoe"],
}

SET17_TRAIT_THRESHOLDS: dict[str, list[int]] = {
    "Bastion": [2, 4, 6], "Brawler": [2, 4, 6], "Vanguard": [2, 4, 6],
    "Sniper": [2, 4, 6], "Stargazer": [3, 5, 7], "SpaceGroove": [3, 5, 7],
    "DarkStar": [3, 6, 9], "Rogue": [2, 4, 6], "PsyOps": [3, 6, 9],
    "AnimaTech": [3, 5, 7], "Marauder": [2, 4, 6], "Sheperd": [2, 4, 6],
    "Primordian": [3, 5, 7], "Arbiter": [2, 4, 6], "Voyager": [2, 4],
    "Challenger": [2, 4, 6], "Timebreaker": [2, 4], "Meeple": [3, 5, 7],
    "NOVA": [2, 4, 6],
}

CHAMP_TO_TRAITS: dict[str, list[str]] = {}
for trait_name, champs in SET17_TRAIT_CHAMPIONS.items():
    for c in champs:
        CHAMP_TO_TRAITS.setdefault(c, []).append(trait_name)

ALL_SET17_TRAITS: list[str] = sorted(list(SET17_TRAIT_CHAMPIONS.keys()))


class TFTBoardFeatureExtractor(BaseEstimator, TransformerMixin):
    """Deep domain-engineered feature extraction for TFT round winner probability models."""

    def __init__(
        self,
        *,
        include_champions: bool = True,
        include_items: bool = True,
        include_augments: bool = True,
        include_placement: bool = True,
        include_traits: bool = True,
        include_absolute: bool = True,
        min_champ_freq: int = 8,
        min_item_freq: int = 8,
        min_aug_freq: int = 8,
    ) -> None:
        self.include_champions = include_champions
        self.include_items = include_items
        self.include_augments = include_augments
        self.include_placement = include_placement
        self.include_traits = include_traits
        self.include_absolute = include_absolute
        self.min_champ_freq = min_champ_freq
        self.min_item_freq = min_item_freq
        self.min_aug_freq = min_aug_freq

        self.champions_vocab_: list[str] = []
        self.items_vocab_: list[str] = []
        self.augments_vocab_: list[str] = []
        self.feature_names_: list[str] = []

    def fit(self, X: Any, y: Any = None) -> TFTBoardFeatureExtractor:
        """Learn vocabularies of champions, items, and augments from training data."""
        df = self._ensure_dataframe(X)

        champ_counts: Counter[str] = Counter()
        item_counts: Counter[str] = Counter()
        aug_counts: Counter[str] = Counter()

        for _, row in df.iterrows():
            board_state = self._parse_input_state(row.get("input_state_json", {}))

            for unit in board_state.get("focal_board", []):
                u_name = unit.get("unit")
                if u_name:
                    champ_counts[u_name] += 1
                for item in unit.get("items", []):
                    if item:
                        item_counts[item] += 1

            for unit in board_state.get("opponent_board", []):
                u_name = unit.get("unit")
                if u_name:
                    champ_counts[u_name] += 1
                for item in unit.get("items", []):
                    if item:
                        item_counts[item] += 1

            for aug in self._parse_augments(row.get("focal_augments")):
                aug_counts[aug] += 1
            for aug in self._parse_augments(row.get("opponent_augments")):
                aug_counts[aug] += 1

        if self.include_champions:
            self.champions_vocab_ = sorted(
                [c for c, count in champ_counts.items() if count >= self.min_champ_freq]
            )
        else:
            self.champions_vocab_ = []

        if self.include_items:
            self.items_vocab_ = sorted(
                [i for i, count in item_counts.items() if count >= self.min_item_freq]
            )
        else:
            self.items_vocab_ = []

        if self.include_augments:
            self.augments_vocab_ = sorted(
                [a for a, count in aug_counts.items() if count >= self.min_aug_freq]
            )
        else:
            self.augments_vocab_ = []

        self.feature_names_ = self._build_feature_names()
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Transform raw round observations into dense feature matrix."""
        df = self._ensure_dataframe(X)
        n_samples = len(df)
        n_features = len(self.feature_names_)

        features = np.zeros((n_samples, n_features), dtype=np.float32)

        champ_to_idx = {c: i for i, c in enumerate(self.champions_vocab_)}
        item_to_idx = {item: i for i, item in enumerate(self.items_vocab_)}
        aug_to_idx = {aug: i for i, aug in enumerate(self.augments_vocab_)}

        # Offsets
        base_dim = len(self._base_feature_names())
        num_champs = len(self.champions_vocab_)
        num_items = len(self.items_vocab_)

        champ_focal_offset = base_dim
        champ_opp_offset = champ_focal_offset + num_champs if self.include_absolute else champ_focal_offset
        champ_diff_offset = champ_opp_offset + num_champs if self.include_absolute else champ_focal_offset
        champ_total_dim = num_champs * 3 if self.include_absolute else num_champs

        item_focal_offset = base_dim + champ_total_dim
        item_opp_offset = item_focal_offset + num_items if self.include_absolute else item_focal_offset
        item_diff_offset = item_opp_offset + num_items if self.include_absolute else item_focal_offset
        item_total_dim = num_items * 3 if self.include_absolute else num_items

        aug_offset = base_dim + champ_total_dim + item_total_dim

        for row_idx, (_, row) in enumerate(df.iterrows()):
            # 1. Context & Status
            stage_num, round_num, total_round, is_aug_round = self._parse_round_stage(
                row.get("round_stage")
            )
            f_lvl = self._safe_num(row.get("focal_level"), default=1.0)
            o_lvl = self._safe_num(row.get("opponent_level"), default=1.0)
            diff_lvl = f_lvl - o_lvl
            f_hp = self._safe_num(row.get("focal_health"), default=100.0)
            o_hp = self._safe_num(row.get("opponent_health"), default=100.0)
            diff_hp = f_hp - o_hp
            f_gold = self._safe_num(row.get("focal_gold"), default=0.0)

            # 2. Parse Boards
            board_state = self._parse_input_state(row.get("input_state_json", {}))
            focal_board = board_state.get("focal_board", [])
            opponent_board = board_state.get("opponent_board", [])

            # Focal Board Summaries & Deep Combat Metrics
            f_units = len(focal_board)
            f_stars_total = 0
            f_star1 = 0
            f_star2 = 0
            f_star3 = 0
            f_items_total = 0
            f_carry_3item = 0
            f_unit_2item = 0
            f_unit_1item = 0
            f_gold_val = 0.0
            f_4cost_2star = 0
            f_5cost_cnt = 0
            f_synergy_item_score = 0.0
            f_unit_powers: list[float] = []
            f_anti_heal = 0
            f_shred = 0
            f_mana = 0

            f_items_on_2star = 0
            f_items_on_1star = 0
            f_2star_3item = 0
            f_2star_2item = 0
            f_3star_3item = 0
            f_tank_density = 0.0
            f_dps_density = 0.0

            for unit in focal_board:
                tier = int(unit.get("tier", 1) or 1)
                u_name = str(unit.get("unit", "") or "")
                cost = CHAMP_BASE_COSTS.get(u_name, 2)
                f_gold_val += cost * (3 ** (tier - 1))
                if cost == 4 and tier >= 2:
                    f_4cost_2star += 1
                if cost == 5:
                    f_5cost_cnt += 1

                f_stars_total += tier
                if tier == 1:
                    f_star1 += 1
                elif tier == 2:
                    f_star2 += 1
                elif tier >= 3:
                    f_star3 += 1

                items = unit.get("items", []) or []
                n_items = len(items)
                f_items_total += n_items
                if n_items >= 3:
                    f_carry_3item += 1
                elif n_items == 2:
                    f_unit_2item += 1
                elif n_items == 1:
                    f_unit_1item += 1

                if tier >= 2:
                    f_items_on_2star += n_items
                    if n_items >= 3:
                        f_2star_3item += 1
                    elif n_items == 2:
                        f_2star_2item += 1
                else:
                    f_items_on_1star += n_items

                if tier >= 3 and n_items >= 3:
                    f_3star_3item += 1

                # Role synergy items
                is_ap = u_name in AP_CARRIES
                is_ad = u_name in AD_CARRIES
                is_tank = u_name in MAIN_TANKS
                n_syn_items = 0
                for it in items:
                    if is_ap and it in AP_ITEMS:
                        f_synergy_item_score += 1.5
                        n_syn_items += 1
                    elif is_ad and it in AD_ITEMS:
                        f_synergy_item_score += 1.5
                        n_syn_items += 1
                    elif is_tank and it in TANK_ITEMS:
                        f_synergy_item_score += 1.5
                        n_syn_items += 1
                    else:
                        f_synergy_item_score += 0.5

                    if it in ANTI_HEAL_ITEMS:
                        f_anti_heal += 1
                    if it in SHRED_ITEMS:
                        f_shred += 1
                    if it in MANA_ITEMS:
                        f_mana += 1

                u_pow = cost * (tier ** 2.2) * (1.0 + 0.6 * n_items + 0.4 * n_syn_items)
                f_unit_powers.append(u_pow)

                if is_tank:
                    f_tank_density += tier * (1.0 + n_items * 0.8)
                if is_ap or is_ad:
                    f_dps_density += u_pow

                if self.include_champions and u_name in champ_to_idx:
                    c_i = champ_to_idx[u_name]
                    if self.include_absolute:
                        features[row_idx, champ_focal_offset + c_i] += tier
                    features[row_idx, champ_diff_offset + c_i] += tier

                if self.include_items:
                    for it in items:
                        if it in item_to_idx:
                            it_i = item_to_idx[it]
                            if self.include_absolute:
                                features[row_idx, item_focal_offset + it_i] += 1.0
                            features[row_idx, item_diff_offset + it_i] += 1.0

            # Opponent Board Summaries
            o_units = len(opponent_board)
            o_stars_total = 0
            o_star1 = 0
            o_star2 = 0
            o_star3 = 0
            o_items_total = 0
            o_carry_3item = 0
            o_unit_2item = 0
            o_unit_1item = 0
            o_gold_val = 0.0
            o_4cost_2star = 0
            o_5cost_cnt = 0
            o_synergy_item_score = 0.0
            o_unit_powers: list[float] = []
            o_anti_heal = 0
            o_shred = 0
            o_mana = 0

            o_items_on_2star = 0
            o_items_on_1star = 0
            o_2star_3item = 0
            o_2star_2item = 0
            o_3star_3item = 0
            o_tank_density = 0.0
            o_dps_density = 0.0

            for unit in opponent_board:
                tier = int(unit.get("tier", 1) or 1)
                u_name = str(unit.get("unit", "") or "")
                cost = CHAMP_BASE_COSTS.get(u_name, 2)
                o_gold_val += cost * (3 ** (tier - 1))
                if cost == 4 and tier >= 2:
                    o_4cost_2star += 1
                if cost == 5:
                    o_5cost_cnt += 1

                o_stars_total += tier
                if tier == 1:
                    o_star1 += 1
                elif tier == 2:
                    o_star2 += 1
                elif tier >= 3:
                    o_star3 += 1

                items = unit.get("items", []) or []
                n_items = len(items)
                o_items_total += n_items
                if n_items >= 3:
                    o_carry_3item += 1
                elif n_items == 2:
                    o_unit_2item += 1
                elif n_items == 1:
                    o_unit_1item += 1

                if tier >= 2:
                    o_items_on_2star += n_items
                    if n_items >= 3:
                        o_2star_3item += 1
                    elif n_items == 2:
                        o_2star_2item += 1
                else:
                    o_items_on_1star += n_items

                if tier >= 3 and n_items >= 3:
                    o_3star_3item += 1

                is_ap = u_name in AP_CARRIES
                is_ad = u_name in AD_CARRIES
                is_tank = u_name in MAIN_TANKS
                n_syn_items = 0
                for it in items:
                    if is_ap and it in AP_ITEMS:
                        o_synergy_item_score += 1.5
                        n_syn_items += 1
                    elif is_ad and it in AD_ITEMS:
                        o_synergy_item_score += 1.5
                        n_syn_items += 1
                    elif is_tank and it in TANK_ITEMS:
                        o_synergy_item_score += 1.5
                        n_syn_items += 1
                    else:
                        o_synergy_item_score += 0.5

                    if it in ANTI_HEAL_ITEMS:
                        o_anti_heal += 1
                    if it in SHRED_ITEMS:
                        o_shred += 1
                    if it in MANA_ITEMS:
                        o_mana += 1

                u_pow = cost * (tier ** 2.2) * (1.0 + 0.6 * n_items + 0.4 * n_syn_items)
                o_unit_powers.append(u_pow)

                if is_tank:
                    o_tank_density += tier * (1.0 + n_items * 0.8)
                if is_ap or is_ad:
                    o_dps_density += u_pow

                if self.include_champions and u_name in champ_to_idx:
                    c_i = champ_to_idx[u_name]
                    if self.include_absolute:
                        features[row_idx, champ_opp_offset + c_i] += tier
                    features[row_idx, champ_diff_offset + c_i] -= tier

                if self.include_items:
                    for it in items:
                        if it in item_to_idx:
                            it_i = item_to_idx[it]
                            if self.include_absolute:
                                features[row_idx, item_opp_offset + it_i] += 1.0
                            features[row_idx, item_diff_offset + it_i] -= 1.0

            # Power metrics
            f_unit_powers_sorted = sorted(f_unit_powers, reverse=True)
            o_unit_powers_sorted = sorted(o_unit_powers, reverse=True)

            f_total_pow = sum(f_unit_powers)
            o_total_pow = sum(o_unit_powers)
            diff_total_pow = f_total_pow - o_total_pow

            f_top3_pow = sum(f_unit_powers_sorted[:3])
            o_top3_pow = sum(o_unit_powers_sorted[:3])
            diff_top3_pow = f_top3_pow - o_top3_pow

            f_carry_pow = f_unit_powers_sorted[0] if f_unit_powers_sorted else 0.0
            o_carry_pow = o_unit_powers_sorted[0] if o_unit_powers_sorted else 0.0
            diff_carry_pow = f_carry_pow - o_carry_pow

            # Differentials
            diff_units = f_units - o_units
            diff_stars_total = f_stars_total - o_stars_total
            diff_star1 = f_star1 - o_star1
            diff_star2 = f_star2 - o_star2
            diff_star3 = f_star3 - o_star3
            diff_items_total = f_items_total - o_items_total
            diff_carry_3item = f_carry_3item - o_carry_3item
            diff_unit_2item = f_unit_2item - o_unit_2item
            diff_unit_1item = f_unit_1item - o_unit_1item
            diff_gold_val = f_gold_val - o_gold_val
            diff_4cost_2star = f_4cost_2star - o_4cost_2star
            diff_5cost_cnt = f_5cost_cnt - o_5cost_cnt
            diff_synergy_item_score = f_synergy_item_score - o_synergy_item_score

            diff_items_on_2star = f_items_on_2star - o_items_on_2star
            diff_items_on_1star = f_items_on_1star - o_items_on_1star
            diff_2star_3item = f_2star_3item - o_2star_3item
            diff_2star_2item = f_2star_2item - o_2star_2item
            diff_3star_3item = f_3star_3item - o_3star_3item
            diff_tank_density = f_tank_density - o_tank_density
            diff_dps_density = f_dps_density - o_dps_density

            diff_anti_heal = f_anti_heal - o_anti_heal
            diff_shred = f_shred - o_shred
            diff_mana = f_mana - o_mana

            f_avg_stars = f_stars_total / max(f_units, 1)
            o_avg_stars = o_stars_total / max(o_units, 1)
            diff_avg_stars = f_avg_stars - o_avg_stars

            # Stage Cross Interactions
            stage_x_diff_pow = stage_num * diff_total_pow
            stage_x_diff_top3 = stage_num * diff_top3_pow
            stage_x_diff_gold = stage_num * diff_gold_val

            # 3. Augments
            f_augs = self._parse_augments(row.get("focal_augments"))
            o_augs = self._parse_augments(row.get("opponent_augments"))
            diff_augs_count = len(f_augs) - len(o_augs)

            if self.include_augments:
                for aug in f_augs:
                    if aug in aug_to_idx:
                        features[row_idx, aug_offset + aug_to_idx[aug]] += 1.0
                for aug in o_augs:
                    if aug in aug_to_idx:
                        features[row_idx, aug_offset + aug_to_idx[aug]] -= 1.0

            # 4. Placement & Tactical Geometry
            f_place = self._extract_placement_metrics(focal_board)
            o_place = self._extract_placement_metrics(opponent_board)

            diff_front_units = f_place["front_units"] - o_place["front_units"]
            diff_back_units = f_place["back_units"] - o_place["back_units"]
            diff_front_stars = f_place["front_stars"] - o_place["front_stars"]
            diff_back_stars = f_place["back_stars"] - o_place["back_stars"]
            diff_front_items = f_place["front_items"] - o_place["front_items"]
            diff_back_items = f_place["back_items"] - o_place["back_items"]
            diff_front_ratio = f_place["front_ratio"] - o_place["front_ratio"]

            carry_same_side = 1.0 if abs(f_place["carry_col"] - o_place["carry_col"]) <= 2 else 0.0
            carry_tank_dist = math.hypot(
                f_place["carry_row"] - f_place["tank_row"],
                f_place["carry_col"] - f_place["tank_col"],
            )

            # Tactical matchup threats
            f_cornered = 1.0 if (f_place["carry_row"] == 3.0 and f_place["carry_col"] in (0.0, 6.0)) else 0.0
            opp_has_blitz = 1.0 if any(u.get("unit") == "TFT17_Blitzcrank" for u in opponent_board) else 0.0
            f_blitz_risk = 1.0 if (f_cornered and opp_has_blitz) else 0.0

            # 5. Trait Synergies & Threshold Engine
            f_traits, f_tiers, f_act_cnt, f_max_tier, f_stat_power = self._extract_trait_metrics(focal_board)
            o_traits, o_tiers, o_act_cnt, o_max_tier, o_stat_power = self._extract_trait_metrics(opponent_board)

            diff_act_traits = f_act_cnt - o_act_cnt
            diff_max_tier = f_max_tier - o_max_tier
            diff_stat_power = f_stat_power - o_stat_power

            # Populate Base Vector
            base_values = [
                stage_num, round_num, total_round, is_aug_round,
                f_lvl, o_lvl, diff_lvl,
                f_hp, o_hp, diff_hp, f_gold,
                f_units, o_units, diff_units,
                f_stars_total, o_stars_total, diff_stars_total,
                f_star1, f_star2, f_star3,
                o_star1, o_star2, o_star3,
                diff_star1, diff_star2, diff_star3,
                f_items_total, o_items_total, diff_items_total,
                f_carry_3item, o_carry_3item, diff_carry_3item,
                f_unit_2item, o_unit_2item, diff_unit_2item,
                f_unit_1item, o_unit_1item, diff_unit_1item,
                f_avg_stars, o_avg_stars, diff_avg_stars,
                len(f_augs), len(o_augs), diff_augs_count,
                # Board Gold Value & High Tier Units
                f_gold_val, o_gold_val, diff_gold_val,
                f_4cost_2star, o_4cost_2star, diff_4cost_2star,
                f_5cost_cnt, o_5cost_cnt, diff_5cost_cnt,
                f_synergy_item_score, o_synergy_item_score, diff_synergy_item_score,
                # Item Efficiency & Star Allocation
                f_items_on_2star, o_items_on_2star, diff_items_on_2star,
                f_items_on_1star, o_items_on_1star, diff_items_on_1star,
                f_2star_3item, o_2star_3item, diff_2star_3item,
                f_2star_2item, o_2star_2item, diff_2star_2item,
                f_3star_3item, o_3star_3item, diff_3star_3item,
                f_tank_density, o_tank_density, diff_tank_density,
                f_dps_density, o_dps_density, diff_dps_density,
                # Combat Power Metrics
                f_total_pow, o_total_pow, diff_total_pow,
                f_top3_pow, o_top3_pow, diff_top3_pow,
                f_carry_pow, o_carry_pow, diff_carry_pow,
                # Item Utility Types
                f_anti_heal, o_anti_heal, diff_anti_heal,
                f_shred, o_shred, diff_shred,
                f_mana, o_mana, diff_mana,
                # Stage Cross Interactions
                stage_x_diff_pow, stage_x_diff_top3, stage_x_diff_gold,
                # Placement Features
                f_place["front_units"], o_place["front_units"], diff_front_units,
                f_place["back_units"], o_place["back_units"], diff_back_units,
                f_place["front_stars"], o_place["front_stars"], diff_front_stars,
                f_place["back_stars"], o_place["back_stars"], diff_back_stars,
                f_place["front_items"], o_place["front_items"], diff_front_items,
                f_place["back_items"], o_place["back_items"], diff_back_items,
                f_place["front_ratio"], o_place["front_ratio"], diff_front_ratio,
                f_place["carry_row"], f_place["carry_col"],
                f_place["tank_row"], f_place["tank_col"],
                carry_same_side, carry_tank_dist, f_blitz_risk,
                # Trait Summaries & Power
                f_act_cnt, o_act_cnt, diff_act_traits,
                f_max_tier, o_max_tier, diff_max_tier,
                f_stat_power, o_stat_power, diff_stat_power,
            ]

            for trait in ALL_SET17_TRAITS:
                diff_t_count = f_traits[trait] - o_traits[trait]
                diff_t_tier = f_tiers[trait] - o_tiers[trait]
                base_values.extend([f_traits[trait], o_traits[trait], diff_t_count, diff_t_tier])

            features[row_idx, :base_dim] = base_values

        return features

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return feature names for transformed matrix."""
        return np.array(self.feature_names_)

    def _build_feature_names(self) -> list[str]:
        names = list(self._base_feature_names())
        if self.include_champions:
            if self.include_absolute:
                names.extend([f"focal_champ__{c}" for c in self.champions_vocab_])
                names.extend([f"opponent_champ__{c}" for c in self.champions_vocab_])
            names.extend([f"champ_diff__{c}" for c in self.champions_vocab_])

        if self.include_items:
            if self.include_absolute:
                names.extend([f"focal_item__{i}" for i in self.items_vocab_])
                names.extend([f"opponent_item__{i}" for i in self.items_vocab_])
            names.extend([f"item_diff__{i}" for i in self.items_vocab_])

        if self.include_augments:
            names.extend([f"aug_diff__{a}" for a in self.augments_vocab_])
        return names

    @staticmethod
    def _base_feature_names() -> list[str]:
        names = [
            "stage_num", "round_num", "total_round", "is_augment_round",
            "focal_level", "opponent_level", "diff_level",
            "focal_health", "opponent_health", "diff_health", "focal_gold",
            "focal_unit_count", "opponent_unit_count", "diff_unit_count",
            "focal_total_stars", "opponent_total_stars", "diff_total_stars",
            "focal_star1_count", "focal_star2_count", "focal_star3_count",
            "opponent_star1_count", "opponent_star2_count", "opponent_star3_count",
            "diff_star1_count", "diff_star2_count", "diff_star3_count",
            "focal_total_items", "opponent_total_items", "diff_total_items",
            "focal_carry_3item_count", "opponent_carry_3item_count", "diff_carry_3item_count",
            "focal_unit_2item_count", "opponent_unit_2item_count", "diff_unit_2item_count",
            "focal_unit_1item_count", "opponent_unit_1item_count", "diff_unit_1item_count",
            "focal_avg_stars", "opponent_avg_stars", "diff_avg_stars",
            "focal_augment_count", "opponent_augment_count", "diff_augment_count",
            # Gold Valuation & High Cost Carries
            "focal_gold_val", "opponent_gold_val", "diff_gold_val",
            "focal_4cost_2star_count", "opponent_4cost_2star_count", "diff_4cost_2star_count",
            "focal_5cost_count", "opponent_5cost_count", "diff_5cost_count",
            "focal_synergy_item_score", "opponent_synergy_item_score", "diff_synergy_item_score",
            # Item Efficiency & Star Allocation
            "focal_items_on_2star_count", "opponent_items_on_2star_count", "diff_items_on_2star_count",
            "focal_items_on_1star_count", "opponent_items_on_1star_count", "diff_items_on_1star_count",
            "focal_2star_3item_count", "opponent_2star_3item_count", "diff_2star_3item_count",
            "focal_2star_2item_count", "opponent_2star_2item_count", "diff_2star_2item_count",
            "focal_3star_3item_count", "opponent_3star_3item_count", "diff_3star_3item_count",
            "focal_tank_density", "opponent_tank_density", "diff_tank_density",
            "focal_dps_density", "opponent_dps_density", "diff_dps_density",
            # Combat Power Metrics
            "focal_total_combat_power", "opponent_total_combat_power", "diff_total_combat_power",
            "focal_top3_combat_power", "opponent_top3_combat_power", "diff_top3_combat_power",
            "focal_carry_combat_power", "opponent_carry_combat_power", "diff_carry_combat_power",
            # Item Utility Types
            "focal_anti_heal_items", "opponent_anti_heal_items", "diff_anti_heal_items",
            "focal_shred_items", "opponent_shred_items", "diff_shred_items",
            "focal_mana_items", "opponent_mana_items", "diff_mana_items",
            # Stage Cross Interactions
            "stage_x_diff_combat_power", "stage_x_diff_top3_power", "stage_x_diff_gold",
            # Placement features
            "focal_front_units", "opponent_front_units", "diff_front_units",
            "focal_back_units", "opponent_back_units", "diff_back_units",
            "focal_front_stars", "opponent_front_stars", "diff_front_stars",
            "focal_back_stars", "opponent_back_stars", "diff_back_stars",
            "focal_front_items", "opponent_front_items", "diff_front_items",
            "focal_back_items", "opponent_back_items", "diff_back_items",
            "focal_front_ratio", "opponent_front_ratio", "diff_front_ratio",
            "focal_carry_row", "focal_carry_col",
            "focal_tank_row", "focal_tank_col",
            "carry_same_side", "carry_tank_dist", "focal_blitz_hook_risk",
            # Trait summaries & Power
            "focal_active_traits_count", "opponent_active_traits_count", "diff_active_traits_count",
            "focal_max_trait_tier", "opponent_max_trait_tier", "diff_max_trait_tier",
            "focal_trait_stat_power", "opponent_trait_stat_power", "diff_trait_stat_power",
        ]

        for trait in ALL_SET17_TRAITS:
            names.extend([
                f"focal_trait__{trait}",
                f"opponent_trait__{trait}",
                f"diff_trait_count__{trait}",
                f"diff_trait_tier__{trait}",
            ])

        return names

    @staticmethod
    def _parse_loc(loc_str: Any) -> tuple[int, int] | None:
        """Parse loc coordinate (e.g. 'A1', 'D7', 'B_4') to (row in [0..3], col in [0..6])."""
        if not loc_str or not isinstance(loc_str, str):
            return None
        c = loc_str.strip().upper().replace("_", "")
        if len(c) < 2:
            return None
        row_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        if c[0] not in row_map:
            return None
        row = row_map[c[0]]
        try:
            col = int(c[1:]) - 1
            if 0 <= col <= 6:
                return row, col
        except ValueError:
            return None
        return None

    def _extract_placement_metrics(self, board_units: list[dict[str, Any]]) -> dict[str, float]:
        """Extract frontline/backline distributions, main carry and main tank spatial coordinates."""
        front_units = 0
        back_units = 0
        front_stars = 0
        back_stars = 0
        front_items = 0
        back_items = 0

        carry_unit: dict[str, Any] | None = None
        carry_max_score = -1.0

        tank_unit: dict[str, Any] | None = None
        tank_max_score = -1.0

        for unit in board_units:
            tier = int(unit.get("tier", 1) or 1)
            u_name = str(unit.get("unit", "") or "")
            cost = CHAMP_BASE_COSTS.get(u_name, 2)
            items = unit.get("items", []) or []
            n_items = len(items)
            loc = self._parse_loc(unit.get("loc"))

            if loc is not None:
                row, _ = loc
                if row in (0, 1):
                    front_units += 1
                    front_stars += tier
                    front_items += n_items
                    tank_score = n_items * 4.0 + tier * 2.0 + cost
                    if tank_score > tank_max_score:
                        tank_max_score = tank_score
                        tank_unit = unit
                else:
                    back_units += 1
                    back_stars += tier
                    back_items += n_items

            carry_score = n_items * 5.0 + tier * 2.5 + cost
            if carry_score > carry_max_score:
                carry_max_score = carry_score
                carry_unit = unit

        total_u = max(front_units + back_units, 1)
        front_ratio = front_units / total_u

        carry_loc = self._parse_loc(carry_unit.get("loc")) if carry_unit else None
        tank_loc = self._parse_loc(tank_unit.get("loc")) if tank_unit else None

        c_row, c_col = carry_loc if carry_loc else (3.0, 3.0)
        t_row, t_col = tank_loc if tank_loc else (0.0, 3.0)

        return {
            "front_units": float(front_units),
            "back_units": float(back_units),
            "front_stars": float(front_stars),
            "back_stars": float(back_stars),
            "front_items": float(front_items),
            "back_items": float(back_items),
            "front_ratio": float(front_ratio),
            "carry_row": float(c_row),
            "carry_col": float(c_col),
            "tank_row": float(t_row),
            "tank_col": float(t_col),
        }

    def _extract_trait_metrics(
        self, board_units: list[dict[str, Any]]
    ) -> tuple[dict[str, float], dict[str, float], float, float, float]:
        """Calculate active trait counts, synergy tiers, summary metrics, and total stat power."""
        counts = {t: 0.0 for t in ALL_SET17_TRAITS}
        tiers = {t: 0.0 for t in ALL_SET17_TRAITS}

        seen_champions: set[str] = set()

        for unit in board_units:
            u_name = unit.get("unit")
            if u_name and u_name not in seen_champions:
                seen_champions.add(u_name)
                for trait in CHAMP_TO_TRAITS.get(u_name, []):
                    counts[trait] += 1.0

            for item in unit.get("items", []) or []:
                if "Emblem" in item:
                    for trait in ALL_SET17_TRAITS:
                        if trait.lower() in item.lower():
                            counts[trait] += 1.0

        active_count = 0.0
        max_tier = 0.0
        total_stat_power = 0.0

        for trait, cnt in counts.items():
            thresholds = SET17_TRAIT_THRESHOLDS.get(trait, [2, 4, 6])
            tier = 0.0
            for t_idx, req in enumerate(thresholds, start=1):
                if cnt >= req:
                    tier = float(t_idx)
            tiers[trait] = tier
            if tier > 0:
                active_count += 1.0
                max_tier = max(max_tier, tier)
                total_stat_power += tier * 2.0 + (tier ** 2)

        return counts, tiers, active_count, max_tier, total_stat_power

    @staticmethod
    def _ensure_dataframe(X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        if isinstance(X, list):
            return pd.DataFrame(X)
        if isinstance(X, dict):
            return pd.DataFrame([X])
        return pd.DataFrame(X)

    @staticmethod
    def _parse_round_stage(stage_raw: Any) -> tuple[float, float, float, float]:
        if not isinstance(stage_raw, str) or not stage_raw.strip():
            return 1.0, 1.0, 11.0, 0.0
        parts = stage_raw.strip().split("-")
        try:
            stage = float(parts[0])
            rnd = float(parts[1]) if len(parts) > 1 else 1.0
            total_rnd = stage * 10.0 + rnd
            is_aug = 1.0 if stage_raw.strip() in {"2-1", "3-2", "4-2"} else 0.0
            return stage, rnd, total_rnd, is_aug
        except (ValueError, IndexError):
            return 1.0, 1.0, 11.0, 0.0

    @staticmethod
    def _parse_input_state(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}

    @staticmethod
    def _parse_augments(raw: Any) -> list[str]:
        if isinstance(raw, list):
            return [str(item).strip() for item in raw if str(item).strip()]
        if isinstance(raw, str) and raw.strip():
            return [item.strip() for item in raw.split(",") if item.strip()]
        return []

    @staticmethod
    def _safe_num(val: Any, default: float = 0.0) -> float:
        if val is None or pd.isna(val):
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default
