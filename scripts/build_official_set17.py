"""Generate official Set 17 profile directly from CommunityDragon Data Dragon dataset.

Extracts champions, traits, base stats, item components, completed items, and crafting recipes.
"""

from __future__ import annotations

import json
from pathlib import Path


def generate_official_set17():
    raw_data = json.loads(Path("data/cdragon_set17.json").read_text(encoding="utf-8"))
    
    # Load all items from CommunityDragon
    import urllib.request
    req = urllib.request.Request(
        "https://raw.communitydragon.org/latest/cdragon/tft/en_us.json",
        headers={"User-Agent": "Mozilla/5.0"}
    )
    raw_full = json.loads(urllib.request.urlopen(req).read().decode("utf-8"))
    all_items_raw = raw_full.get("items", [])

    # 1. Process Champions
    champs_raw = raw_data.get("champions", [])
    valid_champs = []

    for c in champs_raw:
        api_name = c.get("apiName", "")
        cost = c.get("cost", 0)
        traits = c.get("traits", [])
        name = c.get("name", "")

        if cost not in (1, 2, 3, 4, 5):
            continue
        if not api_name.startswith("TFT17_") or api_name.startswith("TFT17_Enemy_"):
            continue
        if not traits:
            continue

        stats = c.get("stats", {})
        hp = stats.get("hp", 0.0)
        armor = stats.get("armor", 0.0)
        mr = stats.get("magicResist", 0.0)
        ad = stats.get("damage", 0.0)
        aspeed = stats.get("attackSpeed", 0.0)
        range_hex = stats.get("range", 1.0)
        mana = stats.get("initialMana", 0.0)
        max_mana = stats.get("mana", 0.0)

        # Determine UnitRole
        role = "UnitRole.UTILITY"
        if cost in (1, 2, 3) and (armor >= 40 or hp >= 700 or "Bastion" in traits or "Brawler" in traits or "Vanguard" in traits):
            role = "UnitRole.TANK"
        elif range_hex >= 3 or "Sniper" in traits or "Challenger" in traits:
            role = "UnitRole.AD_CARRY"
        elif "Conduit" in traits or "Replicator" in traits or "Stargazer" in traits:
            role = "UnitRole.AP_CARRY"
        elif "Rogue" in traits or "Marauder" in traits:
            role = "UnitRole.AD_CARRY" if ad >= 50 else "UnitRole.TANK"
        elif hp >= 900 or armor >= 50:
            role = "UnitRole.TANK"

        is_carry = role in ("UnitRole.AD_CARRY", "UnitRole.AP_CARRY")
        is_tank = role == "UnitRole.TANK"

        if role == "UnitRole.TANK":
            rec_items = ("TFT_Item_WarmogsArmor", "TFT_Item_DragonsClaw", "TFT_Item_BrambleVest")
        elif role == "UnitRole.AP_CARRY":
            rec_items = ("TFT_Item_JeweledGauntlet", "TFT_Item_SpearOfShojin", "TFT_Item_RabadonsDeathcap")
        elif role == "UnitRole.AD_CARRY":
            rec_items = ("TFT_Item_InfinityEdge", "TFT_Item_LastWhisper", "TFT_Item_GuinsoosRageblade")
        else:
            rec_items = ("TFT_Item_SpearOfShojin", "TFT_Item_StatikkShiv", "TFT_Item_Morellonomicon")

        valid_champs.append({
            "api_name": api_name,
            "name": name,
            "cost": cost,
            "traits": tuple(traits),
            "role": role,
            "is_carry": is_carry,
            "is_tank": is_tank,
            "hp": hp,
            "armor": armor,
            "mr": mr,
            "ad": ad,
            "aspeed": aspeed,
            "range": range_hex,
            "mana": mana,
            "max_mana": max_mana,
            "rec_items": rec_items,
        })

    valid_champs.sort(key=lambda x: (x["cost"], x["api_name"]))

    # 2. Process Traits
    traits_raw = raw_data.get("traits", [])
    valid_traits = {}
    trait_members = {}

    for c in valid_champs:
        for t in c["traits"]:
            trait_members.setdefault(t, []).append(c["api_name"])

    for t in traits_raw:
        api_name = t.get("apiName", "")
        name = t.get("name", "")
        desc = t.get("desc", "").replace("<br>", " ").replace("</br>", " ").replace("<br/>", " ").replace("\n", " ").strip()
        effects = t.get("effects", [])
        thresholds = [eff.get("minUnits") for eff in effects if eff.get("minUnits") is not None]
        if not thresholds:
            continue
        if name not in trait_members and api_name not in trait_members:
            continue

        clean_thresh = tuple(sorted(list(set(thresholds))))
        members = tuple(trait_members.get(name, []))

        valid_traits[name] = {
            "api_name": api_name,
            "name": name,
            "thresholds": clean_thresh,
            "desc": desc,
            "members": members,
        }

    # 3. Process Items & Recipes
    core_components = {
        "TFT_Item_BFSword": ("B.F. Sword", "+10 Attack Damage"),
        "TFT_Item_RecurveBow": ("Recurve Bow", "+10% Attack Speed"),
        "TFT_Item_NeedlesslyLargeRod": ("Needlessly Large Rod", "+10 Ability Power"),
        "TFT_Item_TearOfTheGoddess": ("Tear of the Goddess", "+15 Starting Mana"),
        "TFT_Item_ChainVest": ("Chain Vest", "+20 Armor"),
        "TFT_Item_NegatronCloak": ("Negatron Cloak", "+20 Magic Resist"),
        "TFT_Item_GiantsBelt": ("Giant's Belt", "+150 Health"),
        "TFT_Item_SparringGloves": ("Sparring Gloves", "+20% Critical Strike Chance"),
        "TFT_Item_Spatula": ("Spatula", "Special Emblem and Team Size Component"),
    }

    # Canonical standard 36 item mappings
    canonical_items = {}
    canonical_recipes = {}

    for it in all_items_raw:
        api = it.get("apiName") or ""
        name = it.get("name") or ""
        comp = it.get("composition") or []
        desc = it.get("desc") or ""

        if len(comp) == 2 and comp[0] in core_components and comp[1] in core_components:
            key = frozenset([comp[0], comp[1]])
            clean_desc = desc.replace("<br>", " ").replace("</br>", " ").replace("<br/>", " ").replace("\n", " ").strip()
            
            # Map canonical API name
            if comp[0] == "TFT_Item_Spatula" and comp[1] == "TFT_Item_Spatula":
                api = "TFT_Item_TacticiansCrown"
                name = "Tactician's Crown"
            elif "Corrupted" in api:
                # normalize corrupted item IDs to standard TFT_Item_ names
                api = api.replace("Corrupted", "")

            canonical_items[api] = {
                "api_name": api,
                "name": name,
                "composition": tuple(comp),
                "desc": clean_desc,
                "is_component": False,
            }
            canonical_recipes[key] = api

    # Add components to item list
    for cid, (cname, cdesc) in core_components.items():
        canonical_items[cid] = {
            "api_name": cid,
            "name": cname,
            "composition": (),
            "desc": cdesc,
            "is_component": True,
        }

    # Add Set 17 Spatula Emblems
    set17_emblems = {
        frozenset(["TFT_Item_Spatula", "TFT_Item_BFSword"]): ("TFT17_Item_DarkStarEmblemItem", "Dark Star Emblem", "The holder gains the Dark Star trait."),
        frozenset(["TFT_Item_Spatula", "TFT_Item_RecurveBow"]): ("TFT17_Item_PulsefireEmblemItem", "Timebreaker Emblem", "The holder gains the Timebreaker trait."),
        frozenset(["TFT_Item_Spatula", "TFT_Item_NeedlesslyLargeRod"]): ("TFT17_Item_StargazerEmblemItem", "Stargazer Emblem", "The holder gains the Stargazer trait."),
        frozenset(["TFT_Item_Spatula", "TFT_Item_TearOfTheGoddess"]): ("TFT17_Item_SpaceGrooveEmblemItem", "Space Groove Emblem", "The holder gains the Space Groove trait."),
        frozenset(["TFT_Item_Spatula", "TFT_Item_ChainVest"]): ("TFT17_Item_AstronautEmblemItem", "Meeple Emblem", "The holder gains the Meeple trait."),
        frozenset(["TFT_Item_Spatula", "TFT_Item_NegatronCloak"]): ("TFT17_Item_FavoredEmblemItem", "Arbiter Emblem", "The holder gains the Arbiter trait."),
        frozenset(["TFT_Item_Spatula", "TFT_Item_GiantsBelt"]): ("TFT17_Item_PrimordianEmblemItem", "Primordian Emblem", "The holder gains the Primordian trait."),
        frozenset(["TFT_Item_Spatula", "TFT_Item_SparringGloves"]): ("TFT17_Item_DRXEmblemItem", "N.O.V.A. Emblem", "The holder gains the N.O.V.A. trait."),
    }

    for key, (e_id, e_name, e_desc) in set17_emblems.items():
        canonical_items[e_id] = {
            "api_name": e_id,
            "name": e_name,
            "composition": tuple(key),
            "desc": e_desc,
            "is_component": False,
        }
        canonical_recipes[key] = e_id

    # 4. Generate Python Code for set17.py
    lines = [
        '"""Set 17 Complete Profile: Champions, Traits, Items, Recipes, Odds, and Leveling Curves.',
        'Extracted directly from official Riot Games CommunityDragon / Data Dragon dataset.',
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "import json",
        "from dataclasses import asdict, dataclass, field",
        "from pathlib import Path",
        "from typing import Any",
        "",
        "from tft_ai_player.simulation.config import (",
        "    STANDARD_COMPONENTS,",
        "    STANDARD_LEVEL_EXP,",
        "    STANDARD_PASSIVE_GOLD,",
        "    STANDARD_POOL_SIZES,",
        "    STANDARD_SHOP_ODDS,",
        "    STANDARD_STAGE_BASE_DAMAGE,",
        "    ChampionDef,",
        "    ItemDef,",
        "    SetData,",
        "    TraitDef,",
        "    UnitRole,",
        ")",
        "",
        "# =============================================================================",
        f"# SET 17 TRAIT DEFINITIONS & SYNERGY BONUSES ({len(valid_traits)} TRAITS)",
        "# =============================================================================",
        "",
        "@dataclass(frozen=True, slots=True)",
        "class Set17TraitInfo:",
        '    """Detailed trait synergy definition with threshold bonuses."""',
        "",
        "    trait_id: str",
        "    name: str",
        "    thresholds: tuple[int, ...]",
        "    description: str",
        "    champions: tuple[str, ...]",
        "",
        "",
        "SET17_TRAIT_CATALOG: dict[str, Set17TraitInfo] = {",
    ]

    for tname, tinfo in sorted(valid_traits.items()):
        desc_escaped = tinfo["desc"].replace('"', '\\"')
        lines.append(f'    "{tname}": Set17TraitInfo(')
        lines.append(f'        trait_id="{tname}",')
        lines.append(f'        name="{tname}",')
        lines.append(f'        thresholds={tinfo["thresholds"]},')
        lines.append(f'        description="{desc_escaped}",')
        lines.append(f'        champions={tinfo["members"]},')
        lines.append("    ),")

    lines.extend([
        "}",
        "",
        "",
        "# =============================================================================",
        f"# SET 17 CHAMPION CATALOG ({len(valid_champs)} CHAMPIONS)",
        "# =============================================================================",
        "",
        "@dataclass(frozen=True, slots=True)",
        "class Set17ChampionInfo:",
        '    """Complete specification of a Set 17 champion."""',
        "",
        "    champion_id: str",
        "    name: str",
        "    cost: int",
        "    traits: tuple[str, ...]",
        "    role: UnitRole",
        "    is_carry: bool",
        "    is_tank: bool",
        "    hp: float",
        "    armor: float",
        "    mr: float",
        "    ad: float",
        "    aspeed: float",
        "    range: float",
        "    mana: float",
        "    max_mana: float",
        "    recommended_items: tuple[str, ...]",
        "",
        "",
        "SET17_CHAMPION_CATALOG: tuple[Set17ChampionInfo, ...] = (",
    ])

    current_cost = None
    for c in valid_champs:
        if c["cost"] != current_cost:
            current_cost = c["cost"]
            count_for_cost = sum(1 for x in valid_champs if x["cost"] == current_cost)
            lines.append(f"    # --- {current_cost}-Cost ({count_for_cost} Champions) ---")

        escaped_name = c["name"].replace('"', '\\"')
        lines.append(f'    Set17ChampionInfo("{c["api_name"]}", "{escaped_name}", {c["cost"]}, {c["traits"]}, {c["role"]}, {c["is_carry"]}, {c["is_tank"]}, {c["hp"]}, {c["armor"]}, {c["mr"]}, {c["ad"]}, {c["aspeed"]:.2f}, {c["range"]}, {c["mana"]}, {c["max_mana"]}, {c["rec_items"]}),')

    lines.extend([
        ")",
        "",
        "",
        "# =============================================================================",
        f"# SET 17 ITEMS & RECIPES CATALOG ({len(canonical_items)} ITEMS, {len(canonical_recipes)} RECIPES)",
        "# =============================================================================",
        "",
        "@dataclass(frozen=True, slots=True)",
        "class Set17ItemInfo:",
        '    """Official Item definition with recipe composition."""',
        "",
        "    item_id: str",
        "    name: str",
        "    is_component: bool",
        "    recipe: tuple[str, ...]",
        "    description: str",
        "",
        "",
        "SET17_ITEMS_CATALOG: dict[str, Set17ItemInfo] = {",
    ])

    for i_id, i_info in sorted(canonical_items.items()):
        desc_escaped = i_info["desc"].replace('"', '\\"')
        escaped_name = i_info["name"].replace('"', '\\"')
        lines.append(f'    "{i_id}": Set17ItemInfo(')
        lines.append(f'        item_id="{i_id}",')
        lines.append(f'        name="{escaped_name}",')
        lines.append(f'        is_component={i_info["is_component"]},')
        lines.append(f'        recipe={i_info["composition"]},')
        lines.append(f'        description="{desc_escaped}",')
        lines.append("    ),")

    lines.extend([
        "}",
        "",
        "SET17_ITEM_RECIPES: dict[frozenset[str], str] = {",
    ])

    for key, res_id in sorted(canonical_recipes.items(), key=lambda x: sorted(list(x[0]))):
        k_list = sorted(list(key))
        c1 = k_list[0]
        c2 = k_list[0] if len(k_list) == 1 else k_list[1]
        lines.append(f'    frozenset(["{c1}", "{c2}"]): "{res_id}",')

    lines.extend([
        "}",
        "",
        "",
        "# =============================================================================",
        "# SET PROFILE DATA CLASS & EXPORTER",
        "# =============================================================================",
        "",
        "@dataclass(frozen=True, slots=True)",
        "class Set17Profile:",
        '    """Top-level container for Set 17 profile specifications."""',
        "",
        '    set_name: str = "TFTSet17"',
        f"    total_champions: int = {len(valid_champs)}",
        f"    total_traits: int = {len(valid_traits)}",
        f"    total_items: int = {len(canonical_items)}",
        f"    total_recipes: int = {len(canonical_recipes)}",
        "    champions: tuple[Set17ChampionInfo, ...] = SET17_CHAMPION_CATALOG",
        "    traits: dict[str, Set17TraitInfo] = field(default_factory=lambda: dict(SET17_TRAIT_CATALOG))",
        "    items: dict[str, Set17ItemInfo] = field(default_factory=lambda: dict(SET17_ITEMS_CATALOG))",
        "",
        "    def to_dict(self) -> dict[str, Any]:",
        '        """Convert profile to serializable dictionary."""',
        "        return {",
        '            "set_name": self.set_name,',
        '            "stats": {',
        '                "total_champions": len(self.champions),',
        '                "total_traits": len(self.traits),',
        '                "total_items": len(self.items),',
        '                "total_recipes": len(SET17_ITEM_RECIPES),',
        '                "champion_cost_breakdown": {',
        '                    "1_cost": sum(1 for c in self.champions if c.cost == 1),',
        '                    "2_cost": sum(1 for c in self.champions if c.cost == 2),',
        '                    "3_cost": sum(1 for c in self.champions if c.cost == 3),',
        '                    "4_cost": sum(1 for c in self.champions if c.cost == 4),',
        '                    "5_cost": sum(1 for c in self.champions if c.cost == 5),',
        "                },",
        "            },",
        '            "champions": [',
        "                {",
        '                    "id": c.champion_id,',
        '                    "name": c.name,',
        '                    "cost": c.cost,',
        '                    "traits": list(c.traits),',
        '                    "role": c.role.value,',
        '                    "is_carry": c.is_carry,',
        '                    "is_tank": c.is_tank,',
        '                    "hp": c.hp,',
        '                    "armor": c.armor,',
        '                    "mr": c.mr,',
        '                    "ad": c.ad,',
        '                    "aspeed": c.aspeed,',
        '                    "range": c.range,',
        '                    "mana": c.mana,',
        '                    "max_mana": c.max_mana,',
        '                    "recommended_items": list(c.recommended_items),',
        "                }",
        "                for c in self.champions",
        "            ],",
        '            "traits": {',
        "                t.trait_id: {",
        '                    "name": t.name,',
        '                    "thresholds": list(t.thresholds),',
        '                    "description": t.description,',
        '                    "champions": list(t.champions),',
        "                }",
        "                for t in self.traits.values()",
        "            },",
        '            "items": {',
        "                i.item_id: {",
        '                    "name": i.name,',
        '                    "is_component": i.is_component,',
        '                    "recipe": list(i.recipe),',
        '                    "description": i.description,',
        "                }",
        "                for i in self.items.values()",
        "            },",
        '            "recipes": [',
        "                {",
        '                    "components": sorted(list(comp)),',
        '                    "result_item": res_id,',
        '                    "result_name": self.items.get(res_id, Set17ItemInfo(res_id, res_id, False, (), "")).name,',
        "                }",
        "                for comp, res_id in SET17_ITEM_RECIPES.items()",
        "            ],",
        '            "shop_odds": {str(k): list(v) for k, v in STANDARD_SHOP_ODDS.items()},',
        '            "level_exp": {str(k): v for k, v in STANDARD_LEVEL_EXP.items()},',
        '            "pool_sizes": {str(k): v for k, v in STANDARD_POOL_SIZES.items()},',
        "        }",
        "",
        "",
        "def get_set17_data() -> SetData:",
        '    """Build a complete SetData instance for Set 17 simulations."""',
        "    champions: dict[str, ChampionDef] = {",
        "        c.champion_id: ChampionDef(",
        "            champion_id=c.champion_id,",
        "            name=c.name,",
        "            cost=c.cost,",
        "            traits=c.traits,",
        "            role=c.role,",
        "        )",
        "        for c in SET17_CHAMPION_CATALOG",
        "    }",
        "",
        "    traits: dict[str, TraitDef] = {",
        "        t.trait_id: TraitDef(",
        "            trait_id=t.trait_id,",
        "            name=t.name,",
        "            thresholds=t.thresholds,",
        "        )",
        "        for t in SET17_TRAIT_CATALOG.values()",
        "    }",
        "",
        "    items: dict[str, ItemDef] = {",
        "        it.item_id: ItemDef(",
        "            item_id=it.item_id,",
        "            name=it.name,",
        "            is_component=it.is_component,",
        "            recipe=it.recipe if len(it.recipe) == 2 else None,",
        "        )",
        "        for it in SET17_ITEMS_CATALOG.values()",
        "    }",
        "",
        "    recipes: dict[frozenset[str], str] = dict(SET17_ITEM_RECIPES)",
        "",
        "    return SetData(",
        '        set_name="TFTSet17",',
        "        champions=champions,",
        "        traits=traits,",
        "        items=items,",
        "        recipes=recipes,",
        "        shop_odds=STANDARD_SHOP_ODDS,",
        "        pool_sizes=STANDARD_POOL_SIZES,",
        "        level_exp=STANDARD_LEVEL_EXP,",
        "        stage_base_damage=STANDARD_STAGE_BASE_DAMAGE,",
        "        passive_gold_schedule=STANDARD_PASSIVE_GOLD,",
        "    )",
        "",
        "",
        'def export_set17_profile_json(output_path: str | Path = "data/set17_profile.json") -> Path:',
        '    """Export complete Set 17 profile JSON to file."""',
        "    profile = Set17Profile()",
        "    data = profile.to_dict()",
        "    out = Path(output_path).resolve()",
        "    out.parent.mkdir(parents=True, exist_ok=True)",
        '    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")',
        "    return out",
        "",
    ])

    out_file = Path("src/tft_ai_player/simulation/sets/set17.py")
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated complete official {out_file} ({len(lines)} lines)")


if __name__ == "__main__":
    generate_official_set17()
