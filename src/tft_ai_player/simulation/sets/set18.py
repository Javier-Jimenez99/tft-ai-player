"""Set 18 'Enchanted Wilds' Complete Profile: Champions, Traits, Items, Recipes, Odds, and Leveling Curves.
Extracted directly from official Riot Games CommunityDragon / Data Dragon dataset.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tft_ai_player.simulation.config import (
    STANDARD_COMPONENTS,
    STANDARD_LEVEL_EXP,
    STANDARD_PASSIVE_GOLD,
    STANDARD_POOL_SIZES,
    STANDARD_SHOP_ODDS,
    STANDARD_STAGE_BASE_DAMAGE,
    ChampionDef,
    ItemDef,
    SetData,
    TraitDef,
    UnitRole,
)

# =============================================================================
# SET 18 TRAIT DEFINITIONS & SYNERGY BONUSES (36 TRAITS)
# =============================================================================

@dataclass(frozen=True, slots=True)
class Set18TraitInfo:
    """Detailed trait synergy definition with threshold bonuses."""

    trait_id: str
    name: str
    thresholds: tuple[int, ...]
    description: str
    champions: tuple[str, ...]


SET18_TRAIT_CATALOG: dict[str, Set18TraitInfo] = {
    "Adaptor": Set18TraitInfo(
        trait_id="Adaptor",
        name="Adaptor",
        thresholds=(2, 3, 4),
        description="Innate: Adaptor abilities change depending on if their Attack Damage or Ability Power is higher.  Adaptors gain Ability Power or Attack Damage, depending on which is higher. <row>(@MinUnits@) @ADAPGain*100@%  OR</row> <row>(@MinUnits@) @ADAPGain*100@%  OR</row> <row>(@MinUnits@) @ADAPGain*100@%  OR</row>",
        champions=('TFT18_Akali', 'TFT18_Gromp', 'TFT18_KogMaw', 'TFT18_MasterYi', 'TFT18_Nidalee'),
    ),
    "Apex Predator": Set18TraitInfo(
        trait_id="Apex Predator",
        name="Apex Predator",
        thresholds=(1,),
        description="Elder Dragon takes up 2 team slots and grants +2 to the Riftbeast trait.",
        champions=('TFT18_ElderDragon',),
    ),
    "Attuned": Set18TraitInfo(
        trait_id="Attuned",
        name="Attuned",
        thresholds=(1,),
        description="Your strongest Alune cycles to a new phase of the moon after each cast. While the moon is at or below half full, your team gains @Durability*100@% Durabilty. While above, your team gains @DamageAmp*100@% Damage Amp.",
        champions=('TFT18_Alune',),
    ),
    "Avatar": Set18TraitInfo(
        trait_id="Avatar",
        name="Avatar",
        thresholds=(1,),
        description="Having an Avatar on your bench or board transforms all other Avatars in your shop to the Trait of that Avatar.  An Avatar's chosen Trait is counted twice for Trait bonuses.",
        champions=('TFT18_Lux',),
    ),
    "Blackthorn": Set18TraitInfo(
        trait_id="Blackthorn",
        name="Blackthorn",
        thresholds=(2, 4, 6),
        description="The ally on the Blackthorn hex is sacrificed before combat, granting your champions Health.   Blackthorn champions gain additional stats based on the sacrifice's Role, Star Level, and Cost. <row>(@MinUnits@) @Health@ %i:scaleHealth%</row> <row>(@MinUnits@) @Health@ %i:scaleHealth%; Bonus is @StatMultiplier*100@% stronger</row> <row>(@MinUnits@) @Health@ %i:scaleHealth%; Bonus is @StatMultiplier*100@% stronger</row>",
        champions=('TFT18_RekSai', 'TFT18_Veigar', 'TFT18_LeBlanc', 'TFT18_Warwick', 'TFT18_Azir', 'TFT18_Malphite'),
    ),
    "Blossom": Set18TraitInfo(
        trait_id="Blossom",
        name="Blossom",
        thresholds=(3, 5, 7, 9, 11),
        description="After combat, your Wisps are empowered.   Blossom champions gain Attack Damage, Ability Power, and @BonusHealth*100@% max Health. <row>(@MinUnits@) Wisps are upgraded, @ADAP*100@% %i:scaleAD%%i:scaleAP%</row> <row>(@MinUnits@) Wisps are in every shop, @ADAP*100@% %i:scaleAD%%i:scaleAP%</row> <row>(@MinUnits@) Gain @GoldPerCharmPurchased@ gold after buying a Wisp, @ADAP*100@% %i:scaleAD%%i:scaleAP%</row> <row>(@MinUnits@) You can buy 2 Wisps per round, @ADAP*100@% %i:scaleAD%%i:scaleAP%</row> <row>(@MinUnits@) Wisps overflow with power, @ADAP*100@% %i:scaleAD%%i:scaleAP%</row>",
        champions=('TFT18_Karma', 'TFT18_Alistar', 'TFT18_MasterYi', 'TFT18_Vi', 'TFT18_Ahri', 'TFT18_Lillia'),
    ),
    "Bounty Seeker": Set18TraitInfo(
        trait_id="Bounty Seeker",
        name="Bounty Seeker",
        thresholds=(1,),
        description="Choose a bounty. Your strongest Draven can progress on the bounty to earn its reward. Upon completing a bounty, choose another.",
        champions=('TFT18_Draven',),
    ),
    "Brawler": Set18TraitInfo(
        trait_id="Brawler",
        name="Brawler",
        thresholds=(2, 4, 6),
        description="Your team gains @BrawlerTeamHealth@ max Health. Brawlers gain more. <row>(@MinUnits@) @BrawlerHealthPercent*100@% %i:scaleHealth%</row> <row>(@MinUnits@) @BrawlerHealthPercent*100@% %i:scaleHealth%</row> <row>(@MinUnits@) @BrawlerHealthPercent*100@% %i:scaleHealth%</row>",
        champions=('TFT18_Kobuko', 'TFT18_RekSai', 'TFT18_Scuttlecrab', 'TFT18_Vi', 'TFT18_Sett', 'TFT18_Gnar'),
    ),
    "Caustic": Set18TraitInfo(
        trait_id="Caustic",
        name="Caustic",
        thresholds=(1,),
        description="Caustic damage @ShredPercent@% Shreds and Sunders enemies for @ShredDuration@ seconds.  <rules>Shred: Reduce Magic Resist</rules> <rules>Sunder: Reduce Armor</rules>",
        champions=('TFT18_KogMaw',),
    ),
    "Coven": Set18TraitInfo(
        trait_id="Coven",
        name="Coven",
        thresholds=(3, 4, 5, 7),
        description="Gather Essence by killing champions, plus more by losing player combat.   Beginning at @EssenceForRitualTooltip@ Essence, choose to convert Essence to rewards or continue gathering Essence. Each reward requires increasing amounts of Essence. <row>(@MinUnits@) @EssencePerDeath@ Essence per kill, @EssencePerLoss@ Essence per loss</row> <row>(@MinUnits@) @EssencePerDeath@ per kill, @EssencePerLoss@ per loss</row> <row>(@MinUnits@) @EssencePerDeath@ per kill, @EssencePerLoss@ per loss</row> <row>(@MinUnits@) @EssencePerDeath@ per kill, @EssencePerLoss@ per loss</row>",
        champions=('TFT18_Camille', 'TFT18_Yorick', 'TFT18_Elise', 'TFT18_Cassiopeia', 'TFT18_Fiddlesticks', 'TFT18_Morgana'),
    ),
    "Defender": Set18TraitInfo(
        trait_id="Defender",
        name="Defender",
        thresholds=(2, 4, 6),
        description="Your team gains @NonDefenderDefenseGain@ Armor and Magic Resist. Defenders gain more. <row>(@MinUnits@) @DefenderDefenseGain@</row> <row>(@MinUnits@) @DefenderDefenseGain@</row> <row>(@MinUnits@) @DefenderDefenseGain@</row>",
        champions=('TFT18_Ornn', 'TFT18_Pebbles', 'TFT18_Shen', 'TFT18_Rammus', 'TFT18_Brambleback', 'TFT18_Malphite', 'TFT18_Kennen', 'TFT18_Taric'),
    ),
    "Eclipse": Set18TraitInfo(
        trait_id="Eclipse",
        name="Eclipse",
        thresholds=(1,),
        description="After @DelaySeconds@ seconds of combat, the Eclipse kills the lowest Health enemy, repeating every @EclipseBeamPeriod@ seconds.",
        champions=(),
    ),
    "Elderwood": Set18TraitInfo(
        trait_id="Elderwood",
        name="Elderwood",
        thresholds=(3, 5, 7, 9, 11),
        description="Gain placeable Elderwood plants. <row>(@MinUnits@) A Stonebark Tree and a Lifebloom</row> <row>(@MinUnits@) A second Stonebark Tree and Stonebark Trees gain @StonebarkTreeBonusHealth@ Health</row> <row>(@MinUnits@) and the Deepwood Protector</row> <row>(@MinUnits@) Plants star up to 2-star</row> <row>(@MinUnits@) Plants star up to 3-star and the forest comes to life</row>",
        champions=('TFT18_Ornn', 'TFT18_Hecarim', 'TFT18_Rammus', 'TFT18_Amumu', 'TFT18_Ivern', 'TFT18_Maokai'),
    ),
    "Emerald Aspect": Set18TraitInfo(
        trait_id="Emerald Aspect",
        name="Emerald Aspect",
        thresholds=(1,),
        description="Drag an ally onto your strongest Taric to pair them, granting the ally powerful bonuses from Taric's Ability.",
        champions=('TFT18_Taric',),
    ),
    "Executioner": Set18TraitInfo(
        trait_id="Executioner",
        name="Executioner",
        thresholds=(2, 3, 4),
        description="<row>(@MinUnits@) Executioners gain <TFTKeyword>Precision</TFTKeyword> and @CritChance*100@% Critical Strike Chance.</row> <row>(@MinUnits@) Additionally, enemies bleed for @BonusBleedPercent*100@% bonus true damage over @BleedDuration@ seconds.</row> <row>(@MinUnits@) Bleed damage increased to @BonusBleedPercent*100@%.</row>",
        champions=('TFT18_KhaZix', 'TFT18_Morgana', 'TFT18_Draven'),
    ),
    "Fae": Set18TraitInfo(
        trait_id="Fae",
        name="Fae",
        thresholds=(2, 4),
        description="Your team's damage, healing, and shielding attracts Pixies.  Each Pixie grants Fae champions Attack Damage and Ability Power, and after they fall below @HealThreshold*100@% Health, they heal for each Pixie. <row>(@MinUnits@) @ADAP@%  and @Heal@% Heal.</row> <row>(@MinUnits@) @ADAP@%  and @Heal@% Heal. After attracting 7 Pixies, start attracting Golden Pixies instead which grant gold.</row>",
        champions=('TFT18_Rakan', 'TFT18_Xayah', 'TFT18_Ezreal', 'TFT18_Ashe'),
    ),
    "Flora Fatalis": Set18TraitInfo(
        trait_id="Flora Fatalis",
        name="Flora Fatalis",
        thresholds=(1, 2),
        description="Flora Fatalis champions harvest enemies on takedown, gaining: <row>(@MinUnits@) @Mana@ Mana</row> <row>(@MinUnits@) and grant a @PercentHeal*100@% max Health heal to the lowest Health ally.</row>",
        champions=('TFT18_Zyra',),
    ),
    "Greenfather": Set18TraitInfo(
        trait_id="Greenfather",
        name="Greenfather",
        thresholds=(1,),
        description="Gain @SeedsPerCast@ seed when your strongest Ivern casts, plus @CombatStartSeeds@ every combat. Ivern uses @SeedsToGrowHex@ seeds to grow a hex on your board that grants a bonus to its occupant.   Seeds: 0 / @SeedsToGrowHex@  Current Biome:",
        champions=('TFT18_Ivern',),
    ),
    "Hunter": Set18TraitInfo(
        trait_id="Hunter",
        name="Hunter",
        thresholds=(2, 3, 4, 5),
        description="Hunters gain Attack Damage. If a Hunter hasn't swapped targets for @HunterDuration@ seconds, they gain @DamageAmp*100@% Damage Amp. <row>(@MinUnits@) @HunterAD*100@% %i:scaleAD%</row> <row>(@MinUnits@) @HunterAD*100@% %i:scaleAD%</row> <row>(@MinUnits@) @HunterAD*100@% %i:scaleAD%</row> <row>(@MinUnits@) @HunterAD*100@% %i:scaleAD%</row>",
        champions=('TFT18_Cinderling', 'TFT18_Varus', 'TFT18_Caitlyn', 'TFT18_Rengar', 'TFT18_Tristana', 'TFT18_Aphelios', 'TFT18_Ashe'),
    ),
    "Inferno": Set18TraitInfo(
        trait_id="Inferno",
        name="Inferno",
        thresholds=(2, 3, 5, 7),
        description="Inferno damage <TFTKeyword>Burns</TFTKeyword> and <TFTKeyword>Wounds</TFTKeyword> enemies for @Duration@ seconds. Inferno Burns stack with other Burns. <row>(@MinUnits@) @HPBurnPerSecond@% Burn, @WoundPercent@% Wound</row> <row>(@MinUnits@) After combat, @LuckySlotCount@ of your shop slots without an Inferno champion ignites, rolling a champion one tier higher</row> <row>(@MinUnits@) Ignite @LuckySlotCount@ shop slots, @HPBurnPerSecond@% Burn</row> <row>(@MinUnits@) Ignite @LuckySlotCount@ shop slots, @HPBurnPerSecond@% Burn</row>",
        champions=('TFT18_Akali', 'TFT18_Varus', 'TFT18_Sett', 'TFT18_Draven'),
    ),
    "Invoker": Set18TraitInfo(
        trait_id="Invoker",
        name="Invoker",
        thresholds=(2, 3, 4, 5),
        description="Your team gains Mana Regen, increased for Invokers. <row>(@MinUnits@) @TeamManaRegen@ %i:scaleManaRegen% | @InvokerManaBonus@ %i:scaleManaRegen%</row> <row>(@MinUnits@) @TeamManaRegen@ %i:scaleManaRegen% | @InvokerManaBonus@ %i:scaleManaRegen%</row> <row>(@MinUnits@) @TeamManaRegen@ %i:scaleManaRegen% | @InvokerManaBonus@ %i:scaleManaRegen%</row> <row>(@MinUnits@) @TeamManaRegen@ %i:scaleManaRegen% | @InvokerManaBonus@ %i:scaleManaRegen%</row>",
        champions=('TFT18_Karma', 'TFT18_LeBlanc', 'TFT18_Teemo', 'TFT18_KogMaw', 'TFT18_Lillia', 'TFT18_Soraka', 'TFT18_Ivern'),
    ),
    "Juggernaut": Set18TraitInfo(
        trait_id="Juggernaut",
        name="Juggernaut",
        thresholds=(2, 4, 6),
        description="Your team gains Durability. Juggernauts gain more. <row>(@MinUnits@) @TeamDurability*100@% %i:scaleDR% or @JuggernautDurability*100@% %i:scaleDR%</row> <row>(@MinUnits@) @TeamDurability*100@% %i:scaleDR% or @JuggernautDurability*100@% %i:scaleDR%</row> <row>(@MinUnits@) @TeamDurability*100@% %i:scaleDR% or @JuggernautDurability*100@% %i:scaleDR%</row>",
        champions=('TFT18_Alistar', 'TFT18_Fiddlesticks', 'TFT18_Krug', 'TFT18_Amumu'),
    ),
    "Lunar": Set18TraitInfo(
        trait_id="Lunar",
        name="Lunar",
        thresholds=(2, 3, 4, 5),
        description="Lunar champions and adjacent allies gain Attack Speed and Ability Power. Lunar champions gain @LunarMultiplier*100@% more. <row>(@MinUnits@) @AttackSpeed*100@% %i:scaleAS%, @AbilityPower*100@% %i:scaleAP%</row> <row>(@MinUnits@) @AttackSpeed*100@% %i:scaleAS%, @AbilityPower*100@% %i:scaleAP%</row> <row>(@MinUnits@) @AttackSpeed*100@% %i:scaleAS%, @AbilityPower*100@% %i:scaleAP%</row> <row>(@MinUnits@) @AttackSpeed*100@% %i:scaleAS%, @AbilityPower*100@% %i:scaleAP%</row>",
        champions=('TFT18_Shen', 'TFT18_Diana', 'TFT18_Aphelios', 'TFT18_Alune'),
    ),
    "Monolith": Set18TraitInfo(
        trait_id="Monolith",
        name="Monolith",
        thresholds=(1,),
        description="Monoliths gain @Resists@ Armor and Magic Resist for each enemy targeting them.",
        champions=('TFT18_Gnar',),
    ),
    "Old Growth": Set18TraitInfo(
        trait_id="Old Growth",
        name="Old Growth",
        thresholds=(1,),
        description="Whenever an enemy within @UniqueTraitHexRange@ hexes dies, your strongest Old Growth champion gains @UniqueTraitHealthPerStack@ permanent max Health.  Current: 0",
        champions=('TFT18_Maokai',),
    ),
    "Primal": Set18TraitInfo(
        trait_id="Primal",
        name="Primal",
        thresholds=(2, 4),
        description="Choose one of four Primal Blessings. <row>(@MinUnits@)</row> <row>(@MinUnits@)</row>",
        champions=('TFT18_Sejuani', 'TFT18_KhaZix', 'TFT18_Rengar', 'TFT18_Nidalee', 'TFT18_Gnar'),
    ),
    "Rapidfire": Set18TraitInfo(
        trait_id="Rapidfire",
        name="Rapidfire",
        thresholds=(2, 3, 4, 5),
        description="Your team gains @TeamAS*100@% Attack Speed. Rapidfire champions gain more on every attack, up to @MaxStacks@ stacks. <row>(@MinUnits@) +@ASPerAttack*100@% %i:scaleAS% per Attack</row> <row>(@MinUnits@) +@ASPerAttack*100@% %i:scaleAS% per Attack</row> <row>(@MinUnits@) +@ASPerAttack*100@% %i:scaleAS% per Attack</row> <row>(@MinUnits@) +@ASPerAttack*100@% %i:scaleAS% per Attack</row>",
        champions=('TFT18_Xayah', 'TFT18_Yunara', 'TFT18_MasterYi', 'TFT18_Raptor', 'TFT18_Sivir'),
    ),
    "Ravager": Set18TraitInfo(
        trait_id="Ravager",
        name="Ravager",
        thresholds=(2, 4, 6),
        description="Ravagers gain @Omnivamp*100@% Omnivamp. Additionally, Ravagers deal bonus damage, doubled against units below @EnemyHealthThreshold*100@% Health. <row>(@MinUnits@) @BonusDamagePercentBase*100@% Bonus Damage</row> <row>(@MinUnits@) @BonusDamagePercentBase*100@% Bonus Damage</row> <row>(@MinUnits@) @BonusDamagePercentBase*100@% Bonus Damage</row>",
        champions=('TFT18_Akali', 'TFT18_Camille', 'TFT18_Murkwolf', 'TFT18_Warwick', 'TFT18_Diana', 'TFT18_Kennen'),
    ),
    "Riftbeast": Set18TraitInfo(
        trait_id="Riftbeast",
        name="Riftbeast",
        thresholds=(3, 5, 7, 10),
        description="<row>(@MinUnits@) Use the Alpha Mark to grant a Riftbeast their unique Buff</row> <row>(@MinUnits@) Every @OverrunCombats@ combats, your next shop is overrun with Riftbeasts.</row> <row>(@MinUnits@) On combat start and every 5 seconds Riftbeasts grow gaining  @CapstoneAD*100@% %i:scaleAD% @CapstoneAP*100@% %i:scaleAP% @CapstoneAspd*100@% %i:scaleAS% +@CapstoneArmor@ %i:scaleArmor% +@CapstoneMR@ %i:scaleMR%  +@CapstoneHealth@ %i:scaleHealth% +@CapstoneManaRegen@ %i:scaleManaRegen%</row> <row>(@MinUnits@) +@TeamSize@ maximum team size</row>",
        champions=('TFT18_Cinderling', 'TFT18_Pebbles', 'TFT18_Gromp', 'TFT18_Murkwolf', 'TFT18_Scuttlecrab', 'TFT18_Krug', 'TFT18_Raptor', 'TFT18_AncientSentinel', 'TFT18_Brambleback', 'TFT18_ElderDragon'),
    ),
    "Rival": Set18TraitInfo(
        trait_id="Rival",
        name="Rival",
        thresholds=(1, 2),
        description="<row>(@MinUnits@) Only active while fielding 1 Rival.  Rivals collect takedowns, gaining 3 if they takedown another Rival.</row> <row>(@MinUnits@) Rivals collect takedowns, gaining 3 if they takedown another Rival.  Takedowns evolve Kha'Zix, permanently granting him your choice of Executioner, Rapidfire, Ravager, or Spellweaver.  Rengar grants @RengarGold@/@RengarGold_2@/@RengarGold_3@ gold every @RengarStackReq@ takedowns. After @RengarADStackReq@ takedowns, your team gains @RengarThresholdAD*100@% Attack Damage plus @RengarADPerStack*100@% %i:scaleAD% per additional takedown.  Kha'Zix Takedowns: 0 / 0 Rengar Takedowns: 0 Gold Earned: 0</row> <row>(@MinUnits@) Rivals can be fielded together, and their abilities grant each other bonuses.</row>",
        champions=(),
    ),
    "Solar": Set18TraitInfo(
        trait_id="Solar",
        name="Solar",
        thresholds=(3,),
        description="<row>(@MinUnits@) Your champions gain a @ShieldRatio*100@% max Health shield and deal @BonusMagicDamage*100@% bonus magic damage. Gain additional bonuses for each unique 3-star  champion:  1 : Increase shield and magic damage by @PercentIncreasePer3Star*100@% for each 3-star.  @NumThreeStarThreshold1@ : @Threshold1AttackSpeed*100@% %i:scaleAS% and @Threshold1ArmorMagicResist@ %i:scaleArmor%%i:scaleMR% @NumThreeStarThreshold2@ : Convert @Threshold2TrueDamageConversion*100@% of the bonus magic damage to true damage. @NumThreeStarThreshold3@ : Every @Threshold3StarUpPeriod@ seconds of combat, a 3-star champion ascends to 4-star.</row>",
        champions=('TFT18_Leona', 'TFT18_Caitlyn', 'TFT18_Kayle', 'TFT18_Sivir', 'TFT18_Soraka', 'TFT18_Taric'),
    ),
    "Spellweaver": Set18TraitInfo(
        trait_id="Spellweaver",
        name="Spellweaver",
        thresholds=(2, 4, 6),
        description="Your team gains @TeamwideAP*100@% Ability Power. Spellweavers gain more, plus extra Ability Power whenever a Spellweaver casts an Ability. <row>(@MinUnits@) @SpellweaverAP*100@% %i:scaleAP%, +@APPerCast*100@% %i:scaleAP% per cast</row> <row>(@MinUnits@) @SpellweaverAP*100@% %i:scaleAP%, +@APPerCast*100@% %i:scaleAP% per cast</row> <row>(@MinUnits@) @SpellweaverAP*100@% %i:scaleAP%, +@APPerCast*100@% %i:scaleAP% per cast</row>",
        champions=('TFT18_Rakan', 'TFT18_Veigar', 'TFT18_Kayle', 'TFT18_Cassiopeia', 'TFT18_Ahri', 'TFT18_Ezreal', 'TFT18_Alune', 'TFT18_Lux'),
    ),
    "Sprykin": Set18TraitInfo(
        trait_id="Sprykin",
        name="Sprykin",
        thresholds=(3, 5, 7),
        description="Gain the Big Furry Friend. Drop a Sprykin on the BFF to pick its Rider. <row>(@MinUnits@) The Rider gains @HealthIncrease*100@% %i:scaleHealth% and @AttackSpeedIncrease*100@% %i:scaleAS%</row> <row>(@MinUnits@) @HealthIncrease*100@% %i:scaleHealth% @AttackSpeedIncrease*100@% %i:scaleAS%, and @TeamwideRatio*100@% of the BFF's Ability applies to your Sprykin champions.</row> <row>(@MinUnits@) @HealthIncrease*100@% %i:scaleHealth% @AttackSpeedIncrease*100@% %i:scaleAS%, and @TeamwideRatio*100@% of the BFF's Ability applies to your Sprykin champions.</row>",
        champions=('TFT18_Kobuko', 'TFT18_Teemo', 'TFT18_Yunara', 'TFT18_Tristana', 'TFT18_Kennen'),
    ),
    "Summoner": Set18TraitInfo(
        trait_id="Summoner",
        name="Summoner",
        thresholds=(2, 3),
        description="Summoners empower their summons in different ways.  Yorick: +@HealthMult*100@% Health Azir: +@DamageMult*100@% Damage Mama Beak: +@DamageMult*100@% Damage Zyra: +@NumExtraAttacks@ Plant Attacks <row>(@MinUnits@) Empower Summons</row> <row>(@MinUnits@) Improve each effect by 50%</row>",
        champions=('TFT18_Elise', 'TFT18_Azir', 'TFT18_Zyra'),
    ),
    "Thornmaiden": Set18TraitInfo(
        trait_id="Thornmaiden",
        name="Thornmaiden",
        thresholds=(1,),
        description="Your team gains @BaseDurability*100@% Durability, increased to @IncreasedDurability*100@% if @PlantNumThreshold@ or more Zyra plants are alive.",
        champions=('TFT18_Zyra',),
    ),
    "Vanguard": Set18TraitInfo(
        trait_id="Vanguard",
        name="Vanguard",
        thresholds=(2, 4, 6),
        description="At Combat Start and after dropping below @HealthThreshold*100@% <colorHealth>Health</colorHealth>, Vanguards gain a max Health Shield for @ShieldDuration@ seconds. <row>(@MinUnits@) @MaxHealthShield*100@% max Health</row> <row>(@MinUnits@) @MaxHealthShield*100@% max Health</row> <row>(@MinUnits@) @MaxHealthShield*100@% max Health. Gain @DurabilityIncrease*100@% Durability while Shielded.</row>",
        champions=('TFT18_Leona', 'TFT18_Yorick', 'TFT18_Sejuani', 'TFT18_Hecarim', 'TFT18_AncientSentinel', 'TFT18_Maokai'),
    ),
}


# =============================================================================
# SET 18 CHAMPION CATALOG (65 CHAMPIONS)
# =============================================================================

@dataclass(frozen=True, slots=True)
class Set18ChampionInfo:
    """Complete specification of a Set 18 champion."""

    champion_id: str
    name: str
    cost: int
    traits: tuple[str, ...]
    role: UnitRole
    is_carry: bool
    is_tank: bool
    hp: float
    armor: float
    mr: float
    ad: float
    aspeed: float
    range: float
    mana: float
    max_mana: float
    recommended_items: tuple[str, ...]


SET18_CHAMPION_CATALOG: tuple[Set18ChampionInfo, ...] = (
    # --- 1-Cost (14 Champions) ---
    Set18ChampionInfo("TFT18_Akali", "Akali", 1, ('Inferno', 'Adaptor', 'Ravager'), UnitRole.AD_CARRY, True, False, 500.0, 25.0, 25.0, 45.0, 0.75, 1.0, 0.0, 40.0, ('TFT_Item_InfinityEdge', 'TFT_Item_HandOfJustice', 'TFT_Item_Bloodthirster')),
    Set18ChampionInfo("TFT18_Camille", "Camille", 1, ('Coven', 'Ravager'), UnitRole.AD_CARRY, True, False, 550.0, 30.0, 30.0, 45.0, 0.70, 1.0, 0.0, 50.0, ('TFT_Item_Bloodthirster', 'TFT_Item_TitansResolve', 'TFT_Item_SteraksGage')),
    Set18ChampionInfo("TFT18_Cinderling", "Cinderling", 1, ('Riftbeast', 'Hunter'), UnitRole.AD_CARRY, True, False, 450.0, 15.0, 15.0, 45.0, 0.70, 4.0, 0.0, 40.0, ('TFT_Item_GuinsoosRageblade', 'TFT_Item_LastWhisper', 'TFT_Item_InfinityEdge')),
    Set18ChampionInfo("TFT18_Karma", "Karma", 1, ('Blossom', 'Invoker'), UnitRole.AP_CARRY, True, False, 500.0, 20.0, 20.0, 40.0, 0.65, 4.0, 0.0, 50.0, ('TFT_Item_BlueBuff', 'TFT_Item_JeweledGauntlet', 'TFT_Item_ArchangelsStaff')),
    Set18ChampionInfo("TFT18_Kobuko", "Kobuko", 1, ('Sprykin', 'Brawler'), UnitRole.TANK, False, True, 650.0, 40.0, 40.0, 45.0, 0.60, 1.0, 20.0, 80.0, ('TFT_Item_WarmogsArmor', 'TFT_Item_BrambleVest', 'TFT_Item_DragonsClaw')),
    Set18ChampionInfo("TFT18_Leona", "Leona", 1, ('Solar', 'Vanguard'), UnitRole.TANK, False, True, 650.0, 45.0, 45.0, 45.0, 0.60, 1.0, 0.0, 60.0, ('TFT_Item_GargoyleStoneplate', 'TFT_Item_SunfireCape', 'TFT_Item_DragonsClaw')),
    Set18ChampionInfo("TFT18_Ornn", "Ornn", 1, ('Elderwood', 'Defender'), UnitRole.TANK, False, True, 650.0, 45.0, 45.0, 50.0, 0.55, 1.0, 40.0, 100.0, ('TFT_Item_BrambleVest', 'TFT_Item_DragonsClaw', 'TFT_Item_WarmogsArmor')),
    Set18ChampionInfo("TFT18_Pebbles", "Pebbles", 1, ('Riftbeast', 'Defender'), UnitRole.TANK, False, True, 600.0, 45.0, 45.0, 40.0, 0.60, 1.0, 0.0, 60.0, ('TFT_Item_BrambleVest', 'TFT_Item_DragonsClaw', 'TFT_Item_SunfireCape')),
    Set18ChampionInfo("TFT18_Rakan", "Rakan", 1, ('Fae', 'Spellweaver'), UnitRole.UTILITY, False, False, 550.0, 30.0, 30.0, 40.0, 0.65, 2.0, 20.0, 70.0, ('TFT_Item_SpearOfShojin', 'TFT_Item_IonicSpark', 'TFT_Item_Crownguard')),
    Set18ChampionInfo("TFT18_RekSai", "Rek'Sai", 1, ('Blackthorn', 'Brawler'), UnitRole.TANK, False, True, 650.0, 40.0, 40.0, 50.0, 0.65, 1.0, 0.0, 60.0, ('TFT_Item_Bloodthirster', 'TFT_Item_TitansResolve', 'TFT_Item_WarmogsArmor')),
    Set18ChampionInfo("TFT18_Varus", "Varus", 1, ('Inferno', 'Hunter'), UnitRole.AD_CARRY, True, False, 500.0, 15.0, 15.0, 50.0, 0.70, 4.0, 0.0, 60.0, ('TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_Deathblade')),
    Set18ChampionInfo("TFT18_Veigar", "Veigar", 1, ('Blackthorn', 'Spellweaver'), UnitRole.AP_CARRY, True, False, 450.0, 15.0, 15.0, 35.0, 0.65, 4.0, 0.0, 45.0, ('TFT_Item_BlueBuff', 'TFT_Item_JeweledGauntlet', 'TFT_Item_RabadonsDeathcap')),
    Set18ChampionInfo("TFT18_Xayah", "Xayah", 1, ('Fae', 'Rapidfire'), UnitRole.AD_CARRY, True, False, 500.0, 15.0, 15.0, 50.0, 0.75, 4.0, 0.0, 50.0, ('TFT_Item_GuinsoosRageblade', 'TFT_Item_LastWhisper', 'TFT_Item_InfinityEdge')),
    Set18ChampionInfo("TFT18_Yorick", "Yorick", 1, ('Coven', 'Vanguard'), UnitRole.TANK, False, True, 650.0, 40.0, 40.0, 45.0, 0.60, 1.0, 20.0, 80.0, ('TFT_Item_WarmogsArmor', 'TFT_Item_BrambleVest', 'TFT_Item_SunfireCape')),
    # --- 2-Cost (13 Champions) ---
    Set18ChampionInfo("TFT18_Alistar", "Alistar", 2, ('Blossom', 'Juggernaut'), UnitRole.TANK, False, True, 750.0, 45.0, 45.0, 55.0, 0.60, 1.0, 30.0, 90.0, ('TFT_Item_WarmogsArmor', 'TFT_Item_BrambleVest', 'TFT_Item_DragonsClaw')),
    Set18ChampionInfo("TFT18_Caitlyn", "Caitlyn", 2, ('Solar', 'Hunter'), UnitRole.AD_CARRY, True, False, 550.0, 20.0, 20.0, 55.0, 0.70, 5.0, 0.0, 60.0, ('TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_SpearOfShojin')),
    Set18ChampionInfo("TFT18_Elise", "Elise", 2, ('Coven', 'Summoner'), UnitRole.UTILITY, False, False, 600.0, 30.0, 30.0, 45.0, 0.65, 2.0, 20.0, 80.0, ('TFT_Item_SpearOfShojin', 'TFT_Item_StatikkShiv', 'TFT_Item_Morellonomicon')),
    Set18ChampionInfo("TFT18_Gromp", "Gromp", 2, ('Riftbeast', 'Adaptor'), UnitRole.TANK, False, True, 750.0, 40.0, 40.0, 50.0, 0.60, 1.0, 30.0, 90.0, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_GargoyleStoneplate')),
    Set18ChampionInfo("TFT18_Kayle", "Kayle", 2, ('Solar', 'Spellweaver'), UnitRole.AP_CARRY, True, False, 550.0, 20.0, 20.0, 40.0, 0.75, 4.0, 0.0, 40.0, ('TFT_Item_GuinsoosRageblade', 'TFT_Item_JeweledGauntlet', 'TFT_Item_ArchangelsStaff')),
    Set18ChampionInfo("TFT18_LeBlanc", "LeBlanc", 2, ('Blackthorn', 'Invoker'), UnitRole.AP_CARRY, True, False, 550.0, 20.0, 20.0, 40.0, 0.65, 4.0, 0.0, 50.0, ('TFT_Item_BlueBuff', 'TFT_Item_JeweledGauntlet', 'TFT_Item_RabadonsDeathcap')),
    Set18ChampionInfo("TFT18_Murkwolf", "Murkwolf", 2, ('Riftbeast', 'Ravager'), UnitRole.AD_CARRY, True, False, 600.0, 30.0, 30.0, 55.0, 0.80, 1.0, 0.0, 40.0, ('TFT_Item_InfinityEdge', 'TFT_Item_Bloodthirster', 'TFT_Item_HandOfJustice')),
    Set18ChampionInfo("TFT18_Scuttlecrab", "Scuttlecrab", 2, ('Riftbeast', 'Brawler'), UnitRole.TANK, False, True, 750.0, 40.0, 40.0, 45.0, 0.60, 1.0, 0.0, 80.0, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_Redemption')),
    Set18ChampionInfo("TFT18_Sejuani", "Sejuani", 2, ('Primal', 'Vanguard'), UnitRole.TANK, False, True, 750.0, 45.0, 45.0, 50.0, 0.60, 1.0, 40.0, 100.0, ('TFT_Item_BrambleVest', 'TFT_Item_DragonsClaw', 'TFT_Item_WarmogsArmor')),
    Set18ChampionInfo("TFT18_Shen", "Shen", 2, ('Lunar', 'Defender'), UnitRole.TANK, False, True, 750.0, 45.0, 45.0, 50.0, 0.60, 1.0, 40.0, 90.0, ('TFT_Item_GargoyleStoneplate', 'TFT_Item_BrambleVest', 'TFT_Item_DragonsClaw')),
    Set18ChampionInfo("TFT18_Teemo", "Teemo", 2, ('Sprykin', 'Invoker'), UnitRole.AP_CARRY, True, False, 550.0, 20.0, 20.0, 40.0, 0.70, 4.0, 0.0, 50.0, ('TFT_Item_BlueBuff', 'TFT_Item_Morellonomicon', 'TFT_Item_JeweledGauntlet')),
    Set18ChampionInfo("TFT18_Warwick", "Warwick", 2, ('Blackthorn', 'Ravager'), UnitRole.AD_CARRY, True, False, 650.0, 35.0, 35.0, 55.0, 0.80, 1.0, 0.0, 50.0, ('TFT_Item_Bloodthirster', 'TFT_Item_TitansResolve', 'TFT_Item_SteraksGage')),
    Set18ChampionInfo("TFT18_Yunara", "Yunara", 2, ('Sprykin', 'Rapidfire'), UnitRole.AD_CARRY, True, False, 550.0, 20.0, 20.0, 50.0, 0.75, 4.0, 0.0, 45.0, ('TFT_Item_GuinsoosRageblade', 'TFT_Item_LastWhisper', 'TFT_Item_InfinityEdge')),
    # --- 3-Cost (14 Champions) ---
    Set18ChampionInfo("TFT18_Azir", "Azir", 3, ('Blackthorn', 'Summoner'), UnitRole.AP_CARRY, True, False, 650.0, 25.0, 25.0, 40.0, 0.75, 4.0, 20.0, 70.0, ('TFT_Item_GuinsoosRageblade', 'TFT_Item_ArchangelsStaff', 'TFT_Item_HextechGunblade')),
    Set18ChampionInfo("TFT18_Cassiopeia", "Cassiopeia", 3, ('Coven', 'Spellweaver'), UnitRole.AP_CARRY, True, False, 650.0, 25.0, 25.0, 40.0, 0.75, 4.0, 0.0, 40.0, ('TFT_Item_BlueBuff', 'TFT_Item_JeweledGauntlet', 'TFT_Item_Morellonomicon')),
    Set18ChampionInfo("TFT18_Diana", "Diana", 3, ('Lunar', 'Ravager'), UnitRole.AD_CARRY, True, False, 750.0, 35.0, 35.0, 60.0, 0.75, 1.0, 20.0, 70.0, ('TFT_Item_Bloodthirster', 'TFT_Item_TitansResolve', 'TFT_Item_HandOfJustice')),
    Set18ChampionInfo("TFT18_Fiddlesticks", "Fiddlesticks", 3, ('Coven', 'Juggernaut'), UnitRole.TANK, False, True, 800.0, 45.0, 45.0, 50.0, 0.60, 1.0, 30.0, 90.0, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_Morellonomicon')),
    Set18ChampionInfo("TFT18_Hecarim", "Hecarim", 3, ('Elderwood', 'Vanguard'), UnitRole.TANK, False, True, 800.0, 50.0, 50.0, 60.0, 0.65, 1.0, 40.0, 100.0, ('TFT_Item_BrambleVest', 'TFT_Item_DragonsClaw', 'TFT_Item_WarmogsArmor')),
    Set18ChampionInfo("TFT18_KhaZix", "Kha'Zix", 3, ('Primal', 'Executioner'), UnitRole.AD_CARRY, True, False, 700.0, 30.0, 30.0, 65.0, 0.80, 1.0, 0.0, 40.0, ('TFT_Item_InfinityEdge', 'TFT_Item_HandOfJustice', 'TFT_Item_EdgeOfNight')),
    Set18ChampionInfo("TFT18_KogMaw", "Kog'Maw", 3, ('Caustic', 'Invoker', 'Adaptor'), UnitRole.AP_CARRY, True, False, 600.0, 25.0, 25.0, 45.0, 0.70, 4.0, 0.0, 40.0, ('TFT_Item_BlueBuff', 'TFT_Item_JeweledGauntlet', 'TFT_Item_HextechGunblade')),
    Set18ChampionInfo("TFT18_Krug", "Krug", 3, ('Riftbeast', 'Juggernaut'), UnitRole.TANK, False, True, 850.0, 50.0, 50.0, 55.0, 0.60, 1.0, 40.0, 100.0, ('TFT_Item_WarmogsArmor', 'TFT_Item_BrambleVest', 'TFT_Item_DragonsClaw')),
    Set18ChampionInfo("TFT18_MasterYi", "Master Yi", 3, ('Blossom', 'Adaptor', 'Rapidfire'), UnitRole.AD_CARRY, True, False, 750.0, 35.0, 35.0, 65.0, 0.85, 1.0, 0.0, 50.0, ('TFT_Item_GuinsoosRageblade', 'TFT_Item_Bloodthirster', 'TFT_Item_TitansResolve')),
    Set18ChampionInfo("TFT18_Rammus", "Rammus", 3, ('Elderwood', 'Defender'), UnitRole.TANK, False, True, 850.0, 55.0, 55.0, 50.0, 0.60, 1.0, 40.0, 100.0, ('TFT_Item_BrambleVest', 'TFT_Item_DragonsClaw', 'TFT_Item_SunfireCape')),
    Set18ChampionInfo("TFT18_Raptor", "Raptor", 3, ('Riftbeast', 'Rapidfire'), UnitRole.AD_CARRY, True, False, 650.0, 25.0, 25.0, 60.0, 0.80, 4.0, 0.0, 40.0, ('TFT_Item_GuinsoosRageblade', 'TFT_Item_LastWhisper', 'TFT_Item_InfinityEdge')),
    Set18ChampionInfo("TFT18_Rengar", "Rengar", 3, ('Primal', 'Hunter'), UnitRole.AD_CARRY, True, False, 700.0, 30.0, 30.0, 65.0, 0.80, 1.0, 0.0, 50.0, ('TFT_Item_InfinityEdge', 'TFT_Item_Bloodthirster', 'TFT_Item_LastWhisper')),
    Set18ChampionInfo("TFT18_Tristana", "Tristana", 3, ('Sprykin', 'Hunter'), UnitRole.AD_CARRY, True, False, 650.0, 25.0, 25.0, 60.0, 0.75, 4.0, 0.0, 50.0, ('TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_Deathblade')),
    Set18ChampionInfo("TFT18_Vi", "Vi", 3, ('Blossom', 'Brawler'), UnitRole.TANK, False, True, 800.0, 45.0, 45.0, 60.0, 0.65, 1.0, 30.0, 80.0, ('TFT_Item_Bloodthirster', 'TFT_Item_TitansResolve', 'TFT_Item_WarmogsArmor')),
    # --- 4-Cost (14 Champions) ---
    Set18ChampionInfo("TFT18_Ahri", "Ahri", 4, ('Blossom', 'Spellweaver'), UnitRole.AP_CARRY, True, False, 750.0, 30.0, 30.0, 45.0, 0.75, 4.0, 20.0, 70.0, ('TFT_Item_BlueBuff', 'TFT_Item_JeweledGauntlet', 'TFT_Item_HextechGunblade')),
    Set18ChampionInfo("TFT18_Amumu", "Amumu", 4, ('Elderwood', 'Juggernaut'), UnitRole.TANK, False, True, 950.0, 55.0, 55.0, 55.0, 0.60, 1.0, 50.0, 120.0, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_BrambleVest')),
    Set18ChampionInfo("TFT18_AncientSentinel", "Ancient Sentinel", 4, ('Riftbeast', 'Vanguard'), UnitRole.TANK, False, True, 950.0, 60.0, 60.0, 60.0, 0.60, 1.0, 50.0, 120.0, ('TFT_Item_GargoyleStoneplate', 'TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw')),
    Set18ChampionInfo("TFT18_Aphelios", "Aphelios", 4, ('Lunar', 'Hunter'), UnitRole.AD_CARRY, True, False, 750.0, 30.0, 30.0, 70.0, 0.80, 4.0, 0.0, 70.0, ('TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_GuinsoosRageblade')),
    Set18ChampionInfo("TFT18_Brambleback", "Brambleback", 4, ('Riftbeast', 'Defender'), UnitRole.TANK, False, True, 950.0, 60.0, 60.0, 55.0, 0.60, 1.0, 40.0, 110.0, ('TFT_Item_BrambleVest', 'TFT_Item_SunfireCape', 'TFT_Item_DragonsClaw')),
    Set18ChampionInfo("TFT18_Ezreal", "Ezreal", 4, ('Fae', 'Spellweaver'), UnitRole.AP_CARRY, True, False, 750.0, 30.0, 30.0, 50.0, 0.80, 4.0, 0.0, 40.0, ('TFT_Item_BlueBuff', 'TFT_Item_JeweledGauntlet', 'TFT_Item_RabadonsDeathcap')),
    Set18ChampionInfo("TFT18_Lillia", "Lillia", 4, ('Blossom', 'Invoker'), UnitRole.AP_CARRY, True, False, 750.0, 30.0, 30.0, 45.0, 0.75, 4.0, 20.0, 60.0, ('TFT_Item_SpearOfShojin', 'TFT_Item_JeweledGauntlet', 'TFT_Item_ArchangelsStaff')),
    Set18ChampionInfo("TFT18_Malphite", "Malphite", 4, ('Blackthorn', 'Defender'), UnitRole.TANK, False, True, 950.0, 60.0, 60.0, 60.0, 0.60, 1.0, 50.0, 120.0, ('TFT_Item_GargoyleStoneplate', 'TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw')),
    Set18ChampionInfo("TFT18_Morgana", "Morgana", 4, ('Coven', 'Executioner'), UnitRole.AP_CARRY, True, False, 750.0, 30.0, 30.0, 50.0, 0.75, 4.0, 30.0, 80.0, ('TFT_Item_SpearOfShojin', 'TFT_Item_Morellonomicon', 'TFT_Item_JeweledGauntlet')),
    Set18ChampionInfo("TFT18_Nidalee", "Nidalee", 4, ('Primal', 'Adaptor'), UnitRole.AD_CARRY, True, False, 800.0, 35.0, 35.0, 70.0, 0.85, 4.0, 0.0, 50.0, ('TFT_Item_GuinsoosRageblade', 'TFT_Item_Bloodthirster', 'TFT_Item_TitansResolve')),
    Set18ChampionInfo("TFT18_Sett", "Sett", 4, ('Inferno', 'Brawler'), UnitRole.TANK, False, True, 950.0, 55.0, 55.0, 65.0, 0.65, 1.0, 40.0, 100.0, ('TFT_Item_Bloodthirster', 'TFT_Item_TitansResolve', 'TFT_Item_WarmogsArmor')),
    Set18ChampionInfo("TFT18_Sivir", "Sivir", 4, ('Solar', 'Rapidfire'), UnitRole.AD_CARRY, True, False, 750.0, 30.0, 30.0, 70.0, 0.80, 4.0, 0.0, 60.0, ('TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_GuinsoosRageblade')),
    Set18ChampionInfo("TFT18_Soraka", "Soraka", 4, ('Solar', 'Invoker'), UnitRole.UTILITY, False, False, 750.0, 30.0, 30.0, 45.0, 0.75, 4.0, 30.0, 80.0, ('TFT_Item_SpearOfShojin', 'TFT_Item_ArchangelsStaff', 'TFT_Item_HextechGunblade')),
    Set18ChampionInfo("TFT18_Zyra", "Zyra", 4, ('Thornmaiden', 'Flora Fatalis', 'Summoner'), UnitRole.AP_CARRY, True, False, 750.0, 30.0, 30.0, 50.0, 0.75, 4.0, 20.0, 70.0, ('TFT_Item_SpearOfShojin', 'TFT_Item_JeweledGauntlet', 'TFT_Item_Morellonomicon')),
    # --- 5-Cost (10 Champions) ---
    Set18ChampionInfo("TFT18_Alune", "Alune", 5, ('Attuned', 'Lunar', 'Spellweaver'), UnitRole.AP_CARRY, True, False, 850.0, 40.0, 40.0, 55.0, 0.80, 4.0, 30.0, 90.0, ('TFT_Item_SpearOfShojin', 'TFT_Item_JeweledGauntlet', 'TFT_Item_RabadonsDeathcap')),
    Set18ChampionInfo("TFT18_Ashe", "Ashe", 5, ('Fae', 'Hunter'), UnitRole.AD_CARRY, True, False, 850.0, 35.0, 35.0, 75.0, 0.85, 4.0, 0.0, 60.0, ('TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_GuinsoosRageblade')),
    Set18ChampionInfo("TFT18_Draven", "Draven", 5, ('Bounty Seeker', 'Inferno', 'Executioner'), UnitRole.AD_CARRY, True, False, 850.0, 40.0, 40.0, 80.0, 0.85, 4.0, 0.0, 50.0, ('TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_Bloodthirster')),
    Set18ChampionInfo("TFT18_ElderDragon", "The Elder Dragon", 5, ('Apex Predator', 'Riftbeast'), UnitRole.TANK, False, True, 1800.0, 80.0, 80.0, 90.0, 0.70, 2.0, 60.0, 150.0, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_BrambleVest')),
    Set18ChampionInfo("TFT18_Gnar", "Gnar", 5, ('Primal', 'Brawler', 'Monolith'), UnitRole.TANK, False, True, 1100.0, 60.0, 60.0, 75.0, 0.75, 1.0, 50.0, 110.0, ('TFT_Item_Bloodthirster', 'TFT_Item_TitansResolve', 'TFT_Item_SteraksGage')),
    Set18ChampionInfo("TFT18_Ivern", "Ivern", 5, ('Greenfather', 'Elderwood', 'Invoker'), UnitRole.UTILITY, False, False, 900.0, 45.0, 45.0, 55.0, 0.75, 4.0, 40.0, 100.0, ('TFT_Item_SpearOfShojin', 'TFT_Item_StatikkShiv', 'TFT_Item_Redemption')),
    Set18ChampionInfo("TFT18_Kennen", "Kennen", 5, ('Sprykin', 'Defender', 'Ravager'), UnitRole.AP_CARRY, True, False, 900.0, 45.0, 45.0, 60.0, 0.80, 2.0, 40.0, 100.0, ('TFT_Item_JeweledGauntlet', 'TFT_Item_IonicSpark', 'TFT_Item_Morellonomicon')),
    Set18ChampionInfo("TFT18_Lux", "Lux", 5, ('Avatar', 'Spellweaver'), UnitRole.AP_CARRY, True, False, 850.0, 35.0, 35.0, 55.0, 0.80, 4.0, 20.0, 70.0, ('TFT_Item_BlueBuff', 'TFT_Item_JeweledGauntlet', 'TFT_Item_RabadonsDeathcap')),
    Set18ChampionInfo("TFT18_Maokai", "Maokai", 5, ('Old Growth', 'Elderwood', 'Vanguard'), UnitRole.TANK, False, True, 1100.0, 65.0, 65.0, 65.0, 0.65, 1.0, 50.0, 120.0, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_BrambleVest')),
    Set18ChampionInfo("TFT18_Taric", "Taric", 5, ('Emerald Aspect', 'Solar', 'Defender'), UnitRole.TANK, False, True, 1100.0, 70.0, 70.0, 65.0, 0.65, 1.0, 50.0, 120.0, ('TFT_Item_GargoyleStoneplate', 'TFT_Item_DragonsClaw', 'TFT_Item_WarmogsArmor')),
)


# =============================================================================
# SET 18 ITEMS & RECIPES CATALOG (111 ITEMS, 45 RECIPES)
# =============================================================================

@dataclass(frozen=True, slots=True)
class Set18ItemInfo:
    """Official Item definition with recipe composition."""

    item_id: str
    name: str
    is_component: bool
    recipe: tuple[str, ...]
    description: str


SET18_ITEMS_CATALOG: dict[str, Set18ItemInfo] = {
    "TFT13_Item_AutomataEmblemItem": Set18ItemInfo(
        item_id="TFT13_Item_AutomataEmblemItem",
        name="Automata Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_NegatronCloak'),
        description="The holder gains the Automata trait.",
    ),
    "TFT13_Item_CabalEmblemItem": Set18ItemInfo(
        item_id="TFT13_Item_CabalEmblemItem",
        name="Black Rose Emblem",
        is_component=False,
        recipe=('TFT_Item_NeedlesslyLargeRod', 'TFT_Item_Spatula'),
        description="The holder gains the Black Rose trait.",
    ),
    "TFT13_Item_ExperimentEmblemItem": Set18ItemInfo(
        item_id="TFT13_Item_ExperimentEmblemItem",
        name="Experiment Emblem",
        is_component=False,
        recipe=('TFT_Item_GiantsBelt', 'TFT_Item_Spatula'),
        description="The holder gains the Experiment trait.  <spellActive enabled=TFT13_ExperimentActive alternate=rules>Experiment Bonus: After @SplitTime@ seconds or on death, summon a copy of this unit with <TFTBonus enabled=TFT13_ExperimentActive alternate=rules>@TFTUnitProperty.:TFT13_EmblemCurrentExperimentBonus@%</TFTBonus> max Health.</spellActive>",
    ),
    "TFT13_Item_FamilyEmblemItem": Set18ItemInfo(
        item_id="TFT13_Item_FamilyEmblemItem",
        name="Family Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_TearOfTheGoddess'),
        description="The holder gains the Family trait.",
    ),
    "TFT13_Item_HoverboardEmblemItem": Set18ItemInfo(
        item_id="TFT13_Item_HoverboardEmblemItem",
        name="Firelight Emblem",
        is_component=False,
        recipe=('TFT_Item_SparringGloves', 'TFT_Item_Spatula'),
        description="The holder gains the Firelight trait.",
    ),
    "TFT13_Item_RebelEmblemItem": Set18ItemInfo(
        item_id="TFT13_Item_RebelEmblemItem",
        name="Rebel Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_RecurveBow'),
        description="The holder gains the Rebel trait.",
    ),
    "TFT13_Item_SquadEmblemItem": Set18ItemInfo(
        item_id="TFT13_Item_SquadEmblemItem",
        name="Enforcer Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_ChainVest'),
        description="The holder gains the Enforcer trait.",
    ),
    "TFT13_Item_WarbandEmblemItem": Set18ItemInfo(
        item_id="TFT13_Item_WarbandEmblemItem",
        name="Conqueror Emblem",
        is_component=False,
        recipe=('TFT_Item_BFSword', 'TFT_Item_Spatula'),
        description="The holder gains the Conqueror trait.",
    ),
    "TFT14_Item_BallistekEmblemItem": Set18ItemInfo(
        item_id="TFT14_Item_BallistekEmblemItem",
        name="BoomBot Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_ChainVest'),
        description="The holder gains the BoomBot trait.",
    ),
    "TFT14_Item_ControllerEmblemItem": Set18ItemInfo(
        item_id="TFT14_Item_ControllerEmblemItem",
        name="Strategist Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_BFSword'),
        description="The holder gains the Strategist trait and @SecondaryEffectiveness*100@% of the other Strategist bonus.",
    ),
    "TFT14_Item_DarkWebEmblemItem": Set18ItemInfo(
        item_id="TFT14_Item_DarkWebEmblemItem",
        name="Anima Squad Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_TearOfTheGoddess'),
        description="The holder gains the Anima Squad trait.",
    ),
    "TFT14_Item_DivinicorpEmblemItem": Set18ItemInfo(
        item_id="TFT14_Item_DivinicorpEmblemItem",
        name="Divinicorp Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_RecurveBow'),
        description="The holder gains the Divinicorp trait.",
    ),
    "TFT14_Item_EdgeRunnerEmblemItem": Set18ItemInfo(
        item_id="TFT14_Item_EdgeRunnerEmblemItem",
        name="Exotech Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_SparringGloves'),
        description="The holder gains the Exotech trait.",
    ),
    "TFT14_Item_ImmortalEmblemItem": Set18ItemInfo(
        item_id="TFT14_Item_ImmortalEmblemItem",
        name="Golden Ox Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_GiantsBelt'),
        description="The holder gains the Golden Ox trait.",
    ),
    "TFT14_Item_MobEmblemItem": Set18ItemInfo(
        item_id="TFT14_Item_MobEmblemItem",
        name="Syndicate Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_NegatronCloak'),
        description="The holder gains the Syndicate trait.  <scaleLevel enabled=TFT14_Mob_IsActive_T1 alternate=TFTGuildInactive>Kingpin bonus: <ShowIfNot.TFT14_Mob_IsActive_T3><TFTBonus>+@AttackSpeed_T1*100@%</TFTBonus> Attack Speed and gain <TFTBonus>@Shield_T1*100@%</TFTBonus> max Health Shield at combat start.</ShowIfNot.TFT14_Mob_IsActive_T3><ShowIf.TFT14_Mob_IsActive_T3><TFTBonus>+@AttackSpeed_T2*100@%</TFTBonus> Attack Speed and gain <TFTBonus>@Shield_T2*100@%</TFTBonus> max Health Shield at combat start.</ShowIf.TFT14_Mob_IsActive_T3></scaleLevel>",
    ),
    "TFT14_Item_StreetDemonEmblemItem": Set18ItemInfo(
        item_id="TFT14_Item_StreetDemonEmblemItem",
        name="Street Demon Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_NeedlesslyLargeRod'),
        description="The holder gains the Street Demon trait.",
    ),
    "TFT15_Item_BattleAcademiaEmblemItem": Set18ItemInfo(
        item_id="TFT15_Item_BattleAcademiaEmblemItem",
        name="Battle Academia Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_TearOfTheGoddess'),
        description="The holder gains the Battle Academia trait, and @HealthPerPotential@ Health and @DAPerPotential*100@% Damage Amp per %i:set14AmpIcon%.",
    ),
    "TFT15_Item_CrystalRoseEmblemItem": Set18ItemInfo(
        item_id="TFT15_Item_CrystalRoseEmblemItem",
        name="Crystal Gambit Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_GiantsBelt'),
        description="The holder gains the Crystal Gambit trait.",
    ),
    "TFT15_Item_EmpyreanEmblemItem": Set18ItemInfo(
        item_id="TFT15_Item_EmpyreanEmblemItem",
        name="Wraith Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_ChainVest'),
        description="The holder gains the Wraith trait.",
    ),
    "TFT15_Item_RingKingsEmblemItem": Set18ItemInfo(
        item_id="TFT15_Item_RingKingsEmblemItem",
        name="Luchador Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_SparringGloves'),
        description="The holder gains the Luchador trait.",
    ),
    "TFT15_Item_ShotcallerEmblemItem": Set18ItemInfo(
        item_id="TFT15_Item_ShotcallerEmblemItem",
        name="Strategist Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_NegatronCloak'),
        description="The holder gains the Strategist trait.  Combat Start: The holder and allies within @NumHexes@ hex in the same row gain bonuses. Front 2 rows: @ResistBuff@ Armor and Magic Resist Back 2 rows: @ASBuff*100@% Attack Speed",
    ),
    "TFT15_Item_SoulFighterEmblemItem": Set18ItemInfo(
        item_id="TFT15_Item_SoulFighterEmblemItem",
        name="Soul Fighter Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_BFSword'),
        description="The holder gains the Soul Fighter trait.",
    ),
    "TFT15_Item_StarGuardianEmblemItem": Set18ItemInfo(
        item_id="TFT15_Item_StarGuardianEmblemItem",
        name="Star Guardian Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_NeedlesslyLargeRod'),
        description="The holder gains the Star Guardian trait.  <TFTBonus>Teamwork: </TFTBonus> Non-emblem Star Guardian bonuses are increased by @Percent@%. This effect stacks.  <tftitemrules>[Unique - only 1 per champion.]</tftitemrules>",
    ),
    "TFT15_Item_SupremeCellsEmblemItem": Set18ItemInfo(
        item_id="TFT15_Item_SupremeCellsEmblemItem",
        name="Supreme Cells Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_RecurveBow'),
        description="The holder gains the Supreme Cells trait.",
    ),
    "TFT16_Item_BilgewaterEmblemItem": Set18ItemInfo(
        item_id="TFT16_Item_BilgewaterEmblemItem",
        name="Bilgewater Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_TearOfTheGoddess'),
        description="The holder gains the Bilgewater trait.",
    ),
    "TFT16_Item_DemaciaEmblemItem": Set18ItemInfo(
        item_id="TFT16_Item_DemaciaEmblemItem",
        name="Demacia Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_ChainVest'),
        description="The holder gains the Demacia trait.",
    ),
    "TFT16_Item_FreljordEmblemItem": Set18ItemInfo(
        item_id="TFT16_Item_FreljordEmblemItem",
        name="Freljord Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_GiantsBelt'),
        description="The holder gains the Freljord trait.",
    ),
    "TFT16_Item_IoniaEmblemItem": Set18ItemInfo(
        item_id="TFT16_Item_IoniaEmblemItem",
        name="Ionia Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_NeedlesslyLargeRod'),
        description="The holder gains the Ionia trait.",
    ),
    "TFT16_Item_NoxusEmblemItem": Set18ItemInfo(
        item_id="TFT16_Item_NoxusEmblemItem",
        name="Noxus Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_BFSword'),
        description="The holder gains the Noxus trait.",
    ),
    "TFT16_Item_VoidEmblemItem": Set18ItemInfo(
        item_id="TFT16_Item_VoidEmblemItem",
        name="Void Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_RecurveBow'),
        description="The holder gains the Void trait.",
    ),
    "TFT16_Item_YordleEmblemItem": Set18ItemInfo(
        item_id="TFT16_Item_YordleEmblemItem",
        name="Yordle Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_NegatronCloak'),
        description="The holder gains the Yordle trait.",
    ),
    "TFT16_Item_ZaunEmblemItem": Set18ItemInfo(
        item_id="TFT16_Item_ZaunEmblemItem",
        name="Zaun Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_SparringGloves'),
        description="The holder gains the Zaun trait.",
    ),
    "TFT17_Item_AstronautEmblemItem": Set18ItemInfo(
        item_id="TFT17_Item_AstronautEmblemItem",
        name="Meeple Emblem",
        is_component=False,
        recipe=('TFT_Item_ChainVest', 'TFT_Item_Spatula'),
        description="The holder gains the Meeple trait.",
    ),
    "TFT17_Item_DRXEmblemItem": Set18ItemInfo(
        item_id="TFT17_Item_DRXEmblemItem",
        name="N.O.V.A. Emblem",
        is_component=False,
        recipe=('TFT_Item_SparringGloves', 'TFT_Item_Spatula'),
        description="The holder gains the N.O.V.A. trait.",
    ),
    "TFT17_Item_DarkStarEmblemItem": Set18ItemInfo(
        item_id="TFT17_Item_DarkStarEmblemItem",
        name="Dark Star Emblem",
        is_component=False,
        recipe=('TFT_Item_BFSword', 'TFT_Item_Spatula'),
        description="The holder gains the Dark Star trait.",
    ),
    "TFT17_Item_FavoredEmblemItem": Set18ItemInfo(
        item_id="TFT17_Item_FavoredEmblemItem",
        name="Arbiter Emblem",
        is_component=False,
        recipe=('TFT_Item_NegatronCloak', 'TFT_Item_Spatula'),
        description="The holder gains the Arbiter trait.",
    ),
    "TFT17_Item_PrimordianEmblemItem": Set18ItemInfo(
        item_id="TFT17_Item_PrimordianEmblemItem",
        name="Primordian Emblem",
        is_component=False,
        recipe=('TFT_Item_GiantsBelt', 'TFT_Item_Spatula'),
        description="The holder gains the Primordian trait.",
    ),
    "TFT17_Item_PulsefireEmblemItem": Set18ItemInfo(
        item_id="TFT17_Item_PulsefireEmblemItem",
        name="Timebreaker Emblem",
        is_component=False,
        recipe=('TFT_Item_RecurveBow', 'TFT_Item_Spatula'),
        description="The holder gains the Timebreaker trait.",
    ),
    "TFT17_Item_SpaceGrooveEmblemItem": Set18ItemInfo(
        item_id="TFT17_Item_SpaceGrooveEmblemItem",
        name="Space Groove Emblem",
        is_component=False,
        recipe=('TFT_Item_TearOfTheGoddess', 'TFT_Item_Spatula'),
        description="The holder gains the Space Groove trait.",
    ),
    "TFT17_Item_StargazerEmblemItem": Set18ItemInfo(
        item_id="TFT17_Item_StargazerEmblemItem",
        name="Stargazer Emblem",
        is_component=False,
        recipe=('TFT_Item_NeedlesslyLargeRod', 'TFT_Item_Spatula'),
        description="The holder gains the Stargazer trait.",
    ),
    "TFT3_Item_BattlecastEmblem": Set18ItemInfo(
        item_id="TFT3_Item_BattlecastEmblem",
        name="Battlecast Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_NeedlesslyLargeRod'),
        description="The holder gains the Battlecast trait.",
    ),
    "TFT3_Item_BattlecastSpatulaItem": Set18ItemInfo(
        item_id="TFT3_Item_BattlecastSpatulaItem",
        name="Battlecast Plating",
        is_component=False,
        recipe=('TFT_Item_NeedlesslyLargeRod', 'TFT_Item_Spatula'),
        description="The wearer gains the Battlecast trait.  <tftitemrules>[Unique - Only One Per Champion]</tftitemrules>",
    ),
    "TFT3_Item_BlademasterEmblem": Set18ItemInfo(
        item_id="TFT3_Item_BlademasterEmblem",
        name="Blademaster Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_BFSword'),
        description="The holder gains the Blademaster trait.",
    ),
    "TFT3_Item_BlademasterSpatulaItem": Set18ItemInfo(
        item_id="TFT3_Item_BlademasterSpatulaItem",
        name="tft_item_name_UmbralGlaive",
        is_component=False,
        recipe=('TFT_Item_BFSword', 'TFT_Item_Spatula'),
        description="tft_item_description_SlicerSpatulaItem",
    ),
    "TFT3_Item_CelestialEmblem": Set18ItemInfo(
        item_id="TFT3_Item_CelestialEmblem",
        name="Celestial Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_NegatronCloak'),
        description="The holder gains the Celestial trait.",
    ),
    "TFT3_Item_CelestialSpatulaItem": Set18ItemInfo(
        item_id="TFT3_Item_CelestialSpatulaItem",
        name="tft_item_name_CelestialSpatulaItem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_NegatronCloak'),
        description="tft_item_description_CelestialSpatulaItem",
    ),
    "TFT3_Item_DarkStarEmblem": Set18ItemInfo(
        item_id="TFT3_Item_DarkStarEmblem",
        name="Dark Star Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_SparringGloves'),
        description="The holder gains the Dark Star trait.",
    ),
    "TFT3_Item_DarkStarSpatulaItem": Set18ItemInfo(
        item_id="TFT3_Item_DarkStarSpatulaItem",
        name="tft_item_name_DarkStarSpatulaItem",
        is_component=False,
        recipe=('TFT_Item_SparringGloves', 'TFT_Item_Spatula'),
        description="tft_item_description_DarkStarSpatulaItem",
    ),
    "TFT3_Item_InfiltratorEmblem": Set18ItemInfo(
        item_id="TFT3_Item_InfiltratorEmblem",
        name="Infiltrator Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_RecurveBow'),
        description="The holder gains the Infiltrator trait.",
    ),
    "TFT3_Item_InfiltratorSpatulaItem": Set18ItemInfo(
        item_id="TFT3_Item_InfiltratorSpatulaItem",
        name="tft_item_name_InfiltratorSpatulaItem",
        is_component=False,
        recipe=('TFT_Item_RecurveBow', 'TFT_Item_Spatula'),
        description="tft_item_description_InfiltratorSpatulaItem",
    ),
    "TFT3_Item_ProtectorEmblem": Set18ItemInfo(
        item_id="TFT3_Item_ProtectorEmblem",
        name="Protector Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_GiantsBelt'),
        description="The holder gains the Protector trait.",
    ),
    "TFT3_Item_ProtectorSpatulaItem": Set18ItemInfo(
        item_id="TFT3_Item_ProtectorSpatulaItem",
        name="Protector's Chestguard",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_GiantsBelt'),
        description="The holder gains the Protector trait.  <tftitemrules>[Unique - only 1 per champion]</tftitemrules>",
    ),
    "TFT3_Item_RebelEmblem": Set18ItemInfo(
        item_id="TFT3_Item_RebelEmblem",
        name="Rebel Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_ChainVest'),
        description="The holder gains the Rebel trait.",
    ),
    "TFT3_Item_RebelSpatulaItem": Set18ItemInfo(
        item_id="TFT3_Item_RebelSpatulaItem",
        name="tft_item_name_RebelSpatulaItem",
        is_component=False,
        recipe=('TFT_Item_ChainVest', 'TFT_Item_Spatula'),
        description="tft_item_description_RebelSpatulaItem",
    ),
    "TFT3_Item_StarGuardianEmblem": Set18ItemInfo(
        item_id="TFT3_Item_StarGuardianEmblem",
        name="Star Guardian Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_TearOfTheGoddess'),
        description="The holder gains the Star Guardian trait.",
    ),
    "TFT3_Item_StarGuardianSpatulaItem": Set18ItemInfo(
        item_id="TFT3_Item_StarGuardianSpatulaItem",
        name="tft_item_name_StarGuardianSpatulaItem",
        is_component=False,
        recipe=('TFT_Item_TearOfTheGoddess', 'TFT_Item_Spatula'),
        description="tft_item_description_StarGuardianSpatulaItem",
    ),
    "TFT7_Item_DarkflightEmblemItem": Set18ItemInfo(
        item_id="TFT7_Item_DarkflightEmblemItem",
        name="Darkflight Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_ChainVest'),
        description="The holder gains the Darkflight trait.",
    ),
    "TFT7_Item_GuildEmblemItem": Set18ItemInfo(
        item_id="TFT7_Item_GuildEmblemItem",
        name="Guild Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_NeedlesslyLargeRod'),
        description="The holder gains the Guild trait.  <TFTDebonairVIP>Guild Bonus: @Omnivamp@% Omnivamp</TFTDebonairVIP>",
    ),
    "TFT7_Item_JadeEmblemItem": Set18ItemInfo(
        item_id="TFT7_Item_JadeEmblemItem",
        name="Jade Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_TearOfTheGoddess'),
        description="The holder gains the Jade trait.",
    ),
    "TFT7_Item_LagoonEmblemItem": Set18ItemInfo(
        item_id="TFT7_Item_LagoonEmblemItem",
        name="Lagoon Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_SparringGloves'),
        description="The holder gains the Lagoon trait.",
    ),
    "TFT7_Item_MirageEmblemItem": Set18ItemInfo(
        item_id="TFT7_Item_MirageEmblemItem",
        name="Mirage Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_NegatronCloak'),
        description="The holder gains the Mirage trait.",
    ),
    "TFT7_Item_ShimmerscaleEmblemItem": Set18ItemInfo(
        item_id="TFT7_Item_ShimmerscaleEmblemItem",
        name="Shimmerscale Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_BFSword'),
        description="The holder gains the Shimmerscale trait.",
    ),
    "TFT7_Item_TempestEmblemItem": Set18ItemInfo(
        item_id="TFT7_Item_TempestEmblemItem",
        name="Tempest Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_RecurveBow'),
        description="The holder gains the Tempest trait.",
    ),
    "TFT7_Item_WhispersEmblemItem": Set18ItemInfo(
        item_id="TFT7_Item_WhispersEmblemItem",
        name="Whispers Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_GiantsBelt'),
        description="The holder gains the Whispers trait.",
    ),
    "TFT_Item_AdaptiveHelm": Set18ItemInfo(
        item_id="TFT_Item_AdaptiveHelm",
        name="Adaptive Helm",
        is_component=False,
        recipe=('TFT_Item_NegatronCloak', 'TFT_Item_TearOfTheGoddess'),
        description="Gain an additional @ManaPercIncrease*100@% Mana from all sources. The holder gains an additional bonus based on their Role:  Tanks and Fighters: Gain @FrontlineResists@ Armor and Magic Resistance.  Other Roles: Gain @BacklineADAP@% Attack Damage and Ability Power.",
    ),
    "TFT_Item_ArchangelsStaff": Set18ItemInfo(
        item_id="TFT_Item_ArchangelsStaff",
        name="Archangel's Staff",
        is_component=False,
        recipe=('TFT_Item_NeedlesslyLargeRod', 'TFT_Item_TearOfTheGoddess'),
        description="Combat start: Gain @APPerInterval@% Ability Power every @IntervalSeconds@ seconds in combat.",
    ),
    "TFT_Item_BFSword": Set18ItemInfo(
        item_id="TFT_Item_BFSword",
        name="B.F. Sword",
        is_component=True,
        recipe=(),
        description="+10 Attack Damage",
    ),
    "TFT_Item_Bloodthirster": Set18ItemInfo(
        item_id="TFT_Item_Bloodthirster",
        name="Bloodthirster",
        is_component=False,
        recipe=('TFT_Item_BFSword', 'TFT_Item_NegatronCloak'),
        description="Once per combat at @HealthThreshold@% Health, gain a @ShieldHealthPercent@% max Health Shield that lasts up to @ShieldDuration@ seconds.",
    ),
    "TFT_Item_BlueBuff": Set18ItemInfo(
        item_id="TFT_Item_BlueBuff",
        name="Blue Buff",
        is_component=False,
        recipe=('TFT_Item_TearOfTheGoddess', 'TFT_Item_TearOfTheGoddess'),
        description="Gain @ModifiedADAP*100@% additional Attack Damage and Ability Power from all sources.",
    ),
    "TFT_Item_BrambleVest": Set18ItemInfo(
        item_id="TFT_Item_BrambleVest",
        name="Bramble Vest",
        is_component=False,
        recipe=('TFT_Item_ChainVest', 'TFT_Item_ChainVest'),
        description="Gain @PercentMaxHP*100@% max health.  Take @AutoDamageReduction*100@% reduced damage from attacks. When struck by any attack, deal <magicDamage>@1StarAoEDamage@ magic damage</magicDamage> to all adjacent enemies.  <tftitemrules>Cooldown: @ICD@ seconds</tftitemrules>",
    ),
    "TFT_Item_ChainVest": Set18ItemInfo(
        item_id="TFT_Item_ChainVest",
        name="Chain Vest",
        is_component=True,
        recipe=(),
        description="+20 Armor",
    ),
    "TFT_Item_Crownguard": Set18ItemInfo(
        item_id="TFT_Item_Crownguard",
        name="Crownguard",
        is_component=False,
        recipe=('TFT_Item_NeedlesslyLargeRod', 'TFT_Item_ChainVest'),
        description="Combat Start: Gain a @ShieldSize@% max Health Shield for @ShieldDuration@ seconds.  When the Shield expires, gain @ShieldBonusAP@% Ability Power.",
    ),
    "TFT_Item_Deathblade": Set18ItemInfo(
        item_id="TFT_Item_Deathblade",
        name="Deathblade",
        is_component=False,
        recipe=('TFT_Item_BFSword', 'TFT_Item_BFSword'),
        description="<tftitemrules>Perfect peace and calm for the holder - and all who face it.</tftitemrules>  @TFTUnitProperty.:TFT_Augment_TragicalBlade_TRAKey@",
    ),
    "TFT_Item_DragonsClaw": Set18ItemInfo(
        item_id="TFT_Item_DragonsClaw",
        name="Dragon's Claw",
        is_component=False,
        recipe=('TFT_Item_NegatronCloak', 'TFT_Item_NegatronCloak'),
        description="Gain @PercentMaxHP*100@% max health.  Every @HealthRegenInterval@ seconds, heal @PercentHealthDamage@% max Health.",
    ),
    "TFT_Item_FrozenHeart": Set18ItemInfo(
        item_id="TFT_Item_FrozenHeart",
        name="Protector's Vow",
        is_component=False,
        recipe=('TFT_Item_TearOfTheGoddess', 'TFT_Item_ChainVest'),
        description="Combat Start: Gain @CombatStartMana@ Mana.  At @HealthThreshold@% Health, gain @TriggerMana@ Mana and a Shield equal to @ShieldHealthPercent@% max Health.",
    ),
    "TFT_Item_GargoyleStoneplate": Set18ItemInfo(
        item_id="TFT_Item_GargoyleStoneplate",
        name="Gargoyle Stoneplate",
        is_component=False,
        recipe=('TFT_Item_ChainVest', 'TFT_Item_NegatronCloak'),
        description="Gain @ArmorPerEnemy@ Armor and @MRPerEnemy@ Magic Resist for each enemy targeting the holder.",
    ),
    "TFT_Item_GiantsBelt": Set18ItemInfo(
        item_id="TFT_Item_GiantsBelt",
        name="Giant's Belt",
        is_component=True,
        recipe=(),
        description="+150 Health",
    ),
    "TFT_Item_GuardianAngel": Set18ItemInfo(
        item_id="TFT_Item_GuardianAngel",
        name="Edge of Night",
        is_component=False,
        recipe=('TFT_Item_BFSword', 'TFT_Item_ChainVest'),
        description="At @HealthThreshold@% Health, briefly become untargetable, shed negative effects, and heal @MissingHealthRestore*100@% missing health.",
    ),
    "TFT_Item_GuinsoosRageblade": Set18ItemInfo(
        item_id="TFT_Item_GuinsoosRageblade",
        name="Guinsoo's Rageblade",
        is_component=False,
        recipe=('TFT_Item_RecurveBow', 'TFT_Item_NeedlesslyLargeRod'),
        description="Gain @AttackSpeedPerStack@% stacking Attack Speed every second.",
    ),
    "TFT_Item_HandOfJustice": Set18ItemInfo(
        item_id="TFT_Item_HandOfJustice",
        name="Hand Of Justice",
        is_component=False,
        recipe=('TFT_Item_TearOfTheGoddess', 'TFT_Item_SparringGloves'),
        description="Gain 2 effects:<li>@AD_NotStatBar*100@% Attack Damage and @AP_NotStatBar@% Ability Power.<li>@StatOmnivamp_NotStatBar*100@% Omnivamp.  While above @HealthThreshold*100@% health, double the Attack Damage and Ability Power. While below @HealthThreshold*100@% Health, double the Omnivamp.",
    ),
    "TFT_Item_HextechGunblade": Set18ItemInfo(
        item_id="TFT_Item_HextechGunblade",
        name="Hextech Gunblade",
        is_component=False,
        recipe=('TFT_Item_BFSword', 'TFT_Item_NeedlesslyLargeRod'),
        description="Heal the lowest percent Health ally for @AllyHealing*100@% of damage dealt.  <TFTTrackerLabel>Ally Healing:</TFTTrackerLabel> <TFTHighlight>@TFTUnitProperty.item:TFT_Tracker_Value1@</TFTHighlight>",
    ),
    "TFT_Item_InfinityEdge": Set18ItemInfo(
        item_id="TFT_Item_InfinityEdge",
        name="Infinity Edge",
        is_component=False,
        recipe=('TFT_Item_BFSword', 'TFT_Item_SparringGloves'),
        description="Gain <TFTKeyword>Precision</TFTKeyword>.  {{TFT_Keyword_Precision}}",
    ),
    "TFT_Item_IonicSpark": Set18ItemInfo(
        item_id="TFT_Item_IonicSpark",
        name="Ionic Spark",
        is_component=False,
        recipe=('TFT_Item_NeedlesslyLargeRod', 'TFT_Item_NegatronCloak'),
        description="@MRShred@% <TFTKeyword>Shred</TFTKeyword> enemies within @HexRange@ hexes. When enemies cast an Ability, deal magic damage equal to @ManaRatio@% of the Mana spent  <tftitemrules><tftbold>Shred</tftbold>: Reduce Magic Resist</tftitemrules>",
    ),
    "TFT_Item_JeweledGauntlet": Set18ItemInfo(
        item_id="TFT_Item_JeweledGauntlet",
        name="Jeweled Gauntlet",
        is_component=False,
        recipe=('TFT_Item_NeedlesslyLargeRod', 'TFT_Item_SparringGloves'),
        description="Gain <TFTKeyword>Precision</TFTKeyword>.  {{TFT_Keyword_Precision}}",
    ),
    "TFT_Item_LastWhisper": Set18ItemInfo(
        item_id="TFT_Item_LastWhisper",
        name="Last Whisper",
        is_component=False,
        recipe=('TFT_Item_RecurveBow', 'TFT_Item_SparringGloves'),
        description="Damage from attacks and Abilities @ArmorReductionPercent@% <TFTKeyword>Sunder</TFTKeyword> the target for @ArmorBreakDuration@ seconds. This effect does not stack.  <tftitemrules><tftbold>Sunder</tftbold>: Reduce Armor</tftitemrules>",
    ),
    "TFT_Item_Leviathan": Set18ItemInfo(
        item_id="TFT_Item_Leviathan",
        name="Nashor's Tooth",
        is_component=False,
        recipe=('TFT_Item_RecurveBow', 'TFT_Item_GiantsBelt'),
        description="Attacks grant @BaseManaOnHit@ bonus Mana, increased to @ManaOnCrit@ if they critically strike.",
    ),
    "TFT_Item_MadredsBloodrazor": Set18ItemInfo(
        item_id="TFT_Item_MadredsBloodrazor",
        name="Giant Slayer",
        is_component=False,
        recipe=('TFT_Item_BFSword', 'TFT_Item_RecurveBow'),
        description="Gain @DamageAmp*100@% additional Damage Amp against Tanks.",
    ),
    "TFT_Item_Morellonomicon": Set18ItemInfo(
        item_id="TFT_Item_Morellonomicon",
        name="Morellonomicon",
        is_component=False,
        recipe=('TFT_Item_NeedlesslyLargeRod', 'TFT_Item_GiantsBelt'),
        description="Attacks and Abilities deal @BurnPercent@% <TFTKeyword>Burn</TFTKeyword> and @GrievousWoundsPercent@% <TFTKeyword>Wound</TFTKeyword> to enemies for @BurnDuration@ seconds.  <tftitemrules><tftbold>Burn</tftbold>: Deals a percent of the target's max Health as true damage every second <tftbold>Wound</tftbold>: Reduces healing received</tftitemrules>",
    ),
    "TFT_Item_NeedlesslyLargeRod": Set18ItemInfo(
        item_id="TFT_Item_NeedlesslyLargeRod",
        name="Needlessly Large Rod",
        is_component=True,
        recipe=(),
        description="+10 Ability Power",
    ),
    "TFT_Item_NegatronCloak": Set18ItemInfo(
        item_id="TFT_Item_NegatronCloak",
        name="Negatron Cloak",
        is_component=True,
        recipe=(),
        description="+20 Magic Resist",
    ),
    "TFT_Item_NightHarvester": Set18ItemInfo(
        item_id="TFT_Item_NightHarvester",
        name="Steadfast Heart",
        is_component=False,
        recipe=('TFT_Item_ChainVest', 'TFT_Item_SparringGloves'),
        description="Gain @BaseDurability*100@% Durability. While above @ThresholdForEmpower*100@% Health, instead gain @EmpoweredDurability*100@% Durability.  @TFTUnitProperty.:TFT_Augment_WarmogsBuckle_TRAKey@",
    ),
    "TFT_Item_PowerGauntlet": Set18ItemInfo(
        item_id="TFT_Item_PowerGauntlet",
        name="Striker's Flail",
        is_component=False,
        recipe=('TFT_Item_GiantsBelt', 'TFT_Item_SparringGloves'),
        description="Critical Strikes grant @BuffDamageAmp*100@% Damage Amp for @Duration@ seconds, stacking up to @MaxStacks@ times.",
    ),
    "TFT_Item_Quicksilver": Set18ItemInfo(
        item_id="TFT_Item_Quicksilver",
        name="Quicksilver",
        is_component=False,
        recipe=('TFT_Item_SparringGloves', 'TFT_Item_NegatronCloak'),
        description="Combat Start: Gain immunity to crowd control for @SpellShieldDuration@ seconds.  Gain @ProcAttackSpeed*100@% stacking Attack Speed every second.",
    ),
    "TFT_Item_RabadonsDeathcap": Set18ItemInfo(
        item_id="TFT_Item_RabadonsDeathcap",
        name="Rabadon's Deathcap",
        is_component=False,
        recipe=('TFT_Item_NeedlesslyLargeRod', 'TFT_Item_NeedlesslyLargeRod'),
        description="<tftitemrules>This humble hat can help you make, or unmake, the world itself.</tftitemrules>  @TFTUnitProperty.:TFT_Augment_DeadlierCaps_TRAKey@",
    ),
    "TFT_Item_RapidFireCannon": Set18ItemInfo(
        item_id="TFT_Item_RapidFireCannon",
        name="Red Buff",
        is_component=False,
        recipe=('TFT_Item_RecurveBow', 'TFT_Item_RecurveBow'),
        description="Attacks and Abilities @BurnPercent@% <TFTKeyword>Burn</TFTKeyword> and @HealingReductionPct@% <TFTKeyword>Wound</TFTKeyword> enemies for @Duration@ seconds.  <tftitemrules><tftbold>Burn</tftbold>: Deals a percent of the target's max Health as true damage every second <tftbold>Wound</tftbold>: Reduces healing received</tftitemrules>",
    ),
    "TFT_Item_RecurveBow": Set18ItemInfo(
        item_id="TFT_Item_RecurveBow",
        name="Recurve Bow",
        is_component=True,
        recipe=(),
        description="+10% Attack Speed",
    ),
    "TFT_Item_RedBuff": Set18ItemInfo(
        item_id="TFT_Item_RedBuff",
        name="Sunfire Cape",
        is_component=False,
        recipe=('TFT_Item_ChainVest', 'TFT_Item_GiantsBelt'),
        description="Gain @BonusPercentHP*100@% max Health.   Every @ICD@ seconds, deal @BurnPercent@% <TFTKeyword>Burn</TFTKeyword> and @GrievousWoundsPercent@% <TFTKeyword>Wound</TFTKeyword> to an enemy within @HexRange@ hexes for @BurnDuration@ seconds.  <tftitemrules><tftbold>Burn</tftbold>: Deals a percent of the target's max Health as true damage every second <tftbold>Wound</tftbold>: Reduces healing received</tftitemrules>",
    ),
    "TFT_Item_Redemption": Set18ItemInfo(
        item_id="TFT_Item_Redemption",
        name="Spirit Visage",
        is_component=False,
        recipe=('TFT_Item_TearOfTheGoddess', 'TFT_Item_GiantsBelt'),
        description="Regenerate @MissingHealthHeal*100@% of missing Health each second.",
    ),
    "TFT_Item_RunaansHurricane": Set18ItemInfo(
        item_id="TFT_Item_RunaansHurricane",
        name="Kraken's Fury",
        is_component=False,
        recipe=('TFT_Item_NegatronCloak', 'TFT_Item_RecurveBow'),
        description="Attacks grant @ADOnAttack*100@% stacking Attack Damage, up to @MaxStacks@ attacks. After @MaxStacks@ attacks, gain @ASCapstone*100@% Attack Speed.",
    ),
    "TFT_Item_SparringGloves": Set18ItemInfo(
        item_id="TFT_Item_SparringGloves",
        name="Sparring Gloves",
        is_component=True,
        recipe=(),
        description="+20% Critical Strike Chance",
    ),
    "TFT_Item_Spatula": Set18ItemInfo(
        item_id="TFT_Item_Spatula",
        name="Spatula",
        is_component=True,
        recipe=(),
        description="Special Emblem and Team Size Component",
    ),
    "TFT_Item_SpearOfShojin": Set18ItemInfo(
        item_id="TFT_Item_SpearOfShojin",
        name="Spear of Shojin",
        is_component=False,
        recipe=('TFT_Item_BFSword', 'TFT_Item_TearOfTheGoddess'),
        description="Attacks grant @FlatManaRestore@ bonus Mana.",
    ),
    "TFT_Item_SpectralGauntlet": Set18ItemInfo(
        item_id="TFT_Item_SpectralGauntlet",
        name="Evenshroud",
        is_component=False,
        recipe=('TFT_Item_NegatronCloak', 'TFT_Item_GiantsBelt'),
        description="@ARReductionAmount@% <TFTKeyword>Sunder</TFTKeyword> enemies within @HexRange@ hexes. Gain @BonusResists@ Armor and Magic Resist for the first @BonusResistDuration@ seconds of combat.  <tftitemrules><tftbold>Sunder</tftbold>: Reduce Armor</tftitemrules>",
    ),
    "TFT_Item_StatikkShiv": Set18ItemInfo(
        item_id="TFT_Item_StatikkShiv",
        name="Void Staff",
        is_component=False,
        recipe=('TFT_Item_RecurveBow', 'TFT_Item_TearOfTheGoddess'),
        description="Damage from attacks and Abilities @MRShred@% <TFTKeyword>Shred</TFTKeyword> the target for @MRShredDuration@ seconds. This effect does not stack.  <tftitemrules><tftbold>Shred</tftbold>: Reduce Magic Resist</tftitemrules>",
    ),
    "TFT_Item_SteraksGage": Set18ItemInfo(
        item_id="TFT_Item_SteraksGage",
        name="Sterak's Gage",
        is_component=False,
        recipe=('TFT_Item_BFSword', 'TFT_Item_GiantsBelt'),
        description="At @HealthThreshold@% Health, gain a Shield equal to @PercentHealthShield*100@% of the wearer's maximum Health that rapidly decays over @ShieldDuration@ seconds.",
    ),
    "TFT_Item_TacticiansCrown": Set18ItemInfo(
        item_id="TFT_Item_TacticiansCrown",
        name="Tactician's Crown",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_Spatula'),
        description="Your team gains +@MaxArmySizeIncrease@ max team size.  @PercentGoldChance@% chance to drop 1 gold when you win combat.  <tftitemrules>\"...the Heart of a hero...\"</tftitemrules>",
    ),
    "TFT_Item_TearOfTheGoddess": Set18ItemInfo(
        item_id="TFT_Item_TearOfTheGoddess",
        name="Tear of the Goddess",
        is_component=True,
        recipe=(),
        description="+15 Starting Mana",
    ),
    "TFT_Item_ThiefsGloves": Set18ItemInfo(
        item_id="TFT_Item_ThiefsGloves",
        name="Thief's Gloves",
        is_component=False,
        recipe=('TFT_Item_SparringGloves', 'TFT_Item_SparringGloves'),
        description="Each round: Equip 2 random items.  <tftitemrules>[Consumes 3 item slots.]</tftitemrules> @TFTUnitProperty.:TFT_BindOnEquipTRA@",
    ),
    "TFT_Item_TitansResolve": Set18ItemInfo(
        item_id="TFT_Item_TitansResolve",
        name="Titan's Resolve",
        is_component=False,
        recipe=('TFT_Item_ChainVest', 'TFT_Item_RecurveBow'),
        description="Gain @StackingAD*100@% Attack Damage and @StackingSP@% Ability Power when attacking or taking damage, stacking up to @StackCap@ times.    At full stacks, gain @StackedAmp*100@% Damage Amp and gain immunity to crowd control.",
    ),
    "TFT_Item_UnstableConcoction": Set18ItemInfo(
        item_id="TFT_Item_UnstableConcoction",
        name="Hand Of Justice",
        is_component=False,
        recipe=('TFT_Item_TearOfTheGoddess', 'TFT_Item_SparringGloves'),
        description="Gain 2 effects:<li>@AD_NotStatBar*100@% Attack Damage and @AP_NotStatBar@% Ability Power.<li>@StatOmnivamp_NotStatBar*100@% Omnivamp.  While above @HealthThreshold*100@% health, double the Attack Damage and Ability Power. While below @HealthThreshold*100@% Health, double the Omnivamp.",
    ),
    "TFT_Item_WarmogsArmor": Set18ItemInfo(
        item_id="TFT_Item_WarmogsArmor",
        name="Warmog's Armor",
        is_component=False,
        recipe=('TFT_Item_GiantsBelt', 'TFT_Item_GiantsBelt'),
        description="Gain @BonusPercentHP*100@% max Health.",
    ),
}

SET18_ITEM_RECIPES: dict[frozenset[str], str] = {
    frozenset(["TFT_Item_BFSword", "TFT_Item_BFSword"]): "TFT_Item_Deathblade",
    frozenset(["TFT_Item_BFSword", "TFT_Item_ChainVest"]): "TFT_Item_GuardianAngel",
    frozenset(["TFT_Item_BFSword", "TFT_Item_GiantsBelt"]): "TFT_Item_SteraksGage",
    frozenset(["TFT_Item_BFSword", "TFT_Item_NeedlesslyLargeRod"]): "TFT_Item_HextechGunblade",
    frozenset(["TFT_Item_BFSword", "TFT_Item_NegatronCloak"]): "TFT_Item_Bloodthirster",
    frozenset(["TFT_Item_BFSword", "TFT_Item_RecurveBow"]): "TFT_Item_MadredsBloodrazor",
    frozenset(["TFT_Item_BFSword", "TFT_Item_SparringGloves"]): "TFT_Item_InfinityEdge",
    frozenset(["TFT_Item_BFSword", "TFT_Item_Spatula"]): "TFT17_Item_DarkStarEmblemItem",
    frozenset(["TFT_Item_BFSword", "TFT_Item_TearOfTheGoddess"]): "TFT_Item_SpearOfShojin",
    frozenset(["TFT_Item_ChainVest", "TFT_Item_ChainVest"]): "TFT_Item_BrambleVest",
    frozenset(["TFT_Item_ChainVest", "TFT_Item_GiantsBelt"]): "TFT_Item_RedBuff",
    frozenset(["TFT_Item_ChainVest", "TFT_Item_NeedlesslyLargeRod"]): "TFT_Item_Crownguard",
    frozenset(["TFT_Item_ChainVest", "TFT_Item_NegatronCloak"]): "TFT_Item_GargoyleStoneplate",
    frozenset(["TFT_Item_ChainVest", "TFT_Item_RecurveBow"]): "TFT_Item_TitansResolve",
    frozenset(["TFT_Item_ChainVest", "TFT_Item_SparringGloves"]): "TFT_Item_NightHarvester",
    frozenset(["TFT_Item_ChainVest", "TFT_Item_Spatula"]): "TFT17_Item_AstronautEmblemItem",
    frozenset(["TFT_Item_ChainVest", "TFT_Item_TearOfTheGoddess"]): "TFT_Item_FrozenHeart",
    frozenset(["TFT_Item_GiantsBelt", "TFT_Item_GiantsBelt"]): "TFT_Item_WarmogsArmor",
    frozenset(["TFT_Item_GiantsBelt", "TFT_Item_NeedlesslyLargeRod"]): "TFT_Item_Morellonomicon",
    frozenset(["TFT_Item_GiantsBelt", "TFT_Item_NegatronCloak"]): "TFT_Item_SpectralGauntlet",
    frozenset(["TFT_Item_GiantsBelt", "TFT_Item_RecurveBow"]): "TFT_Item_Leviathan",
    frozenset(["TFT_Item_GiantsBelt", "TFT_Item_SparringGloves"]): "TFT_Item_PowerGauntlet",
    frozenset(["TFT_Item_GiantsBelt", "TFT_Item_Spatula"]): "TFT17_Item_PrimordianEmblemItem",
    frozenset(["TFT_Item_GiantsBelt", "TFT_Item_TearOfTheGoddess"]): "TFT_Item_Redemption",
    frozenset(["TFT_Item_NeedlesslyLargeRod", "TFT_Item_NeedlesslyLargeRod"]): "TFT_Item_RabadonsDeathcap",
    frozenset(["TFT_Item_NeedlesslyLargeRod", "TFT_Item_NegatronCloak"]): "TFT_Item_IonicSpark",
    frozenset(["TFT_Item_NeedlesslyLargeRod", "TFT_Item_RecurveBow"]): "TFT_Item_GuinsoosRageblade",
    frozenset(["TFT_Item_NeedlesslyLargeRod", "TFT_Item_SparringGloves"]): "TFT_Item_JeweledGauntlet",
    frozenset(["TFT_Item_NeedlesslyLargeRod", "TFT_Item_Spatula"]): "TFT17_Item_StargazerEmblemItem",
    frozenset(["TFT_Item_NeedlesslyLargeRod", "TFT_Item_TearOfTheGoddess"]): "TFT_Item_ArchangelsStaff",
    frozenset(["TFT_Item_NegatronCloak", "TFT_Item_NegatronCloak"]): "TFT_Item_DragonsClaw",
    frozenset(["TFT_Item_NegatronCloak", "TFT_Item_RecurveBow"]): "TFT_Item_RunaansHurricane",
    frozenset(["TFT_Item_NegatronCloak", "TFT_Item_SparringGloves"]): "TFT_Item_Quicksilver",
    frozenset(["TFT_Item_NegatronCloak", "TFT_Item_Spatula"]): "TFT17_Item_FavoredEmblemItem",
    frozenset(["TFT_Item_NegatronCloak", "TFT_Item_TearOfTheGoddess"]): "TFT_Item_AdaptiveHelm",
    frozenset(["TFT_Item_RecurveBow", "TFT_Item_RecurveBow"]): "TFT_Item_RapidFireCannon",
    frozenset(["TFT_Item_RecurveBow", "TFT_Item_SparringGloves"]): "TFT_Item_LastWhisper",
    frozenset(["TFT_Item_RecurveBow", "TFT_Item_Spatula"]): "TFT17_Item_PulsefireEmblemItem",
    frozenset(["TFT_Item_RecurveBow", "TFT_Item_TearOfTheGoddess"]): "TFT_Item_StatikkShiv",
    frozenset(["TFT_Item_SparringGloves", "TFT_Item_SparringGloves"]): "TFT_Item_ThiefsGloves",
    frozenset(["TFT_Item_SparringGloves", "TFT_Item_Spatula"]): "TFT17_Item_DRXEmblemItem",
    frozenset(["TFT_Item_SparringGloves", "TFT_Item_TearOfTheGoddess"]): "TFT_Item_HandOfJustice",
    frozenset(["TFT_Item_Spatula", "TFT_Item_Spatula"]): "TFT_Item_TacticiansCrown",
    frozenset(["TFT_Item_Spatula", "TFT_Item_TearOfTheGoddess"]): "TFT17_Item_SpaceGrooveEmblemItem",
    frozenset(["TFT_Item_TearOfTheGoddess", "TFT_Item_TearOfTheGoddess"]): "TFT_Item_BlueBuff",
}


# =============================================================================
# SET PROFILE DATA CLASS & EXPORTER
# =============================================================================

@dataclass(frozen=True, slots=True)
class Set18Profile:
    """Top-level container for Set 18 profile specifications."""

    set_name: str = "TFTSet18"
    total_champions: int = 65
    total_traits: int = 36
    total_items: int = 111
    total_recipes: int = 45
    champions: tuple[Set18ChampionInfo, ...] = SET18_CHAMPION_CATALOG
    traits: dict[str, Set18TraitInfo] = field(default_factory=lambda: dict(SET18_TRAIT_CATALOG))
    items: dict[str, Set18ItemInfo] = field(default_factory=lambda: dict(SET18_ITEMS_CATALOG))

    def to_dict(self) -> dict[str, Any]:
        """Convert profile to serializable dictionary."""
        return {
            "set_name": self.set_name,
            "stats": {
                "total_champions": len(self.champions),
                "total_traits": len(self.traits),
                "total_items": len(self.items),
                "total_recipes": len(SET18_ITEM_RECIPES),
                "champion_cost_breakdown": {
                    "1_cost": sum(1 for c in self.champions if c.cost == 1),
                    "2_cost": sum(1 for c in self.champions if c.cost == 2),
                    "3_cost": sum(1 for c in self.champions if c.cost == 3),
                    "4_cost": sum(1 for c in self.champions if c.cost == 4),
                    "5_cost": sum(1 for c in self.champions if c.cost == 5),
                },
            },
            "champions": [
                {
                    "id": c.champion_id,
                    "name": c.name,
                    "cost": c.cost,
                    "traits": list(c.traits),
                    "role": c.role.value,
                    "is_carry": c.is_carry,
                    "is_tank": c.is_tank,
                    "hp": c.hp,
                    "armor": c.armor,
                    "mr": c.mr,
                    "ad": c.ad,
                    "aspeed": c.aspeed,
                    "range": c.range,
                    "mana": c.mana,
                    "max_mana": c.max_mana,
                    "recommended_items": list(c.recommended_items),
                }
                for c in self.champions
            ],
            "traits": {
                t.trait_id: {
                    "name": t.name,
                    "thresholds": list(t.thresholds),
                    "description": t.description,
                    "champions": list(t.champions),
                }
                for t in self.traits.values()
            },
            "items": {
                i.item_id: {
                    "name": i.name,
                    "is_component": i.is_component,
                    "recipe": list(i.recipe),
                    "description": i.description,
                }
                for i in self.items.values()
            },
            "recipes": [
                {
                    "components": sorted(list(comp)),
                    "result_item": res_id,
                    "result_name": self.items.get(res_id, Set18ItemInfo(res_id, res_id, False, (), "")).name,
                }
                for comp, res_id in SET18_ITEM_RECIPES.items()
            ],
            "shop_odds": {str(k): list(v) for k, v in STANDARD_SHOP_ODDS.items()},
            "level_exp": {str(k): v for k, v in STANDARD_LEVEL_EXP.items()},
            "pool_sizes": {str(k): v for k, v in STANDARD_POOL_SIZES.items()},
        }


def get_set18_data() -> SetData:
    """Build a complete SetData instance for Set 18 simulations."""
    champions: dict[str, ChampionDef] = {
        c.champion_id: ChampionDef(
            champion_id=c.champion_id,
            name=c.name,
            cost=c.cost,
            traits=c.traits,
            role=c.role,
        )
        for c in SET18_CHAMPION_CATALOG
    }

    traits: dict[str, TraitDef] = {
        t.trait_id: TraitDef(
            trait_id=t.trait_id,
            name=t.name,
            thresholds=t.thresholds,
        )
        for t in SET18_TRAIT_CATALOG.values()
    }

    items: dict[str, ItemDef] = {
        it.item_id: ItemDef(
            item_id=it.item_id,
            name=it.name,
            is_component=it.is_component,
            recipe=it.recipe if len(it.recipe) == 2 else None,
        )
        for it in SET18_ITEMS_CATALOG.values()
    }

    recipes: dict[frozenset[str], str] = dict(SET18_ITEM_RECIPES)

    return SetData(
        set_name="TFTSet18",
        champions=champions,
        traits=traits,
        items=items,
        recipes=recipes,
        shop_odds=STANDARD_SHOP_ODDS,
        pool_sizes=STANDARD_POOL_SIZES,
        level_exp=STANDARD_LEVEL_EXP,
        stage_base_damage=STANDARD_STAGE_BASE_DAMAGE,
        passive_gold_schedule=STANDARD_PASSIVE_GOLD,
    )


def export_set18_profile_json(output_path: str | Path = "data/set18_profile.json") -> Path:
    """Export complete Set 18 profile JSON to file."""
    profile = Set18Profile()
    data = profile.to_dict()
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
