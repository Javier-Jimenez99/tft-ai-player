"""Set 17 Complete Profile: Champions, Traits, Items, Recipes, Odds, and Leveling Curves.
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
# SET 17 TRAIT DEFINITIONS & SYNERGY BONUSES (35 TRAITS)
# =============================================================================

@dataclass(frozen=True, slots=True)
class Set17TraitInfo:
    """Detailed trait synergy definition with threshold bonuses."""

    trait_id: str
    name: str
    thresholds: tuple[int, ...]
    description: str
    champions: tuple[str, ...]


SET17_TRAIT_CATALOG: dict[str, Set17TraitInfo] = {
    "Anima": Set17TraitInfo(
        trait_id="Anima",
        name="Anima",
        thresholds=(3, 5),
        description="After losing a player combat, gain @TechPerCombat@ Tech, plus additional Tech equal to @TechPerLoss@ times the length of your loss streak. Additionally, gain @TechPerKill@ Tech per Anima takedown.  Each time Animas get @TechBreakpoint@ Tech, they prototype new Anima Weapons. You can take them, or save your Tech to get more powerful weapons next time.  <row>(@MinUnits@) Start Researching! </row> <row>(@MinUnits@) After each player combat, gain loot!</row>  <ShowIf.TFT17_AnimaSquad_IsDoubleUp><rules>Gain @PairsPartnerWinMultiplier*100@% Tech from losing if your partner wins a combat in Double Up.</rules></ShowIf.TFT17_AnimaSquad_IsDoubleUp>",
        champions=('TFT17_Briar', 'TFT17_Jinx', 'TFT17_Aurora', 'TFT17_Illaoi', 'TFT17_Fiora'),
    ),
    "Arbiter": Set17TraitInfo(
        trait_id="Arbiter",
        name="Arbiter",
        thresholds=(2, 3),
        description="Scribe a unique divine law, allowing you to choose an effect to apply to Arbiters when a chosen cause occurs.  <row>(@MinUnits@) Choose a cause and effect for your law</row> <row>(@MinUnits@) Effects are stronger. </row>",
        champions=('TFT17_Leona', 'TFT17_Zoe', 'TFT17_Diana', 'TFT17_Leblanc'),
    ),
    "Bastion": Set17TraitInfo(
        trait_id="Bastion",
        name="Bastion",
        thresholds=(2, 4, 6),
        description="Your team gains @TeamwideResists@ Armor and Magic Resist.  Bastions gain more, and the value doubles in the first @Duration@ seconds of combat.  <row>(@MinUnits@) @BonusArmor@ %i:scaleArmor%%i:scaleMR%</row> <row>(@MinUnits@) @BonusArmor@ %i:scaleArmor%%i:scaleMR%</row> <row>(@MinUnits@) @BonusArmor@ %i:scaleArmor%%i:scaleMR%; Non-Bastions gain an additional @EnhancedTeamwideArmor@ %i:scaleArmor%%i:scaleMR%.</row>",
        champions=('TFT17_Aatrox', 'TFT17_Poppy', 'TFT17_Jax', 'TFT17_Ornn', 'TFT17_Rammus', 'TFT17_Shen'),
    ),
    "Brawler": Set17TraitInfo(
        trait_id="Brawler",
        name="Brawler",
        thresholds=(2, 4, 6),
        description="Your team gains @TeamwideBonus*100@% Health. Brawlers gain more.  <expandRow>(@MinUnits@) +@HealthBonus*100@% maximum Health</expandRow>",
        champions=('TFT17_Chogath', 'TFT17_RekSai', 'TFT17_Gnar', 'TFT17_Gragas', 'TFT17_Pantheon', 'TFT17_Maokai', 'TFT17_Urgot', 'TFT17_TahmKench'),
    ),
    "Bulwark": Set17TraitInfo(
        trait_id="Bulwark",
        name="Bulwark",
        thresholds=(1,),
        description="Summon a placeable relic. At the start of combat, it grants adjacent allies a @PercentHealthShield*100@% max Health shield and @AttackSpeed*100@% Attack Speed.",
        champions=('TFT17_Shen',),
    ),
    "Challenger": Set17TraitInfo(
        trait_id="Challenger",
        name="Challenger",
        thresholds=(2, 3, 4, 5),
        description="Your team gains @TeamwideAS*100@% Attack Speed. Challengers gain bonus Attack Speed. When their target dies, Challengers dash to a new target and increase their Attack Speed bonus by @BurstPercent*100@% for @BurstDuration@ seconds.  <row>(@MinUnits@) @AttackSpeedPercent*100@%&nbsp;%i:scaleAS%</row> <row>(@MinUnits@) @AttackSpeedPercent*100@%&nbsp;%i:scaleAS%</row> <row>(@MinUnits@) @AttackSpeedPercent*100@%&nbsp;%i:scaleAS%</row> <row>(@MinUnits@) @AttackSpeedPercent*100@%&nbsp;%i:scaleAS%</row>",
        champions=('TFT17_Belveth', 'TFT17_Jinx', 'TFT17_Diana', 'TFT17_Kindred'),
    ),
    "Commander": Set17TraitInfo(
        trait_id="Commander",
        name="Commander",
        thresholds=(1,),
        description="(@MinUnits@) Sona gives you a random Command Mod every @RoundsPerMod@ rounds which allows you to alter the way an ally behaves during combat. Command Mods last @RoundsPerMod@ player combats even if they are not equipped.",
        champions=('TFT17_Sona',),
    ),
    "Conduit": Set17TraitInfo(
        trait_id="Conduit",
        name="Conduit",
        thresholds=(2, 3, 4, 5),
        description="Innate: Conduits gain @InnateManaGain*100@% additional Mana from all sources.  Your team gains Mana Regen, increased for Conduits.  <expandRow>(@MinUnits@) @TeamManaRegen@ %i:TFTManaRegen% | @ChannelerManaRegen@ %i:TFTManaRegen%</expandRow>",
        champions=('TFT17_Mordekaiser', 'TFT17_Zoe', 'TFT17_Viktor', 'TFT17_AurelionSol', 'TFT17_Morgana', 'TFT17_Bard'),
    ),
    "Dark Lady": Set17TraitInfo(
        trait_id="Dark Lady",
        name="Dark Lady",
        thresholds=(1,),
        description="Your team gains @Durability*100@% Durability, increased to @TransformedDurability*100@% when Morgana is in her Dark Form.",
        champions=('TFT17_Morgana',),
    ),
    "Dark Star": Set17TraitInfo(
        trait_id="Dark Star",
        name="Dark Star",
        thresholds=(2, 4, 6, 9),
        description="<row>(@MinUnits@) Dark Stars create a black hole that consumes enemies at <ShowIf.TFT17_DarkStar_HasNeutronStar><TFTBonus>@TFTUnitProperty.trait:TFT17_Augment_DarkStar_NeutronStar_BonusExecutePercent*100@%</TFTBonus></ShowIf.TFT17_DarkStar_HasNeutronStar><ShowIfNot.TFT17_DarkStar_HasNeutronStar>@ExecuteHPPercent*100@%</ShowIfNot.TFT17_DarkStar_HasNeutronStar> max health.</row> <row>(@MinUnits@) AND they gain @ADAP@% %i:scaleAD%%i:scaleAP%.</row> <row>(@MinUnits@) AND the strongest Dark Star unit goes supermassive, gaining @SupermassivePercentBonus*100@% effectiveness from Dark Star, and creates 2 minor Black Holes.</row> <row>(@MinUnits@) All Dark Stars are supermassive. At level 10, CONSUME EVERYONE. </row>",
        champions=('TFT17_Chogath', 'TFT17_Lissandra', 'TFT17_Mordekaiser', 'TFT17_Kaisa', 'TFT17_Karma', 'TFT17_Jhin'),
    ),
    "Divine Duelist": Set17TraitInfo(
        trait_id="Divine Duelist",
        name="Divine Duelist",
        thresholds=(1,),
        description="Your Tactician heals for @PlayerOmnivamp*100@% of player damage dealt from winning.  Fiora always wins a one on one duel.",
        champions=('TFT17_Fiora',),
    ),
    "Doomer": Set17TraitInfo(
        trait_id="Doomer",
        name="Doomer",
        thresholds=(1,),
        description="Combat Start: Mark all enemies with Doom.  The first time enemies are damaged each combat, their Doom is consumed, stealing @ADAP1@% Attack Damage and Ability Power from them and granting it to your strongest Vex.",
        champions=('TFT17_Vex',),
    ),
    "Eradicator": Set17TraitInfo(
        trait_id="Eradicator",
        name="Eradicator",
        thresholds=(1,),
        description="Enemies have @PctResists*100@% less Armor and Magic Resist.",
        champions=('TFT17_Jhin',),
    ),
    "Factory New": Set17TraitInfo(
        trait_id="Factory New",
        name="Factory New",
        thresholds=(1,),
        description="After participating in combat, open an armory to purchase a permanent upgrade for your strongest Graves.  Every @NumberOfUpgradesBeforeRoundCostIncrease@ upgrades, future upgrades will take an additional round.  <rules>Next Upgrade: @TFTUnitProperty.trait:TFT17_GravesTrait_RoundsUntilUpgrade@ Rounds.</rules>",
        champions=('TFT17_Graves',),
    ),
    "Fateweaver": Set17TraitInfo(
        trait_id="Fateweaver",
        name="Fateweaver",
        thresholds=(2, 4),
        description="Innate: Fateweavers have <TFTKeyword>Precision</TFTKeyword>.  <row>(@MinUnits@) Chance effects on abilities are <TFTKeyword>Lucky</TFTKeyword>.</row> <row>(@MinUnits@) Gain @CritChance*100@% Crit Chance and @CritDamage@%&nbsp;Crit Damage. Critical strikes are also <TFTKeyword>Lucky</TFTKeyword>.</row>  {{TFT_Keyword_Precision}} <rules>Lucky: Check twice and take the better outcome.</rules>",
        champions=('TFT17_Caitlyn', 'TFT17_TwistedFate', 'TFT17_Milio', 'TFT17_Corki'),
    ),
    "Galaxy Hunter": Set17TraitInfo(
        trait_id="Galaxy Hunter",
        name="Galaxy Hunter",
        thresholds=(1,),
        description="Zed is obtained from the Invader Zed augment.  While at least one clone is alive, Zed gains @BonusAD*100@% bonus Attack Damage.",
        champions=('TFT17_Zed',),
    ),
    "Gun Goddess": Set17TraitInfo(
        trait_id="Gun Goddess",
        name="Gun Goddess",
        thresholds=(1,),
        description="When you field Miss Fortune, choose between Conduit Mode, Challenger Mode, and Replicator Mode. Miss Fortune has a unique ability based on her mode and gains the associated trait.",
        champions=('TFT17_MissFortune',),
    ),
    "Marauder": Set17TraitInfo(
        trait_id="Marauder",
        name="Marauder",
        thresholds=(2, 4, 6),
        description="Your team gains @TeamwideBonus*100@% Omnivamp. Marauders gain more Omnivamp, Attack Damage, and their Omnivamp overhealing is converted into Shield (up to @MaxPercentHealthShield*100@% max Health.)  <row>(@MinUnits@) @Omnivamp*100@% %i:scaleSV%, @AD*100@% %i:scaleAD%</row> <row>(@MinUnits@) @Omnivamp*100@% %i:scaleSV%, @AD*100@% %i:scaleAD%</row> <row>(@MinUnits@) @Omnivamp*100@% %i:scaleSV%, @AD*100@% %i:scaleAD%.</row>",
        champions=('TFT17_Akali', 'TFT17_Belveth', 'TFT17_Urgot', 'TFT17_MasterYi', 'TFT17_Fiora'),
    ),
    "Mecha": Set17TraitInfo(
        trait_id="Mecha",
        name="Mecha",
        thresholds=(3, 4, 6),
        description="Innate: Mecha units can transform into their Ultimate form, upgrading their ability and gaining @TransformedPercentHealth*100@% Health. Transformed Mechas take up two team slots and count twice for the Mecha trait.  <row>(@MinUnits@) Energy Cells: Mechas gain @AP@%&nbsp;%i:scaleAD%%i:scaleAP%.</row> <row>(@MinUnits@) Overclocked Cells: Increased to @AP@%&nbsp;%i:scaleAD%%i:scaleAP%.</row> <row>(@MinUnits@) Precision Engineering: +@TeamSize@ max team size</row>  <rules>Use the Mecha-Former item to toggle the forms of your Mecha units</rules>",
        champions=('TFT17_Urgot', 'TFT17_AurelionSol', 'TFT17_Galio'),
    ),
    "Meeple": Set17TraitInfo(
        trait_id="Meeple",
        name="Meeple",
        thresholds=(3, 5, 7, 10),
        description="Meeple attract Meeps that empower Meeple abilities in meepy ways. They also gain bonus Health.  <row>(@MinUnits@) @Meeps@ %i:set14AmpIcon%, @BonusHealth@ %i:scaleHealth%</row> <row>(@MinUnits@) @Meeps@ %i:set14AmpIcon%, @BonusHealth@ %i:scaleHealth%</row> <row>(@MinUnits@) @Meeps@ %i:set14AmpIcon%, @BonusHealth@ %i:scaleHealth%. Create a Cloning Slot on your bench. Gain gold and a 1-star copy of the champion placed there when cloning completes.</row> <row>(@MinUnits@) @Meeps@ %i:set14AmpIcon%, @BonusHealth@ %i:scaleHealth%. SUMMON THE FOUR MEEPLORDS!</row>  <rules>Cloning time = Champion cost</rules>",
        champions=('TFT17_Poppy', 'TFT17_Veigar', 'TFT17_Gnar', 'TFT17_IvernMinion', 'TFT17_Fizz', 'TFT17_Corki', 'TFT17_Rammus', 'TFT17_Bard'),
    ),
    "N.O.V.A.": Set17TraitInfo(
        trait_id="N.O.V.A.",
        name="N.O.V.A.",
        thresholds=(2, 5),
        description="<row>(@MinUnits@) @TeamAttackDelay@ seconds into combat, N.O.V.A. grant a power surge to allies based on champions.</row> <row>(@MinUnits@) Gain a Striker selector. The chosen N.O.V.A. activates their Strike during the power surge.</row>  <ShowIf.TFT17_DRX_HasAatrox><status>Aatrox:</status> Ally Damage @ShredAndSunder*100@% <TFTKeyword>Shred</TFTKeyword> and <TFTKeyword>Sunders</TFTKeyword> enemies</ShowIf.TFT17_DRX_HasAatrox><ShowIfNot.TFT17_DRX_HasAatrox><TFTGuildInactive>Aatrox: Shred and Sunder enemies</TFTGuildInactive></ShowIfNot.TFT17_DRX_HasAatrox> <ShowIf.TFT17_DRX_HasCaitlyn><status>Caitlyn:</status> Grant allies @AS*100@% Attack Speed</ShowIf.TFT17_DRX_HasCaitlyn><ShowIfNot.TFT17_DRX_HasCaitlyn><TFTGuildInactive>Caitlyn: Grant Attack Speed</TFTGuildInactive></ShowIfNot.TFT17_DRX_HasCaitlyn> <ShowIf.TFT17_DRX_HasAkali><status>Akali:</status> Allies gain <TFTKeyword>Precision</TFTKeyword></ShowIf.TFT17_DRX_HasAkali><ShowIfNot.TFT17_DRX_HasAkali><TFTGuildInactive>Akali: Allies gain Precision</TFTGuildInactive></ShowIfNot.TFT17_DRX_HasAkali> <ShowIf.TFT17_DRX_HasMaokai><status>Maokai:</status> Allies heal @Heal*100@% max Health</ShowIf.TFT17_DRX_HasMaokai><ShowIfNot.TFT17_DRX_HasMaokai><TFTGuildInactive>Maokai: Heal allies</TFTGuildInactive></ShowIfNot.TFT17_DRX_HasMaokai> <ShowIf.TFT17_DRX_HasKindred><status>Kindred:</status> Shield the strongest Tank for <TFTBonus>@ShieldValue@</TFTBonus></ShowIf.TFT17_DRX_HasKindred><ShowIfNot.TFT17_DRX_HasKindred><TFTGuildInactive>Kindred: Shield an ally</TFTGuildInactive></ShowIfNot.TFT17_DRX_HasKindred> <ShowIf.TFT17_DRX_HasEmblem><status>Emblem:</status> Allies deal @BonusTrueDamage*100@% stacking bonus true damage</ShowIf.TFT17_DRX_HasEmblem><ShowIfNot.TFT17_DRX_HasEmblem><TFTGuildInactive>Emblem: Deal bonus true damage</TFTGuildInactive></ShowIfNot.TFT17_DRX_HasEmblem>",
        champions=('TFT17_Aatrox', 'TFT17_Caitlyn', 'TFT17_Akali', 'TFT17_Maokai', 'TFT17_Kindred'),
    ),
    "Oracle": Set17TraitInfo(
        trait_id="Oracle",
        name="Oracle",
        thresholds=(1,),
        description="Every @Rounds@ rounds, Tahm Kench grants a reward!  Rounds Remaining: @TFTUnitProperty.trait:TFT17_TahmKench_RoundsRemaining@ Last Reward: @TFTUnitProperty.trait:TFT17_TahmKench_LastReward@",
        champions=('TFT17_TahmKench',),
    ),
    "Party Animal": Set17TraitInfo(
        trait_id="Party Animal",
        name="Party Animal",
        thresholds=(1,),
        description="Once per combat, after falling below @HealthThreshold*100@% percent Health, become untargetable and repair @PercentHealthHeal*100@% max Health per second. Upon reaching full Health, or when no other allies remain, return to combat. If fully healed, for the rest of combat Blitzcrank is in {{TFT17_SpaceGroove_TheGroove}} and Party Crasher's passive fires bolts four times as fast.",
        champions=('TFT17_Blitzcrank',),
    ),
    "Primordian": Set17TraitInfo(
        trait_id="Primordian",
        name="Primordian",
        thresholds=(2, 3),
        description="<row>(@MinUnits@) Dealing damage spawns Swarmlings based on unique Primordian star level.</row> <row>(@MinUnits@) Spawn @PercentMoreSwarmlings@% more Swarmlings! After each player combat, gain a random 1 or 2-cost champion. </row>  <rules>@DamageTakenPercentModifier*100@% of damage taken contributes to damage dealt.</rules>",
        champions=('TFT17_Briar', 'TFT17_RekSai', 'TFT17_Belveth'),
    ),
    "Psionic": Set17TraitInfo(
        trait_id="Psionic",
        name="Psionic",
        thresholds=(2, 4),
        description="Gain Psionic items that can be equipped to any ally.  <row>(@MinUnits@) Gain the @TFTUnitProperty.trait:TFT17_PsyOps_Item1@</row> <row>(@MinUnits@) Gain the @TFTUnitProperty.trait:TFT17_PsyOps_Item2@, Psionic items gain extra effects on Psionic units</row>",
        champions=('TFT17_Gragas', 'TFT17_Pyke', 'TFT17_Viktor', 'TFT17_MasterYi', 'TFT17_Sona'),
    ),
    "Redeemer": Set17TraitInfo(
        trait_id="Redeemer",
        name="Redeemer",
        thresholds=(1,),
        description="<row>(@MinUnits@) For each non-unique trait you have active, your team gains @BonusOffensiveStat1*100@% Attack Speed, and @BonusDefensiveStat1@ Armor and Magic Resist.</row>  Teamwide Attack Speed: @TFTUnitProperty.trait:TFT17_RhaastUnique_OffensiveStatToGain@% %i:scaleAS% Teamwide Resists: @TFTUnitProperty.trait:TFT17_RhaastUnique_DefensiveStatToGain@ %i:scaleArmor%%i:scaleMR%",
        champions=('TFT17_Rhaast',),
    ),
    "Replicator": Set17TraitInfo(
        trait_id="Replicator",
        name="Replicator",
        thresholds=(2, 4),
        description="Replicator abilities occur a second time at reduced effectiveness.  <expandRow>(@MinUnits@) @Effectiveness*100@% strength</expandRow>",
        champions=('TFT17_Lissandra', 'TFT17_Veigar', 'TFT17_Pantheon', 'TFT17_Lulu', 'TFT17_Nami'),
    ),
    "Rogue": Set17TraitInfo(
        trait_id="Rogue",
        name="Rogue",
        thresholds=(2, 3, 4, 5),
        description="Rogues gain Attack Damage and Ability Power. The first time they fall below @HealthThreshold*100@% health, they slip into shadows. Enemies targeting them are redirected to a nearby unit, preferring Tanks.  <row>(@MinUnits@) @AP@% %i:scaleAD% %i:scaleAP%</row> <row>(@MinUnits@) @AP@% %i:scaleAD% %i:scaleAP%</row> <row>(@MinUnits@) @AP@% %i:scaleAD% %i:scaleAP%</row> <row>(@MinUnits@) @AP@% %i:scaleAD% %i:scaleAP%</row>",
        champions=('TFT17_Briar', 'TFT17_Talon', 'TFT17_Gwen', 'TFT17_Fizz', 'TFT17_Kaisa', 'TFT17_Riven'),
    ),
    "Shepherd": Set17TraitInfo(
        trait_id="Shepherd",
        name="Shepherd",
        thresholds=(3, 5, 7),
        description="Shepherds summon the Bond of the Stars to aid them in battle.  <row>(@MinUnits@) Summon Bia</row> <row>(@MinUnits@) Summon Bayin</row> <row>(@MinUnits@) Bia and Bayin's bond grows deeper</row>  <rules>Bia and Bayin's power are increased by the total star level of all Shepherds.</rules>",
        champions=('TFT17_Lissandra', 'TFT17_Teemo', 'TFT17_IvernMinion', 'TFT17_Illaoi', 'TFT17_Leblanc', 'TFT17_Sona'),
    ),
    "Sniper": Set17TraitInfo(
        trait_id="Sniper",
        name="Sniper",
        thresholds=(2, 3, 4),
        description="Snipers gain Damage Amp, increased against targets farther away.  <row>(@MinUnits@) @PercentDamageIncrease@%&nbsp;%i:scaleDA%; +@PerHexIncrease@%&nbsp;%i:scaleDA% per hex</row> <row>(@MinUnits@) @PercentDamageIncrease@%&nbsp;%i:scaleDA%; +@PerHexIncrease@%&nbsp;%i:scaleDA% per hex</row> <row>(@MinUnits@) @PercentDamageIncrease@%&nbsp;%i:scaleDA%; +@PerHexIncrease@%&nbsp;%i:scaleDA% per hex</row>",
        champions=('TFT17_Ezreal', 'TFT17_Gnar', 'TFT17_Samira', 'TFT17_Xayah', 'TFT17_Jhin'),
    ),
    "Space Groove": Set17TraitInfo(
        trait_id="Space Groove",
        name="Space Groove",
        thresholds=(1, 3, 5, 7, 10),
        description="<row>(@MinUnits@) Groovians can enter {{TFT17_SpaceGroove_TheGroove}}. While in it, they gain Attack Speed and max Health Regen, increased per Groovian on your team.</row> <row>(@MinUnits@) All Groovians start combat in {{TFT17_SpaceGroove_TheGroove}} for @StartOfCombatDuration@ seconds.</row> <row>(@MinUnits@) Each second spent in {{TFT17_SpaceGroove_TheGroove}} grants @ADAPPerSecond@% stacking Attack Damage and Ability Power.</row> <row>(@MinUnits@) Increase these effects by @EffectBonus@%!</row> <row>(@MinUnits@) {{TFT17_SpaceGroove_Groove}}</row>  {{TFT17_SpaceGroove_TheGroove}}: @TFTUnitProperty.:TFT17_SpaceGroove_AS*100@% %i:scaleAS%, @TFTUnitProperty.:TFT17_SpaceGroove_HealthRegen*100@% %i:scaleHPRegen%",
        champions=('TFT17_Nasus', 'TFT17_Teemo', 'TFT17_Gwen', 'TFT17_Ornn', 'TFT17_Samira', 'TFT17_Nami', 'TFT17_Blitzcrank'),
    ),
    "Stargazer": Set17TraitInfo(
        trait_id="Stargazer",
        name="Stargazer",
        thresholds=(3, 5, 7, 8, 9, 10),
        description="Stargazers chart a different constellation every game.  Stargazers in empowered hexes gain various bonuses, starting at (@MinUnits@) units.  <rules>More hexes reveal at each player level.</rules>",
        champions=('TFT17_Talon', 'TFT17_TwistedFate', 'TFT17_Jax', 'TFT17_Lulu', 'TFT17_Nunu', 'TFT17_Xayah', 'TFT17_Vex'),
    ),
    "Timebreaker": Set17TraitInfo(
        trait_id="Timebreaker",
        name="Timebreaker",
        thresholds=(2, 3, 4),
        description="<row>(@MinUnits@) Allies gain @AttackSpeed*100@% Attack Speed.</row> <row>(@MinUnits@) AND When you lose, gain free rerolls. When you win, store XP in a Temporal Core (scales with stage).</row> <row>(@MinUnits@) AND Timebreakers gain an additional @TimebreakerAdditionalAS*100@% Attack Speed</row>  <rules>Rerolls on Loss: @TFTUnitProperty.:TFT17_Timebreaker_NumRerollsTooltip@ XP on Win: @TFTUnitProperty.:TFT17_Timebreaker_NumXPTooltip@</rules>",
        champions=('TFT17_Ezreal', 'TFT17_Milio', 'TFT17_Pantheon', 'TFT17_Riven', 'TFT17_Shen'),
    ),
    "Vanguard": Set17TraitInfo(
        trait_id="Vanguard",
        name="Vanguard",
        thresholds=(2, 4, 6),
        description="Vanguards gain @DamageReductionPct*100@% Durability while Shielded.   Combat start and @HealthThreshold*100@%&nbsp;Health: Gain a max Health Shield for @ShieldDuration@&nbsp;seconds.  <row>(@MinUnits@) @ShieldPercentAmount*100@% max Health</row> <row>(@MinUnits@) @ShieldPercentAmount*100@% max Health</row> <row>(@MinUnits@) @ShieldPercentAmount*100@% max Health; @EnhancedDurability*100@%&nbsp;%i:scaleDR% while Shielded</row>",
        champions=('TFT17_Leona', 'TFT17_Nasus', 'TFT17_Mordekaiser', 'TFT17_Illaoi', 'TFT17_Nunu', 'TFT17_Blitzcrank'),
    ),
    "Voyager": Set17TraitInfo(
        trait_id="Voyager",
        name="Voyager",
        thresholds=(2, 3, 4, 5, 6),
        description="Combat Start: Your Tanks gain a Shield for @ShieldDuration@ seconds. Your other allies gain Damage Amp.   Voyagers gain double.  <expandRow>(@MinUnits@) @ShieldHP@ Shield; @BonusDA*100@% %i:scaleDA%</expandRow>",
        champions=('TFT17_IvernMinion', 'TFT17_Pyke', 'TFT17_Aurora', 'TFT17_Galio', 'TFT17_Karma'),
    ),
}


# =============================================================================
# SET 17 CHAMPION CATALOG (63 CHAMPIONS)
# =============================================================================

@dataclass(frozen=True, slots=True)
class Set17ChampionInfo:
    """Complete specification of a Set 17 champion."""

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


SET17_CHAMPION_CATALOG: tuple[Set17ChampionInfo, ...] = (
    # --- 1-Cost (14 Champions) ---
    Set17ChampionInfo("TFT17_Aatrox", "Aatrox", 1, ('N.O.V.A.', 'Bastion'), UnitRole.TANK, False, True, 700.0, 45.0, 45.0, 50.0, 0.60, 1.0, 30.0, 90.0, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_BrambleVest')),
    Set17ChampionInfo("TFT17_Briar", "Briar", 1, ('Anima', 'Primordian', 'Rogue'), UnitRole.TANK, False, True, 650.0, 35.0, 35.0, 40.0, 0.75, 1.0, 0, 40.0, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_BrambleVest')),
    Set17ChampionInfo("TFT17_Caitlyn", "Caitlyn", 1, ('N.O.V.A.', 'Fateweaver'), UnitRole.AD_CARRY, True, False, 500.0, 15.0, 15.0, 65.0, 0.55, 4.0, 0, 0.0, ('TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_GuinsoosRageblade')),
    Set17ChampionInfo("TFT17_Chogath", "Cho'Gath", 1, ('Dark Star', 'Brawler'), UnitRole.TANK, False, True, 700.0, 40.0, 40.0, 45.0, 0.60, 1.0, 30.0, 70.0, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_BrambleVest')),
    Set17ChampionInfo("TFT17_Ezreal", "Ezreal", 1, ('Timebreaker', 'Sniper'), UnitRole.AD_CARRY, True, False, 450.0, 15.0, 15.0, 45.0, 0.70, 6.0, 0, 30.0, ('TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_GuinsoosRageblade')),
    Set17ChampionInfo("TFT17_Leona", "Leona", 1, ('Arbiter', 'Vanguard'), UnitRole.TANK, False, True, 700.0, 40.0, 40.0, 50.0, 0.60, 1.0, 50.0, 110.0, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_BrambleVest')),
    Set17ChampionInfo("TFT17_Lissandra", "Lissandra", 1, ('Dark Star', 'Shepherd', 'Replicator'), UnitRole.AD_CARRY, True, False, 450.0, 15.0, 15.0, 30.0, 0.70, 4.0, 0, 30.0, ('TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_GuinsoosRageblade')),
    Set17ChampionInfo("TFT17_Nasus", "Nasus", 1, ('Space Groove', 'Vanguard'), UnitRole.TANK, False, True, 700.0, 45.0, 45.0, 40.0, 0.65, 1.0, 60.0, 120.0, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_BrambleVest')),
    Set17ChampionInfo("TFT17_Poppy", "Poppy", 1, ('Meeple', 'Bastion'), UnitRole.TANK, False, True, 700.0, 45.0, 45.0, 60.0, 0.65, 1.0, 30.0, 100, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_BrambleVest')),
    Set17ChampionInfo("TFT17_RekSai", "Rek'Sai", 1, ('Primordian', 'Brawler'), UnitRole.TANK, False, True, 700.0, 45.0, 45.0, 50.0, 0.60, 1.0, 40.0, 100, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_BrambleVest')),
    Set17ChampionInfo("TFT17_Talon", "Talon", 1, ('Stargazer', 'Rogue'), UnitRole.AP_CARRY, True, False, 650.0, 35.0, 35.0, 35.0, 0.75, 1.0, 0, 30.0, ('TFT_Item_JeweledGauntlet', 'TFT_Item_SpearOfShojin', 'TFT_Item_RabadonsDeathcap')),
    Set17ChampionInfo("TFT17_Teemo", "Teemo", 1, ('Space Groove', 'Shepherd'), UnitRole.AD_CARRY, True, False, 450.0, 15.0, 15.0, 15.0, 0.70, 4.0, 0, 50.0, ('TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_GuinsoosRageblade')),
    Set17ChampionInfo("TFT17_TwistedFate", "Twisted Fate", 1, ('Stargazer', 'Fateweaver'), UnitRole.AD_CARRY, True, False, 500.0, 15.0, 15.0, 30.0, 0.70, 4.0, 0, 50.0, ('TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_GuinsoosRageblade')),
    Set17ChampionInfo("TFT17_Veigar", "Veigar", 1, ('Meeple', 'Replicator'), UnitRole.AD_CARRY, True, False, 500.0, 15.0, 15.0, 30.0, 0.70, 4.0, 10.0, 50.0, ('TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_GuinsoosRageblade')),
    # --- 2-Cost (13 Champions) ---
    Set17ChampionInfo("TFT17_Akali", "Akali", 2, ('N.O.V.A.', 'Marauder'), UnitRole.TANK, False, True, 750.0, 45.0, 45.0, 45.0, 0.80, 1.0, 0, 30.0, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_BrambleVest')),
    Set17ChampionInfo("TFT17_Belveth", "Bel'Veth", 2, ('Primordian', 'Challenger', 'Marauder'), UnitRole.TANK, False, True, 750.0, 45.0, 45.0, 47.0, 0.75, 2.0, 0, 50.0, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_BrambleVest')),
    Set17ChampionInfo("TFT17_Gnar", "Gnar", 2, ('Meeple', 'Brawler', 'Sniper'), UnitRole.TANK, False, True, 550.0, 20.0, 20.0, 48.0, 0.75, 6.0, 0, 5.0, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_BrambleVest')),
    Set17ChampionInfo("TFT17_Gragas", "Gragas", 2, ('Psionic', 'Brawler'), UnitRole.TANK, False, True, 950.0, 45.0, 45.0, 50.0, 0.60, 1.0, 30.0, 80.0, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_BrambleVest')),
    Set17ChampionInfo("TFT17_Gwen", "Gwen", 2, ('Space Groove', 'Rogue'), UnitRole.TANK, False, True, 750.0, 50.0, 50.0, 50.0, 0.80, 2.0, 0, 30.0, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_BrambleVest')),
    Set17ChampionInfo("TFT17_IvernMinion", "Meepsie", 2, ('Meeple', 'Shepherd', 'Voyager'), UnitRole.TANK, False, True, 950.0, 45.0, 45.0, 20.0, 0.65, 1.0, 0, 55.0, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_BrambleVest')),
    Set17ChampionInfo("TFT17_Jax", "Jax", 2, ('Stargazer', 'Bastion'), UnitRole.TANK, False, True, 950.0, 45.0, 45.0, 50.0, 0.65, 1.0, 20.0, 80.0, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_BrambleVest')),
    Set17ChampionInfo("TFT17_Jinx", "Jinx", 2, ('Anima', 'Challenger'), UnitRole.AD_CARRY, True, False, 550.0, 20.0, 20.0, 55.0, 0.75, 4.0, 20.0, 80.0, ('TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_GuinsoosRageblade')),
    Set17ChampionInfo("TFT17_Milio", "Milio", 2, ('Timebreaker', 'Fateweaver'), UnitRole.AD_CARRY, True, False, 550.0, 20.0, 20.0, 30.0, 0.70, 4.0, 0, 40.0, ('TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_GuinsoosRageblade')),
    Set17ChampionInfo("TFT17_Mordekaiser", "Mordekaiser", 2, ('Dark Star', 'Conduit', 'Vanguard'), UnitRole.TANK, False, True, 950.0, 45.0, 45.0, 40.0, 0.60, 1.0, 40.0, 100, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_BrambleVest')),
    Set17ChampionInfo("TFT17_Pantheon", "Pantheon", 2, ('Timebreaker', 'Brawler', 'Replicator'), UnitRole.TANK, False, True, 900.0, 45.0, 45.0, 50.0, 0.60, 1.0, 20.0, 80.0, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_BrambleVest')),
    Set17ChampionInfo("TFT17_Pyke", "Pyke", 2, ('Psionic', 'Voyager'), UnitRole.TANK, False, True, 700.0, 45.0, 45.0, 45.0, 0.80, 1.0, 0, 40.0, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_BrambleVest')),
    Set17ChampionInfo("TFT17_Zoe", "Zoe", 2, ('Arbiter', 'Conduit'), UnitRole.AD_CARRY, True, False, 550.0, 20.0, 20.0, 30.0, 0.70, 4.0, 0, 50.0, ('TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_GuinsoosRageblade')),
    # --- 3-Cost (13 Champions) ---
    Set17ChampionInfo("TFT17_Aurora", "Aurora", 3, ('Anima', 'Voyager'), UnitRole.TANK, False, True, 700.0, 25.0, 25.0, 30.0, 0.80, 4.0, 20.0, 80.0, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_BrambleVest')),
    Set17ChampionInfo("TFT17_Diana", "Diana", 3, ('Arbiter', 'Challenger'), UnitRole.TANK, False, True, 850.0, 50.0, 50.0, None, 0.80, 1.0, 0, 40.0, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_BrambleVest')),
    Set17ChampionInfo("TFT17_Fizz", "Fizz", 3, ('Meeple', 'Rogue'), UnitRole.TANK, False, True, 850.0, 55.0, 55.0, 30.0, 0.85, 1.0, 0, 20.0, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_BrambleVest')),
    Set17ChampionInfo("TFT17_Illaoi", "Illaoi", 3, ('Anima', 'Vanguard', 'Shepherd'), UnitRole.TANK, False, True, 1100.0, 50.0, 50.0, 50.0, 0.65, 1.0, 40.0, 100, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_BrambleVest')),
    Set17ChampionInfo("TFT17_Kaisa", "Kai'Sa", 3, ('Dark Star', 'Rogue'), UnitRole.AD_CARRY, True, False, 650.0, 25.0, 25.0, 45.0, 0.80, 4.0, 0, 50.0, ('TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_GuinsoosRageblade')),
    Set17ChampionInfo("TFT17_Lulu", "Lulu", 3, ('Stargazer', 'Replicator'), UnitRole.AD_CARRY, True, False, 650.0, 25.0, 25.0, 30.0, 0.75, 4.0, 0, 55.0, ('TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_GuinsoosRageblade')),
    Set17ChampionInfo("TFT17_Maokai", "Maokai", 3, ('N.O.V.A.', 'Brawler'), UnitRole.TANK, False, True, 1100.0, 40.0, 40.0, 60.0, 0.70, 1.0, 30.0, 100, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_BrambleVest')),
    Set17ChampionInfo("TFT17_MissFortune", "Miss Fortune", 3, ('Gun Goddess', 'Choose Trait'), UnitRole.AD_CARRY, True, False, 650.0, 30.0, 30.0, 50.0, 0.75, 6.0, 0, 100, ('TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_GuinsoosRageblade')),
    Set17ChampionInfo("TFT17_Ornn", "Ornn", 3, ('Space Groove', 'Bastion'), UnitRole.TANK, False, True, 950.0, 40.0, 40.0, 50.0, 0.65, 1.0, 40.0, 100, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_BrambleVest')),
    Set17ChampionInfo("TFT17_Rhaast", "Rhaast", 3, ('Redeemer',), UnitRole.TANK, False, True, 1200.0, 60.0, 60.0, 60.0, 0.65, 1.0, 30.0, 90.0, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_BrambleVest')),
    Set17ChampionInfo("TFT17_Samira", "Samira", 3, ('Space Groove', 'Sniper'), UnitRole.AD_CARRY, True, False, 650.0, 25.0, 25.0, 50.0, 0.75, 6.0, 0, 65.0, ('TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_GuinsoosRageblade')),
    Set17ChampionInfo("TFT17_Urgot", "Urgot", 3, ('Mecha', 'Brawler', 'Marauder'), UnitRole.TANK, False, True, 600.0, 45.0, 45.0, 60.0, 0.80, 2.0, 0, 50.0, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_BrambleVest')),
    Set17ChampionInfo("TFT17_Viktor", "Viktor", 3, ('Psionic', 'Conduit'), UnitRole.AD_CARRY, True, False, 650.0, 25.0, 25.0, 30.0, 0.80, 4.0, 20.0, 80.0, ('TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_GuinsoosRageblade')),
    # --- 4-Cost (14 Champions) ---
    Set17ChampionInfo("TFT17_AurelionSol", "Aurelion Sol", 4, ('Mecha', 'Conduit'), UnitRole.AD_CARRY, True, False, 850.0, 30.0, 30.0, 30.0, 0.75, 6.0, 15.0, 75.0, ('TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_GuinsoosRageblade')),
    Set17ChampionInfo("TFT17_Corki", "Corki", 4, ('Meeple', 'Fateweaver'), UnitRole.AD_CARRY, True, False, 850.0, 30.0, 30.0, 45.0, 0.80, 4.0, 0, 60.0, ('TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_GuinsoosRageblade')),
    Set17ChampionInfo("TFT17_Galio", "The Mighty Mech", 4, ('Mecha', 'Voyager'), UnitRole.TANK, False, True, 1300.0, 60.0, 60.0, 70.0, 0.65, 1.0, 40.0, 100, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_BrambleVest')),
    Set17ChampionInfo("TFT17_Karma", "Karma", 4, ('Dark Star', 'Voyager'), UnitRole.AD_CARRY, True, False, 850.0, 30.0, 30.0, 40.0, 0.80, 4.0, 10.0, 55.0, ('TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_GuinsoosRageblade')),
    Set17ChampionInfo("TFT17_Kindred", "Kindred", 4, ('N.O.V.A.', 'Challenger'), UnitRole.AD_CARRY, True, False, 850.0, 30.0, 30.0, 58.0, 0.80, 6.0, 0, 40.0, ('TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_GuinsoosRageblade')),
    Set17ChampionInfo("TFT17_Leblanc", "LeBlanc", 4, ('Arbiter', 'Shepherd'), UnitRole.AD_CARRY, True, False, 850.0, 30.0, 30.0, 0.0, 0.80, 4.0, 0, 40.0, ('TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_GuinsoosRageblade')),
    Set17ChampionInfo("TFT17_MasterYi", "Master Yi", 4, ('Psionic', 'Marauder'), UnitRole.AD_CARRY, True, False, 1100.0, 65.0, 65.0, 60.0, 0.85, 1.0, 20.0, 60.0, ('TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_GuinsoosRageblade')),
    Set17ChampionInfo("TFT17_Morgana", "Morgana", 4, ('Dark Lady', 'Conduit'), UnitRole.AP_CARRY, True, False, 1300.0, 70.0, 70.0, 60.0, 0.65, 1.0, 45.0, 95.0, ('TFT_Item_JeweledGauntlet', 'TFT_Item_SpearOfShojin', 'TFT_Item_RabadonsDeathcap')),
    Set17ChampionInfo("TFT17_Nami", "Nami", 4, ('Space Groove', 'Replicator'), UnitRole.AD_CARRY, True, False, 850.0, 30.0, 30.0, 40.0, 0.80, 4.0, 20.0, 65.0, ('TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_GuinsoosRageblade')),
    Set17ChampionInfo("TFT17_Nunu", "Nunu & Willump", 4, ('Stargazer', 'Vanguard'), UnitRole.AP_CARRY, True, False, 1300.0, 60.0, 60.0, 60.0, 0.65, 1.0, 40.0, 145.0, ('TFT_Item_JeweledGauntlet', 'TFT_Item_SpearOfShojin', 'TFT_Item_RabadonsDeathcap')),
    Set17ChampionInfo("TFT17_Rammus", "Rammus", 4, ('Meeple', 'Bastion'), UnitRole.TANK, False, True, 1300.0, 60.0, 60.0, 60.0, 0.65, 1.0, 20.0, 80.0, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_BrambleVest')),
    Set17ChampionInfo("TFT17_Riven", "Riven", 4, ('Timebreaker', 'Rogue'), UnitRole.TANK, False, True, 1100.0, 60.0, 60.0, 0.0, 0.85, 1.0, 0, 20.0, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_BrambleVest')),
    Set17ChampionInfo("TFT17_TahmKench", "Tahm Kench", 4, ('Oracle', 'Brawler'), UnitRole.TANK, False, True, 1300.0, 60.0, 60.0, 75.0, 0.50, 1.0, 50.0, 110.0, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_BrambleVest')),
    Set17ChampionInfo("TFT17_Xayah", "Xayah", 4, ('Stargazer', 'Sniper'), UnitRole.AD_CARRY, True, False, 850.0, 30.0, 30.0, 49.0, 0.75, 6.0, 0, 50.0, ('TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_GuinsoosRageblade')),
    # --- 5-Cost (9 Champions) ---
    Set17ChampionInfo("TFT17_Bard", "Bard", 5, ('Meeple', 'Conduit'), UnitRole.AD_CARRY, True, False, 900.0, 40.0, 40.0, 30.0, 0.85, 4.0, 0, 65.0, ('TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_GuinsoosRageblade')),
    Set17ChampionInfo("TFT17_Blitzcrank", "Blitzcrank", 5, ('Party Animal', 'Space Groove', 'Vanguard'), UnitRole.TANK, False, True, 1000.0, 50.0, 50.0, 50.0, 0.90, 1.0, 30.0, 120.0, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_BrambleVest')),
    Set17ChampionInfo("TFT17_Fiora", "Fiora", 5, ('Divine Duelist', 'Anima', 'Marauder'), UnitRole.AD_CARRY, True, False, 1200.0, 65.0, 65.0, 80.0, 0.90, 1.0, 0, 70.0, ('TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_GuinsoosRageblade')),
    Set17ChampionInfo("TFT17_Graves", "Graves", 5, ('Factory New',), UnitRole.AD_CARRY, True, False, 900.0, 40.0, 40.0, 60.0, 0.75, 4.0, 0, 60.0, ('TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_GuinsoosRageblade')),
    Set17ChampionInfo("TFT17_Jhin", "Jhin", 5, ('Dark Star', 'Eradicator', 'Sniper'), UnitRole.AD_CARRY, True, False, 900.0, 40.0, 40.0, 80.0, 0.90, 6.0, 0, 44.0, ('TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_GuinsoosRageblade')),
    Set17ChampionInfo("TFT17_Shen", "Shen", 5, ('Bulwark', 'Timebreaker', 'Bastion'), UnitRole.TANK, False, True, 1300.0, 65.0, 65.0, 50.0, 0.90, 1.0, 20.0, 70.0, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_BrambleVest')),
    Set17ChampionInfo("TFT17_Sona", "Sona", 5, ('Commander', 'Psionic', 'Shepherd'), UnitRole.AD_CARRY, True, False, 900.0, 40.0, 40.0, 35.0, 0.90, 4.0, 0, 25.0, ('TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_GuinsoosRageblade')),
    Set17ChampionInfo("TFT17_Vex", "Vex", 5, ('Doomer', 'Stargazer'), UnitRole.AD_CARRY, True, False, 900.0, 40.0, 40.0, 15.0, 0.80, 6.0, 0, 60.0, ('TFT_Item_InfinityEdge', 'TFT_Item_LastWhisper', 'TFT_Item_GuinsoosRageblade')),
    Set17ChampionInfo("TFT17_Zed", "Zed", 5, ('Galaxy Hunter',), UnitRole.TANK, False, True, 1100.0, 60.0, 60.0, 85.0, 0.85, 1.0, 40.0, 100, ('TFT_Item_WarmogsArmor', 'TFT_Item_DragonsClaw', 'TFT_Item_BrambleVest')),
)


# =============================================================================
# SET 17 ITEMS & RECIPES CATALOG (111 ITEMS, 45 RECIPES)
# =============================================================================

@dataclass(frozen=True, slots=True)
class Set17ItemInfo:
    """Official Item definition with recipe composition."""

    item_id: str
    name: str
    is_component: bool
    recipe: tuple[str, ...]
    description: str


SET17_ITEMS_CATALOG: dict[str, Set17ItemInfo] = {
    "TFT13_Item_AutomataEmblemItem": Set17ItemInfo(
        item_id="TFT13_Item_AutomataEmblemItem",
        name="Automata Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_NegatronCloak'),
        description="The holder gains the Automata trait.",
    ),
    "TFT13_Item_CabalEmblemItem": Set17ItemInfo(
        item_id="TFT13_Item_CabalEmblemItem",
        name="Black Rose Emblem",
        is_component=False,
        recipe=('TFT_Item_NeedlesslyLargeRod', 'TFT_Item_Spatula'),
        description="The holder gains the Black Rose trait.",
    ),
    "TFT13_Item_ExperimentEmblemItem": Set17ItemInfo(
        item_id="TFT13_Item_ExperimentEmblemItem",
        name="Experiment Emblem",
        is_component=False,
        recipe=('TFT_Item_GiantsBelt', 'TFT_Item_Spatula'),
        description="The holder gains the Experiment trait.  <spellActive enabled=TFT13_ExperimentActive alternate=rules>Experiment Bonus: After @SplitTime@ seconds or on death, summon a copy of this unit with <TFTBonus enabled=TFT13_ExperimentActive alternate=rules>@TFTUnitProperty.:TFT13_EmblemCurrentExperimentBonus@%</TFTBonus> max Health.</spellActive>",
    ),
    "TFT13_Item_FamilyEmblemItem": Set17ItemInfo(
        item_id="TFT13_Item_FamilyEmblemItem",
        name="Family Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_TearOfTheGoddess'),
        description="The holder gains the Family trait.",
    ),
    "TFT13_Item_HoverboardEmblemItem": Set17ItemInfo(
        item_id="TFT13_Item_HoverboardEmblemItem",
        name="Firelight Emblem",
        is_component=False,
        recipe=('TFT_Item_SparringGloves', 'TFT_Item_Spatula'),
        description="The holder gains the Firelight trait.",
    ),
    "TFT13_Item_RebelEmblemItem": Set17ItemInfo(
        item_id="TFT13_Item_RebelEmblemItem",
        name="Rebel Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_RecurveBow'),
        description="The holder gains the Rebel trait.",
    ),
    "TFT13_Item_SquadEmblemItem": Set17ItemInfo(
        item_id="TFT13_Item_SquadEmblemItem",
        name="Enforcer Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_ChainVest'),
        description="The holder gains the Enforcer trait.",
    ),
    "TFT13_Item_WarbandEmblemItem": Set17ItemInfo(
        item_id="TFT13_Item_WarbandEmblemItem",
        name="Conqueror Emblem",
        is_component=False,
        recipe=('TFT_Item_BFSword', 'TFT_Item_Spatula'),
        description="The holder gains the Conqueror trait.",
    ),
    "TFT14_Item_BallistekEmblemItem": Set17ItemInfo(
        item_id="TFT14_Item_BallistekEmblemItem",
        name="BoomBot Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_ChainVest'),
        description="The holder gains the BoomBot trait.",
    ),
    "TFT14_Item_ControllerEmblemItem": Set17ItemInfo(
        item_id="TFT14_Item_ControllerEmblemItem",
        name="Strategist Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_BFSword'),
        description="The holder gains the Strategist trait and @SecondaryEffectiveness*100@% of the other Strategist bonus.",
    ),
    "TFT14_Item_DarkWebEmblemItem": Set17ItemInfo(
        item_id="TFT14_Item_DarkWebEmblemItem",
        name="Anima Squad Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_TearOfTheGoddess'),
        description="The holder gains the Anima Squad trait.",
    ),
    "TFT14_Item_DivinicorpEmblemItem": Set17ItemInfo(
        item_id="TFT14_Item_DivinicorpEmblemItem",
        name="Divinicorp Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_RecurveBow'),
        description="The holder gains the Divinicorp trait.",
    ),
    "TFT14_Item_EdgeRunnerEmblemItem": Set17ItemInfo(
        item_id="TFT14_Item_EdgeRunnerEmblemItem",
        name="Exotech Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_SparringGloves'),
        description="The holder gains the Exotech trait.",
    ),
    "TFT14_Item_ImmortalEmblemItem": Set17ItemInfo(
        item_id="TFT14_Item_ImmortalEmblemItem",
        name="Golden Ox Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_GiantsBelt'),
        description="The holder gains the Golden Ox trait.",
    ),
    "TFT14_Item_MobEmblemItem": Set17ItemInfo(
        item_id="TFT14_Item_MobEmblemItem",
        name="Syndicate Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_NegatronCloak'),
        description="The holder gains the Syndicate trait.  <scaleLevel enabled=TFT14_Mob_IsActive_T1 alternate=TFTGuildInactive>Kingpin bonus: <ShowIfNot.TFT14_Mob_IsActive_T3><TFTBonus>+@AttackSpeed_T1*100@%</TFTBonus> Attack Speed and gain <TFTBonus>@Shield_T1*100@%</TFTBonus> max Health Shield at combat start.</ShowIfNot.TFT14_Mob_IsActive_T3><ShowIf.TFT14_Mob_IsActive_T3><TFTBonus>+@AttackSpeed_T2*100@%</TFTBonus> Attack Speed and gain <TFTBonus>@Shield_T2*100@%</TFTBonus> max Health Shield at combat start.</ShowIf.TFT14_Mob_IsActive_T3></scaleLevel>",
    ),
    "TFT14_Item_StreetDemonEmblemItem": Set17ItemInfo(
        item_id="TFT14_Item_StreetDemonEmblemItem",
        name="Street Demon Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_NeedlesslyLargeRod'),
        description="The holder gains the Street Demon trait.",
    ),
    "TFT15_Item_BattleAcademiaEmblemItem": Set17ItemInfo(
        item_id="TFT15_Item_BattleAcademiaEmblemItem",
        name="Battle Academia Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_TearOfTheGoddess'),
        description="The holder gains the Battle Academia trait, and @HealthPerPotential@ Health and @DAPerPotential*100@% Damage Amp per %i:set14AmpIcon%.",
    ),
    "TFT15_Item_CrystalRoseEmblemItem": Set17ItemInfo(
        item_id="TFT15_Item_CrystalRoseEmblemItem",
        name="Crystal Gambit Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_GiantsBelt'),
        description="The holder gains the Crystal Gambit trait.",
    ),
    "TFT15_Item_EmpyreanEmblemItem": Set17ItemInfo(
        item_id="TFT15_Item_EmpyreanEmblemItem",
        name="Wraith Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_ChainVest'),
        description="The holder gains the Wraith trait.",
    ),
    "TFT15_Item_RingKingsEmblemItem": Set17ItemInfo(
        item_id="TFT15_Item_RingKingsEmblemItem",
        name="Luchador Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_SparringGloves'),
        description="The holder gains the Luchador trait.",
    ),
    "TFT15_Item_ShotcallerEmblemItem": Set17ItemInfo(
        item_id="TFT15_Item_ShotcallerEmblemItem",
        name="Strategist Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_NegatronCloak'),
        description="The holder gains the Strategist trait.  Combat Start: The holder and allies within @NumHexes@ hex in the same row gain bonuses. Front 2 rows: @ResistBuff@ Armor and Magic Resist Back 2 rows: @ASBuff*100@% Attack Speed",
    ),
    "TFT15_Item_SoulFighterEmblemItem": Set17ItemInfo(
        item_id="TFT15_Item_SoulFighterEmblemItem",
        name="Soul Fighter Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_BFSword'),
        description="The holder gains the Soul Fighter trait.",
    ),
    "TFT15_Item_StarGuardianEmblemItem": Set17ItemInfo(
        item_id="TFT15_Item_StarGuardianEmblemItem",
        name="Star Guardian Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_NeedlesslyLargeRod'),
        description="The holder gains the Star Guardian trait.  <TFTBonus>Teamwork: </TFTBonus> Non-emblem Star Guardian bonuses are increased by @Percent@%. This effect stacks.  <tftitemrules>[Unique - only 1 per champion.]</tftitemrules>",
    ),
    "TFT15_Item_SupremeCellsEmblemItem": Set17ItemInfo(
        item_id="TFT15_Item_SupremeCellsEmblemItem",
        name="Supreme Cells Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_RecurveBow'),
        description="The holder gains the Supreme Cells trait.",
    ),
    "TFT16_Item_BilgewaterEmblemItem": Set17ItemInfo(
        item_id="TFT16_Item_BilgewaterEmblemItem",
        name="Bilgewater Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_TearOfTheGoddess'),
        description="The holder gains the Bilgewater trait.",
    ),
    "TFT16_Item_DemaciaEmblemItem": Set17ItemInfo(
        item_id="TFT16_Item_DemaciaEmblemItem",
        name="Demacia Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_ChainVest'),
        description="The holder gains the Demacia trait.",
    ),
    "TFT16_Item_FreljordEmblemItem": Set17ItemInfo(
        item_id="TFT16_Item_FreljordEmblemItem",
        name="Freljord Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_GiantsBelt'),
        description="The holder gains the Freljord trait.",
    ),
    "TFT16_Item_IoniaEmblemItem": Set17ItemInfo(
        item_id="TFT16_Item_IoniaEmblemItem",
        name="Ionia Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_NeedlesslyLargeRod'),
        description="The holder gains the Ionia trait.",
    ),
    "TFT16_Item_NoxusEmblemItem": Set17ItemInfo(
        item_id="TFT16_Item_NoxusEmblemItem",
        name="Noxus Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_BFSword'),
        description="The holder gains the Noxus trait.",
    ),
    "TFT16_Item_VoidEmblemItem": Set17ItemInfo(
        item_id="TFT16_Item_VoidEmblemItem",
        name="Void Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_RecurveBow'),
        description="The holder gains the Void trait.",
    ),
    "TFT16_Item_YordleEmblemItem": Set17ItemInfo(
        item_id="TFT16_Item_YordleEmblemItem",
        name="Yordle Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_NegatronCloak'),
        description="The holder gains the Yordle trait.",
    ),
    "TFT16_Item_ZaunEmblemItem": Set17ItemInfo(
        item_id="TFT16_Item_ZaunEmblemItem",
        name="Zaun Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_SparringGloves'),
        description="The holder gains the Zaun trait.",
    ),
    "TFT17_Item_AstronautEmblemItem": Set17ItemInfo(
        item_id="TFT17_Item_AstronautEmblemItem",
        name="Meeple Emblem",
        is_component=False,
        recipe=('TFT_Item_ChainVest', 'TFT_Item_Spatula'),
        description="The holder gains the Meeple trait.",
    ),
    "TFT17_Item_DRXEmblemItem": Set17ItemInfo(
        item_id="TFT17_Item_DRXEmblemItem",
        name="N.O.V.A. Emblem",
        is_component=False,
        recipe=('TFT_Item_SparringGloves', 'TFT_Item_Spatula'),
        description="The holder gains the N.O.V.A. trait.",
    ),
    "TFT17_Item_DarkStarEmblemItem": Set17ItemInfo(
        item_id="TFT17_Item_DarkStarEmblemItem",
        name="Dark Star Emblem",
        is_component=False,
        recipe=('TFT_Item_BFSword', 'TFT_Item_Spatula'),
        description="The holder gains the Dark Star trait.",
    ),
    "TFT17_Item_FavoredEmblemItem": Set17ItemInfo(
        item_id="TFT17_Item_FavoredEmblemItem",
        name="Arbiter Emblem",
        is_component=False,
        recipe=('TFT_Item_NegatronCloak', 'TFT_Item_Spatula'),
        description="The holder gains the Arbiter trait.",
    ),
    "TFT17_Item_PrimordianEmblemItem": Set17ItemInfo(
        item_id="TFT17_Item_PrimordianEmblemItem",
        name="Primordian Emblem",
        is_component=False,
        recipe=('TFT_Item_GiantsBelt', 'TFT_Item_Spatula'),
        description="The holder gains the Primordian trait.",
    ),
    "TFT17_Item_PulsefireEmblemItem": Set17ItemInfo(
        item_id="TFT17_Item_PulsefireEmblemItem",
        name="Timebreaker Emblem",
        is_component=False,
        recipe=('TFT_Item_RecurveBow', 'TFT_Item_Spatula'),
        description="The holder gains the Timebreaker trait.",
    ),
    "TFT17_Item_SpaceGrooveEmblemItem": Set17ItemInfo(
        item_id="TFT17_Item_SpaceGrooveEmblemItem",
        name="Space Groove Emblem",
        is_component=False,
        recipe=('TFT_Item_TearOfTheGoddess', 'TFT_Item_Spatula'),
        description="The holder gains the Space Groove trait.",
    ),
    "TFT17_Item_StargazerEmblemItem": Set17ItemInfo(
        item_id="TFT17_Item_StargazerEmblemItem",
        name="Stargazer Emblem",
        is_component=False,
        recipe=('TFT_Item_NeedlesslyLargeRod', 'TFT_Item_Spatula'),
        description="The holder gains the Stargazer trait.",
    ),
    "TFT3_Item_BattlecastEmblem": Set17ItemInfo(
        item_id="TFT3_Item_BattlecastEmblem",
        name="Battlecast Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_NeedlesslyLargeRod'),
        description="The holder gains the Battlecast trait.",
    ),
    "TFT3_Item_BattlecastSpatulaItem": Set17ItemInfo(
        item_id="TFT3_Item_BattlecastSpatulaItem",
        name="Battlecast Plating",
        is_component=False,
        recipe=('TFT_Item_NeedlesslyLargeRod', 'TFT_Item_Spatula'),
        description="The wearer gains the Battlecast trait.  <tftitemrules>[Unique - Only One Per Champion]</tftitemrules>",
    ),
    "TFT3_Item_BlademasterEmblem": Set17ItemInfo(
        item_id="TFT3_Item_BlademasterEmblem",
        name="Blademaster Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_BFSword'),
        description="The holder gains the Blademaster trait.",
    ),
    "TFT3_Item_BlademasterSpatulaItem": Set17ItemInfo(
        item_id="TFT3_Item_BlademasterSpatulaItem",
        name="tft_item_name_UmbralGlaive",
        is_component=False,
        recipe=('TFT_Item_BFSword', 'TFT_Item_Spatula'),
        description="tft_item_description_SlicerSpatulaItem",
    ),
    "TFT3_Item_CelestialEmblem": Set17ItemInfo(
        item_id="TFT3_Item_CelestialEmblem",
        name="Celestial Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_NegatronCloak'),
        description="The holder gains the Celestial trait.",
    ),
    "TFT3_Item_CelestialSpatulaItem": Set17ItemInfo(
        item_id="TFT3_Item_CelestialSpatulaItem",
        name="tft_item_name_CelestialSpatulaItem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_NegatronCloak'),
        description="tft_item_description_CelestialSpatulaItem",
    ),
    "TFT3_Item_DarkStarEmblem": Set17ItemInfo(
        item_id="TFT3_Item_DarkStarEmblem",
        name="Dark Star Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_SparringGloves'),
        description="The holder gains the Dark Star trait.",
    ),
    "TFT3_Item_DarkStarSpatulaItem": Set17ItemInfo(
        item_id="TFT3_Item_DarkStarSpatulaItem",
        name="tft_item_name_DarkStarSpatulaItem",
        is_component=False,
        recipe=('TFT_Item_SparringGloves', 'TFT_Item_Spatula'),
        description="tft_item_description_DarkStarSpatulaItem",
    ),
    "TFT3_Item_InfiltratorEmblem": Set17ItemInfo(
        item_id="TFT3_Item_InfiltratorEmblem",
        name="Infiltrator Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_RecurveBow'),
        description="The holder gains the Infiltrator trait.",
    ),
    "TFT3_Item_InfiltratorSpatulaItem": Set17ItemInfo(
        item_id="TFT3_Item_InfiltratorSpatulaItem",
        name="tft_item_name_InfiltratorSpatulaItem",
        is_component=False,
        recipe=('TFT_Item_RecurveBow', 'TFT_Item_Spatula'),
        description="tft_item_description_InfiltratorSpatulaItem",
    ),
    "TFT3_Item_ProtectorEmblem": Set17ItemInfo(
        item_id="TFT3_Item_ProtectorEmblem",
        name="Protector Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_GiantsBelt'),
        description="The holder gains the Protector trait.",
    ),
    "TFT3_Item_ProtectorSpatulaItem": Set17ItemInfo(
        item_id="TFT3_Item_ProtectorSpatulaItem",
        name="Protector's Chestguard",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_GiantsBelt'),
        description="The holder gains the Protector trait.  <tftitemrules>[Unique - only 1 per champion]</tftitemrules>",
    ),
    "TFT3_Item_RebelEmblem": Set17ItemInfo(
        item_id="TFT3_Item_RebelEmblem",
        name="Rebel Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_ChainVest'),
        description="The holder gains the Rebel trait.",
    ),
    "TFT3_Item_RebelSpatulaItem": Set17ItemInfo(
        item_id="TFT3_Item_RebelSpatulaItem",
        name="tft_item_name_RebelSpatulaItem",
        is_component=False,
        recipe=('TFT_Item_ChainVest', 'TFT_Item_Spatula'),
        description="tft_item_description_RebelSpatulaItem",
    ),
    "TFT3_Item_StarGuardianEmblem": Set17ItemInfo(
        item_id="TFT3_Item_StarGuardianEmblem",
        name="Star Guardian Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_TearOfTheGoddess'),
        description="The holder gains the Star Guardian trait.",
    ),
    "TFT3_Item_StarGuardianSpatulaItem": Set17ItemInfo(
        item_id="TFT3_Item_StarGuardianSpatulaItem",
        name="tft_item_name_StarGuardianSpatulaItem",
        is_component=False,
        recipe=('TFT_Item_TearOfTheGoddess', 'TFT_Item_Spatula'),
        description="tft_item_description_StarGuardianSpatulaItem",
    ),
    "TFT7_Item_DarkflightEmblemItem": Set17ItemInfo(
        item_id="TFT7_Item_DarkflightEmblemItem",
        name="Darkflight Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_ChainVest'),
        description="The holder gains the Darkflight trait.",
    ),
    "TFT7_Item_GuildEmblemItem": Set17ItemInfo(
        item_id="TFT7_Item_GuildEmblemItem",
        name="Guild Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_NeedlesslyLargeRod'),
        description="The holder gains the Guild trait.  <TFTDebonairVIP>Guild Bonus: @Omnivamp@% Omnivamp</TFTDebonairVIP>",
    ),
    "TFT7_Item_JadeEmblemItem": Set17ItemInfo(
        item_id="TFT7_Item_JadeEmblemItem",
        name="Jade Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_TearOfTheGoddess'),
        description="The holder gains the Jade trait.",
    ),
    "TFT7_Item_LagoonEmblemItem": Set17ItemInfo(
        item_id="TFT7_Item_LagoonEmblemItem",
        name="Lagoon Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_SparringGloves'),
        description="The holder gains the Lagoon trait.",
    ),
    "TFT7_Item_MirageEmblemItem": Set17ItemInfo(
        item_id="TFT7_Item_MirageEmblemItem",
        name="Mirage Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_NegatronCloak'),
        description="The holder gains the Mirage trait.",
    ),
    "TFT7_Item_ShimmerscaleEmblemItem": Set17ItemInfo(
        item_id="TFT7_Item_ShimmerscaleEmblemItem",
        name="Shimmerscale Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_BFSword'),
        description="The holder gains the Shimmerscale trait.",
    ),
    "TFT7_Item_TempestEmblemItem": Set17ItemInfo(
        item_id="TFT7_Item_TempestEmblemItem",
        name="Tempest Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_RecurveBow'),
        description="The holder gains the Tempest trait.",
    ),
    "TFT7_Item_WhispersEmblemItem": Set17ItemInfo(
        item_id="TFT7_Item_WhispersEmblemItem",
        name="Whispers Emblem",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_GiantsBelt'),
        description="The holder gains the Whispers trait.",
    ),
    "TFT_Item_AdaptiveHelm": Set17ItemInfo(
        item_id="TFT_Item_AdaptiveHelm",
        name="Adaptive Helm",
        is_component=False,
        recipe=('TFT_Item_NegatronCloak', 'TFT_Item_TearOfTheGoddess'),
        description="Gain an additional @ManaPercIncrease*100@% Mana from all sources. The holder gains an additional bonus based on their Role:  Tanks and Fighters: Gain @FrontlineResists@ Armor and Magic Resistance.  Other Roles: Gain @BacklineADAP@% Attack Damage and Ability Power.",
    ),
    "TFT_Item_ArchangelsStaff": Set17ItemInfo(
        item_id="TFT_Item_ArchangelsStaff",
        name="Archangel's Staff",
        is_component=False,
        recipe=('TFT_Item_NeedlesslyLargeRod', 'TFT_Item_TearOfTheGoddess'),
        description="Combat start: Gain @APPerInterval@% Ability Power every @IntervalSeconds@ seconds in combat.",
    ),
    "TFT_Item_BFSword": Set17ItemInfo(
        item_id="TFT_Item_BFSword",
        name="B.F. Sword",
        is_component=True,
        recipe=(),
        description="+10 Attack Damage",
    ),
    "TFT_Item_Bloodthirster": Set17ItemInfo(
        item_id="TFT_Item_Bloodthirster",
        name="Bloodthirster",
        is_component=False,
        recipe=('TFT_Item_BFSword', 'TFT_Item_NegatronCloak'),
        description="Once per combat at @HealthThreshold@% Health, gain a @ShieldHealthPercent@% max Health Shield that lasts up to @ShieldDuration@ seconds.",
    ),
    "TFT_Item_BlueBuff": Set17ItemInfo(
        item_id="TFT_Item_BlueBuff",
        name="Blue Buff",
        is_component=False,
        recipe=('TFT_Item_TearOfTheGoddess', 'TFT_Item_TearOfTheGoddess'),
        description="Gain @ModifiedADAP*100@% additional Attack Damage and Ability Power from all sources.",
    ),
    "TFT_Item_BrambleVest": Set17ItemInfo(
        item_id="TFT_Item_BrambleVest",
        name="Bramble Vest",
        is_component=False,
        recipe=('TFT_Item_ChainVest', 'TFT_Item_ChainVest'),
        description="Gain @PercentMaxHP*100@% max health.  Take @AutoDamageReduction*100@% reduced damage from attacks. When struck by any attack, deal <magicDamage>@1StarAoEDamage@ magic damage</magicDamage> to all adjacent enemies.  <tftitemrules>Cooldown: @ICD@ seconds</tftitemrules>",
    ),
    "TFT_Item_ChainVest": Set17ItemInfo(
        item_id="TFT_Item_ChainVest",
        name="Chain Vest",
        is_component=True,
        recipe=(),
        description="+20 Armor",
    ),
    "TFT_Item_Crownguard": Set17ItemInfo(
        item_id="TFT_Item_Crownguard",
        name="Crownguard",
        is_component=False,
        recipe=('TFT_Item_NeedlesslyLargeRod', 'TFT_Item_ChainVest'),
        description="Combat Start: Gain a @ShieldSize@% max Health Shield for @ShieldDuration@ seconds.  When the Shield expires, gain @ShieldBonusAP@% Ability Power.",
    ),
    "TFT_Item_Deathblade": Set17ItemInfo(
        item_id="TFT_Item_Deathblade",
        name="Deathblade",
        is_component=False,
        recipe=('TFT_Item_BFSword', 'TFT_Item_BFSword'),
        description="<tftitemrules>Perfect peace and calm for the holder - and all who face it.</tftitemrules>  @TFTUnitProperty.:TFT_Augment_TragicalBlade_TRAKey@",
    ),
    "TFT_Item_DragonsClaw": Set17ItemInfo(
        item_id="TFT_Item_DragonsClaw",
        name="Dragon's Claw",
        is_component=False,
        recipe=('TFT_Item_NegatronCloak', 'TFT_Item_NegatronCloak'),
        description="Gain @PercentMaxHP*100@% max health.  Every @HealthRegenInterval@ seconds, heal @PercentHealthDamage@% max Health.",
    ),
    "TFT_Item_FrozenHeart": Set17ItemInfo(
        item_id="TFT_Item_FrozenHeart",
        name="Protector's Vow",
        is_component=False,
        recipe=('TFT_Item_TearOfTheGoddess', 'TFT_Item_ChainVest'),
        description="Combat Start: Gain @CombatStartMana@ Mana.  At @HealthThreshold@% Health, gain @TriggerMana@ Mana and a Shield equal to @ShieldHealthPercent@% max Health.",
    ),
    "TFT_Item_GargoyleStoneplate": Set17ItemInfo(
        item_id="TFT_Item_GargoyleStoneplate",
        name="Gargoyle Stoneplate",
        is_component=False,
        recipe=('TFT_Item_ChainVest', 'TFT_Item_NegatronCloak'),
        description="Gain @ArmorPerEnemy@ Armor and @MRPerEnemy@ Magic Resist for each enemy targeting the holder.",
    ),
    "TFT_Item_GiantsBelt": Set17ItemInfo(
        item_id="TFT_Item_GiantsBelt",
        name="Giant's Belt",
        is_component=True,
        recipe=(),
        description="+150 Health",
    ),
    "TFT_Item_GuardianAngel": Set17ItemInfo(
        item_id="TFT_Item_GuardianAngel",
        name="Edge of Night",
        is_component=False,
        recipe=('TFT_Item_BFSword', 'TFT_Item_ChainVest'),
        description="At @HealthThreshold@% Health, briefly become untargetable, shed negative effects, and heal @MissingHealthRestore*100@% missing health.",
    ),
    "TFT_Item_GuinsoosRageblade": Set17ItemInfo(
        item_id="TFT_Item_GuinsoosRageblade",
        name="Guinsoo's Rageblade",
        is_component=False,
        recipe=('TFT_Item_RecurveBow', 'TFT_Item_NeedlesslyLargeRod'),
        description="Gain @AttackSpeedPerStack@% stacking Attack Speed every second.",
    ),
    "TFT_Item_HandOfJustice": Set17ItemInfo(
        item_id="TFT_Item_HandOfJustice",
        name="Hand Of Justice",
        is_component=False,
        recipe=('TFT_Item_TearOfTheGoddess', 'TFT_Item_SparringGloves'),
        description="Gain 2 effects:<li>@AD_NotStatBar*100@% Attack Damage and @AP_NotStatBar@% Ability Power.<li>@StatOmnivamp_NotStatBar*100@% Omnivamp.  While above @HealthThreshold*100@% health, double the Attack Damage and Ability Power. While below @HealthThreshold*100@% Health, double the Omnivamp.",
    ),
    "TFT_Item_HextechGunblade": Set17ItemInfo(
        item_id="TFT_Item_HextechGunblade",
        name="Hextech Gunblade",
        is_component=False,
        recipe=('TFT_Item_BFSword', 'TFT_Item_NeedlesslyLargeRod'),
        description="Heal the lowest percent Health ally for @AllyHealing*100@% of damage dealt.  <TFTTrackerLabel>Ally Healing:</TFTTrackerLabel> <TFTHighlight>@TFTUnitProperty.item:TFT_Tracker_Value1@</TFTHighlight>",
    ),
    "TFT_Item_InfinityEdge": Set17ItemInfo(
        item_id="TFT_Item_InfinityEdge",
        name="Infinity Edge",
        is_component=False,
        recipe=('TFT_Item_BFSword', 'TFT_Item_SparringGloves'),
        description="Gain <TFTKeyword>Precision</TFTKeyword>.  {{TFT_Keyword_Precision}}",
    ),
    "TFT_Item_IonicSpark": Set17ItemInfo(
        item_id="TFT_Item_IonicSpark",
        name="Ionic Spark",
        is_component=False,
        recipe=('TFT_Item_NeedlesslyLargeRod', 'TFT_Item_NegatronCloak'),
        description="@MRShred@% <TFTKeyword>Shred</TFTKeyword> enemies within @HexRange@ hexes. When enemies cast an Ability, deal magic damage equal to @ManaRatio@% of the Mana spent  <tftitemrules><tftbold>Shred</tftbold>: Reduce Magic Resist</tftitemrules>",
    ),
    "TFT_Item_JeweledGauntlet": Set17ItemInfo(
        item_id="TFT_Item_JeweledGauntlet",
        name="Jeweled Gauntlet",
        is_component=False,
        recipe=('TFT_Item_NeedlesslyLargeRod', 'TFT_Item_SparringGloves'),
        description="Gain <TFTKeyword>Precision</TFTKeyword>.  {{TFT_Keyword_Precision}}",
    ),
    "TFT_Item_LastWhisper": Set17ItemInfo(
        item_id="TFT_Item_LastWhisper",
        name="Last Whisper",
        is_component=False,
        recipe=('TFT_Item_RecurveBow', 'TFT_Item_SparringGloves'),
        description="Damage from attacks and Abilities @ArmorReductionPercent@% <TFTKeyword>Sunder</TFTKeyword> the target for @ArmorBreakDuration@ seconds. This effect does not stack.  <tftitemrules><tftbold>Sunder</tftbold>: Reduce Armor</tftitemrules>",
    ),
    "TFT_Item_Leviathan": Set17ItemInfo(
        item_id="TFT_Item_Leviathan",
        name="Nashor's Tooth",
        is_component=False,
        recipe=('TFT_Item_RecurveBow', 'TFT_Item_GiantsBelt'),
        description="Attacks grant @BaseManaOnHit@ bonus Mana, increased to @ManaOnCrit@ if they critically strike.",
    ),
    "TFT_Item_MadredsBloodrazor": Set17ItemInfo(
        item_id="TFT_Item_MadredsBloodrazor",
        name="Giant Slayer",
        is_component=False,
        recipe=('TFT_Item_BFSword', 'TFT_Item_RecurveBow'),
        description="Gain @DamageAmp*100@% additional Damage Amp against Tanks.",
    ),
    "TFT_Item_Morellonomicon": Set17ItemInfo(
        item_id="TFT_Item_Morellonomicon",
        name="Morellonomicon",
        is_component=False,
        recipe=('TFT_Item_NeedlesslyLargeRod', 'TFT_Item_GiantsBelt'),
        description="Attacks and Abilities deal @BurnPercent@% <TFTKeyword>Burn</TFTKeyword> and @GrievousWoundsPercent@% <TFTKeyword>Wound</TFTKeyword> to enemies for @BurnDuration@ seconds.  <tftitemrules><tftbold>Burn</tftbold>: Deals a percent of the target's max Health as true damage every second <tftbold>Wound</tftbold>: Reduces healing received</tftitemrules>",
    ),
    "TFT_Item_NeedlesslyLargeRod": Set17ItemInfo(
        item_id="TFT_Item_NeedlesslyLargeRod",
        name="Needlessly Large Rod",
        is_component=True,
        recipe=(),
        description="+10 Ability Power",
    ),
    "TFT_Item_NegatronCloak": Set17ItemInfo(
        item_id="TFT_Item_NegatronCloak",
        name="Negatron Cloak",
        is_component=True,
        recipe=(),
        description="+20 Magic Resist",
    ),
    "TFT_Item_NightHarvester": Set17ItemInfo(
        item_id="TFT_Item_NightHarvester",
        name="Steadfast Heart",
        is_component=False,
        recipe=('TFT_Item_ChainVest', 'TFT_Item_SparringGloves'),
        description="Gain @BaseDurability*100@% Durability. While above @ThresholdForEmpower*100@% Health, instead gain @EmpoweredDurability*100@% Durability.  @TFTUnitProperty.:TFT_Augment_WarmogsBuckle_TRAKey@",
    ),
    "TFT_Item_PowerGauntlet": Set17ItemInfo(
        item_id="TFT_Item_PowerGauntlet",
        name="Striker's Flail",
        is_component=False,
        recipe=('TFT_Item_GiantsBelt', 'TFT_Item_SparringGloves'),
        description="Critical Strikes grant @BuffDamageAmp*100@% Damage Amp for @Duration@ seconds, stacking up to @MaxStacks@ times.",
    ),
    "TFT_Item_Quicksilver": Set17ItemInfo(
        item_id="TFT_Item_Quicksilver",
        name="Quicksilver",
        is_component=False,
        recipe=('TFT_Item_SparringGloves', 'TFT_Item_NegatronCloak'),
        description="Combat Start: Gain immunity to crowd control for @SpellShieldDuration@ seconds.  Gain @ProcAttackSpeed*100@% stacking Attack Speed every second.",
    ),
    "TFT_Item_RabadonsDeathcap": Set17ItemInfo(
        item_id="TFT_Item_RabadonsDeathcap",
        name="Rabadon's Deathcap",
        is_component=False,
        recipe=('TFT_Item_NeedlesslyLargeRod', 'TFT_Item_NeedlesslyLargeRod'),
        description="<tftitemrules>This humble hat can help you make, or unmake, the world itself.</tftitemrules>  @TFTUnitProperty.:TFT_Augment_DeadlierCaps_TRAKey@",
    ),
    "TFT_Item_RapidFireCannon": Set17ItemInfo(
        item_id="TFT_Item_RapidFireCannon",
        name="Red Buff",
        is_component=False,
        recipe=('TFT_Item_RecurveBow', 'TFT_Item_RecurveBow'),
        description="Attacks and Abilities @BurnPercent@% <TFTKeyword>Burn</TFTKeyword> and @HealingReductionPct@% <TFTKeyword>Wound</TFTKeyword> enemies for @Duration@ seconds.  <tftitemrules><tftbold>Burn</tftbold>: Deals a percent of the target's max Health as true damage every second <tftbold>Wound</tftbold>: Reduces healing received</tftitemrules>",
    ),
    "TFT_Item_RecurveBow": Set17ItemInfo(
        item_id="TFT_Item_RecurveBow",
        name="Recurve Bow",
        is_component=True,
        recipe=(),
        description="+10% Attack Speed",
    ),
    "TFT_Item_RedBuff": Set17ItemInfo(
        item_id="TFT_Item_RedBuff",
        name="Sunfire Cape",
        is_component=False,
        recipe=('TFT_Item_ChainVest', 'TFT_Item_GiantsBelt'),
        description="Gain @BonusPercentHP*100@% max Health.   Every @ICD@ seconds, deal @BurnPercent@% <TFTKeyword>Burn</TFTKeyword> and @GrievousWoundsPercent@% <TFTKeyword>Wound</TFTKeyword> to an enemy within @HexRange@ hexes for @BurnDuration@ seconds.  <tftitemrules><tftbold>Burn</tftbold>: Deals a percent of the target's max Health as true damage every second <tftbold>Wound</tftbold>: Reduces healing received</tftitemrules>",
    ),
    "TFT_Item_Redemption": Set17ItemInfo(
        item_id="TFT_Item_Redemption",
        name="Spirit Visage",
        is_component=False,
        recipe=('TFT_Item_TearOfTheGoddess', 'TFT_Item_GiantsBelt'),
        description="Regenerate @MissingHealthHeal*100@% of missing Health each second.",
    ),
    "TFT_Item_RunaansHurricane": Set17ItemInfo(
        item_id="TFT_Item_RunaansHurricane",
        name="Kraken's Fury",
        is_component=False,
        recipe=('TFT_Item_NegatronCloak', 'TFT_Item_RecurveBow'),
        description="Attacks grant @ADOnAttack*100@% stacking Attack Damage, up to @MaxStacks@ attacks. After @MaxStacks@ attacks, gain @ASCapstone*100@% Attack Speed.",
    ),
    "TFT_Item_SparringGloves": Set17ItemInfo(
        item_id="TFT_Item_SparringGloves",
        name="Sparring Gloves",
        is_component=True,
        recipe=(),
        description="+20% Critical Strike Chance",
    ),
    "TFT_Item_Spatula": Set17ItemInfo(
        item_id="TFT_Item_Spatula",
        name="Spatula",
        is_component=True,
        recipe=(),
        description="Special Emblem and Team Size Component",
    ),
    "TFT_Item_SpearOfShojin": Set17ItemInfo(
        item_id="TFT_Item_SpearOfShojin",
        name="Spear of Shojin",
        is_component=False,
        recipe=('TFT_Item_BFSword', 'TFT_Item_TearOfTheGoddess'),
        description="Attacks grant @FlatManaRestore@ bonus Mana.",
    ),
    "TFT_Item_SpectralGauntlet": Set17ItemInfo(
        item_id="TFT_Item_SpectralGauntlet",
        name="Evenshroud",
        is_component=False,
        recipe=('TFT_Item_NegatronCloak', 'TFT_Item_GiantsBelt'),
        description="@ARReductionAmount@% <TFTKeyword>Sunder</TFTKeyword> enemies within @HexRange@ hexes. Gain @BonusResists@ Armor and Magic Resist for the first @BonusResistDuration@ seconds of combat.  <tftitemrules><tftbold>Sunder</tftbold>: Reduce Armor</tftitemrules>",
    ),
    "TFT_Item_StatikkShiv": Set17ItemInfo(
        item_id="TFT_Item_StatikkShiv",
        name="Void Staff",
        is_component=False,
        recipe=('TFT_Item_RecurveBow', 'TFT_Item_TearOfTheGoddess'),
        description="Damage from attacks and Abilities @MRShred@% <TFTKeyword>Shred</TFTKeyword> the target for @MRShredDuration@ seconds. This effect does not stack.  <tftitemrules><tftbold>Shred</tftbold>: Reduce Magic Resist</tftitemrules>",
    ),
    "TFT_Item_SteraksGage": Set17ItemInfo(
        item_id="TFT_Item_SteraksGage",
        name="Sterak's Gage",
        is_component=False,
        recipe=('TFT_Item_BFSword', 'TFT_Item_GiantsBelt'),
        description="At @HealthThreshold@% Health, gain a Shield equal to @PercentHealthShield*100@% of the wearer's maximum Health that rapidly decays over @ShieldDuration@ seconds.",
    ),
    "TFT_Item_TacticiansCrown": Set17ItemInfo(
        item_id="TFT_Item_TacticiansCrown",
        name="Tactician's Crown",
        is_component=False,
        recipe=('TFT_Item_Spatula', 'TFT_Item_Spatula'),
        description="Your team gains +@MaxArmySizeIncrease@ max team size.  @PercentGoldChance@% chance to drop 1 gold when you win combat.  <tftitemrules>\"...the Heart of a hero...\"</tftitemrules>",
    ),
    "TFT_Item_TearOfTheGoddess": Set17ItemInfo(
        item_id="TFT_Item_TearOfTheGoddess",
        name="Tear of the Goddess",
        is_component=True,
        recipe=(),
        description="+15 Starting Mana",
    ),
    "TFT_Item_ThiefsGloves": Set17ItemInfo(
        item_id="TFT_Item_ThiefsGloves",
        name="Thief's Gloves",
        is_component=False,
        recipe=('TFT_Item_SparringGloves', 'TFT_Item_SparringGloves'),
        description="Each round: Equip 2 random items.  <tftitemrules>[Consumes 3 item slots.]</tftitemrules> @TFTUnitProperty.:TFT_BindOnEquipTRA@",
    ),
    "TFT_Item_TitansResolve": Set17ItemInfo(
        item_id="TFT_Item_TitansResolve",
        name="Titan's Resolve",
        is_component=False,
        recipe=('TFT_Item_ChainVest', 'TFT_Item_RecurveBow'),
        description="Gain @StackingAD*100@% Attack Damage and @StackingSP@% Ability Power when attacking or taking damage, stacking up to @StackCap@ times.    At full stacks, gain @StackedAmp*100@% Damage Amp and gain immunity to crowd control.",
    ),
    "TFT_Item_UnstableConcoction": Set17ItemInfo(
        item_id="TFT_Item_UnstableConcoction",
        name="Hand Of Justice",
        is_component=False,
        recipe=('TFT_Item_TearOfTheGoddess', 'TFT_Item_SparringGloves'),
        description="Gain 2 effects:<li>@AD_NotStatBar*100@% Attack Damage and @AP_NotStatBar@% Ability Power.<li>@StatOmnivamp_NotStatBar*100@% Omnivamp.  While above @HealthThreshold*100@% health, double the Attack Damage and Ability Power. While below @HealthThreshold*100@% Health, double the Omnivamp.",
    ),
    "TFT_Item_WarmogsArmor": Set17ItemInfo(
        item_id="TFT_Item_WarmogsArmor",
        name="Warmog's Armor",
        is_component=False,
        recipe=('TFT_Item_GiantsBelt', 'TFT_Item_GiantsBelt'),
        description="Gain @BonusPercentHP*100@% max Health.",
    ),
}

SET17_ITEM_RECIPES: dict[frozenset[str], str] = {
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
class Set17Profile:
    """Top-level container for Set 17 profile specifications."""

    set_name: str = "TFTSet17"
    total_champions: int = 63
    total_traits: int = 35
    total_items: int = 111
    total_recipes: int = 45
    champions: tuple[Set17ChampionInfo, ...] = SET17_CHAMPION_CATALOG
    traits: dict[str, Set17TraitInfo] = field(default_factory=lambda: dict(SET17_TRAIT_CATALOG))
    items: dict[str, Set17ItemInfo] = field(default_factory=lambda: dict(SET17_ITEMS_CATALOG))

    def to_dict(self) -> dict[str, Any]:
        """Convert profile to serializable dictionary."""
        return {
            "set_name": self.set_name,
            "stats": {
                "total_champions": len(self.champions),
                "total_traits": len(self.traits),
                "total_items": len(self.items),
                "total_recipes": len(SET17_ITEM_RECIPES),
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
                    "result_name": self.items.get(res_id, Set17ItemInfo(res_id, res_id, False, (), "")).name,
                }
                for comp, res_id in SET17_ITEM_RECIPES.items()
            ],
            "shop_odds": {str(k): list(v) for k, v in STANDARD_SHOP_ODDS.items()},
            "level_exp": {str(k): v for k, v in STANDARD_LEVEL_EXP.items()},
            "pool_sizes": {str(k): v for k, v in STANDARD_POOL_SIZES.items()},
        }


def get_set17_data() -> SetData:
    """Build a complete SetData instance for Set 17 simulations."""
    champions: dict[str, ChampionDef] = {
        c.champion_id: ChampionDef(
            champion_id=c.champion_id,
            name=c.name,
            cost=c.cost,
            traits=c.traits,
            role=c.role,
        )
        for c in SET17_CHAMPION_CATALOG
    }

    traits: dict[str, TraitDef] = {
        t.trait_id: TraitDef(
            trait_id=t.trait_id,
            name=t.name,
            thresholds=t.thresholds,
        )
        for t in SET17_TRAIT_CATALOG.values()
    }

    items: dict[str, ItemDef] = {
        it.item_id: ItemDef(
            item_id=it.item_id,
            name=it.name,
            is_component=it.is_component,
            recipe=it.recipe if len(it.recipe) == 2 else None,
        )
        for it in SET17_ITEMS_CATALOG.values()
    }

    recipes: dict[frozenset[str], str] = dict(SET17_ITEM_RECIPES)

    return SetData(
        set_name="TFTSet17",
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


def export_set17_profile_json(output_path: str | Path = "data/set17_profile.json") -> Path:
    """Export complete Set 17 profile JSON to file."""
    profile = Set17Profile()
    data = profile.to_dict()
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
