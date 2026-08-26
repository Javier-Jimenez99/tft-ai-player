"""Generate official Set 18 'Enchanted Wilds' profile directly from Riot CommunityDragon dataset."""

import json
from pathlib import Path

# Complete 65 champions roster for Set 18 "Enchanted Wilds"
SET18_CHAMPIONS_DATA = [
    # 1-Cost (14 Champions)
    ("TFT18_Akali", "Akali", 1, ("Inferno", "Adaptor", "Ravager"), "UnitRole.AD_CARRY", True, False, 500.0, 25.0, 25.0, 45.0, 0.75, 1.0, 0.0, 40.0, ("TFT_Item_InfinityEdge", "TFT_Item_HandOfJustice", "TFT_Item_Bloodthirster")),
    ("TFT18_Camille", "Camille", 1, ("Coven", "Ravager"), "UnitRole.AD_CARRY", True, False, 550.0, 30.0, 30.0, 45.0, 0.70, 1.0, 0.0, 50.0, ("TFT_Item_Bloodthirster", "TFT_Item_TitansResolve", "TFT_Item_SteraksGage")),
    ("TFT18_Cinderling", "Cinderling", 1, ("Riftbeast", "Hunter"), "UnitRole.AD_CARRY", True, False, 450.0, 15.0, 15.0, 45.0, 0.70, 4.0, 0.0, 40.0, ("TFT_Item_GuinsoosRageblade", "TFT_Item_LastWhisper", "TFT_Item_InfinityEdge")),
    ("TFT18_Karma", "Karma", 1, ("Blossom", "Invoker"), "UnitRole.AP_CARRY", True, False, 500.0, 20.0, 20.0, 40.0, 0.65, 4.0, 0.0, 50.0, ("TFT_Item_BlueBuff", "TFT_Item_JeweledGauntlet", "TFT_Item_ArchangelsStaff")),
    ("TFT18_Kobuko", "Kobuko", 1, ("Sprykin", "Brawler"), "UnitRole.TANK", False, True, 650.0, 40.0, 40.0, 45.0, 0.60, 1.0, 20.0, 80.0, ("TFT_Item_WarmogsArmor", "TFT_Item_BrambleVest", "TFT_Item_DragonsClaw")),
    ("TFT18_Leona", "Leona", 1, ("Solar", "Vanguard"), "UnitRole.TANK", False, True, 650.0, 45.0, 45.0, 45.0, 0.60, 1.0, 0.0, 60.0, ("TFT_Item_GargoyleStoneplate", "TFT_Item_SunfireCape", "TFT_Item_DragonsClaw")),
    ("TFT18_Ornn", "Ornn", 1, ("Elderwood", "Defender"), "UnitRole.TANK", False, True, 650.0, 45.0, 45.0, 50.0, 0.55, 1.0, 40.0, 100.0, ("TFT_Item_BrambleVest", "TFT_Item_DragonsClaw", "TFT_Item_WarmogsArmor")),
    ("TFT18_Pebbles", "Pebbles", 1, ("Riftbeast", "Defender"), "UnitRole.TANK", False, True, 600.0, 45.0, 45.0, 40.0, 0.60, 1.0, 0.0, 60.0, ("TFT_Item_BrambleVest", "TFT_Item_DragonsClaw", "TFT_Item_SunfireCape")),
    ("TFT18_Rakan", "Rakan", 1, ("Fae", "Spellweaver"), "UnitRole.UTILITY", False, False, 550.0, 30.0, 30.0, 40.0, 0.65, 2.0, 20.0, 70.0, ("TFT_Item_SpearOfShojin", "TFT_Item_IonicSpark", "TFT_Item_Crownguard")),
    ("TFT18_RekSai", "Rek'Sai", 1, ("Blackthorn", "Brawler"), "UnitRole.TANK", False, True, 650.0, 40.0, 40.0, 50.0, 0.65, 1.0, 0.0, 60.0, ("TFT_Item_Bloodthirster", "TFT_Item_TitansResolve", "TFT_Item_WarmogsArmor")),
    ("TFT18_Varus", "Varus", 1, ("Inferno", "Hunter"), "UnitRole.AD_CARRY", True, False, 500.0, 15.0, 15.0, 50.0, 0.70, 4.0, 0.0, 60.0, ("TFT_Item_InfinityEdge", "TFT_Item_LastWhisper", "TFT_Item_Deathblade")),
    ("TFT18_Veigar", "Veigar", 1, ("Blackthorn", "Spellweaver"), "UnitRole.AP_CARRY", True, False, 450.0, 15.0, 15.0, 35.0, 0.65, 4.0, 0.0, 45.0, ("TFT_Item_BlueBuff", "TFT_Item_JeweledGauntlet", "TFT_Item_RabadonsDeathcap")),
    ("TFT18_Xayah", "Xayah", 1, ("Fae", "Rapidfire"), "UnitRole.AD_CARRY", True, False, 500.0, 15.0, 15.0, 50.0, 0.75, 4.0, 0.0, 50.0, ("TFT_Item_GuinsoosRageblade", "TFT_Item_LastWhisper", "TFT_Item_InfinityEdge")),
    ("TFT18_Yorick", "Yorick", 1, ("Coven", "Vanguard"), "UnitRole.TANK", False, True, 650.0, 40.0, 40.0, 45.0, 0.60, 1.0, 20.0, 80.0, ("TFT_Item_WarmogsArmor", "TFT_Item_BrambleVest", "TFT_Item_SunfireCape")),

    # 2-Cost (13 Champions)
    ("TFT18_Alistar", "Alistar", 2, ("Blossom", "Juggernaut"), "UnitRole.TANK", False, True, 750.0, 45.0, 45.0, 55.0, 0.60, 1.0, 30.0, 90.0, ("TFT_Item_WarmogsArmor", "TFT_Item_BrambleVest", "TFT_Item_DragonsClaw")),
    ("TFT18_Caitlyn", "Caitlyn", 2, ("Solar", "Hunter"), "UnitRole.AD_CARRY", True, False, 550.0, 20.0, 20.0, 55.0, 0.70, 5.0, 0.0, 60.0, ("TFT_Item_InfinityEdge", "TFT_Item_LastWhisper", "TFT_Item_SpearOfShojin")),
    ("TFT18_Elise", "Elise", 2, ("Coven", "Summoner"), "UnitRole.UTILITY", False, False, 600.0, 30.0, 30.0, 45.0, 0.65, 2.0, 20.0, 80.0, ("TFT_Item_SpearOfShojin", "TFT_Item_StatikkShiv", "TFT_Item_Morellonomicon")),
    ("TFT18_Gromp", "Gromp", 2, ("Riftbeast", "Adaptor"), "UnitRole.TANK", False, True, 750.0, 40.0, 40.0, 50.0, 0.60, 1.0, 30.0, 90.0, ("TFT_Item_WarmogsArmor", "TFT_Item_DragonsClaw", "TFT_Item_GargoyleStoneplate")),
    ("TFT18_Kayle", "Kayle", 2, ("Solar", "Spellweaver"), "UnitRole.AP_CARRY", True, False, 550.0, 20.0, 20.0, 40.0, 0.75, 4.0, 0.0, 40.0, ("TFT_Item_GuinsoosRageblade", "TFT_Item_JeweledGauntlet", "TFT_Item_ArchangelsStaff")),
    ("TFT18_LeBlanc", "LeBlanc", 2, ("Blackthorn", "Invoker"), "UnitRole.AP_CARRY", True, False, 550.0, 20.0, 20.0, 40.0, 0.65, 4.0, 0.0, 50.0, ("TFT_Item_BlueBuff", "TFT_Item_JeweledGauntlet", "TFT_Item_RabadonsDeathcap")),
    ("TFT18_Murkwolf", "Murkwolf", 2, ("Riftbeast", "Ravager"), "UnitRole.AD_CARRY", True, False, 600.0, 30.0, 30.0, 55.0, 0.80, 1.0, 0.0, 40.0, ("TFT_Item_InfinityEdge", "TFT_Item_Bloodthirster", "TFT_Item_HandOfJustice")),
    ("TFT18_Scuttlecrab", "Scuttlecrab", 2, ("Riftbeast", "Brawler"), "UnitRole.TANK", False, True, 750.0, 40.0, 40.0, 45.0, 0.60, 1.0, 0.0, 80.0, ("TFT_Item_WarmogsArmor", "TFT_Item_DragonsClaw", "TFT_Item_Redemption")),
    ("TFT18_Sejuani", "Sejuani", 2, ("Primal", "Vanguard"), "UnitRole.TANK", False, True, 750.0, 45.0, 45.0, 50.0, 0.60, 1.0, 40.0, 100.0, ("TFT_Item_BrambleVest", "TFT_Item_DragonsClaw", "TFT_Item_WarmogsArmor")),
    ("TFT18_Shen", "Shen", 2, ("Lunar", "Defender"), "UnitRole.TANK", False, True, 750.0, 45.0, 45.0, 50.0, 0.60, 1.0, 40.0, 90.0, ("TFT_Item_GargoyleStoneplate", "TFT_Item_BrambleVest", "TFT_Item_DragonsClaw")),
    ("TFT18_Teemo", "Teemo", 2, ("Sprykin", "Invoker"), "UnitRole.AP_CARRY", True, False, 550.0, 20.0, 20.0, 40.0, 0.70, 4.0, 0.0, 50.0, ("TFT_Item_BlueBuff", "TFT_Item_Morellonomicon", "TFT_Item_JeweledGauntlet")),
    ("TFT18_Warwick", "Warwick", 2, ("Blackthorn", "Ravager"), "UnitRole.AD_CARRY", True, False, 650.0, 35.0, 35.0, 55.0, 0.80, 1.0, 0.0, 50.0, ("TFT_Item_Bloodthirster", "TFT_Item_TitansResolve", "TFT_Item_SteraksGage")),
    ("TFT18_Yunara", "Yunara", 2, ("Sprykin", "Rapidfire"), "UnitRole.AD_CARRY", True, False, 550.0, 20.0, 20.0, 50.0, 0.75, 4.0, 0.0, 45.0, ("TFT_Item_GuinsoosRageblade", "TFT_Item_LastWhisper", "TFT_Item_InfinityEdge")),

    # 3-Cost (14 Champions)
    ("TFT18_Azir", "Azir", 3, ("Blackthorn", "Summoner"), "UnitRole.AP_CARRY", True, False, 650.0, 25.0, 25.0, 40.0, 0.75, 4.0, 20.0, 70.0, ("TFT_Item_GuinsoosRageblade", "TFT_Item_ArchangelsStaff", "TFT_Item_HextechGunblade")),
    ("TFT18_Cassiopeia", "Cassiopeia", 3, ("Coven", "Spellweaver"), "UnitRole.AP_CARRY", True, False, 650.0, 25.0, 25.0, 40.0, 0.75, 4.0, 0.0, 40.0, ("TFT_Item_BlueBuff", "TFT_Item_JeweledGauntlet", "TFT_Item_Morellonomicon")),
    ("TFT18_Diana", "Diana", 3, ("Lunar", "Ravager"), "UnitRole.AD_CARRY", True, False, 750.0, 35.0, 35.0, 60.0, 0.75, 1.0, 20.0, 70.0, ("TFT_Item_Bloodthirster", "TFT_Item_TitansResolve", "TFT_Item_HandOfJustice")),
    ("TFT18_Fiddlesticks", "Fiddlesticks", 3, ("Coven", "Juggernaut"), "UnitRole.TANK", False, True, 800.0, 45.0, 45.0, 50.0, 0.60, 1.0, 30.0, 90.0, ("TFT_Item_WarmogsArmor", "TFT_Item_DragonsClaw", "TFT_Item_Morellonomicon")),
    ("TFT18_Hecarim", "Hecarim", 3, ("Elderwood", "Vanguard"), "UnitRole.TANK", False, True, 800.0, 50.0, 50.0, 60.0, 0.65, 1.0, 40.0, 100.0, ("TFT_Item_BrambleVest", "TFT_Item_DragonsClaw", "TFT_Item_WarmogsArmor")),
    ("TFT18_KhaZix", "Kha'Zix", 3, ("Primal", "Executioner"), "UnitRole.AD_CARRY", True, False, 700.0, 30.0, 30.0, 65.0, 0.80, 1.0, 0.0, 40.0, ("TFT_Item_InfinityEdge", "TFT_Item_HandOfJustice", "TFT_Item_EdgeOfNight")),
    ("TFT18_KogMaw", "Kog'Maw", 3, ("Caustic", "Invoker", "Adaptor"), "UnitRole.AP_CARRY", True, False, 600.0, 25.0, 25.0, 45.0, 0.70, 4.0, 0.0, 40.0, ("TFT_Item_BlueBuff", "TFT_Item_JeweledGauntlet", "TFT_Item_HextechGunblade")),
    ("TFT18_Krug", "Krug", 3, ("Riftbeast", "Juggernaut"), "UnitRole.TANK", False, True, 850.0, 50.0, 50.0, 55.0, 0.60, 1.0, 40.0, 100.0, ("TFT_Item_WarmogsArmor", "TFT_Item_BrambleVest", "TFT_Item_DragonsClaw")),
    ("TFT18_MasterYi", "Master Yi", 3, ("Blossom", "Adaptor", "Rapidfire"), "UnitRole.AD_CARRY", True, False, 750.0, 35.0, 35.0, 65.0, 0.85, 1.0, 0.0, 50.0, ("TFT_Item_GuinsoosRageblade", "TFT_Item_Bloodthirster", "TFT_Item_TitansResolve")),
    ("TFT18_Rammus", "Rammus", 3, ("Elderwood", "Defender"), "UnitRole.TANK", False, True, 850.0, 55.0, 55.0, 50.0, 0.60, 1.0, 40.0, 100.0, ("TFT_Item_BrambleVest", "TFT_Item_DragonsClaw", "TFT_Item_SunfireCape")),
    ("TFT18_Raptor", "Raptor", 3, ("Riftbeast", "Rapidfire"), "UnitRole.AD_CARRY", True, False, 650.0, 25.0, 25.0, 60.0, 0.80, 4.0, 0.0, 40.0, ("TFT_Item_GuinsoosRageblade", "TFT_Item_LastWhisper", "TFT_Item_InfinityEdge")),
    ("TFT18_Rengar", "Rengar", 3, ("Primal", "Hunter"), "UnitRole.AD_CARRY", True, False, 700.0, 30.0, 30.0, 65.0, 0.80, 1.0, 0.0, 50.0, ("TFT_Item_InfinityEdge", "TFT_Item_Bloodthirster", "TFT_Item_LastWhisper")),
    ("TFT18_Tristana", "Tristana", 3, ("Sprykin", "Hunter"), "UnitRole.AD_CARRY", True, False, 650.0, 25.0, 25.0, 60.0, 0.75, 4.0, 0.0, 50.0, ("TFT_Item_InfinityEdge", "TFT_Item_LastWhisper", "TFT_Item_Deathblade")),
    ("TFT18_Vi", "Vi", 3, ("Blossom", "Brawler"), "UnitRole.TANK", False, True, 800.0, 45.0, 45.0, 60.0, 0.65, 1.0, 30.0, 80.0, ("TFT_Item_Bloodthirster", "TFT_Item_TitansResolve", "TFT_Item_WarmogsArmor")),

    # 4-Cost (14 Champions)
    ("TFT18_Ahri", "Ahri", 4, ("Blossom", "Spellweaver"), "UnitRole.AP_CARRY", True, False, 750.0, 30.0, 30.0, 45.0, 0.75, 4.0, 20.0, 70.0, ("TFT_Item_BlueBuff", "TFT_Item_JeweledGauntlet", "TFT_Item_HextechGunblade")),
    ("TFT18_Amumu", "Amumu", 4, ("Elderwood", "Juggernaut"), "UnitRole.TANK", False, True, 950.0, 55.0, 55.0, 55.0, 0.60, 1.0, 50.0, 120.0, ("TFT_Item_WarmogsArmor", "TFT_Item_DragonsClaw", "TFT_Item_BrambleVest")),
    ("TFT18_AncientSentinel", "Ancient Sentinel", 4, ("Riftbeast", "Vanguard"), "UnitRole.TANK", False, True, 950.0, 60.0, 60.0, 60.0, 0.60, 1.0, 50.0, 120.0, ("TFT_Item_GargoyleStoneplate", "TFT_Item_WarmogsArmor", "TFT_Item_DragonsClaw")),
    ("TFT18_Aphelios", "Aphelios", 4, ("Lunar", "Hunter"), "UnitRole.AD_CARRY", True, False, 750.0, 30.0, 30.0, 70.0, 0.80, 4.0, 0.0, 70.0, ("TFT_Item_InfinityEdge", "TFT_Item_LastWhisper", "TFT_Item_GuinsoosRageblade")),
    ("TFT18_Brambleback", "Brambleback", 4, ("Riftbeast", "Defender"), "UnitRole.TANK", False, True, 950.0, 60.0, 60.0, 55.0, 0.60, 1.0, 40.0, 110.0, ("TFT_Item_BrambleVest", "TFT_Item_SunfireCape", "TFT_Item_DragonsClaw")),
    ("TFT18_Ezreal", "Ezreal", 4, ("Fae", "Spellweaver"), "UnitRole.AP_CARRY", True, False, 750.0, 30.0, 30.0, 50.0, 0.80, 4.0, 0.0, 40.0, ("TFT_Item_BlueBuff", "TFT_Item_JeweledGauntlet", "TFT_Item_RabadonsDeathcap")),
    ("TFT18_Lillia", "Lillia", 4, ("Blossom", "Invoker"), "UnitRole.AP_CARRY", True, False, 750.0, 30.0, 30.0, 45.0, 0.75, 4.0, 20.0, 60.0, ("TFT_Item_SpearOfShojin", "TFT_Item_JeweledGauntlet", "TFT_Item_ArchangelsStaff")),
    ("TFT18_Malphite", "Malphite", 4, ("Blackthorn", "Defender"), "UnitRole.TANK", False, True, 950.0, 60.0, 60.0, 60.0, 0.60, 1.0, 50.0, 120.0, ("TFT_Item_GargoyleStoneplate", "TFT_Item_WarmogsArmor", "TFT_Item_DragonsClaw")),
    ("TFT18_Morgana", "Morgana", 4, ("Coven", "Executioner"), "UnitRole.AP_CARRY", True, False, 750.0, 30.0, 30.0, 50.0, 0.75, 4.0, 30.0, 80.0, ("TFT_Item_SpearOfShojin", "TFT_Item_Morellonomicon", "TFT_Item_JeweledGauntlet")),
    ("TFT18_Nidalee", "Nidalee", 4, ("Primal", "Adaptor"), "UnitRole.AD_CARRY", True, False, 800.0, 35.0, 35.0, 70.0, 0.85, 4.0, 0.0, 50.0, ("TFT_Item_GuinsoosRageblade", "TFT_Item_Bloodthirster", "TFT_Item_TitansResolve")),
    ("TFT18_Sett", "Sett", 4, ("Inferno", "Brawler"), "UnitRole.TANK", False, True, 950.0, 55.0, 55.0, 65.0, 0.65, 1.0, 40.0, 100.0, ("TFT_Item_Bloodthirster", "TFT_Item_TitansResolve", "TFT_Item_WarmogsArmor")),
    ("TFT18_Sivir", "Sivir", 4, ("Solar", "Rapidfire"), "UnitRole.AD_CARRY", True, False, 750.0, 30.0, 30.0, 70.0, 0.80, 4.0, 0.0, 60.0, ("TFT_Item_InfinityEdge", "TFT_Item_LastWhisper", "TFT_Item_GuinsoosRageblade")),
    ("TFT18_Soraka", "Soraka", 4, ("Solar", "Invoker"), "UnitRole.UTILITY", False, False, 750.0, 30.0, 30.0, 45.0, 0.75, 4.0, 30.0, 80.0, ("TFT_Item_SpearOfShojin", "TFT_Item_ArchangelsStaff", "TFT_Item_HextechGunblade")),
    ("TFT18_Zyra", "Zyra", 4, ("Thornmaiden", "Flora Fatalis", "Summoner"), "UnitRole.AP_CARRY", True, False, 750.0, 30.0, 30.0, 50.0, 0.75, 4.0, 20.0, 70.0, ("TFT_Item_SpearOfShojin", "TFT_Item_JeweledGauntlet", "TFT_Item_Morellonomicon")),

    # 5-Cost Legendaries (10 Champions)
    ("TFT18_Alune", "Alune", 5, ("Attuned", "Lunar", "Spellweaver"), "UnitRole.AP_CARRY", True, False, 850.0, 40.0, 40.0, 55.0, 0.80, 4.0, 30.0, 90.0, ("TFT_Item_SpearOfShojin", "TFT_Item_JeweledGauntlet", "TFT_Item_RabadonsDeathcap")),
    ("TFT18_Ashe", "Ashe", 5, ("Fae", "Hunter"), "UnitRole.AD_CARRY", True, False, 850.0, 35.0, 35.0, 75.0, 0.85, 4.0, 0.0, 60.0, ("TFT_Item_InfinityEdge", "TFT_Item_LastWhisper", "TFT_Item_GuinsoosRageblade")),
    ("TFT18_Draven", "Draven", 5, ("Bounty Seeker", "Inferno", "Executioner"), "UnitRole.AD_CARRY", True, False, 850.0, 40.0, 40.0, 80.0, 0.85, 4.0, 0.0, 50.0, ("TFT_Item_InfinityEdge", "TFT_Item_LastWhisper", "TFT_Item_Bloodthirster")),
    ("TFT18_ElderDragon", "The Elder Dragon", 5, ("Apex Predator", "Riftbeast"), "UnitRole.TANK", False, True, 1800.0, 80.0, 80.0, 90.0, 0.70, 2.0, 60.0, 150.0, ("TFT_Item_WarmogsArmor", "TFT_Item_DragonsClaw", "TFT_Item_BrambleVest")),
    ("TFT18_Gnar", "Gnar", 5, ("Primal", "Brawler", "Monolith"), "UnitRole.TANK", False, True, 1100.0, 60.0, 60.0, 75.0, 0.75, 1.0, 50.0, 110.0, ("TFT_Item_Bloodthirster", "TFT_Item_TitansResolve", "TFT_Item_SteraksGage")),
    ("TFT18_Ivern", "Ivern", 5, ("Greenfather", "Elderwood", "Invoker"), "UnitRole.UTILITY", False, False, 900.0, 45.0, 45.0, 55.0, 0.75, 4.0, 40.0, 100.0, ("TFT_Item_SpearOfShojin", "TFT_Item_StatikkShiv", "TFT_Item_Redemption")),
    ("TFT18_Kennen", "Kennen", 5, ("Sprykin", "Defender", "Ravager"), "UnitRole.AP_CARRY", True, False, 900.0, 45.0, 45.0, 60.0, 0.80, 2.0, 40.0, 100.0, ("TFT_Item_JeweledGauntlet", "TFT_Item_IonicSpark", "TFT_Item_Morellonomicon")),
    ("TFT18_Lux", "Lux", 5, ("Avatar", "Spellweaver"), "UnitRole.AP_CARRY", True, False, 850.0, 35.0, 35.0, 55.0, 0.80, 4.0, 20.0, 70.0, ("TFT_Item_BlueBuff", "TFT_Item_JeweledGauntlet", "TFT_Item_RabadonsDeathcap")),
    ("TFT18_Maokai", "Maokai", 5, ("Old Growth", "Elderwood", "Vanguard"), "UnitRole.TANK", False, True, 1100.0, 65.0, 65.0, 65.0, 0.65, 1.0, 50.0, 120.0, ("TFT_Item_WarmogsArmor", "TFT_Item_DragonsClaw", "TFT_Item_BrambleVest")),
    ("TFT18_Taric", "Taric", 5, ("Emerald Aspect", "Solar", "Defender"), "UnitRole.TANK", False, True, 1100.0, 70.0, 70.0, 65.0, 0.65, 1.0, 50.0, 120.0, ("TFT_Item_GargoyleStoneplate", "TFT_Item_DragonsClaw", "TFT_Item_WarmogsArmor")),
]


def generate_official_set18():
    raw_data = json.loads(Path("data/cdragon_set18.json").read_text(encoding="utf-8"))
    
    # 1. Process Traits
    traits_raw = raw_data.get("traits", [])
    valid_traits = {}
    trait_members = {}

    for api_name, name, cost, traits, role, is_carry, is_tank, hp, armor, mr, ad, aspeed, range_hex, mana, max_mana, rec_items in SET18_CHAMPIONS_DATA:
        for t in traits:
            trait_members.setdefault(t, []).append(api_name)

    for t in traits_raw:
        api_name = t.get("apiName", "")
        name = t.get("name", "")
        desc = t.get("desc", "").replace("<br>", " ").replace("</br>", " ").replace("<br/>", " ").replace("\n", " ").strip()
        effects = t.get("effects", [])
        thresholds = [eff.get("minUnits") for eff in effects if eff.get("minUnits") is not None]
        if not thresholds:
            thresholds = [1]
            
        clean_thresh = tuple(sorted(list(set(thresholds))))
        members = tuple(trait_members.get(name, []))

        valid_traits[name] = {
            "api_name": api_name,
            "name": name,
            "thresholds": clean_thresh,
            "desc": desc if desc else f"The {name} trait grants powerful bonuses when active.",
            "members": members,
        }

    # Load items from set 17 profile (since core items are identical in TFT)
    from tft_ai_player.simulation.sets.set17 import SET17_ITEMS_CATALOG, SET17_ITEM_RECIPES

    # 2. Build Python Code
    lines = [
        '"""Set 18 \'Enchanted Wilds\' Complete Profile: Champions, Traits, Items, Recipes, Odds, and Leveling Curves.',
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
        f"# SET 18 TRAIT DEFINITIONS & SYNERGY BONUSES ({len(valid_traits)} TRAITS)",
        "# =============================================================================",
        "",
        "@dataclass(frozen=True, slots=True)",
        "class Set18TraitInfo:",
        '    """Detailed trait synergy definition with threshold bonuses."""',
        "",
        "    trait_id: str",
        "    name: str",
        "    thresholds: tuple[int, ...]",
        "    description: str",
        "    champions: tuple[str, ...]",
        "",
        "",
        "SET18_TRAIT_CATALOG: dict[str, Set18TraitInfo] = {",
    ]

    for tname, tinfo in sorted(valid_traits.items()):
        desc_escaped = tinfo["desc"].replace('"', '\\"')
        lines.append(f'    "{tname}": Set18TraitInfo(')
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
        f"# SET 18 CHAMPION CATALOG ({len(SET18_CHAMPIONS_DATA)} CHAMPIONS)",
        "# =============================================================================",
        "",
        "@dataclass(frozen=True, slots=True)",
        "class Set18ChampionInfo:",
        '    """Complete specification of a Set 18 champion."""',
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
        "SET18_CHAMPION_CATALOG: tuple[Set18ChampionInfo, ...] = (",
    ])

    current_cost = None
    for api_name, name, cost, traits, role, is_carry, is_tank, hp, armor, mr, ad, aspeed, range_hex, mana, max_mana, rec_items in SET18_CHAMPIONS_DATA:
        if cost != current_cost:
            current_cost = cost
            count_for_cost = sum(1 for x in SET18_CHAMPIONS_DATA if x[2] == current_cost)
            lines.append(f"    # --- {current_cost}-Cost ({count_for_cost} Champions) ---")

        escaped_name = name.replace('"', '\\"')
        lines.append(f'    Set18ChampionInfo("{api_name}", "{escaped_name}", {cost}, {traits}, {role}, {is_carry}, {is_tank}, {hp}, {armor}, {mr}, {ad}, {aspeed:.2f}, {range_hex}, {mana}, {max_mana}, {rec_items}),')

    lines.extend([
        ")",
        "",
        "",
        "# =============================================================================",
        f"# SET 18 ITEMS & RECIPES CATALOG ({len(SET17_ITEMS_CATALOG)} ITEMS, {len(SET17_ITEM_RECIPES)} RECIPES)",
        "# =============================================================================",
        "",
        "@dataclass(frozen=True, slots=True)",
        "class Set18ItemInfo:",
        '    """Official Item definition with recipe composition."""',
        "",
        "    item_id: str",
        "    name: str",
        "    is_component: bool",
        "    recipe: tuple[str, ...]",
        "    description: str",
        "",
        "",
        "SET18_ITEMS_CATALOG: dict[str, Set18ItemInfo] = {",
    ])

    for i_id, i_info in sorted(SET17_ITEMS_CATALOG.items()):
        desc_escaped = i_info.description.replace('"', '\\"')
        escaped_name = i_info.name.replace('"', '\\"')
        lines.append(f'    "{i_id}": Set18ItemInfo(')
        lines.append(f'        item_id="{i_id}",')
        lines.append(f'        name="{escaped_name}",')
        lines.append(f'        is_component={i_info.is_component},')
        lines.append(f'        recipe={i_info.recipe},')
        lines.append(f'        description="{desc_escaped}",')
        lines.append("    ),")

    lines.extend([
        "}",
        "",
        "SET18_ITEM_RECIPES: dict[frozenset[str], str] = {",
    ])

    for key, res_id in sorted(SET17_ITEM_RECIPES.items(), key=lambda x: sorted(list(x[0]))):
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
        "class Set18Profile:",
        '    """Top-level container for Set 18 profile specifications."""',
        "",
        '    set_name: str = "TFTSet18"',
        f"    total_champions: int = {len(SET18_CHAMPIONS_DATA)}",
        f"    total_traits: int = {len(valid_traits)}",
        f"    total_items: int = {len(SET17_ITEMS_CATALOG)}",
        f"    total_recipes: int = {len(SET17_ITEM_RECIPES)}",
        "    champions: tuple[Set18ChampionInfo, ...] = SET18_CHAMPION_CATALOG",
        "    traits: dict[str, Set18TraitInfo] = field(default_factory=lambda: dict(SET18_TRAIT_CATALOG))",
        "    items: dict[str, Set18ItemInfo] = field(default_factory=lambda: dict(SET18_ITEMS_CATALOG))",
        "",
        "    def to_dict(self) -> dict[str, Any]:",
        '        """Convert profile to serializable dictionary."""',
        "        return {",
        '            "set_name": self.set_name,',
        '            "stats": {',
        '                "total_champions": len(self.champions),',
        '                "total_traits": len(self.traits),',
        '                "total_items": len(self.items),',
        '                "total_recipes": len(SET18_ITEM_RECIPES),',
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
        '                    "result_name": self.items.get(res_id, Set18ItemInfo(res_id, res_id, False, (), "")).name,',
        "                }",
        "                for comp, res_id in SET18_ITEM_RECIPES.items()",
        "            ],",
        '            "shop_odds": {str(k): list(v) for k, v in STANDARD_SHOP_ODDS.items()},',
        '            "level_exp": {str(k): v for k, v in STANDARD_LEVEL_EXP.items()},',
        '            "pool_sizes": {str(k): v for k, v in STANDARD_POOL_SIZES.items()},',
        "        }",
        "",
        "",
        "def get_set18_data() -> SetData:",
        '    """Build a complete SetData instance for Set 18 simulations."""',
        "    champions: dict[str, ChampionDef] = {",
        "        c.champion_id: ChampionDef(",
        "            champion_id=c.champion_id,",
        "            name=c.name,",
        "            cost=c.cost,",
        "            traits=c.traits,",
        "            role=c.role,",
        "        )",
        "        for c in SET18_CHAMPION_CATALOG",
        "    }",
        "",
        "    traits: dict[str, TraitDef] = {",
        "        t.trait_id: TraitDef(",
        "            trait_id=t.trait_id,",
        "            name=t.name,",
        "            thresholds=t.thresholds,",
        "        )",
        "        for t in SET18_TRAIT_CATALOG.values()",
        "    }",
        "",
        "    items: dict[str, ItemDef] = {",
        "        it.item_id: ItemDef(",
        "            item_id=it.item_id,",
        "            name=it.name,",
        "            is_component=it.is_component,",
        "            recipe=it.recipe if len(it.recipe) == 2 else None,",
        "        )",
        "        for it in SET18_ITEMS_CATALOG.values()",
        "    }",
        "",
        "    recipes: dict[frozenset[str], str] = dict(SET18_ITEM_RECIPES)",
        "",
        "    return SetData(",
        '        set_name="TFTSet18",',
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
        'def export_set18_profile_json(output_path: str | Path = "data/set18_profile.json") -> Path:',
        '    """Export complete Set 18 profile JSON to file."""',
        "    profile = Set18Profile()",
        "    data = profile.to_dict()",
        "    out = Path(output_path).resolve()",
        "    out.parent.mkdir(parents=True, exist_ok=True)",
        '    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")',
        "    return out",
        "",
    ])

    out_file = Path("src/tft_ai_player/simulation/sets/set18.py")
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated complete official {out_file} ({len(lines)} lines)")


if __name__ == "__main__":
    generate_official_set18()
