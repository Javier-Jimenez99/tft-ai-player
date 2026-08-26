"""Set-Agnostic TFT Simulation and Gymnasium Environment Package."""

from __future__ import annotations

import gymnasium as gym

from tft_ai_player.simulation.actions import (
    TOTAL_DISCRETE_ACTIONS,
    ActionType,
    execute_action,
    get_action_mask,
)
from tft_ai_player.simulation.bots import (
    BaseBot,
    GreedyBankerBot,
    RandomBot,
    StandardTempoBot,
)
from tft_ai_player.simulation.combat import (
    CombatResolver,
    CombatResult,
    HeuristicCombatResolver,
    MLCombatResolver,
)
from tft_ai_player.simulation.config import (
    ChampionDef,
    ItemDef,
    SetData,
    TraitDef,
    UnitRole,
    get_default_set17_data,
)
from tft_ai_player.simulation.game import TFTGame
from tft_ai_player.simulation.gym_env import TFTEnv
from tft_ai_player.simulation.matchmaking import MatchmakingEngine, Matchup
from tft_ai_player.simulation.models import (
    ChampionInstance,
    ChampionPool,
    ItemInstance,
    Player,
    Shop,
)
from tft_ai_player.simulation.observations import ObservationEncoder
from tft_ai_player.simulation.sets import (
    AVAILABLE_SETS,
    Set17Profile,
    Set18Profile,
    get_set_data,
    get_set_profile,
    get_set17_data,
    get_set18_data,
)
from tft_ai_player.simulation.stage_manager import (
    RoundInfo,
    RoundType,
    StageManager,
)
from tft_ai_player.simulation.visualizer import (
    GameRecorder,
    generate_visual_html,
)

# Register with Gymnasium env registry
gym.register(
    id="TFT-v0",
    entry_point="tft_ai_player.simulation.gym_env:TFTEnv",
)

__all__ = [
    "AVAILABLE_SETS",
    "ActionType",
    "BaseBot",
    "ChampionDef",
    "ChampionInstance",
    "ChampionPool",
    "CombatResolver",
    "CombatResult",
    "GameRecorder",
    "GreedyBankerBot",
    "HeuristicCombatResolver",
    "ItemDef",
    "ItemInstance",
    "MLCombatResolver",
    "MatchmakingEngine",
    "Matchup",
    "ObservationEncoder",
    "Player",
    "RandomBot",
    "RoundInfo",
    "RoundType",
    "Set17Profile",
    "Set18Profile",
    "SetData",
    "Shop",
    "StageManager",
    "StandardTempoBot",
    "TFTEnv",
    "TFTGame",
    "TOTAL_DISCRETE_ACTIONS",
    "TraitDef",
    "UnitRole",
    "execute_action",
    "generate_visual_html",
    "get_action_mask",
    "get_default_set17_data",
    "get_set_data",
    "get_set_profile",
    "get_set17_data",
    "get_set18_data",
]
