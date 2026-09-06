"""Reinforcement Learning and AlphaStar League Pipeline for Teamfight Tactics."""

from __future__ import annotations

from tft_ai_player.rl.agent_policy import RLBot
from tft_ai_player.rl.algorithms.ppo import MaskablePPO, RolloutBuffer
from tft_ai_player.rl.evaluation.evaluator import BenchmarkBotEvaluator, TournamentEvaluator
from tft_ai_player.rl.evaluation.report import generate_league_markdown_report, print_league_terminal_summary
from tft_ai_player.rl.league.checkpoints import CheckpointManager
from tft_ai_player.rl.league.elo import MultilateralEloSystem
from tft_ai_player.rl.league.league_manager import LeagueManager
from tft_ai_player.rl.logger import WandBLogger, check_collapse_warnings
from tft_ai_player.rl.models.distributions import MaskedCategorical, numpy_masked_sample, numpy_masked_softmax
from tft_ai_player.rl.models.networks import ShopBenchFeatureExtractor, TFTActorCritic
from tft_ai_player.rl.planner import ShopBeamSearchPlanner
from tft_ai_player.rl.train import LeagueTrainer
from tft_ai_player.rl.types import AgentProfile, AgentRole, EloRating, MatchResult

__all__ = [
    "AgentProfile",
    "AgentRole",
    "BenchmarkBotEvaluator",
    "CheckpointManager",
    "EloRating",
    "LeagueManager",
    "LeagueTrainer",
    "MaskablePPO",
    "MaskedCategorical",
    "MatchResult",
    "MultilateralEloSystem",
    "RLBot",
    "RolloutBuffer",
    "ShopBeamSearchPlanner",
    "ShopBenchFeatureExtractor",
    "TFTActorCritic",
    "TournamentEvaluator",
    "WandBLogger",
    "check_collapse_warnings",
    "generate_league_markdown_report",
    "numpy_masked_sample",
    "numpy_masked_softmax",
    "print_league_terminal_summary",
]
