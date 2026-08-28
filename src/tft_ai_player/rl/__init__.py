"""Reinforcement Learning and League Self-Play module for Teamfight Tactics."""

from __future__ import annotations

from tft_ai_player.rl.agent_policy import RLBot
from tft_ai_player.rl.algorithms.ppo import MaskablePPO, RolloutBuffer
from tft_ai_player.rl.evaluation.evaluator import TournamentEvaluator
from tft_ai_player.rl.evaluation.report import generate_league_markdown_report, print_league_terminal_summary
from tft_ai_player.rl.league.checkpoints import CheckpointManager
from tft_ai_player.rl.league.elo import MultilateralEloSystem
from tft_ai_player.rl.league.league_manager import LeagueManager
from tft_ai_player.rl.models.distributions import MaskedCategorical
from tft_ai_player.rl.models.networks import TFTActorCritic
from tft_ai_player.rl.train import LeagueTrainer
from tft_ai_player.rl.types import AgentProfile, AgentRole, EloRating, MatchResult

__all__ = [
    "AgentProfile",
    "AgentRole",
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
    "TFTActorCritic",
    "TournamentEvaluator",
    "generate_league_markdown_report",
    "print_league_terminal_summary",
]
