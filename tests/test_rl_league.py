"""Unit tests for Reinforcement Learning (PPO) and AlphaStar Multi-Agent League Pipeline."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from tft_ai_player.rl.agent_policy import RLBot
from tft_ai_player.rl.algorithms.ppo import MaskablePPO, RolloutBuffer
from tft_ai_player.rl.evaluation.evaluator import BenchmarkBotEvaluator, TournamentEvaluator
from tft_ai_player.rl.evaluation.report import generate_league_markdown_report, print_league_terminal_summary
from tft_ai_player.rl.league.checkpoints import CheckpointManager
from tft_ai_player.rl.league.elo import MultilateralEloSystem
from tft_ai_player.rl.league.league_manager import LeagueManager
from tft_ai_player.rl.logger import check_collapse_warnings
from tft_ai_player.rl.models.distributions import MaskedCategorical, numpy_masked_sample, numpy_masked_softmax
from tft_ai_player.rl.models.networks import ShopBenchFeatureExtractor, TFTActorCritic
from tft_ai_player.rl.train import LeagueTrainer
from tft_ai_player.rl.types import AgentProfile, AgentRole, EloRating, MatchResult
from tft_ai_player.simulation.actions import TOTAL_DISCRETE_ACTIONS, get_action_mask
from tft_ai_player.simulation.bots import BotAlphaFast8, BotBetaHyperroll, BotGammaGreedy
from tft_ai_player.simulation.config import get_default_set17_data
from tft_ai_player.simulation.gym_env import TFTEnv, TFTStateEncoder


def test_numpy_masked_softmax() -> None:
    """Verify numpy masked softmax assigns exact zero probability to invalid actions."""
    logits = np.array([2.0, 1.0, 5.0, 0.5])
    mask = np.array([True, False, True, False])

    probs = numpy_masked_softmax(logits, mask)
    assert probs[1] == 0.0
    assert probs[3] == 0.0
    assert np.isclose(np.sum(probs), 1.0)
    assert probs[2] > probs[0]

    action, log_prob = numpy_masked_sample(logits, mask, deterministic=True)
    assert action == 2
    assert np.isclose(log_prob, np.log(probs[2]))


def test_pytorch_masked_categorical() -> None:
    """Verify PyTorch MaskedCategorical never samples invalid actions."""
    logits = torch.tensor([[1.0, 10.0, -2.0, 4.0]])
    mask = torch.tensor([[True, False, True, True]])

    dist = MaskedCategorical(logits=logits, mask=mask)
    action, log_prob = dist.sample_action(deterministic=True)

    # Action 1 had highest logit (10.0) but was masked False -> should pick action 3 (4.0)
    assert action.item() == 3
    probs = dist.probs
    assert probs[0, 1].item() == 0.0
    assert torch.isclose(probs.sum(), torch.tensor(1.0))


def test_actor_critic_forward() -> None:
    """Verify TFTActorCritic accepts 704D state observations and outputs 111 action logits and 1 value."""
    model = TFTActorCritic(obs_dim=704, action_dim=TOTAL_DISCRETE_ACTIONS, hidden_dim=512)

    batch_size = 4
    dummy_obs = torch.randn(batch_size, 704)
    dummy_mask = torch.ones(batch_size, TOTAL_DISCRETE_ACTIONS, dtype=torch.bool)

    logits, values = model(dummy_obs)
    assert logits.shape == (batch_size, 111)
    assert values.shape == (batch_size, 1)

    actions, log_probs, vals = model.get_action(dummy_obs, dummy_mask)
    assert actions.shape == (batch_size,)
    assert log_probs.shape == (batch_size,)
    assert vals.shape == (batch_size,)

    eval_log_probs, entropy, eval_vals = model.evaluate_actions(dummy_obs, actions, dummy_mask)
    assert eval_log_probs.shape == (batch_size,)
    assert entropy.shape == (batch_size,)
    assert eval_vals.shape == (batch_size,)


def test_shop_bench_feature_extractor() -> None:
    """Verify shop (160D->64D) and bench (64D->64D) projection module."""
    extractor = ShopBenchFeatureExtractor(shop_in=160, bench_in=64, out_dim=64)
    shop_in = torch.randn(2, 160)
    bench_in = torch.randn(2, 64)
    shop_feat, bench_feat = extractor(shop_in, bench_in)
    assert shop_feat.shape == (2, 64)
    assert bench_feat.shape == (2, 64)


def test_rollout_buffer_and_ppo_update() -> None:
    """Verify RolloutBuffer stores transitions and MaskablePPO computes clipped surrogate gradients."""
    model = TFTActorCritic(obs_dim=704, action_dim=TOTAL_DISCRETE_ACTIONS, hidden_dim=256)
    ppo = MaskablePPO(model=model, lr=1e-3, clip_ratio=0.2)
    buffer = RolloutBuffer(buffer_size=64, obs_dim=704, action_dim=TOTAL_DISCRETE_ACTIONS)

    for _ in range(64):
        obs = np.random.randn(704).astype(np.float32)
        mask = np.ones(TOTAL_DISCRETE_ACTIONS, dtype=bool)
        action = 0
        reward = 1.0
        value = 0.5
        log_prob = -1.0
        done = False
        buffer.add(obs, action, mask, reward, value, log_prob, done)

    buffer.compute_returns_and_advantages(last_value=0.5, done=True)
    metrics = ppo.update(buffer, num_epochs=2, batch_size=16)

    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "policy_entropy" in metrics
    assert "explained_variance" in metrics
    assert not np.isnan(metrics["policy_loss"])


def test_multilateral_elo_system() -> None:
    """Verify 8-player multilateral Elo rating updates and leaderboard generation."""
    elo_sys = MultilateralEloSystem(k_factor=32.0, base_rating=1200.0)

    profiles = {
        f"p{i}": AgentProfile(agent_id=f"p{i}", name=f"Player {i}", role=AgentRole.BASELINE)
        for i in range(1, 9)
    }

    # Placements: p1 is 1st, p8 is 8th
    placements = {f"p{i}": i for i in range(1, 9)}
    deltas = elo_sys.update_lobby_ratings(placements, profiles)

    # 1st place should gain rating, 8th place should lose rating
    assert deltas["p1"] > 0
    assert deltas["p8"] < 0
    assert profiles["p1"].elo.rating > 1200.0
    assert profiles["p8"].elo.rating < 1200.0

    leaderboard = elo_sys.generate_leaderboard(profiles)
    assert len(leaderboard) == 8
    assert leaderboard[0]["agent_id"] == "p1"


def test_league_manager_and_pfsp() -> None:
    """Verify AlphaStar LeagueManager initializes 15 Exploiters and PFSP sampling."""
    league = LeagueManager(checkpoint_dir="checkpoints/test_league", num_exploiters=15)

    assert "main_agent" in league.profiles
    assert "bot_alpha" in league.profiles
    assert "bot_beta" in league.profiles
    assert "bot_gamma" in league.profiles
    assert "exploiter_z01" in league.profiles
    assert "exploiter_z15" in league.profiles

    opponents = league.sample_pfsp_opponents(focal_agent_id="main_agent", num_opponents=7)
    assert len(opponents) == 7

    # Test historical snapshot archival
    snap_id = league.archive_snapshot("main_agent", generation=50)
    assert snap_id in league.profiles
    assert league.profiles[snap_id].role == AgentRole.HISTORICAL

    leaderboard = league.get_leaderboard()
    assert len(leaderboard) >= 19


def test_benchmark_bot_evaluator() -> None:
    """Verify BenchmarkBotEvaluator evaluates Main Agent against Bot Alpha, Beta, Gamma."""
    set_data = get_default_set17_data()
    model = TFTActorCritic(obs_dim=704, action_dim=TOTAL_DISCRETE_ACTIONS, hidden_dim=128)
    evaluator = BenchmarkBotEvaluator(set_data=set_data)

    metrics = evaluator.evaluate_main_agent(model, num_matches=2, base_seed=42)
    assert "eval_avg_placement" in metrics
    assert "eval_win_rate" in metrics
    assert "bot_alpha_win_rate" in metrics
    assert "bot_beta_win_rate" in metrics
    assert "bot_gamma_win_rate" in metrics
    assert 1.0 <= metrics["eval_avg_placement"] <= 8.0


def test_tft_state_encoder_and_gym_env() -> None:
    """Verify TFTStateEncoder constructs 704D state and TFTEnv executes steps with multi-objective rewards."""
    set_data = get_default_set17_data()
    env = TFTEnv(set_data=set_data, alpha=0.0, beta=0.3)

    obs, info = env.reset(seed=42)
    assert obs.shape == (704,)
    assert "action_mask" in info
    assert info["action_mask"].shape == (TOTAL_DISCRETE_ACTIONS,)
    assert info["action_mask"][0] == True  # PASS_ROUND is legal

    # Execute action 0 (PASS_ROUND)
    next_obs, reward, terminated, truncated, next_info = env.step(0)
    assert next_obs.shape == (704,)
    assert isinstance(reward, float)


def test_check_collapse_warnings() -> None:
    """Verify collapse detection identifies low entropy, negative explained variance, and high KL."""
    healthy_metrics = {
        "policy_entropy": 1.5,
        "explained_variance": 0.75,
        "approx_kl": 0.01,
        "action_mask_rejection_rate": 0.0,
        "macro_z_cosine": 0.80,
        "micro_world_model_cosine": 0.85,
    }
    assert len(check_collapse_warnings(healthy_metrics)) == 0

    collapsed_metrics = {
        "policy_entropy": 0.05,
        "explained_variance": -0.2,
        "approx_kl": 0.08,
        "action_mask_rejection_rate": 0.05,
        "macro_z_cosine": 0.20,
        "micro_world_model_cosine": 0.30,
    }
    warnings = check_collapse_warnings(collapsed_metrics)
    assert len(warnings) == 6
