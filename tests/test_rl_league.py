"""Unit and integration tests for TFT Reinforcement Learning, AlphaStar League, and Elo systems."""

import numpy as np
import pytest
import torch

from tft_ai_player.rl.agent_policy import RLBot
from tft_ai_player.rl.algorithms.ppo import MaskablePPO, RolloutBuffer
from tft_ai_player.rl.evaluation.evaluator import TournamentEvaluator
from tft_ai_player.rl.evaluation.report import generate_league_markdown_report
from tft_ai_player.rl.league.checkpoints import CheckpointManager
from tft_ai_player.rl.league.elo import MultilateralEloSystem
from tft_ai_player.rl.league.league_manager import LeagueManager
from tft_ai_player.rl.models.distributions import MaskedCategorical, numpy_masked_sample, numpy_masked_softmax
from tft_ai_player.rl.models.networks import TFTActorCritic
from tft_ai_player.rl.train import LeagueTrainer
from tft_ai_player.rl.types import AgentProfile, AgentRole, EloRating
from tft_ai_player.simulation.config import get_default_set17_data
from tft_ai_player.simulation.game import TFTGame
from tft_ai_player.simulation.gym_env import TFTEnv


def test_numpy_masked_softmax() -> None:
    """Verify numpy masked softmax assigns zero probability to invalid actions."""
    logits = np.array([2.0, 1.0, 5.0, 0.5])
    mask = np.array([True, False, True, False])

    probs = numpy_masked_softmax(logits, mask)
    assert probs[1] == 0.0
    assert probs[3] == 0.0
    assert np.isclose(np.sum(probs), 1.0)
    assert probs[2] > probs[0]

    # Sample action
    action, log_prob = numpy_masked_sample(logits, mask, deterministic=True)
    assert action == 2
    assert log_prob == np.log(probs[2])


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
    """Verify TFTActorCritic extracts multi-modal embeddings and outputs valid dimensions."""
    set_data = get_default_set17_data()
    model = TFTActorCritic(
        num_champs=len(set_data.champions) + 1,
        num_items=len(set_data.items) + 1,
        champ_embed_dim=16,
        item_embed_dim=8,
        hidden_dim=128,
    )

    batch_size = 2
    dummy_obs = {
        "player_stats": torch.randn(batch_size, 9),
        "board": torch.zeros(batch_size, 28, 5),
        "bench": torch.zeros(batch_size, 9, 5),
        "item_bench": torch.zeros(batch_size, 10),
        "shop": torch.zeros(batch_size, 5),
        "opponents": torch.randn(batch_size, 7, 8),
        "opponents_boards": torch.zeros(batch_size, 7, 28, 5),
        "opponents_benches": torch.zeros(batch_size, 7, 9, 5),
    }
    dummy_mask = torch.ones(batch_size, 1721, dtype=torch.bool)

    logits, values = model(dummy_obs)
    assert logits.shape == (batch_size, 1721)
    assert values.shape == (batch_size, 1)

    actions, log_probs, vals = model.get_action(dummy_obs, dummy_mask)
    assert actions.shape == (batch_size,)
    assert log_probs.shape == (batch_size,)
    assert vals.shape == (batch_size,)


def test_multilateral_elo_system() -> None:
    """Verify 8-player multilateral Elo rating updates and leaderboard metrics."""
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
    assert profiles["p1"].elo.wins == 1
    assert profiles["p1"].elo.top4s == 1
    assert profiles["p8"].elo.wins == 0
    assert profiles["p8"].elo.top4s == 0

    leaderboard = elo_sys.generate_leaderboard(list(profiles.values()))
    assert leaderboard[0]["agent_id"] == "p1"
    assert leaderboard[-1]["agent_id"] == "p8"


def test_league_manager_and_pfsp(tmp_path: pytest.TempPathFactory) -> None:
    """Verify LeagueManager registration, PFSP opponent sampling, and Hall of Fame snapshotting."""
    league = LeagueManager(checkpoint_dir=str(tmp_path))

    # Should have default baseline bots registered
    assert "bot_standard_tempo" in league.profiles
    assert "bot_greedy_banker" in league.profiles
    assert "bot_random" in league.profiles

    # Register main agent
    main_prof = league.register_agent("main_v1", "Main Agent", AgentRole.MAIN)
    assert main_prof.agent_id == "main_v1"

    # Sample 7 opponents via PFSP
    opps = league.sample_pfsp_opponents("main_v1", num_opponents=7)
    assert len(opps) == 7

    # Create Hall of Fame snapshot
    hof = league.create_hall_of_fame_snapshot("main_v1", generation=10)
    assert hof.role == AgentRole.HALL_OF_FAME
    assert "hof_main_v1_gen10" in league.profiles


def test_paired_seed_benchmark(tmp_path: pytest.TempPathFactory) -> None:
    """Verify paired seed Common Random Numbers benchmark is reproducible and mitigates luck."""
    set_data = get_default_set17_data()
    league = LeagueManager(checkpoint_dir=str(tmp_path))
    evaluator = TournamentEvaluator(league, set_data=set_data)

    res1 = evaluator.run_paired_benchmark(
        candidate_agent_id="bot_standard_tempo",
        opponent_agent_ids=["bot_greedy_banker", "bot_random"],
        num_seeds=2,
        base_seed=123,
    )

    res2 = evaluator.run_paired_benchmark(
        candidate_agent_id="bot_standard_tempo",
        opponent_agent_ids=["bot_greedy_banker", "bot_random"],
        num_seeds=2,
        base_seed=123,
    )

    # Identical seeds must produce identical match outcomes
    assert res1["placements"] == res2["placements"]
    assert res1["avg_placement"] == res2["avg_placement"]


def test_rlbot_adapter_turn() -> None:
    """Verify RLBot can execute valid turns inside TFTGame simulation."""
    set_data = get_default_set17_data()
    model = TFTActorCritic(
        num_champs=len(set_data.champions) + 1,
        num_items=len(set_data.items) + 1,
        champ_embed_dim=16,
        item_embed_dim=8,
        hidden_dim=128,
    )

    bot = RLBot(model=model, set_data=set_data, deterministic=True, max_micro_actions=5)
    game = TFTGame(set_data=set_data, seed=42)

    p0 = game.players[0]
    initial_gold = p0.gold
    bot.take_turn(
        player=p0,
        pool=game.pool,
        set_data=set_data,
        stage=2,
        round_in_stage=1,
        all_players=game.players,
        stage_manager=game.stage_manager,
    )

    # Player state is preserved and valid
    assert p0.alive is True
    assert p0.gold >= 0


def test_league_trainer_mini_iteration(tmp_path: pytest.TempPathFactory) -> None:
    """Verify LeagueTrainer collects rollouts and executes PPO gradient update."""
    set_data = get_default_set17_data()
    trainer = LeagueTrainer(
        set_data=set_data,
        lr=1e-3,
        buffer_size=64,
        batch_size=16,
        num_epochs=1,
        checkpoint_dir=str(tmp_path),
    )

    metrics = trainer.train_iteration(generation=1, rollout_steps=32, eval_every=1, eval_num_seeds=1)
    assert "loss" in metrics
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "eval_avg_placement" in metrics
    assert metrics["steps"] >= 32
