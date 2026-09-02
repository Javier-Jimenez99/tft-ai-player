from pathlib import Path

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
from tft_ai_player.simulation.actions import TOTAL_DISCRETE_ACTIONS
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
    dummy_mask = torch.ones(batch_size, TOTAL_DISCRETE_ACTIONS, dtype=torch.bool)

    logits, values = model(dummy_obs)
    assert logits.shape == (batch_size, TOTAL_DISCRETE_ACTIONS)
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
        round_winner_model_path=None,
    )

    metrics = trainer.train_iteration(generation=1, rollout_steps=32, eval_every=1, eval_num_seeds=1)
    assert "loss" in metrics
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "eval_avg_placement" in metrics
    assert metrics["steps"] >= 32


def test_unit_synergy_transformer_and_recurrent_memory() -> None:
    """Verify UnitSynergyTransformer cross-attention and recurrent GRU hidden state updates."""
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
    dummy_mask = torch.ones(batch_size, TOTAL_DISCRETE_ACTIONS, dtype=torch.bool)

    # Initial step without hidden state
    action, log_prob, val, hidden1 = model.get_action(
        dummy_obs, dummy_mask, hidden_state=None, return_hidden=True
    )
    assert action.shape == (batch_size,)
    assert isinstance(hidden1, tuple)
    h1, c1 = hidden1
    assert h1.shape == (2, batch_size, 128)
    assert c1.shape == (2, batch_size, 128)

    # Sequential step passing hidden state
    action2, log_prob2, val2, hidden2 = model.get_action(
        dummy_obs, dummy_mask, hidden_state=hidden1, return_hidden=True
    )
    assert isinstance(hidden2, tuple)
    h2, c2 = hidden2
    assert h2.shape == (2, batch_size, 128)
    assert c2.shape == (2, batch_size, 128)
    assert not torch.allclose(h1, h2)  # Hidden state evolved across steps


def test_entropy_schedule_decay() -> None:
    """Verify dynamic entropy schedule decays as generations advance."""
    set_data = get_default_set17_data()
    model = TFTActorCritic(
        num_champs=len(set_data.champions) + 1,
        num_items=len(set_data.items) + 1,
        hidden_dim=64,
    )
    ppo = MaskablePPO(
        model,
        entropy_coef_start=0.05,
        entropy_coef_min=0.001,
        entropy_decay_rate=0.90,
    )

    c0 = ppo.step_entropy_schedule(0)
    assert np.isclose(c0, 0.05)

    c5 = ppo.step_entropy_schedule(5)
    assert c5 < c0

    c100 = ppo.step_entropy_schedule(100)
    assert np.isclose(c100, 0.001)  # Clamped at min entropy coef


def test_checkpoint_state_serialization(tmp_path: pytest.TempPathFactory) -> None:
    """Verify full training checkpointing saves and restores training generation and weights."""
    set_data = get_default_set17_data()
    trainer1 = LeagueTrainer(
        set_data=set_data,
        hidden_dim=64,
        checkpoint_dir=str(tmp_path),
        round_winner_model_path=None,
    )

    # Save checkpoint at generation 7
    ckpt_path = trainer1.league.checkpoints.save_training_state(
        model=trainer1.model,
        optimizer=trainer1.ppo.optimizer,
        generation=7,
        entropy_coef=0.025,
        metrics_history=[{"generation": 7, "loss": 0.5}],
    )
    assert ckpt_path.exists()

    # Load into new trainer instance
    trainer2 = LeagueTrainer(
        set_data=set_data,
        hidden_dim=64,
        checkpoint_dir=str(tmp_path),
        round_winner_model_path=None,
    )
    resumed_gen = trainer2.load_checkpoint(ckpt_path)
    assert resumed_gen == 7
    assert np.isclose(trainer2.ppo.entropy_coef, 0.025)
    assert len(trainer2.metrics_history) == 1


def test_wandb_and_decomposed_rewards(tmp_path: pytest.TempPathFactory) -> None:
    """Verify decomposed reward breakdown and WandB multi-agent metric logging."""
    from tft_ai_player.rl.logger import WandBLogger

    logger = WandBLogger(
        project="tft-test-league",
        run_name="unit_test_run",
        enabled=False,  # Test local processing without requiring wandb network call
    )

    # Test environment reward breakdown
    set_data = get_default_set17_data()
    env = TFTEnv(set_data=set_data)
    obs, info = env.reset(seed=42)
    obs, reward, term, trunc, info = env.step(1)  # Buy champion

    assert "reward_breakdown" in info
    rb = info["reward_breakdown"]
    assert "block_combat_outcome" in rb
    assert "block_board_power" in rb
    assert "block_constraints_economy" in rb
    assert "rew_potential_delta" in rb
    assert "rew_round_win" in rb

    # Test WandB logging payload formatting with Tri-Tier multi-agent channels
    mock_metrics = {
        "mean_reward": reward,
        "loss": -0.05,
        "policy_loss": -0.03,
        "value_loss": 0.01,
        "entropy": 2.5,
        "entropy_coef": 0.02,
        "action_distribution": {
            "Pass": 0.40,
            "Buy": 0.30,
            "Reroll": 0.10,
            "EXP": 0.05,
            "Lock": 0.0,
            "Sell": 0.05,
            "Move": 0.05,
            "Equip": 0.03,
            "Combine": 0.02,
        },
        "tri_tier": {
            "Main_Agent": {"elo": 1250.0, "mean_reward": 0.15, "loss": 0.02},
            "Main_Exploiter": {"elo": 1210.0, "mean_reward": 0.12, "loss": 0.03},
            "League_Exploiter": {"elo": 1190.0, "mean_reward": 0.08, "loss": 0.04},
        },
        **rb,
    }
    logger.log_generation(1, mock_metrics)
    logger.close()


def test_action_grouping_helpers() -> None:
    """Verify get_action_group accurately partitions discrete action indices."""
    from tft_ai_player.simulation.actions import ACTION_GROUP_NAMES, get_action_group

    assert get_action_group(0) == "Pass"
    assert get_action_group(1) == "Buy"
    assert get_action_group(5) == "Buy"
    assert get_action_group(6) == "Reroll"
    assert get_action_group(7) == "EXP"
    assert get_action_group(8) == "Lock"
    assert get_action_group(9) == "Sell"
    assert get_action_group(17) == "Sell"
    assert get_action_group(18) == "Sell"
    assert get_action_group(29) == "Sell"
    assert get_action_group(30) == "Deploy"
    assert get_action_group(38) == "Deploy"
    assert get_action_group(39) == "Recall"
    assert get_action_group(50) == "Recall"
    assert get_action_group(51) == "Equip"
    assert get_action_group(260) == "Equip"
    assert get_action_group(261) == "Combine"
    assert get_action_group(305) == "Combine"
    assert set(ACTION_GROUP_NAMES) == {"Pass", "Buy", "Reroll", "EXP", "Lock", "Sell", "Deploy", "Recall", "Equip", "Combine"}


def test_alphastar_five_core_principles(tmp_path: Path) -> None:
    """Verify the 5 core AlphaStar league design pillars:
    1. Multilateral Elo: 8-player lobby decomposes into 28 1v1 matchups for global ranking.
    2. PFSP: Dynamic matchmaking weighting P ~ (1 - win_rate)^T.
    3. Tri-Tier Bot Ecosystem: Main, Main Exploiter, and League Exploiter role matchmaking.
    4. Deterministic Rule-Based Bots: Baseline anchor bots never mutate.
    5. Static Evaluation Environment: Frozen evaluation ladder tested across paired seeds.
    """
    league = LeagueManager(checkpoint_dir=str(tmp_path / "checkpoints"))

    # 1. Multilateral Elo System
    league.register_agent("main_agent", name="Main Agent", role=AgentRole.MAIN)
    league.register_agent("main_exploiter", name="Main Exploiter", role=AgentRole.MAIN_EXPLOITER)
    league.register_agent("league_exploiter", name="League Exploiter", role=AgentRole.LEAGUE_EXPLOITER)

    placements = {
        "main_agent": 1,
        "main_exploiter": 2,
        "league_exploiter": 3,
        "bot_standard_tempo": 4,
        "bot_greedy_banker": 5,
        "bot_hyper_roll_exploiter": 6,
        "bot_fast9_econ_exploiter": 7,
        "bot_random": 8,
    }
    deltas = league.elo_system.update_lobby_ratings(placements, league.profiles)
    assert len(deltas) == 8
    # 1st place gains Elo, 8th place loses Elo
    assert deltas["main_agent"] > 0
    assert deltas["bot_random"] < 0
    # Head-to-head 28 pairings recorded
    main_prof = league.profiles["main_agent"]
    assert "bot_random" in main_prof.h2h_records
    assert main_prof.h2h_records["bot_random"][0] == 1  # 1 win against bot_random
    assert main_prof.get_win_rate_vs("bot_random") == 1.0

    # 2. PFSP Matchmaking Engine
    # Force low win-rate against a specific opponent
    main_prof.h2h_records["bot_hyper_roll_exploiter"] = [0, 10]  # 0 wins, 10 losses (0.0 win rate)
    assert main_prof.get_win_rate_vs("bot_hyper_roll_exploiter") == 0.0
    opponents = league.sample_pfsp_opponents("main_agent", num_opponents=5, temperature=2.0)
    assert len(opponents) == 5

    # 3. Tri-Tier Role Matchmaking
    main_opps = league.sample_role_pfsp_opponents(AgentRole.MAIN, "main_agent", num_opponents=7)
    assert len(main_opps) == 7
    me_opps = league.sample_role_pfsp_opponents(AgentRole.MAIN_EXPLOITER, "main_exploiter", num_opponents=7)
    assert len(me_opps) == 7
    le_opps = league.sample_role_pfsp_opponents(AgentRole.LEAGUE_EXPLOITER, "league_exploiter", num_opponents=7)
    assert len(le_opps) == 7

    # 4. Deterministic Rule-Based Anchor Bots
    assert "bot_standard_tempo" in league.baselines
    assert "bot_greedy_banker" in league.baselines
    assert league.profiles["bot_standard_tempo"].role == AgentRole.BASELINE

    # 5. Static Evaluation Environment
    evaluator = TournamentEvaluator(league)
    benchmark_res = evaluator.run_tiered_benchmark("main_agent", num_seeds_per_tier=1, base_seed=42)
    assert "eval_avg_placement" in benchmark_res
    assert "bench_Tier1_Random_placement" in benchmark_res
    assert "bench_Tier2_Banker_placement" in benchmark_res
    assert "bench_Tier3_Tempo_placement" in benchmark_res
    assert "bench_Tier4_Exploiters_placement" in benchmark_res



