"""Sanity test script for ShopBeamSearchPlanner and TFTEnv dense reward shaping."""

import time
import torch
import numpy as np

from tft_ai_player.simulation.sets.set18 import get_set18_data
from tft_ai_player.simulation.game import TFTGame
from tft_ai_player.simulation.gym_env import TFTEnv, TFTStateEncoder
from tft_ai_player.rl.planner import ShopBeamSearchPlanner
from tft_ai_player.rl.models.networks import TFTActorCritic
from tft_ai_player.rl.agent_policy import RLBot

def test_planner_and_env():
    print(" [+] Initializing Set18 Data and Env...")
    set_data = get_set18_data()
    env = TFTEnv(set_data=set_data, device="cpu")
    obs, info = env.reset()

    focal = env.game.get_focal_player()
    focal.gold = 50
    focal.level = 6
    focal.shop.refresh(level=focal.level, pool=env.game.pool, set_data=set_data)
    print(f"     Focal Shop: {focal.shop.slots}")
    print(f"     Focal Gold: {focal.gold}, Level: {focal.level}")

    # Test Planner
    print(" [+] Testing ShopBeamSearchPlanner...")
    planner = ShopBeamSearchPlanner(
        set_data=set_data,
        encoder=env.encoder,
        world_model=None,
        beam_width=8,
        device=torch.device("cpu"),
    )

    t0 = time.time()
    planned_actions = planner.plan_shop_sequence(
        player=focal,
        pool=env.game.pool,
        stage=2,
        round_in_stage=1,
    )
    dt_ms = (time.time() - t0) * 1000
    print(f"     Planned Actions: {planned_actions} (took {dt_ms:.2f} ms)")
    assert isinstance(planned_actions, list)

    # Test Step Reward Shaping
    print(" [+] Testing TFTEnv step with dense potential rewards...")
    # Execute first action from planner or buy_shop_0 (action 1)
    test_action = planned_actions[0] if planned_actions else 1
    next_obs, reward, terminated, truncated, next_info = env.step(test_action)
    rb = next_info["reward_breakdown"]
    print(f"     Step Action: {test_action} | Reward: {reward:.4f}")
    print(f"     Reward breakdown: {rb}")

    # Test PASS_ROUND (action 0)
    print(" [+] Testing PASS_ROUND (action 0) combat resolution...")
    pass_obs, pass_reward, term, trunc, pass_info = env.step(0)
    pass_rb = pass_info["reward_breakdown"]
    print(f"     Pass Reward: {pass_reward:.4f}")
    print(f"     Pass breakdown: {pass_rb}")

    # Test RLBot with Planner
    print(" [+] Testing RLBot take_turn with Planner...")
    model = TFTActorCritic(obs_dim=704, action_dim=111, hidden_dim=256)
    bot = RLBot(model=model, set_data=set_data, deterministic=True, use_planner=True)
    bot.take_turn(focal, env.game.pool, set_data, stage=2, round_in_stage=2)
    print(f"     After bot turn: Gold: {focal.gold}, Board units: {focal.board_unit_count}")

    print("\n [SUCCESS] All unit sanity checks passed with ZERO errors!")

if __name__ == "__main__":
    test_planner_and_env()
