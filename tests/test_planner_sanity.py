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
from tft_ai_player.simulation.models import ChampionInstance, Player


class _ItemAwareEncoder:
    def extract_state_vector(self, player, stage, round_in_stage, target_z=None):
        equipped_items = sum(len(unit.items) for unit in player.board.values())
        state = torch.tensor([float(equipped_items)])
        return np.zeros(1, dtype=np.float32), state, state


class _ItemAwareBoardEvaluator(torch.nn.Module):
    def forward_fused(self, fused_state):
        equipped_items = fused_state[:, 0]
        placement = 8.0 - equipped_items * 7.0
        top4_logits = torch.stack((torch.zeros_like(equipped_items), equipped_items * 10.0), dim=-1)
        return placement, top4_logits

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


def test_planner_selects_item_action_when_value_oracle_prefers_it():
    """The value oracle must be able to choose a legacy item-action branch."""
    set_data = get_set18_data()
    player = Player(0, set_data)
    player.board[(0, 0)] = ChampionInstance("TFT18_Maokai", cost=1, star_level=1)
    player.add_item("TFT_Item_BFSword")
    pool = TFTGame(set_data=set_data, seed=7).pool

    planner = ShopBeamSearchPlanner(
        set_data=set_data,
        encoder=_ItemAwareEncoder(),
        board_evaluator=_ItemAwareBoardEvaluator(),
        use_neural_eval=True,
        device=torch.device("cpu"),
    )
    actions = planner.plan_shop_sequence(player, pool, stage=2, round_in_stage=1)

    assert actions
    assert actions[0] == 101

if __name__ == "__main__":
    test_planner_and_env()
