"""Interactive demo: Simulate an entire 8-player TFT match using the Gymnasium environment."""

from __future__ import annotations

import gymnasium as gym

import tft_ai_player.simulation  # Registers 'TFT-v0'
from tft_ai_player.simulation import (
    StandardTempoBot,
    TFTEnv,
)


def run_demo() -> None:
    print("\n" + "=" * 70)
    print(" [START] TFT GYMNASIUM SIMULATION DEMO")
    print("=" * 70)

    # Initialize environment
    env = TFTEnv(render_mode="human")
    obs, info = env.reset(seed=42)

    terminated = False
    truncated = False
    round_count = 0

    focal_bot = StandardTempoBot()

    # Step 1: Run Gym RL interaction loop while focal agent is alive
    while not terminated and not truncated:
        mask = info["action_mask"]
        r_stage = info["stage"]

        # Run focal bot to pick valid moves during planning phase
        focal_player = env.game.get_focal_player()
        rinfo = env.game.stage_manager.get_current_round_info()

        if focal_player.alive:
            focal_bot.take_turn(
                player=focal_player,
                pool=env.game.pool,
                set_data=env.set_data,
                stage=rinfo.stage,
                round_in_stage=rinfo.round_in_stage,
                rng=env.game.rng,
            )

        # Advance round with PASS (action 0)
        obs, reward, terminated, truncated, info = env.step(0)
        round_count += 1

        print(
            f"\n[-] Round {round_count} finished ({r_stage}) | "
            f"Step Reward: {reward:+.3f} | Focal HP: {info['health']} | Alive: {info['alive']}"
        )

        if env.game.round_combat_results:
            print(" [x] Combat Outcomes:")
            for res in env.game.round_combat_results:
                ghost_tag = " (vs Ghost)" if res.is_ghost_b else ""
                print(
                    f"   * P{res.winner_id} defeated P{res.loser_id}{ghost_tag} "
                    f"(Damage: {res.damage_dealt}, P(Win): {res.win_prob_a:.1%})"
                )

    print(f"\n[!] Focal Agent match ended with Placement #{info['placement'] or 8}!")

    # Step 2: Continue simulating remaining bots until 1st place champion is crowned
    if not env.game.is_over:
        print("\n[+] Continuing lobby combat until final champion is crowned...")
        while not env.game.is_over and round_count < 60:
            env.game.execute_bot_turns()
            env.game.resolve_round_phase()
            round_count += 1

    print("\n" + "=" * 70)
    print(" [WINNERS] FINAL TOURNAMENT STANDINGS")
    print("=" * 70)
    rankings = env.game.get_rankings()
    for rank, player in rankings:
        status_tag = "ALIVE (WINNER)" if player.alive else f"Eliminated (HP: {player.health})"
        print(
            f"  Rank #{rank}: {player.name} -> {status_tag} | "
            f"Board Value: {player.get_board_value():.0f}g | Level: {player.level}"
        )
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
