"""Run a full TFT match, record all game state snapshots, and generate an interactive HTML dashboard."""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

from tft_ai_player.simulation import StandardTempoBot, TFTGame, get_set_data
from tft_ai_player.simulation.visualizer import GameRecorder, generate_visual_html


def run_and_export_dashboard(
    set_name: str = "TFTSet17",
    seed: int = 42,
    output_html_path: str | Path | None = None,
    output_dir: str | Path = "dashboards",
    open_browser: bool = False,
    max_rounds: int = 60,
) -> Path:
    """Run a simulated TFT game and export the interactive replay dashboard to an HTML file.

    Parameters
    ----------
    set_name : str
        TFT set to simulate (e.g., '17', '18', 'TFTSet17', 'TFTSet18'). Default: 'TFTSet17'.
    seed : int
        RNG seed for the simulation.
    output_html_path : str | Path | None
        Exact target file path. If None, saves to `{output_dir}/tft_simulation_{set_slug}_seed_{seed}.html`.
    output_dir : str | Path
        Directory where dashboards will be saved if output_html_path is not explicitly provided.
    open_browser : bool
        Whether to open the exported HTML dashboard in the default web browser.
    max_rounds : int
        Maximum number of rounds to simulate before stopping.
    """
    set_data = get_set_data(set_name)
    set_slug = set_data.set_name.lower()

    print("\n" + "=" * 70)
    print(f" [TFT VISUAL SIMULATION] Running match ({set_data.set_name}, seed={seed}) and recording visual replay...")
    print("=" * 70)

    game = TFTGame(set_data=set_data, seed=seed)
    recorder = GameRecorder(game)

    # Initial frame
    recorder.capture_snapshot(event_type="GAME_START")

    round_count = 0
    while not game.is_over and round_count < max_rounds:
        round_count += 1
        rinfo = game.stage_manager.get_current_round_info()

        # Bot planning actions
        game.execute_bot_turns()

        # Also let focal player 0 execute bot turns
        focal_p = game.get_focal_player()
        if focal_p.alive:
            focal_bot = StandardTempoBot()
            focal_bot.take_turn(
                player=focal_p,
                pool=game.pool,
                set_data=game.set_data,
                stage=rinfo.stage,
                round_in_stage=rinfo.round_in_stage,
                rng=game.rng,
            )

        # Resolve combat phase
        combat_results = game.resolve_round_phase()

        # Capture snapshot
        recorder.capture_snapshot(
            event_type="ROUND_RESOLVED",
            combat_results=combat_results,
        )

        print(f"  Recorded Stage {rinfo.stage_str} ({rinfo.round_type.value}) - {len(game.players)} Players Monitored")

    print(f"\n[+] Simulation complete! Recorded {len(recorder.frames)} visual snapshots.")

    # Determine output path
    if output_html_path is not None:
        out_path = Path(output_html_path).resolve()
    else:
        dir_path = Path(output_dir).resolve()
        out_path = dir_path / f"tft_simulation_{set_slug}_seed_{seed}.html"

    # Ensure parent directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate and write HTML
    html_content = generate_visual_html(recorder.frames, title=f"TFT Simulation ({set_data.set_name}, Seed {seed})")
    out_path.write_text(html_content, encoding="utf-8")

    print(f" [+] Visual Dashboard exported to: file:///{out_path.as_posix()}")

    if open_browser:
        print(" [+] Opening visual dashboard in default browser...")
        try:
            webbrowser.open(out_path.as_uri())
        except Exception:
            pass

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TFT simulation and export interactive visual dashboard.")
    parser.add_argument(
        "--set",
        "-s",
        type=str,
        default="TFTSet17",
        help="TFT set to simulate (e.g. '17', '18', 'TFTSet17', 'TFTSet18'). Default: 'TFTSet17'.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for simulation (default: 42).")
    parser.add_argument("--output-path", type=str, default=None, help="Custom output file path for the HTML dashboard.")
    parser.add_argument("--output-dir", type=str, default="dashboards", help="Output directory when output-path is not set (default: dashboards).")
    parser.add_argument("--max-rounds", type=int, default=60, help="Max rounds to simulate (default: 60).")
    parser.add_argument("--open-browser", action="store_true", help="Automatically open generated dashboard in browser.")
    args = parser.parse_args()

    run_and_export_dashboard(
        set_name=args.set,
        seed=args.seed,
        output_html_path=args.output_path,
        output_dir=args.output_dir,
        open_browser=args.open_browser,
        max_rounds=args.max_rounds,
    )


if __name__ == "__main__":
    main()
