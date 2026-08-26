"""Observation space encodings and public scouting feature vector generators for reinforcement learning."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from tft_ai_player.simulation.actions import get_action_mask, index_to_hex
from tft_ai_player.simulation.config import SetData
from tft_ai_player.simulation.models import ChampionPool, Player
from tft_ai_player.simulation.stage_manager import StageManager


class ObservationEncoder:
    """Encodes complete TFT game state and full public opponent scouting into Gymnasium spaces."""

    def __init__(self, set_data: SetData) -> None:
        self.set_data = set_data

        # Build stable vocabularies (0 index reserved for empty/none)
        sorted_champs = sorted(list(set_data.champions.keys()))
        self.champ_to_idx: dict[str, int] = {cid: idx + 1 for idx, cid in enumerate(sorted_champs)}
        self.idx_to_champ: dict[int, str] = {idx + 1: cid for idx, cid in enumerate(sorted_champs)}

        sorted_items = sorted(list(set_data.items.keys()))
        self.item_to_idx: dict[str, int] = {iid: idx + 1 for idx, iid in enumerate(sorted_items)}
        self.idx_to_item: dict[int, str] = {idx + 1: iid for idx, iid in enumerate(sorted_items)}

        self.num_champs = len(self.champ_to_idx) + 1
        self.num_items = len(self.item_to_idx) + 1

    def encode_dict(
        self,
        focal_player: Player,
        opponents: Sequence[Player],
        pool: ChampionPool,
        stage_manager: StageManager,
    ) -> dict[str, np.ndarray]:
        """Produce structured dictionary observation with full public opponent scouting."""
        rinfo = stage_manager.get_current_round_info()

        # 1. Focal Player Stats (9,)
        stats = np.array(
            [
                focal_player.health / 100.0,
                focal_player.gold / 100.0,
                focal_player.level / 10.0,
                focal_player.exp / 100.0,
                focal_player.streak / 10.0,
                float(focal_player.alive),
                float(rinfo.stage),
                float(rinfo.round_in_stage),
                float(stage_manager.total_rounds_elapsed),
            ],
            dtype=np.float32,
        )

        # 2. Focal Board Grid (28, 5) -> [champ_idx, star, item1, item2, item3]
        board_obs = np.zeros((28, 5), dtype=np.float32)
        for hex_idx in range(28):
            pos = index_to_hex(hex_idx, self.set_data.board_cols)
            unit = focal_player.board.get(pos)
            if unit is not None:
                c_idx = self.champ_to_idx.get(unit.champion_id, 0)
                board_obs[hex_idx, 0] = float(c_idx)
                board_obs[hex_idx, 1] = float(unit.star_level)
                for it_idx, item_id in enumerate(unit.items[:3]):
                    board_obs[hex_idx, 2 + it_idx] = float(self.item_to_idx.get(item_id, 0))

        # 3. Focal Bench (9, 5)
        bench_obs = np.zeros((9, 5), dtype=np.float32)
        for bench_slot, unit in enumerate(focal_player.bench):
            if unit is not None:
                c_idx = self.champ_to_idx.get(unit.champion_id, 0)
                bench_obs[bench_slot, 0] = float(c_idx)
                bench_obs[bench_slot, 1] = float(unit.star_level)
                for it_idx, item_id in enumerate(unit.items[:3]):
                    bench_obs[bench_slot, 2 + it_idx] = float(self.item_to_idx.get(item_id, 0))

        # 4. Focal Item Bench (10,)
        item_bench_obs = np.zeros(10, dtype=np.float32)
        for it_idx, it_inst in enumerate(focal_player.item_bench[:10]):
            item_bench_obs[it_idx] = float(self.item_to_idx.get(it_inst.item_id, 0))

        # 5. Focal Shop (5,)
        shop_obs = np.zeros(5, dtype=np.float32)
        for s_idx, card_id in enumerate(focal_player.shop.slots):
            if card_id is not None:
                shop_obs[s_idx] = float(self.champ_to_idx.get(card_id, 0))

        # 6. Public Opponent Scouting - Summary Matrix (7, 8)
        opp_summary = np.zeros((7, 8), dtype=np.float32)
        for opp_idx, opp in enumerate(opponents[:7]):
            opp_summary[opp_idx, 0] = opp.health / 100.0
            opp_summary[opp_idx, 1] = opp.gold / 100.0
            opp_summary[opp_idx, 2] = opp.level / 10.0
            opp_summary[opp_idx, 3] = opp.exp / 100.0
            opp_summary[opp_idx, 4] = opp.streak / 10.0
            opp_summary[opp_idx, 5] = opp.get_board_value() / 100.0
            opp_summary[opp_idx, 6] = opp.board_unit_count / 10.0
            opp_summary[opp_idx, 7] = float(opp.alive)

        # 7. Public Opponent Scouting - Scouted Boards (7, 28, 5)
        opp_boards = np.zeros((7, 28, 5), dtype=np.float32)
        for opp_idx, opp in enumerate(opponents[:7]):
            if not opp.alive:
                continue
            for hex_idx in range(28):
                pos = index_to_hex(hex_idx, self.set_data.board_cols)
                unit = opp.board.get(pos)
                if unit is not None:
                    c_idx = self.champ_to_idx.get(unit.champion_id, 0)
                    opp_boards[opp_idx, hex_idx, 0] = float(c_idx)
                    opp_boards[opp_idx, hex_idx, 1] = float(unit.star_level)
                    for it_idx, item_id in enumerate(unit.items[:3]):
                        opp_boards[opp_idx, hex_idx, 2 + it_idx] = float(self.item_to_idx.get(item_id, 0))

        # 8. Public Opponent Scouting - Scouted Benches (7, 9, 5)
        opp_benches = np.zeros((7, 9, 5), dtype=np.float32)
        for opp_idx, opp in enumerate(opponents[:7]):
            if not opp.alive:
                continue
            for bench_slot, unit in enumerate(opp.bench):
                if unit is not None:
                    c_idx = self.champ_to_idx.get(unit.champion_id, 0)
                    opp_benches[opp_idx, bench_slot, 0] = float(c_idx)
                    opp_benches[opp_idx, bench_slot, 1] = float(unit.star_level)
                    for it_idx, item_id in enumerate(unit.items[:3]):
                        opp_benches[opp_idx, bench_slot, 2 + it_idx] = float(self.item_to_idx.get(item_id, 0))

        # 9. Public Champion Pool Remaining Counts (num_champs,)
        pool_counts = np.zeros(self.num_champs, dtype=np.float32)
        for cid, count in pool.counts.items():
            c_idx = self.champ_to_idx.get(cid, 0)
            if c_idx > 0:
                pool_counts[c_idx] = float(count)

        # 10. Action Mask (1721,)
        mask = get_action_mask(focal_player, self.set_data)

        return {
            "player_stats": stats,
            "board": board_obs,
            "bench": bench_obs,
            "item_bench": item_bench_obs,
            "shop": shop_obs,
            "opponents": opp_summary,
            "opponents_boards": opp_boards,
            "opponents_benches": opp_benches,
            "pool_counts": pool_counts,
            "action_mask": mask,
        }

    def encode_flat(
        self,
        focal_player: Player,
        opponents: Sequence[Player],
        pool: ChampionPool,
        stage_manager: StageManager,
    ) -> np.ndarray:
        """Flatten all observation sub-tensors into a single 1D vector."""
        d = self.encode_dict(focal_player, opponents, pool, stage_manager)
        return np.concatenate(
            [
                d["player_stats"],
                d["board"].flatten(),
                d["bench"].flatten(),
                d["item_bench"],
                d["shop"],
                d["opponents"].flatten(),
                d["opponents_boards"].flatten(),
                d["opponents_benches"].flatten(),
                d["pool_counts"],
            ]
        ).astype(np.float32)

    @property
    def flat_observation_dim(self) -> int:
        """Calculate total dimensions of the flattened observation vector."""
        # stats (9) + board (28*5=140) + bench (9*5=45) + items (10) + shop (5)
        # + opp_summary (7*8=56) + opp_boards (7*28*5=980) + opp_benches (7*9*5=315) + pool_counts (num_champs)
        return 9 + (28 * 5) + (9 * 5) + 10 + 5 + (7 * 8) + (7 * 28 * 5) + (7 * 9 * 5) + self.num_champs
