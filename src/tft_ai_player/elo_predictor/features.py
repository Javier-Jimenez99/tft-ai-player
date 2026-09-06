"""Full-game feature extractor for player skill and Elo regression.

Transforms a complete match trajectory into a dense feature vector capturing:
  - Economy management & interest discipline
  - Level curve & tempo timings (levels 6, 7, 8, 9)
  - Board quality, carry itemization, and star levels
  - Survival, HP preservation, and round win rates
  - Active trait synergy coherence
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence
import numpy as np

from tft_ai_player.embeddings.dataset import parse_stage_string


FEATURE_NAMES = [
    # Survival & Performance
    "final_placement",
    "rounds_survived",
    "pvp_win_rate",
    "final_health",
    "hp_at_stage_3_2",
    "hp_at_stage_4_2",
    "hp_loss_per_defeat",
    # Economy
    "avg_gold",
    "max_gold",
    "interest_50g_rounds_ratio",
    "interest_40g_rounds_ratio",
    "min_gold_midgame",
    # Level Timings & Tempo
    "final_level",
    "round_reached_lvl_6",
    "round_reached_lvl_7",
    "round_reached_lvl_8",
    "avg_level",
    # Board Quality & Composition Architecture
    "final_board_cost",
    "final_star_2_count",
    "final_star_3_count",
    "final_tier_4_count",
    "final_tier_5_count",
    "final_completed_items",
    "three_item_carries_count",
    # Synergy Coherence
    "active_traits_count",
    # Neural Round-by-Round Board Quality (from BoardQualityNet & Trunk)
    "neural_quality_mean",
    "neural_quality_stage_2",
    "neural_quality_stage_3",
    "neural_quality_stage_4",
    "neural_quality_stage_5",
    "neural_quality_growth",
    "neural_top4_prob_mean",
    "neural_top4_prob_stage_4",
    "neural_quality_peak",
    "neural_quality_final",
]

_DEFAULT_BOARD_EVALUATOR = None
_DEFAULT_VOCAB = None
_DEFAULT_ITEM_VOCAB = None
_DEFAULT_TRAIT_VOCAB = None


def get_default_board_evaluator():
    """Lazy-load cached default BoardQualityNet for fast neural feature extraction."""
    global _DEFAULT_BOARD_EVALUATOR, _DEFAULT_VOCAB, _DEFAULT_ITEM_VOCAB, _DEFAULT_TRAIT_VOCAB
    if _DEFAULT_BOARD_EVALUATOR is None:
        from pathlib import Path
        import torch
        ckpt = Path("models/board_evaluator/board_quality_best.pt")
        if ckpt.exists():
            try:
                from tft_ai_player.board_evaluator.model import BoardQualityNet
                from tft_ai_player.embeddings.vocab import ChampionVocabulary, ItemVocabulary, TraitVocabulary
                dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                _DEFAULT_BOARD_EVALUATOR = BoardQualityNet.load_checkpoint(ckpt, device=dev)
                _DEFAULT_VOCAB = ChampionVocabulary()
                _DEFAULT_ITEM_VOCAB = ItemVocabulary()
                _DEFAULT_TRAIT_VOCAB = TraitVocabulary()
            except Exception:
                _DEFAULT_BOARD_EVALUATOR = False
        else:
            _DEFAULT_BOARD_EVALUATOR = False
    return _DEFAULT_BOARD_EVALUATOR, _DEFAULT_VOCAB, _DEFAULT_ITEM_VOCAB, _DEFAULT_TRAIT_VOCAB


def parse_stage_round_index(stage_str: str) -> int:
    """Convert stage string (e.g. '3-2') to continuous round index (1-indexed)."""
    major, minor = parse_stage_string(stage_str)
    if major <= 0:
        return 0
    return (major - 2) * 6 + minor



def extract_game_features(
    trajectory: Sequence[Mapping[str, Any]],
    board_evaluator: Any | None = None,
    vocab: Any | None = None,
    item_vocab: Any | None = None,
    trait_vocab: Any | None = None,
    use_neural: bool = True,
) -> np.ndarray:
    """Extract a dense 35-dimensional hybrid feature vector (25 macro + 10 neural quality metrics)."""
    if not trajectory:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    # Sort rounds chronologically
    sorted_traj = sorted(trajectory, key=lambda x: parse_stage_string(str(x.get("round_stage", "2-1"))))
    total_rounds = len(sorted_traj)
    last_round = sorted_traj[-1]

    # 1. Survival & Placement
    last_stage = parse_stage_string(str(last_round.get("round_stage", "2-1")))
    final_hp = float(last_round.get("focal_health", 0) or 0)
    
    # Infer placement
    major, minor = last_stage
    cont_round = (major - 2) * 6 + minor
    if final_hp > 0:
        if major >= 6 and minor >= 4:
            placement = 1.0
        elif major >= 6:
            placement = 2.0
        elif major >= 5 and minor >= 5:
            placement = 2.5
        else:
            placement = 3.0
    else:
        if cont_round >= 26:
            placement = 2.0
        elif cont_round >= 23:
            placement = 3.0
        elif cont_round >= 20:
            placement = 4.0
        elif cont_round >= 17:
            placement = 5.0
        elif cont_round >= 14:
            placement = 6.0
        elif cont_round >= 10:
            placement = 7.0
        else:
            placement = 8.0

    victories = 0
    defeats = 0
    hp_at_3_2 = 100.0
    hp_at_4_2 = 100.0
    golds: list[float] = []
    levels: list[float] = []
    round_lvl_6 = 30.0
    round_lvl_7 = 35.0
    round_lvl_8 = 40.0

    hp_diffs = []
    prev_hp = 100.0

    for idx, r in enumerate(sorted_traj):
        stg = str(r.get("round_stage", ""))
        maj, minr = parse_stage_string(stg)
        hp = float(r.get("focal_health", 100) or 100)
        gold = float(r.get("focal_gold", 0) or 0)
        lvl = float(r.get("focal_level", 1) or 1)
        outcome = str(r.get("outcome", "")).lower()

        golds.append(gold)
        levels.append(lvl)

        if outcome == "victory" or r.get("label") in (1, "1"):
            victories += 1
        elif outcome == "defeat" or r.get("label") in (0, "0"):
            defeats += 1
            if prev_hp > hp:
                hp_diffs.append(prev_hp - hp)

        if (maj, minr) == (3, 2) or (maj == 3 and minr <= 2):
            hp_at_3_2 = hp
        if (maj, minr) == (4, 2) or (maj == 4 and minr <= 2):
            hp_at_4_2 = hp

        c_idx = (maj - 2) * 6 + minr
        if lvl >= 6 and round_lvl_6 == 30.0:
            round_lvl_6 = float(c_idx)
        if lvl >= 7 and round_lvl_7 == 35.0:
            round_lvl_7 = float(c_idx)
        if lvl >= 8 and round_lvl_8 == 40.0:
            round_lvl_8 = float(c_idx)

        prev_hp = hp

    pvp_wr = victories / max(1, (victories + defeats))
    hp_loss_per_def = float(np.mean(hp_diffs)) if hp_diffs else 0.0

    avg_gold = float(np.mean(golds)) if golds else 0.0
    max_gold = float(np.max(golds)) if golds else 0.0
    int_50_ratio = sum(1 for g in golds if g >= 50.0) / max(1, total_rounds)
    int_40_ratio = sum(1 for g in golds if g >= 40.0) / max(1, total_rounds)

    # Min gold in mid-game (rounds between stage 3-1 and 4-5)
    mid_golds = [g for i, g in enumerate(golds) if 7 <= i <= 18]
    min_gold_mid = float(np.min(mid_golds)) if mid_golds else avg_gold

    final_lvl = float(last_round.get("focal_level", 1) or 1)
    avg_lvl = float(np.mean(levels)) if levels else final_lvl

    # Final board characteristics
    final_board = last_round.get("focal_board") or []
    board_cost = 0.0
    star_2 = 0
    star_3 = 0
    tier_4 = 0
    tier_5 = 0
    completed_items = 0
    three_item_carries = 0
    traits_set = set()

    for u in final_board:
        if not isinstance(u, dict):
            continue
        c_cost = float(u.get("cost") or 1.0)
        c_star = int(u.get("tier") or u.get("star_level") or 1)
        items = u.get("items") or []

        board_cost += c_cost * (3 ** (c_star - 1))
        if c_star == 2:
            star_2 += 1
        elif c_star >= 3:
            star_3 += 1

        if c_cost == 4:
            tier_4 += 1
        elif c_cost >= 5:
            tier_5 += 1

        completed_items += len(items)
        if len(items) >= 3:
            three_item_carries += 1

        # Synergy traits
        u_traits = u.get("traits") or []
        for t in u_traits:
            if t:
                traits_set.add(str(t))

    macro_feats = [
        placement,
        float(total_rounds),
        pvp_wr,
        final_hp,
        hp_at_3_2,
        hp_at_4_2,
        hp_loss_per_def,
        avg_gold,
        max_gold,
        int_50_ratio,
        int_40_ratio,
        min_gold_mid,
        final_lvl,
        round_lvl_6,
        round_lvl_7,
        round_lvl_8,
        avg_lvl,
        board_cost,
        float(star_2),
        float(star_3),
        float(tier_4),
        float(tier_5),
        float(completed_items),
        float(three_item_carries),
        float(len(traits_set)),
    ]

    neural_feats = (
        extract_neural_metrics(
            sorted_traj,
            board_evaluator=board_evaluator,
            vocab=vocab,
            item_vocab=item_vocab,
            trait_vocab=trait_vocab,
        )
        if use_neural
        else [0.0] * 10
    )

    feat_vector = np.array(macro_feats + neural_feats, dtype=np.float32)
    return feat_vector


def extract_neural_metrics(
    trajectory: Sequence[Mapping[str, Any]],
    board_evaluator: Any | None = None,
    vocab: Any | None = None,
    item_vocab: Any | None = None,
    trait_vocab: Any | None = None,
) -> list[float]:
    """Extract 10 round-by-round neural quality metrics using BoardQualityNet."""
    if not trajectory:
        return [0.0] * 10

    evaluator = board_evaluator
    v = vocab
    iv = item_vocab
    tv = trait_vocab

    if evaluator is None:
        def_eval, def_v, def_iv, def_tv = get_default_board_evaluator()
        if def_eval:
            evaluator = def_eval
            v = def_v
            iv = def_iv
            tv = def_tv

    if not evaluator:
        return [0.0] * 10

    try:
        import torch
        import torch.nn.functional as F
        from tft_ai_player.embeddings.dataset import parse_loc_to_row_col

        dev = getattr(evaluator, "device", None)
        if dev is None:
            try:
                dev = next(evaluator.parameters()).device
            except Exception:
                dev = torch.device("cpu")

        N = len(trajectory)
        champs = torch.zeros((N, 28), dtype=torch.long, device=dev)
        stars = torch.zeros((N, 28), dtype=torch.long, device=dev)
        items = torch.zeros((N, 28, 3), dtype=torch.long, device=dev)
        traits = torch.zeros((N, 60), dtype=torch.float32, device=dev)
        scalars = torch.zeros((N, 8), dtype=torch.float32, device=dev)

        stages = []
        for i, r in enumerate(trajectory):
            stg = str(r.get("round_stage", "2-1"))
            maj, minr = parse_stage_string(stg)
            stages.append((maj, minr))
            scalars[i, 0] = float(r.get("focal_health", 100) or 100) / 100.0
            scalars[i, 1] = float(r.get("focal_level", 1) or 1) / 10.0
            scalars[i, 2] = float(r.get("focal_gold", 0) or 0) / 100.0
            b = r.get("focal_board") or []
            scalars[i, 3] = float(len(b)) / 10.0
            scalars[i, 4] = float(sum(len(u.get("items", []) or []) for u in b if isinstance(u, dict))) / 15.0
            scalars[i, 6] = float(maj) / 8.0
            scalars[i, 7] = float(minr) / 7.0

            champ_names_for_round: list[str] = []
            if v is not None:
                for u in b:
                    if not isinstance(u, dict):
                        continue
                    u_name = u.get("unit") or u.get("champion") or u.get("name")
                    if u_name:
                        champ_names_for_round.append(str(u_name))
                    loc = u.get("loc") or u.get("location") or 0
                    if isinstance(loc, str):
                        parsed = parse_loc_to_row_col(loc)
                        hex_idx = parsed[0] * 7 + parsed[1] if parsed else 0
                    else:
                        try:
                            hex_idx = int(loc)
                        except (ValueError, TypeError):
                            hex_idx = 0
                    if 0 <= hex_idx < 28:
                        champs[i, hex_idx] = v.encode(u_name)
                        stars[i, hex_idx] = int(u.get("tier") or u.get("star_level") or 1)
                        if iv is not None:
                            for it_idx, it_name in enumerate((u.get("items") or [])[:3]):
                                items[i, hex_idx, it_idx] = iv.encode(it_name)

            if tv is not None and champ_names_for_round:
                t_vec = tv.compute_trait_vector(champ_names_for_round)
                traits[i] = torch.as_tensor(t_vec, dtype=torch.float32, device=dev)

        with torch.no_grad():
            pred_place, logits_top4 = evaluator(champs, stars, items, traits, scalars)
            top4_p = F.softmax(logits_top4, dim=-1)[:, 1].cpu().numpy()
            quality = (8.0 - pred_place.squeeze(-1).cpu().numpy()) * (10.0 / 7.0)
            quality = np.clip(quality, 0.0, 10.0)

        stg2_q = [quality[i] for i, (m, _) in enumerate(stages) if m == 2]
        stg3_q = [quality[i] for i, (m, _) in enumerate(stages) if m == 3]
        stg4_q = [quality[i] for i, (m, _) in enumerate(stages) if m == 4]
        stg5_q = [quality[i] for i, (m, _) in enumerate(stages) if m >= 5]
        stg4_top4 = [top4_p[i] for i, (m, _) in enumerate(stages) if m == 4]

        mean_q = float(np.mean(quality))
        q2 = float(np.mean(stg2_q)) if stg2_q else mean_q
        q3 = float(np.mean(stg3_q)) if stg3_q else mean_q
        q4 = float(np.mean(stg4_q)) if stg4_q else mean_q
        q5 = float(np.mean(stg5_q)) if stg5_q else q4
        growth = float(q5 - q2)

        return [
            mean_q,
            q2,
            q3,
            q4,
            q5,
            growth,
            float(np.mean(top4_p)),
            float(np.mean(stg4_top4)) if stg4_top4 else float(np.mean(top4_p)),
            float(np.max(quality)),
            float(quality[-1]),
        ]
    except Exception:
        return [0.0] * 10

