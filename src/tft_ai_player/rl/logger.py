"""Weights & Biases (WandB) experiment tracking, health telemetry, and executive dashboard logger for TFT RL."""

from __future__ import annotations

import base64
import json
import logging
import netrc
import os
import time
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


def check_collapse_warnings(metrics: dict[str, Any]) -> list[str]:
    """Check health metrics against collapse warning thresholds.

    Returns human-readable diagnostic warnings if any training metric breaches safety bounds.
    """
    warnings: list[str] = []

    # 1. Policy Entropy
    entropy = metrics.get("policy_entropy")
    if entropy is not None:
        if entropy < 0.05:
            warnings.append(
                f"Entropy Collapse (entropy={entropy:.3f} < 0.05): Policy prematurely collapsing into single action."
            )
        elif entropy > 4.0:
            warnings.append(
                f"Entropy Anomaly (entropy={entropy:.3f} > 4.0): Policy failing to converge."
            )

    # 2. Explained Variance
    exp_var = metrics.get("explained_variance")
    if exp_var is not None and exp_var < 0.0:
        warnings.append(
            f"Critic Failure (explained_variance={exp_var:.3f} < 0.0): Value network corrupted."
        )

    # 3. Approximate KL
    approx_kl = metrics.get("approx_kl")
    if approx_kl is not None and approx_kl > 0.05:
        warnings.append(
            f"Policy Explosion (approx_kl={approx_kl:.4f} > 0.05): Policy update too aggressive."
        )

    # 4. Action Mask Rejection Rate
    mask_rej = metrics.get("action_mask_rejection_rate")
    if mask_rej is not None and mask_rej > 0.00001:
        warnings.append(
            f"Masking Leak (action_mask_rejection_rate={mask_rej:.4f} > 0.0%): Invalid actions leaking."
        )

    # 5. Macro Z Cosine
    macro_z = metrics.get("macro_z_cosine")
    if macro_z is not None and macro_z < 0.40:
        warnings.append(
            f"Specialist Divergence (macro_z_cosine={macro_z:.3f} < 0.40): Drifting off target archetype."
        )

    # 6. Micro World Model Cosine
    micro_wm = metrics.get("micro_world_model_cosine")
    if micro_wm is not None and micro_wm < 0.50:
        warnings.append(
            f"Macro Regression (micro_world_model_cosine={micro_wm:.3f} < 0.50): Diverging from planning trajectory."
        )

    return warnings


def format_wandb_payload(raw_metrics: dict[str, Any]) -> dict[str, Any]:
    """Format metrics into a clean, hierarchical structure with a 5-plot 'Principal' executive section.

    The 'Principal' section contains at most 5 essential charts for immediate go/no-go assessment:
    1. Principal/1_Average_Placement: Game placement performance (1.0 to 8.0, lower is better)
    2. Principal/2_Critic_Explained_Variance: Critic value health (>0.50 is healthy, <0 is failure)
    3. Principal/3_Policy_Entropy: Action exploration (2.5 -> 0.8 is healthy, <0.15 is collapse)
    4. Principal/4_Policy_Stability_KL: PPO step stability (0.005 -> 0.02 is healthy, >0.05 is explosion)
    5. Principal/5_Benchmark_WinRate: Overall win rate vs fixed benchmark bots (Alpha, Beta, Gamma)
    """
    payload: dict[str, Any] = {}

    # =========================================================================
    # SECTION 1: PRINCIPAL (Max 5 Executive Monitoring Plots)
    # =========================================================================
    # 1. Placement Performance
    avg_place = raw_metrics.get("eval_avg_placement", raw_metrics.get("avg_placement"))
    if avg_place is not None:
        payload["Principal/1_Average_Placement"] = float(avg_place)

    # 2. Critic Health
    exp_var = raw_metrics.get("explained_variance")
    if exp_var is not None:
        payload["Principal/2_Critic_Explained_Variance"] = float(exp_var)

    # 3. Policy Exploration / Entropy
    entropy = raw_metrics.get("policy_entropy")
    if entropy is not None:
        payload["Principal/3_Policy_Entropy"] = float(entropy)

    # 4. Policy Stability (Approx KL)
    approx_kl = raw_metrics.get("approx_kl")
    if approx_kl is not None:
        payload["Principal/4_Policy_Stability_KL"] = float(approx_kl)

    # 5. Benchmark Performance
    bot_a_wr = raw_metrics.get("bot_alpha_win_rate")
    bot_b_wr = raw_metrics.get("bot_beta_win_rate")
    bot_g_wr = raw_metrics.get("bot_gamma_win_rate")
    if bot_a_wr is not None and bot_b_wr is not None and bot_g_wr is not None:
        avg_bot_wr = (float(bot_a_wr) + float(bot_b_wr) + float(bot_g_wr)) / 3.0
        payload["Principal/5_Benchmark_WinRate"] = avg_bot_wr * 100.0
    elif "mean_reward" in raw_metrics:
        payload["Principal/5_Mean_Episode_Reward"] = float(raw_metrics["mean_reward"])

    # =========================================================================
    # SECTION 2: PERFORMANCE (Detailed Tournament & Match Metrics)
    # =========================================================================
    if "top4_rate" in raw_metrics and raw_metrics["top4_rate"] is not None:
        payload["Performance/Top4_Rate"] = float(raw_metrics["top4_rate"]) * 100.0
    if "win_rate" in raw_metrics and raw_metrics["win_rate"] is not None:
        payload["Performance/Win_Rate"] = float(raw_metrics["win_rate"]) * 100.0
    if "mean_reward" in raw_metrics and raw_metrics["mean_reward"] is not None:
        payload["Performance/Mean_Reward"] = float(raw_metrics["mean_reward"])
    if "eval_top4_rate" in raw_metrics and raw_metrics["eval_top4_rate"] is not None:
        payload["Performance/Eval_Top4_Rate"] = float(raw_metrics["eval_top4_rate"]) * 100.0
    if "eval_win_rate" in raw_metrics and raw_metrics["eval_win_rate"] is not None:
        payload["Performance/Eval_Win_Rate"] = float(raw_metrics["eval_win_rate"]) * 100.0
    if "bot_alpha_win_rate" in raw_metrics and raw_metrics["bot_alpha_win_rate"] is not None:
        payload["Performance/Bot_Alpha_Fast8_WinRate"] = float(raw_metrics["bot_alpha_win_rate"]) * 100.0
    if "bot_beta_win_rate" in raw_metrics and raw_metrics["bot_beta_win_rate"] is not None:
        payload["Performance/Bot_Beta_Hyperroll_WinRate"] = float(raw_metrics["bot_beta_win_rate"]) * 100.0
    if "bot_gamma_win_rate" in raw_metrics and raw_metrics["bot_gamma_win_rate"] is not None:
        payload["Performance/Bot_Gamma_Greedy_WinRate"] = float(raw_metrics["bot_gamma_win_rate"]) * 100.0
    if "league_elo" in raw_metrics and raw_metrics["league_elo"] is not None:
        payload["Performance/League_Elo"] = float(raw_metrics["league_elo"])

    # =========================================================================
    # SECTION 3: OPTIMIZATION (Gradients, Losses, Learning Rates)
    # =========================================================================
    if "policy_loss" in raw_metrics and raw_metrics["policy_loss"] is not None:
        payload["Optimization/Policy_Loss"] = float(raw_metrics["policy_loss"])
    if "value_loss" in raw_metrics and raw_metrics["value_loss"] is not None:
        payload["Optimization/Value_Loss"] = float(raw_metrics["value_loss"])
    if "entropy_loss" in raw_metrics and raw_metrics["entropy_loss"] is not None:
        payload["Optimization/Entropy_Loss"] = float(raw_metrics["entropy_loss"])
    if "total_loss" in raw_metrics and raw_metrics["total_loss"] is not None:
        payload["Optimization/Total_Loss"] = float(raw_metrics["total_loss"])
    if "learning_rate" in raw_metrics and raw_metrics["learning_rate"] is not None:
        payload["Optimization/Learning_Rate"] = float(raw_metrics["learning_rate"])
    if "entropy_coef" in raw_metrics and raw_metrics["entropy_coef"] is not None:
        payload["Optimization/Entropy_Coef"] = float(raw_metrics["entropy_coef"])

    # =========================================================================
    # SECTION 4: DIAGNOSTICS & ALIGNMENT (Action Masking, Z-Index, World Model)
    # =========================================================================
    if "action_mask_rejection_rate" in raw_metrics and raw_metrics["action_mask_rejection_rate"] is not None:
        payload["Diagnostics/Action_Mask_Rejection_Rate"] = float(raw_metrics["action_mask_rejection_rate"]) * 100.0
    if "macro_z_cosine" in raw_metrics and raw_metrics["macro_z_cosine"] is not None:
        payload["Diagnostics/Macro_Z_Cosine"] = float(raw_metrics["macro_z_cosine"])
    if "micro_world_model_cosine" in raw_metrics and raw_metrics["micro_world_model_cosine"] is not None:
        payload["Diagnostics/Micro_World_Model_Cosine"] = float(raw_metrics["micro_world_model_cosine"])
    if "current_beta" in raw_metrics and raw_metrics["current_beta"] is not None:
        payload["Diagnostics/Reward_Beta_Weight"] = float(raw_metrics["current_beta"])
    if "current_alpha" in raw_metrics and raw_metrics["current_alpha"] is not None:
        payload["Diagnostics/Reward_Alpha_Weight"] = float(raw_metrics["current_alpha"])
    if "actions_per_round" in raw_metrics and raw_metrics["actions_per_round"] is not None:
        payload["Diagnostics/Actions_Per_Round"] = float(raw_metrics["actions_per_round"])

    # =========================================================================
    # SECTION 5: REWARD BLOCKS (Complete Multi-Objective Decomposition)
    # =========================================================================
    if "reward_breakdown" in raw_metrics and isinstance(raw_metrics["reward_breakdown"], dict):
        rb = raw_metrics["reward_breakdown"]
        key_mapping = {
            "r_combat": "Rewards/1_Combat_HP_Reward",
            "r_interest": "Rewards/2_Interest_Gold_Reward",
            "r_terminal": "Rewards/3_Placement_Terminal_Reward",
            "r_micro": "Rewards/4_Micro_Transition_Alignment",
            "r_macro": "Rewards/5_Macro_Cluster_Strategy",
            "r_env": "Rewards/6_Total_Environment_Game_Reward",
        }
        for k, v in rb.items():
            if isinstance(v, (int, float)):
                dest_name = key_mapping.get(k, f"Rewards/{k}")
                payload[dest_name] = float(v)

    # =========================================================================
    # SECTION 6: ACTIONS (Action Distribution Breakdown %)
    # =========================================================================
    if "action_buy_xp_pct" in raw_metrics and raw_metrics["action_buy_xp_pct"] is not None:
        payload["Actions/1_Buy_XP_Pct"] = float(raw_metrics["action_buy_xp_pct"])
    if "action_pass_pct" in raw_metrics and raw_metrics["action_pass_pct"] is not None:
        payload["Actions/2_Pass_Round_Pct"] = float(raw_metrics["action_pass_pct"])
    if "action_equip_item_pct" in raw_metrics and raw_metrics["action_equip_item_pct"] is not None:
        payload["Actions/3_Equip_Item_Pct"] = float(raw_metrics["action_equip_item_pct"])
    if "action_buy_shop_pct" in raw_metrics and raw_metrics["action_buy_shop_pct"] is not None:
        payload["Actions/4_Buy_Shop_Pct"] = float(raw_metrics["action_buy_shop_pct"])
    if "action_deploy_board_pct" in raw_metrics and raw_metrics["action_deploy_board_pct"] is not None:
        payload["Actions/5_Deploy_Board_Pct"] = float(raw_metrics["action_deploy_board_pct"])
    if "action_reroll_pct" in raw_metrics and raw_metrics["action_reroll_pct"] is not None:
        payload["Actions/6_Reroll_Shop_Pct"] = float(raw_metrics["action_reroll_pct"])
    # =========================================================================
    # SECTION 7: COMPOSITIONS & Z-INDEX MASTERY
    # =========================================================================
    if "macro_alignment_cosine" in raw_metrics and raw_metrics["macro_alignment_cosine"] is not None:
        payload["Compositions/1_Macro_Alignment_Cosine"] = float(raw_metrics["macro_alignment_cosine"])
    if "target_cluster_match_rate" in raw_metrics and raw_metrics["target_cluster_match_rate"] is not None:
        payload["Compositions/2_Target_Cluster_Match_Rate"] = float(raw_metrics["target_cluster_match_rate"])

    # =========================================================================
    # SECTION 8: UNIVERSAL GOAL-CONDITIONED EXPLOITER METRICS
    # =========================================================================
    for k, v in raw_metrics.items():
        if k.startswith("Exploiter/") and isinstance(v, (int, float)):
            payload[k] = float(v)

    return payload


class WandBSingleRun:
    """Represents a single autonomous WandB run using native FileStream & GraphQL."""

    def __init__(
        self,
        project: str,
        entity: str,
        group: str,
        display_name: str,
        run_id: str,
        config: dict[str, Any] | None,
        api_key: str,
    ) -> None:
        self.project = project
        self.entity = entity
        self.group = group
        self.display_name = display_name
        self.run_id = run_id
        self.api_key = api_key
        self.stream_offset = 0
        self.dashboard_url = f"https://wandb.ai/{self.entity}/{self.project}/runs/{self.run_id}"

        # Upsert bucket
        mutation = """
        mutation UpsertBucket($input: UpsertBucketInput!) {
            upsertBucket(input: $input) {
                bucket {
                    id
                    name
                    displayName
                }
            }
        }
        """
        input_vars = {
            "entityName": self.entity,
            "modelName": self.project,
            "name": self.run_id,
            "displayName": self.display_name,
            "groupName": self.group,
            "config": json.dumps(config or {}),
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req = urllib.request.Request(
            "https://api.wandb.ai/graphql",
            data=json.dumps({"query": mutation, "variables": {"input": input_vars}}).encode("utf-8"),
            headers=headers,
        )
        urllib.request.urlopen(req, timeout=10)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Stream metrics to WandB FileStream API."""
        try:
            payload = dict(metrics)
            payload["_step"] = step
            payload["_runtime"] = time.time()
            if self.stream_offset < step:
                self.stream_offset = step

            basic_auth = base64.b64encode(f"api:{self.api_key}".encode("utf-8")).decode("ascii")
            fs_headers = {
                "Content-Type": "application/json",
                "Authorization": f"Basic {basic_auth}",
            }
            fs_payload = {
                "files": {
                    "wandb-history.jsonl": {
                        "offset": self.stream_offset,
                        "content": [json.dumps(payload) + "\n"],
                    }
                }
            }
            fs_req = urllib.request.Request(
                f"https://api.wandb.ai/files/{self.entity}/{self.project}/{self.run_id}/file_stream",
                data=json.dumps(fs_payload).encode("utf-8"),
                headers=fs_headers,
            )
            urllib.request.urlopen(fs_req, timeout=5)
            self.stream_offset += 1
            self._update_summary(payload)
        except Exception as e:
            logger.debug(f"Direct WandB stream error for {self.run_id}: {e}")

    def _update_summary(self, summary_dict: dict[str, Any]) -> None:
        try:
            mutation = """
            mutation UpsertBucket($input: UpsertBucketInput!) {
                upsertBucket(input: $input) {
                    bucket {
                        id
                    }
                }
            }
            """
            input_vars = {
                "entityName": self.entity,
                "modelName": self.project,
                "name": self.run_id,
                "summaryMetrics": json.dumps(summary_dict),
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            req = urllib.request.Request(
                "https://api.wandb.ai/graphql",
                data=json.dumps({"query": mutation, "variables": {"input": input_vars}}).encode("utf-8"),
                headers=headers,
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass

    def close(self) -> None:
        try:
            basic_auth = base64.b64encode(f"api:{self.api_key}".encode("utf-8")).decode("ascii")
            fs_headers = {
                "Content-Type": "application/json",
                "Authorization": f"Basic {basic_auth}",
            }
            fs_payload = {"complete": True, "exitcode": 0}
            fs_req = urllib.request.Request(
                f"https://api.wandb.ai/files/{self.entity}/{self.project}/{self.run_id}/file_stream",
                data=json.dumps(fs_payload).encode("utf-8"),
                headers=fs_headers,
            )
            urllib.request.urlopen(fs_req, timeout=5)
        except Exception:
            pass


class WandBLogger:
    """WandB Logger for TFT Reinforcement Learning and AlphaStar League."""

    def __init__(
        self,
        project: str = "tft-ai-league",
        entity: str | None = None,
        group: str | None = None,
        run_name: str | None = None,
        run_id: str | None = None,
        config: dict[str, Any] | None = None,
        enabled: bool = True,
    ) -> None:
        self.project = project
        self.group = group or run_name or "ppo_league_v1"
        self.run_name = run_name or self.group
        self.run_id = run_id or self.run_name
        self.enabled = enabled
        self.api_key: str | None = None
        self.entity: str = entity or ""
        self.single_run: WandBSingleRun | None = None

        if not self.enabled:
            return

        try:
            self._resolve_credentials()
            self.single_run = WandBSingleRun(
                project=self.project,
                entity=self.entity,
                group=self.group,
                display_name=self.run_name,
                run_id=self.run_id,
                config=config,
                api_key=self.api_key,
            )
            print(f"  [WandB Run] {self.run_name}: {self.single_run.dashboard_url}")
        except Exception as e:
            logger.warning(f"Could not initialize WandB run: {e}. Running in local logging mode.")
            self.enabled = False

    def _resolve_credentials(self) -> None:
        self.api_key = os.environ.get("WANDB_API_KEY")
        if not self.api_key:
            netrc_paths = [os.path.expanduser("~/_netrc"), os.path.expanduser("~/.netrc")]
            for np in netrc_paths:
                if os.path.exists(np):
                    try:
                        n = netrc.netrc(np)
                        auth = n.authenticators("api.wandb.ai")
                        if auth and len(auth) >= 3 and auth[2]:
                            self.api_key = auth[2]
                            break
                    except Exception:
                        pass

        if not self.api_key:
            raise ValueError("WandB API key not found in environment or netrc.")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req = urllib.request.Request(
            "https://api.wandb.ai/graphql",
            data=json.dumps({"query": "query Viewer { viewer { id entity username } }"}).encode("utf-8"),
            headers=headers,
        )
        res = urllib.request.urlopen(req, timeout=10)
        viewer_data = json.loads(res.read())
        resolved_entity = viewer_data.get("data", {}).get("viewer", {}).get("entity", "")
        if resolved_entity and not self.entity:
            self.entity = resolved_entity

    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Format and log metrics dictionary at given step/generation with the 5 Principal plots."""
        if not self.enabled or self.single_run is None:
            return
        formatted_payload = format_wandb_payload(metrics)
        self.single_run.log(formatted_payload, step=step)

    def log_strategy_progression(self, results: dict[str, Any], step: int) -> None:
        """Upload AlphaStar strategy landscape progression artifacts (PNG, GIF, HTML) to Weights & Biases."""
        if not self.enabled:
            return
        try:
            import wandb
            from pathlib import Path

            if wandb.run is None and self.api_key:
                os.environ["WANDB_API_KEY"] = self.api_key
                wandb.init(
                    project=self.project,
                    entity=self.entity,
                    id=self.run_id,
                    resume="allow",
                )

            media_payload = {}
            if "png" in results and Path(results["png"]).exists():
                media_payload["Strategy/Landscape_Progression_PNG"] = wandb.Image(
                    str(results["png"]), caption=f"AlphaStar Strategy Landscape (Gen {step})"
                )
            if "gif" in results and Path(results["gif"]).exists():
                media_payload["Strategy/Progression_Animation_GIF"] = wandb.Video(
                    str(results["gif"]), format="gif", caption=f"Training Trajectory across 8 Z-Index Clusters"
                )
            if "html" in results and Path(results["html"]).exists():
                media_payload["Strategy/Interactive_Dashboard_HTML"] = wandb.Html(
                    str(results["html"])
                )

            if media_payload and wandb.run is not None:
                wandb.log(media_payload, step=step)
                print(f"  [+] Logged AlphaStar strategy progression artifacts to WandB at step {step}!")
        except Exception as e:
            logger.debug(f"Could not upload strategy progression media to WandB: {e}")

    def close(self) -> None:
        if self.single_run is not None:
            self.single_run.close()

