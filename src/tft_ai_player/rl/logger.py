"""Weights & Biases (WandB) experiment tracking and multi-agent metric logger for TFT RL."""

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


class WandBSingleRun:
    """Represents a single autonomous WandB run within a multi-agent league group."""

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

        # Initialize run via WandB GraphQL UpsertBucket
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
        """Stream metrics to WandB FileStream API and update summaryMetrics index."""
        try:
            payload = dict(metrics)
            payload["_step"] = step
            payload["_runtime"] = time.time()
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

            # Update summaryMetrics via GraphQL on every step so WandB UI dropdowns index all metric keys
            self._update_summary(payload)
        except Exception as e:
            logger.debug(f"Direct WandB stream error for {self.run_id}: {e}")

    def _update_summary(self, summary_dict: dict[str, Any]) -> None:
        """Update summaryMetrics in WandB to populate UI panel selectors."""
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
        """Mark run complete."""
        try:
            basic_auth = base64.b64encode(f"api:{self.api_key}".encode("utf-8")).decode("ascii")
            fs_headers = {
                "Content-Type": "application/json",
                "Authorization": f"Basic {basic_auth}",
            }
            fs_payload = {
                "complete": True,
                "exitcode": 0,
            }
            fs_req = urllib.request.Request(
                f"https://api.wandb.ai/files/{self.entity}/{self.project}/{self.run_id}/file_stream",
                data=json.dumps(fs_payload).encode("utf-8"),
                headers=fs_headers,
            )
            urllib.request.urlopen(fs_req, timeout=5)
        except Exception:
            pass


class WandBLogger:
    """Manages grouped multi-agent WandB runs for Tri-Tier RL League.

    Creates 3 parallel runs under the same Group container (run-name):
    1. <run-name>_main_agent       (Role: Main Agent / Generalist)
    2. <run-name>_main_exploiter   (Role: Main Exploiter / Hyper-Roll)
    3. <run-name>_league_exploiter (Role: League Exploiter / Fast-8/9)
    """

    def __init__(
        self,
        project: str = "tft-ai-league",
        entity: str | None = None,
        group: str | None = None,
        run_name: str | None = None,
        run_id: str | None = None,
        config: dict[str, Any] | None = None,
        resume: str | None = None,
        enabled: bool = True,
    ) -> None:
        self.project = project
        self.group = group or run_name or "ppo_tri_tier_league_v1"
        self.run_name = run_name or self.group
        self.run_id = run_id or self.run_name
        self.enabled = enabled
        self.api_key: str | None = None
        self.entity: str = entity or ""
        self.agent_runs: dict[str, WandBSingleRun] = {}
        self.group_url: str | None = None

        if not self.enabled:
            return

        try:
            self._resolve_credentials()
            self._init_tri_tier_runs(base_config=config or {})
        except Exception as e:
            logger.warning(f"Could not initialize WandB Grouped Runs: {e}. Running in local logging mode.")
            self.enabled = False

    def _resolve_credentials(self) -> None:
        """Resolve WandB API key and entity."""
        self.api_key = os.environ.get("WANDB_API_KEY")
        if not self.api_key:
            netrc_paths = [
                os.path.expanduser("~/_netrc"),
                os.path.expanduser("~/.netrc"),
            ]
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

        # Query entity from API
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

    def _init_tri_tier_runs(self, base_config: dict[str, Any]) -> None:
        """Initialize the 3 parallel runs grouped under self.group."""
        agents_spec = [
            ("Main_Agent", "main_agent", "Main Agent (Generalist)", "generalist"),
            ("Main_Exploiter", "main_exploiter", "Main Exploiter (Hyper-Roll)", "hyper_roll"),
            ("League_Exploiter", "league_exploiter", "League Exploiter (Fast-8/9)", "fast8_flex"),
        ]

        print(f"  [WandB] Project: {self.project} | Group: {self.group}")
        for agent_key, suffix, display_suffix, archetype in agents_spec:
            agent_run_id = f"{self.run_name}_{suffix}"
            agent_display_name = f"{self.run_name} ({display_suffix})"
            agent_cfg = dict(base_config)
            agent_cfg["agent_name"] = agent_key
            agent_cfg["archetype"] = archetype
            agent_cfg["league_group"] = self.group

            run_obj = WandBSingleRun(
                project=self.project,
                entity=self.entity,
                group=self.group,
                display_name=agent_display_name,
                run_id=agent_run_id,
                config=agent_cfg,
                api_key=self.api_key,
            )
            self.agent_runs[agent_key] = run_obj
            print(f"  [WandB Run] {agent_key:<16}: {run_obj.dashboard_url}")

        self.group_url = f"https://wandb.ai/{self.entity}/{self.project}/groups/{self.group}"
        print(f"  [WandB Group View] Overlaid Charts Dashboard: {self.group_url}")

    def log_generation(self, generation: int, metrics: dict[str, Any]) -> None:
        """Log generation metrics individually to each of the 3 grouped runs."""
        if not self.enabled or not self.agent_runs:
            return

        tri_tier = metrics.get("tri_tier", {})
        if not isinstance(tri_tier, dict):
            return

        for agent_key, agent_run in self.agent_runs.items():
            agent_data = tri_tier.get(agent_key)
            if not isinstance(agent_data, dict):
                continue

            payload: dict[str, Any] = {
                "generation": generation,
            }

            # 1. Performance & Value
            if "elo" in agent_data and agent_data["elo"] is not None:
                payload["Performance/Elo"] = float(agent_data["elo"])
            if "mean_reward" in agent_data and agent_data["mean_reward"] is not None:
                payload["Performance/Mean_Reward"] = float(agent_data["mean_reward"])
            if "avg_placement" in agent_data and agent_data["avg_placement"] is not None:
                payload["Performance/Avg_Placement"] = float(agent_data["avg_placement"])

            # 2. Optimization Losses & Entropy
            if "loss" in agent_data and agent_data["loss"] is not None:
                payload["Optimization/Total_Loss"] = float(agent_data["loss"])
            if "policy_loss" in agent_data and agent_data["policy_loss"] is not None:
                payload["Optimization/Policy_Loss"] = float(agent_data["policy_loss"])
            if "value_loss" in agent_data and agent_data["value_loss"] is not None:
                payload["Optimization/Value_Loss"] = float(agent_data["value_loss"])
            if "entropy" in agent_data and agent_data["entropy"] is not None:
                payload["Optimization/Entropy"] = float(agent_data["entropy"])
            if "explained_variance" in agent_data and agent_data["explained_variance"] is not None:
                payload["Optimization/Explained_Variance"] = float(agent_data["explained_variance"])

            # 3. Action Distributions
            act_dist = agent_data.get("action_distribution")
            if isinstance(act_dist, dict):
                for act_name, pct in act_dist.items():
                    payload[f"Action_Dist/{act_name}"] = float(pct) * 100.0

            # 4. Turn Efficiency
            if "actions_per_round" in agent_data:
                payload["Turn_Efficiency/APM"] = float(agent_data["actions_per_round"])
            if "pass_clean_econ_rate" in agent_data:
                payload["Turn_Efficiency/Clean_Econ_Pass_Rate"] = float(agent_data["pass_clean_econ_rate"]) * 100.0
            if "pass_missed_craft_rate" in agent_data:
                payload["Turn_Efficiency/Missed_Craft_Rate"] = float(agent_data["pass_missed_craft_rate"]) * 100.0
            if "pass_missed_upgrade_rate" in agent_data:
                payload["Turn_Efficiency/Missed_Upgrade_Rate"] = float(agent_data["pass_missed_upgrade_rate"]) * 100.0

            # 5. Reward Blocks
            for blk in ["block_combat_outcome", "block_board_power", "block_constraints_economy", "block_economy", "block_experience", "block_board_building", "block_combat", "block_placement"]:
                if blk in agent_data:
                    payload[f"Rewards_Blocks/{blk.replace('block_', '').title()}"] = float(agent_data[blk])

            # 6. Micro-Rewards
            for rew in ["rew_round_win", "rew_round_loss", "rew_potential_delta", "rew_elimination_bounty", "rew_interest", "rew_pair_shop_buy", "rew_level_up", "rew_star_2", "rew_star_3", "rew_item_slam", "rew_synergy_tier", "rew_hp_loss", "rew_stage_survival", "rew_placement"]:
                if rew in agent_data:
                    payload[f"Rewards_Micro/{rew.replace('rew_', '').title()}"] = float(agent_data[rew])

            # 7. Standardized Static Benchmark Ladder (Logged to Main Agent)
            if agent_key == "Main_Agent":
                bench_keys = [
                    ("Benchmark_Placement/Tier1_Random", "bench_Tier1_Random_placement"),
                    ("Benchmark_WinRate/Tier1_Random", "bench_Tier1_Random_win_rate"),
                    ("Benchmark_Placement/Tier2_Banker", "bench_Tier2_Banker_placement"),
                    ("Benchmark_WinRate/Tier2_Banker", "bench_Tier2_Banker_win_rate"),
                    ("Benchmark_Placement/Tier3_Tempo", "bench_Tier3_Tempo_placement"),
                    ("Benchmark_WinRate/Tier3_Tempo", "bench_Tier3_Tempo_win_rate"),
                    ("Benchmark_Placement/Tier4_Exploiters", "bench_Tier4_Exploiters_placement"),
                    ("Benchmark_WinRate/Tier4_Exploiters", "bench_Tier4_Exploiters_win_rate"),
                ]
                for tag, key in bench_keys:
                    if key in metrics and metrics[key] is not None:
                        payload[tag] = float(metrics[key])

            # Stream payload to this agent's run
            agent_run.log(payload, step=generation)

    def close(self) -> None:
        """Finish and close all 3 grouped runs."""
        for run_obj in self.agent_runs.values():
            run_obj.close()

