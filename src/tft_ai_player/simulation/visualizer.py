"""Interactive Visual Dashboard and Replay Generator for TFT Simulation."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tft_ai_player.simulation.bots import BaseBot, StandardTempoBot
from tft_ai_player.simulation.config import SetData, get_default_set17_data
from tft_ai_player.simulation.game import TFTGame


class GameRecorder:
    """Records full step-by-step game snapshots for interactive visualization."""

    def __init__(self, game: TFTGame) -> None:
        self.game = game
        self.frames: list[dict[str, Any]] = []

    def capture_snapshot(self, event_type: str = "ROUND_START", combat_results: list[Any] | None = None) -> dict[str, Any]:
        """Record the complete state of all 8 players, shop, pool, stage, and events."""
        rinfo = self.game.stage_manager.get_current_round_info()

        players_data: list[dict[str, Any]] = []
        for p in self.game.players:
            board_units: list[dict[str, Any]] = []
            for (r, c), u in p.board.items():
                board_units.append({
                    "row": r,
                    "col": c,
                    "champion_id": u.champion_id,
                    "name": u.champion_id.replace("TFT17_", ""),
                    "cost": u.cost,
                    "star_level": u.star_level,
                    "items": list(u.items),
                })

            bench_units: list[dict[str, Any] | None] = []
            for slot_idx, u in enumerate(p.bench):
                if u is not None:
                    bench_units.append({
                        "slot": slot_idx,
                        "champion_id": u.champion_id,
                        "name": u.champion_id.replace("TFT17_", ""),
                        "cost": u.cost,
                        "star_level": u.star_level,
                        "items": list(u.items),
                    })
                else:
                    bench_units.append(None)

            item_bench_data = [
                {"item_id": it.item_id, "name": it.name, "is_component": it.is_component}
                for it in p.item_bench
            ]

            shop_data = [
                {
                    "slot": idx,
                    "champion_id": card,
                    "name": card.replace("TFT17_", "") if card else None,
                    "cost": self.game.set_data.champions[card].cost if card and card in self.game.set_data.champions else 1,
                }
                if card else None
                for idx, card in enumerate(p.shop.slots)
            ]

            players_data.append({
                "player_id": p.player_id,
                "name": p.name,
                "health": p.health,
                "gold": p.gold,
                "level": p.level,
                "exp": p.exp,
                "max_exp": self.game.set_data.level_exp.get(p.level, 0),
                "streak": p.streak,
                "alive": p.alive,
                "placement": p.placement,
                "board_value": p.get_board_value(),
                "active_traits": p.get_active_traits(),
                "board": board_units,
                "bench": bench_units,
                "item_bench": item_bench_data,
                "shop": shop_data,
                "shop_locked": p.shop.locked,
            })

        combats_data: list[dict[str, Any]] = []
        if combat_results:
            for res in combat_results:
                combats_data.append({
                    "winner_id": res.winner_id,
                    "loser_id": res.loser_id,
                    "damage_dealt": res.damage_dealt,
                    "win_prob_a": res.win_prob_a,
                    "surviving_units": res.surviving_units,
                    "is_ghost_b": res.is_ghost_b,
                })

        snapshot = {
            "round_index": self.game.stage_manager.total_rounds_elapsed,
            "stage": rinfo.stage,
            "round_in_stage": rinfo.round_in_stage,
            "stage_str": rinfo.stage_str,
            "round_type": rinfo.round_type.value,
            "event_type": event_type,
            "is_over": self.game.is_over,
            "players": players_data,
            "combats": combats_data,
            "rankings": [{"placement": rank, "player_id": p.player_id, "name": p.name} for rank, p in self.game.get_rankings()],
        }

        self.frames.append(snapshot)
        return snapshot


def generate_visual_html(replay_data: list[dict[str, Any]], title: str = "TFT AI Simulation Dashboard") -> str:
    """Generate self-contained, responsive HTML5/CSS/JS visual interactive dashboard."""
    replay_json = json.dumps(replay_data, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
:root {{
  --bg-primary: #0a0d14;
  --bg-card: rgba(18, 24, 38, 0.85);
  --bg-card-hover: rgba(28, 36, 56, 0.95);
  --border-color: rgba(255, 255, 255, 0.08);
  --accent-blue: #3b82f6;
  --accent-cyan: #06b6d4;
  --accent-gold: #f59e0b;
  --accent-purple: #a855f7;
  --accent-green: #10b981;
  --accent-red: #ef4444;
  --text-main: #f8fafc;
  --text-muted: #94a3b8;
  --cost-1: #94a3b8;
  --cost-2: #22c55e;
  --cost-3: #3b82f6;
  --cost-4: #c084fc;
  --cost-5: #fbbf24;
}}

* {{
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}}

body {{
  font-family: 'Outfit', sans-serif;
  background-color: var(--bg-primary);
  color: var(--text-main);
  min-height: 100vh;
  padding: 20px;
  background-image: 
    radial-gradient(circle at 15% 15%, rgba(59, 130, 246, 0.08) 0%, transparent 40%),
    radial-gradient(circle at 85% 85%, rgba(168, 85, 247, 0.08) 0%, transparent 40%);
}}

.app-container {{
  max-width: 1500px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 20px;
}}

/* Top Navigation Bar */
.top-nav {{
  background: var(--bg-card);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border-color);
  border-radius: 16px;
  padding: 16px 24px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 16px;
}}

.logo-group {{
  display: flex;
  align-items: center;
  gap: 12px;
}}

.logo-badge {{
  background: linear-gradient(135deg, #3b82f6, #8b5cf6);
  padding: 8px 14px;
  border-radius: 10px;
  font-weight: 800;
  font-size: 1.1rem;
  letter-spacing: 1px;
}}

.title-text h1 {{
  font-size: 1.3rem;
  font-weight: 700;
}}

.title-text p {{
  font-size: 0.85rem;
  color: var(--text-muted);
}}

.stage-badge {{
  display: flex;
  align-items: center;
  gap: 12px;
  background: rgba(0, 0, 0, 0.4);
  padding: 8px 16px;
  border-radius: 12px;
  border: 1px solid var(--border-color);
}}

.stage-label {{
  font-size: 0.75rem;
  color: var(--text-muted);
  text-transform: uppercase;
}}

.stage-val {{
  font-size: 1.2rem;
  font-weight: 800;
  color: var(--accent-cyan);
}}

/* Controls */
.controls-group {{
  display: flex;
  align-items: center;
  gap: 10px;
}}

.btn {{
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid var(--border-color);
  color: var(--text-main);
  padding: 8px 16px;
  border-radius: 10px;
  font-family: inherit;
  font-weight: 600;
  font-size: 0.9rem;
  cursor: pointer;
  transition: all 0.2s ease;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}}

.btn:hover {{
  background: rgba(255, 255, 255, 0.12);
  border-color: var(--accent-blue);
  transform: translateY(-1px);
}}

.btn-primary {{
  background: linear-gradient(135deg, #2563eb, #7c3aed);
  border: none;
}}

.btn-primary:hover {{
  background: linear-gradient(135deg, #1d4ed8, #6d28d9);
  box-shadow: 0 4px 14px rgba(59, 130, 246, 0.4);
}}

/* Main Layout Grid */
.main-grid {{
  display: grid;
  grid-template-columns: 360px 1fr;
  gap: 20px;
}}

@media (max-width: 1024px) {{
  .main-grid {{
    grid-template-columns: 1fr;
  }}
}}

/* Left Sidebar: Scoreboard */
.sidebar-panel {{
  background: var(--bg-card);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border-color);
  border-radius: 16px;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}}

.panel-header {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid var(--border-color);
  padding-bottom: 12px;
}}

.panel-header h2 {{
  font-size: 1.1rem;
  font-weight: 700;
}}

.player-list {{
  display: flex;
  flex-direction: column;
  gap: 8px;
}}

.player-card {{
  background: rgba(0, 0, 0, 0.25);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 12px;
  cursor: pointer;
  transition: all 0.2s ease;
  position: relative;
  overflow: hidden;
}}

.player-card:hover {{
  background: var(--bg-card-hover);
  border-color: var(--accent-blue);
}}

.player-card.active {{
  border-color: var(--accent-cyan);
  background: rgba(6, 182, 212, 0.08);
  box-shadow: 0 0 16px rgba(6, 182, 212, 0.2);
}}

.player-card.dead {{
  opacity: 0.5;
  filter: grayscale(80%);
}}

.card-top {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}}

.player-badge {{
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 700;
  font-size: 0.95rem;
}}

.rank-pill {{
  background: rgba(255, 255, 255, 0.1);
  padding: 2px 6px;
  border-radius: 6px;
  font-size: 0.75rem;
}}

.rank-top4 {{
  background: rgba(16, 185, 129, 0.2);
  color: #34d399;
}}

.hp-bar-bg {{
  height: 6px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 3px;
  overflow: hidden;
  margin-bottom: 8px;
}}

.hp-bar-fill {{
  height: 100%;
  background: var(--accent-green);
  transition: width 0.3s ease;
}}

.card-stats {{
  display: flex;
  justify-content: space-between;
  font-size: 0.8rem;
  color: var(--text-muted);
  font-family: 'JetBrains Mono', monospace;
}}

/* Right Board View Area */
.view-panel {{
  display: flex;
  flex-direction: column;
  gap: 20px;
}}

.scout-header {{
  background: var(--bg-card);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border-color);
  border-radius: 16px;
  padding: 16px 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
}}

.scout-title {{
  display: flex;
  align-items: center;
  gap: 10px;
}}

.scout-title h3 {{
  font-size: 1.2rem;
  font-weight: 700;
}}

.econ-chips {{
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}}

.chip {{
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid var(--border-color);
  padding: 6px 12px;
  border-radius: 8px;
  font-size: 0.85rem;
  display: flex;
  align-items: center;
  gap: 6px;
  font-family: 'JetBrains Mono', monospace;
}}

/* Board Grid (4 Rows x 7 Cols) */
.board-container {{
  background: var(--bg-card);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border-color);
  border-radius: 16px;
  padding: 24px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
}}

.board-row {{
  display: flex;
  gap: 10px;
}}

.board-row:nth-child(even) {{
  margin-left: 36px;
}}

.hex-cell {{
  width: 76px;
  height: 86px;
  background: rgba(15, 23, 42, 0.6);
  border: 2px solid rgba(255, 255, 255, 0.08);
  border-radius: 12px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: space-between;
  padding: 6px;
  transition: all 0.2s ease;
  position: relative;
}}

.hex-cell.occupied {{
  background: rgba(30, 41, 59, 0.85);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}}

.hex-cell.cost-1 {{ border-color: var(--cost-1); }}
.hex-cell.cost-2 {{ border-color: var(--cost-2); }}
.hex-cell.cost-3 {{ border-color: var(--cost-3); }}
.hex-cell.cost-4 {{ border-color: var(--cost-4); }}
.hex-cell.cost-5 {{ border-color: var(--cost-5); }}

.unit-stars {{
  font-size: 0.75rem;
  color: var(--accent-gold);
  font-weight: 700;
  letter-spacing: 1px;
}}

.unit-name {{
  font-size: 0.75rem;
  font-weight: 700;
  text-align: center;
  line-height: 1.1;
  word-break: break-word;
}}

.unit-items {{
  display: flex;
  gap: 2px;
}}

.item-dot {{
  width: 8px;
  height: 8px;
  border-radius: 2px;
  background: var(--accent-cyan);
}}

/* Bench & Items */
.lower-deck {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}}

@media (max-width: 800px) {{
  .lower-deck {{
    grid-template-columns: 1fr;
  }}
}}

.deck-panel {{
  background: var(--bg-card);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border-color);
  border-radius: 16px;
  padding: 16px 20px;
}}

.deck-title {{
  font-size: 0.95rem;
  font-weight: 700;
  margin-bottom: 12px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}}

.bench-grid {{
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}}

.bench-slot {{
  width: 58px;
  height: 66px;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid var(--border-color);
  border-radius: 10px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  font-size: 0.75rem;
  padding: 4px;
}}

/* Shop */
.shop-panel {{
  background: var(--bg-card);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border-color);
  border-radius: 16px;
  padding: 16px 20px;
}}

.shop-cards {{
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 10px;
}}

.shop-card {{
  background: rgba(0, 0, 0, 0.4);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 12px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  min-height: 80px;
}}

.shop-card.cost-1 {{ border-left: 4px solid var(--cost-1); }}
.shop-card.cost-2 {{ border-left: 4px solid var(--cost-2); }}
.shop-card.cost-3 {{ border-left: 4px solid var(--cost-3); }}
.shop-card.cost-4 {{ border-left: 4px solid var(--cost-4); }}
.shop-card.cost-5 {{ border-left: 4px solid var(--cost-5); }}

.shop-name {{
  font-weight: 700;
  font-size: 0.9rem;
}}

.shop-cost {{
  font-size: 0.8rem;
  color: var(--accent-gold);
  font-family: 'JetBrains Mono', monospace;
  font-weight: 600;
}}

/* Combat Feed */
.combat-feed {{
  background: var(--bg-card);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border-color);
  border-radius: 16px;
  padding: 16px 20px;
}}

.combat-log {{
  display: flex;
  flex-direction: column;
  gap: 8px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.85rem;
}}

.combat-item {{
  background: rgba(0, 0, 0, 0.3);
  border-radius: 8px;
  padding: 8px 12px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}}

.win-tag {{ color: var(--accent-green); font-weight: 700; }}
.loss-tag {{ color: var(--accent-red); font-weight: 700; }}
</style>
</head>
<body>

<div class="app-container">
  <!-- Top Navigation Bar -->
  <header class="top-nav">
    <div class="logo-group">
      <div class="logo-badge">TFT SIM</div>
      <div class="title-text">
        <h1>Autonomous Player Dashboard</h1>
        <p>Set 17 Multi-Agent RL Simulation & Combat Engine</p>
      </div>
    </div>

    <div class="stage-badge">
      <div>
        <div class="stage-label">Current Stage</div>
        <div id="stage-display" class="stage-val">1-1 (CAROUSEL)</div>
      </div>
      <div>
        <div class="stage-label">Round</div>
        <div id="round-display" class="stage-val" style="color: var(--text-main);">#1</div>
      </div>
    </div>

    <div class="controls-group">
      <button id="btn-prev" class="btn">⏮ Prev</button>
      <button id="btn-play" class="btn btn-primary">▶ Play</button>
      <button id="btn-next" class="btn">Next ⏭</button>
      <input type="range" id="speed-slider" min="50" max="1500" value="400" style="accent-color: var(--accent-blue); width: 90px;">
    </div>
  </header>

  <!-- Main Grid -->
  <div class="main-grid">
    <!-- Left Sidebar: 8-Player Lobby -->
    <aside class="sidebar-panel">
      <div class="panel-header">
        <h2>Lobby Standings</h2>
        <span id="alive-count" class="badge" style="font-size: 0.8rem; color: var(--accent-green);">8 Alive</span>
      </div>
      <div id="player-list" class="player-list"></div>
    </aside>

    <!-- Right: Fielded Board View of Scouted Player -->
    <main class="view-panel">
      <!-- Player Header Info -->
      <div class="scout-header">
        <div class="scout-title">
          <h3 id="scouted-player-name">Player 0 (Focal Agent)</h3>
          <span id="scouted-status-badge" class="chip" style="color: var(--accent-green);">ALIVE</span>
        </div>

        <div class="econ-chips">
          <div class="chip">❤️ <span id="scout-hp">100</span> HP</div>
          <div class="chip">💰 <span id="scout-gold">2</span> Gold</div>
          <div class="chip">⭐ Level <span id="scout-level">1</span> (<span id="scout-xp">0/2</span> XP)</div>
          <div class="chip">🔥 Streak <span id="scout-streak">0</span></div>
          <div class="chip">💎 Board Value: <span id="scout-board-val">0</span>g</div>
        </div>
      </div>

      <!-- Hex Board (4x7) -->
      <div class="board-container">
        <div id="board-grid" style="display: flex; flex-direction: column; gap: 8px;"></div>
      </div>

      <!-- Lower Deck: Bench & Items -->
      <div class="lower-deck">
        <div class="deck-panel">
          <div class="deck-title">Champion Bench (9 Slots)</div>
          <div id="bench-slots" class="bench-grid"></div>
        </div>

        <div class="deck-panel">
          <div class="deck-title">Item Bench (10 Slots)</div>
          <div id="item-slots" class="bench-grid"></div>
        </div>
      </div>

      <!-- Shop -->
      <div class="shop-panel">
        <div class="panel-header" style="margin-bottom: 12px;">
          <div class="deck-title" style="margin: 0;">Shop Offerings (2g Reroll)</div>
          <span id="shop-lock-status" class="chip" style="font-size: 0.75rem;">🔓 Unlocked</span>
        </div>
        <div id="shop-cards" class="shop-cards"></div>
      </div>

      <!-- Combat Feed -->
      <div class="combat-feed">
        <div class="panel-header" style="margin-bottom: 12px;">
          <h2>Latest Round Combat Outcomes</h2>
        </div>
        <div id="combat-log" class="combat-log"></div>
      </div>
    </main>
  </div>
</div>

<script>
const frames = {replay_json};
let currentFrameIdx = 0;
let scoutedPlayerId = 0;
let isPlaying = false;
let playTimer = null;

function renderFrame(idx) {{
  if (idx < 0 || idx >= frames.length) return;
  currentFrameIdx = idx;
  const frame = frames[idx];

  // Top Nav
  document.getElementById("stage-display").textContent = `${{frame.stage_str}} (${{frame.round_type}})`;
  document.getElementById("round-display").textContent = `#${{frame.round_index + 1}}`;

  // Players List
  const playerListEl = document.getElementById("player-list");
  playerListEl.innerHTML = "";

  const alivePlayers = frame.players.filter(p => p.alive);
  document.getElementById("alive-count").textContent = `${{alivePlayers.length}} Alive`;

  frame.players.forEach(p => {{
    const card = document.createElement("div");
    card.className = `player-card ${{p.player_id === scoutedPlayerId ? 'active' : ''}} ${{!p.alive ? 'dead' : ''}}`;
    card.onclick = () => {{
      scoutedPlayerId = p.player_id;
      renderFrame(currentFrameIdx);
    }};

    const hpPercent = Math.max(0, Math.min(100, p.health));
    const hpColor = hpPercent > 50 ? 'var(--accent-green)' : (hpPercent > 20 ? 'var(--accent-gold)' : 'var(--accent-red)');
    const rankPill = p.alive ? `<span class="rank-pill rank-top4">Rank #1</span>` : `<span class="rank-pill" style="background: rgba(239,68,68,0.2); color:#f87171;">#${{p.placement}}</span>`;

    card.innerHTML = `
      <div class="card-top">
        <div class="player-badge">
          <span>${{p.name}}</span>
          ${{!p.alive ? rankPill : ''}}
        </div>
        <div style="font-weight: 700; font-family: 'JetBrains Mono'; color: ${{hpColor}};">${{p.health}} HP</div>
      </div>
      <div class="hp-bar-bg">
        <div class="hp-bar-fill" style="width: ${{hpPercent}}%; background: ${{hpColor}};"></div>
      </div>
      <div class="card-stats">
        <span>💰 ${{p.gold}}g</span>
        <span>⭐ Lv ${{p.level}}</span>
        <span>♟️ ${{p.board.length}}u (${{p.board_value}}g)</span>
      </div>
    `;
    playerListEl.appendChild(card);
  }});

  // Scouted Player View
  const scouted = frame.players.find(p => p.player_id === scoutedPlayerId) || frame.players[0];
  document.getElementById("scouted-player-name").textContent = scouted.name;
  document.getElementById("scouted-status-badge").textContent = scouted.alive ? 'ALIVE' : `ELIMINATED (#${{scouted.placement}})`;
  document.getElementById("scouted-status-badge").style.color = scouted.alive ? 'var(--accent-green)' : 'var(--accent-red)';
  document.getElementById("scout-hp").textContent = scouted.health;
  document.getElementById("scout-gold").textContent = scouted.gold;
  document.getElementById("scout-level").textContent = scouted.level;
  document.getElementById("scout-xp").textContent = `${{scouted.exp}}/${{scouted.max_exp || 'MAX'}}`;
  document.getElementById("scout-streak").textContent = scouted.streak >= 0 ? `+${{scouted.streak}}` : scouted.streak;
  document.getElementById("scout-board-val").textContent = scouted.board_value;

  // Board (4x7)
  const boardGridEl = document.getElementById("board-grid");
  boardGridEl.innerHTML = "";

  const boardMap = {{}};
  scouted.board.forEach(u => {{
    boardMap[`${{u.row}},${{u.col}}`] = u;
  }});

  for (let r = 0; r < 4; r++) {{
    const rowEl = document.createElement("div");
    rowEl.className = "board-row";
    for (let c = 0; c < 7; c++) {{
      const cell = document.createElement("div");
      const unit = boardMap[`${{r}},${{c}}`];

      if (unit) {{
        cell.className = `hex-cell occupied cost-${{unit.cost}}`;
        const starStr = "★".repeat(unit.star_level);
        const itemDots = unit.items.map(it => `<div class="item-dot" title="${{it}}"></div>`).join("");
        cell.innerHTML = `
          <div class="unit-stars">${{starStr}}</div>
          <div class="unit-name">${{unit.name}}</div>
          <div class="unit-items">${{itemDots}}</div>
        `;
      }} else {{
        cell.className = "hex-cell";
        cell.innerHTML = `<span style="color: rgba(255,255,255,0.05); font-size: 0.7rem;">${{r}},${{c}}</span>`;
      }}
      rowEl.appendChild(cell);
    }}
    boardGridEl.appendChild(rowEl);
  }}

  // Bench (9 slots)
  const benchEl = document.getElementById("bench-slots");
  benchEl.innerHTML = "";
  for (let bIdx = 0; bIdx < 9; bIdx++) {{
    const bSlot = document.createElement("div");
    bSlot.className = "bench-slot";
    const u = scouted.bench[bIdx];
    if (u) {{
      bSlot.style.borderColor = `var(--cost-${{u.cost}})`;
      bSlot.innerHTML = `
        <div style="color: var(--accent-gold); font-size: 0.65rem;">${{"★".repeat(u.star_level)}}</div>
        <div style="font-weight: 700; font-size: 0.75rem;">${{u.name}}</div>
        <div style="font-size: 0.65rem; color: var(--text-muted);">${{u.cost}}g</div>
      `;
    }} else {{
      bSlot.innerHTML = `<span style="color: rgba(255,255,255,0.15);">${{bIdx}}</span>`;
    }}
    benchEl.appendChild(bSlot);
  }}

  // Item Bench (10 slots)
  const itemEl = document.getElementById("item-slots");
  itemEl.innerHTML = "";
  for (let iIdx = 0; iIdx < 10; iIdx++) {{
    const itSlot = document.createElement("div");
    itSlot.className = "bench-slot";
    const it = scouted.item_bench[iIdx];
    if (it) {{
      itSlot.style.borderColor = "var(--accent-cyan)";
      const shortName = it.name.replace("TFT_Item_", "").slice(0, 10);
      itSlot.innerHTML = `<div style="font-size: 0.7rem; font-weight: 600; text-align: center;">${{shortName}}</div>`;
    }} else {{
      itSlot.innerHTML = `<span style="color: rgba(255,255,255,0.15);">${{iIdx}}</span>`;
    }}
    itemEl.appendChild(itSlot);
  }}

  // Shop
  const shopEl = document.getElementById("shop-cards");
  shopEl.innerHTML = "";
  document.getElementById("shop-lock-status").textContent = scouted.shop_locked ? "🔒 Locked" : "🔓 Unlocked";

  scouted.shop.forEach(card => {{
    const cardEl = document.createElement("div");
    if (card) {{
      cardEl.className = `shop-card cost-${{card.cost}}`;
      cardEl.innerHTML = `
        <div class="shop-name">${{card.name}}</div>
        <div class="shop-cost">${{card.cost}} Gold</div>
      `;
    }} else {{
      cardEl.className = "shop-card";
      cardEl.style.opacity = "0.3";
      cardEl.innerHTML = `<div style="color: var(--text-muted); font-size: 0.8rem;">(Empty)</div>`;
    }}
    shopEl.appendChild(cardEl);
  }});

  // Combat Log
  const combatLogEl = document.getElementById("combat-log");
  combatLogEl.innerHTML = "";
  if (frame.combats && frame.combats.length > 0) {{
    frame.combats.forEach(c => {{
      const item = document.createElement("div");
      item.className = "combat-item";
      const ghostStr = c.is_ghost_b ? " (Ghost)" : "";
      const pWinStr = (c.win_prob_a * 100).toFixed(1);
      item.innerHTML = `
        <div>⚔️ <strong style="color: #60a5fa;">Player ${{c.winner_id}}</strong> defeated <strong style="color: #f87171;">Player ${{c.loser_id}}${{ghostStr}}</strong></div>
        <div style="display: flex; gap: 12px;">
          <span style="color: var(--accent-gold);">-${{c.damage_dealt}} HP (${{c.surviving_units}} survived)</span>
          <span style="color: var(--text-muted);">P(Win): ${{pWinStr}}%</span>
        </div>
      `;
      combatLogEl.appendChild(item);
    }});
  }} else {{
    combatLogEl.innerHTML = `<div style="color: var(--text-muted); padding: 8px;">No combat battles occurred during this phase (${{frame.round_type}}).</div>`;
  }}
}}

// Control Listeners
document.getElementById("btn-prev").onclick = () => {{
  if (currentFrameIdx > 0) renderFrame(currentFrameIdx - 1);
}};

document.getElementById("btn-next").onclick = () => {{
  if (currentFrameIdx < frames.length - 1) renderFrame(currentFrameIdx + 1);
}};

const playBtn = document.getElementById("btn-play");
playBtn.onclick = () => {{
  isPlaying = !isPlaying;
  playBtn.textContent = isPlaying ? "⏸ Pause" : "▶ Play";
  if (isPlaying) {{
    const speed = 1600 - document.getElementById("speed-slider").value;
    playTimer = setInterval(() => {{
      if (currentFrameIdx < frames.length - 1) {{
        renderFrame(currentFrameIdx + 1);
      }} else {{
        isPlaying = false;
        playBtn.textContent = "▶ Play";
        clearInterval(playTimer);
      }}
    }}, speed);
  }} else {{
    clearInterval(playTimer);
  }}
}};

// Initial Render
renderFrame(0);
</script>
</body>
</html>
"""
