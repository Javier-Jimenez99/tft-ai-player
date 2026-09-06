"""Interactive network graph visualizer and multi-panel plot generator for TFT player lobbies."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from .player_graph import PlayerGraph, PlayerNode

# TFT Rank Colors Palette
TIER_COLORS: dict[str, str] = {
    "CHALLENGER": "#F4C430",   # Radiant Gold
    "GRANDMASTER": "#E23636",  # Crimson Red
    "MASTER": "#9B59B6",       # Arcane Purple
    "DIAMOND": "#3498DB",      # Crystal Sapphire Blue
    "EMERALD": "#2ECC71",      # Vivid Emerald Green
    "PLATINUM": "#1ABC9C",     # Teal Cyan
    "GOLD": "#F39C12",         # Warm Amber Gold
    "SILVER": "#BDC3C7",       # Mist Silver
    "BRONZE": "#CD7F32",       # Copper Bronze
    "IRON": "#7F8C8D",         # Slate Iron Gray
    "UNKNOWN": "#95A5A6",      # Neutral Gray
}


def build_networkx_graph(graph: PlayerGraph) -> nx.Graph:
    """Construct a NetworkX graph from PlayerGraph nodes and lobby edges."""
    G = nx.Graph()
    for node in graph.nodes.values():
        G.add_node(
            node.riot_id,
            tier=node.tier,
            rank_text=node.rank_text,
            region=node.region,
            depth=node.depth,
            scanned=node.scanned,
            is_app_user=node.is_app_user,
            app_matches=node.app_matches,
            color=TIER_COLORS.get(node.tier, "#95A5A6"),
        )

    for edge in graph.edges:
        if edge.player_a in graph.nodes and edge.player_b in graph.nodes:
            G.add_edge(edge.player_a, edge.player_b, match_uuid=edge.match_uuid)

    return G


def compute_constellation_layout(G: nx.Graph, graph: PlayerGraph) -> dict[str, tuple[float, float]]:
    """Compute an organic constellation layout of match lobbies without artificial boundary circles.

    In a standard spring layout, isolated/unscanned nodes have no attractive edges and get repelled
    by the dense central cluster to the perimeter of the unit circle.
    This function computes local physics per connected lobby component and packs them as celestial
    lobby clusters across the plane.
    """
    import math

    pos: dict[str, tuple[float, float]] = {}
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    if not comps:
        return pos

    # 1. Layout connected components of size >= 2
    major_comps = [c for c in comps if len(c) >= 2]
    angle_step = 2.0 * math.pi / max(1, len(major_comps) - 1)

    for i, comp in enumerate(major_comps):
        sub = G.subgraph(comp)
        n_comp = len(comp)

        if i == 0:
            cx, cy = 0.0, 0.0
            scale = 4.5
        else:
            ring_r = 6.5 + math.sqrt(n_comp) * 0.7
            angle = (i - 1) * angle_step
            cx = ring_r * math.cos(angle)
            cy = ring_r * math.sin(angle)
            scale = max(1.5, math.sqrt(n_comp) * 0.55)

        sub_pos = nx.spring_layout(sub, k=1.2 / math.sqrt(max(2, n_comp)), iterations=45, seed=42 + i)
        for node, (lx, ly) in sub_pos.items():
            pos[node] = (cx + lx * scale, cy + ly * scale)

    # 2. Arrange isolated unscanned nodes in a clean peripheral shelf at the bottom
    single_nodes = [list(c)[0] for c in comps if len(c) == 1]
    if single_nodes:
        tier_groups = defaultdict(list)
        for n in single_nodes:
            tier_groups[graph.nodes[n].tier].append(n)

        tiers_order = ("CHALLENGER", "GRANDMASTER", "MASTER", "DIAMOND", "EMERALD", "PLATINUM", "GOLD", "SILVER", "BRONZE", "IRON", "UNKNOWN")
        dock_y = -11.0
        curr_x = -13.0
        for tier in tiers_order:
            t_nodes = tier_groups.get(tier, [])
            for j, node in enumerate(t_nodes[:25]):
                pos[node] = (curr_x + (j % 15) * 0.45, dock_y - (j // 15) * 0.45)
            if t_nodes:
                curr_x += min(15, len(t_nodes)) * 0.45 + 0.6

    return pos


def export_static_plot(graph: PlayerGraph, output_path: Path | str) -> Path:
    """Generate a high-resolution 4-panel publication-ready dashboard image of the graph."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    G = build_networkx_graph(graph)
    total_nodes = len(graph.nodes)
    total_edges = len(graph.edges)
    total_games = len(graph.games)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(20, 14), dpi=150)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1.0], width_ratios=[1.3, 1.0], hspace=0.25, wspace=0.2)

    # Panel 1: Network Graph Visualization
    ax_net = fig.add_subplot(gs[0, :])
    ax_net.set_title(
        f"TFT Player Lobby Network Graph ({total_nodes:,} Players, {total_edges:,} Lobby Connections, {total_games:,} Manifest Games)",
        fontsize=16,
        fontweight="bold",
        pad=15,
        color="#F8F9FA",
    )

    if len(G) > 0:
        pos = compute_constellation_layout(G, graph)

        # Draw edges with low opacity
        nx.draw_networkx_edges(G, pos, ax=ax_net, alpha=0.20, edge_color="#4A5568", width=0.8)

        # Group nodes by tier to draw with tier colors
        for tier, color in TIER_COLORS.items():
            t_nodes = [n for n, d in G.nodes(data=True) if d.get("tier") == tier and n in pos]
            if not t_nodes:
                continue

            app_users = [n for n in t_nodes if G.nodes[n].get("is_app_user")]
            regular_users = [n for n in t_nodes if not G.nodes[n].get("is_app_user")]

            # Draw regular nodes
            if regular_users:
                nx.draw_networkx_nodes(
                    G,
                    pos,
                    nodelist=regular_users,
                    ax=ax_net,
                    node_color=color,
                    node_size=25,
                    alpha=0.75,
                )

            # Draw app users with larger size and golden glow
            if app_users:
                nx.draw_networkx_nodes(
                    G,
                    pos,
                    nodelist=app_users,
                    ax=ax_net,
                    node_color=color,
                    node_size=80,
                    edgecolors="#FFFFFF",
                    linewidths=1.2,
                    alpha=0.95,
                    label=tier,
                )

    ax_net.axis("off")
    # Legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=col, markersize=10, label=t)
        for t, col in TIER_COLORS.items() if t != "UNKNOWN"
    ]
    ax_net.legend(
        handles=handles,
        loc="upper right",
        ncol=5,
        fontsize=9,
        framealpha=0.3,
        facecolor="#1A202C",
        edgecolor="#2D3748",
    )

    # Panel 2: Candidate Games Distribution by League
    ax_games = fig.add_subplot(gs[1, 0])
    games_dist = graph.get_games_tier_distribution()
    tiers_order = ("CHALLENGER", "GRANDMASTER", "MASTER", "DIAMOND", "EMERALD", "PLATINUM", "GOLD", "SILVER", "BRONZE", "IRON")
    game_counts = [games_dist.get(t, 0) for t in tiers_order]
    bar_colors = [TIER_COLORS[t] for t in tiers_order]

    bars = ax_games.barh(tiers_order[::-1], game_counts[::-1], color=bar_colors[::-1], edgecolor="#2D3748", height=0.65)
    ax_games.set_title("Candidate Games per League (1K Cap Target)", fontsize=13, fontweight="bold", pad=10)
    ax_games.set_xlabel("Candidate Matches Cataloged", fontsize=10)
    ax_games.axvline(1000, color="#E74C3C", linestyle="--", linewidth=1.5, label="1,000 Cap Target")
    ax_games.legend(loc="lower right", fontsize=9)
    ax_games.grid(axis="x", alpha=0.15)

    # Add count labels
    for bar in bars:
        width = bar.get_width()
        if width > 0:
            ax_games.text(
                width + 100,
                bar.get_y() + bar.get_height() / 2,
                f"{int(width):,}",
                va="center",
                ha="left",
                fontsize=8.5,
                color="#E2E8F0",
            )

    # Panel 3: Players Scanned vs App Users vs In Queue
    ax_players = fig.add_subplot(gs[1, 1])
    scanned_counts = [sum(1 for n in graph.nodes.values() if n.scanned and n.tier == t) for t in tiers_order]
    app_counts = [sum(1 for n in graph.nodes.values() if n.scanned and n.is_app_user and n.tier == t) for t in tiers_order]
    queued_counts = [sum(1 for n in graph.nodes.values() if not n.scanned and n.tier == t) for t in tiers_order]

    import numpy as np
    y_indices = np.arange(len(tiers_order))
    h = 0.25

    ax_players.barh(y_indices - h, scanned_counts[::-1], height=h, color="#48BB78", label="Scanned Players", edgecolor="#1A202C")
    ax_players.barh(y_indices, app_counts[::-1], height=h, color="#ED8936", label="App Users (Telemetry)", edgecolor="#1A202C")
    ax_players.barh(y_indices + h, queued_counts[::-1], height=h, color="#4299E1", label="In Queue (Unscanned)", edgecolor="#1A202C")

    ax_players.set_yticks(y_indices)
    ax_players.set_yticklabels(tiers_order[::-1], fontsize=9)
    ax_players.set_title("Player Graph Node Status", fontsize=13, fontweight="bold", pad=10)
    ax_players.set_xlabel("Number of Players", fontsize=10)
    ax_players.legend(loc="lower right", fontsize=8.5)
    ax_players.grid(axis="x", alpha=0.15)

    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="#0F172A")
    plt.close(fig)
    return out


def export_interactive_html(graph: PlayerGraph, output_path: Path | str) -> Path:
    """Generate an interactive HTML visualization with Force-Directed Graph physics, search, and filters."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    nodes_json = []
    for node in graph.nodes.values():
        nodes_json.append({
            "id": node.riot_id,
            "label": node.riot_id,
            "tier": node.tier,
            "region": node.region.upper(),
            "rank_text": node.rank_text,
            "depth": node.depth,
            "scanned": node.scanned,
            "is_app_user": node.is_app_user,
            "app_matches": node.app_matches,
            "color": TIER_COLORS.get(node.tier, "#95A5A6"),
            "size": 16 if node.is_app_user else (8 if node.scanned else 4),
        })

    edges_json = []
    seen = set()
    for edge in graph.edges:
        pair = tuple(sorted((edge.player_a, edge.player_b)))
        if pair not in seen:
            seen.add(pair)
            edges_json.append({
                "from": edge.player_a,
                "to": edge.player_b,
                "match_id": edge.match_uuid,
            })

    total_players = len(graph.nodes)
    total_edges = len(edges_json)
    total_games = len(graph.games)
    scanned_cnt = sum(1 for n in graph.nodes.values() if n.scanned)
    app_cnt = sum(1 for n in graph.nodes.values() if n.is_app_user)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TFT Player Lobby Graph Navigator</title>
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', sans-serif;
      background: #0B0F19;
      color: #F3F4F6;
      display: flex;
      height: 100vh;
      overflow: hidden;
    }}
    #sidebar {{
      width: 360px;
      background: #111827;
      border-right: 1px solid #1F2937;
      display: flex;
      flex-direction: column;
      padding: 24px;
      gap: 20px;
      overflow-y: auto;
      z-index: 10;
      box-shadow: 4px 0 24px rgba(0,0,0,0.5);
    }}
    #header h1 {{
      font-size: 20px;
      font-weight: 800;
      background: linear-gradient(135deg, #60A5FA, #A78BFA, #F472B6);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 6px;
    }}
    #header p {{
      font-size: 12px;
      color: #9CA3AF;
    }}
    .stat-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }}
    .stat-card {{
      background: #1F2937;
      border: 1px solid #374151;
      border-radius: 10px;
      padding: 12px;
    }}
    .stat-val {{
      font-size: 18px;
      font-weight: 700;
      color: #60A5FA;
    }}
    .stat-lbl {{
      font-size: 11px;
      color: #9CA3AF;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .control-section {{
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .control-title {{
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: #E5E7EB;
    }}
    #search-box {{
      width: 100%;
      padding: 10px 14px;
      background: #1F2937;
      border: 1px solid #374151;
      border-radius: 8px;
      color: #F9FAFB;
      font-size: 13px;
      outline: none;
      transition: border-color 0.2s;
    }}
    #search-box:focus {{
      border-color: #60A5FA;
    }}
    .tier-pills {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .tier-pill {{
      padding: 4px 10px;
      border-radius: 20px;
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
      border: 1px solid transparent;
      transition: transform 0.1s, opacity 0.2s;
      user-select: none;
    }}
    .tier-pill:hover {{
      transform: scale(1.05);
    }}
    .tier-pill.inactive {{
      opacity: 0.3;
      filter: grayscale(1);
    }}
    #node-details {{
      background: #1F2937;
      border: 1px solid #374151;
      border-radius: 10px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    #node-details h3 {{
      font-size: 15px;
      color: #F3F4F6;
      word-break: break-all;
    }}
    .detail-row {{
      display: flex;
      justify-content: space-between;
      font-size: 12px;
    }}
    .detail-row span:first-child {{
      color: #9CA3AF;
    }}
    .detail-row span:last-child {{
      font-weight: 600;
      color: #E5E7EB;
    }}
    #graph-container {{
      flex: 1;
      height: 100%;
      position: relative;
    }}
    #network {{
      width: 100%;
      height: 100%;
    }}
    .hud {{
      position: absolute;
      top: 20px;
      right: 20px;
      background: rgba(17, 24, 39, 0.85);
      backdrop-filter: blur(8px);
      border: 1px solid #374151;
      border-radius: 10px;
      padding: 10px 16px;
      font-size: 12px;
      color: #9CA3AF;
      display: flex;
      gap: 16px;
      z-index: 5;
    }}
  </style>
</head>
<body>
  <div id="sidebar">
    <div id="header">
      <h1>TFT Lobby Graph</h1>
      <p>Multi-Tier Player Navigation & Lobby Topology</p>
    </div>

    <div class="stat-grid">
      <div class="stat-card">
        <div class="stat-val">{total_players:,}</div>
        <div class="stat-lbl">Total Players</div>
      </div>
      <div class="stat-card">
        <div class="stat-val">{total_games:,}</div>
        <div class="stat-lbl">Candidate Games</div>
      </div>
      <div class="stat-card">
        <div class="stat-val">{total_edges:,}</div>
        <div class="stat-lbl">Lobby Edges</div>
      </div>
      <div class="stat-card">
        <div class="stat-val">{app_cnt:,}</div>
        <div class="stat-lbl">App Users</div>
      </div>
    </div>

    <div class="control-section">
      <div class="control-title">Search Player</div>
      <input type="text" id="search-box" placeholder="e.g. Dishsoap#NA1 or Javi#401..." oninput="searchPlayer(this.value)">
    </div>

    <div class="control-section">
      <div class="control-title">Filter Leagues / Tiers</div>
      <div class="tier-pills" id="tier-filters">
        <!-- Rendered by JS -->
      </div>
    </div>

    <div class="control-section">
      <div class="control-title">Selected Player Details</div>
      <div id="node-details">
        <p style="color:#9CA3AF; font-size:12px;">Click any player node in the graph to inspect rank, lobby depth, and candidate matches.</p>
      </div>
    </div>
  </div>

  <div id="graph-container">
    <div class="hud">
      <span>🖱️ Drag to pan</span>
      <span>🔍 Scroll to zoom</span>
      <span>✨ Click node to focus</span>
    </div>
    <div id="network"></div>
  </div>

  <script>
    const rawNodes = {json.dumps(nodes_json)};
    const rawEdges = {json.dumps(edges_json)};
    const tierColors = {json.dumps(TIER_COLORS)};

    let activeTiers = new Set(Object.keys(tierColors));

    // Render Tier Filter Buttons
    const pillsContainer = document.getElementById('tier-filters');
    Object.keys(tierColors).forEach(tier => {{
      if (tier === 'UNKNOWN') return;
      const pill = document.createElement('div');
      pill.className = 'tier-pill';
      pill.style.background = tierColors[tier] + '22';
      pill.style.color = tierColors[tier];
      pill.style.borderColor = tierColors[tier] + '66';
      pill.innerText = tier;
      pill.onclick = () => toggleTier(tier, pill);
      pillsContainer.appendChild(pill);
    }});

    function toggleTier(tier, element) {{
      if (activeTiers.has(tier)) {{
        activeTiers.delete(tier);
        element.classList.add('inactive');
      }} else {{
        activeTiers.add(tier);
        element.classList.remove('inactive');
      }}
      applyFilter();
    }}

    const container = document.getElementById('network');
    const nodesDataset = new vis.DataSet(rawNodes.map(n => ({{
      id: n.id,
      label: n.is_app_user ? n.label : '',
      title: n.label + ' (' + n.tier + ')',
      color: {{
        background: n.color,
        border: n.is_app_user ? '#FFFFFF' : n.color,
        highlight: {{ background: '#FFFFFF', border: n.color }}
      }},
      size: n.size,
      borderWidth: n.is_app_user ? 2 : 1,
      shape: n.is_app_user ? 'diamond' : 'dot',
      tier: n.tier,
      font: {{ color: '#F3F4F6', size: 10, strokeWidth: 2, strokeColor: '#000' }}
    }})));

    const edgesDataset = new vis.DataSet(rawEdges.map(e => ({{
      from: e.from,
      to: e.to,
      color: {{ color: '#4B5563', opacity: 0.25 }},
      width: 0.8
    }})));

    const data = {{ nodes: nodesDataset, edges: edgesDataset }};
    const options = {{
      nodes: {{ scaling: {{ min: 4, max: 30 }} }},
      edges: {{ smooth: {{ type: 'continuous' }} }},
      physics: {{
        stabilization: {{ iterations: 100 }},
        barnesHut: {{
          gravitationalConstant: -1800,
          centralGravity: 0.2,
          springLength: 45,
          springConstant: 0.04,
          damping: 0.09
        }}
      }},
      interaction: {{ hover: true, tooltipDelay: 100, zoomView: true }}
    }};

    const network = new vis.Network(container, data, options);

    network.on('click', function(params) {{
      if (params.nodes.length > 0) {{
        const nodeId = params.nodes[0];
        const node = rawNodes.find(n => n.id === nodeId);
        if (node) showNodeDetails(node);
      }}
    }});

    function showNodeDetails(node) {{
      const details = document.getElementById('node-details');
      details.innerHTML = `
        <h3>${{node.id}}</h3>
        <div class="detail-row"><span>League / Tier:</span><span style="color:${{node.color}}">${{node.tier}} (${{node.rank_text}})</span></div>
        <div class="detail-row"><span>Region:</span><span>${{node.region}}</span></div>
        <div class="detail-row"><span>App User:</span><span>${{node.is_app_user ? '✅ Yes' : '❌ No'}}</span></div>
        <div class="detail-row"><span>Tracked Matches:</span><span>${{node.app_matches}}</span></div>
        <div class="detail-row"><span>Lobby Hop Depth:</span><span>Level ${{node.depth}}</span></div>
        <div class="detail-row"><span>Status:</span><span>${{node.scanned ? 'Scanned' : 'In Queue'}}</span></div>
      `;
    }}

    function searchPlayer(query) {{
      if (!query) return;
      const clean = query.trim().toLowerCase();
      const match = rawNodes.find(n => n.id.toLowerCase().includes(clean));
      if (match) {{
        network.focus(match.id, {{
          scale: 1.5,
          animation: {{ duration: 600, easingFunction: 'easeInOutQuad' }}
        }});
        network.selectNodes([match.id]);
        showNodeDetails(match);
      }}
    }}

    function applyFilter() {{
      const filtered = rawNodes.filter(n => activeTiers.has(n.tier)).map(n => n.id);
      const allIds = rawNodes.map(n => n.id);
      const hiddenIds = allIds.filter(id => !filtered.includes(id));
      nodesDataset.update(filtered.map(id => ({{ id, hidden: false }})));
      nodesDataset.update(hiddenIds.map(id => ({{ id, hidden: true }})));
    }}
  </script>
</body>
</html>
"""
    out.write_text(html_content, encoding="utf-8")
    return out


def export_progression_gif(
    graph: PlayerGraph,
    output_path: Path | str,
    num_frames: int | None = 0,
    fps: int = 24,
    step_by_step: bool = True,
) -> Path:
    """Generate a dynamic animated GIF showing player nodes discovered one-by-one with live telemetry.

    Optimized for high-speed playback (20+ FPS) without player name clutter, supporting the entire
    network graph across all connected match lobby constellations.
    """
    import io
    import math
    import numpy as np
    from PIL import Image
    from matplotlib.collections import LineCollection

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    G_full = build_networkx_graph(graph)
    total_nodes = len(graph.nodes)
    if total_nodes == 0:
        raise ValueError("Cannot create animation from empty graph")

    target_frames = total_nodes if (num_frames is None or num_frames <= 0) else min(num_frames, total_nodes)

    # Compute stable organic constellation positions for all nodes
    fixed_pos = compute_constellation_layout(G_full, graph)
    comps = sorted(nx.connected_components(G_full), key=len, reverse=True)

    # Identify exploration seeds, prioritizing diverse tiers (Bronze/Iron, Gold/Silver, High Elo)
    seeds_ranked = []
    for seed_candidate in ["bombos973#EUW", "Gin Ichimaru#BLCH", "Conso#Prepu"]:
        if seed_candidate in G_full:
            seeds_ranked.append(seed_candidate)
    for n in graph.nodes.values():
        if n.depth == 0 and n.scanned and n.riot_id in G_full and n.riot_id not in seeds_ranked:
            seeds_ranked.append(n.riot_id)

    # Fallback if no depth 0 scanned seed exists
    if not seeds_ranked:
        for c in comps:
            nodes_in_c = list(c)
            min_d = min(graph.nodes[nid].depth for nid in nodes_in_c)
            for nid in nodes_in_c:
                if graph.nodes[nid].depth == min_d and nid not in seeds_ranked:
                    seeds_ranked.append(nid)
                    break

    sequence: list[tuple[str, str | None, str]] = []  # (node_id, parent_id, match_uuid)
    seen_nodes: set[str] = set()

    # 1. Traverse priority seeds and their components first
    for seed_id in seeds_ranked:
        if len(sequence) >= target_frames:
            break
        if seed_id in seen_nodes:
            continue

        comp = [c for c in comps if seed_id in c]
        if not comp:
            continue
        comp_nodes = comp[0]
        sub = G_full.subgraph(comp_nodes)
        edges = list(nx.bfs_edges(sub, seed_id))

        seen_nodes.add(seed_id)
        sequence.append((seed_id, None, ""))

        for u, v in edges:
            if len(sequence) >= target_frames:
                break
            if v in seen_nodes:
                continue
            seen_nodes.add(v)
            edge_data = G_full.get_edge_data(u, v, default={})
            muuid = edge_data.get("match_uuid", "")
            sequence.append((v, u, muuid))

    # 2. Traverse all remaining components to ensure full graph coverage
    for comp in comps:
        if len(sequence) >= target_frames:
            break
        comp_nodes = list(comp)
        min_d = min(graph.nodes[nid].depth for nid in comp_nodes)
        roots = [nid for nid in comp_nodes if graph.nodes[nid].depth == min_d]
        root = roots[0]

        if root not in seen_nodes:
            seen_nodes.add(root)
            sequence.append((root, None, ""))

        sub = G_full.subgraph(comp)
        for u, v in nx.bfs_edges(sub, root):
            if len(sequence) >= target_frames:
                break
            if v in seen_nodes:
                continue
            seen_nodes.add(v)
            edge_data = G_full.get_edge_data(u, v, default={})
            muuid = edge_data.get("match_uuid", "")
            sequence.append((v, u, muuid))

    # 3. Fallback for any remaining isolated nodes
    if len(sequence) < target_frames:
        for n_id in list(G_full.nodes()):
            if len(sequence) >= target_frames:
                break
            if n_id not in seen_nodes:
                seen_nodes.add(n_id)
                sequence.append((n_id, None, ""))

    # Ensure all nodes in sequence have coordinates
    for n_id, _, _ in sequence:
        if n_id not in fixed_pos:
            fixed_pos[n_id] = (0.0, 0.0)

    # Map candidate games to players
    player_to_games: dict[str, list[Any]] = defaultdict(list)
    for g in graph.games.values():
        player_to_games[g.focal_player_riot_id].append(g)

    tiers_order = ("CHALLENGER", "GRANDMASTER", "MASTER", "DIAMOND", "EMERALD", "PLATINUM", "GOLD", "SILVER", "BRONZE", "IRON")
    frames: list[Image.Image] = []

    all_x = [fixed_pos[n][0] for n, _, _ in sequence if n in fixed_pos]
    all_y = [fixed_pos[n][1] for n, _, _ in sequence if n in fixed_pos]
    pad = 2.0
    x_min, x_max = (min(all_x) - pad, max(all_x) + pad) if all_x else (-5.0, 5.0)
    y_min, y_max = (min(all_y) - pad, max(all_y) + pad) if all_y else (-5.0, 5.0)

    total_steps = len(sequence)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(15, 8.5), dpi=75)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0], height_ratios=[1.2, 0.8], hspace=0.25, wspace=0.2)
    ax_net = fig.add_subplot(gs[:, 0])
    ax_hud = fig.add_subplot(gs[0, 1])
    ax_games = fig.add_subplot(gs[1, 1])

    edge_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    active_xs: list[float] = []
    active_ys: list[float] = []
    active_colors: list[str] = []

    cur_games_dist: dict[str, int] = defaultdict(int)
    edges_active = 0

    for step_idx in range(total_steps):
        cur_node_id, cur_parent_id, cur_match_uuid = sequence[step_idx]
        cur_node = graph.nodes[cur_node_id]

        if cur_node_id in fixed_pos:
            cx, cy = fixed_pos[cur_node_id]
            active_xs.append(cx)
            active_ys.append(cy)
            active_colors.append(TIER_COLORS.get(cur_node.tier, "#FFFFFF"))

        if cur_parent_id and cur_parent_id in fixed_pos and cur_node_id in fixed_pos:
            edge_segments.append((fixed_pos[cur_parent_id], fixed_pos[cur_node_id]))
            edges_active += 1

        for g in player_to_games[cur_node_id]:
            cur_games_dist[g.tier] += 1
        total_active_games = sum(cur_games_dist.values())

        # Clear axes for fast redraw
        ax_net.cla()
        ax_hud.cla()
        ax_games.cla()

        # 1. Left Panel: Step-by-step Network Graph (NO PLAYER NAMES, FAST SCATTER & LINECOLLECTION)
        ax_net.set_title(
            f"TFT Player Lobby Network Discovery: Step {step_idx + 1:04d}/{total_steps:04d} (Full Graph Stream)",
            fontsize=12,
            fontweight="bold",
            color="#F8F9FA",
            pad=10,
        )

        # Draw existing edges via LineCollection
        if edge_segments:
            lc = LineCollection(edge_segments, colors="#4A5568", alpha=0.30, linewidths=0.75)
            ax_net.add_collection(lc)

        # Highlight newly added edge in bright cyan
        if cur_parent_id and cur_parent_id in fixed_pos and cur_node_id in fixed_pos:
            active_lc = LineCollection(
                [(fixed_pos[cur_parent_id], fixed_pos[cur_node_id])],
                colors="#38BDF8",
                alpha=0.95,
                linewidths=2.2,
            )
            ax_net.add_collection(active_lc)

        # Draw nodes via fast scatter
        if active_xs:
            ax_net.scatter(active_xs, active_ys, c=active_colors, s=26, alpha=0.85)

        # Highlight newly added node with glowing halo (NO TEXT LABELS)
        if cur_node_id in fixed_pos:
            cx, cy = fixed_pos[cur_node_id]
            ax_net.scatter([cx], [cy], s=170, c="#00F0FF", alpha=0.40)
            ax_net.scatter(
                [cx],
                [cy],
                s=70,
                c=TIER_COLORS.get(cur_node.tier, "#FFFFFF"),
                edgecolors="#FFFFFF",
                linewidths=1.6,
                alpha=1.0,
            )

        ax_net.set_xlim(x_min, x_max)
        ax_net.set_ylim(y_min, y_max)
        ax_net.axis("off")

        # 2. Right Top Panel: Fast Live Telemetry HUD Card
        ax_hud.axis("off")
        role_label = "Seed / App User" if cur_node.is_app_user else "Lobby Opponent"

        hud_text = (
            f"CRAWLER DISCOVERY TELEMETRY\n"
            f"Step {step_idx + 1:04d} / {total_steps:04d}  [FULL GRAPH RAPID STREAM]\n"
            f"-----------------------------------------\n"
            f" * League Tier: {cur_node.tier} ({cur_node.rank_text})\n"
            f" * Hop Depth  : Depth {cur_node.depth} (lobby distance)\n"
            f" * Role       : {role_label}\n"
            f"-----------------------------------------\n"
            f" * Network    : {step_idx + 1:,} Nodes | {edges_active:,} Edges\n"
            f" * Manifest   : {total_active_games:,} Matches Cataloged"
        )
        ax_hud.text(
            0.04, 0.5,
            hud_text,
            va="center",
            ha="left",
            fontsize=10.5,
            family="monospace",
            color="#67E8F9",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#111827", edgecolor="#38BDF8", linewidth=1.5),
        )

        # 3. Right Bottom Panel: Candidate Games Progress per League
        counts = [cur_games_dist.get(t, 0) for t in tiers_order]
        colors = [TIER_COLORS[t] for t in tiers_order]
        bars = ax_games.barh(tiers_order[::-1], counts[::-1], color=colors[::-1], height=0.65, edgecolor="#1A202C")
        ax_games.set_title("Candidate Games per League", fontsize=11, fontweight="bold", pad=8)
        ax_games.axvline(1000, color="#E74C3C", linestyle="--", linewidth=1.2, label="1K Cap Target")
        max_c = max(1500, max(counts) + 200 if counts else 1500)
        ax_games.set_xlim(0, max_c)
        ax_games.legend(loc="lower right", fontsize=8)
        ax_games.grid(axis="x", alpha=0.15)

        for bar in bars:
            w = bar.get_width()
            if w > 0:
                ax_games.text(w + (max_c * 0.02), bar.get_y() + bar.get_height() / 2, f"{int(w):,}", va="center", fontsize=8, color="#E2E8F0")

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(rgba).convert("RGB"))

    plt.close(fig)

    # Add pause frames at the end (approx 0.8s pause)
    pause_cnt = max(5, int(fps * 0.8))
    for _ in range(pause_cnt):
        frames.append(frames[-1])

    duration_ms = max(15, int(1000 / fps))
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return out
