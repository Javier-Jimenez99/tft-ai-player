"""Generates a standalone, beautiful HTML interactive audit dashboard for the TFT-AI Model Ecosystem."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_html_report(audit_data: dict[str, Any], output_path: str | Path) -> None:
    """Render self-contained HTML report with modern dark theme and interactive tables."""
    reports = audit_data.get("reports", {})
    timestamp = audit_data.get("timestamp", "N/A")
    elapsed = audit_data.get("elapsed_seconds", 0.0)
    device = audit_data.get("device", "cpu")

    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    warn_tests = 0

    for rep in reports.values():
        for test in rep.get("test_results", []):
            total_tests += 1
            st = test.get("status", "PASS")
            if st == "PASS":
                passed_tests += 1
            elif st == "WARN":
                warn_tests += 1
            else:
                failed_tests += 1

    health_pct = (passed_tests / max(1, total_tests)) * 100

    # Build Model Cards HTML
    cards_html = []
    for model_key, rep in reports.items():
        name = rep.get("model_name", model_key)
        ckpt = rep.get("checkpoint_path", "")
        status = rep.get("overall_status", "PASS")
        diagnosis = rep.get("diagnosis", "")
        tests = rep.get("test_results", [])
        empirical = rep.get("empirical_metrics", {})

        status_class = f"status-{status.lower()}"
        status_badge = f'<span class="badge {status_class}">{status}</span>'

        # Tests table
        tests_rows = []
        for t in tests:
            t_status = t.get("status", "PASS")
            t_badge = f'<span class="badge badge-sm status-{t_status.lower()}">{t_status}</span>'
            tests_rows.append(f"""
                <tr>
                    <td class="font-medium">{t.get("name", "")}</td>
                    <td class="text-dim text-sm">{t.get("description", "")}</td>
                    <td class="font-mono text-sm">{t.get("law", "")}</td>
                    <td class="font-mono text-sm">{t.get("expected", "")}</td>
                    <td class="font-mono text-sm font-semibold">{t.get("actual", "")}</td>
                    <td>{t_badge}</td>
                </tr>
            """)
        tests_table = "".join(tests_rows)

        # Empirical metrics list
        metrics_html = ""
        if empirical:
            metric_items = []
            for mk, mv in empirical.items():
                label = mk.replace("_", " ").capitalize()
                metric_items.append(f"""
                    <div class="metric-pill">
                        <span class="text-dim">{label}:</span>
                        <span class="font-mono font-semibold">{mv}</span>
                    </div>
                """)
            metrics_html = f"""
                <div class="metrics-container">
                    <div class="section-title">Empirical Replay Metrics</div>
                    <div class="metrics-grid">{"".join(metric_items)}</div>
                </div>
            """

        diag_html = ""
        if diagnosis:
            diag_class = "diag-fail" if status == "FAIL" else "diag-warn"
            diag_html = f"""
                <div class="diagnosis-box {diag_class}">
                    <strong>Diagnosis & Root Cause:</strong> {diagnosis}
                </div>
            """

        cards_html.append(f"""
            <div class="model-card">
                <div class="card-header">
                    <div>
                        <h2 class="card-title">{name}</h2>
                        <div class="card-subtitle font-mono">{ckpt}</div>
                    </div>
                    <div>{status_badge}</div>
                </div>
                {diag_html}
                {metrics_html}
                <div class="table-container">
                    <div class="section-title">Deterministic Axiomatic Law Tests</div>
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Test Name</th>
                                <th>Description</th>
                                <th>Physical TFT Law</th>
                                <th>Expected</th>
                                <th>Actual Measured</th>
                                <th>Verdict</th>
                            </tr>
                        </thead>
                        <tbody>
                            {tests_table}
                        </tbody>
                    </table>
                </div>
            </div>
        """)

    all_cards = "\n".join(cards_html)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TFT-AI Model Ecosystem Sanity & Benchmark Audit</title>
    <style>
        :root {{
            --bg-primary: #0b0f19;
            --bg-secondary: #121a2d;
            --bg-card: #18233c;
            --border: #263554;
            --text-main: #e2e8f0;
            --text-dim: #94a3b8;
            --accent: #3b82f6;
            --pass-bg: #064e3b;
            --pass-text: #34d399;
            --warn-bg: #78350f;
            --warn-text: #fbbf24;
            --fail-bg: #7f1d1d;
            --fail-text: #f87171;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        body {{
            background-color: var(--bg-primary);
            color: var(--text-main);
            padding: 2rem;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1300px;
            margin: 0 auto;
        }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 2rem;
            border-bottom: 1px solid var(--border);
            margin-bottom: 2rem;
        }}
        h1 {{
            font-size: 1.8rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #f8fafc;
        }}
        .meta-info {{
            font-size: 0.875rem;
            color: var(--text-dim);
            margin-top: 0.25rem;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .stat-card {{
            background-color: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1.25rem;
        }}
        .stat-title {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-dim);
            font-weight: 600;
        }}
        .stat-value {{
            font-size: 1.8rem;
            font-weight: 700;
            margin-top: 0.25rem;
            color: #f8fafc;
        }}
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.65rem;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }}
        .badge-sm {{
            padding: 0.15rem 0.45rem;
            font-size: 0.7rem;
        }}
        .status-pass {{
            background-color: var(--pass-bg);
            color: var(--pass-text);
            border: 1px solid var(--pass-text);
        }}
        .status-warn {{
            background-color: var(--warn-bg);
            color: var(--warn-text);
            border: 1px solid var(--warn-text);
        }}
        .status-fail {{
            background-color: var(--fail-bg);
            color: var(--fail-text);
            border: 1px solid var(--fail-text);
        }}
        .model-card {{
            background-color: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            border-bottom: 1px solid var(--border);
            padding-bottom: 1rem;
            margin-bottom: 1.25rem;
        }}
        .card-title {{
            font-size: 1.3rem;
            font-weight: 600;
            color: #ffffff;
        }}
        .card-subtitle {{
            font-size: 0.8rem;
            color: var(--text-dim);
            margin-top: 0.2rem;
        }}
        .diagnosis-box {{
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1.25rem;
            font-size: 0.9rem;
            line-height: 1.6;
        }}
        .diag-fail {{
            background-color: rgba(127, 29, 29, 0.4);
            border-left: 4px solid var(--fail-text);
            color: #fecaca;
        }}
        .diag-warn {{
            background-color: rgba(120, 53, 15, 0.4);
            border-left: 4px solid var(--warn-text);
            color: #fde68a;
        }}
        .section-title {{
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--accent);
            margin-bottom: 0.75rem;
        }}
        .metrics-container {{
            margin-bottom: 1.25rem;
        }}
        .metrics-grid {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}
        .metric-pill {{
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.35rem 0.75rem;
            font-size: 0.85rem;
            display: flex;
            gap: 0.4rem;
        }}
        .table-container {{
            overflow-x: auto;
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875rem;
            text-align: left;
        }}
        .data-table th {{
            background-color: var(--bg-card);
            color: var(--text-dim);
            padding: 0.75rem 1rem;
            font-weight: 600;
            border-bottom: 1px solid var(--border);
        }}
        .data-table td {{
            padding: 0.75rem 1rem;
            border-bottom: 1px solid rgba(38, 53, 84, 0.5);
            vertical-align: middle;
        }}
        .data-table tr:hover {{
            background-color: rgba(24, 35, 60, 0.4);
        }}
        .font-mono {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }}
        .font-medium {{ font-weight: 500; }}
        .font-semibold {{ font-weight: 600; }}
        .text-dim {{ color: var(--text-dim); }}
        .text-sm {{ font-size: 0.8rem; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>TFT-AI Model Ecosystem Sanity & Benchmark Audit</h1>
                <div class="meta-info">
                    Timestamp: <strong>{timestamp}</strong> | Device: <strong>{device}</strong> | Audit Runtime: <strong>{elapsed}s</strong>
                </div>
            </div>
            <div>
                <span class="badge status-{'pass' if failed_tests == 0 else 'fail'}" style="font-size: 1rem; padding: 0.5rem 1rem;">
                    Ecosystem Health: {health_pct:.1f}%
                </span>
            </div>
        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-title">Models Audited</div>
                <div class="stat-value">{len(reports)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-title">Axiomatic Tests Executed</div>
                <div class="stat-value">{total_tests}</div>
            </div>
            <div class="stat-card">
                <div class="stat-title">Tests Passed</div>
                <div class="stat-value" style="color: var(--pass-text);">{passed_tests}</div>
            </div>
            <div class="stat-card">
                <div class="stat-title">Critical Law Violations</div>
                <div class="stat-value" style="color: var(--fail-text);">{failed_tests}</div>
            </div>
        </div>

        {all_cards}
    </div>
</body>
</html>
"""

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            data = json.load(f)
        generate_html_report(data, sys.argv[2])
