"""Monitor — ASCII-based real-time monitoring dashboard for agent pipelines.

No Grafana, no Prometheus, no setup. Works in any terminal.
Renders agent performance table, Gantt chart timeline, and exports JSON.
"""

from __future__ import annotations

import json
from typing import Any

from orchestrator.base_agent import AgentOutput


class Monitor:
    """ASCII monitoring dashboard for pipeline executions.

    Usage:
        monitor = Monitor()
        monitor.record(results, timeline)
        print(monitor.dashboard())
        monitor.export_json("run_001.json")
    """

    def __init__(self) -> None:
        self._runs: list[dict[str, Any]] = []

    def record(
        self,
        results: dict[str, AgentOutput],
        timeline: list[dict[str, Any]],
    ) -> None:
        """Record a pipeline execution for monitoring."""
        self._runs.append({"results": results, "timeline": timeline})

    def dashboard(self, run_index: int = -1) -> str:
        """Render an ASCII dashboard for a pipeline run.

        Args:
            run_index: Which run to display (-1 for latest).

        Returns:
            Formatted ASCII dashboard string.
        """
        if not self._runs:
            return "No runs recorded yet."

        run = self._runs[run_index]
        results: dict[str, AgentOutput] = run["results"]
        timeline: list[dict[str, Any]] = run["timeline"]

        sections = [
            self._render_header(len(self._runs), run_index),
            self._render_performance_table(results),
            self._render_gantt(timeline),
            self._render_cost_summary(results),
        ]
        return "\n".join(sections)

    def _render_header(self, total_runs: int, run_index: int) -> str:
        actual_index = run_index if run_index >= 0 else total_runs + run_index
        return (
            f"\n{'=' * 60}\n"
            f"  AGENT PIPELINE MONITOR — Run {actual_index + 1}/{total_runs}\n"
            f"{'=' * 60}"
        )

    def _render_performance_table(self, results: dict[str, AgentOutput]) -> str:
        header = f"\n{'Agent':<20} {'Status':<8} {'Duration':<12} {'Tokens':<10} {'Cost':<10}"
        separator = "-" * 60
        rows = [header, separator]

        for name, result in results.items():
            status = "PASS" if result.success else "FAIL"
            duration = f"{result.duration_ms:.0f}ms"
            tokens = str(result.token_usage) if result.token_usage else "-"
            cost = (
                f"${result.estimated_cost_usd:.4f}"
                if result.estimated_cost_usd
                else "-"
            )
            rows.append(
                f"{name:<20} {status:<8} {duration:<12} {tokens:<10} {cost:<10}"
            )

        return "\n".join(rows)

    def _render_gantt(self, timeline: list[dict[str, Any]]) -> str:
        if not timeline:
            return "\nNo timeline data."

        max_end = max(t["end"] for t in timeline) if timeline else 1.0
        chart_width = 40
        lines = [f"\n{'Timeline':^60}", "-" * 60]

        for entry in timeline:
            name = entry["agent"][:16]
            start_pos = int((entry["start"] / max_end) * chart_width)
            end_pos = max(start_pos + 1, int((entry["end"] / max_end) * chart_width))
            bar_char = "#" if entry["success"] else "!"

            bar = (
                "." * start_pos
                + bar_char * (end_pos - start_pos)
                + "." * (chart_width - end_pos)
            )
            duration = entry["duration_ms"]
            lines.append(f"  {name:<16} |{bar}| {duration:.0f}ms")

        scale_label = f"0{'':>{chart_width - 6}}{max_end * 1000:.0f}ms"
        lines.append(f"  {'':16} |{scale_label}|")
        return "\n".join(lines)

    def _render_cost_summary(self, results: dict[str, AgentOutput]) -> str:
        total_tokens = sum(r.token_usage for r in results.values())
        total_cost = sum(r.estimated_cost_usd for r in results.values())
        total_duration = sum(r.duration_ms for r in results.values())
        success_count = sum(1 for r in results.values() if r.success)

        return (
            f"\n{'=' * 60}\n"
            f"  Agents: {success_count}/{len(results)} passed | "
            f"Duration: {total_duration:.0f}ms | "
            f"Tokens: {total_tokens} | "
            f"Cost: ${total_cost:.4f}\n"
            f"{'=' * 60}\n"
        )

    def export_json(self, filepath: str, run_index: int = -1) -> None:
        """Export a run's data as JSON for external analysis."""
        if not self._runs:
            return

        run = self._runs[run_index]
        export = {
            "timeline": run["timeline"],
            "results": {
                name: {
                    "success": r.success,
                    "duration_ms": r.duration_ms,
                    "token_usage": r.token_usage,
                    "estimated_cost_usd": r.estimated_cost_usd,
                    "error": r.error,
                    "data_keys": list(r.data.keys()),
                }
                for name, r in run["results"].items()
            },
        }
        with open(filepath, "w") as f:
            json.dump(export, f, indent=2)

    def compare_runs(self) -> str:
        """Compare all recorded runs side by side."""
        if len(self._runs) < 2:
            return "Need at least 2 runs to compare."

        lines = [f"\n{'Run Comparison':^60}", "=" * 60]
        header = f"{'Agent':<20}"
        for i in range(len(self._runs)):
            header += f" {'Run ' + str(i + 1):<12}"
        lines.append(header)
        lines.append("-" * 60)

        all_agents = list(self._runs[0]["results"].keys())
        for agent in all_agents:
            row = f"{agent:<20}"
            for run in self._runs:
                result = run["results"].get(agent)
                if result:
                    row += f" {result.duration_ms:>6.0f}ms    "
                else:
                    row += f" {'N/A':>10}  "
            lines.append(row)

        return "\n".join(lines)
