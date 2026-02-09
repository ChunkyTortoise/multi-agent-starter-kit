"""ReportAgent — Generates a structured report from analysis results.

Demonstrates consuming multiple upstream results and producing
final output. In production, replace with LLM-powered report
generation, PDF export, email delivery, etc.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from orchestrator.base_agent import AgentInput, BaseAgent


class ReportAgent(BaseAgent):
    """Produces a final structured report from analysis.

    Depends on: AnalysisAgent (consumes its insights).
    Replace execute() body with your actual report generation.
    """

    name = "report"
    timeout_seconds = 10.0

    def execute(self, agent_input: AgentInput) -> dict:
        analysis = agent_input.upstream_results.get("analysis", {})
        research = agent_input.upstream_results.get("research", {})
        query = analysis.get("query", "unknown")

        insights = analysis.get("insights", [])
        risk_factors = analysis.get("risk_factors", [])
        recommendation = analysis.get("recommendation", "UNKNOWN")

        # --- Replace this block with your actual report generation ---
        time.sleep(0.03)  # Simulate formatting

        executive_summary = (
            f"Analysis of '{query}' complete. "
            f"{len(insights)} key insights identified across "
            f"{analysis.get('sources_analyzed', 0)} sources. "
            f"Recommendation: {recommendation}."
        )

        sections = [
            {
                "title": "Executive Summary",
                "content": executive_summary,
            },
            {
                "title": "Key Findings",
                "content": "\n".join(
                    f"  {i['rank']}. [{i['confidence']}% confidence] {i['insight']}"
                    for i in insights
                ),
            },
        ]

        if risk_factors:
            sections.append(
                {
                    "title": "Risk Factors",
                    "content": "\n".join(f"  - {r}" for r in risk_factors),
                }
            )

        sections.append(
            {
                "title": "Recommendation",
                "content": (
                    f"Action: {recommendation}\n"
                    f"  Average relevance: {analysis.get('avg_relevance', 'N/A')}\n"
                    f"  Data points: {research.get('total_data_points', 'N/A')}"
                ),
            }
        )
        # --- End replaceable block ---

        report_text = ""
        for section in sections:
            report_text += (
                f"\n{'=' * 40}\n{section['title']}\n{'=' * 40}\n{section['content']}\n"
            )

        return {
            "query": query,
            "report_text": report_text,
            "sections": sections,
            "recommendation": recommendation,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "_token_usage": 850,
            "_cost_usd": 0.0013,
        }
