"""AnalysisAgent — Processes research findings into structured analysis.

Demonstrates how to consume upstream agent results via
agent_input.upstream_results. In production, replace the mock
logic with your actual analysis (LLM reasoning, statistical
models, ML inference, etc.).
"""

from __future__ import annotations

import time

from orchestrator.base_agent import BaseAgent, AgentInput


class AnalysisAgent(BaseAgent):
    """Analyzes research findings and produces structured insights.

    Depends on: ResearchAgent (consumes its findings).
    Replace execute() body with your actual analysis logic.
    """

    name = "analysis"
    timeout_seconds = 20.0

    def execute(self, agent_input: AgentInput) -> dict:
        research = agent_input.upstream_results.get("research", {})
        findings = research.get("findings", [])
        query = research.get("query", agent_input.context.get("query", "unknown"))

        if not findings:
            raise ValueError("No research findings to analyze — check upstream agent")

        # --- Replace this block with your actual analysis logic ---
        time.sleep(0.08)  # Simulate processing time

        # Score and rank findings
        scored = sorted(findings, key=lambda f: f["relevance"], reverse=True)
        avg_relevance = sum(f["relevance"] for f in findings) / len(findings)

        insights = []
        for i, finding in enumerate(scored[:3]):
            insights.append(
                {
                    "rank": i + 1,
                    "source": finding["source"],
                    "insight": f"High-confidence signal from {finding['source']}: {finding['summary']}",
                    "confidence": round(finding["relevance"] * 100),
                    "actionable": finding["relevance"] > 0.8,
                }
            )

        risk_factors = []
        if avg_relevance < 0.7:
            risk_factors.append("Low average relevance — findings may be tangential")
        if len(findings) < 3:
            risk_factors.append("Insufficient data sources for robust analysis")

        recommendation = "PROCEED" if avg_relevance > 0.75 and not risk_factors else "REVIEW"
        # --- End replaceable block ---

        return {
            "query": query,
            "insights": insights,
            "risk_factors": risk_factors,
            "recommendation": recommendation,
            "avg_relevance": round(avg_relevance, 3),
            "sources_analyzed": len(findings),
            "_token_usage": 2100,
            "_cost_usd": 0.0032,
        }
