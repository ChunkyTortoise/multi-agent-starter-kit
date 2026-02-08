"""ResearchAgent — Gathers and synthesizes information from context.

Example agent demonstrating the BaseAgent pattern. In production,
you'd replace the mock logic with actual API calls (web search,
database queries, LLM calls, etc.).
"""

from __future__ import annotations

import hashlib
import time

from orchestrator.base_agent import BaseAgent, AgentInput


class ResearchAgent(BaseAgent):
    """Gathers information relevant to a query.

    Simulates research by generating structured findings.
    Replace the execute() body with your actual research logic
    (API calls, web scraping, database queries, LLM prompts).
    """

    name = "research"
    timeout_seconds = 15.0

    def execute(self, agent_input: AgentInput) -> dict:
        query = agent_input.context.get("query", "general analysis")
        depth = agent_input.context.get("research_depth", "standard")

        # --- Replace this block with your actual research logic ---
        time.sleep(0.05)  # Simulate API latency

        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        findings = [
            {
                "source": "market_data",
                "relevance": 0.92,
                "summary": f"Current market trends indicate strong demand in '{query}' sector",
                "data_points": 47,
            },
            {
                "source": "competitor_analysis",
                "relevance": 0.85,
                "summary": f"3 major competitors identified in '{query}' space",
                "data_points": 23,
            },
            {
                "source": "historical_patterns",
                "relevance": 0.78,
                "summary": f"Historical data shows cyclical patterns for '{query}'",
                "data_points": 156,
            },
        ]

        if depth == "deep":
            findings.append(
                {
                    "source": "academic_literature",
                    "relevance": 0.71,
                    "summary": f"12 peer-reviewed papers found for '{query}'",
                    "data_points": 89,
                }
            )
        # --- End replaceable block ---

        return {
            "query": query,
            "research_id": query_hash,
            "findings": findings,
            "total_sources": len(findings),
            "total_data_points": sum(f["data_points"] for f in findings),
            "_token_usage": 1250,
            "_cost_usd": 0.0019,
        }
