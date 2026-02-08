"""Complete example: Research -> Analysis -> Report pipeline.

Run this file to see the full orchestration in action:
    python -m examples.pipeline

Or from the project root:
    python examples/pipeline.py
"""

import sys
import logging
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator import DAGOrchestrator, AgentNode, Monitor
from agents import ResearchAgent, AnalysisAgent, ReportAgent

# Configure logging to see agent execution
logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    # 1. Define the pipeline as a DAG
    orchestrator = DAGOrchestrator()
    orchestrator.add_node(
        AgentNode(agent=ResearchAgent())
    )
    orchestrator.add_node(
        AgentNode(agent=AnalysisAgent(), depends_on=["research"])
    )
    orchestrator.add_node(
        AgentNode(agent=ReportAgent(), depends_on=["analysis", "research"])
    )

    # 2. Run the pipeline
    print("\n--- Running Pipeline ---\n")
    results = orchestrator.run(context={
        "query": "AI agent orchestration market trends",
        "research_depth": "deep",
    })

    # 3. Print the execution summary
    print("\n" + orchestrator.summary())

    # 4. Print the generated report
    report = results.get("report")
    if report and report.success:
        print("\n--- Generated Report ---")
        print(report.data["report_text"])

    # 5. Display the monitoring dashboard
    monitor = Monitor()
    monitor.record(results, orchestrator.timeline)
    print(monitor.dashboard())

    # 6. Export metrics to JSON
    monitor.export_json("pipeline_metrics.json")
    print("Metrics exported to pipeline_metrics.json")


if __name__ == "__main__":
    main()
