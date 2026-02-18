"""HITL example: Human-in-the-loop approval gate in an agent pipeline.

Demonstrates two modes:
1. Auto-approve (non-interactive, for CI/automated tests)
2. External approval (webhook-style, where another thread resolves the gate)

Run from project root:
    python examples/hitl_pipeline.py
"""

import logging
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator import AgentInput, AgentNode, DAGOrchestrator, HITLGate
from orchestrator.base_agent import BaseAgent

logging.basicConfig(level=logging.INFO, format="%(message)s")


# --- Agent definitions ---


class DataIngestionAgent(BaseAgent):
    name = "ingest"

    def execute(self, agent_input: AgentInput) -> dict:
        dataset = agent_input.context.get("dataset", "sales_q1.csv")
        print(f"  [ingest] Loading dataset: {dataset}")
        time.sleep(0.05)  # Simulate I/O
        return {
            "row_count": 12_847,
            "columns": ["date", "revenue", "region", "product"],
            "dataset": dataset,
        }


class TransformationAgent(BaseAgent):
    name = "transform"

    def execute(self, agent_input: AgentInput) -> dict:
        row_count = agent_input.upstream_results["ingest"]["row_count"]
        print(f"  [transform] Applying transformations to {row_count:,} rows")
        time.sleep(0.05)
        return {
            "cleaned_rows": row_count - 43,
            "dropped_rows": 43,
            "transformations": ["dedup", "normalize_dates", "fill_nulls"],
        }


class ExportAgent(BaseAgent):
    name = "export"

    def execute(self, agent_input: AgentInput) -> dict:
        cleaned = agent_input.upstream_results["transform"]["cleaned_rows"]
        print(f"  [export] Writing {cleaned:,} rows to data warehouse")
        time.sleep(0.05)
        return {"status": "exported", "destination": "bigquery://prod/sales"}


# --- Mode 1: Auto-approve after timeout (non-interactive) ---


def demo_auto_approve():
    print("\n" + "=" * 60)
    print("  DEMO 1: Auto-approve after timeout")
    print("=" * 60)

    # Gate auto-approves after 0.5s if no human responds
    review_gate = HITLGate(
        name="data_quality_review",
        description="Approve before writing transformed data to production",
        auto_approve_after=0.5,
        interactive=False,  # No stdin prompts
    )

    orchestrator = DAGOrchestrator()
    orchestrator.add_node(AgentNode(agent=DataIngestionAgent()))
    orchestrator.add_node(
        AgentNode(
            agent=TransformationAgent(),
            depends_on=["ingest"],
            hitl_gate=review_gate,  # Gate triggers AFTER transform succeeds
        )
    )
    orchestrator.add_node(AgentNode(agent=ExportAgent(), depends_on=["transform"]))

    results = orchestrator.run(context={"dataset": "sales_q4.csv"})
    print("\n" + orchestrator.summary())
    print(f"\nGate history: {review_gate.summary()}")


# --- Mode 2: External approval (webhook-style) ---


def demo_external_approval():
    print("\n" + "=" * 60)
    print("  DEMO 2: External approval (simulating webhook)")
    print("=" * 60)

    approvals_received: list[str] = []

    def on_approve(req):
        """Simulates a Slack bot or webhook handler approving the request."""
        approvals_received.append(req.gate_id)

    deploy_gate = HITLGate(
        name="deploy_approval",
        description="Senior engineer must approve before deploying to production",
        auto_approve_after=10.0,  # Long timeout as safety net
        interactive=False,
        on_resolve=on_approve,
    )

    def external_approver():
        """Simulates an external system (Slack, PagerDuty) resolving the gate.
        Polls until the gate appears (pipeline takes ~100ms to reach the gate).
        """
        deadline = time.time() + 5.0
        while time.time() < deadline:
            pending = deploy_gate.pending
            if pending:
                req = pending[0]
                print(f"\n  [external] Webhook received gate_id={req.gate_id}")
                print(f"  [external] Approving: {req.description}")
                deploy_gate.approve(req.gate_id, approver="senior_eng@company.com", notes="Reviewed and LGTM")
                return
            time.sleep(0.05)

    # Start the approver in a background thread (simulating async webhook)
    t = threading.Thread(target=external_approver, daemon=True)
    t.start()

    orchestrator = DAGOrchestrator()
    orchestrator.add_node(AgentNode(agent=DataIngestionAgent()))
    orchestrator.add_node(
        AgentNode(
            agent=TransformationAgent(),
            depends_on=["ingest"],
            hitl_gate=deploy_gate,
        )
    )
    orchestrator.add_node(AgentNode(agent=ExportAgent(), depends_on=["transform"]))

    results = orchestrator.run(context={"dataset": "production_batch.csv"})
    t.join(timeout=5.0)

    print("\n" + orchestrator.summary())
    print(f"\nGate approval rate: {deploy_gate.approval_rate():.0%}")
    if deploy_gate.history:
        approved_by = deploy_gate.history[0].approver
        print(f"Approved by: {approved_by}")


# --- Mode 3: Rejection demonstration ---


def demo_rejection():
    print("\n" + "=" * 60)
    print("  DEMO 3: Gate rejection (pipeline halts)")
    print("=" * 60)

    quality_gate = HITLGate(
        name="quality_check",
        description="Data quality must pass before proceeding",
        auto_approve_after=10.0,
        interactive=False,
    )

    def external_rejector():
        deadline = time.time() + 5.0
        while time.time() < deadline:
            pending = quality_gate.pending
            if pending:
                req = pending[0]
                print(f"\n  [reviewer] Rejecting gate {req.gate_id} — data quality too low")
                quality_gate.reject(req.gate_id, approver="qa_team", reason="Error rate > 5%")
                return
            time.sleep(0.05)

    t = threading.Thread(target=external_rejector, daemon=True)
    t.start()

    # Use DataIngestionAgent (no upstream deps) so it succeeds before the gate
    orchestrator = DAGOrchestrator()
    orchestrator.add_node(
        AgentNode(
            agent=DataIngestionAgent(),
            hitl_gate=quality_gate,  # Gate triggers after ingest succeeds
        )
    )
    orchestrator.add_node(AgentNode(agent=ExportAgent(), depends_on=["ingest"]))

    results = orchestrator.run(context={"dataset": "bad_data.csv"})
    t.join(timeout=5.0)

    print("\n" + orchestrator.summary())
    ingest_result = results["ingest"]
    export_result = results["export"]
    print(f"\nIngest status: {'PASS' if ingest_result.success else 'BLOCKED by gate rejection'}")
    print(f"Export status: {'RAN' if export_result.success else 'SKIPPED (blocked by gate rejection)'}")


if __name__ == "__main__":
    demo_auto_approve()
    demo_external_approval()
    demo_rejection()
