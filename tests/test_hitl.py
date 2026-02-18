"""Tests for the HITL (Human-in-the-Loop) approval gate system."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator import AgentInput, AgentNode, DAGOrchestrator, HITLGate, ApprovalStatus
from orchestrator.base_agent import BaseAgent


# --- Stub agents ---


class EchoAgent(BaseAgent):
    name = "echo"

    def execute(self, agent_input: AgentInput) -> dict:
        return {"message": agent_input.context.get("msg", "hello")}


class FailAgent(BaseAgent):
    name = "fail"
    max_retries = 0

    def execute(self, agent_input: AgentInput) -> dict:
        raise RuntimeError("intentional failure")


# ---------------------------------------------------------------------------
# HITLGate unit tests
# ---------------------------------------------------------------------------


class TestHITLGateAutoApprove:
    def test_auto_approve_after_timeout(self):
        gate = HITLGate(
            "test_gate",
            description="Test",
            auto_approve_after=0.1,
            interactive=False,
        )
        request = gate.request_approval(context={"x": 1}, agent_output={"y": 2})
        assert request.status == ApprovalStatus.AUTO_APPROVED
        assert request.gate_name == "test_gate"
        assert request.wait_time_seconds >= 0

    def test_external_approve_before_timeout(self):
        gate = HITLGate(
            "ext_gate",
            auto_approve_after=5.0,  # long timeout
            interactive=False,
        )

        request_holder: list = []

        def _approve_thread():
            # Wait a tiny bit for the request to be created
            time.sleep(0.05)
            for req in gate.pending:
                gate.approve(req.gate_id, approver="alice", notes="LGTM")

        t = threading.Thread(target=_approve_thread, daemon=True)
        t.start()

        request = gate.request_approval(context={}, agent_output={"result": "ok"})
        t.join(timeout=2.0)

        assert request.status == ApprovalStatus.APPROVED
        assert request.approver == "alice"
        assert request.notes == "LGTM"

    def test_external_reject(self):
        gate = HITLGate(
            "reject_gate",
            auto_approve_after=5.0,
            interactive=False,
        )

        def _reject_thread():
            time.sleep(0.05)
            for req in gate.pending:
                gate.reject(req.gate_id, approver="bob", reason="Not ready")

        t = threading.Thread(target=_reject_thread, daemon=True)
        t.start()

        request = gate.request_approval(context={}, agent_output={})
        t.join(timeout=2.0)

        assert request.status == ApprovalStatus.REJECTED
        assert request.approver == "bob"

    def test_approve_unknown_gate_id_returns_false(self):
        gate = HITLGate("g", auto_approve_after=0.1, interactive=False)
        result = gate.approve("nonexistent-id")
        assert result is False

    def test_reject_unknown_gate_id_returns_false(self):
        gate = HITLGate("g", auto_approve_after=0.1, interactive=False)
        result = gate.reject("nonexistent-id")
        assert result is False

    def test_history_recorded(self):
        gate = HITLGate("history_gate", auto_approve_after=0.05, interactive=False)
        gate.request_approval({}, {})
        gate.request_approval({}, {})
        assert len(gate.history) == 2

    def test_approval_rate(self):
        gate = HITLGate("rate_gate", auto_approve_after=0.05, interactive=False)
        gate.request_approval({}, {})  # AUTO_APPROVED
        gate.request_approval({}, {})  # AUTO_APPROVED
        assert gate.approval_rate() == 1.0

    def test_approval_rate_zero_history(self):
        gate = HITLGate("empty_gate", auto_approve_after=0.05, interactive=False)
        assert gate.approval_rate() == 0.0

    def test_summary_string(self):
        gate = HITLGate("summary_gate", auto_approve_after=0.05, interactive=False)
        gate.request_approval({}, {})
        summary = gate.summary()
        assert "summary_gate" in summary
        assert "Total: 1" in summary

    def test_request_captures_context_snapshot(self):
        gate = HITLGate("ctx_gate", auto_approve_after=0.05, interactive=False)
        ctx = {"key": "value", "count": 42}
        request = gate.request_approval(context=ctx, agent_output={"data": "x"})
        assert request.context_snapshot["key"] == "value"
        assert request.context_snapshot["count"] == 42

    def test_request_captures_agent_output(self):
        gate = HITLGate("out_gate", auto_approve_after=0.05, interactive=False)
        request = gate.request_approval(context={}, agent_output={"score": 99})
        assert request.agent_output["score"] == 99

    def test_to_dict(self):
        gate = HITLGate("dict_gate", auto_approve_after=0.05, interactive=False)
        request = gate.request_approval({}, {})
        d = request.to_dict()
        assert d["gate_name"] == "dict_gate"
        assert "status" in d
        assert "wait_time_seconds" in d

    def test_on_request_callback(self):
        fired: list[str] = []

        def callback(req):
            fired.append(req.gate_id)

        gate = HITLGate(
            "cb_gate",
            auto_approve_after=0.05,
            interactive=False,
            on_request=callback,
        )
        req = gate.request_approval({}, {})
        assert req.gate_id in fired

    def test_on_resolve_callback(self):
        resolved: list[str] = []

        def callback(req):
            resolved.append(req.status.value)

        gate = HITLGate(
            "resolve_gate",
            auto_approve_after=0.05,
            interactive=False,
            on_resolve=callback,
        )
        gate.request_approval({}, {})
        assert len(resolved) == 1
        assert resolved[0] in ("auto_approved", "approved")


# ---------------------------------------------------------------------------
# HITL integration with DAGOrchestrator
# ---------------------------------------------------------------------------


class TestHITLInDAG:
    def test_gate_approved_pipeline_continues(self):
        gate = HITLGate("approval_gate", auto_approve_after=0.1, interactive=False)
        dag = DAGOrchestrator()
        dag.add_node(AgentNode(agent=EchoAgent(), hitl_gate=gate))
        results = dag.run(context={"msg": "approved!"})
        assert results["echo"].success is True
        assert results["echo"].data["message"] == "approved!"

    def test_gate_rejected_marks_agent_failed(self):
        gate = HITLGate("reject_gate", auto_approve_after=5.0, interactive=False)

        def _reject():
            time.sleep(0.05)
            for req in gate.pending:
                gate.reject(req.gate_id, approver="reviewer", reason="Needs rework")

        t = threading.Thread(target=_reject, daemon=True)
        t.start()

        dag = DAGOrchestrator()
        dag.add_node(AgentNode(agent=EchoAgent(), hitl_gate=gate))
        results = dag.run(context={"msg": "test"})
        t.join(timeout=2.0)

        assert results["echo"].success is False
        assert "reject_gate" in results["echo"].error
        assert "reviewer" in results["echo"].error

    def test_gate_rejection_blocks_downstream(self):
        gate = HITLGate("block_gate", auto_approve_after=5.0, interactive=False)

        def _reject():
            time.sleep(0.05)
            for req in gate.pending:
                gate.reject(req.gate_id, approver="sys")

        t = threading.Thread(target=_reject, daemon=True)
        t.start()

        class DownstreamAgent(BaseAgent):
            name = "downstream"

            def execute(self, agent_input):
                return {"status": "ran"}

        dag = DAGOrchestrator()
        dag.add_node(AgentNode(agent=EchoAgent(), hitl_gate=gate))
        dag.add_node(AgentNode(agent=DownstreamAgent(), depends_on=["echo"]))
        results = dag.run(context={"msg": "hi"})
        t.join(timeout=2.0)

        assert results["echo"].success is False
        assert results["downstream"].success is False
        assert "Skipped" in results["downstream"].error

    def test_gate_not_triggered_on_agent_failure(self):
        called: list[bool] = []

        def callback(_req):
            called.append(True)

        gate = HITLGate(
            "no_trigger_gate",
            auto_approve_after=0.1,
            interactive=False,
            on_request=callback,
        )

        dag = DAGOrchestrator()
        dag.add_node(AgentNode(agent=FailAgent(), hitl_gate=gate))
        results = dag.run()
        assert results["fail"].success is False
        # Gate should NOT be triggered when agent fails
        assert len(called) == 0
