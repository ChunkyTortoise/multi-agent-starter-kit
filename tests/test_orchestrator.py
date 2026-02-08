"""Tests for the DAG orchestrator, BaseAgent, and Monitor."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator import (
    DAGOrchestrator,
    AgentNode,
    BaseAgent,
    AgentInput,
    AgentOutput,
    Monitor,
)


# --- Test agents ---


class AddAgent(BaseAgent):
    name = "add"

    def execute(self, agent_input: AgentInput) -> dict:
        a = agent_input.context.get("a", 0)
        b = agent_input.context.get("b", 0)
        return {"sum": a + b}


class DoubleAgent(BaseAgent):
    name = "double"

    def execute(self, agent_input: AgentInput) -> dict:
        upstream_sum = agent_input.upstream_results["add"]["sum"]
        return {"doubled": upstream_sum * 2}


class FailingAgent(BaseAgent):
    name = "failing"
    max_retries = 0

    def execute(self, agent_input: AgentInput) -> dict:
        raise ValueError("Intentional failure")


class TokenAgent(BaseAgent):
    name = "token_tracker"

    def execute(self, agent_input: AgentInput) -> dict:
        return {"answer": "hello", "_token_usage": 150, "_cost_usd": 0.0045}


# --- BaseAgent tests ---


class TestBaseAgent:
    def test_agent_run_success(self):
        agent = AddAgent()
        result = agent.run(AgentInput(context={"a": 3, "b": 7}))
        assert result.success is True
        assert result.data["sum"] == 10
        assert result.agent_name == "add"
        assert result.duration_ms > 0

    def test_agent_run_failure(self):
        agent = FailingAgent()
        result = agent.run(AgentInput())
        assert result.success is False
        assert "Intentional failure" in result.error
        assert result.duration_ms >= 0

    def test_agent_token_tracking(self):
        agent = TokenAgent()
        result = agent.run(AgentInput())
        assert result.token_usage == 150
        assert result.estimated_cost_usd == 0.0045

    def test_agent_repr(self):
        agent = AddAgent()
        assert "AddAgent" in repr(agent)
        assert "add" in repr(agent)


# --- AgentInput/AgentOutput tests ---


class TestModels:
    def test_agent_input_defaults(self):
        inp = AgentInput()
        assert inp.context == {}
        assert inp.upstream_results == {}

    def test_agent_output_defaults(self):
        out = AgentOutput(agent_name="test")
        assert out.success is True
        assert out.data == {}
        assert out.error is None
        assert out.token_usage == 0


# --- DAGOrchestrator tests ---


class TestDAGOrchestrator:
    def test_single_agent(self):
        dag = DAGOrchestrator()
        dag.add_node(AgentNode(agent=AddAgent()))
        results = dag.run(context={"a": 5, "b": 3})
        assert results["add"].success is True
        assert results["add"].data["sum"] == 8

    def test_chained_agents(self):
        dag = DAGOrchestrator()
        dag.add_node(AgentNode(agent=AddAgent()))
        dag.add_node(AgentNode(agent=DoubleAgent(), depends_on=["add"]))
        results = dag.run(context={"a": 4, "b": 6})
        assert results["add"].data["sum"] == 10
        assert results["double"].data["doubled"] == 20

    def test_add_node_chaining(self):
        dag = DAGOrchestrator()
        returned = dag.add_node(AgentNode(agent=AddAgent()))
        assert returned is dag

    def test_duplicate_agent_raises(self):
        dag = DAGOrchestrator()
        dag.add_node(AgentNode(agent=AddAgent()))
        try:
            dag.add_node(AgentNode(agent=AddAgent()))
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "already exists" in str(e)

    def test_missing_dependency_raises(self):
        dag = DAGOrchestrator()
        dag.add_node(AgentNode(agent=DoubleAgent(), depends_on=["nonexistent"]))
        try:
            dag.run()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "nonexistent" in str(e)

    def test_cycle_detection(self):
        class AgentA(BaseAgent):
            name = "a"

            def execute(self, agent_input):
                return {}

        class AgentB(BaseAgent):
            name = "b"

            def execute(self, agent_input):
                return {}

        dag = DAGOrchestrator()
        dag.add_node(AgentNode(agent=AgentA(), depends_on=["b"]))
        dag.add_node(AgentNode(agent=AgentB(), depends_on=["a"]))
        try:
            dag.run()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Cycle" in str(e)

    def test_optional_agent_failure(self):
        dag = DAGOrchestrator()
        dag.add_node(AgentNode(agent=FailingAgent(), optional=True))
        dag.add_node(AgentNode(agent=AddAgent()))
        results = dag.run(context={"a": 1, "b": 2})
        assert results["failing"].success is False
        assert results["add"].success is True
        assert results["add"].data["sum"] == 3

    def test_required_dependency_failure_skips_downstream(self):
        dag = DAGOrchestrator()
        dag.add_node(AgentNode(agent=FailingAgent()))
        dag.add_node(AgentNode(agent=DoubleAgent(), depends_on=["failing"]))
        results = dag.run()
        assert results["failing"].success is False
        assert results["double"].success is False
        assert "Skipped" in results["double"].error

    def test_timeline(self):
        dag = DAGOrchestrator()
        dag.add_node(AgentNode(agent=AddAgent()))
        dag.run(context={"a": 1, "b": 1})
        assert len(dag.timeline) == 1
        assert dag.timeline[0]["agent"] == "add"
        assert dag.timeline[0]["success"] is True

    def test_total_cost_and_tokens(self):
        dag = DAGOrchestrator()
        dag.add_node(AgentNode(agent=TokenAgent()))
        dag.run()
        assert dag.total_tokens == 150
        assert dag.total_cost == 0.0045

    def test_summary(self):
        dag = DAGOrchestrator()
        dag.add_node(AgentNode(agent=AddAgent()))
        dag.run(context={"a": 1, "b": 2})
        summary = dag.summary()
        assert "add" in summary
        assert "PASS" in summary


# --- Monitor tests ---


class TestMonitor:
    def _make_run(self):
        dag = DAGOrchestrator()
        dag.add_node(AgentNode(agent=AddAgent()))
        dag.add_node(AgentNode(agent=DoubleAgent(), depends_on=["add"]))
        results = dag.run(context={"a": 3, "b": 4})
        return results, dag.timeline

    def test_no_runs(self):
        monitor = Monitor()
        assert "No runs" in monitor.dashboard()

    def test_record_and_dashboard(self):
        monitor = Monitor()
        results, timeline = self._make_run()
        monitor.record(results, timeline)
        dashboard = monitor.dashboard()
        assert "AGENT PIPELINE MONITOR" in dashboard
        assert "add" in dashboard
        assert "double" in dashboard
        assert "PASS" in dashboard

    def test_export_json(self):
        monitor = Monitor()
        results, timeline = self._make_run()
        monitor.record(results, timeline)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            monitor.export_json(f.name)
            data = json.loads(Path(f.name).read_text())
        assert "timeline" in data
        assert "results" in data
        assert "add" in data["results"]

    def test_compare_runs_needs_two(self):
        monitor = Monitor()
        assert "Need at least 2" in monitor.compare_runs()

    def test_compare_runs(self):
        monitor = Monitor()
        for _ in range(2):
            results, timeline = self._make_run()
            monitor.record(results, timeline)
        comparison = monitor.compare_runs()
        assert "Run 1" in comparison
        assert "Run 2" in comparison

    def test_gantt_chart(self):
        monitor = Monitor()
        results, timeline = self._make_run()
        monitor.record(results, timeline)
        dashboard = monitor.dashboard()
        assert "#" in dashboard  # Gantt bar characters
