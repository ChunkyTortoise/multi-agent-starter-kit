"""Tests for the Agent Evaluation and Benchmarking framework."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator import AgentInput, AgentEvaluator, BenchmarkSuite, EvalReport, TestCase
from orchestrator.base_agent import BaseAgent
from orchestrator.eval import BenchmarkResult, EvalResult


# --- Test agents ---


class AddAgent(BaseAgent):
    name = "add"

    def execute(self, agent_input: AgentInput) -> dict:
        a = agent_input.context.get("a", 0)
        b = agent_input.context.get("b", 0)
        return {"sum": a + b, "_token_usage": 10, "_cost_usd": 0.001}


class EchoAgent(BaseAgent):
    name = "echo"

    def execute(self, agent_input: AgentInput) -> dict:
        return {"message": agent_input.context.get("msg", "")}


class SlowAgent(BaseAgent):
    name = "slow"

    def execute(self, agent_input: AgentInput) -> dict:
        import time
        time.sleep(0.01)
        return {"result": "done"}


class FailingAgent(BaseAgent):
    name = "failing"
    max_retries = 0

    def execute(self, agent_input: AgentInput) -> dict:
        raise ValueError("always fails")


# ---------------------------------------------------------------------------
# TestCase tests
# ---------------------------------------------------------------------------


class TestTestCase:
    def test_default_judge_all_keys_present(self):
        tc = TestCase(id="tc1", expected_keys=["sum", "extra"])
        result = tc.default_judge({"sum": 5, "extra": 10})
        passed, score = result
        assert passed is True
        assert score == 1.0

    def test_default_judge_missing_key(self):
        tc = TestCase(id="tc2", expected_keys=["sum", "extra"])
        passed, score = tc.default_judge({"sum": 5})
        assert passed is False
        assert score == 0.5

    def test_default_judge_no_expected_keys(self):
        tc = TestCase(id="tc3")
        passed, score = tc.default_judge({"anything": "ok"})
        assert passed is True
        assert score == 1.0

    def test_default_judge_empty_output(self):
        tc = TestCase(id="tc4", expected_keys=["required"])
        passed, score = tc.default_judge({})
        assert passed is False
        assert score == 0.0


# ---------------------------------------------------------------------------
# EvalResult tests
# ---------------------------------------------------------------------------


class TestEvalResult:
    def test_to_dict(self):
        result = EvalResult(
            test_case_id="tc1",
            agent_name="add",
            passed=True,
            score=1.0,
            duration_ms=5.0,
            cost_usd=0.001,
            tokens=10,
        )
        d = result.to_dict()
        assert d["test_case_id"] == "tc1"
        assert d["passed"] is True
        assert d["score"] == 1.0
        assert d["duration_ms"] == 5.0


# ---------------------------------------------------------------------------
# AgentEvaluator tests
# ---------------------------------------------------------------------------


class TestAgentEvaluator:
    def _cases(self):
        return [
            TestCase(
                id="add_3_4",
                input_context={"a": 3, "b": 4},
                judge=lambda out: out.get("sum") == 7,
            ),
            TestCase(
                id="add_0_0",
                input_context={"a": 0, "b": 0},
                judge=lambda out: out.get("sum") == 0,
            ),
            TestCase(
                id="add_neg",
                input_context={"a": -5, "b": 3},
                judge=lambda out: out.get("sum") == -2,
            ),
        ]

    def test_all_pass(self):
        evaluator = AgentEvaluator(AddAgent())
        report = evaluator.evaluate(self._cases())
        assert report.pass_rate == 1.0
        assert report.avg_score == 1.0

    def test_report_agent_name(self):
        report = AgentEvaluator(AddAgent()).evaluate(self._cases())
        assert report.agent_name == "add"

    def test_result_count(self):
        report = AgentEvaluator(AddAgent()).evaluate(self._cases())
        assert len(report.results) == 3

    def test_failing_agent_all_fail(self):
        cases = [TestCase(id="f1", expected_keys=["result"])]
        report = AgentEvaluator(FailingAgent()).evaluate(cases)
        assert report.pass_rate == 0.0
        assert report.failures()[0].error is not None

    def test_partial_pass(self):
        cases = [
            TestCase(id="pass", input_context={"a": 2, "b": 3}, judge=lambda o: o.get("sum") == 5),
            TestCase(id="fail", input_context={"a": 2, "b": 3}, judge=lambda o: o.get("sum") == 99),
        ]
        report = AgentEvaluator(AddAgent()).evaluate(cases)
        assert report.pass_rate == 0.5

    def test_judge_returns_tuple(self):
        """Judge can return (passed, score) for partial credit."""
        cases = [
            TestCase(
                id="partial",
                input_context={"a": 5, "b": 5},
                judge=lambda o: (o.get("sum") == 10, 0.75),
            )
        ]
        report = AgentEvaluator(AddAgent()).evaluate(cases)
        assert report.results[0].score == 0.75

    def test_expected_keys_judge(self):
        cases = [TestCase(id="keys", expected_keys=["sum"])]
        report = AgentEvaluator(AddAgent()).evaluate(cases)
        assert report.pass_rate == 1.0

    def test_cost_and_tokens_tracked(self):
        cases = [TestCase(id="cost", input_context={"a": 1, "b": 2})]
        report = AgentEvaluator(AddAgent()).evaluate(cases)
        assert report.total_tokens == 10
        assert report.total_cost_usd > 0

    def test_duration_positive(self):
        cases = [TestCase(id="dur", input_context={"a": 1, "b": 2})]
        report = AgentEvaluator(AddAgent()).evaluate(cases)
        assert report.avg_duration_ms >= 0

    def test_failures_list(self):
        cases = [
            TestCase(id="ok", judge=lambda o: True),
            TestCase(id="bad", judge=lambda o: False),
        ]
        report = AgentEvaluator(EchoAgent()).evaluate(cases)
        failures = report.failures()
        assert len(failures) == 1
        assert failures[0].test_case_id == "bad"

    def test_summary_string(self):
        report = AgentEvaluator(AddAgent()).evaluate(self._cases())
        summary = report.summary()
        assert "add" in summary
        assert "100" in summary  # "100.0%" on Python 3.14

    def test_export_json(self):
        report = AgentEvaluator(AddAgent()).evaluate(self._cases())
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            report.export_json(f.name)
            data = json.loads(Path(f.name).read_text())
        assert data["agent_name"] == "add"
        assert "results" in data
        assert len(data["results"]) == 3

    def test_to_dict(self):
        report = AgentEvaluator(AddAgent()).evaluate(self._cases())
        d = report.to_dict()
        assert "pass_rate" in d
        assert "avg_score" in d
        assert "results" in d

    def test_results_by_tag(self):
        cases = [
            TestCase(id="t1", tags=["smoke"], judge=lambda o: True),
            TestCase(id="t2", tags=["regression"], judge=lambda o: True),
            TestCase(id="t3", tags=["smoke"], judge=lambda o: True),
        ]
        report = AgentEvaluator(EchoAgent()).evaluate(cases)
        smoke_results = report.results_by_tag("smoke", cases)
        assert len(smoke_results) == 2

    def test_compare(self):
        cases = [TestCase(id="c1", judge=lambda o: True)]
        evaluator = AgentEvaluator(AddAgent())
        comparison = evaluator.compare(EchoAgent(), cases)
        assert "add" in comparison
        assert "echo" in comparison
        assert "Winner" in comparison


# ---------------------------------------------------------------------------
# BenchmarkResult tests
# ---------------------------------------------------------------------------


class TestBenchmarkResult:
    def test_summary(self):
        result = BenchmarkResult(
            agent_name="test",
            runs=3,
            test_case_count=2,
            p50_ms=10.0,
            p95_ms=50.0,
            p99_ms=100.0,
            min_ms=5.0,
            max_ms=120.0,
            error_rate=0.0,
            avg_cost_usd=0.001,
        )
        summary = result.summary()
        assert "test" in summary
        assert "P50" in summary
        assert "P95" in summary
        assert "P99" in summary

    def test_to_dict(self):
        result = BenchmarkResult(
            agent_name="bench",
            runs=5,
            test_case_count=3,
            p50_ms=10.0,
            p95_ms=20.0,
            p99_ms=30.0,
            min_ms=5.0,
            max_ms=50.0,
            error_rate=0.1,
            avg_cost_usd=0.005,
        )
        d = result.to_dict()
        assert d["p50_ms"] == 10.0
        assert d["error_rate"] == 0.1


# ---------------------------------------------------------------------------
# AgentEvaluator.benchmark tests
# ---------------------------------------------------------------------------


class TestBenchmark:
    def _cases(self):
        return [
            TestCase(id="b1", input_context={"a": 1, "b": 2}),
            TestCase(id="b2", input_context={"a": 3, "b": 4}),
        ]

    def test_benchmark_runs_count(self):
        bench = AgentEvaluator(AddAgent()).benchmark(self._cases(), runs=3)
        # 3 runs × 2 cases = 6 total observations
        assert bench.runs == 3
        assert bench.test_case_count == 2

    def test_percentile_ordering(self):
        bench = AgentEvaluator(AddAgent()).benchmark(self._cases(), runs=5)
        assert bench.p50_ms <= bench.p95_ms <= bench.p99_ms
        assert bench.min_ms <= bench.p50_ms

    def test_error_rate_zero_for_good_agent(self):
        bench = AgentEvaluator(AddAgent()).benchmark(self._cases(), runs=3)
        assert bench.error_rate == 0.0

    def test_error_rate_one_for_failing_agent(self):
        bench = AgentEvaluator(FailingAgent()).benchmark(self._cases(), runs=3)
        assert bench.error_rate == 1.0

    def test_latency_positive(self):
        bench = AgentEvaluator(SlowAgent()).benchmark([TestCase(id="slow")], runs=3)
        assert bench.p50_ms > 0


# ---------------------------------------------------------------------------
# BenchmarkSuite tests
# ---------------------------------------------------------------------------


class TestBenchmarkSuite:
    def _cases(self):
        return [
            TestCase(id="s1", judge=lambda o: True),
            TestCase(id="s2", judge=lambda o: True),
        ]

    def test_add_returns_self(self):
        suite = BenchmarkSuite(self._cases())
        returned = suite.add(AddAgent())
        assert returned is suite

    def test_run_returns_reports(self):
        suite = BenchmarkSuite(self._cases())
        suite.add(AddAgent()).add(EchoAgent())
        reports = suite.run()
        assert "add" in reports
        assert "echo" in reports

    def test_leaderboard_with_agents(self):
        suite = BenchmarkSuite(self._cases())
        suite.add(AddAgent()).add(EchoAgent())
        leaderboard = suite.leaderboard()
        assert "add" in leaderboard
        assert "echo" in leaderboard
        assert "Leaderboard" in leaderboard

    def test_leaderboard_triggers_run(self):
        suite = BenchmarkSuite(self._cases())
        suite.add(AddAgent())
        # Should auto-run on first call to leaderboard
        leaderboard = suite.leaderboard()
        assert "add" in leaderboard

    def test_single_agent_leaderboard(self):
        suite = BenchmarkSuite(self._cases())
        suite.add(EchoAgent())
        leaderboard = suite.leaderboard()
        assert "echo" in leaderboard
