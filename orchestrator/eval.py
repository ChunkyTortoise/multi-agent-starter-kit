"""Eval — Agent evaluation and benchmarking framework.

Measure agent quality systematically: accuracy, latency, cost, and reliability.
Run deterministic test suites, score with custom judge functions, and export
structured reports — all without external dependencies.

Usage:
    # Define test cases
    cases = [
        TestCase(id="tc1", input_context={"query": "Hello"}, expected_keys=["answer"]),
        TestCase(id="tc2", input_context={"a": 3, "b": 4}, judge=lambda out: out.get("sum") == 7),
    ]

    # Evaluate
    evaluator = AgentEvaluator(agent=MyAgent())
    report = evaluator.evaluate(cases)
    print(report.summary())

    # Benchmark (latency + reliability)
    bench = evaluator.benchmark(cases, runs=5)
    print(bench.summary())
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from orchestrator.base_agent import AgentInput, AgentOutput, BaseAgent

# Judge function signature: receives agent output dict, returns pass/fail + score
JudgeFn = Callable[[dict[str, Any]], bool | tuple[bool, float]]


# ---------------------------------------------------------------------------
# Test case
# ---------------------------------------------------------------------------


@dataclass
class TestCase:
    """A single evaluation test case.

    Args:
        id: Unique identifier for this test case.
        input_context: Dict passed to AgentInput.context.
        expected_keys: Keys that must be present in agent output.data.
        judge: Custom function that receives output.data and returns True/False
               or a (passed, score) tuple for partial credit.
        tags: Labels for grouping/filtering test cases.
        description: Human-readable explanation of what's being tested.
    """

    id: str
    input_context: dict[str, Any] = field(default_factory=dict)
    expected_keys: list[str] = field(default_factory=list)
    judge: JudgeFn | None = None
    tags: list[str] = field(default_factory=list)
    description: str = ""

    def default_judge(self, output: dict[str, Any]) -> tuple[bool, float]:
        """Check that all expected_keys are present. Returns (passed, score)."""
        if not self.expected_keys:
            return True, 1.0
        present = sum(1 for k in self.expected_keys if k in output)
        score = present / len(self.expected_keys)
        return score == 1.0, round(score, 3)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """The result of running one test case against one agent.

    Attributes:
        test_case_id: ID of the TestCase.
        agent_name: Name of the evaluated agent.
        passed: Whether the judge function returned True.
        score: Numeric score in [0, 1]. 1.0 = perfect, 0.0 = total failure.
        duration_ms: Agent execution time in milliseconds.
        cost_usd: Estimated LLM cost for this run.
        tokens: Token usage for this run.
        actual_output: The agent's output.data dict.
        error: Error message if the agent raised an exception.
    """

    test_case_id: str
    agent_name: str
    passed: bool
    score: float
    duration_ms: float
    cost_usd: float = 0.0
    tokens: int = 0
    actual_output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_case_id": self.test_case_id,
            "agent_name": self.agent_name,
            "passed": self.passed,
            "score": self.score,
            "duration_ms": self.duration_ms,
            "cost_usd": self.cost_usd,
            "tokens": self.tokens,
            "error": self.error,
        }


@dataclass
class EvalReport:
    """Aggregated evaluation report for an agent across all test cases.

    Attributes:
        agent_name: Name of the evaluated agent.
        results: Individual EvalResult per test case.
    """

    agent_name: str
    results: list[EvalResult]

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return round(sum(1 for r in self.results if r.passed) / len(self.results), 3)

    @property
    def avg_score(self) -> float:
        if not self.results:
            return 0.0
        return round(statistics.mean(r.score for r in self.results), 3)

    @property
    def avg_duration_ms(self) -> float:
        if not self.results:
            return 0.0
        return round(statistics.mean(r.duration_ms for r in self.results), 2)

    @property
    def total_cost_usd(self) -> float:
        return round(sum(r.cost_usd for r in self.results), 6)

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens for r in self.results)

    def results_by_tag(self, tag: str, test_cases: list[TestCase]) -> list[EvalResult]:
        """Filter results to only test cases with the given tag."""
        tagged_ids = {tc.id for tc in test_cases if tag in tc.tags}
        return [r for r in self.results if r.test_case_id in tagged_ids]

    def failures(self) -> list[EvalResult]:
        return [r for r in self.results if not r.passed]

    def summary(self) -> str:
        lines = [
            f"Eval Report — {self.agent_name}",
            "=" * 40,
            f"  Test cases : {len(self.results)}",
            f"  Pass rate  : {self.pass_rate:.1%}",
            f"  Avg score  : {self.avg_score:.3f}",
            f"  Avg latency: {self.avg_duration_ms:.1f}ms",
            f"  Total cost : ${self.total_cost_usd:.6f}",
            f"  Total tokens: {self.total_tokens}",
        ]
        if self.failures():
            lines.append(f"\n  Failed cases ({len(self.failures())}):")
            for r in self.failures():
                err = f" — {r.error}" if r.error else ""
                lines.append(f"    ✗ {r.test_case_id}{err}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "pass_rate": self.pass_rate,
            "avg_score": self.avg_score,
            "avg_duration_ms": self.avg_duration_ms,
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
            "results": [r.to_dict() for r in self.results],
        }

    def export_json(self, path: str) -> None:
        """Export report to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Benchmark (multi-run latency analysis)
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Latency and reliability statistics across multiple runs.

    Attributes:
        agent_name: Name of the benchmarked agent.
        runs: Number of runs per test case.
        test_case_count: Number of test cases.
        p50_ms: 50th percentile latency.
        p95_ms: 95th percentile latency.
        p99_ms: 99th percentile latency.
        min_ms: Minimum latency observed.
        max_ms: Maximum latency observed.
        error_rate: Fraction of runs that raised errors.
        avg_cost_usd: Average cost per full run.
    """

    agent_name: str
    runs: int
    test_case_count: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    error_rate: float
    avg_cost_usd: float

    def summary(self) -> str:
        return (
            f"Benchmark — {self.agent_name} "
            f"({self.runs} runs × {self.test_case_count} cases)\n"
            f"  P50: {self.p50_ms:.1f}ms | P95: {self.p95_ms:.1f}ms | P99: {self.p99_ms:.1f}ms\n"
            f"  Min: {self.min_ms:.1f}ms | Max: {self.max_ms:.1f}ms\n"
            f"  Error rate: {self.error_rate:.1%} | Avg cost/run: ${self.avg_cost_usd:.6f}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "runs": self.runs,
            "test_case_count": self.test_case_count,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "error_rate": self.error_rate,
            "avg_cost_usd": self.avg_cost_usd,
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class AgentEvaluator:
    """Evaluate and benchmark an agent against a suite of test cases.

    Args:
        agent: The BaseAgent instance to evaluate.

    Example:
        evaluator = AgentEvaluator(MyAgent())
        report = evaluator.evaluate(test_cases)
        bench  = evaluator.benchmark(test_cases, runs=10)
    """

    def __init__(self, agent: BaseAgent) -> None:
        self.agent = agent

    def evaluate(self, test_cases: list[TestCase]) -> EvalReport:
        """Run each test case once and return an EvalReport.

        Args:
            test_cases: List of TestCase definitions.

        Returns:
            EvalReport with per-case results and aggregate statistics.
        """
        results: list[EvalResult] = []
        for tc in test_cases:
            result = self._run_case(tc)
            results.append(result)
        return EvalReport(agent_name=self.agent.name, results=results)

    def benchmark(
        self,
        test_cases: list[TestCase],
        runs: int = 5,
    ) -> BenchmarkResult:
        """Run each test case `runs` times and compute latency percentiles.

        Args:
            test_cases: List of TestCase definitions.
            runs: Number of repetitions per test case.

        Returns:
            BenchmarkResult with P50/P95/P99 latency and error statistics.
        """
        all_durations: list[float] = []
        all_costs: list[float] = []
        error_count = 0
        total_runs = len(test_cases) * runs

        for _ in range(runs):
            for tc in test_cases:
                result = self._run_case(tc)
                all_durations.append(result.duration_ms)
                all_costs.append(result.cost_usd)
                if result.error:
                    error_count += 1

        all_durations.sort()
        n = len(all_durations)

        def percentile(data: list[float], p: float) -> float:
            idx = max(0, int(math.ceil(p / 100 * n)) - 1)
            return round(data[idx], 2)

        import math

        return BenchmarkResult(
            agent_name=self.agent.name,
            runs=runs,
            test_case_count=len(test_cases),
            p50_ms=percentile(all_durations, 50),
            p95_ms=percentile(all_durations, 95),
            p99_ms=percentile(all_durations, 99),
            min_ms=round(all_durations[0], 2) if all_durations else 0,
            max_ms=round(all_durations[-1], 2) if all_durations else 0,
            error_rate=round(error_count / total_runs, 3) if total_runs else 0,
            avg_cost_usd=round(
                sum(all_costs) / len(all_costs) if all_costs else 0, 6
            ),
        )

    def compare(
        self,
        other_agent: BaseAgent,
        test_cases: list[TestCase],
    ) -> str:
        """Compare this agent against another on the same test suite."""
        report_a = self.evaluate(test_cases)
        report_b = AgentEvaluator(other_agent).evaluate(test_cases)

        winner = "TIE"
        if report_a.avg_score > report_b.avg_score:
            winner = self.agent.name
        elif report_b.avg_score > report_a.avg_score:
            winner = other_agent.name

        lines = [
            f"Agent Comparison: {self.agent.name} vs {other_agent.name}",
            "=" * 50,
            f"  {'Metric':<20} {self.agent.name:<15} {other_agent.name:<15}",
            f"  {'Pass rate':<20} {report_a.pass_rate:.1%}{'':>10} {report_b.pass_rate:.1%}",
            f"  {'Avg score':<20} {report_a.avg_score:.3f}{'':>10} {report_b.avg_score:.3f}",
            f"  {'Avg latency':<20} {report_a.avg_duration_ms:.1f}ms{'':>8} {report_b.avg_duration_ms:.1f}ms",
            f"  {'Total cost':<20} ${report_a.total_cost_usd:.6f}{'':>5} ${report_b.total_cost_usd:.6f}",
            "=" * 50,
            f"  Winner: {winner}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_case(self, tc: TestCase) -> EvalResult:
        """Execute one test case and score the result."""
        agent_input = AgentInput(context=tc.input_context)
        start = time.perf_counter()
        output: AgentOutput = self.agent.run(agent_input)
        duration_ms = (time.perf_counter() - start) * 1000

        if not output.success:
            return EvalResult(
                test_case_id=tc.id,
                agent_name=self.agent.name,
                passed=False,
                score=0.0,
                duration_ms=round(duration_ms, 2),
                cost_usd=output.estimated_cost_usd,
                tokens=output.token_usage,
                actual_output={},
                error=output.error,
            )

        # Score with custom judge or default key-presence check
        if tc.judge:
            judge_result = tc.judge(output.data)
            if isinstance(judge_result, tuple):
                passed, score = judge_result
            else:
                passed = bool(judge_result)
                score = 1.0 if passed else 0.0
        else:
            passed, score = tc.default_judge(output.data)

        return EvalResult(
            test_case_id=tc.id,
            agent_name=self.agent.name,
            passed=passed,
            score=score,
            duration_ms=round(duration_ms, 2),
            cost_usd=output.estimated_cost_usd,
            tokens=output.token_usage,
            actual_output=output.data,
        )


# ---------------------------------------------------------------------------
# Convenience: BenchmarkSuite
# ---------------------------------------------------------------------------


class BenchmarkSuite:
    """Compare multiple agents on the same test suite.

    Args:
        test_cases: Shared test cases for all agents.

    Example:
        suite = BenchmarkSuite(test_cases)
        suite.add(AgentV1())
        suite.add(AgentV2())
        print(suite.leaderboard())
    """

    def __init__(self, test_cases: list[TestCase]) -> None:
        self.test_cases = test_cases
        self._agents: list[BaseAgent] = []
        self._reports: dict[str, EvalReport] = {}

    def add(self, agent: BaseAgent) -> BenchmarkSuite:
        """Add an agent to the suite. Returns self for chaining."""
        self._agents.append(agent)
        return self

    def run(self) -> dict[str, EvalReport]:
        """Evaluate all agents and store reports."""
        for agent in self._agents:
            evaluator = AgentEvaluator(agent)
            self._reports[agent.name] = evaluator.evaluate(self.test_cases)
        return self._reports

    def leaderboard(self) -> str:
        """Ranked leaderboard by average score."""
        if not self._reports:
            self.run()

        ranked = sorted(
            self._reports.values(),
            key=lambda r: (-r.avg_score, r.avg_duration_ms),
        )

        lines = [
            "Benchmark Leaderboard",
            "=" * 60,
            f"  {'Rank':<5} {'Agent':<20} {'Pass%':<10} {'Score':<8} {'Latency':<12} {'Cost'}",
            "-" * 60,
        ]
        for rank, report in enumerate(ranked, 1):
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"  {rank}.")
            lines.append(
                f"  {medal:<5} {report.agent_name:<20} "
                f"{report.pass_rate:.0%}{'':>5} "
                f"{report.avg_score:.3f}{'':>3} "
                f"{report.avg_duration_ms:.1f}ms{'':>5} "
                f"${report.total_cost_usd:.6f}"
            )
        return "\n".join(lines)
