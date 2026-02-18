"""Eval demo: Agent evaluation and benchmarking framework.

Shows how to:
1. Define test cases with custom judge functions
2. Evaluate an agent for pass rate and score
3. Run benchmark for P50/P95/P99 latency
4. Compare multiple agents on the same suite
5. Export reports to JSON

Run from project root:
    python examples/eval_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator import AgentInput, AgentEvaluator, BenchmarkSuite, TestCase
from orchestrator.base_agent import BaseAgent


# --- Agents to evaluate ---


class MathAgent(BaseAgent):
    """Performs basic arithmetic. Fast and reliable."""

    name = "math_agent"

    def execute(self, agent_input: AgentInput) -> dict:
        a = agent_input.context.get("a", 0)
        b = agent_input.context.get("b", 0)
        op = agent_input.context.get("op", "add")

        if op == "add":
            result = a + b
        elif op == "multiply":
            result = a * b
        elif op == "subtract":
            result = a - b
        else:
            raise ValueError(f"Unknown op: {op}")

        return {
            "result": result,
            "op": op,
            "_token_usage": 5,
            "_cost_usd": 0.00001,
        }


class SloppyMathAgent(BaseAgent):
    """Gets some answers wrong — demonstrates partial pass rates."""

    name = "sloppy_math"

    def execute(self, agent_input: AgentInput) -> dict:
        a = agent_input.context.get("a", 0)
        b = agent_input.context.get("b", 0)
        op = agent_input.context.get("op", "add")

        # Deliberately wrong on multiply
        if op == "multiply":
            result = a + b  # Bug: uses add instead of multiply
        else:
            result = eval(f"{a} {'+' if op == 'add' else '-'} {b}")

        return {"result": result, "op": op}


# --- Test suite ---


def build_test_cases() -> list[TestCase]:
    return [
        TestCase(
            id="add_small",
            description="Basic addition",
            input_context={"a": 3, "b": 4, "op": "add"},
            judge=lambda out: out.get("result") == 7,
            tags=["arithmetic", "smoke"],
        ),
        TestCase(
            id="add_large",
            description="Large number addition",
            input_context={"a": 1_000_000, "b": 999_999, "op": "add"},
            judge=lambda out: out.get("result") == 1_999_999,
            tags=["arithmetic"],
        ),
        TestCase(
            id="multiply",
            description="Multiplication",
            input_context={"a": 12, "b": 15, "op": "multiply"},
            judge=lambda out: out.get("result") == 180,
            tags=["arithmetic", "smoke"],
        ),
        TestCase(
            id="subtract",
            description="Subtraction",
            input_context={"a": 100, "b": 37, "op": "subtract"},
            judge=lambda out: out.get("result") == 63,
            tags=["arithmetic"],
        ),
        TestCase(
            id="output_has_op",
            description="Output always includes op field",
            expected_keys=["result", "op"],
            input_context={"a": 1, "b": 1, "op": "add"},
            tags=["schema"],
        ),
        # Partial credit example
        TestCase(
            id="add_negative",
            description="Negative number addition (partial credit if result is numeric)",
            input_context={"a": -5, "b": 3, "op": "add"},
            judge=lambda out: (
                out.get("result") == -2,
                1.0 if out.get("result") == -2 else (0.5 if isinstance(out.get("result"), (int, float)) else 0.0),
            ),
            tags=["arithmetic", "edge"],
        ),
    ]


def main():
    cases = build_test_cases()

    # -----------------------------------------------------------------------
    # 1. Evaluate MathAgent
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  EVALUATION: MathAgent")
    print("=" * 60)

    evaluator = AgentEvaluator(MathAgent())
    report = evaluator.evaluate(cases)
    print(report.summary())

    if report.failures():
        print("\n  Failed cases:")
        for r in report.failures():
            print(f"    ✗ {r.test_case_id}: {r.error or 'judge returned False'}")

    # -----------------------------------------------------------------------
    # 2. Evaluate SloppyMathAgent
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  EVALUATION: SloppyMathAgent")
    print("=" * 60)

    sloppy_evaluator = AgentEvaluator(SloppyMathAgent())
    sloppy_report = sloppy_evaluator.evaluate(cases)
    print(sloppy_report.summary())

    # -----------------------------------------------------------------------
    # 3. Filter by tag
    # -----------------------------------------------------------------------
    print("\n  Smoke tests only (MathAgent):")
    smoke_results = report.results_by_tag("smoke", cases)
    smoke_pass = sum(1 for r in smoke_results if r.passed)
    print(f"  {smoke_pass}/{len(smoke_results)} smoke tests passed")

    # -----------------------------------------------------------------------
    # 4. Benchmark (latency)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  BENCHMARK: MathAgent (5 runs)")
    print("=" * 60)

    bench = evaluator.benchmark(cases, runs=5)
    print(bench.summary())

    # -----------------------------------------------------------------------
    # 5. Compare two agents
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  COMPARISON: MathAgent vs SloppyMathAgent")
    print("=" * 60)

    comparison = evaluator.compare(SloppyMathAgent(), cases)
    print(comparison)

    # -----------------------------------------------------------------------
    # 6. BenchmarkSuite leaderboard
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  LEADERBOARD: All agents")
    print("=" * 60)

    suite = BenchmarkSuite(cases)
    suite.add(MathAgent()).add(SloppyMathAgent())
    print(suite.leaderboard())

    # -----------------------------------------------------------------------
    # 7. Export report
    # -----------------------------------------------------------------------
    report.export_json("eval_report.json")
    print("\n  Report exported to eval_report.json")


if __name__ == "__main__":
    main()
