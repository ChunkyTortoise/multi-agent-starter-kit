"""DAG Orchestrator — Directed Acyclic Graph execution engine for agents.

Defines agent pipelines as a DAG. Handles:
- Topological sort for execution order
- Retry with exponential backoff
- Per-agent timeouts
- Graceful degradation (optional agents can fail without killing the pipeline)
- Result passing between dependent agents
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from orchestrator.base_agent import AgentInput, AgentOutput, BaseAgent

logger = logging.getLogger(__name__)

# Avoid circular import: HITLGate is referenced by string type only at class
# definition time, imported lazily when actually used.
_HITLGate = None


def _get_hitl_class():
    global _HITLGate
    if _HITLGate is None:
        from orchestrator.hitl import HITLGate  # noqa: PLC0415
        _HITLGate = HITLGate
    return _HITLGate


@dataclass
class AgentNode:
    """A node in the orchestration DAG.

    Args:
        agent: The agent instance to execute.
        depends_on: Names of agents that must complete before this one runs.
        optional: If True, failure won't halt downstream agents.
        retry_count: Override the agent's default max_retries.
        retry_delay: Base delay in seconds between retries (doubles each attempt).
        hitl_gate: Optional HITLGate to pause for human approval after this
                   agent succeeds. Rejection marks the result as failed.
    """

    agent: BaseAgent
    depends_on: list[str] = field(default_factory=list)
    optional: bool = False
    retry_count: int | None = None
    retry_delay: float = 1.0
    hitl_gate: Any = None  # HITLGate | None — typed as Any to avoid circular import

    @property
    def name(self) -> str:
        return self.agent.name


class DAGOrchestrator:
    """Execute a pipeline of agents respecting dependency order.

    Usage:
        orchestrator = DAGOrchestrator()
        orchestrator.add_node(AgentNode(agent=ResearchAgent()))
        orchestrator.add_node(AgentNode(agent=AnalysisAgent(), depends_on=["research"]))
        orchestrator.add_node(AgentNode(agent=ReportAgent(), depends_on=["analysis"]))
        results = orchestrator.run(context={"query": "AI trends"})
    """

    def __init__(self) -> None:
        self._nodes: dict[str, AgentNode] = {}
        self._results: dict[str, AgentOutput] = {}
        self._execution_order: list[str] = []
        self._timeline: list[dict[str, Any]] = []

    def add_node(self, node: AgentNode) -> DAGOrchestrator:
        """Add an agent node to the DAG. Returns self for chaining."""
        if node.name in self._nodes:
            raise ValueError(f"Agent '{node.name}' already exists in the DAG")
        self._nodes[node.name] = node
        return self

    def _topological_sort(self) -> list[str]:
        """Kahn's algorithm for topological ordering."""
        in_degree: dict[str, int] = {name: 0 for name in self._nodes}
        adjacency: dict[str, list[str]] = {name: [] for name in self._nodes}

        for name, node in self._nodes.items():
            for dep in node.depends_on:
                if dep not in self._nodes:
                    raise ValueError(
                        f"Agent '{name}' depends on '{dep}' which is not in the DAG"
                    )
                adjacency[dep].append(name)
                in_degree[name] += 1

        queue = [name for name, degree in in_degree.items() if degree == 0]
        order: list[str] = []

        while queue:
            queue.sort()  # Deterministic ordering
            current = queue.pop(0)
            order.append(current)
            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self._nodes):
            raise ValueError("Cycle detected in the DAG — check your depends_on fields")

        return order

    def _execute_with_retry(
        self, node: AgentNode, agent_input: AgentInput
    ) -> AgentOutput:
        """Execute an agent with exponential backoff retry."""
        max_retries = (
            node.retry_count if node.retry_count is not None else node.agent.max_retries
        )
        last_result: AgentOutput | None = None

        for attempt in range(max_retries + 1):
            if attempt > 0:
                delay = node.retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"[{node.name}] Retry {attempt}/{max_retries} after {delay:.1f}s"
                )
                time.sleep(delay)

            result = node.agent.run(agent_input)
            last_result = result

            if result.success:
                return result

        return last_result  # type: ignore[return-value]

    def run(self, context: dict[str, Any] | None = None) -> dict[str, AgentOutput]:
        """Execute the full pipeline in topological order.

        Args:
            context: Initial context dict passed to all agents.

        Returns:
            Dict mapping agent names to their AgentOutput results.
        """
        self._results = {}
        self._timeline = []
        self._execution_order = self._topological_sort()
        context = context or {}
        pipeline_start = time.perf_counter()

        logger.info(f"Pipeline execution order: {' -> '.join(self._execution_order)}")

        for agent_name in self._execution_order:
            node = self._nodes[agent_name]

            # Check if dependencies succeeded
            failed_deps = [
                dep
                for dep in node.depends_on
                if dep in self._results and not self._results[dep].success
            ]
            if failed_deps and not node.optional:
                skip_deps = [d for d in failed_deps if not self._nodes[d].optional]
                if skip_deps:
                    logger.warning(
                        f"[{agent_name}] Skipping — required dependency failed: {skip_deps}"
                    )
                    self._results[agent_name] = AgentOutput(
                        agent_name=agent_name,
                        success=False,
                        error=f"Skipped due to failed dependencies: {skip_deps}",
                    )
                    continue

            # Build input with upstream results
            upstream = {
                dep: self._results[dep].data
                for dep in node.depends_on
                if dep in self._results and self._results[dep].success
            }
            agent_input = AgentInput(context=context, upstream_results=upstream)

            # Execute
            step_start = time.perf_counter()
            result = self._execute_with_retry(node, agent_input)
            step_end = time.perf_counter()

            # HITL gate: pause for human approval if configured
            if result.success and node.hitl_gate is not None:
                approval = node.hitl_gate.request_approval(
                    context=context,
                    agent_output=result.data,
                )
                from orchestrator.hitl import ApprovalStatus  # noqa: PLC0415
                if approval.status == ApprovalStatus.REJECTED:
                    result = AgentOutput(
                        agent_name=agent_name,
                        success=False,
                        error=f"HITL gate '{node.hitl_gate.name}' rejected by "
                              f"'{approval.approver}': {approval.notes}",
                        duration_ms=result.duration_ms,
                    )

            self._results[agent_name] = result
            self._timeline.append(
                {
                    "agent": agent_name,
                    "start": step_start - pipeline_start,
                    "end": step_end - pipeline_start,
                    "success": result.success,
                    "duration_ms": result.duration_ms,
                }
            )

            status = "OK" if result.success else "FAILED"
            if not result.success and node.optional:
                status = "FAILED (optional)"
            logger.info(f"[{agent_name}] {status} — {result.duration_ms:.1f}ms")

        total_ms = (time.perf_counter() - pipeline_start) * 1000
        succeeded = sum(1 for r in self._results.values() if r.success)
        logger.info(
            f"Pipeline complete: {succeeded}/{len(self._results)} agents succeeded "
            f"in {total_ms:.0f}ms"
        )

        return self._results

    @property
    def timeline(self) -> list[dict[str, Any]]:
        """Execution timeline for monitoring/visualization."""
        return self._timeline

    @property
    def total_cost(self) -> float:
        """Sum of estimated costs across all agents."""
        return sum(r.estimated_cost_usd for r in self._results.values())

    @property
    def total_tokens(self) -> int:
        """Sum of token usage across all agents."""
        return sum(r.token_usage for r in self._results.values())

    def summary(self) -> str:
        """Human-readable execution summary."""
        lines = ["Pipeline Summary", "=" * 40]
        for name in self._execution_order:
            result = self._results.get(name)
            if not result:
                lines.append(f"  {name}: NOT RUN")
                continue
            status = "PASS" if result.success else "FAIL"
            lines.append(f"  {name}: {status} ({result.duration_ms:.0f}ms)")
            if result.error:
                lines.append(f"    Error: {result.error}")
        lines.append("=" * 40)
        lines.append(f"Total tokens: {self.total_tokens}")
        lines.append(f"Total cost: ${self.total_cost:.4f}")
        return "\n".join(lines)
