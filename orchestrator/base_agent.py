"""BaseAgent — Abstract base class for all agents in the orchestration pipeline.

Extend this class to build any agent. Provides:
- Pydantic input/output validation
- Execution timing
- Structured logging
- Cost tracking
- Retry-safe interface
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AgentInput(BaseModel):
    """Base input model. Extend with your agent's specific fields."""

    context: dict[str, Any] = Field(default_factory=dict)
    upstream_results: dict[str, Any] = Field(default_factory=dict)


class AgentOutput(BaseModel):
    """Base output model. Every agent returns this structure."""

    agent_name: str
    success: bool = True
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    duration_ms: float = 0.0
    token_usage: int = 0
    estimated_cost_usd: float = 0.0


class BaseAgent(ABC):
    """Abstract base class for pipeline agents.

    Subclass and implement `execute()` to create a new agent.
    The orchestrator calls `run()`, which wraps your execute()
    with timing, error handling, and structured output.

    Example:
        class MyAgent(BaseAgent):
            name = "my_agent"

            def execute(self, agent_input: AgentInput) -> dict:
                result = do_something(agent_input.context["query"])
                return {"answer": result}
    """

    name: str = "base_agent"
    timeout_seconds: float = 30.0
    max_retries: int = 2

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def execute(self, agent_input: AgentInput) -> dict[str, Any]:
        """Core logic. Return a dict that becomes AgentOutput.data.

        Args:
            agent_input: Validated input with context and upstream results.

        Returns:
            Dict of results. Will be wrapped in AgentOutput automatically.
        """
        ...

    def run(self, agent_input: AgentInput) -> AgentOutput:
        """Execute the agent with timing, error handling, and structured output.

        This is what the orchestrator calls. Do not override this —
        override execute() instead.
        """
        start = time.perf_counter()
        try:
            logger.info(f"[{self.name}] Starting execution")
            result = self.execute(agent_input)
            duration = (time.perf_counter() - start) * 1000
            logger.info(f"[{self.name}] Completed in {duration:.1f}ms")
            return AgentOutput(
                agent_name=self.name,
                success=True,
                data=result,
                duration_ms=round(duration, 2),
                token_usage=result.get("_token_usage", 0),
                estimated_cost_usd=result.get("_cost_usd", 0.0),
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            logger.error(f"[{self.name}] Failed after {duration:.1f}ms: {e}")
            return AgentOutput(
                agent_name=self.name,
                success=False,
                error=str(e),
                duration_ms=round(duration, 2),
            )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
