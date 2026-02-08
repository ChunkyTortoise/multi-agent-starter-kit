"""Multi-Agent Orchestration Starter Kit — DAG-based agent orchestration."""

from orchestrator.dag import DAGOrchestrator, AgentNode
from orchestrator.base_agent import BaseAgent, AgentInput, AgentOutput
from orchestrator.monitor import Monitor

__version__ = "1.0.0"
__all__ = [
    "DAGOrchestrator",
    "AgentNode",
    "BaseAgent",
    "AgentInput",
    "AgentOutput",
    "Monitor",
]
