"""Multi-Agent Orchestration Starter Kit — DAG-based agent orchestration."""

from orchestrator.base_agent import AgentInput, AgentOutput, BaseAgent
from orchestrator.dag import AgentNode, DAGOrchestrator
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
