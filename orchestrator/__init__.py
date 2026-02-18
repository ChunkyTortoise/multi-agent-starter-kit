"""Multi-Agent Orchestration Starter Kit — DAG-based agent orchestration."""

from orchestrator.base_agent import AgentInput, AgentOutput, BaseAgent
from orchestrator.dag import AgentNode, DAGOrchestrator
from orchestrator.eval import AgentEvaluator, BenchmarkSuite, EvalReport, EvalResult, TestCase
from orchestrator.hitl import ApprovalRequest, ApprovalStatus, HITLGate
from orchestrator.monitor import Monitor
from orchestrator.rag import Document, KnowledgeBase, RAGAgent, RetrievalResult

__version__ = "2.0.0"
__all__ = [
    # Core orchestration
    "DAGOrchestrator",
    "AgentNode",
    "BaseAgent",
    "AgentInput",
    "AgentOutput",
    "Monitor",
    # Human-in-the-loop
    "HITLGate",
    "ApprovalRequest",
    "ApprovalStatus",
    # Agentic RAG
    "RAGAgent",
    "KnowledgeBase",
    "Document",
    "RetrievalResult",
    # Evaluation
    "AgentEvaluator",
    "BenchmarkSuite",
    "EvalReport",
    "EvalResult",
    "TestCase",
]
