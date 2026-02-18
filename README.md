# Multi-Agent Orchestration Starter Kit

Production-grade multi-agent orchestration for Python. Pure Python, zero framework dependencies.

![CI](https://github.com/ChunkyTortoise/multi-agent-starter-kit/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Tests](https://img.shields.io/badge/tests-101%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

## What This Solves

- **Agent coordination complexity** — Run multi-step workflows with explicit dependencies and safe retries
- **High-stakes decisions** — Human-in-the-loop approval gates pause pipelines until a human signs off
- **RAG without the bloat** — Multi-KB retrieval, self-correcting queries, and answer validation — no vector DB required
- **Blind deployment** — Evaluate agents on test suites before they hit production; get P50/P95/P99 latency reports
- **Framework bloat** — No heavyweight dependencies; a compact, readable orchestration core

## Quick Start

```bash
pip install pydantic
python examples/pipeline.py         # Basic DAG pipeline
python examples/hitl_pipeline.py    # Human-in-the-loop approval gates
python examples/eval_demo.py        # Agent evaluation + benchmarking
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Your Agent Pipeline                 │
│                                                     │
│  Agent A  →  HITL Gate  →  Agent B  →  Agent C     │
│     ↑             ↑            ↑                    │
│  BaseAgent    Human/API    RAGAgent                 │
└─────────────────────────────────────────────────────┘
                      ↓
             DAGOrchestrator
     (topological sort, retry, timeout)
                      ↓
    ┌─────────────────────────────────┐
    │  Monitor  │  Eval  │  HITL Log  │
    └─────────────────────────────────┘
```

## What's Included

### Core Orchestration
- **DAG Orchestrator** — Define pipelines as a directed acyclic graph. Topological sort, exponential backoff retry, per-agent timeouts, graceful degradation.
- **BaseAgent** — Abstract base class with Pydantic validation, execution timing, structured logging, and LLM cost tracking. Build any agent in under 20 lines.
- **ASCII Monitor** — Performance table, Gantt chart timeline, JSON export. No Grafana required.

### Human-in-the-Loop (HITL)
Pause pipeline execution at critical decisions — model deployment, data deletion, payment processing — and require human sign-off before proceeding.

```python
from orchestrator import HITLGate, AgentNode, DAGOrchestrator

# Gate auto-approves after 60s, or wait for external approve()
gate = HITLGate(
    name="deploy_approval",
    description="Senior engineer must review before production deploy",
    auto_approve_after=60.0,   # Safety net
    interactive=False,          # Webhook/API mode
)

dag = DAGOrchestrator()
dag.add_node(AgentNode(agent=BuildAgent()))
dag.add_node(AgentNode(
    agent=DeployAgent(),
    depends_on=["build"],
    hitl_gate=gate,            # Blocks here until approved
))

# From your Slack bot or webhook handler:
gate.approve(gate_id, approver="alice@company.com", notes="LGTM")
```

**Features**:
- CLI interactive mode (`y/n/details` prompt) for development
- Non-interactive mode for webhook/API integration
- `on_request` and `on_resolve` callbacks (Slack, PagerDuty, etc.)
- Rejection halts pipeline and blocks downstream agents
- Full audit trail: who approved, when, how long it waited

### Agentic RAG
Multi-knowledge-base retrieval with self-correcting queries and answer validation.

```python
from orchestrator import KnowledgeBase, Document, RAGAgent

# Build knowledge bases from any source
kb_hr = KnowledgeBase("hr_policies")
kb_hr.add_document(Document(id="pto", content="Employees receive 20 PTO days per year...", source="handbook"))

kb_legal = KnowledgeBase("legal")
kb_legal.add_document(Document(id="nda", content="All employees must sign an NDA...", source="legal_guide"))

# Create a RAG agent — subclass and optionally override synthesize_answer()
class PolicyBot(RAGAgent):
    name = "policy_bot"

    def synthesize_answer(self, query, results):
        # Replace with LLM call in production
        return results[0].document.content[:300] if results else "Not found."

bot = PolicyBot(knowledge_bases=[kb_hr, kb_legal], confidence_threshold=0.65)
result = bot.run(AgentInput(context={"query": "What is the PTO policy?"}))

print(result.data["answer"])      # Synthesized answer
print(result.data["confidence"])  # 0.82
print(result.data["sources"])     # ["hr_policies"]
print(result.data["iterations"])  # 1 (no refinement needed)
```

**Features**:
- TF-IDF relevance scoring (zero external deps) — swap `search()` for embeddings
- Multi-KB query: merges results from all knowledge bases, re-sorted by score
- Self-correction: low confidence → query refinement → retry (configurable iterations)
- Answer validation: groundedness check, confidence blending, structured warnings
- Drop-in for OpenAI/Cohere embeddings by overriding `KnowledgeBase.search()`

### Agent Evaluation Framework
Measure agent quality systematically before deployment.

```python
from orchestrator import AgentEvaluator, BenchmarkSuite, TestCase

# Define test cases with custom judge functions
cases = [
    TestCase(
        id="basic_query",
        input_context={"query": "What is PTO policy?"},
        judge=lambda out: "days" in out.get("answer", "").lower(),
        tags=["smoke"],
    ),
    TestCase(
        id="confidence_threshold",
        input_context={"query": "vacation days"},
        judge=lambda out: (out.get("confidence", 0) > 0.5, out.get("confidence", 0)),
    ),
]

# Evaluate
evaluator = AgentEvaluator(PolicyBot(knowledge_bases=[kb_hr]))
report = evaluator.evaluate(cases)
print(report.summary())
# → Pass rate: 100.0% | Avg score: 1.000 | Avg latency: 2.3ms

# Benchmark (P50/P95/P99 latency across 10 runs)
bench = evaluator.benchmark(cases, runs=10)
print(bench.summary())
# → P50: 2.1ms | P95: 4.8ms | P99: 6.2ms | Error rate: 0.0%

# Compare multiple agents on the same suite
suite = BenchmarkSuite(cases)
suite.add(AgentV1()).add(AgentV2()).add(AgentV3())
print(suite.leaderboard())
```

**Features**:
- Custom judge functions (exact match, partial credit, LLM-as-judge)
- P50/P95/P99 latency from multi-run benchmarks
- Tag-based filtering (smoke tests, regression, edge cases)
- Agent comparison and leaderboard
- JSON export for CI integration

## Project Structure

```
multi-agent-starter-kit/
  orchestrator/
    dag.py           # DAG execution engine + HITL integration
    base_agent.py    # BaseAgent abstract class
    monitor.py       # ASCII monitoring dashboard
    hitl.py          # Human-in-the-loop approval gates
    rag.py           # Agentic RAG: KnowledgeBase, RAGAgent
    eval.py          # Evaluation + benchmarking framework
  agents/
    research_agent.py
    analysis_agent.py
    report_agent.py
  examples/
    pipeline.py         # Basic DAG pipeline
    hitl_pipeline.py    # HITL approval gates (3 modes)
    eval_demo.py        # Evaluation + leaderboard demo
    custom_agent.py     # Minimal custom agent template
  tests/
    test_orchestrator.py  # DAG + BaseAgent + Monitor (23 tests)
    test_hitl.py          # HITL gates + DAG integration (19 tests)
    test_rag.py           # RAG: KB, retrieval, RAGAgent (34 tests)
    test_eval.py          # Eval + benchmark framework (25 tests)
  docs/
    architecture.md
    quickstart.md
```

## Service Mapping

- Service 4: Multi-Agent Workflows (Agentic AI Systems)
- Service 6: AI-Powered Personal and Business Automation

## Certification Mapping

- IBM RAG and Agentic AI Professional Certificate
- Duke University LLMOps Specialization
- Vanderbilt Generative AI Strategic Leader Specialization
- Google Cloud Generative AI Leader Certificate

## License

MIT
