# Multi-Agent Orchestration Starter Kit

Production-grade DAG-based agent orchestration. Pure Python, zero framework dependencies.

## What's Included

- **DAG Orchestrator** — Define agents and dependencies as a directed acyclic graph. Topological sort, retry with exponential backoff, per-agent timeouts, graceful degradation.
- **BaseAgent Framework** — Abstract base class with Pydantic validation, execution timing, structured logging, and cost tracking. Build any agent in under 20 lines.
- **3 Example Agents** — ResearchAgent, AnalysisAgent, ReportAgent. Fully functional pipeline you can run immediately.
- **ASCII Monitoring Dashboard** — Performance table, Gantt chart timeline, JSON export. No Grafana required.
- **Architecture Docs** — System diagrams, data flow, extension guide.

## Quick Start

```bash
pip install pydantic
python examples/pipeline.py
```

## Project Structure

```
multi-agent-starter-kit/
  orchestrator/
    dag.py           # DAG execution engine
    base_agent.py    # BaseAgent abstract class
    monitor.py       # ASCII monitoring dashboard
  agents/
    research_agent.py
    analysis_agent.py
    report_agent.py
  examples/
    pipeline.py      # Full pipeline example
    custom_agent.py  # Minimal custom agent example
  docs/
    architecture.md  # System design + ASCII diagrams
    quickstart.md    # Getting started guide
```

## License

MIT
