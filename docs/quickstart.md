# Quickstart Guide

## Setup (30 seconds)

```bash
pip install pydantic
```

That's it. No framework installs, no config files, no API keys needed to run the examples.

## Run the Example Pipeline

```bash
python examples/pipeline.py
```

You'll see:
1. Agent execution logs (research -> analysis -> report)
2. Pipeline summary (pass/fail per agent)
3. The generated report
4. ASCII monitoring dashboard with Gantt chart
5. JSON metrics exported to `pipeline_metrics.json`

## Build Your First Custom Agent

```python
from orchestrator import BaseAgent, AgentInput

class MyAgent(BaseAgent):
    name = "my_agent"

    def execute(self, agent_input: AgentInput) -> dict:
        query = agent_input.context["query"]
        # Your logic here
        return {"result": f"Processed: {query}"}
```

## Wire It Into a Pipeline

```python
from orchestrator import DAGOrchestrator, AgentNode

dag = DAGOrchestrator()
dag.add_node(AgentNode(agent=MyAgent()))
results = dag.run(context={"query": "hello world"})
print(results["my_agent"].data)
```

## Add Dependencies

```python
dag = DAGOrchestrator()
dag.add_node(AgentNode(agent=FetchAgent()))
dag.add_node(AgentNode(agent=ProcessAgent(), depends_on=["fetch"]))
dag.add_node(AgentNode(agent=OutputAgent(), depends_on=["process"]))
results = dag.run(context={"url": "https://example.com"})
```

## Make Agents Optional

```python
# If enrichment fails, the pipeline continues without it
dag.add_node(AgentNode(
    agent=EnrichmentAgent(),
    depends_on=["fetch"],
    optional=True,   # Won't block downstream agents on failure
    retry_count=3,   # Try 3 times before giving up
    retry_delay=2.0, # 2s, 4s, 8s exponential backoff
))
```

## Monitor Your Runs

```python
from orchestrator import Monitor

monitor = Monitor()

# Run pipeline multiple times
for query in queries:
    results = dag.run(context={"query": query})
    monitor.record(results, dag.timeline)

# View dashboard for latest run
print(monitor.dashboard())

# Compare all runs
print(monitor.compare_runs())

# Export for external analysis
monitor.export_json("metrics.json")
```

## Next Steps

1. Replace mock logic in example agents with real API calls
2. Read `docs/architecture.md` for the full system design
3. See `examples/custom_agent.py` for a minimal agent template
4. Add your own agents to `agents/` and wire them into DAGs
