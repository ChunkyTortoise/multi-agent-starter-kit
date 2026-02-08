# Architecture

## System Overview

```
                    +-------------------+
                    |  DAGOrchestrator  |
                    |  (Execution Core) |
                    +--------+----------+
                             |
              +--------------+--------------+
              |              |              |
         +----+----+   +----+----+   +----+----+
         | Research |   | Analysis|   |  Report |
         |  Agent   |-->|  Agent  |-->|  Agent  |
         +---------+   +---------+   +---------+
              |              |              |
              v              v              v
         +-------------------------------------------+
         |              Monitor                      |
         | (Performance Table + Gantt + JSON Export)  |
         +-------------------------------------------+
```

## Components

### DAGOrchestrator (`orchestrator/dag.py`)

The execution engine. Takes a set of `AgentNode` objects, resolves
their dependency order via topological sort, and executes them sequentially.

Key features:
- **Topological sort** (Kahn's algorithm) — deterministic execution order
- **Retry with exponential backoff** — configurable per agent
- **Graceful degradation** — optional agents can fail without killing the pipeline
- **Result passing** — upstream outputs are injected into downstream inputs
- **Cycle detection** — raises immediately if your DAG has circular dependencies

### BaseAgent (`orchestrator/base_agent.py`)

Abstract base class. You implement `execute()`, the framework handles everything else.

What you get for free:
- **Pydantic validation** on inputs and outputs
- **Execution timing** — precise millisecond duration tracking
- **Error handling** — exceptions are caught, logged, and wrapped in AgentOutput
- **Cost tracking** — return `_token_usage` and `_cost_usd` from execute()

### Monitor (`orchestrator/monitor.py`)

ASCII dashboard that works in any terminal. No external dependencies.

Output includes:
- **Performance table** — agent name, status, duration, tokens, cost
- **Gantt chart** — visual timeline showing execution overlap
- **Cost summary** — total tokens and cost across all agents
- **JSON export** — full run data for external analysis
- **Run comparison** — side-by-side comparison of multiple runs

## Data Flow

```
Context (dict)
    |
    v
AgentInput {
    context: {...}           <-- Original context
    upstream_results: {...}  <-- Results from dependency agents
}
    |
    v
BaseAgent.execute()          <-- Your logic here
    |
    v
AgentOutput {
    agent_name: str
    success: bool
    data: {...}              <-- Your results
    duration_ms: float       <-- Auto-measured
    token_usage: int         <-- From _token_usage
    estimated_cost_usd: float
}
```

## Dependency Resolution

Dependencies are declared per-node, not per-agent:

```python
AgentNode(agent=ReportAgent(), depends_on=["analysis", "research"])
```

This means the ReportAgent receives BOTH analysis and research results
in `upstream_results`. The orchestrator guarantees both complete before
ReportAgent starts.

## Error Handling Strategy

```
Agent fails
    |
    +-- Has retries remaining?
    |       |
    |       YES --> Wait (exponential backoff) --> Retry
    |       NO  --> Mark as FAILED
    |                   |
    |                   +-- Is agent optional?
    |                   |       |
    |                   |       YES --> Continue pipeline
    |                   |       NO  --> Skip downstream dependents
    |
    +-- Success --> Pass results to downstream agents
```

## Extending the Kit

### Adding a new agent

1. Create a file in `agents/`
2. Subclass `BaseAgent`
3. Set `name` (unique identifier used in depends_on)
4. Implement `execute()` — return a dict
5. Add to your DAG with `AgentNode`

### Adding LLM calls

Replace the mock logic in any agent's `execute()` with actual API calls:

```python
import anthropic

class ClaudeAgent(BaseAgent):
    name = "claude_reasoner"

    def __init__(self):
        self.client = anthropic.Anthropic()

    def execute(self, agent_input: AgentInput) -> dict:
        response = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{"role": "user", "content": agent_input.context["prompt"]}]
        )
        return {
            "response": response.content[0].text,
            "_token_usage": response.usage.input_tokens + response.usage.output_tokens,
            "_cost_usd": (response.usage.input_tokens * 0.003 + response.usage.output_tokens * 0.015) / 1000,
        }
```

### Parallel execution (advanced)

The current implementation executes sequentially. For parallel execution
of independent agents, replace the `for` loop in `DAGOrchestrator.run()`
with `concurrent.futures.ThreadPoolExecutor`:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

# Group agents by "level" (agents with no unmet dependencies)
# Execute each level in parallel
```
