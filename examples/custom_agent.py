"""Example: Build your own custom agent in under 20 lines.

Shows the minimal code needed to create a production-ready agent
that plugs into the DAG orchestrator.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator import AgentInput, AgentNode, BaseAgent, DAGOrchestrator


class SentimentAgent(BaseAgent):
    """Minimal custom agent example — sentiment analysis."""

    name = "sentiment"

    def execute(self, agent_input: AgentInput) -> dict:
        text = agent_input.context.get("text", "")
        # Replace with actual sentiment analysis (TextBlob, HuggingFace, LLM, etc.)
        positive_words = {"good", "great", "excellent", "amazing", "love", "best"}
        negative_words = {"bad", "terrible", "worst", "hate", "awful", "poor"}

        words = set(text.lower().split())
        pos = len(words & positive_words)
        neg = len(words & negative_words)
        total = pos + neg or 1

        return {
            "sentiment": "positive"
            if pos > neg
            else "negative"
            if neg > pos
            else "neutral",
            "confidence": round(max(pos, neg) / total, 2),
            "positive_count": pos,
            "negative_count": neg,
        }


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    dag = DAGOrchestrator()
    dag.add_node(AgentNode(agent=SentimentAgent()))
    results = dag.run(
        context={
            "text": "This product is great and amazing but the support is terrible"
        }
    )

    print(dag.summary())
    result = results["sentiment"]
    print(
        f"\nSentiment: {result.data['sentiment']} (confidence: {result.data['confidence']})"
    )
