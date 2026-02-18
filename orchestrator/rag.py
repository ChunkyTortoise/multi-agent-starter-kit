"""Agentic RAG — Retrieval-Augmented Generation with self-correction.

Production patterns for multi-knowledge-base retrieval, iterative query
refinement, and answer validation. No external vector-DB dependencies —
uses TF-IDF relevance scoring out of the box, with a clean interface for
swapping in embeddings (OpenAI, Cohere, etc.).

Usage:
    kb = KnowledgeBase("company_docs")
    kb.add_document(Document(id="doc1", content="...", source="handbook"))

    class MyRAGAgent(RAGAgent):
        name = "rag"

    agent = MyRAGAgent(knowledge_bases=[kb], confidence_threshold=0.65)
    result = agent.run(AgentInput(context={"query": "What is our PTO policy?"}))
    print(result.data["answer"])  # → "Employees receive 15 days..."
    print(result.data["confidence"])  # → 0.82
    print(result.data["sources"])  # → ["handbook"]
"""

from __future__ import annotations

import math
import re
from abc import ABC
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from orchestrator.base_agent import AgentInput, BaseAgent


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Document:
    """A document in a knowledge base.

    Attributes:
        id: Unique document identifier.
        content: Full text content.
        source: Origin label (filename, URL, table name, etc.).
        metadata: Arbitrary key-value metadata.
    """

    id: str
    content: str
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def word_tokens(self) -> list[str]:
        """Lowercase word tokens for TF-IDF."""
        return re.findall(r"\b[a-z]{2,}\b", self.content.lower())


@dataclass
class RetrievalResult:
    """A retrieved document with its relevance score.

    Attributes:
        document: The matched document.
        score: Relevance score in [0, 1]. Higher is more relevant.
        kb_name: Name of the knowledge base this came from.
    """

    document: Document
    score: float
    kb_name: str = ""

    def snippet(self, query: str, max_chars: int = 200) -> str:
        """Return the most relevant passage from the document."""
        terms = set(re.findall(r"\b[a-z]{2,}\b", query.lower()))
        sentences = re.split(r"(?<=[.!?])\s+", self.document.content)
        best = max(
            sentences,
            key=lambda s: sum(1 for t in terms if t in s.lower()),
            default=self.document.content[:max_chars],
        )
        return best[:max_chars]


@dataclass
class RAGContext:
    """The full retrieval context for one query iteration.

    Attributes:
        query: The (possibly refined) query used for retrieval.
        results: Retrieved documents sorted by score.
        confidence: Overall confidence in the retrieval quality.
        iteration: Which refinement iteration this is (0-indexed).
    """

    query: str
    results: list[RetrievalResult]
    confidence: float
    iteration: int = 0

    @property
    def top_result(self) -> RetrievalResult | None:
        return self.results[0] if self.results else None

    @property
    def sources(self) -> list[str]:
        return list({r.kb_name or r.document.source for r in self.results if r.score > 0})


@dataclass
class ValidatedAnswer:
    """An answer that has been validated against retrieved sources.

    Attributes:
        answer: The synthesized answer text.
        confidence: Confidence in the answer (0.0–1.0).
        sources: List of source labels used.
        grounded: Whether the answer is grounded in the retrieved documents.
        warnings: Any validation warnings (e.g. low confidence, missing sources).
    """

    answer: str
    confidence: float
    sources: list[str]
    grounded: bool
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Knowledge Base
# ---------------------------------------------------------------------------


class KnowledgeBase:
    """In-memory knowledge base with TF-IDF retrieval.

    Swap the `search()` method for embedding-based retrieval in production.

    Args:
        name: Unique name for this knowledge base.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._documents: list[Document] = []
        self._idf_cache: dict[str, float] | None = None

    def add_document(self, doc: Document) -> KnowledgeBase:
        """Add a document and invalidate IDF cache. Returns self for chaining."""
        self._documents.append(doc)
        self._idf_cache = None
        return self

    def add_documents(self, docs: list[Document]) -> KnowledgeBase:
        for doc in docs:
            self.add_document(doc)
        return self

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve the top-k most relevant documents using TF-IDF cosine similarity.

        Args:
            query: Search query.
            top_k: Number of results to return.

        Returns:
            List of RetrievalResult, sorted by descending score.

        Note:
            Replace this method with embedding-based search for production use:
            scores = cosine_similarity(embed(query), self._embeddings)
        """
        if not self._documents:
            return []

        query_tokens = re.findall(r"\b[a-z]{2,}\b", query.lower())
        if not query_tokens:
            return []

        idf = self._compute_idf()
        query_tf = Counter(query_tokens)

        scored: list[tuple[float, Document]] = []
        for doc in self._documents:
            doc_tokens = doc.word_tokens()
            doc_tf = Counter(doc_tokens)
            doc_len = len(doc_tokens) or 1

            score = 0.0
            for term, qf in query_tf.items():
                if term in idf:
                    tf = doc_tf.get(term, 0) / doc_len
                    score += tf * idf[term] * qf

            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda x: -x[0])
        max_score = scored[0][0] if scored else 1.0

        return [
            RetrievalResult(
                document=doc,
                score=round(score / max_score, 4),
                kb_name=self.name,
            )
            for score, doc in scored[:top_k]
        ]

    def _compute_idf(self) -> dict[str, float]:
        """Compute inverse document frequency for all terms."""
        if self._idf_cache is not None:
            return self._idf_cache

        n = len(self._documents)
        df: Counter[str] = Counter()
        for doc in self._documents:
            df.update(set(doc.word_tokens()))

        self._idf_cache = {
            term: math.log((n + 1) / (count + 1)) + 1
            for term, count in df.items()
        }
        return self._idf_cache

    def __len__(self) -> int:
        return len(self._documents)

    def __repr__(self) -> str:
        return f"<KnowledgeBase name={self.name!r} docs={len(self)}>"


# ---------------------------------------------------------------------------
# RAG Agent base class
# ---------------------------------------------------------------------------


class RAGAgent(BaseAgent, ABC):
    """Base class for Retrieval-Augmented Generation agents.

    Subclass this and implement `synthesize_answer()` to create a RAG agent.
    The retrieval, self-correction, and validation loops are handled for you.

    Args:
        knowledge_bases: One or more KnowledgeBase instances to query.
        confidence_threshold: Minimum confidence to accept without refinement (0–1).
        max_refinement_iterations: Max query refinement loops before accepting best result.
        top_k_per_kb: Number of results to retrieve per knowledge base per iteration.

    Example:
        class PolicyBot(RAGAgent):
            name = "policy_bot"

            def synthesize_answer(self, query, results):
                top = results[0].document.content if results else "No data found."
                return f"Based on our policy: {top[:200]}"

        bot = PolicyBot(knowledge_bases=[kb])
        out = bot.run(AgentInput(context={"query": "What is the vacation policy?"}))
    """

    name = "rag_agent"
    knowledge_bases: list[KnowledgeBase] = []
    confidence_threshold: float = 0.7
    max_refinement_iterations: int = 3
    top_k_per_kb: int = 3

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not self.knowledge_bases:
            self.knowledge_bases = []

    def execute(self, agent_input: AgentInput) -> dict[str, Any]:
        """Run the full agentic RAG loop: retrieve → evaluate → refine → validate."""
        query = agent_input.context.get("query", "")
        if not query:
            return {
                "answer": "",
                "confidence": 0.0,
                "sources": [],
                "iterations": 0,
                "grounded": False,
                "warnings": ["No query provided"],
            }

        best_context: RAGContext | None = None
        current_query = query

        for iteration in range(self.max_refinement_iterations):
            results = self._retrieve_all(current_query)
            confidence = self._compute_confidence(results)

            ctx = RAGContext(
                query=current_query,
                results=results,
                confidence=confidence,
                iteration=iteration,
            )

            if best_context is None or confidence > best_context.confidence:
                best_context = ctx

            if confidence >= self.confidence_threshold:
                break

            # Self-correction: refine query for next iteration
            current_query = self._refine_query(query, current_query, results)

        assert best_context is not None
        answer_text = self.synthesize_answer(best_context.query, best_context.results)
        validated = self._validate_answer(answer_text, best_context)

        return {
            "answer": validated.answer,
            "confidence": round(validated.confidence, 3),
            "sources": validated.sources,
            "iterations": (best_context.iteration + 1),
            "grounded": validated.grounded,
            "warnings": validated.warnings,
            "result_count": len(best_context.results),
        }

    def synthesize_answer(self, query: str, results: list[RetrievalResult]) -> str:
        """Produce an answer from retrieved documents.

        Override this to integrate with an LLM. The default implementation
        returns the most relevant passage from the top result.

        Args:
            query: The (possibly refined) search query.
            results: Retrieved documents sorted by relevance.

        Returns:
            Answer string.
        """
        if not results:
            return "No relevant information found in the knowledge base."
        top = results[0]
        return top.snippet(query, max_chars=300)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _retrieve_all(self, query: str) -> list[RetrievalResult]:
        """Query all knowledge bases and merge results."""
        all_results: list[RetrievalResult] = []
        for kb in self.knowledge_bases:
            all_results.extend(kb.search(query, top_k=self.top_k_per_kb))
        # Re-sort merged results by score
        all_results.sort(key=lambda r: -r.score)
        return all_results

    def _compute_confidence(self, results: list[RetrievalResult]) -> float:
        """Estimate retrieval confidence from top-result scores.

        Heuristic: weighted average of top-3 scores with diminishing weight.
        """
        if not results:
            return 0.0
        weights = [0.6, 0.3, 0.1]
        score = 0.0
        for i, result in enumerate(results[:3]):
            score += result.score * weights[i]
        return round(score, 4)

    def _refine_query(
        self,
        original_query: str,
        current_query: str,
        results: list[RetrievalResult],
    ) -> str:
        """Refine the query based on low-confidence retrieval.

        Strategy: expand with terms from top result titles/metadata,
        and strip generic stopwords from the query.
        """
        _STOPWORDS = {
            "what", "is", "are", "the", "a", "an", "of", "in", "to", "for",
            "and", "or", "how", "does", "do", "can", "we", "our", "my",
        }

        # Extract key terms from original query
        key_terms = [
            t
            for t in re.findall(r"\b[a-z]{3,}\b", original_query.lower())
            if t not in _STOPWORDS
        ]

        # Extract candidate expansion terms from top result
        expansion_terms: list[str] = []
        if results:
            top_content = results[0].document.content.lower()
            # Find terms near the query terms in the document
            for term in key_terms[:2]:
                idx = top_content.find(term)
                if idx >= 0:
                    window = top_content[max(0, idx - 50) : idx + 100]
                    nearby = [
                        t
                        for t in re.findall(r"\b[a-z]{4,}\b", window)
                        if t not in _STOPWORDS and t not in key_terms
                    ]
                    expansion_terms.extend(nearby[:2])

        refined_parts = key_terms + list(dict.fromkeys(expansion_terms))[:2]
        return " ".join(refined_parts) if refined_parts else current_query

    def _validate_answer(
        self,
        answer: str,
        context: RAGContext,
    ) -> ValidatedAnswer:
        """Validate that the answer is grounded in retrieved sources."""
        warnings: list[str] = []

        if not answer or answer.strip() == "":
            return ValidatedAnswer(
                answer="Unable to generate an answer.",
                confidence=0.0,
                sources=[],
                grounded=False,
                warnings=["Empty answer generated"],
            )

        # Groundedness check: answer terms should overlap with source content
        answer_terms = set(re.findall(r"\b[a-z]{3,}\b", answer.lower()))
        source_terms: set[str] = set()
        for r in context.results[:3]:
            source_terms.update(re.findall(r"\b[a-z]{3,}\b", r.document.content.lower()))

        overlap = answer_terms & source_terms
        grounded = len(overlap) >= max(3, len(answer_terms) * 0.2)

        if not grounded:
            warnings.append(
                f"Low groundedness: only {len(overlap)} answer terms found in sources"
            )

        if context.confidence < 0.3:
            warnings.append(f"Low retrieval confidence: {context.confidence:.2f}")

        sources = context.sources or []
        if not sources:
            warnings.append("No source labels available")

        # Final confidence = blend of retrieval confidence and groundedness
        groundedness_score = min(1.0, len(overlap) / max(len(answer_terms), 1))
        final_confidence = round(
            0.6 * context.confidence + 0.4 * groundedness_score, 3
        )

        return ValidatedAnswer(
            answer=answer,
            confidence=final_confidence,
            sources=sources,
            grounded=grounded,
            warnings=warnings,
        )
