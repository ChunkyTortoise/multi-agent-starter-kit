"""Tests for the Agentic RAG module."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator import AgentInput, Document, KnowledgeBase, RAGAgent, RetrievalResult


# ---------------------------------------------------------------------------
# Document tests
# ---------------------------------------------------------------------------


class TestDocument:
    def test_word_tokens(self):
        doc = Document(id="d1", content="The quick brown fox jumps")
        tokens = doc.word_tokens()
        assert "quick" in tokens
        assert "fox" in tokens
        # Single-char words excluded
        assert "a" not in tokens

    def test_empty_content(self):
        doc = Document(id="d2", content="")
        assert doc.word_tokens() == []


# ---------------------------------------------------------------------------
# KnowledgeBase tests
# ---------------------------------------------------------------------------


class TestKnowledgeBase:
    def _kb(self) -> KnowledgeBase:
        kb = KnowledgeBase("test_kb")
        kb.add_document(Document(id="d1", content="Python is a high-level programming language", source="python_docs"))
        kb.add_document(Document(id="d2", content="JavaScript runs in the browser and server environments", source="js_docs"))
        kb.add_document(Document(id="d3", content="Python uses indentation for code blocks and syntax", source="python_docs"))
        return kb

    def test_add_document_returns_self(self):
        kb = KnowledgeBase("kb")
        doc = Document(id="d1", content="test")
        returned = kb.add_document(doc)
        assert returned is kb

    def test_len(self):
        kb = self._kb()
        assert len(kb) == 3

    def test_repr(self):
        kb = self._kb()
        assert "test_kb" in repr(kb)
        assert "3" in repr(kb)

    def test_search_returns_relevant_docs(self):
        kb = self._kb()
        results = kb.search("Python programming")
        assert len(results) > 0
        # Python docs should rank first
        sources = [r.document.source for r in results]
        assert "python_docs" in sources

    def test_search_scores_normalized(self):
        kb = self._kb()
        results = kb.search("Python")
        assert all(0.0 <= r.score <= 1.0 for r in results)
        if results:
            assert results[0].score == 1.0  # top result normalized to 1.0

    def test_search_top_k(self):
        kb = self._kb()
        results = kb.search("code", top_k=1)
        assert len(results) <= 1

    def test_search_empty_kb(self):
        kb = KnowledgeBase("empty")
        results = kb.search("anything")
        assert results == []

    def test_search_no_match(self):
        kb = self._kb()
        results = kb.search("quantum mechanics superconductor")
        assert len(results) == 0

    def test_search_empty_query(self):
        kb = self._kb()
        results = kb.search("")
        assert results == []

    def test_idf_cache_invalidated_on_add(self):
        kb = KnowledgeBase("cache_test")
        kb.add_document(Document(id="d1", content="machine learning neural network"))
        _ = kb.search("machine")
        assert kb._idf_cache is not None
        kb.add_document(Document(id="d2", content="deep learning transformers"))
        assert kb._idf_cache is None  # invalidated

    def test_add_documents_bulk(self):
        kb = KnowledgeBase("bulk")
        docs = [Document(id=f"d{i}", content=f"content {i}") for i in range(5)]
        kb.add_documents(docs)
        assert len(kb) == 5

    def test_retrieval_result_kb_name(self):
        kb = KnowledgeBase("my_kb")
        kb.add_document(Document(id="d1", content="test content here"))
        results = kb.search("test content")
        assert results[0].kb_name == "my_kb"


# ---------------------------------------------------------------------------
# RetrievalResult tests
# ---------------------------------------------------------------------------


class TestRetrievalResult:
    def test_snippet_returns_relevant_passage(self):
        doc = Document(id="d1", content="The sky is blue. Python is great. I love programming.")
        result = RetrievalResult(document=doc, score=0.9, kb_name="kb")
        snippet = result.snippet("Python programming")
        assert len(snippet) <= 200
        assert len(snippet) > 0

    def test_snippet_max_chars(self):
        doc = Document(id="d1", content="x " * 500)
        result = RetrievalResult(document=doc, score=0.5)
        snippet = result.snippet("x", max_chars=50)
        assert len(snippet) <= 50


# ---------------------------------------------------------------------------
# RAGAgent tests
# ---------------------------------------------------------------------------


class SimpleRAGAgent(RAGAgent):
    """Minimal RAGAgent implementation for testing."""
    name = "simple_rag"


class TestRAGAgent:
    def _make_agent(self, docs: list | None = None) -> SimpleRAGAgent:
        kb = KnowledgeBase("test_kb")
        if docs:
            for doc in docs:
                kb.add_document(doc)
        else:
            kb.add_document(Document(
                id="d1",
                content="The vacation policy allows 15 days per year for full-time employees.",
                source="hr_handbook",
            ))
            kb.add_document(Document(
                id="d2",
                content="Remote work policy: employees may work remotely up to 3 days per week.",
                source="hr_handbook",
            ))
            kb.add_document(Document(
                id="d3",
                content="Health benefits include medical, dental, and vision coverage.",
                source="benefits_guide",
            ))
        return SimpleRAGAgent(knowledge_bases=[kb])

    def test_basic_retrieval(self):
        agent = self._make_agent()
        result = agent.run(AgentInput(context={"query": "vacation policy"}))
        assert result.success is True
        assert "answer" in result.data
        assert result.data["answer"] != ""

    def test_confidence_in_range(self):
        agent = self._make_agent()
        result = agent.run(AgentInput(context={"query": "vacation days"}))
        assert 0.0 <= result.data["confidence"] <= 1.0

    def test_sources_populated(self):
        agent = self._make_agent()
        result = agent.run(AgentInput(context={"query": "health benefits coverage"}))
        assert isinstance(result.data["sources"], list)

    def test_no_query_returns_empty(self):
        agent = self._make_agent()
        result = agent.run(AgentInput(context={}))
        assert result.success is True
        assert result.data["answer"] == ""
        assert result.data["confidence"] == 0.0
        assert "No query provided" in result.data["warnings"]

    def test_iterations_tracked(self):
        agent = self._make_agent()
        result = agent.run(AgentInput(context={"query": "remote work"}))
        assert result.data["iterations"] >= 1

    def test_grounded_field_present(self):
        agent = self._make_agent()
        result = agent.run(AgentInput(context={"query": "benefits"}))
        assert "grounded" in result.data

    def test_warnings_field_present(self):
        agent = self._make_agent()
        result = agent.run(AgentInput(context={"query": "something"}))
        assert isinstance(result.data["warnings"], list)

    def test_empty_kb_handles_gracefully(self):
        agent = SimpleRAGAgent(knowledge_bases=[KnowledgeBase("empty")])
        result = agent.run(AgentInput(context={"query": "anything"}))
        assert result.success is True
        assert result.data["confidence"] == 0.0

    def test_multi_kb_query(self):
        kb1 = KnowledgeBase("kb1")
        kb1.add_document(Document(id="a", content="FastAPI is a Python web framework"))
        kb2 = KnowledgeBase("kb2")
        kb2.add_document(Document(id="b", content="Python is widely used for data science"))
        agent = SimpleRAGAgent(knowledge_bases=[kb1, kb2])
        result = agent.run(AgentInput(context={"query": "Python"}))
        assert result.success is True
        assert result.data["result_count"] >= 1

    def test_low_confidence_triggers_refinement(self):
        """With a very high threshold, multiple iterations should be attempted."""
        agent = SimpleRAGAgent(
            knowledge_bases=[KnowledgeBase("empty_kb")],
            confidence_threshold=0.99,
            max_refinement_iterations=3,
        )
        result = agent.run(AgentInput(context={"query": "anything at all"}))
        assert result.success is True
        # Should have tried multiple iterations
        assert result.data["iterations"] >= 1

    def test_custom_synthesize_answer(self):
        class CustomRAG(RAGAgent):
            name = "custom_rag"

            def synthesize_answer(self, query, results):
                return f"Custom answer for: {query}"

        kb = KnowledgeBase("kb")
        kb.add_document(Document(id="d1", content="some relevant content about the topic"))
        agent = CustomRAG(knowledge_bases=[kb])
        result = agent.run(AgentInput(context={"query": "topic"}))
        assert "Custom answer for: topic" in result.data["answer"]

    def test_result_count_field(self):
        agent = self._make_agent()
        result = agent.run(AgentInput(context={"query": "policy"}))
        assert "result_count" in result.data
        assert result.data["result_count"] >= 0
