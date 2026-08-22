"""Tests for prompt construction."""

from nexus.generation.prompts import PromptManager
from nexus.models.search import SearchResult


def test_build_context_block_empty():
    pm = PromptManager()
    block = pm.build_context_block([])
    assert "No relevant context" in block


def test_build_context_block_with_results():
    pm = PromptManager()
    results = [
        SearchResult(
            chunk_id="c1",
            document_id="d1",
            content="The capital of France is Paris.",
            score=0.95,
            metadata={"source": "geography.pdf"},
        ),
        SearchResult(
            chunk_id="c2",
            document_id="d2",
            content="Paris is known for the Eiffel Tower.",
            score=0.87,
            metadata={"source": "travel.md"},
        ),
    ]
    block = pm.build_context_block(results)
    assert "[Source 1]" in block
    assert "[Source 2]" in block
    assert "geography.pdf" in block
    assert "Paris" in block


def test_build_user_prompt_includes_query():
    pm = PromptManager()
    results = [
        SearchResult(
            chunk_id="c1",
            document_id="d1",
            content="Some context.",
            score=0.9,
        )
    ]
    prompt = pm.build_user_prompt("What is the capital?", results)
    assert "What is the capital?" in prompt
    assert "Some context." in prompt


def test_build_citations_truncates_long_content():
    pm = PromptManager(excerpt_length=50)
    long_content = "X" * 100
    results = [
        SearchResult(
            chunk_id="c1",
            document_id="d1",
            content=long_content,
            score=0.9,
            metadata={"source": "doc.txt"},
        )
    ]
    citations = pm.build_citations(results)
    assert len(citations) == 1
    assert len(citations[0].excerpt) <= 53  # 50 + "..."
    assert citations[0].excerpt.endswith("...")


def test_system_prompt_prevents_hallucination():
    pm = PromptManager()
    system = pm.build_system_prompt()
    assert "Only use information from the provided context" in system
    assert "don't have enough information" in system.lower()
