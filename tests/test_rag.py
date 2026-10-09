"""Generation receives retrieved evidence; the guard makes no model call."""

from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from rag_agent.agent.rag import answer_question
from rag_agent.agent.state import ChunkMetadata, RetrievedChunk


def test_no_context_guard_does_not_call_llm():
    store, model = Mock(), Mock()
    store.query.return_value = []
    result = answer_question("History of Rome?", store, model)
    assert result.no_context_found
    assert result.sources == []
    model.invoke.assert_not_called()


def test_blank_question_rejected():
    with pytest.raises(ValueError):
        answer_question(" ", Mock(), Mock())


def test_grounded_generation_receives_source_and_filters():
    store, model = Mock(), Mock()
    store.query.return_value = [
        RetrievedChunk(
            "123",
            "The forget gate removes information from the cell state.",
            ChunkMetadata("LSTM", "intermediate", "concept_explanation", "test_rag.md"),
            0.8,
        )
    ]
    model.invoke.return_value = AIMessage(
        content="The forget gate removes information."
    )
    result = answer_question(
        "What does the forget gate do?",
        store,
        model,
        topic_filter="LSTM",
        difficulty_filter="intermediate",
    )
    store.query.assert_called_once_with(
        "What does the forget gate do?",
        topic_filter="LSTM",
        difficulty_filter="intermediate",
    )
    messages = model.invoke.call_args.args[0]
    assert "[SOURCE: LSTM | test_rag.md]" in messages[-1].content
    assert "removes information from the cell state" in messages[-1].content
    assert "Answer ONLY from the provided context" in messages[0].content
    assert result.sources == ["[LSTM | intermediate | test_rag.md]"]
    assert result.confidence == 0.8


def test_relevant_context_requires_real_model():
    store = Mock()
    store.query.return_value = [
        RetrievedChunk(
            "1",
            "LSTM gates",
            ChunkMetadata("LSTM", "beginner", "concept_explanation", "x.md"),
            0.9,
        )
    ]
    with pytest.raises(EnvironmentError, match="GROQ_API_KEY"):
        answer_question("LSTM?", store, None)
