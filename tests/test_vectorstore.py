"""Integration tests against persistent ChromaDB and real local embeddings."""

import pytest

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.config import Settings
from rag_agent.vectorstore.store import VectorStoreManager


@pytest.fixture
def store(tmp_path):
    return VectorStoreManager(
        Settings(
            _env_file=None,
            chroma_db_path=str(tmp_path / "chroma"),
            similarity_threshold=0.3,
        )
    )


@pytest.fixture
def sample_chunk():
    text = (
        "Long Short-Term Memory networks use the forget gate to decide what "
        "information should be removed from the cell state. The input gate "
        "stores new information and the output gate controls the next hidden state."
    )
    return DocumentChunk(
        VectorStoreManager.generate_chunk_id("lstm_intermediate.md", text),
        text,
        ChunkMetadata(
            "LSTM",
            "intermediate",
            "concept_explanation",
            "lstm_intermediate.md",
            ["RNN"],
            False,
        ),
    )


@pytest.fixture
def bonus_chunk():
    text = (
        "Generative Adversarial Networks consist of a generator producing synthetic "
        "data and a discriminator distinguishing real from generated samples."
    )
    return DocumentChunk(
        VectorStoreManager.generate_chunk_id("gan_advanced.md", text),
        text,
        ChunkMetadata("GAN", "advanced", "architecture", "gan_advanced.md", [], True),
    )


def test_chunk_ids_are_deterministic_and_source_specific():
    generate = VectorStoreManager.generate_chunk_id
    value = generate("a.md", "text")
    assert value == generate("a.md", "text")
    assert value != generate("a.md", "different")
    assert value != generate("b.md", "text")
    assert len(value) == 16
    assert all(character in "0123456789abcdef" for character in value)


def test_duplicate_guard(store, sample_chunk):
    assert not store.check_duplicate(sample_chunk.chunk_id)
    first = store.ingest([sample_chunk])
    assert first.ingested == 1 and not first.errors
    assert store.check_duplicate(sample_chunk.chunk_id)
    second = store.ingest([sample_chunk])
    assert second.ingested == 0 and second.skipped == 1
    assert store.get_collection_stats()["total_chunks"] == 1


def test_duplicate_inside_same_batch(store, sample_chunk):
    result = store.ingest([sample_chunk, sample_chunk])
    assert (result.ingested, result.skipped) == (1, 1)


def test_relevant_query_returns_cited_text(store, sample_chunk):
    store.ingest([sample_chunk])
    results = store.query("What does the LSTM forget gate do?", k=100)
    assert results
    assert results[0].metadata.source == "lstm_intermediate.md"
    assert results[0].score >= 0.3


def test_off_topic_and_blank_queries_return_empty(store, sample_chunk):
    store.ingest([sample_chunk])
    assert store.query("history of the Roman empire") == []
    assert store.query("   ") == []
    assert store.query("LSTM", k=0) == []


def test_empty_collection_returns_empty(store):
    assert store.query("LSTM forget gate") == []


def test_combined_filters_and_score_order(store, sample_chunk, bonus_chunk):
    store.ingest([sample_chunk, bonus_chunk])
    filtered = store.query(
        sample_chunk.chunk_text, topic_filter="LSTM", difficulty_filter="intermediate"
    )
    assert filtered and all(chunk.metadata.topic == "LSTM" for chunk in filtered)
    assert store.query("LSTM", topic_filter="LSTM", difficulty_filter="advanced") == []
    store._settings.similarity_threshold = 0.0
    results = store.query(sample_chunk.chunk_text)
    assert len(results) == 2
    assert [chunk.score for chunk in results] == sorted(
        [chunk.score for chunk in results], reverse=True
    )


def test_metadata_round_trip_and_delete(store, sample_chunk, bonus_chunk):
    store.ingest([sample_chunk, bonus_chunk])
    chunks = store.get_document_chunks("lstm_intermediate.md")
    assert len(chunks) == 1
    assert chunks[0].metadata == sample_chunk.metadata
    stats = store.get_collection_stats()
    assert stats["topics"] == ["GAN", "LSTM"]
    assert stats["bonus_topics_present"]
    assert len(store.list_documents()) == 2
    assert store.delete_document("lstm_intermediate.md") == 1
    assert store.get_collection_stats()["total_chunks"] == 1
    assert store.get_document_chunks("lstm_intermediate.md") == []


def test_data_survives_new_store_instance(store, sample_chunk):
    store.ingest([sample_chunk])
    reopened = VectorStoreManager(store._settings)
    assert reopened.check_duplicate(sample_chunk.chunk_id)
    assert reopened.query("forget gate")


def test_strict_threshold_removes_weak_matches(store, sample_chunk):
    store.ingest([sample_chunk])
    store._settings.similarity_threshold = 0.999
    assert store.query("What is LSTM?") == []


def test_empty_chunk_is_reported(store, sample_chunk):
    sample_chunk.chunk_text = " "
    result = store.ingest([sample_chunk])
    assert result.ingested == 0 and len(result.errors) == 1
