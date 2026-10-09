"""Markdown boundaries, metadata, and reproducibility checks."""

from pathlib import Path

import pytest

from rag_agent.corpus.chunker import DocumentChunker


def test_provided_note_is_chunked_without_losing_content():
    chunks = DocumentChunker().chunk_file(
        Path("test_rag.md"), metadata_overrides={"topic": "LSTM"}
    )
    assert chunks and all(chunk.metadata.source == "test_rag.md" for chunk in chunks)
    assert any("forget gate" in chunk.chunk_text for chunk in chunks)
    repeated = DocumentChunker().chunk_file(Path("test_rag.md"), {"topic": "LSTM"})
    assert [chunk.chunk_id for chunk in chunks] == [
        chunk.chunk_id for chunk in repeated
    ]


def test_headers_are_preserved_and_long_sections_are_split(tmp_path):
    path = tmp_path / "lstm_beginner.md"
    path.write_text(
        "# LSTM\n\n## Forget Gate\n" + "Memory is controlled by gates. " * 30,
        encoding="utf-8",
    )
    chunks = DocumentChunker().chunk_file(path, chunk_size=180, chunk_overlap=25)
    assert len(chunks) > 1
    assert all(len(chunk.chunk_text) <= 180 for chunk in chunks)
    assert chunks[0].chunk_text.startswith("# LSTM")
    assert chunks[0].metadata.difficulty == "beginner"


def test_metadata_overrides_and_bonus_topics(tmp_path):
    path = tmp_path / "gan_advanced.md"
    path.write_text("# GAN\nA generator and discriminator compete.", encoding="utf-8")
    chunker = DocumentChunker()
    assert chunker.chunk_file(path)[0].metadata.is_bonus
    overridden = chunker.chunk_file(path, {"topic": "CNN", "difficulty": "beginner"})[0]
    assert overridden.metadata.topic == "CNN"
    assert not overridden.metadata.is_bonus


def test_invalid_empty_and_missing_files(tmp_path):
    chunker = DocumentChunker()
    with pytest.raises(FileNotFoundError):
        chunker.chunk_file(tmp_path / "missing.md")
    path = tmp_path / "empty.md"
    path.write_text("", encoding="utf-8")
    assert chunker.chunk_file(path) == []
    with pytest.raises(ValueError):
        chunker.chunk_file(path, chunk_size=10, chunk_overlap=10)
    pdf = tmp_path / "note.pdf"
    pdf.write_text("not a PDF", encoding="utf-8")
    with pytest.raises(ValueError, match="Markdown"):
        chunker.chunk_file(pdf)
