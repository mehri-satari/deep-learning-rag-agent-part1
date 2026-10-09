"""Header-aware Markdown chunking for the Part 1 workshop."""

from __future__ import annotations

from pathlib import Path

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.config import Settings, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


class DocumentChunker:
    """Prepare Markdown chunks with headings and source attribution."""

    DEFAULT_CHUNK_SIZE = 512  # Characters, not tokens.
    DEFAULT_CHUNK_OVERLAP = 50

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def chunk_file(
        self,
        file_path: Path,
        metadata_overrides: dict | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> list[DocumentChunk]:
        file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        if file_path.suffix.lower() != ".md":
            raise ValueError("Part 1 supports Markdown (.md) documents only.")
        metadata = self._infer_metadata(file_path, metadata_overrides)
        return [
            DocumentChunk(
                VectorStoreManager.generate_chunk_id(metadata.source, item["text"]),
                item["text"],
                metadata,
            )
            for item in self._chunk_markdown(file_path, chunk_size, chunk_overlap)
        ]

    def _chunk_markdown(
        self, file_path: Path, chunk_size: int, chunk_overlap: int
    ) -> list[dict]:
        if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError(
                "Use a positive chunk size and a smaller nonnegative overlap."
            )
        text = file_path.read_text(encoding="utf-8-sig")
        if not text.strip():
            return []
        headers = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "title"),
                ("##", "section"),
                ("###", "subsection"),
            ],
            strip_headers=False,
        )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return [
            {"text": document.page_content, "header": document.metadata}
            for document in splitter.split_documents(headers.split_text(text))
            if document.page_content.strip()
        ]

    def _infer_metadata(
        self, file_path: Path, overrides: dict | None = None
    ) -> ChunkMetadata:
        stem = file_path.stem
        topic_name, separator, suffix = stem.rpartition("_")
        if separator and suffix.lower() in {"beginner", "intermediate", "advanced"}:
            difficulty = suffix.lower()
        else:
            topic_name, difficulty = stem, "intermediate"
        topics = {
            "ann": "ANN",
            "cnn": "CNN",
            "rnn": "RNN",
            "lstm": "LSTM",
            "seq2seq": "Seq2Seq",
            "autoencoder": "Autoencoder",
            "som": "SOM",
            "boltzmannmachine": "BoltzmannMachine",
            "gan": "GAN",
        }
        values = {
            "topic": topics.get(topic_name.lower(), topic_name.upper()),
            "difficulty": difficulty,
            "type": "concept_explanation",
            "source": file_path.name,
            "related_topics": [],
        }
        values.update(overrides or {})
        if values["difficulty"] not in {"beginner", "intermediate", "advanced"}:
            raise ValueError("Difficulty must be beginner, intermediate, or advanced.")
        values.setdefault(
            "is_bonus", values["topic"] in {"SOM", "BoltzmannMachine", "GAN"}
        )
        return ChunkMetadata(**values)
