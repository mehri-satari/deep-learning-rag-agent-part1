"""Persistent ChromaDB ingestion, duplicate detection, and cosine retrieval."""

from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb
from loguru import logger

from rag_agent.agent.state import (
    ChunkMetadata,
    DocumentChunk,
    IngestionResult,
    RetrievedChunk,
)
from rag_agent.config import EmbeddingFactory, Settings, get_settings


class VectorStoreManager:
    """Keep document text, metadata, and local embeddings together."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embeddings = EmbeddingFactory(self._settings).create()
        self._initialise()

    def _initialise(self) -> None:
        path = Path(self._settings.chroma_db_path)
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=self._settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
        logger.info(
            "Collection {} ready: {} chunks",
            self._collection.name,
            self._collection.count(),
        )

    @staticmethod
    def generate_chunk_id(source: str, chunk_text: str) -> str:
        """The same source and text always produce the same 16-character ID."""
        return hashlib.sha256(f"{source}::{chunk_text}".encode()).hexdigest()[:16]

    def check_duplicate(self, chunk_id: str) -> bool:
        return bool(self._collection.get(ids=[chunk_id], include=[])["ids"])

    def ingest(self, chunks: list[DocumentChunk]) -> IngestionResult:
        result = IngestionResult()
        pending = []
        seen = set()
        for chunk in chunks:
            if chunk.chunk_id in seen or self.check_duplicate(chunk.chunk_id):
                result.skipped += 1
            elif not chunk.chunk_text.strip():
                result.errors.append(f"{chunk.metadata.source}: empty chunk")
            else:
                pending.append(chunk)
            seen.add(chunk.chunk_id)
        for offset in range(0, len(pending), 100):
            batch = pending[offset : offset + 100]
            try:
                vectors = self._embeddings.embed_documents(
                    [chunk.chunk_text for chunk in batch]
                )
                self._collection.upsert(
                    ids=[chunk.chunk_id for chunk in batch],
                    embeddings=vectors,
                    documents=[chunk.chunk_text for chunk in batch],
                    metadatas=[chunk.metadata.to_dict() for chunk in batch],
                )
                result.ingested += len(batch)
                result.document_ids.extend(chunk.metadata.source for chunk in batch)
            except Exception as exc:
                logger.error("Ingestion batch failed: {}", type(exc).__name__)
                result.errors.extend(
                    f"{chunk.metadata.source}: ingestion failed ({type(exc).__name__})"
                    for chunk in batch
                )
        result.document_ids = sorted(set(result.document_ids))
        logger.info("Ingested {}; skipped {}", result.ingested, result.skipped)
        return result

    def query(
        self,
        query_text: str,
        k: int | None = None,
        topic_filter: str | None = None,
        difficulty_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        count = self._collection.count()
        limit = self._settings.retrieval_k if k is None else k
        if not query_text.strip() or not count or limit <= 0:
            return []
        filters = []
        if topic_filter:
            filters.append({"topic": topic_filter})
        if difficulty_filter:
            filters.append({"difficulty": difficulty_filter})
        where = (
            {"$and": filters} if len(filters) > 1 else filters[0] if filters else None
        )
        response = self._collection.query(
            query_embeddings=[self._embeddings.embed_query(query_text)],
            n_results=min(limit, count),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        retrieved = []
        for chunk_id, text, metadata, distance in zip(
            response["ids"][0],
            response["documents"][0],
            response["metadatas"][0],
            response["distances"][0],
            strict=True,
        ):
            score = max(0.0, min(1.0, 1.0 - distance))
            if score >= self._settings.similarity_threshold:
                retrieved.append(
                    RetrievedChunk(
                        chunk_id, text, ChunkMetadata.from_dict(metadata), score
                    )
                )
        return sorted(retrieved, key=lambda chunk: chunk.score, reverse=True)

    def list_documents(self) -> list[dict]:
        grouped = {}
        for metadata in self._collection.get(include=["metadatas"])["metadatas"]:
            source = metadata["source"]
            if source not in grouped:
                grouped[source] = {
                    "source": source,
                    "topic": metadata["topic"],
                    "chunk_count": 0,
                }
            grouped[source]["chunk_count"] += 1
        return [grouped[source] for source in sorted(grouped)]

    def get_document_chunks(self, source: str) -> list[DocumentChunk]:
        response = self._collection.get(
            where={"source": source}, include=["documents", "metadatas"]
        )
        return [
            DocumentChunk(chunk_id, text, ChunkMetadata.from_dict(metadata))
            for chunk_id, text, metadata in zip(
                response["ids"],
                response["documents"],
                response["metadatas"],
                strict=True,
            )
        ]

    def get_collection_stats(self) -> dict:
        metadata = self._collection.get(include=["metadatas"])["metadatas"]
        return {
            "total_chunks": self._collection.count(),
            "topics": sorted({item["topic"] for item in metadata}),
            "sources": sorted({item["source"] for item in metadata}),
            "bonus_topics_present": any(
                item.get("is_bonus") == "true" for item in metadata
            ),
        }

    def delete_document(self, source: str) -> int:
        count = len(self._collection.get(where={"source": source}, include=[])["ids"])
        self._collection.delete(where={"source": source})
        return count
