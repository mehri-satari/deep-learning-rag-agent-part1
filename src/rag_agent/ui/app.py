"""Streamlit document ingestion, source viewer, and grounded RAG chat."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st

from rag_agent.agent.rag import answer_question
from rag_agent.config import LLMFactory, get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager


@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    return VectorStoreManager()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    return DocumentChunker()


@st.cache_resource
def get_llm():
    return LLMFactory().create()


def initialise_session_state() -> None:
    for key, default in {"chat_history": [], "last_ingestion_result": None}.items():
        if key not in st.session_state:
            st.session_state[key] = default


def render_ingestion_panel(store: VectorStoreManager, chunker: DocumentChunker) -> None:
    st.sidebar.header("Document ingestion")
    files = st.sidebar.file_uploader(
        "Upload study notes (.md)", type=["md"], accept_multiple_files=True
    )
    topic = st.sidebar.text_input("Topic (optional)", placeholder="Example: LSTM")
    difficulty = st.sidebar.selectbox(
        "Note difficulty", ["intermediate", "beginner", "advanced"]
    )
    if st.sidebar.button("Ingest Documents", disabled=not files, type="primary"):
        added = skipped = 0
        errors = []
        with st.spinner("Preparing and storing your notes..."):
            with TemporaryDirectory() as directory:
                for uploaded in files:
                    path = Path(directory) / Path(uploaded.name).name
                    path.write_bytes(uploaded.getvalue())
                    overrides = {"difficulty": difficulty}
                    if topic.strip():
                        overrides["topic"] = topic.strip().upper()
                    try:
                        chunks = chunker.chunk_file(path, metadata_overrides=overrides)
                        if not chunks:
                            errors.append(f"{path.name}: the document is empty.")
                            continue
                        result = store.ingest(chunks)
                        added += result.ingested
                        skipped += result.skipped
                        errors.extend(result.errors)
                    except (ValueError, FileNotFoundError, UnicodeError):
                        errors.append(
                            f"{path.name}: use a readable UTF-8 Markdown file."
                        )
        st.session_state.last_ingestion_result = (added, skipped, errors)
    if st.session_state.last_ingestion_result:
        added, skipped, errors = st.session_state.last_ingestion_result
        message = f"{added} chunks added, {skipped} duplicates skipped"
        if errors:
            st.sidebar.error("\n".join(errors))
        if added:
            st.sidebar.success(message)
        elif skipped:
            st.sidebar.info(message)
    st.sidebar.metric(
        "Total ChromaDB chunks", store.get_collection_stats()["total_chunks"]
    )
    st.sidebar.caption(
        "Upload a note, ingest it, then ask a question about its content."
    )


def render_document_viewer(store: VectorStoreManager) -> None:
    st.subheader("Study material")
    documents = store.list_documents()
    if not documents:
        st.info("Upload and ingest a Markdown note to view it here.")
        return
    selected = st.selectbox(
        "Select document", [document["source"] for document in documents]
    )
    chunks = store.get_document_chunks(selected)
    st.caption(f"{selected} · {len(chunks)} chunks")
    with st.container(height=420):
        for chunk in chunks:
            st.caption(f"{chunk.metadata.topic} · {chunk.metadata.difficulty}")
            st.markdown(chunk.chunk_text)
            st.divider()


def render_chat_interface(store: VectorStoreManager) -> None:
    st.subheader("Ask your notes")
    topics = store.get_collection_stats()["topics"]
    left, right = st.columns(2)
    topic = left.selectbox("Topic filter", ["All topics", *topics])
    difficulty = right.selectbox(
        "Difficulty filter", ["All levels", "beginner", "intermediate", "advanced"]
    )
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources", expanded=True):
                    for source in message["sources"]:
                        st.caption(source)
                    st.caption(f"Retrieval similarity: {message['confidence']:.2f}")
    query = st.chat_input("Ask about your uploaded study material...")
    if query:
        history = st.session_state.chat_history.copy()
        st.session_state.chat_history.append({"role": "user", "content": query})
        try:
            with st.spinner("Finding relevant passages and generating an answer..."):
                # Missing credentials never prevent ingestion or the no-context guard.
                settings = get_settings()
                llm = get_llm() if settings.groq_api_key.strip() else None
                response = answer_question(
                    query,
                    store,
                    llm,
                    history,
                    topic_filter=None if topic == "All topics" else topic,
                    difficulty_filter=None
                    if difficulty == "All levels"
                    else difficulty,
                )
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": response.answer,
                    "sources": response.sources,
                    "confidence": response.confidence,
                }
            )
        except EnvironmentError:
            st.session_state.chat_history.pop()
            st.error(
                "Add your Groq API key to the local .env file, then restart the app."
            )
            return
        except Exception as exc:
            st.session_state.chat_history.pop()
            st.error(
                f"The model request could not complete ({type(exc).__name__}). "
                "Check your Groq credentials and connection, then try again."
            )
            return
        st.rerun()


def main() -> None:
    settings = get_settings()
    st.set_page_config(page_title=settings.app_title, page_icon="📚", layout="wide")
    st.title("Deep Learning RAG Workshop")
    st.caption(f"Part 1 · Markdown study notes · Groq {settings.groq_model}")
    initialise_session_state()
    store = get_vector_store()
    render_ingestion_panel(store, get_chunker())
    viewer, chat = st.columns([1, 1], gap="large")
    with viewer:
        render_document_viewer(store)
    with chat:
        render_chat_interface(store)


if __name__ == "__main__":
    main()
