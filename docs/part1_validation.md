# Part 1 validation

The implementation follows the supplied RAG_INSTRUCTIONS.txt (Markdown-only scope).

## Automated checks

- `uv sync` completed.
- `uv add torchvision` completed; dependency and lockfile committed.
- `uv run pytest tests/ -q`: **19 passed**, no skipped tests.
- Ruff checks passed for implemented files and tests.
- Tests use real local all-MiniLM-L6-v2 embeddings and temporary persistent
  ChromaDB collections. The generation unit tests explicitly use a test double.

## Real browser demo

- Streamlit ran locally at `http://127.0.0.1:8501`.
- Uploaded the instructor-provided `test_rag.md` through the sidebar with topic LSTM.
- First ingestion: **1 chunk added, 0 duplicates skipped**, total chunks **1**.
- Second ingestion: **0 chunks added, 1 duplicate skipped**, total chunks stayed **1**.
- Actual Groq model: **openai/gpt-oss-20b**.
- Question: **What does the forget gate do?**
- Real generated answer:
  > The forget gate decides what information should be removed from the cell state.
  > [SOURCE: LSTM | test_rag.md]
- Expanded Sources displayed **[LSTM | intermediate | test_rag.md]**.
- Retrieval similarity displayed **0.35** (rounded).
- Off-topic question: **What is the history of the Roman empire?**
- The UI displayed a no-relevant-information message rather than an unsupported answer.
- Screenshot: [rag_demo.png](../screenshots/rag_demo.png), captured from the actual
  running browser session; no simulated answer or edited screenshot content.

## Publication checks

- `.env` is ignored by Git.
- The local Groq key is absent from all publication files.
- `.venv/` and `data/chroma_db/` are ignored.
- The source, sample Markdown note, screenshot, README, and uv.lock are included.
- Reproduce the demo by cloning, configuring your own local key, and ingesting the note.

The optional LangGraph workflow, alternative providers, and PDF ingestion are not
part of the required Markdown demo. The student will submit the GitHub URL in Canvas.
