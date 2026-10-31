# agentic-docqa

Multi-user document QA. Upload PDFs or markdown, ask questions, get answers grounded in your own
files. Built on LangGraph, Qdrant hybrid search, and any OpenAI-compatible LLM (Ollama by default).

## What it does

- Accounts with bcrypt passwords and cookie sessions. Each user gets their own Qdrant
  collection, parent store and chat list, so retrieval never crosses users.
- Parent/child chunking: markdown is split on headings into ~2-4k char parents; each parent is
  split into small overlapping children. Children get embedded, parents are fetched for context.
- Hybrid retrieval: dense (Ollama `nomic-embed-text`) + sparse BM25 (fastembed) in one Qdrant
  collection, then a cross-encoder rerank (`bge-reranker-base`), deduped to one hit per parent.
- A grading step (CRAG style) after every search. Weak results send the agent back to search
  again with feedback; after `CRAG_MAX_RETRIES` it answers from what it has and says what's missing.
- Context compression when the research log outgrows its token budget, with a list of searches
  already run so the agent doesn't repeat itself.
- Document versioning: uploading or deleting a file bumps the user's library version. A chat
  started on an older version gets a fresh agent thread on its next message, so it can't answer
  from documents that are gone or changed.
- Streaming UI (SSE) showing which step the agent is on.

## The graph

```
router ──► direct_chat | inventory | overview | clarify | no_documents
   └────► summarize ─► rewrite ─┬─► await_clarification (interrupt, resumes into rewrite)
                                └─► Send(research) per sub-question ─► aggregate

research (one per sub-question):
orchestrator ─► tools ─► grade ─┬─► maybe_compress ─► (compress) ─► orchestrator
     │                          ├─► orchestrator          (retry with feedback)
     │                          └─► fallback ─► collect   (retry limit hit)
     └─► collect                (no more tool calls)
```

Compound questions get split by `rewrite` and researched in parallel via `Send`, then merged.
The research subgraph runs isolated, so its tool calls never land in the chat's message list.

## Run

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

pip install -r requirements.txt
cp .env.example .env
python run.py            # http://127.0.0.1:8000
```

Or `docker compose up` for app + Qdrant + Ollama (then pull the two models inside the ollama
container).

Any OpenAI-compatible endpoint works for the LLM: set `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`.
The model needs tool calling; qwen2.5 7B and up is fine.

### Google Colab

Download [`scripts/run_agentic_docqa_colab.py`](scripts/run_agentic_docqa_colab.py), choose a GPU
under **Runtime > Change runtime type** in Colab, and upload the Python file using the Files sidebar.
Run these commands in separate notebook cells:

```python
%run /content/run_agentic_docqa_colab.py
```

```python
docqa.upload()  # select PDF, Markdown or TXT documents
```

```python
answer = docqa.ask("Summarize my document")
```

The runner includes a snapshot of the project source, installs dependencies and Ollama, downloads
the models, and checks GPU use. It requires no Docker or GitHub token. The app runs locally inside
Colab and is used through the notebook. Initial setup and model downloads take time.
The runner mounts Google Drive and saves to **My Drive/Agentic DocQA**. Allow the mount prompt.
Project source and model/download caches persist there; uploads, extracted text, chunks, chats,
account state, logs, and the search index are checkpointed after each upload or completed question.
Previous checkpoints are retained in `snapshots/`, and the latest one is restored on a fresh runtime.
Use `docqa.save()` for an extra checkpoint and `docqa.stop()` before rerunning setup.
The live database stays on Colab's local disk; backups use SQLite's online backup API.
Python packages are reinstalled on a fresh runtime, and in-progress agent execution restarts after
a disconnect. Model caches occupy several GB of Drive storage.

After the first run, the runner itself is saved in Drive, so subsequent sessions can start with:

```python
from google.colab import drive
drive.mount("/content/drive")
%run "/content/drive/MyDrive/Agentic DocQA/run_agentic_docqa_colab.py"
```

## Config

See `.env.example`. Agent limits (`MAX_TOOL_CALLS`, `MAX_ITERATIONS`, `CRAG_MAX_RETRIES`) and
chunk sizes are in `app/config.py` and can all be set by env var.

Set `RERANKER=llm` to rerank with the LLM instead of downloading a cross-encoder, or `none` to skip.
Set `LLAMA_CLOUD_API_KEY` to parse PDFs with LlamaParse instead of pymupdf4llm.

## Layout

```
app/
  agent/      graph, nodes, state, tools, prompts
  core/       user store, chats + versioning, documents, per-user rag runtime, reranker
  db/         qdrant hybrid store, parent store
  web/        fastapi app + templates
  chunker.py  parent/child splitting
run.py
```

## Limits

- Users, sessions and chats are JSON files. Fine for a handful of users, not for real traffic.
- LangGraph checkpoints are in memory, so a restart drops agent threads (chat history survives).
- Scanned PDFs need OCR before upload.
