"""The two tools the orchestrator gets: child search and parent fetch."""
from langchain_core.tools import tool

from app import config
from app.core.reranker import get_reranker
from app.agent.observability import emit


def build_tools(vector_store, parent_store, llm=None, *, reranker_mode=None, observer=None):
    reranker = get_reranker(llm, kind=reranker_mode)

    @tool
    def search_child_chunks(query: str, limit: int = 10) -> str:
        """Hybrid (dense + BM25) search over small document chunks. Returns matching
        snippets with their Parent ID and file name."""
        try:
            k = max(int(limit or 0), config.SEARCH_TOP_K)
            hits = vector_store.similarity_search_with_score(query, k=k)
            hits = [(d, s) for d, s in hits if s >= config.SCORE_THRESHOLD]
            if not hits:
                emit(observer, "search", query=query, chunks=[], error=None)
                return "NO_RELEVANT_CHUNKS"
            chunks = [{"content": d.page_content.strip(), "parent_id": d.metadata.get("parent_id", ""),
                       "source": d.metadata.get("source", "")} for d, _ in hits]
            if reranker:
                chunks = [c for c, _ in reranker.rerank(query, chunks)][: config.RERANK_TOP_M]
            picked, seen = [], set()
            for c in chunks:  # one snippet per parent
                if c["parent_id"] in seen:
                    continue
                seen.add(c["parent_id"])
                picked.append(c)
                if len(picked) >= config.FINAL_TOP_N:
                    break
            output = "\n\n".join(
                f"[Chunk {i}]\nParent ID: {c['parent_id']}\nFile: {c['source']}\n{c['content']}"
                for i, c in enumerate(picked, 1))
            emit(observer, "search", query=query, chunks=picked, content=output, error=None)
            return output
        except Exception as e:
            emit(observer, "search", query=query, chunks=[], error=str(e))
            return f"RETRIEVAL_ERROR: {e}"

    @tool
    def retrieve_parent_chunks(parent_id: str) -> str:
        """Fetch the full parent section for a Parent ID returned by search_child_chunks."""
        p = parent_store.get(parent_id)
        if not p:
            emit(observer, "parent", parent_id=parent_id, error="not found")
            return f"NO_PARENT_DOCUMENT: {parent_id}"
        output = f"Parent ID: {parent_id}\nFile: {p['metadata'].get('source', '?')}\n{p['content'].strip()}"
        emit(observer, "parent", parent_id=parent_id, source=p['metadata'].get('source', '?'),
             content=output, text=p['content'].strip(), error=None)
        return output

    return [search_child_chunks, retrieve_parent_chunks]
