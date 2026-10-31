"""Qdrant hybrid store: dense (Ollama) + sparse (BM25 via fastembed), one collection per user."""
import atexit
import threading

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app import config


class TaskPrefixEmbeddings(Embeddings):
    """Apply Nomic task instructions only to embedding inputs, never stored text."""
    def __init__(self, base, model, enabled=True):
        self.base = base
        self.enabled = enabled and model.lower().startswith("nomic-embed")

    def embed_documents(self, texts):
        inputs = ["search_document: " + text for text in texts] if self.enabled else list(texts)
        return self.base.embed_documents(inputs)

    def embed_query(self, text):
        return self.base.embed_query("search_query: " + text if self.enabled else text)

    async def aembed_documents(self, texts):
        inputs = ["search_document: " + text for text in texts] if self.enabled else list(texts)
        return await self.base.aembed_documents(inputs)

    async def aembed_query(self, text):
        return await self.base.aembed_query("search_query: " + text if self.enabled else text)


class VectorStore:
    _lock = threading.Lock()
    _client = None
    _dense = None
    _sparse = None

    def __init__(self):
        # embedded qdrant locks its directory, so one client per process
        with VectorStore._lock:
            if VectorStore._client is None:
                VectorStore._client = (
                    QdrantClient(url=config.QDRANT_URL) if config.QDRANT_URL
                    else QdrantClient(path=config.QDRANT_PATH)
                )
                atexit.register(VectorStore._close)
                VectorStore._dense = OllamaEmbeddings(model=config.DENSE_MODEL, base_url=config.OLLAMA_HOST)
                VectorStore._sparse = FastEmbedSparse(model_name=config.SPARSE_MODEL)
        self.client = VectorStore._client

    @staticmethod
    def _close():
        try:
            if VectorStore._client:
                VectorStore._client.close()
        except Exception:
            pass

    def ensure(self, name):
        if not self.client.collection_exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config=qm.VectorParams(size=config.DENSE_DIM, distance=qm.Distance.COSINE),
                sparse_vectors_config={config.SPARSE_NAME: qm.SparseVectorParams()},
            )

    def store(self, name, retrieval_mode=None, *, nomic_prefixes=None) -> QdrantVectorStore:
        mode = retrieval_mode or config.RETRIEVAL_MODE
        modes = {"dense": RetrievalMode.DENSE, "hybrid": RetrievalMode.HYBRID}
        if mode not in modes:
            raise ValueError("retrieval_mode must be dense or hybrid")
        self.ensure(name)
        return QdrantVectorStore(
            client=self.client,
            collection_name=name,
            embedding=TaskPrefixEmbeddings(VectorStore._dense, config.DENSE_MODEL,
                                           config.NOMIC_PREFIXES if nomic_prefixes is None else nomic_prefixes),
            sparse_embedding=VectorStore._sparse,
            retrieval_mode=modes[mode],
            sparse_vector_name=config.SPARSE_NAME,
            validate_collection_config=False,
        )

    def _ids_for_source(self, name, source):
        flt = qm.Filter(must=[qm.FieldCondition(key="metadata.source", match=qm.MatchValue(value=source))])
        ids, offset = [], None
        while True:
            batch, offset = self.client.scroll(name, scroll_filter=flt, limit=256, offset=offset,
                                               with_payload=False, with_vectors=False)
            ids += [p.id for p in batch]
            if offset is None:
                return ids

    def delete_source(self, name, source):
        if not self.client.collection_exists(name):
            return 0
        ids = self._ids_for_source(name, source)
        if ids:
            self.client.delete(name, points_selector=qm.PointIdsList(points=ids), wait=True)
        return len(ids)

    def count(self, name):
        if not self.client.collection_exists(name):
            return 0
        return self.client.count(name).count
