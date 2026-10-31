"""Env-driven settings."""
import os
from pathlib import Path


def _s(name, default=""):
    v = os.environ.get(name, "").strip()
    return v or default


def _i(name, default):
    return int(_s(name, str(default)))


def _f(name, default):
    return float(_s(name, str(default)))


def _b(name, default):
    v = _s(name, "")
    return default if not v else v.lower() in {"1", "true", "yes", "on"}


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(_s("DATA_DIR", str(BASE_DIR / "data")))
QDRANT_PATH = _s("QDRANT_PATH", str(BASE_DIR / "qdrant_db"))
QDRANT_URL = _s("QDRANT_URL", "")

# models
LLM_BASE_URL = _s("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = _s("LLM_API_KEY", "ollama")
LLM_MODEL = _s("LLM_MODEL", "qwen2.5:7b")
LLM_TEMPERATURE = _f("LLM_TEMPERATURE", 0.0)
OLLAMA_HOST = _s("OLLAMA_HOST", "http://localhost:11434")
DENSE_MODEL = _s("DENSE_MODEL", "nomic-embed-text")
DENSE_DIM = _i("DENSE_DIM", 768)
SPARSE_MODEL = _s("SPARSE_MODEL", "Qdrant/bm25")
SPARSE_NAME = "sparse"
RETRIEVAL_MODE = _s("RETRIEVAL_MODE", "hybrid").lower()

# reranker: cross_encoder | llm | none
RERANKER = _s("RERANKER", "cross_encoder").lower()
RERANK_MODEL = _s("RERANK_MODEL", "BAAI/bge-reranker-base")
RERANK_DEVICE = _s("RERANK_DEVICE", "")  # empty preserves sentence-transformers' automatic device choice
SEARCH_TOP_K = _i("SEARCH_TOP_K", 10)
RERANK_TOP_M = _i("RERANK_TOP_M", 5)
FINAL_TOP_N = _i("FINAL_TOP_N", 5)
SCORE_THRESHOLD = _f("SCORE_THRESHOLD", 0.0)

# chunking
CHILD_SIZE = _i("CHILD_SIZE", 500)
CHILD_OVERLAP = _i("CHILD_OVERLAP", 100)
MIN_PARENT = _i("MIN_PARENT", 2000)
MAX_PARENT = _i("MAX_PARENT", 4000)
HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
MAX_UPLOAD_MB = _i("MAX_UPLOAD_MB", 100)

# agent limits
MAX_TOOL_CALLS = _i("MAX_TOOL_CALLS", 8)
MAX_ITERATIONS = _i("MAX_ITERATIONS", 10)
CRAG_MAX_RETRIES = _i("CRAG_MAX_RETRIES", 2)
RECURSION_LIMIT = _i("RECURSION_LIMIT", 50)
COMPRESS_BASE_TOKENS = _i("COMPRESS_BASE_TOKENS", 2000)
COMPRESS_GROWTH = _f("COMPRESS_GROWTH", 0.9)

# auth
SESSION_TTL = _i("SESSION_TTL", 7 * 24 * 3600)
COOKIE_NAME = "docqa_session"

# observability
LANGFUSE_ENABLED = _b("LANGFUSE_ENABLED", False)
