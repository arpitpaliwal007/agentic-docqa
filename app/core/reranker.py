"""Rerankers. Cross-encoder by default; LLM-scored as a no-download fallback."""
import logging

from app import config
from app.utils import parse_json

log = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query, chunks):
        if not chunks:
            return []
        scores = self.model.predict([(query, c["content"]) for c in chunks])
        return sorted(zip(chunks, map(float, scores)), key=lambda x: x[1], reverse=True)


class LLMReranker:
    def __init__(self, llm):
        self.llm = llm

    def rerank(self, query, chunks):
        listing = "\n\n".join(f"[{i}] {c['content'][:600]}" for i, c in enumerate(chunks))
        prompt = (
            "Score each passage 0-10 for how well it answers the query. "
            'Reply with JSON only: {"scores": [..one number per passage..]}\n\n'
            f"Query: {query}\n\nPassages:\n{listing}"
        )
        try:
            scores = (parse_json(self.llm.invoke(prompt).content) or {}).get("scores") or []
        except Exception as e:
            log.warning("llm rerank failed: %s", e)
            scores = []
        scores = [float(s) for s in scores][: len(chunks)] + [0.0] * max(0, len(chunks) - len(scores))
        return sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)


_cache = {}


def get_reranker(llm=None):
    kind = config.RERANKER
    if kind == "none":
        return None
    if kind not in _cache:
        try:
            _cache[kind] = CrossEncoderReranker(config.RERANK_MODEL) if kind == "cross_encoder" else LLMReranker(llm)
        except Exception as e:
            log.warning("reranker %s unavailable (%s), continuing without", kind, e)
            _cache[kind] = None
    return _cache[kind]
