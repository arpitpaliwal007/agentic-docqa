"""Real production ingestion, retrieval tools and UserRAG, in an isolated eval DB."""
import copy
from pathlib import Path
import re
import secrets
import shutil
import time
import uuid

from eval.qasper.metrics import captured_evidence, retrieval_metrics
from eval.trace import Trace

MODES = ('dense', 'hybrid', 'hybrid_rerank')


class PaperSession:
    def __init__(self, backend, username, paper_ids):
        from app.core.documents import DocumentManager
        from app.core.rag import UserRAG
        from app.utils import safe_stem
        self.backend, self.username = backend, username
        self.collection = backend.users.collection(username)
        self.directory = backend.users.user_dir(username)
        ok, message = backend.users.register(username, secrets.token_urlsafe(24))
        if not ok:
            raise RuntimeError(message)
        self.docs = DocumentManager(self.directory, self.collection)
        for pid in paper_ids:
            paper = backend.papers[pid]
            self.docs.add(paper['source'], paper['markdown'].encode())
            if backend.prefix_ablation:
                _, children = self.docs.chunker.chunk(self.docs.md_dir / f"{safe_stem(paper['source'])}.md",
                                                       source=paper['source'], stem=safe_stem(paper['source']))
                backend.vdb.store(self.collection + '_prefix_off', nomic_prefixes=False).add_documents(children)
            backend.indexed_papers.append(pid)
        self.trace = Trace()
        self.rag = UserRAG(backend.users, username, observer=self.trace)

    def retrieval(self, paper, question):
        from app.agent.tools import build_tools
        output = {}
        for prefixes in ((False, True) if self.backend.prefix_ablation else (True,)):
            collection = self.collection if prefixes else self.collection + '_prefix_off'
            for mode in MODES:
                trace = Trace()
                store = self.backend.vdb.store(collection, retrieval_mode='dense' if mode == 'dense' else 'hybrid', nomic_prefixes=prefixes)
                search = build_tools(store, self.docs.parents, reranker_mode='cross_encoder' if mode == 'hybrid_rerank' else 'none', observer=trace)[0]
                start = time.perf_counter()
                raw = search.invoke({'query':question['question'], 'limit':10})
                events = [event for event in trace.events if event['kind'] == 'search']
                if not events or events[-1].get('error') or raw.startswith('RETRIEVAL_ERROR'):
                    raise RuntimeError(f'{mode} search failed: {raw}')
                hits = events[-1]['chunks']
                output[f"prefix_{'on' if prefixes else 'off'}_{mode}"] = {
                    'prefixes':prefixes, 'mode':mode, 'metrics':retrieval_metrics(paper, question, hits),
                    'hits':hits, 'query':question['question'], 'latency_seconds':time.perf_counter()-start}
        return output

    def ask(self, question):
        self.trace.reset()
        thread_id = str(uuid.uuid4())
        record = {'thread_id':thread_id}
        started = time.perf_counter()
        try:
            record['answer'] = self.rag.ask(thread_id, question, history=[])
        except Exception as error:
            record['execution_error'] = f'{type(error).__name__}: {error}'
        finally:
            record.update(latency_seconds=time.perf_counter()-started, events=copy.deepcopy(self.trace.events), crag=self.trace.stats())
            record['evidence'] = captured_evidence(record['events'])
            record['agent_searches'] = [event for event in record['events'] if event['kind'] == 'search']
            record['agent_parents'] = [event for event in record['events'] if event['kind'] == 'parent']
            self.rag.drop_thread(thread_id)
        return record

    def close(self):
        for name in (self.collection, self.collection + '_prefix_off'):
            if self.backend.vdb.client.collection_exists(name):
                self.backend.vdb.client.delete_collection(name)
        if self.directory.resolve().is_relative_to(self.backend.work.resolve()):
            shutil.rmtree(self.directory)
        else:
            raise ValueError('Refusing to clean a user directory outside the isolated eval work root')


class QasperBackend:
    def __init__(self, work, papers, mode, prefix_ablation):
        from app.core.user_store import UserStore
        from app.core.reranker import get_reranker
        from app.db.vector_store import VectorStore
        self.work, self.papers = Path(work), papers
        self.mode, self.prefix_ablation = mode, prefix_ablation
        reranker = get_reranker(kind='cross_encoder')
        if reranker is None:
            raise RuntimeError('Cross-encoder unavailable; refusing to report a reranked condition')
        self.reranker_device = str(reranker.model.device)
        self.users, self.vdb = UserStore(), VectorStore()
        self.sessions, self.indexed_papers = {}, []

    def session(self, paper_id, all_paper_ids):
        key = paper_id if self.mode == 'per_paper' else 'pooled'
        if key not in self.sessions:
            name = 'qasper_' + re.sub(r'[^a-zA-Z0-9_.-]', '_', key)
            if len(name) > 32:
                import hashlib
                name = 'qasper_' + hashlib.sha256(key.encode()).hexdigest()[:24]
            ids = [paper_id] if self.mode == 'per_paper' else all_paper_ids
            self.sessions[key] = PaperSession(self, name, ids)
        return self.sessions[key]

    def finish_paper(self, paper_id):
        if self.mode == 'per_paper' and paper_id in self.sessions:
            self.sessions.pop(paper_id).close()

    def close(self):
        from app.db.vector_store import VectorStore
        try:
            for session in list(self.sessions.values()):
                session.close()
        finally:
            self.vdb.client.close()
            with VectorStore._lock:
                VectorStore._client = VectorStore._dense = VectorStore._sparse = None
