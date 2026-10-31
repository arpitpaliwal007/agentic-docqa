"""Offline evaluation tests; the optional integration cases also need requirements installed."""
import ast
import copy
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eval.metrics import retrieval_metrics, validate_golden, validate_judgment
from eval.run_eval import ROOT, fixtures, judge_answer
from eval.run_eval import main
from eval.storage import Results, prepare_workdir
from eval.trace import Trace
from eval.report import render_report


class FixtureAndMetricTests(unittest.TestCase):
    def setUp(self):
        self.corpus, self.golden = fixtures()

    def test_golden_schema_and_literal_support(self):
        counts = validate_golden(self.golden, self.corpus)
        self.assertEqual(sum(counts.values()), 20)
        self.assertEqual(counts["unanswerable"], 4)

    def test_unsupported_evidence_is_rejected(self):
        broken = copy.deepcopy(self.golden)
        broken[0]["evidence"][0]["phrase"] = "a nonexistent policy"
        with self.assertRaises(ValueError):
            validate_golden(broken, self.corpus)

    def test_hit_mrr_precision_and_source_pairing(self):
        q = self.golden[0]
        hits = [{"source": "wrong.md", "content": "30 calendar days"},
                {"source": "refunds.md", "content": "Eligible within 30 CALENDAR\nDAYS of delivery"}]
        metrics = retrieval_metrics(hits, q)
        self.assertEqual(metrics["hit_at_5"], 1)
        self.assertEqual(metrics["mrr_at_5"], .5)
        self.assertEqual(metrics["precision_at_5"], .2)
        self.assertEqual(metrics["evidence_coverage_at_5"], 1)

    def test_multisource_coverage_and_unanswerable_exclusion(self):
        q = self.golden[6]
        metrics = retrieval_metrics([{"source": "warranty.md", "content": "24 months"}], q)
        self.assertEqual(metrics["evidence_coverage_at_5"], .5)
        self.assertEqual(retrieval_metrics([], self.golden[0])["mrr_at_5"], 0)
        self.assertTrue(all(v is None for v in retrieval_metrics([], self.golden[-1]).values()))

    def test_invalid_judge_numbers_and_invented_quotes_are_rejected(self):
        for number in (True, float("nan"), -1, 2):
            with self.assertRaises(ValueError):
                validate_judgment({"correctness": number, "abstained": False, "claims": []}, "")
        with self.assertRaises(ValueError):
            validate_judgment({"correctness": 1, "abstained": False, "claims": [
                {"claim": "invented", "supported": True, "evidence_quote": "not in evidence"}]}, "real evidence")
        score = validate_judgment({"correctness": .5, "abstained": False, "claims": [
            {"claim": "one", "supported": True, "evidence_quote": "real"},
            {"claim": "two", "supported": False}]}, "real evidence")
        self.assertEqual(score["faithfulness"], .5)

    def test_trace_counts_retry_events_after_counter_reset(self):
        trace = Trace()
        trace({"kind": "grade", "verdict": "insufficient", "origin": "llm", "retry_scheduled": True})
        trace({"kind": "grade", "verdict": "sufficient", "origin": "llm", "retry_scheduled": False})
        trace({"kind": "fallback", "reason": "agent_budget"})
        self.assertEqual(trace.stats()["retry_events"], 1)
        self.assertEqual(trace.stats()["llm_insufficient_events"], 1)
        self.assertEqual(trace.stats()["fallback_events"], 1)

    def test_evidence_contains_only_captured_context_and_deduplicates_children(self):
        trace = Trace()
        trace({"kind": "parent", "content": "File: refunds.md\nA literal source sentence.", "error": None})
        trace({"kind": "search", "chunks": [{"content": "A literal source sentence.", "source": "refunds.md", "parent_id": "p0"}], "error": None})
        self.assertEqual(trace.evidence().count("A literal source sentence."), 1)

    def test_resume_preserves_completed_questions_and_rejects_incompatible_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Results(directory, "stable", {"fingerprint": "same"})
            state.put({"id": "q01", "status": "complete"})
            state.put({"id": "q02", "status": "error"})
            resumed = Results(directory, "stable", {"fingerprint": "same"})
            self.assertTrue(resumed.completed("q01"))
            self.assertFalse(resumed.completed("q02"))
            with self.assertRaises(ValueError):
                Results(directory, "stable", {"fingerprint": "different"})

    def test_cleanup_is_confined_to_marked_eval_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unrelated = root / "data"
            unrelated.mkdir()
            (unrelated / "keep.txt").write_text("private user data")
            work = prepare_workdir(root, "safe", "fingerprint")
            (work / "stale.txt").write_text("eval state")
            prepare_workdir(root, "safe", "fingerprint")
            self.assertFalse((work / "stale.txt").exists())
            self.assertEqual((unrelated / "keep.txt").read_text(), "private user data")
            with self.assertRaises(ValueError):
                prepare_workdir(root, "safe", "other")
            with self.assertRaises(ValueError):
                prepare_workdir(root, "../data", "fingerprint")

    def test_failed_partial_write_preserves_last_completed_question(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Results(directory, "stable", {"fingerprint": "same"})
            state.put({"id": "q01", "status": "complete"})
            with patch("eval.storage.json.dump", side_effect=OSError("Drive disconnected")):
                with self.assertRaises(OSError):
                    state.put({"id": "q02", "status": "complete"})
            restored = Results(directory, "stable", {"fingerprint": "same"})
            self.assertTrue(restored.completed("q01"))
            self.assertFalse(restored.completed("q02"))

    def test_empty_report_shows_unavailable_scores_without_inventing_failures(self):
        value = {"metadata": {"run_id": "toy", "model": "test", "judge_model": "test", "self_judge": True,
                              "expected_questions": 20, "fingerprint": "example"}, "results": {}}
        text = render_report(value)
        self.assertIn("| correctness | N/A | 0 |", text)
        self.assertIn("No scored failures are available", text)

    def test_notebook_valid_json_and_code_with_no_outputs(self):
        notebook = json.loads((ROOT / "eval/colab_eval.ipynb").read_text())
        self.assertEqual(notebook["nbformat"], 4)
        ids = set()
        for index, cell in enumerate(notebook["cells"]):
            self.assertNotIn(cell["id"], ids)
            ids.add(cell["id"])
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(notebook["cells"][index - 1]["cell_type"], "markdown")
                ast.parse("".join(cell["source"]))


HAVE_LANGGRAPH = importlib.util.find_spec("langgraph") is not None


@unittest.skipUnless(HAVE_LANGGRAPH, "install project dependencies for real graph integration tests")
class PipelineIntegrationTests(unittest.TestCase):
    def fake_model(self, answers):
        from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
        class ToolModel(FakeMessagesListChatModel):
            def bind_tools(self, tools, **kwargs):
                return self
        return ToolModel(responses=answers)

    def tools(self, observer=None):
        from langchain_core.documents import Document
        from app.agent.tools import build_tools
        class Store:
            def similarity_search_with_score(self, query, k):
                return [(Document(page_content="Returns within 30 calendar days.", metadata={"source": "refunds.md", "parent_id": "p0"}), .9)]
        class Parents:
            def get(self, parent_id):
                return {"content": "Returns within 30 calendar days.", "metadata": {"source": "refunds.md"}}
        return build_tools(Store(), Parents(), reranker_mode="none", observer=observer)

    def test_real_graph_and_user_rag_ask_preserve_answer_with_observer(self):
        from langchain_core.messages import AIMessage
        from app.agent.graph import build_graph
        from app.core.rag import UserRAG
        from langgraph.checkpoint.memory import InMemorySaver
        from types import SimpleNamespace
        replies = [AIMessage('{"route":"document_qa"}'),
                   AIMessage('{"is_clear":true,"questions":["return window"]}'),
                   AIMessage(content="", tool_calls=[{"name": "search_child_chunks", "args": {"query": "return window"}, "id": "call0", "type": "tool_call"}]),
                   AIMessage('{"grade":"sufficient","reason":"has the time limit"}'),
                   AIMessage("30 calendar days. Sources: refunds.md")]
        answers = []
        trace = Trace()
        for observer in (None, trace):
            rag = object.__new__(UserRAG)
            rag.docs = SimpleNamespace(list=lambda: ["refunds.md"], previews=lambda: [])
            rag.graph = build_graph(self.fake_model(replies), self.tools(observer), InMemorySaver(), observer)
            answers.append(rag.ask("isolated-test-thread", "According to the document, what is the return window?", []))
        self.assertEqual(answers, ["30 calendar days. Sources: refunds.md"] * 2)
        self.assertEqual(trace.stats()["grade_events"], 1)
        self.assertEqual(trace.stats()["routes"], ["document_qa"])
        self.assertIn("30 calendar days", trace.evidence())

    def test_real_grade_insufficient_and_retry_limit_fallback(self):
        from langchain_core.messages import AIMessage, ToolMessage
        from app.agent.nodes import grade
        from app import config
        state = {"index": 0, "question": "return window", "messages": [ToolMessage("[Chunk 1]\n30 calendar days", tool_call_id="t")], "retries": 0}
        trace = Trace()
        with patch.object(config, "CRAG_MAX_RETRIES", 1):
            first = grade(state, self.fake_model([AIMessage('{"grade":"insufficient","reason":"missing"}')]), trace)
            state["retries"] = 1
            last = grade(state, self.fake_model([AIMessage('{"grade":"insufficient","reason":"missing"}')]), trace)
        self.assertEqual(first.goto, "orchestrator")
        self.assertEqual(last.goto, "fallback")
        self.assertEqual(trace.stats()["retry_events"], 1)
        self.assertEqual(trace.stats()["insufficient_events"], 2)

    def test_observer_failure_does_not_change_tool_output(self):
        def broken(event):
            raise RuntimeError("observer failed")
        with self.assertLogs("app.agent.observability", level="ERROR"):
            actual = self.tools(broken)[0].invoke({"query": "returns"})
        expected = self.tools()[0].invoke({"query": "returns"})
        self.assertEqual(actual, expected)

    def test_dense_and_hybrid_use_the_same_real_qdrant_collection(self):
        from langchain_core.embeddings import Embeddings
        from langchain_core.documents import Document
        from langchain_qdrant import RetrievalMode
        from langchain_qdrant.sparse_embeddings import SparseEmbeddings, SparseVector
        from qdrant_client import QdrantClient
        from app.db.vector_store import VectorStore
        from app import config
        class Dense(Embeddings):
            def embed_query(self, text):
                return [1., 0., 0.] if "refund" in text else [0., 1., 0.]
            def embed_documents(self, texts):
                return [self.embed_query(text) for text in texts]
        class Sparse(SparseEmbeddings):
            def embed_query(self, text):
                return SparseVector(indices=[0 if "refund" in text else 1], values=[1.])
            def embed_documents(self, texts):
                return [self.embed_query(text) for text in texts]
        client = QdrantClient(":memory:")
        self.addCleanup(client.close)
        with patch.object(VectorStore, "_client", client), patch.object(VectorStore, "_dense", Dense()), patch.object(VectorStore, "_sparse", Sparse()), patch.object(config, "DENSE_DIM", 3):
            store = VectorStore()
            hybrid = store.store("test_eval_shared")
            self.assertEqual(hybrid.retrieval_mode, RetrievalMode.HYBRID)
            hybrid.add_documents([Document(page_content="refund window", metadata={"source":"refunds.md"}),
                                  Document(page_content="shipping window", metadata={"source":"shipping.md"})])
            dense = store.store("test_eval_shared", retrieval_mode="dense")
            self.assertEqual(dense.retrieval_mode, RetrievalMode.DENSE)
            for variant in (dense, hybrid):
                self.assertEqual(variant.similarity_search("refund", k=1)[0].metadata["source"], "refunds.md")
            with self.assertRaises(ValueError):
                store.store("test_eval_shared", retrieval_mode="unknown")

    def test_judge_failure_and_context_overflow_are_explicit(self):
        from langchain_core.messages import AIMessage
        invalid = self.fake_model([AIMessage("not JSON")])
        q = fixtures()[1][0]
        value, attempts, _ = judge_answer(invalid, q, "answer", "context", 24000)
        self.assertIsNone(value)
        self.assertEqual(len(attempts), 2)
        valid = self.fake_model([AIMessage('{"correctness":1,"abstained":true,"claims":[]}')])
        value, _, _ = judge_answer(valid, q, "I cannot answer", "x" * 2000, 1000)
        self.assertIsNone(value["faithfulness"])
        self.assertIn("context budget", value["faithfulness_unavailable_reason"])

    def test_cli_writes_durable_results_and_skips_completed_questions(self):
        from types import SimpleNamespace
        from app import config
        from app.db.vector_store import VectorStore  # Load NumPy before the temporary sys.modules patch.
        from eval.report import MODES
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            users = SimpleNamespace(collection=lambda name: "isolated_eval_collection")
            fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
            observation = {mode: {"metrics": retrieval_metrics([], fixtures()[1][0]), "hits": [], "latency_seconds": 0.0} for mode in MODES}
            args = ["--model", "test-model", "--skip-e2e", "--limit", "1", "--out", str(root / "results"), "--work-root", str(root / "work")]
            with patch.dict(os.environ), patch.dict("sys.modules", {"torch": fake_torch}), \
                 patch.multiple(config, DATA_DIR=root / "unused", QDRANT_PATH=str(root / "unused_db"), LLM_MODEL="test-model", RERANK_DEVICE="cpu"), \
                 patch("eval.run_eval.runtime_info", return_value={"packages": {}, "gpu": "mock"}), \
                 patch("eval.run_eval.ollama_info", return_value={"version": "mock", "model_digests": {}}), \
                 patch("eval.run_eval.initialize_pipeline", return_value=(users, None, None, None, Trace(), "cpu")) as setup, \
                 patch("eval.run_eval.score_retrieval", return_value=observation) as score, \
                 patch("app.db.vector_store.VectorStore._close"), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
                self.assertEqual(main(args), 0)
                self.assertEqual(setup.call_count, 1)
                self.assertEqual(score.call_count, 1)
            files = list((root / "results").glob("*.partial.json"))
            self.assertEqual(len(files), 1)
            saved = json.loads(files[0].read_text())
            self.assertEqual(list(saved["results"]), ["q01"])
            self.assertEqual(saved["results"]["q01"]["status"], "complete")
            self.assertEqual(len(list((root / "results").glob("*.md"))), 1)


if __name__ == "__main__":
    unittest.main()
