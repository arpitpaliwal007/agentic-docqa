#!/usr/bin/env python3
"""Evaluate the real pipeline. Offline fixture check: python eval/run_eval.py --validate-only."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.metrics import METRIC_POLICY, add_diagnostics, classify_failure, retrieval_metrics, validate_golden
from eval.report import MODES, render_report
from eval.storage import Results, atomic_json, code_fingerprint, digest, prepare_workdir
from eval.trace import Trace
from eval.judge import JUDGE_POLICY, JUDGE_VARIANTS, judge_answer, judge_failure, make_judge


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=os.environ.get("EVAL_LLM_MODEL", "qwen2.5:7b"))
    p.add_argument("--judge-model", default=os.environ.get("EVAL_JUDGE_MODEL"))
    p.add_argument("--limit", type=int, help="First N questions; use a separate --out for smoke tests")
    p.add_argument("--out", type=Path, default=ROOT / "results")
    p.add_argument("--run-id", help="Optional stable label; incompatible existing results are rejected")
    p.add_argument("--work-root", type=Path, default=Path(tempfile.gettempdir()) / "agentic-docqa-eval-work")
    p.add_argument("--skip-e2e", action="store_true")
    p.add_argument("--prefix-ablation", action=argparse.BooleanOptionalAction, default=True,
                   help="Compare independently indexed prefixes-off/on collections (default: on)")
    p.add_argument("--validate-only", action="store_true", help="Validate corpus/golden data without model dependencies")
    p.add_argument("--judge-context-chars", type=int, default=24000)
    args = p.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        p.error("--limit must be positive")
    if args.judge_context_chars < 1000:
        p.error("--judge-context-chars must be at least 1000")
    return args


def fixtures():
    corpus = {p.name: p.read_text(encoding="utf-8") for p in sorted((ROOT / "eval/corpus").glob("*.md"))}
    golden = json.loads((ROOT / "eval/golden.json").read_text())
    validate_golden(golden, corpus)
    return corpus, golden


def runtime_info():
    packages = {}
    for name in ("langgraph", "langchain-core", "langchain-openai", "langchain-ollama", "langchain-qdrant",
                 "qdrant-client", "fastembed", "sentence-transformers", "torch", "numpy", "pydantic"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        gpu = "unavailable"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        commit = "ZIP source (see code fingerprint)"
    return {"python": sys.version, "packages": packages, "gpu": gpu, "git_commit": commit}


def ollama_info(models):
    base = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    with urllib.request.urlopen(base + "/api/version", timeout=10) as response:
        version = json.load(response)
    with urllib.request.urlopen(base + "/api/tags", timeout=10) as response:
        tags = json.load(response).get("models", [])
    digests = {name: next((item.get("digest") for item in tags
                          if item.get("name") in {name, name + ":latest"}), "not pulled") for name in models}
    return {"version": version, "model_digests": digests,
            "context_length_environment": os.environ.get("OLLAMA_CONTEXT_LENGTH", "server default")}


def initialize_pipeline(work, corpus, skip_e2e, prefix_ablation=True):
    from app.core.user_store import UserStore
    from app.core.documents import DocumentManager
    from app.db.vector_store import VectorStore
    from app.core.reranker import get_reranker
    users = UserStore()
    ok, message = users.register("eval_user", secrets.token_urlsafe(24))
    if not ok:
        raise RuntimeError(f"Could not create isolated evaluation user: {message}")
    docs = DocumentManager(users.user_dir("eval_user"), users.collection("eval_user"))
    for name, content in corpus.items():
        docs.add(name, content.encode())
    reranker = get_reranker(kind="cross_encoder")
    if reranker is None:
        raise RuntimeError("Cross-encoder failed to load; refusing to label an unreranked run as reranked")
    vector_store = VectorStore()
    if prefix_ablation:
        # This second collection exists only inside the freshly rebuilt eval DB.
        # Re-embedding is necessary: querying prefixed vectors with unprefixed queries
        # would confound the ablation. Stored chunks and parent text stay identical.
        from app.utils import safe_stem
        unprefixed = vector_store.store(users.collection("eval_user") + "_prefix_off", nomic_prefixes=False)
        for name in corpus:
            stem = safe_stem(name)
            _, children = docs.chunker.chunk(docs.md_dir / f"{stem}.md", source=name, stem=stem)
            unprefixed.add_documents(children)
    trace = Trace()
    rag = None
    if not skip_e2e:
        from app.core.rag import UserRAG
        rag = UserRAG(users, "eval_user", observer=trace)
    return users, docs, vector_store, rag, trace, str(reranker.model.device)


def score_retrieval(q, docs, vdb, collection, prefix_ablation=True):
    from app.agent.tools import build_tools
    result = {}
    for prefixes in ((False, True) if prefix_ablation else (True,)):
        for mode in MODES:
            label = f"prefix_{'on' if prefixes else 'off'}_{mode}"
            trace = Trace()
            name = collection if prefixes else collection + "_prefix_off"
            store = vdb.store(name, retrieval_mode="dense" if mode == "dense" else "hybrid", nomic_prefixes=prefixes)
            tool = build_tools(store, docs.parents, reranker_mode="cross_encoder" if mode == "hybrid_rerank" else "none", observer=trace)[0]
            started = time.perf_counter()
            output = tool.invoke({"query": q["question"], "limit": 10})
            elapsed = time.perf_counter() - started
            searches = [e for e in trace.events if e["kind"] == "search"]
            if not searches or searches[-1].get("error") or str(output).startswith("RETRIEVAL_ERROR"):
                raise RuntimeError(f"{label} retrieval failed: {output}")
            hits = searches[-1]["chunks"]
            result[label] = {"prefixes": prefixes, "mode": mode, "collection": name,
                             "metrics": retrieval_metrics(hits, q), "hits": hits, "latency_seconds": elapsed}
    return result


def failure_type(q, result):
    return classify_failure(q, result)


def write_reports(results):
    metadata = results.value["metadata"]
    base = results.path.parent / metadata["report_basename"]
    atomic_json(base.with_suffix(".json"), results.value)
    temporary = base.with_suffix(".md.tmp")
    temporary.write_text(render_report(results.value), encoding="utf-8")
    temporary.replace(base.with_suffix(".md"))
    print(f"Report: {base.with_suffix('.md')}", flush=True)
    return base.with_suffix(".md")


def main(argv=None):
    args = parse_args(argv)
    corpus, golden = fixtures()
    if args.validate_only:
        print(f"Validated {len(corpus)} documents and {len(golden)} golden questions; no models loaded.")
        return 0
    # Configure BEFORE importing app modules: the app caches its settings and LLM.
    os.environ.update({"LLM_MODEL": args.model, "LLM_TEMPERATURE": "0", "RETRIEVAL_MODE": "hybrid",
                       "RERANKER": "cross_encoder", "DENSE_MODEL": "nomic-embed-text", "DENSE_DIM": "768",
                       "SEARCH_TOP_K": "10", "RERANK_TOP_M": "5", "FINAL_TOP_N": "5", "SCORE_THRESHOLD": "0",
                       "QDRANT_URL": "", "LANGFUSE_ENABLED": "false"})
    os.environ["NOMIC_PREFIXES"] = "true"
    import torch
    os.environ["RERANK_DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
    from app import config
    excluded = {"BASE_DIR", "DATA_DIR", "QDRANT_PATH", "LLM_API_KEY", "COOKIE_NAME", "SESSION_TTL"}
    settings = {key: value for key, value in vars(config).items()
                if key.isupper() and key not in excluded and isinstance(value, (str, int, float, bool, list))}
    judge_model = args.judge_model or args.model
    runtime = runtime_info()
    model_info = ollama_info(["nomic-embed-text"] if args.skip_e2e else [args.model, judge_model, "nomic-embed-text"])
    identity = {"code_hash": code_fingerprint(ROOT), "corpus_hash": digest(corpus), "golden_hash": digest(golden),
                "settings": settings, "judge_model": judge_model, "skip_e2e": args.skip_e2e,
                "judge_context_chars": args.judge_context_chars, "packages": runtime["packages"], "gpu": runtime["gpu"],
                "ollama": model_info, "python_version": sys.version.split()[0], "judge_policy": JUDGE_POLICY,
                "judge_variants": JUDGE_VARIANTS, "metric_policy": METRIC_POLICY, "eval_version": 2,
                "prefix_ablation": args.prefix_ablation}
    fingerprint = digest(identity)
    run_id = args.run_id or fingerprint[:16]
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    answer_digest = model_info["model_digests"].get(args.model)
    same_weights = answer_digest not in {None, "not pulled"} and answer_digest == model_info["model_digests"].get(judge_model)
    metadata = {**identity, "fingerprint": fingerprint, "run_id": run_id, "model": args.model,
                "self_judge": judge_model == args.model or same_weights, "expected_questions": len(golden),
                "report_basename": f"{now}-{run_id}", "runtime": runtime}
    results = Results(args.out, run_id, metadata)
    results.value.pop("last_setup_or_runtime_error", None)
    selected = golden[:args.limit] if args.limit else golden
    pending = [q for q in selected if not results.completed(q["id"])]
    if not pending:
        print("All selected questions are already complete; no inference repeated.")
        write_reports(results)
        return 0
    work = prepare_workdir(args.work_root, run_id, fingerprint)
    # config.py was already loaded for fingerprinting; only isolated storage paths change.
    config.DATA_DIR = work / "data"
    config.QDRANT_PATH = str(work / "qdrant_db")
    atomic_json(work / "settings.json", identity)
    atomic_json(work / "runtime.json", metadata["runtime"])
    try:
        users, docs, vdb, rag, trace, device = initialize_pipeline(work, corpus, args.skip_e2e, args.prefix_ablation)
        results.value["metadata"]["reranker_device"] = device
        if config.RERANK_DEVICE == "cuda" and not device.startswith("cuda"):
            raise RuntimeError("Requested GPU reranking but the cross-encoder is not on CUDA")
        judge = None
        if not args.skip_e2e:
            judge = make_judge(judge_model, config.LLM_BASE_URL, config.LLM_API_KEY)
        for q in pending:
            print(f"Evaluating {q['id']}: {q['question']}", flush=True)
            result = {**q, "status": "complete", "retrieval": {}, "errors": []}
            try:
                result["retrieval"] = score_retrieval(q, docs, vdb, users.collection("eval_user"), args.prefix_ablation)
                if not args.skip_e2e:
                    trace.reset()
                    thread_id = str(uuid.uuid4())
                    started = time.perf_counter()
                    try:
                        answer = rag.ask(thread_id, q["question"], history=[])
                    finally:
                        elapsed = time.perf_counter() - started
                        result["e2e"] = {"thread_id": thread_id, "latency_seconds": elapsed,
                                         "crag": trace.stats(), "events": list(trace.events), "evidence": trace.evidence()}
                        rag.drop_thread(thread_id)
                    result["e2e"]["answer"] = answer
                    judgment, attempts, judge_latency = judge_answer(judge, q, answer, trace.evidence(), args.judge_context_chars)
                    result["e2e"].update(judgment=judgment, judge_attempts=attempts, judge_latency_seconds=judge_latency,
                                         judge_policy=JUDGE_POLICY, agent_nomic_prefixes=True)
                    if judgment is None:
                        result["errors"].append(judge_failure(attempts))
                        print(f"{q['id']}: {result['errors'][-1]}", flush=True)
                    for event in trace.events:
                        if event["kind"] in {"search", "parent"} and event.get("error"):
                            result["errors"].append(f"Tool failure: {event['error']}")
            except Exception as exc:
                result["errors"].append(f"{type(exc).__name__}: {exc}")
            result["status"] = "error" if result["errors"] else "complete"
            add_diagnostics(q, result)
            result["failure_type"] = failure_type(q, result)
            results.put(result)  # Persist after every question, directly in --out (Drive in the notebook).
            write_reports(results)
        return 1 if any(results.value["results"][q["id"]]["status"] == "error" for q in selected) else 0
    except BaseException as exc:
        results.value["last_setup_or_runtime_error"] = f"{type(exc).__name__}: {exc}"
        results.save()
        write_reports(results)
        raise
    finally:
        from app.db.vector_store import VectorStore
        VectorStore._close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Evaluation failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
