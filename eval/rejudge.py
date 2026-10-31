#!/usr/bin/env python3
"""Rejudge saved answers/evidence, without indexing documents or running the agent."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.judge import JUDGE_POLICY, judge_answer, judge_failure, make_judge
from eval.storage import Results, digest
from eval.run_eval import failure_type, ollama_info, runtime_info, write_reports


def eligible(record):
    e2e = record.get("e2e", {})
    return isinstance(e2e.get("answer"), str) and bool(e2e["answer"].strip()) and isinstance(e2e.get("evidence"), str)


def judge_only_errors(record):
    errors = record.get("errors", [])
    return bool(errors) and all(error.startswith("Judge failed") for error in errors)


def rejudge_records(source, results, judge, context_limit, only_failed=False):
    """Checkpoint each new judgment; never mutate the source result object."""
    targets = [key for key, record in source["results"].items()
               if eligible(record) and (not only_failed or judge_only_errors(record))]
    if not results.value["results"]:
        results.value["results"] = copy.deepcopy(source["results"])
        for key in targets:
            record = results.value["results"][key]
            e2e = record["e2e"]
            e2e["previous_judgment"] = e2e.get("judgment")
            e2e["previous_judge_attempts"] = e2e.get("judge_attempts", [])
            e2e["judgment"] = None
            record["status"] = "pending_judge"
        results.save()
    for key in targets:
        current = results.value["results"][key]
        if current.get("rejudge_fingerprint") == results.value["metadata"]["fingerprint"] and not current.get("rejudge_error"):
            continue
        original = source["results"][key]
        record = copy.deepcopy(current)
        e2e = record["e2e"]
        print(f"Rejudging {key} from its saved answer and actual retrieved evidence...", flush=True)
        judgment, attempts, latency = judge_answer(judge, original, original["e2e"]["answer"],
                                                   original["e2e"]["evidence"], context_limit)
        e2e.update(judgment=judgment, judge_attempts=attempts, judge_latency_seconds=latency,
                   judge_policy=JUDGE_POLICY)
        record["errors"] = [error for error in original.get("errors", []) if not error.startswith("Judge failed")]
        record["rejudge_fingerprint"] = results.value["metadata"]["fingerprint"]
        record["rejudge_error"] = judgment is None
        if judgment is None:
            record["errors"].append(judge_failure(attempts))
            print(f"{key}: {record['errors'][-1]}", flush=True)
        record["status"] = "error" if record["errors"] else "complete"
        record["failure_type"] = failure_type(record, record)
        results.put(record)
        write_reports(results)
    write_reports(results)
    return 1 if any(record["status"] != "complete" for record in results.value["results"].values()) else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original .partial.json or full JSON report")
    parser.add_argument("--out", type=Path, help="Default: a rejudged/ folder beside the source")
    parser.add_argument("--judge-model", help="Default: the judge model from the saved run")
    parser.add_argument("--only-failed", action="store_true", help="Faster repair; explicitly retain previous successful judgments")
    parser.add_argument("--judge-context-chars", type=int, help="Default: the original run's evidence budget")
    args = parser.parse_args(argv)
    raw = args.source.read_bytes()
    source = json.loads(raw)
    if not isinstance(source.get("metadata"), dict) or not isinstance(source.get("results"), dict) or not source["results"]:
        parser.error("Source must contain saved evaluation metadata and nonempty results")
    previous = source["metadata"]
    model = args.judge_model or previous["judge_model"]
    context_limit = args.judge_context_chars if args.judge_context_chars is not None else previous.get("judge_context_chars", 24000)
    if context_limit < 1000:
        parser.error("--judge-context-chars must be at least 1000")
    if not any(eligible(record) and (not args.only_failed or judge_only_errors(record)) for record in source["results"].values()):
        parser.error("No eligible saved answers/evidence match this repair mode")
    out = args.out or args.source.parent / "rejudged"
    if out.resolve() == args.source.parent.resolve():
        parser.error("Use a separate output folder to preserve the original run")
    base_url = os.environ.get("LLM_BASE_URL", previous.get("settings", {}).get("LLM_BASE_URL", "http://localhost:11434/v1"))
    host, separator, _ = base_url.rstrip("/").rpartition("/v1")
    if separator:
        os.environ["OLLAMA_HOST"] = host
    runtime = runtime_info()
    model_info = ollama_info([model])
    answer_digest = previous.get("ollama", {}).get("model_digests", {}).get(previous["model"])
    same_weights = answer_digest not in {None, "not pulled"} and answer_digest == model_info["model_digests"].get(model)
    code_hash = digest({name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                        for name in ("eval/judge.py", "eval/metrics.py", "eval/rejudge.py")})
    identity = {"source_sha256": hashlib.sha256(raw).hexdigest(), "pipeline_fingerprint": previous["fingerprint"],
                "judge_policy": JUDGE_POLICY, "judge_code_hash": code_hash, "judge_model": model,
                "judge_context_chars": context_limit, "only_failed": args.only_failed,
                "packages": runtime["packages"], "python_version": sys.version.split()[0], "ollama": model_info}
    fingerprint = digest(identity)
    run_id = "rejudge-" + fingerprint[:16]
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    retained = sum(bool(record.get("e2e", {}).get("judgment")) and not judge_only_errors(record)
                   for record in source["results"].values()) if args.only_failed else 0
    metadata = {**copy.deepcopy(previous), "fingerprint": fingerprint, "run_id": run_id,
                "report_basename": f"{now}-{run_id}", "judge_model": model, "judge_policy": JUDGE_POLICY,
                "self_judge": model == previous["model"] or same_weights, "rejudged_from": str(args.source),
                "judge_context_chars": context_limit,
                "pipeline_fingerprint": previous["fingerprint"], "retained_previous_judgments": retained,
                "rejudge_identity": identity, "rejudge_runtime": runtime}
    results = Results(out, run_id, metadata)
    judge = make_judge(model, base_url, os.environ.get("LLM_API_KEY", "ollama"))
    return rejudge_records(source, results, judge, context_limit, args.only_failed)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Rejudging failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
