"""Readable reports built only from recorded evaluation observations."""
from eval.metrics import mean

MODES = ("dense", "hybrid", "hybrid_rerank")


def show(value):
    return "N/A" if value is None else f"{value:.3f}"


def cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_report(value):
    metadata = value["metadata"]
    results = list(value["results"].values())
    completed = [r for r in results if r["status"] == "complete"]
    judged = [r for r in completed if r.get("e2e", {}).get("judgment")]
    lines = ["# Agentic DocQA evaluation", "", f"Run: `{metadata['run_id']}`",
             f"Model: `{metadata['model']}`; judge: `{metadata['judge_model']}`.",
             f"Completed: {len(completed)}/{metadata['expected_questions']}; errors: {len(results) - len(completed)}.", "",
             "This is a 20-question synthetic diagnostic, not an estimate of production quality.",
             "Judge scores are model estimates. " + ("The answering model also judges itself." if metadata["self_judge"] else "A separately configured model performs judging."),
             "", "## Summary", "", "| Metric | Value | Scored questions |", "|---|---:|---:|"]
    if value.get("last_setup_or_runtime_error"):
        lines[6:6] = [f"Setup/runtime error: {cell(value['last_setup_or_runtime_error'])}", ""]
    if metadata.get("skip_e2e"):
        lines[6:6] = ["Retrieval-only run: end-to-end generation and judging were skipped.", ""]
    for key in ("correctness", "faithfulness"):
        scores = [r["e2e"]["judgment"].get(key) for r in judged]
        lines.append(f"| {key} | {show(mean(scores))} | {sum(v is not None for v in scores)} |")
    for answerable, label in ((False, "Correct refusal on unanswerable"), (True, "Wrong refusal on answerable")):
        group = [r for r in judged if r["answerable"] == answerable]
        lines.append(f"| {label} | {show(mean([float(r['e2e']['judgment']['abstained']) for r in group]))} | {len(group)} |")
    latencies = [r["e2e"]["latency_seconds"] for r in completed if r.get("e2e") and "latency_seconds" in r["e2e"]]
    lines.append(f"| Mean answer latency (seconds; excludes judge/setup) | {show(mean(latencies))} | {len(latencies)} |")
    lines += ["", "## Retrieval comparison", "", "Unanswerable questions are excluded. Relevance requires a correct source AND its paired evidence phrase.",
              "Precision@5 always divides by 5, including when parent deduplication returns fewer hits.",
              "All modes use the same candidate count, threshold, and production parent deduplication; reranking keeps the production pre-dedup cutoff.",
              "", "| Mode | Questions | Hit@5 | MRR@5 | Precision@5 | Evidence coverage@5 |", "|---|---:|---:|---:|---:|---:|"]
    for mode in MODES:
        rows = [r["retrieval"][mode]["metrics"] for r in results if r["answerable"] and mode in r.get("retrieval", {}) and "metrics" in r["retrieval"][mode]]
        scores = [show(mean([row[key] for row in rows])) for key in ("hit_at_5", "mrr_at_5", "precision_at_5", "evidence_coverage_at_5")]
        lines.append(f"| {mode} | {len(rows)} | " + " | ".join(scores) + " |")
    lines += ["", "## CRAG observations", ""]
    for key in ("grade_events", "llm_insufficient_events", "insufficient_events", "grader_errors", "retry_events", "fallback_events"):
        lines.append(f"- {key}: {sum(r.get('e2e', {}).get('crag', {}).get(key, 0) for r in results)}")
    lines += ["", "`insufficient_events` also includes empty retrieval and grading failures; `llm_insufficient_events` counts explicit LLM insufficient judgments only.",
              "Retry events count scheduled retries, including those followed by a successful grade that resets the agent's counter.",
              "", "| Observed group | Questions | Mean correctness |", "|---|---:|---:|"]
    for retried in (False, True):
        group = [r for r in judged if bool(r["e2e"]["crag"]["retry_events"]) == retried]
        lines.append(f"| {'With retry' if retried else 'Without retry'} | {len(group)} | {show(mean([r['e2e']['judgment']['correctness'] for r in group]))} |")
    lines += ["", "These groups are descriptive. Harder questions may trigger retries; this is not a causal retry ablation.",
              "", "## Per-question results", "", "| ID | Type | Status | Hybrid+rerank hit@5 | Correctness | Faithfulness | Seconds |", "|---|---|---|---:|---:|---:|---:|"]
    for r in results:
        e2e = r.get("e2e", {})
        judgment = e2e.get("judgment") or {}
        hit = r.get("retrieval", {}).get("hybrid_rerank", {}).get("metrics", {}).get("hit_at_5")
        lines.append(f"| {r['id']} | {r['type']} | {r['status']} | {show(hit)} | {show(judgment.get('correctness'))} | {show(judgment.get('faithfulness'))} | {show(e2e.get('latency_seconds'))} |")
    candidates = [r for r in results if r["status"] == "error" or (r.get("e2e", {}).get("judgment") or {}).get("correctness", 1) < 1 or r.get("retrieval", {}).get("hybrid_rerank", {}).get("metrics", {}).get("hit_at_5") == 0]
    candidates.sort(key=lambda r: (-1 if r["status"] == "error" else (r.get("e2e", {}).get("judgment") or {}).get("correctness", 1), r["id"]))
    lines += ["", "## Three worst observed questions", "", "Failure labels are diagnostic hypotheses, not proof of cause."]
    if not candidates:
        lines.append("No scored failures are available; no worst questions have been invented.")
    for r in candidates[:3]:
        label = r.get("failure_type", "retrieval miss")
        lines += ["", f"### {r['id']}: {label}", cell(r["question"]), "", f"Reference: {cell(r['reference_answer'])}",
                  f"Answer: {cell(r.get('e2e', {}).get('answer', '(end-to-end skipped or failed)'))}",
                  f"Errors: {cell(r.get('errors', []))}"]
    lines += ["", "## Reproducibility", "", f"Fingerprint: `{metadata['fingerprint']}`",
              "Full settings, package versions, GPU information, raw judge replies, tool evidence and CRAG events are in the JSON report.",
              "Latency includes the actual answer pipeline, with model warm-up performed by the notebook. Cold caches and runtime load can still affect timing.", ""]
    return "\n".join(lines)
