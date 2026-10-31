"""Reports from recorded observations; rule signals never replace judge scores."""
from eval.metrics import agent_retrieval_metrics, answer_metrics, classify_failure, mean

MODES = ("dense", "hybrid", "hybrid_rerank")
TYPES = ("single_fact", "multi_part", "paraphrase", "exact_identifier", "unanswerable")


def show(value):
    return "N/A" if value is None else f"{value:.3f}"


def cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def ids(rows):
    return ", ".join(row["id"] for row in rows) or "none observed"


def diagnosis(row):
    e2e = row.get("e2e", {})
    return e2e.get("answer_metrics") or answer_metrics(row, e2e.get("answer", ""), e2e.get("judgment"))


def agent_metrics(row):
    e2e = row.get("e2e", {})
    return e2e.get("agent_retrieval") or agent_retrieval_metrics(e2e.get("events", []), row)


def retrieval_row(row, mode, prefixes, metadata):
    values = row.get("retrieval", {})
    key = f"prefix_{'on' if prefixes else 'off'}_{mode}"
    if key in values:
        return values[key]
    # Reports rejudging v1 answers retain the original, unprefixed direct searches.
    legacy_prefixes = metadata.get("settings", {}).get("NOMIC_PREFIXES", False)
    return values.get(mode) if prefixes == legacy_prefixes else None


def render_report(value):
    metadata = value["metadata"]
    results = list(value["results"].values())
    completed = [row for row in results if row["status"] == "complete"]
    answered = [row for row in results if "answer" in row.get("e2e", {})]
    judged = [row for row in completed if row.get("e2e", {}).get("judgment")]
    policy = metadata.get("judge_policy", "legacy / not recorded")
    retained = metadata.get("retained_previous_judgments", 0)
    lines = ["# Agentic DocQA evaluation v2", "", f"Run: `{metadata['run_id']}`",
             f"Answer model: `{metadata['model']}`; judge model: `{metadata['judge_model']}`.",
             f"Completed: {len(completed)}/{metadata['expected_questions']}; errors: {sum(row['status'] == 'error' for row in results)}; pending: {sum(row['status'] not in {'complete', 'error'} for row in results)}.",
             f"Judge policy: `{policy}`. Retained earlier judgments: {retained}.", ""]
    lines.append("Mixed-policy report (explicit selective-repair opt-in)." if retained else
                 "One policy is used for all new judgments; v2 requires both separately prompted passes before scoring a question.")
    if metadata.get("eval_version", 1) < 2:
        lines.append("Legacy observations: two-pass consistency and prefix-on ablation may be unavailable. Missing observations are N/A.")
    if metadata.get("rejudged_from"):
        lines += [f"Saved answers/evidence rejudged from `{metadata['pipeline_fingerprint']}`; the agent and retrieval were not rerun."]
    if metadata.get("skip_e2e"):
        lines.append("Retrieval-only run: agent answers and judging were skipped.")
    if value.get("last_setup_or_runtime_error"):
        lines.append(f"Setup/runtime error: {cell(value['last_setup_or_runtime_error'])}")
    lines += ["", "n=20 synthetic questions; this is a diagnostic, not an estimate of production quality.",
              "There are 16 answerable questions. One hit/miss changes full-set Hit@5 by 6.25 percentage points; partial runs use their observed denominator.",
              "Judge scores are model estimates. " + ("The answering model also judges itself." if metadata["self_judge"] else "A separately configured model performs judging."),
              "", "## Summary", "", "| Metric | Value | Scored questions |", "|---|---:|---:|"]

    def metric(label, scores):
        lines.append(f"| {label} | {show(mean(scores))} | {sum(score is not None for score in scores)} |")

    for key in ("correctness", "faithfulness"):
        metric(key, [row["e2e"]["judgment"].get(key) for row in judged])
    metric("Judge two-pass disagreement rate", [float(row["e2e"]["judgment"]["judge_run_disagreement"])
           if "judge_run_disagreement" in row["e2e"]["judgment"] else None for row in judged])
    metric("key_fact_recall", [diagnosis(row)["key_fact_recall"] for row in answered])
    metric("Key facts vs judge disagreement rate", [float(diagnosis(row)["judge_disagreement"])
           if diagnosis(row)["judge_disagreement"] is not None else None for row in judged])
    metric("Rule vs judge refusal disagreement rate", [float(diagnosis(row)["refusal_disagreement"])
           if diagnosis(row)["refusal_disagreement"] is not None else None for row in answered])
    for answerable, label in ((False, "Correct refusal on unanswerable"), (True, "Wrong refusal on answerable")):
        group = [row for row in answered if row["answerable"] == answerable]
        for key, kind in (("rule_refusal", "rule"), ("judge_abstained", "judge"), ("combined_abstained", "combined")):
            metric(f"{label} ({kind})", [float(diagnosis(row)[key]) if diagnosis(row)[key] is not None else None for row in group])
    metric("Partial refusal rate (answerable)", [float(diagnosis(row)["partial_refusal"]) for row in answered if row["answerable"]])
    metric("Mean answer latency (seconds; excludes judge/setup)", [row["e2e"].get("latency_seconds") for row in answered])
    metric("agent_retrieval_hit (answerable)", [agent_metrics(row)["agent_retrieval_hit"] for row in answered if row["answerable"]])
    lines += ["", "Correctness is the arithmetic mean of two temperature-zero, separately prompted judgments; retries repair a pass and do not add votes.",
              "Two-pass disagreement means correctness differs (tolerance 1e-9) or abstention flags differ. Judge abstention is true if either pass abstains.",
              "Combined abstention = rule_refusal OR judge_abstained. These rates include partial refusals; their IDs are listed separately.",
              "Key-fact recall is the fraction of normalized must_contain phrases found, with currency/unit normalization. It is a match signal, not semantic correctness; negated or quoted facts can match. Judge scores are never replaced by this signal.",
              "Faithfulness is averaged over available claim-bearing passes; a pure-refusal pass has N/A. JSON records the scored-pass count.",
              "", "## Breakdown by question type", "", "| Type | Answers | Judged | Correctness | Faithfulness | Key fact recall | Refusal rule | Refusal judge | Refusal combined | Agent hit |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for kind in TYPES:
        group = [row for row in answered if row["type"] == kind]
        scored = [row for row in judged if row["type"] == kind]
        scores = [mean([row["e2e"]["judgment"].get(key) for row in scored]) for key in ("correctness", "faithfulness")]
        scores += [mean([diagnosis(row)[key] for row in group]) for key in ("key_fact_recall",)]
        scores += [mean([float(diagnosis(row)[key]) if diagnosis(row)[key] is not None else None for row in group])
                   for key in ("rule_refusal", "judge_abstained", "combined_abstained")]
        scores += [mean([agent_metrics(row)["agent_retrieval_hit"] for row in group])]
        lines.append(f"| {kind} | {len(group)} | {len(scored)} | " + " | ".join(show(score) for score in scores) + " |")
    lines += ["", "## Retrieval comparison: prefix ablation", "",
              "Each condition uses independently indexed document vectors and matching query prefixes. Chunk text, sparse model, corpus, candidate count and production deduplication are identical. Prefixes apply only at the dense embedding boundary.",
              "Unanswerable questions are excluded. Relevance requires the expected source AND its paired evidence phrase; precision@5 always divides by five.",
              "", "| Prefixes | Mode | Questions | Hit@5 | MRR@5 | Precision@5 | Evidence coverage@5 |", "|---|---|---:|---:|---:|---:|---:|"]
    for prefixes in (False, True):
        for mode in MODES:
            rows = [retrieval_row(row, mode, prefixes, metadata) for row in results if row["answerable"]]
            metrics = [row["metrics"] for row in rows if row and "metrics" in row]
            scores = [show(mean([row[key] for row in metrics])) for key in ("hit_at_5", "mrr_at_5", "precision_at_5", "evidence_coverage_at_5")]
            lines.append(f"| {'on' if prefixes else 'off'} | {mode} | {len(metrics)} | " + " | ".join(scores) + " |")
    if metadata.get("prefix_ablation") is False:
        lines.append("\nPrefix-off ablation was disabled for this run; its rows are N/A.")
    lines += ["", "Agent retrieval is measured separately from these direct-question searches, across actual search snippets, fetched parents and observed previews, without a top-five cutoff.",
              "Grader checks use the exact (possibly truncated) text supplied to each grading call, so parallel research branches are not paired by event order. Insufficient with full required-fact coverage, or sufficient with no required fact, is flagged as a grader issue; partial coverage is inconclusive. These are diagnostic hypotheses against the golden facts.",
              "", "## Flagged questions", "",
              f"- Key facts vs judge (`judge_disagreement`): {ids([row for row in judged if diagnosis(row)['judge_disagreement']])}.",
              f"- Two-pass judge disagreements: {ids([row for row in judged if row['e2e']['judgment'].get('judge_run_disagreement')])}.",
              f"- Rule vs judge refusal disagreements: {ids([row for row in answered if diagnosis(row)['refusal_disagreement']])}.",
              f"- Partial refusals: {ids([row for row in answered if diagnosis(row)['partial_refusal']])}.",
              "", "## CRAG observations", ""]
    for key in ("grade_events", "llm_insufficient_events", "insufficient_events", "grader_errors", "retry_events", "fallback_events"):
        lines.append(f"- {key}: {sum(row.get('e2e', {}).get('crag', {}).get(key, 0) for row in results)}")
    lines += ["", "Retry counts record scheduled retries before counter resets. Comparisons are descriptive, not a causal retry ablation.",
              "", "| Observed group | Questions | Mean correctness |", "|---|---:|---:|"]
    for retried in (False, True):
        group = [row for row in judged if bool(row["e2e"].get("crag", {}).get("retry_events", 0)) == retried]
        lines.append(f"| {'With retry' if retried else 'Without retry'} | {len(group)} | {show(mean([row['e2e']['judgment']['correctness'] for row in group]))} |")
    lines += ["", "## Per-question results", "", "| ID | Type | Status | Agent hit | Correctness | Faithfulness | Key facts | Rule refusal | Judge refusal | Combined | Partial | Seconds |",
              "|---|---|---|---:|---:|---:|---:|---|---|---|---|---:|"]
    for row in results:
        e2e = row.get("e2e", {})
        judgment = e2e.get("judgment") or {}
        diag = diagnosis(row)
        hit = agent_metrics(row)["agent_retrieval_hit"] if "answer" in e2e else None
        lines.append(f"| {row['id']} | {row['type']} | {row['status']} | {show(hit)} | {show(judgment.get('correctness'))} | {show(judgment.get('faithfulness'))} | {show(diag['key_fact_recall'])} | {diag['rule_refusal']} | {diag['judge_abstained']} | {diag['combined_abstained']} | {diag['partial_refusal']} | {show(e2e.get('latency_seconds'))} |")
    candidates = [row for row in results if row["status"] == "error" or classify_failure(row, row) != "none"]
    candidates.sort(key=lambda row: (-1 if row["status"] == "error" else (row.get("e2e", {}).get("judgment") or {}).get("correctness", 1), row["id"]))
    lines += ["", "## Three worst observed questions", "", "Failure labels use the agent's evidence and per-call grading trace, not the direct-question retrieval benchmark."]
    if not candidates:
        lines.append("No scored failures are available; no worst questions have been invented.")
    for row in candidates[:3]:
        agent = agent_metrics(row)
        returned = [{"query": search["query"], "chunks": [{"source": hit.get("source"), "parent_id": hit.get("parent_id"), "chunk_id": hit.get("chunk_id")} for hit in search["chunks"]]}
                    for search in agent["searches"]]
        lines += ["", f"### {row['id']}: {classify_failure(row, row)}", cell(row["question"]), "",
                  f"Reference: {cell(row['reference_answer'])}", f"Answer: {cell(row.get('e2e', {}).get('answer', '(unavailable)'))}",
                  f"Actual agent queries: {cell(agent['queries'])}", f"Returned IDs: {cell(returned)}",
                  f"Fetched parent IDs: {cell([hit['parent_id'] for hit in agent['parents']])}", f"Errors: {cell(row.get('errors', []))}"]
    lines += ["", "## Reproducibility", "", f"Fingerprint: `{metadata['fingerprint']}`",
              "JSON contains both judge passes, retries, scope checks, raw replies, settings, model digests, package/GPU versions, actual tool evidence and grading evidence. Answer latency excludes judging/setup.", ""]
    return "\n".join(lines)
