# Evaluation v2 on Colab

Open `colab_eval.ipynb`, choose a T4 GPU, and use the updated
`My Drive/Agentic DocQA/agentic-docqa-v2.zip`, upload that ZIP, or select GitHub
with a repository-scoped Colab Secret `GITHUB_TOKEN`. Old project ZIPs are rejected.
The repo stays private; no Docker, tunnel or HTTP application server is needed.

The notebook checks the GPU, loads source, installs requirements, runs offline
checks, starts Ollama, pulls selected models, verifies GPU loading, runs a two-question
smoke test, then runs all 20 questions. Code/output cells in the committed notebook
have no saved benchmark outputs. New reports, partial saves and logs use
`My Drive/Agentic DocQA/evaluation-results/v2/`; smoke uses `v2/smoke/`.

Form fields:

- `ANSWER_MODEL`: `qwen2.5:7b` by default, with `qwen2.5:3b` available.
- `JUDGE_MODEL`: empty means the answer model; an alternative is pulled and GPU-checked.
- `RUN_PREFIX_ABLATION`: true by default; measures all six retrieval conditions.
- `EVAL_MODE`: full evaluation, or uniform rejudging of saved answers.
- `SAVED_RESULTS`: the source JSON for rejudging (pre-filled with the previous run).

A separate judge such as the requested `qwen2.5:14b` or `llama3.1:8b` is optional.
This change has not tested either model's T4 memory fit. The notebook loads one
model at a time and rejects substantial CPU offload during GPU preflight. Model
switching between answers and judgments can affect observed answer latency.
The report names both models and identifies self-judging (including identical
weight digests under different names).

## Commands

```bash
# No Ollama or downloads needed:
python eval/run_eval.py --validate-only
python -m unittest discover -s tests -v

# Ollama running, requirements installed and selected models already pulled:
python eval/run_eval.py --model qwen2.5:7b --judge-model qwen2.5:7b --limit 2 --out results/v2/smoke
python eval/run_eval.py --model qwen2.5:7b --judge-model qwen2.5:7b --prefix-ablation --out results/v2
python eval/run_eval.py --skip-e2e --prefix-ablation --out results/v2/retrieval-only
python eval/run_eval.py --no-prefix-ablation --out results/v2/no-ablation

# ALL saved answers, both judge passes, one current policy. No agent/retrieval calls:
python eval/rejudge.py /path/to/old.partial.json --out results/v2/rejudged-all
```

The notebook's **Rejudge saved answers** mode uses the last command and writes to
`v2/rejudged-all/`. It does not retain old successful judgments. Every eligible
saved answer, including all 20 in the previous run, is rejudged. The source file,
answers, actual evidence, direct retrieval observations, traces and answer latency
are preserved. Old unprefixed retrieval results cannot supply prefix-on ablation
numbers: run full evaluation to measure the embedding change.

`--only-failed` is an explicit compatibility opt-in that retains earlier successes;
its report is prominently labelled mixed-policy. It is never used by either v2
notebook mode. Original judgments/attempts are retained separately for audit even
when all current scores are replaced. Do not report a mixed-policy selective
repair as a uniform rejudging run.

For an already configured Colab runtime, the standalone
`scripts/repair_eval_judge.py` can patch the evaluation modules and rejudge all saved
answers by default. Upload the file and run `%run /content/repair_eval_judge.py`.
This helper changes evaluation code only; use the v2 project ZIP for the app's
embedding change and the full prefix ablation. Its code backups and live log are
saved beside the new reports. It does not install dependencies or pull models.

## Metrics and diagnostics

The five fictional Northstar policy documents support 20 questions: six single
facts, three multi-part, four paraphrases, three exact identifiers and four
unanswerable questions. Fixture validation checks source-specific reference
phrases against actual corpus text.

**Two-pass judging.** Every answer is scored twice at temperature zero, once with
the standard instruction and once with a reworded assessment instruction. The
second pass does not see the first pass. Each pass can repair invalid JSON/evidence
once, using validation feedback; retries are not votes. Both passes must validate
before a question gets an average correctness score. Otherwise it remains an
explicit error, never zero by default. Raw replies, attempts and per-pass scores
are in JSON. Two-pass disagreement means correctness differs by more than 1e-9 or
abstention flags differ. Faithfulness is the average of available claim-bearing
pass scores; the count of scored passes is recorded, and pure-refusal passes are N/A.
Two similar judgments do not establish correctness; same-model judging is biased.

**Evidence checking.** JSON-schema quote choices come only from actual captured
source lines/paragraphs (plus an empty string), and supported quotes still undergo
literal validation. An abstention that includes claims gets a separate answer-only
claim extraction; only an empty extraction constrains repair to `claims=[]`.
Mixed answers retain their factual assertions. Neither unsupported facts nor
missing evidence are erased to manufacture a score. Truncated evidence makes
faithfulness N/A with an explicit reason.

**Refusals.** The deterministic phrase detector identifies expressions such as
"do not contain", "not mentioned", "not specified", "no information", "not
explicitly stated", "cannot find", "isn't covered" and "doesn't say". Policy
negations such as "not required" and "not refunded" are not refusal phrases.
The report shows rule-based, judge-based and combined rates. Judge abstention is
true if either judge pass abstains; combined = rule OR judge. A partial refusal is
a rule refusal plus at least one matched required fact. Combined rates include
partial refusals; flagged IDs and rule-vs-judge disagreements remain visible.
These are heuristics and can misclassify caveats, quotations or unusual wording.

**Key-fact recall.** The fraction of `must_contain` phrases present after case,
whitespace, punctuation, currency (`$12`/`12 dollars`) and common unit normalization.
Unanswerable entries have N/A recall. It is a literal matching diagnostic, not
semantic truth: a negated fact or quotation may match and a paraphrase may not.
`judge_disagreement` flags recall=1 with mean judge correctness<0.5, or recall=0
with correctness>0.5. It never overwrites a judge score.

**Retrieval.** The real `search_child_chunks` tool compares dense, hybrid without
reranking and hybrid with a cross-encoder, each with prefixes off and on: six rows.
Separate collections re-embed the same children for each condition. Document and
query prefixes match within each condition. Chunk text, sparse inputs, candidate
count (10), score threshold, pre-dedup reranking cutoff and production parent
deduplication are held constant. Only returned snippets are scored. Precision@5
always divides by five. Source AND paired evidence phrase determine relevance.
Unanswerables are excluded. With 16 answerable questions, one hit/miss changes
full-set Hit@5 by 6.25 percentage points; partial runs show observed denominators.

**Agent evidence and failure labels.** Agent questions use fresh thread IDs and
empty history. Search events record actual query strings, source/chunk/parent IDs
and returned text. Fetched parents and observed previews also count as evidence
actually seen by the agent. `agent_retrieval_hit` uses all of that evidence, without
a top-five cutoff; it does not substitute direct-question benchmark searches.
Grading events capture the exact text passed to each grader, including its existing
truncation, so parallel branches are not matched by event order. A grader marking
full required-fact coverage insufficient, or no required fact sufficient, is flagged
as a `grader issue`; partial coverage is inconclusive. Otherwise unseen required
evidence gives `retrieval miss (agent queries)`, and a wrong/refused answer despite
seeing evidence gives `generation error`. Grader contradictions take precedence.
These labels are diagnostic hypotheses against the golden facts, not proof of cause.
Older traces lacking per-call grading text cannot retroactively diagnose that grader.

Reports include summary, type breakdown, all six retrieval rows, agent hit rate,
flagged question IDs, CRAG counts, per-question results and the worst three with
actual agent queries and returned IDs. Retry comparisons remain descriptive rather
than causal. Answer latency excludes judge/setup, but caches and model reloads matter.

## Prefix migration and isolation

Nomic's [official model card](https://huggingface.co/nomic-ai/nomic-embed-text-v1#task-instruction-prefixes)
requires `search_document: ` for indexed text and `search_query: ` for queries.
`TaskPrefixEmbeddings` wraps the existing dense embedder and adds them only at the
embedding boundary when `DENSE_MODEL` starts with `nomic-embed` and
`NOMIC_PREFIXES=true` (the default). Stored `page_content`, sparse embeddings,
reranker input, tool output and agent prompts are not prefixed.

Changing prefix configuration invalidates old vectors. Existing app users must
re-upload all documents; normal libraries are not erased automatically. Every fresh
eval process resets only its fingerprint-marked evaluation work directory and
rebuilds the isolated `eval_user`, including both dense-vector conditions. It never
uses or deletes normal app users, libraries, indexes or Drive source results.
The answering agent always uses the prefix-on collection, even if ablation is disabled.

## Resume and provenance

A stable `<run-id>.partial.json` is atomically replaced after each question.
Timestamped JSON and Markdown are regenerated too. Compatible completed questions
are skipped on resume; failed questions are retried. Rejudging checkpoints each new
judgment, and resumes completed new judgments under the same policy/source identity.
One process per output/run ID is supported. A disconnected runtime rebuilds its
small evaluation corpus; in-progress graph execution is not checkpointed.

Identity includes app/eval code, fixtures, model names/digests, prefix configuration,
ablation setting, both judge variants, metric/judge policies, package/runtime/GPU
information and context budget. `--limit` is not identity, allowing a longer run to
continue compatible questions (smoke still has a separate directory). A readable
`--run-id` cannot reuse incompatible results. Rejudging records the original file
hash and pipeline fingerprint separately from the current judge/runtime identity.
Use a fresh subprocess so app configuration is applied before cached imports.

No new dependencies or version pins were added. Offline tests use fake model
responses and embeddings, real LangGraph, and embedded in-memory Qdrant when project
dependencies are installed. Saved-answer regression fixtures contain exact synthetic
answers from the previous run, without benchmark scores. No v2 benchmark has been
run or estimated by this change; full inference and optional larger-judge memory
fit must be checked on Colab.
