# QASPER evaluation

This adapter evaluates the unchanged production `DocumentManager`, retrieval tools
and `UserRAG.ask` on scientific papers. All benchmark-specific implementation,
statistics, formatting, scientific-judge adapter and offline tests live here.
`tests/test_qasper.py` only exposes these tests to the main test discovery command.
The shared Colab notebook and source bundle are the integration points. No app
routing, answering prompt, graph transition or production ingestion is changed.

## First Colab run

Use the updated `colab_eval.ipynb` and `agentic-docqa-qasper.zip` in the shared
`My Drive/Agentic DocQA` folder. The notebook uses the ZIP by default; uploading
it or loading private GitHub main with a Colab Secret also works.

Choose a **T4 GPU** and these exact fields:

| Field | First-run setting |
|---|---|
| BENCHMARK | QASPER |
| ANSWER_MODEL | qwen2.5:7b |
| JUDGE_MODEL | empty (defaults to answer model) |
| RUN_PREFIX_ABLATION | true |
| QASPER_N | 250 |
| QASPER_MODE | per_paper |
| QASPER_SPLIT | test |
| QASPER_SEED | 42 |
| QASPER_FORMAT_MODEL | empty (defaults to answer model) |

Run the common setup, then the QASPER section. Synthetic execution/report cells
are skipped when BENCHMARK=QASPER. The section installs the evaluation-only
`datasets` dependency, caches the data/sample, runs three smoke questions, runs
the full sample with `--resume`, then displays Markdown. Smoke outputs are separate.
Use `pooled` for a later, separately labelled experiment.

Full results go to
`/content/drive/MyDrive/Agentic DocQA/evaluation-results/qasper/<mode>/`.
The raw release, canonical papers, HF cache and sample mirror go to
`/content/drive/MyDrive/Agentic DocQA/cache/qasper/`.
Download/model caches use the notebook's existing shared Drive cache settings.
A task-supplied planning range is 1.5–3 hours for 250 questions on a T4 with
qwen2.5:7b; this is **unmeasured, not a runtime guarantee**. Indexing many papers,
multiple agent calls, formatting, two judge passes and prefix ablations can take
longer. No benchmark scores have been estimated or bundled.

## CLI

Install the app's requirements, plus `pip install -r eval/qasper/requirements.txt`.
Start Ollama and pull the answer/formatter/judge models plus `nomic-embed-text`.
Run in a fresh process so configuration is applied before cached app imports.

```bash
# Only download/cache data and create the sample. No Ollama or GPU required:
python -m eval.qasper.run --prepare-only --split test --n 250 --seed 42 \
  --cache-dir /path/to/cache/qasper --out results/qasper/per_paper

# Smoke and full use the SAME 250-question sample; do not use --n 3 for smoke.
python -m eval.qasper.run --split test --n 250 --seed 42 --mode per_paper \
  --answer-model qwen2.5:7b --prefix-ablation --limit 3 \
  --out results/qasper/per_paper/smoke --resume
python -m eval.qasper.run --split test --n 250 --seed 42 --mode per_paper \
  --answer-model qwen2.5:7b --prefix-ablation \
  --out results/qasper/per_paper --resume

# Separately labelled pooled experiment:
python -m eval.qasper.run --mode pooled --n 250 --seed 42 --prefix-ablation --resume
```

Additional flags include `--judge-model`, `--format-model`, `--cache-dir`,
`--sample-dir`, `--work-root`, `--judge-context-chars` (24000) and
`--report-every` (10). Without `--prefix-ablation`, direct comparisons use prefixes
on only. The answering agent always uses the existing Nomic-prefix-on setting.
A matching existing run requires `--resume`; it is never silently overwritten.

## Data and reproducibility

The first source is `datasets.load_dataset("allenai/qasper", split=...)`.
The official dataset loader records these v0.3 release URLs:

- Test: https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz
- Train/validation: https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz

If HF loading fails (including newer `datasets` versions rejecting legacy builder
scripts), the adapter downloads the appropriate official archive, reads its single
JSON member without extracting paths, and records its URL/hash and the HF failure.
`validation` uses `qasper-dev-v0.3.json`. Canonical cache files have checksum checks.
The [dataset card](https://huggingface.co/datasets/allenai/qasper) identifies the data
as CC BY 4.0; retain its provenance when sharing results.

Papers become Markdown with title, abstract and section headings. Normalized
whitespace is preserved for paragraph text; scoring identities use SQuAD-normalized
text. Repeated identical normalized paragraphs share a stable paragraph ID.
Unmapped gold evidence is retained as an unmatched ID, not dropped. Annotation
fields including highlighted evidence are retained; highlighted spans are not
substituted for gold paragraph evidence. Figure/table metadata is not ingested.
Evidence containing `FLOAT SELECTED` is counted separately and excluded from text
matching, as in the official text-only evaluator. Float-only questions stay in
answer scoring and appear in a dedicated failure/scope category.

Sampling uses majority annotation answer type, fixed type-order tie breaking, and
constrained proportional allocation. Every available type gets at least 20; a type
with fewer than 20 contributes all its questions. Infeasible n values are rejected
with the required minimum; `--limit` is the smoke control. Floors necessarily
change natural proportions when a type would otherwise receive fewer than 20.
Sample means are unweighted, so they are not estimates for the full test distribution.

`eval/qasper/sample_<split>_<n>_<seed>.json` contains only question/paper IDs and
sampling provenance and is ignored by git. It is mirrored into the Drive cache
and output folder. An incompatible existing manifest is rejected rather than
silently resampled. Run identity includes the data/sample/code hashes, split,
mode, model names/weight digests, prefix condition, actual app settings, package/GPU
information and evaluation policies. `--limit` and output path are not identity.

In per-paper mode, a dedicated `qasper_<paper_id>` eval user/collection indexes
that paper once and is reused for all its pending questions. Unsafe/long IDs are
sanitized or hashed. Collections and user directories are released after that paper.
Pooled mode indexes every sampled paper once into one eval user, including for
smoke so the retrieval corpus matches the full experiment. Each question still
has a fresh thread, empty history and its exact original question text.
Direct-chat routing is recorded, never repaired by adding a document-QA prefix.

The embedded DB and users live in a fingerprint-marked local work directory.
Cleanup validates ownership, closes Qdrant and removes only that work directory.
No normal app data, source cache or Drive result is deleted. Resuming rebuilds
needed indexes and skips compatible completed questions. Failed questions retry.
JSON saves atomically after every attempted question, including explicit errors;
Markdown/CIs refresh every ten and at completion/error. In-progress agent execution
is not checkpointed. Run one process per output/run ID.

## Metric definitions and differences

The [official evaluator, pinned at e996b6c7](https://github.com/allenai/qasper-led-baseline/blob/e996b6c7b1b5f95d9308a74e3586416c6e780df1/scripts/evaluator.py)
is the reference for normalization, token-F1, paragraph-set F1, max-over-annotators
and empty-gold behavior. F1 is stored as 0–1 and shown as 0–100.

- **Answer F1:** lowercase, remove ASCII punctuation, remove a/an/the, collapse
  whitespace; multiset token overlap F1. Take the maximum over annotators.
  Empty-vs-empty answer F1 is 0 in the release. **The official script does not
  impose categorical exact match:** `Yes indeed` vs `Yes` receives 2/3 token F1.
  This adapter matches that behavior, while recording categorical exact match
  separately. Space-joining extracted spans instead of comma-space joining gives
  the same normalized tokens. Answer type is none, then non-null yes/no, then
  extracted spans, else abstractive as requested; the release prioritizes spans
  and free-form text before yes/no on contradictory mixed-field annotations.
- **Breakdown:** like the evaluator, type for a scored prediction comes from its
  best-matching annotation, with stable first-annotation ties. Short and full
  predictions can have different type counts. Majority strata are also reported.
- **Evidence F1 (agent-level):** predict all gold-paper paragraph identities found
  in actual search snippets and parent fetches, not just gold-labelled paragraphs.
  Matching uses normalized substring containment; fragments across actual snippets
  count if the union of covered token positions is **strictly greater than 80%**.
  Overlapping snippets do not double-count coverage. Wrong-paper source names never
  match in pooled mode. Precision/recall F1 on unique paragraph-ID sets is maximized
  independently over annotation evidence sets. Both empty = 1; only one empty = 0.
  Our mapping/fragment rule is a text-retrieval adaptation, not the official
  evidence-selection model. The official core divides by list lengths, so duplicate
  evidence entries can differ; we use deduplicated sets as requested.
- **Short vs full:** strip a trailing Sources section, then a separate temperature-zero
  LLM sees only question/answer and returns JSON. Added content words cause fallback
  to the stripped full answer with a flag. Yes/No/Unanswerable are allowed special
  outputs. This guard checks words, not entailment or negation scope. Both primary
  short-answer F1 and secondary stripped-full F1 are reported, alongside fallback
  rates and rule-vs-formatter refusal disagreements.
- **Refusals:** reuse the existing deterministic detector; the formatter cross-check
  is exactly normalized `Unanswerable`. None/answerable groups for refusal rates use
  majority annotation type. Conflicting annotators can therefore produce a high
  max-reference F1 and a majority-based false-refusal flag simultaneously.
- **Judge:** reuse the existing two-pass literal-evidence judge with an evaluation-only
  system-message adapter for scientific claims. It sees the best short-F1 reference,
  full agent answer and actual tool evidence. Raw passes, reference selection and
  scope checks remain available; judge failure never creates a zero judge score.
- **Retrieval:** real direct search calls compare dense/hybrid/reranked, with matching
  independently indexed prefix conditions. Hit@5/MRR@5 require an individual top-five
  chunk to match gold text evidence (>80% fallback allowed). Agent evidence F1 can
  combine chunks. No-text-evidence questions are excluded from retrieval denominators.
- **Failures:** direct-chat route, no seen gold evidence, imperfect F1 despite seen
  evidence, answerable majority with refusal, float-only evidence and execution errors
  are non-exclusive diagnostic categories. They are not proofs of cause.
- **Uncertainty:** Wilson intervals for binary rates; 2,000 deterministic question-level
  percentile bootstrap samples for continuous means and paired hit differences.
  Exact two-sided McNemar tests use matched questions and say when differences are
  not significant. Tests are exploratory/unadjusted; repeated questions per paper
  are not cluster-adjusted and can violate independence assumptions. Unattempted
  questions remain unscored. Attempted missing predictions get official F1=0;
  model-only metrics remain unavailable.

The [paper](https://arxiv.org/pdf/2105.03011), Table 2, reports 33.63 Answer F1 for
LED-base with evidence scaffolding and a human lower-bound estimate of 60.92.
Our sampled retrieval agent plus LLM formatter differs; this is not a like-for-like
comparison. We do not treat those reference points as our results.

## Offline tests

```bash
python -m unittest discover -s tests -v
python -m unittest eval.qasper.tests -v
```

Tests cover known SQuAD examples, repeated tokens, annotation typing, HF row/column
forms, max references, exact/split/overlapping evidence, float-only and unmapped
paragraphs, empty gold, formatter word rejection, deterministic stratification,
cache/fallback safety, Wilson/McNemar, actual isolated/pooled ingestion with fake
embeddings, exact agent question/thread calls, durable resume and owned cleanup.

For parity tests, download the pinned public evaluator into a temporary location:

```bash
curl -L https://raw.githubusercontent.com/allenai/qasper-led-baseline/e996b6c7b1b5f95d9308a74e3586416c6e780df1/scripts/evaluator.py \
  -o /tmp/qasper-official-evaluator.py
QASPER_OFFICIAL_EVALUATOR=/tmp/qasper-official-evaluator.py python -m unittest eval.qasper.tests -v
```

It is not vendored or fetched by ordinary unit tests. If absent, that one optional
parity test skips; hand-computed scoring tests still run. Neither the benchmark
corpus, generated manifests, results nor model weights are committed.
