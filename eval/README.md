# Evaluation on Colab

Open `colab_eval.ipynb` in Colab and select a T4 GPU. The notebook's default source
is `My Drive/Agentic DocQA/agentic-docqa.zip`; this must be the updated project ZIP
that includes `eval/`. You can instead upload a ZIP or use GitHub with the Colab
Secret `GITHUB_TOKEN`. The repository remains private. No Docker, tunnel or HTTP
application server is used. The evaluation calls the actual `UserRAG.ask` pipeline.

The notebook installs requirements, checks fixtures, runs offline tests, starts
Ollama and verifies GPU use, pulls the models, runs a two-question smoke test, then
runs all 20 questions. The default answer model is `qwen2.5:7b`, with `qwen2.5:3b`
available in a form field. `EVAL_LLM_MODEL` or `--model` also selects the answer
model. `EVAL_JUDGE_MODEL` or `--judge-model` optionally chooses a different judge;
by default the answer model judges itself and the report states that limitation.

## CLI

```bash
# These checks need no Ollama or model downloads.
python eval/run_eval.py --validate-only
python -m unittest discover -s tests -v

# After installing requirements and starting Ollama with both models pulled:
python eval/run_eval.py --model qwen2.5:7b --limit 2 --out results/smoke
python eval/run_eval.py --model qwen2.5:7b --out results
python eval/run_eval.py --skip-e2e --out results/retrieval-only
```

The retrieval-only command still requires the embedding model, BM25 model and
cross-encoder, but makes no LLM calls. Reranking explicitly uses CUDA when available;
if the requested cross-encoder cannot load, the evaluation fails instead of silently
reporting an unreranked run as reranked. Its actual device is recorded.

## Resume and isolation

On Colab, output is written directly to `My Drive/agentic-docqa-eval`. A stable
`<run-id>.partial.json` is replaced after **every completed question**, before moving
on. Repeating the command skips compatible completed questions and retries errors.
Timestamped JSON and Markdown reports are regenerated after each question too.
Smoke results live in a separate `smoke/` directory. `--limit` is not part of run
identity, so increasing a limit in the same output directory can continue a run.

Run identity includes corpus, golden questions, relevant app/eval source, settings,
answer/judge models and weight digests, Ollama version, dependency versions and GPU information. `--run-id` can supply
a readable stable label; an incompatible existing label is rejected. The CLI is
intended to run as a fresh Python process, with settings applied before app imports.

The dedicated `eval_user` is created in a separate local work directory, with its
own embedded Qdrant database. Only a directory carrying the matching evaluation
ownership marker is reset. Normal app users, Drive backups and search indexes are
never used or deleted. A disconnected runtime rebuilds the small corpus and resumes
question results from Drive; in-progress graph execution is not checkpointed.
Run one evaluation process per output/run ID at a time.
Model and package download caches use the existing `My Drive/Agentic DocQA` folder;
Ollama logs also persist under the evaluation results folder.

## Metrics and limitations

- The corpus has five fictional policy documents of 1,500-4,000 characters. There
  are 20 questions: six single facts, three multi-part, four paraphrases, three
  exact identifiers, and four unanswerable questions. The fixture validator checks
  source-specific evidence phrases literally against the corpus and references.
- Each retrieval mode calls the real `search_child_chunks` tool. Dense-only,
  hybrid/no reranker, and hybrid/cross-encoder share ten candidates, a zero score
  threshold and the production output/deduplication rules. Reranking retains the
  production cutoff before parent deduplication. This evaluates the app's returned
  snippets, not a separately invented search implementation.
- Relevance requires the expected source and its paired evidence phrase. Hit@5,
  MRR@5 and precision@5 are reported on answerable questions only; precision always
  divides by five. Evidence coverage@5 measures how many required facts appear.
  Unanswerable retrieval metrics are N/A, not zeros.
- End-to-end questions use fresh thread IDs and empty history. Optional observers
  capture actual search results, parent fetches, previews, grading decisions and
  fallback events inside the parallel research graph. Observation does not alter
  prompts, transitions, answers or the app's default retrieval settings.
- The temperature-zero judge compares correctness against the reference and checks
  atomic answer claims against the evidence that the agent actually retrieved.
  Supporting quotes must occur in that captured context. Faithfulness is the
  fraction of extracted claims judged supported; a pure refusal has no factual
  claims and gets N/A faithfulness. Claim extraction and support judgments remain
  model estimates, not independently verified truth. If the evidence exceeds the
  judge budget, faithfulness is N/A and the truncation reason is recorded.
- Invalid judge output gets one retry and then an explicit error, never a default
  score of zero. Raw replies, errors, answers and evidence remain in JSON. The
  summary separates correct refusals on unanswerable questions from wrong refusals
  on answerable questions. Clarification is not automatically treated as refusal.
- CRAG counts distinguish explicit LLM insufficiency from empty retrieval and
  malformed/failed grading calls. Retry events are counted before the agent can
  reset its retry counter. Correctness for questions with/without retries is a
  descriptive comparison with group sizes, **not a causal ablation**. Worst-question
  failure labels, especially suspected grader issues, are diagnostic hypotheses.
- Answer latency excludes the separate judge and setup. The notebook warms the
  answer model; cache state and runtime load can still influence comparisons.

No requirements were pinned for this harness. It adds no dependencies beyond the
app's requirements; reports record the versions actually used. The offline suite
includes real LangGraph/tool tests with fake model replies when dependencies are
installed. Fixture/metric/storage/notebook checks also run without those packages.
The full GPU evaluation must be executed on Colab; no benchmark numbers have been
invented or committed.

## Application changes

| File | Additive change |
|---|---|
| `app/config.py` | `RETRIEVAL_MODE` defaults to hybrid; optional `RERANK_DEVICE` preserves automatic choice when unset. |
| `app/db/vector_store.py` | Select dense or hybrid retrieval per store, using the same collection. |
| `app/core/reranker.py` | Optional explicit reranker kind/device; existing default and fallback behavior remain. |
| `app/agent/observability.py` | Best-effort event delivery; observer failures are logged without changing decisions. |
| `app/agent/tools.py` | Optional reranker override and observation of actual search/parent results. |
| `app/agent/nodes.py` | Observe routing, preview evidence, sufficient/insufficient grades, scheduled retries and fallbacks. |
| `app/agent/graph.py` | Pass the observer into both outer and research graphs. |
| `app/core/rag.py` | Optional observer parameter on `UserRAG`; existing callers need no changes. |

The standalone `scripts/run_agentic_docqa_colab.py` source snapshot is refreshed to
include the new files, and its SQLite backup connections are explicitly closed.
Existing Drive persistence tests verify the refreshed source too.
