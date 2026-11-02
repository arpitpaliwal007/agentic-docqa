"""QASPER summaries, intervals, paired tests and explicitly scoped comparisons."""
from collections import Counter
from eval.qasper.load import TYPES
from eval.qasper.pipeline import MODES
from eval.qasper.stats import mean_ci, paired_test
from eval.storage import atomic_json

OFFICIAL_EVALUATOR = 'https://github.com/allenai/qasper-led-baseline/blob/e996b6c7b1b5f95d9308a74e3586416c6e780df1/scripts/evaluator.py'
PAPER = 'https://arxiv.org/pdf/2105.03011'
BINARY = {'unanswerable_rule', 'unanswerable_format', 'wrong_refusal_rule', 'wrong_refusal_format', 'format_refusal_disagreement', 'format_fallback'}
LABELS = {'answer_f1_short':'Answer F1 — short (primary)', 'answer_f1_full':'Answer F1 — stripped full (secondary)',
          'evidence_f1':'Evidence F1 (agent-level, text only)', 'unanswerable_rule':'Unanswerable accuracy — rule',
          'unanswerable_format':'Unanswerable accuracy — formatter', 'wrong_refusal_rule':'Wrong refusal — rule',
          'wrong_refusal_format':'Wrong refusal — formatter', 'judge_correctness':'LLM-judge correctness',
          'faithfulness':'LLM-judge faithfulness', 'latency_seconds':'Mean agent latency (seconds)',
          'format_refusal_disagreement':'Rule vs formatter refusal disagreement', 'format_fallback':'Formatter fallback rate'}


def summarize(value):
    records = list(value['results'].values())
    seed = value['metadata']['seed']
    summary = {'metrics':{}, 'by_type':{}, 'retrieval':{}, 'paired_tests':{}, 'failures':{},
               'attempted':len(records), 'complete':sum(r['status']=='complete' for r in records),
               'errors':sum(r['status']=='error' for r in records),
               'float_only_questions':sum(r['float_only'] for r in records),
               'float_evidence_entries':sum(r['float_evidence_count'] for r in records),
               'missing_predictions':sum(r.get('missing_prediction', False) for r in records)}
    for key in LABELS:
        summary['metrics'][key] = mean_ci([r.get(key) for r in records], binary=key in BINARY, seed=seed)
    for kind in TYPES:
        short = [r.get('answer_f1_short') for r in records if r.get('best_type_short') == kind]
        full = [r.get('answer_f1_full') for r in records if r.get('best_type_full') == kind]
        summary['by_type'][kind] = {'short':mean_ci(short, seed=seed), 'full':mean_ci(full, seed=seed),
                                    'majority_n':sum(r['majority_type']==kind for r in records)}
    conditions = [f'prefix_{prefix}_{mode}' for prefix in (('off','on') if value['metadata']['prefix_ablation'] else ('on',)) for mode in MODES]
    def observations(condition, key):
        return [r.get('retrieval',{}).get(condition,{}).get('metrics',{}).get(key) for r in records]
    for condition in conditions:
        summary['retrieval'][condition] = {'hit_at_5':mean_ci(observations(condition,'hit_at_5'), binary=True, seed=seed),
                                           'mrr_at_5':mean_ci(observations(condition,'mrr_at_5'), seed=seed)}
    comparisons = []
    for prefix in (('off','on') if value['metadata']['prefix_ablation'] else ('on',)):
        comparisons += [(f'prefix_{prefix}_dense', f'prefix_{prefix}_hybrid'),
                        (f'prefix_{prefix}_hybrid', f'prefix_{prefix}_hybrid_rerank')]
    if value['metadata']['prefix_ablation']:
        comparisons += [(f'prefix_off_{mode}',f'prefix_on_{mode}') for mode in MODES]
    for left, right in comparisons:
        summary['paired_tests'][f'{left} -> {right}'] = paired_test(observations(left,'hit_at_5'), observations(right,'hit_at_5'), seed=seed)
    counts = Counter(category for r in records for category in r.get('failure_categories',[]))
    summary['failures'] = dict(counts)
    return summary


def formatted(interval, *, scale=100):
    if interval['mean'] is None:
        return 'N/A | N/A | 0'
    return f"{interval['mean']*scale:.2f} | [{interval['low']*scale:.2f}, {interval['high']*scale:.2f}] | {interval['n']}"


def text(value):
    return str(value).replace('|','\\|').replace('\n',' ').replace('`','\\`')


def render_report(value, summary=None):
    m = value['metadata']
    s = summary or summarize(value)
    records = list(value['results'].values())
    lines = ['# QASPER: Agentic DocQA', '',
             f"Split: **{m['split']}**; sample n={m['sampled_n']} (requested {m['n']}); seed={m['seed']}; corpus mode=**{m['mode']}**.",
             f"Answer: `{m['answer_model']}`; judge: `{m['judge_model']}`; formatter: `{m['format_model']}`.",
             f"Agent Nomic prefixes: on; direct-search prefix ablation: {'off + on' if m['prefix_ablation'] else 'on only'}.",
             f"Attempted {s['attempted']}/{m['sampled_n']}; complete {s['complete']}; errors {s['errors']}; missing predictions {s['missing_predictions']}.",
             f"Judge policy: `{m['judge_policy']}`. Formatter policy: `{m['format_policy']}`.",
             f"Dataset: {text(m['data_source'])}; fingerprint `{m['fingerprint']}`.", '']
    if value.get('last_error'):
        lines.append('Setup/runtime error: ' + text(value['last_error']))
    lines += ['## Main metrics', '', 'F1 and rates are shown on a 0–100 scale; latency is seconds. Unattempted questions are not assigned scores. Attempted missing predictions get official answer/evidence F1=0; unavailable model judgments remain N/A.', '',
              '| Metric | Mean | 95% CI | n |', '|---|---:|---|---:|']
    for key, label in LABELS.items():
        lines.append(f"| {label} | {formatted(s['metrics'][key], scale=1 if key=='latency_seconds' else 100)} |")
    lines += ['', 'Intervals: Wilson for binary rates; deterministic question-level percentile bootstrap (2,000 resamples) for F1, MRR, judge scores and latency. Questions from the same paper can be correlated; these intervals/tests are not paper-cluster adjusted. Small strata have unstable intervals.', '',
              '## Answer F1 by annotation answer type', '',
              'Like the official evaluator, each prediction is grouped by its best-matching reference type (first annotation wins ties). Short/full predictions can select different types. The majority type is used only for sampling and refusal-rate denominators.', '',
              '| Type | Majority sample n | Short F1 | 95% CI | Short n | Full F1 | 95% CI | Full n |', '|---|---:|---:|---|---:|---:|---|---:|']
    for kind, row in s['by_type'].items():
        lines.append(f"| {kind} | {row['majority_n']} | {formatted(row['short'])} | {formatted(row['full'])} |")
    lines += ['', '## Sampling and evidence scope', '',
              f"Population by majority type: {text(m['sample']['population'])}. Selected quotas: {text(m['sample']['quotas'])}.",
              f"Rare types (take all available): {text(m['sample']['rare_types_take_all'])}. Minimum per type: {m['sample']['minimum_per_type']}. Floors can alter natural proportions.",
              f"Float-only evidence questions in this attempted set: {s['float_only_questions']}; float evidence entries excluded: {s['float_evidence_entries']}.",
              f"Sample float-only total: {m['sample_float_only']}; unmatched text-evidence IDs in sample: {m['unmapped_evidence_count']}.",
              'Figures/tables are not indexed or scored as text evidence. Float-only questions remain in answer scoring and are flagged separately. Empty text-evidence gold follows the official empty-set convention, so Evidence F1 can be 1 for empty predictions on such questions.', '',
              '## Direct retrieval comparison', '',
              '| Condition | Evidence Hit@5 | 95% CI | n | MRR@5 | 95% CI | n |', '|---|---:|---|---:|---:|---|---:|']
    for condition, row in s['retrieval'].items():
        lines.append(f"| {condition} | {formatted(row['hit_at_5'])} | {formatted(row['mrr_at_5'])} |")
    lines += ['', 'Only questions with annotated text evidence are eligible. Hit@5 requires an individual top-five chunk to match a gold paragraph (substring or >80% token-position coverage). Agent-level evidence F1 can combine fragments across multiple snippets/parents. Wrong-paper chunks never match the gold paper in pooled mode.', '',
              '| Paired comparison (right minus left) | Paired n | Difference | 95% CI | McNemar p | Interpretation |', '|---|---:|---:|---|---:|---|']
    for name, result in s['paired_tests'].items():
        ci = result['difference_right_minus_left']
        difference = 'N/A' if ci['mean'] is None else f"{100*ci['mean']:.2f}"
        bounds = 'N/A' if ci['mean'] is None else f"[{100*ci['low']:.2f}, {100*ci['high']:.2f}]"
        p = 'N/A' if result['p_value'] is None else f"{result['p_value']:.5f}"
        label = 'No paired observations' if not result['n'] else ('Difference significant at 0.05' if result['significant_0_05'] else 'Difference is not significant at 0.05')
        lines.append(f"| {name} | {result['n']} | {difference} | {bounds} | {p} | {label} |")
    lines += ['', 'Exact two-sided McNemar tests use matched binary hit outcomes only; discordant counts are in JSON. Comparisons are exploratory, unadjusted for multiple testing, and not causal ablations of overall answer quality.', '', '## Failure categories', '']
    for category in ('routed_to_direct_chat','retrieval_miss','generation_error','false_refusal','float_only_evidence','execution_error'):
        examples = [r for r in records if category in r.get('failure_categories',[])][:5]
        lines += [f"### {category}: {s['failures'].get(category,0)}", '']
        for row in examples:
            gold = [ann['reference'] for ann in row['annotations']]
            lines += [f"- **{text(row['id'])}** — {text(row['question'])}",
                      f"  Gold: {text(gold)}; agent: {text(row.get('answer','(missing)'))}; short: {text(row.get('short_answer','(missing)'))}."]
    lines += ['', '## Comparability and caveats', '',
              f"The [QASPER paper]({PAPER}), Table 2, reports about **33 Answer F1** for LED-base (33.63 with evidence scaffolding) and about **60** for humans (60.92, a lower-bound estimate on a multi-annotator subset). Our retrieval agent, sampled corpus, LLM formatting step and scoring scope differ; this is not a like-for-like comparison.",
              f"Answer normalization, multiset token F1, max over annotations and empty evidence scoring follow the [released evaluator]({OFFICIAL_EVALUATOR}). Unlike the prompt's exact-match assumption, the released script gives partial token credit to categorical answers with extra words; categorical exact match is recorded separately in JSON. Space-joining extractive spans has the same normalized tokens as the official comma-space join.",
              'Text evidence is mapped to normalized gold-paper paragraph identities, with a >80% token-position union fallback for split paragraphs. This mapping is our text-retrieval adaptation, not the official evidence-selection model. Duplicate normalized paragraphs share one ID; unmapped gold evidence is kept as an unmatched ID, not discarded.',
              'The formatter is an additional LLM, sees only the original question/answer and can still change meaning without introducing new words. Its lexical guard does not prove entailment; stripped full-answer scores and refusal/fallback disagreements are reported as checks.',
              'The answering model also judges itself.' if m['self_judge'] else 'A separately configured model judges the answers.',
              'The existing two-pass judge is adapted only in evaluation to scientific assertions. Raw judge replies, formatter proposals, guard errors, exact agent queries/chunk/parent texts, evidence mappings and reference selection are in JSON. Latency measures the agent call, excluding indexing, direct retrieval, formatting and judging.',
              'Questions are passed verbatim to UserRAG.ask with fresh threads and empty history. No router prompt or answering policy is changed; direct-chat routes are recorded as failures.', '']
    return '\n'.join(lines)


def write_reports(state, *, markdown=True):
    base = state.path.parent / state.value['metadata']['report_basename']
    if markdown:
        state.value['summary'] = summarize(state.value)
    else:
        state.value.pop('summary', None)
    state.save()
    atomic_json(base.with_suffix('.json'), state.value)
    if markdown:
        temporary = base.with_suffix('.md.tmp')
        temporary.write_text(render_report(state.value, state.value['summary']), encoding='utf-8')
        temporary.replace(base.with_suffix('.md'))
        print('Report:', base.with_suffix('.md'), flush=True)
    return base.with_suffix('.md')
