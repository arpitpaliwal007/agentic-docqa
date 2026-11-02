"""Official answer scoring plus explicitly text-only, agent-level evidence mapping."""
from collections import Counter
from difflib import SequenceMatcher
import re
import string


def normalize(text):
    text = ''.join(ch for ch in str(text).lower() if ch not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def token_f1(prediction, reference):
    predicted, gold = normalize(prediction).split(), normalize(reference).split()
    shared = sum((Counter(predicted) & Counter(gold)).values())
    if not shared:
        return 0.
    precision, recall = shared / len(predicted), shared / len(gold)
    return 2 * precision * recall / (precision + recall)


def set_f1(predicted, gold):
    predicted, gold = set(predicted), set(gold)
    if not predicted and not gold:
        return 1.
    shared = len(predicted & gold)
    return 2 * shared / (len(predicted) + len(gold)) if shared else 0.


def answer_score(prediction, annotations):
    if not annotations:
        raise ValueError('A question needs at least one annotated answer')
    scores = [token_f1(prediction, ann['reference']) for ann in annotations]
    # First annotation wins exact ties, matching the stable sort in the release.
    index = max(range(len(scores)), key=scores.__getitem__)
    return {'f1': scores[index], 'reference_index': index, 'answer_type': annotations[index]['type'],
            'categorical_exact_match': float(normalize(prediction) == normalize(annotations[index]['reference']))
                if annotations[index]['type'] in {'none', 'boolean'} else None}


def tool_hits(events):
    hits = []
    for event in events:
        if event['kind'] == 'search' and not event.get('error'):
            hits.extend(event.get('chunks', []))
        elif event['kind'] == 'parent' and not event.get('error'):
            hits.append({'source': event['source'], 'parent_id': event.get('parent_id'),
                         'content': event.get('text', '')})
    return hits


def captured_evidence(events):
    hits = tool_hits(events)
    return '\n\n'.join(dict.fromkeys(f"File: {hit['source']}\nParent ID: {hit.get('parent_id', '')}\n{hit['content']}" for hit in hits))


def match_paragraphs(paper, hits):
    """Union token-position coverage across actual same-paper snippets, not token bags."""
    snippets = [normalize(hit['content']) for hit in hits if hit.get('source') == paper['source']]
    matched, details = set(), []
    for pid, text in paper['paragraphs'].items():
        paragraph = normalize(text)
        tokens = paragraph.split()
        if not tokens:
            continue
        if any(paragraph in snippet for snippet in snippets):
            matched.add(pid)
            details.append({'paragraph_id': pid, 'coverage': 1., 'method': 'substring'})
            continue
        covered = set()
        for snippet in snippets:
            for block in SequenceMatcher(None, tokens, snippet.split(), autojunk=False).get_matching_blocks():
                covered.update(range(block.a, block.a + block.size))
        coverage = len(covered) / len(tokens)
        if coverage > .8:
            matched.add(pid)
            details.append({'paragraph_id': pid, 'coverage': coverage, 'method': 'split-token-coverage'})
    return matched, details


def evidence_score(predicted_ids, annotations):
    return max(set_f1(predicted_ids, ann['evidence_ids']) for ann in annotations)


def retrieval_metrics(paper, question, hits, k=5):
    gold = set().union(*(set(ann['evidence_ids']) for ann in question['annotations']))
    if not gold:
        return {'hit_at_5': None, 'mrr_at_5': None, 'eligible': False}
    flags = [bool(match_paragraphs(paper, [hit])[0] & gold) for hit in hits[:k]]
    return {'hit_at_5': float(any(flags)), 'mrr_at_5': next((1./rank for rank, hit in enumerate(flags, 1) if hit), 0.), 'eligible': True}


def failure_categories(question, result):
    if 'answer' not in result:
        return ['execution_error'] if result.get('errors') else []
    categories = []
    if 'general_chat' in result.get('crag', {}).get('routes', []):
        categories.append('routed_to_direct_chat')
    if question['float_only']:
        categories.append('float_only_evidence')
    if question['majority_type'] != 'none' and (result.get('rule_refusal') or result.get('format_refusal')):
        categories.append('false_refusal')
    gold = set().union(*(set(ann['evidence_ids']) for ann in question['annotations']))
    if gold and not set(result.get('predicted_evidence_ids', [])) & gold:
        categories.append('retrieval_miss')
    elif result.get('answer_f1_short', 0) < 1 and result.get('predicted_evidence_ids'):
        categories.append('generation_error')
    if result.get('errors'):
        categories.append('execution_error')
    return list(dict.fromkeys(categories))
