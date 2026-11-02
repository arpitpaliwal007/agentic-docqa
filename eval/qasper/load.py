"""Canonical QASPER papers, evidence maps and reproducible constrained sampling."""
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import random
import tarfile
import urllib.request

from eval.storage import atomic_json, digest
from eval.qasper.metrics import normalize

TYPES = ('extractive', 'abstractive', 'boolean', 'none')
LOADER_POLICY = 'qasper-text-v1'
RELEASES = {
    'test': ('https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz', 'qasper-test-v0.3.json'),
    'validation': ('https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz', 'qasper-dev-v0.3.json'),
    'train': ('https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz', 'qasper-train-v0.3.json'),
}


def rows(value):
    """HF Sequence(dict) can be column-oriented; official JSON uses row lists."""
    if isinstance(value, list):
        return value
    if not value:
        return []
    lengths = {len(item) for item in value.values() if isinstance(item, list)}
    if len(lengths) != 1 or any(not isinstance(item, list) for item in value.values()):
        raise ValueError('Invalid column-oriented QASPER sequence')
    return [{key: item[i] for key, item in value.items()} for i in range(next(iter(lengths)))]


def annotation_type(answer):
    if answer.get('unanswerable'):
        return 'none'
    if answer.get('yes_no') is not None:
        return 'boolean'
    return 'extractive' if answer.get('extractive_spans') else 'abstractive'


def reference_text(answer):
    kind = annotation_type(answer)
    if kind == 'none':
        return 'Unanswerable'
    if kind == 'boolean':
        return 'Yes' if answer['yes_no'] else 'No'
    if kind == 'extractive':
        return ' '.join(answer['extractive_spans'])
    return answer.get('free_form_answer') or ''


def canonical_paper(raw):
    paper_id = str(raw['id'])
    source = 'qasper-' + hashlib.sha256(paper_id.encode()).hexdigest()[:20] + '.md'
    paragraphs, by_text, markdown = {}, {}, [f"# {raw['title']}", '## Abstract']
    def add(text):
        exact = ' '.join(str(text).split())
        if not exact:
            return
        markdown.append(exact)
        key = normalize(exact)
        if key and key not in by_text:
            pid = f'{paper_id}:p{len(paragraphs):04d}'
            by_text[key] = pid
            paragraphs[pid] = exact
    add(raw.get('abstract', ''))
    for section in rows(raw.get('full_text', [])):
        markdown.append('## ' + (section.get('section_name') or 'Untitled section'))
        for paragraph in section.get('paragraphs') or []:
            add(paragraph)
    questions = []
    for qa in rows(raw.get('qas', [])):
        annotations = []
        for item in rows(qa.get('answers', [])):
            answer = item.get('answer', item)
            if isinstance(answer, list):
                raise ValueError('Expected one answer object per annotation')
            kind = annotation_type(answer)
            evidence = [] if kind == 'none' else list(answer.get('evidence') or [])
            floats = [text for text in evidence if 'FLOAT SELECTED' in text]
            text_evidence = [text for text in evidence if 'FLOAT SELECTED' not in text]
            evidence_ids = []
            for text in text_evidence:
                key = normalize(text)
                if not key:
                    continue
                pid = by_text.get(key, paper_id + ':unmapped:' + hashlib.sha256(key.encode()).hexdigest()[:20])
                evidence_ids.append(pid)
            annotations.append({'type': kind, 'reference': reference_text(answer),
                                'evidence_ids': sorted(set(evidence_ids)), 'text_evidence': text_evidence,
                                'float_evidence': floats, 'annotation_id': item.get('annotation_id'),
                                'answer': {field: answer.get(field) for field in ('unanswerable', 'yes_no', 'extractive_spans', 'free_form_answer', 'evidence', 'highlighted_evidence')}})
        if not annotations:
            raise ValueError(f"No annotations for {qa.get('question_id')}")
        counts = Counter(ann['type'] for ann in annotations)
        majority = max(TYPES, key=lambda kind: counts[kind])
        gold_text = set().union(*(set(ann['evidence_ids']) for ann in annotations))
        float_count = sum(len(ann['float_evidence']) for ann in annotations)
        questions.append({'id': qa['question_id'], 'paper_id': paper_id, 'question': qa['question'],
                          'majority_type': majority, 'annotations': annotations,
                          'float_evidence_count': float_count, 'float_only': bool(float_count and not gold_text),
                          'unmapped_evidence_ids': sorted(gold_text - set(paragraphs))})
    return {'id': paper_id, 'title': raw['title'], 'source': source, 'markdown': '\n\n'.join(markdown) + '\n',
            'paragraphs': paragraphs, 'questions': questions}


def load_split(split, cache_dir, *, hf_loader=None):
    if split not in RELEASES:
        raise ValueError('split must be train, validation or test')
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    canonical = cache / f'{split}-{LOADER_POLICY}.json'
    if canonical.exists():
        value = json.loads(canonical.read_text())
        if digest(value['papers']) != value['metadata']['dataset_hash']:
            raise ValueError('QASPER cache checksum mismatch; remove the corrupt cache file and prepare again')
        return value['papers'], value['metadata']
    try:
        if hf_loader is None:
            from datasets import load_dataset
            hf_loader = load_dataset
        dataset = hf_loader('allenai/qasper', split=split, cache_dir=str(cache / 'huggingface'))
        papers = {str(item['id']): canonical_paper(item) for item in dataset}
        origin = {'provider': 'huggingface', 'dataset': 'allenai/qasper', 'hf_fingerprint': getattr(dataset, '_fingerprint', None)}
    except Exception as error:
        url, member_name = RELEASES[split]
        archive_path = cache / url.rsplit('/', 1)[1]
        if not archive_path.exists():
            temporary = archive_path.with_suffix('.part')
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as output:
                import shutil
                shutil.copyfileobj(response, output)
            temporary.replace(archive_path)
        with tarfile.open(archive_path, 'r:gz') as archive:
            member = next((item for item in archive.getmembers() if item.isfile() and Path(item.name).name == member_name), None)
            if member is None:
                raise ValueError(f'Official release lacks {member_name}')
            stream = archive.extractfile(member)  # Read one member; never extract archive paths.
            raw = json.load(stream)
        papers = {str(pid): canonical_paper({**item, 'id': str(pid)}) for pid, item in raw.items()}
        origin = {'provider': 'official-release', 'url': url, 'member': member_name,
                  'archive_sha256': hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                  'hf_failure': f'{type(error).__name__}: {str(error)[:300]}'}
    questions = [q for paper in papers.values() for q in paper['questions']]
    if len({q['id'] for q in questions}) != len(questions):
        raise ValueError('QASPER question IDs are not unique')
    metadata = {**origin, 'split': split, 'loader_policy': LOADER_POLICY, 'dataset_hash': digest(papers),
                'paper_count': len(papers), 'question_count': len(questions)}
    atomic_json(canonical, {'metadata': metadata, 'papers': papers})
    return papers, metadata


def sample_questions(papers, n=250, seed=42, minimum=20):
    if n <= 0 or minimum < 0:
        raise ValueError('n must be positive and minimum must be nonnegative')
    groups = {kind: [] for kind in TYPES}
    for paper in papers.values():
        for question in paper['questions']:
            groups[question['majority_type']].append(question)
    population = {kind: len(group) for kind, group in groups.items()}
    total = sum(population.values())
    size = min(n, total)
    if not total:
        raise ValueError('No QASPER questions available')
    quotas = {kind: min(minimum, population[kind]) for kind in TYPES}
    if sum(quotas.values()) > size:
        raise ValueError(f'n={n} cannot satisfy the type minimums; use n >= {sum(quotas.values())} (or --limit for smoke)')
    targets = {kind: size * population[kind] / total for kind in TYPES}
    while sum(quotas.values()) < size:
        choices = [kind for kind in TYPES if quotas[kind] < population[kind]]
        winner = max(choices, key=lambda kind: targets[kind] - quotas[kind])
        quotas[winner] += 1
    rng = random.Random(seed)
    selected = []
    for kind in TYPES:
        selected.extend(rng.sample(sorted(groups[kind], key=lambda q: (q['paper_id'], q['id'])), quotas[kind]))
    rng.shuffle(selected)
    details = {'requested_n': n, 'sampled_n': len(selected), 'seed': seed, 'minimum_per_type': minimum,
               'population': population, 'quotas': quotas,
               'rare_types_take_all': [kind for kind in TYPES if population[kind] < minimum],
               'notes': ['Constrained proportional allocation; type minimums can change natural proportions.',
                         'Majority ties use extractive, abstractive, boolean, none in that fixed order.']}
    return selected, details


def prepare_sample(papers, metadata, n, seed, sample_dir, cache_dir, minimum=20):
    selected, details = sample_questions(papers, n, seed, minimum)
    manifest = {**details, 'split': metadata['split'], 'dataset_hash': metadata['dataset_hash'],
                'questions': [{'question_id': q['id'], 'paper_id': q['paper_id']} for q in selected]}
    filename = f"sample_{metadata['split']}_{n}_{seed}.json"
    for directory in (Path(sample_dir), Path(cache_dir) / 'samples'):
        path = directory / filename
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError(f'Sample manifest differs from current data/policy: {path}')
        atomic_json(path, manifest)
    return selected, manifest
