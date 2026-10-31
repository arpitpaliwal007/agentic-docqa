"""Deterministic metrics and fixture validation; no model dependencies."""
from collections import Counter
import json
import math
import re


def normalize(text):
    return " ".join(str(text).casefold().split())


def validate_golden(questions, corpus):
    if not isinstance(questions, list) or len(questions) != 20:
        raise ValueError("golden.json must contain exactly 20 questions")
    counts, ids = Counter(), set()
    allowed = {"single_fact", "multi_part", "paraphrase", "exact_identifier", "unanswerable"}
    for name, text in corpus.items():
        if not 1500 <= len(text) <= 4000 or not text.startswith("# ") or "\n## " not in text:
            raise ValueError(f"Invalid corpus document length or headings: {name}")
    if not 4 <= len(corpus) <= 6:
        raise ValueError("Expected 4-6 corpus documents")
    for q in questions:
        if not isinstance(q.get("id"), str) or not re.fullmatch(r"q\d{2}", q["id"]) or q["id"] in ids:
            raise ValueError("Question IDs must be unique qNN strings")
        ids.add(q["id"])
        if not all(isinstance(q.get(key), str) and q[key].strip() for key in ("question", "reference_answer")):
            raise ValueError("Missing question or reference_answer")
        if type(q.get("answerable")) is not bool or q.get("type") not in allowed:
            raise ValueError("Invalid answerable/type")
        counts[q["type"]] += 1
        for key in ("source", "must_contain", "evidence"):
            if not isinstance(q.get(key), list):
                raise ValueError(f"{key} must be a list")
        if not q["answerable"]:
            if q["type"] != "unanswerable" or any(q[k] for k in ("source", "must_contain", "evidence")):
                raise ValueError("Unanswerable questions must have no positive evidence")
            continue
        if q["type"] == "unanswerable" or not 1 <= len(q["must_contain"]) <= 3 or not q["evidence"]:
            raise ValueError("Answerable questions require 1-3 phrases and evidence")
        sources, phrases = set(), set()
        for unit in q["evidence"]:
            source, phrase = unit.get("source"), unit.get("phrase")
            if source not in corpus or not isinstance(phrase, str) or not phrase.strip():
                raise ValueError(f"Invalid evidence in {q['id']}")
            if normalize(phrase) not in normalize(corpus[source]) or normalize(phrase) not in normalize(q["reference_answer"]):
                raise ValueError(f"Unsupported evidence/reference in {q['id']}: {phrase}")
            sources.add(source)
            phrases.add(phrase)
        if sources != set(q["source"]) or phrases != set(q["must_contain"]):
            raise ValueError("Source and phrase fields must agree with evidence")
    for kind, minimum in {"unanswerable": 4, "multi_part": 3, "paraphrase": 4, "exact_identifier": 3}.items():
        if counts[kind] < minimum:
            raise ValueError(f"Need at least {minimum} {kind} questions")
    return dict(counts)


def retrieval_metrics(hits, question, k=5):
    if k <= 0:
        raise ValueError("k must be positive")
    if not question["answerable"]:
        return {key: None for key in ("hit_at_5", "mrr_at_5", "precision_at_5", "evidence_coverage_at_5")}
    units = question["evidence"]
    flags, covered = [], set()
    for hit in hits[:k]:
        matching = {i for i, unit in enumerate(units)
                    if hit["source"] == unit["source"] and normalize(unit["phrase"]) in normalize(hit["content"])}
        flags.append(bool(matching))
        covered.update(matching)
    return {"hit_at_5": float(any(flags)),
            "mrr_at_5": next((1.0 / i for i, flag in enumerate(flags, 1) if flag), 0.0),
            "precision_at_5": sum(flags) / k,
            "evidence_coverage_at_5": len(covered) / len(units)}


def mean(values):
    values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]
    return sum(values) / len(values) if values else None


def validate_judgment(value, context):
    json.dumps(value, allow_nan=False)  # Reject nonfinite values even in optional judge fields.
    score = value.get("correctness")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Judge correctness must be a finite number in [0, 1]")
    if type(value.get("abstained")) is not bool or not isinstance(value.get("claims"), list):
        raise ValueError("Judge must supply abstained and claims")
    for index, claim in enumerate(value["claims"]):
        if not isinstance(claim.get("claim"), str) or not claim["claim"].strip() or type(claim.get("supported")) is not bool:
            raise ValueError("Invalid judged claim")
        if claim["supported"]:
            quote = claim.get("evidence_quote")
            if not isinstance(quote, str) or not quote.strip() or normalize(quote) not in normalize(context):
                raise ValueError(f"claims[{index}]: supported claim needs a literal quote from the captured evidence")
    value = dict(value)
    claims = value["claims"]
    value["faithfulness"] = sum(c["supported"] for c in claims) / len(claims) if claims else None
    return value
