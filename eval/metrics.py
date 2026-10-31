"""Deterministic metrics and fixture validation; no model dependencies."""
from collections import Counter
import json
import math
import re
import unicodedata

METRIC_POLICY = "eval-v2-rule-refusal-key-facts-agent-evidence-v1"

REFUSAL = re.compile(
    r"\b(?:do|does|did)\s+not\s+(?:contain|mention|specify|state|provide|say)\b|"
    r"\bnot\s+(?:explicitly\s+)?(?:mentioned|specified|stated|covered|provided|available|given)\b|"
    r"\bno\s+(?:(?:specific|explicit|direct|relevant|additional|further)\s+)*(?:information|details?|mention)\b|"
    r"\b(?:cannot|can['’]t|could\s+not|couldn['’]t)\s+(?:find|determine|identify|provide)\b|"
    r"\b(?:isn['’]t|aren['’]t|wasn['’]t|weren['’]t)\s+(?:explicitly\s+)?(?:covered|mentioned|specified|stated|given)\b|"
    r"\b(?:doesn['’]t|don['’]t)\s+(?:say|contain|mention|specify|state|provide)\b", re.I)


def refusal_detector(answer):
    """A phrase detector, not a semantic claim that the whole answer is a refusal."""
    return bool(REFUSAL.search(answer))


def normalize_answer(text):
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    text = re.sub(r"\$\s*(\d+(?:\.\d+)?)", r"\1 dollars", text)
    text = re.sub(r"\b(?:usd|us dollars?)\s*(\d+(?:\.\d+)?)", r"\1 dollars", text)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"[^\w\s.]", " ", text)
    text = re.sub(r"\b(dollars?|days?|months?|minutes?|members?|devices?|codes?)\b",
                  lambda match: match.group().rstrip("s"), text)
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)
    return " ".join(text.split())


def key_fact_recall(answer, question):
    if not question["answerable"] or not question["must_contain"]:
        return None
    text = " " + normalize_answer(answer) + " "
    return sum(" " + normalize_answer(phrase) + " " in text for phrase in question["must_contain"]) / len(question["must_contain"])


def answer_metrics(question, answer, judgment):
    judgment = judgment or {}
    rule = refusal_detector(answer)
    judge = judgment.get("abstained")
    recall = key_fact_recall(answer, question)
    correctness = judgment.get("correctness")
    return {"key_fact_recall": recall, "rule_refusal": rule, "judge_abstained": judge,
            "combined_abstained": True if rule else judge,
            "refusal_disagreement": rule != judge if type(judge) is bool else None,
            "partial_refusal": rule and recall is not None and recall > 0,
            "judge_disagreement": ((recall == 1 and correctness < .5) or (recall == 0 and correctness > .5))
                                  if recall is not None and correctness is not None else None}


def tool_evidence_hits(text):
    """Parse the exact tool text a grader saw; no cross-thread event matching."""
    hits = []
    for block in re.split(r"(?=\[Chunk \d+\])", text):
        source = re.search(r"^File:\s*([^\n]+)", block, re.M)
        parent = re.search(r"^Parent ID:\s*([^\n]+)", block, re.M)
        if source:
            hits.append({"source": source[1].strip(), "parent_id": parent[1].strip() if parent else "",
                         "content": block[source.end():].strip()})
    return hits


def agent_retrieval_metrics(events, question):
    hits, searches, parents, checks = [], [], [], []
    for event in events:
        kind = event["kind"]
        if kind == "search":
            searches.append({"query": event["query"], "chunks": event.get("chunks", []), "error": event.get("error")})
            if not event.get("error"):
                hits.extend(event.get("chunks", []))
        elif kind == "parent" and not event.get("error"):
            hit = {"source": event["source"], "parent_id": event.get("parent_id", ""), "content": event["text"]}
            parents.append(hit)
            hits.append(hit)
        elif kind == "preview":
            try:
                previews = json.loads(event["content"])
                hits.extend({"source": item["name"], "content": item["preview"]} for item in previews)
            except (ValueError, KeyError, TypeError):
                pass
        elif kind == "grade" and event.get("origin") == "llm" and "evidence" in event and question["answerable"]:
            graded = tool_evidence_hits(event["evidence"])
            metrics = retrieval_metrics(graded, question, k=max(1, len(graded)))
            issue = ((event["verdict"] == "insufficient" and metrics["evidence_coverage_at_5"] == 1) or
                     (event["verdict"] == "sufficient" and metrics["hit_at_5"] == 0))
            checks.append({"index": event.get("index"), "question": event.get("question"),
                           "verdict": event["verdict"], "evidence": event["evidence"], "metrics": metrics, "issue": issue})
    metrics = retrieval_metrics(hits, question, k=max(1, len(hits)))
    return {"agent_retrieval_hit": metrics["hit_at_5"], "evidence_coverage": metrics["evidence_coverage_at_5"],
            "queries": [item["query"] for item in searches], "searches": searches, "parents": parents,
            "grader_checks": checks, "grader_issue": any(check["issue"] for check in checks)}


def add_diagnostics(question, record):
    e2e = record.get("e2e", {})
    if "answer" in e2e:
        e2e["answer_metrics"] = answer_metrics(question, e2e["answer"], e2e.get("judgment"))
        e2e["agent_retrieval"] = agent_retrieval_metrics(e2e.get("events", []), question)
        e2e["metric_policy"] = METRIC_POLICY


def classify_failure(question, record):
    if record.get("errors"):
        return "execution/judge error"
    e2e = record.get("e2e", {})
    judgment = e2e.get("judgment") or {}
    agent = e2e.get("agent_retrieval") or agent_retrieval_metrics(e2e.get("events", []), question)
    if agent.get("grader_issue"):
        return "grader issue"
    if question["answerable"] and "answer" in e2e and agent.get("agent_retrieval_hit") == 0:
        return "retrieval miss (agent queries)"
    metrics = e2e.get("answer_metrics") or answer_metrics(question, e2e.get("answer", ""), judgment)
    if judgment.get("correctness", 1) < 1 or (question["answerable"] and metrics.get("combined_abstained")):
        return "generation error"
    return "none"


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
