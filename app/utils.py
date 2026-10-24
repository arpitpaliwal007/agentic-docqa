import re


def parse_json(text):
    """First {...} in a model reply, or None."""
    import json
    m = re.search(r"\{.*\}", str(text or ""), re.S)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except Exception:
        return None


def estimate_tokens(messages):
    # ~4 chars/token, good enough to decide when to compress
    return sum(len(str(getattr(m, "content", "") or "")) for m in messages) // 4


def safe_stem(name):
    stem = re.sub(r"[^\w\-]+", "_", name.rsplit(".", 1)[0]).strip("_")
    return stem or "document"
