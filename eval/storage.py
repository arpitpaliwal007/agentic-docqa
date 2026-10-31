"""Stable run identity, incremental results and isolated, disposable evaluation data."""
import hashlib
import json
from pathlib import Path
import re
import shutil


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def code_fingerprint(root):
    paths = [*root.joinpath("app").rglob("*.py"), *root.joinpath("eval").rglob("*.py"),
             *root.joinpath("eval/corpus").glob("*.md"), root / "eval/golden.json", root / "requirements.txt"]
    return digest({p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)})


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
    temporary.replace(path)


def prepare_workdir(work_root, run_id, fingerprint):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_id):
        raise ValueError("run_id must contain only letters, numbers, underscores or hyphens")
    parent = Path(work_root).resolve()
    path = parent / ("eval-" + run_id)
    if path.is_symlink():
        raise ValueError("Evaluation work directory cannot be a symlink")
    marker = path / ".eval-owned.json"
    expected = {"owner": "agentic-docqa-eval", "fingerprint": fingerprint}
    if path.exists():
        if not marker.is_file() or json.loads(marker.read_text()) != expected:
            raise ValueError("Refusing to clear an unrecognized work directory")
        if path.resolve().parent != parent:
            raise ValueError("Work directory is outside the isolated evaluation root")
        shutil.rmtree(path)  # Only this marked, isolated eval directory; never the app's data.
    path.mkdir(parents=True)
    atomic_json(marker, expected)
    return path


class Results:
    def __init__(self, out, run_id, metadata):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_id):
            raise ValueError("Invalid run_id")
        self.path = Path(out) / f"{run_id}.partial.json"
        if self.path.exists():
            self.value = json.loads(self.path.read_text())
            if self.value["metadata"]["fingerprint"] != metadata["fingerprint"]:
                raise ValueError("Existing run uses different corpus, code or settings. Choose another run ID.")
        else:
            self.value = {"metadata": metadata, "results": {}}
            self.save()

    def save(self):
        atomic_json(self.path, self.value)

    def completed(self, question_id):
        return self.value["results"].get(question_id, {}).get("status") == "complete"

    def put(self, result):
        self.value["results"][result["id"]] = result
        self.save()
