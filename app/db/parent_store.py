"""Parent chunks live as JSON files on disk; Qdrant only holds the children."""
import json
import re
from pathlib import Path


class ParentStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def _file(self, pid):
        return self.path / f"{pid}.json"

    def save_many(self, parents):
        for pid, doc in parents:
            self._file(pid).write_text(
                json.dumps({"content": doc.page_content, "metadata": doc.metadata}, ensure_ascii=False),
                encoding="utf-8",
            )

    def get(self, pid):
        f = self._file(pid)
        if not f.exists():
            return None
        data = json.loads(f.read_text(encoding="utf-8"))
        return {"parent_id": pid, **data}

    def get_many(self, pids):
        def order(p):
            m = re.search(r"_p(\d+)$", p)
            return int(m.group(1)) if m else 0
        return [d for d in (self.get(p) for p in sorted(set(pids), key=order)) if d]

    def delete_doc(self, stem):
        n = 0
        for f in self.path.glob(f"{stem}_p*.json"):
            f.unlink()
            n += 1
        return n

    def count(self):
        return sum(1 for _ in self.path.glob("*.json"))
