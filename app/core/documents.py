"""Upload -> markdown -> parent/child chunks -> Qdrant + parent store. Also delete/list."""
import logging
import os
import shutil
from pathlib import Path

from app import config
from app.chunker import Chunker
from app.db.parent_store import ParentStore
from app.db.vector_store import VectorStore
from app.utils import safe_stem

log = logging.getLogger(__name__)
ALLOWED = {".pdf", ".md", ".markdown", ".txt"}


def to_markdown(src: Path, dst: Path):
    ext = src.suffix.lower()
    if ext in {".md", ".markdown", ".txt"}:
        shutil.copyfile(src, dst)
        return
    if os.environ.get("LLAMA_CLOUD_API_KEY"):
        try:
            from llama_cloud_services import LlamaParse
            docs = LlamaParse(result_type="markdown").load_data(str(src))
            dst.write_text("\n\n".join(d.text for d in docs), encoding="utf-8")
            return
        except Exception as e:
            log.warning("LlamaParse failed (%s), falling back to pymupdf4llm", e)
    import pymupdf4llm
    dst.write_text(pymupdf4llm.to_markdown(str(src)), encoding="utf-8")


class DocumentManager:
    def __init__(self, user_dir: Path, collection: str):
        self.md_dir = user_dir / "markdown"
        self.upload_dir = user_dir / "uploads"
        self.parents = ParentStore(user_dir / "parents")
        self.collection = collection
        self.vdb = VectorStore()
        self.chunker = Chunker()

    def list(self):
        return sorted(p.name for p in self.upload_dir.iterdir() if p.is_file())

    def previews(self, chars=600):
        out = []
        for name in self.list():
            md = self.md_dir / f"{safe_stem(name)}.md"
            text = md.read_text(encoding="utf-8")[:chars] if md.exists() else ""
            out.append({"name": name, "preview": text})
        return out

    def add(self, filename, data: bytes):
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED:
            raise ValueError(f"Unsupported file type {ext}. Use PDF, Markdown or TXT.")
        if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
            raise ValueError(f"File is over the {config.MAX_UPLOAD_MB} MB limit.")
        name = Path(filename).name
        if name in self.list():
            self.remove(name)  # re-upload replaces the old version
        src = self.upload_dir / name
        src.write_bytes(data)
        stem = safe_stem(name)
        md = self.md_dir / f"{stem}.md"
        try:
            to_markdown(src, md)
            parents, children = self.chunker.chunk(md, source=name, stem=stem)
            if not children:
                raise ValueError("No text could be extracted from that file.")
            self.parents.save_many(parents)
            self.vdb.store(self.collection).add_documents(children)
        except Exception:
            self.remove(name)
            raise
        return {"name": name, "parents": len(parents), "children": len(children)}

    def remove(self, name):
        stem = safe_stem(name)
        removed = self.vdb.delete_source(self.collection, name)
        self.parents.delete_doc(stem)
        for f in (self.upload_dir / name, self.md_dir / f"{stem}.md"):
            f.unlink(missing_ok=True)
        return removed

    def stats(self):
        return {"documents": len(self.list()), "parents": self.parents.count(),
                "children": self.vdb.count(self.collection)}
