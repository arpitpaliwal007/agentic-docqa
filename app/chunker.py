"""Parent/child chunking.

Parents: split on markdown headings, then merged/split so each lands between
MIN_PARENT and MAX_PARENT chars. Children: small overlapping windows over each
parent, which is what gets embedded. Search hits a child, the agent can then
pull the whole parent for context.
"""
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app import config


def _join(a, b):
    a.page_content += "\n\n" + b.page_content
    for k, v in b.metadata.items():
        a.metadata[k] = f"{a.metadata[k]} -> {v}" if k in a.metadata and a.metadata[k] != v else v


class Chunker:
    def __init__(self):
        self.header_split = MarkdownHeaderTextSplitter(config.HEADERS, strip_headers=False)
        self.child_split = RecursiveCharacterTextSplitter(
            chunk_size=config.CHILD_SIZE, chunk_overlap=config.CHILD_OVERLAP)
        self.big_split = RecursiveCharacterTextSplitter(
            chunk_size=config.MAX_PARENT, chunk_overlap=config.CHILD_OVERLAP)

    def _sized_parents(self, text):
        sections = self.header_split.split_text(text)
        merged, cur = [], None
        for s in sections:
            if cur is None:
                cur = s
            else:
                _join(cur, s)
            if len(cur.page_content) >= config.MIN_PARENT:
                merged.append(cur)
                cur = None
        if cur is not None:
            if merged:
                _join(merged[-1], cur)
            else:
                merged.append(cur)
        out = []
        for p in merged:
            out += [p] if len(p.page_content) <= config.MAX_PARENT else self.big_split.split_documents([p])
        return out

    def chunk(self, md_path, source, stem):
        text = Path(md_path).read_text(encoding="utf-8")
        parents, children = [], []
        for i, p in enumerate(self._sized_parents(text)):
            pid = f"{stem}_p{i}"
            p.metadata.update({"source": source, "parent_id": pid})
            parents.append((pid, p))
            children += self.child_split.split_documents([p])
        return parents, children
