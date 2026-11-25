"""Supporting document inspection tools over the document store.

open_doc   : Cleans web boilerplate whitespace, splits text on natural paragraph/sentence boundaries,
             and provides navigation hints for long documents.
find_in_doc: Shares exact pagination boundaries with open_doc so (page N) hints are always aligned.
"""
from __future__ import annotations

import re
from typing import Any, List, Tuple

import config
from agent.types import ToolResult
from retrieval.base import DocStore
from tools.base import Tool


_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def clean_doc_text(text: str) -> str:
    """Normalize web-scraped boilerplate whitespace while preserving paragraph structure."""
    if not text:
        return ""
    s = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_MULTI_SPACE_RE.sub(" ", line).strip() for line in s.split("\n")]
    s = "\n".join(lines)
    return _MULTI_NEWLINE_RE.sub("\n\n", s).strip()


def paginate_text(text: str, page_chars: int = config.OPEN_DOC_CHARS) -> List[Tuple[int, int]]:
    """Split text into contiguous (start, end) page spans snapped to natural boundaries."""
    n = len(text)
    if n == 0:
        return [(0, 0)]
    step = max(200, int(page_chars))
    spans: List[Tuple[int, int]] = []
    start = 0
    while start < n:
        hard_end = min(n, start + step)
        if hard_end >= n:
            spans.append((start, n))
            break
        # Look back up to 20% of page_chars for a clean paragraph, line, sentence, or word boundary
        min_cut = start + int(step * 0.8)
        cut = -1
        for sep in ("\n\n", "\n", ". ", "? ", "! ", " "):
            pos = text.rfind(sep, min_cut, hard_end)
            if pos != -1:
                cut = pos + len(sep)
                break
        if cut <= start:
            cut = hard_end
        spans.append((start, cut))
        start = cut
    return spans


def page_for_offset(spans: List[Tuple[int, int]], offset: int) -> int:
    """Return the exact page index containing character `offset`."""
    for idx, (s, e) in enumerate(spans):
        if s <= offset < e:
            return idx
    return max(0, len(spans) - 1)


class OpenDocTool(Tool):
    name = "open_doc"
    description = (
        "Read a document by id, one page at a time (page starts at 0). "
        "Use after search to inspect full text; for long documents, use find_in_doc to jump to the right page."
    )
    parameters = {"doc_id": "string", "page": "int, optional - page index (default 0)"}
    json_schema = {
        "type": "object",
        "properties": {
            "doc_id": {"type": "string", "description": "Document identifier returned by search"},
            "page": {"type": "integer", "description": "Zero-based page index (default 0)"},
        },
        "required": ["doc_id"],
    }

    def __init__(self, store: DocStore):
        self.store = store

    def run(self, doc_id: str = "", page: int = 0, **_: Any) -> ToolResult:
        doc = self.store.get(str(doc_id))
        if doc is None:
            return ToolResult(f"Error: doc_id {doc_id!r} not found", error=True)
        text = clean_doc_text(doc.get("text", ""))
        spans = paginate_text(text, config.OPEN_DOC_CHARS)
        pages = len(spans)
        try:
            page = int(page)
        except (TypeError, ValueError):
            return ToolResult(f"Error: invalid page {page!r}", error=True)
        if page < 0 or page >= pages:
            return ToolResult(f"Error: page {page} out of range (valid: 0..{pages - 1})", error=True)

        start, end = spans[page]
        chunk = text[start:end]
        header = (
            f"doc_id={doc_id} url={doc.get('url', '')} "
            f"page={page + 1}/{pages} (index={page}, chars={start}-{end}/{len(text)})"
        )
        hint = ""
        if pages > 2 and page == 0:
            hint = f"\n[Long document ({pages} pages): use find_in_doc(doc_id={doc_id!r}, keyword=...) to jump directly to relevant pages.]"
        return ToolResult(f"{header}\n{chunk}{hint}", doc_ids=[str(doc_id)])


class FindInDocTool(Tool):
    name = "find_in_doc"
    description = "Find a keyword/phrase inside one document; returns matching snippets with exact open_doc page indices."
    parameters = {"doc_id": "string", "keyword": "string", "window": "int, optional - context chars (default 150)"}
    json_schema = {
        "type": "object",
        "properties": {
            "doc_id": {"type": "string", "description": "Document identifier"},
            "keyword": {"type": "string", "description": "Keyword or short phrase to locate inside the document"},
            "window": {"type": "integer", "description": "Context window size in characters around each match (default 150)"},
        },
        "required": ["doc_id", "keyword"],
    }

    def __init__(self, store: DocStore):
        self.store = store

    def run(self, doc_id: str = "", keyword: str = "", window: int = 150, **_: Any) -> ToolResult:
        doc = self.store.get(str(doc_id))
        if doc is None:
            return ToolResult(f"Error: doc_id {doc_id!r} not found", error=True)
        if not str(keyword).strip():
            return ToolResult("Error: empty keyword", error=True)
        text = clean_doc_text(doc.get("text", ""))
        spans = paginate_text(text, config.OPEN_DOC_CHARS)
        w = max(30, int(window))
        out: List[str] = []
        for m in re.finditer(re.escape(str(keyword).strip()), text, flags=re.I):
            page_idx = page_for_offset(spans, m.start())
            snippet = "..." + text[max(0, m.start() - w): m.end() + w].replace("\n", " ") + "..."
            out.append(f"(page {page_idx}) {snippet}")
            if len(out) >= 5:
                break
        if not out:
            return ToolResult(f"No match for {keyword!r} in {doc_id}")
        return ToolResult("\n".join(f"[{i + 1}] {s}" for i, s in enumerate(out)), doc_ids=[str(doc_id)])
