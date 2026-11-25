"""Unified retrieval tool exposed to the search agent.

All underlying retrieval categories (sparse BM25S, encoder dense, LLM instruction-aware dense,
and RRF fusion) are encapsulated behind the single `Retriever` interface.
"""
from __future__ import annotations

from typing import Any

import config
from agent.types import ToolResult
from retrieval.base import Retriever
from tools.base import Tool


class SearchTool(Tool):
    name = "search"
    description = ("Search the document collection. Returns ranked doc ids, scores and short snippets. "
                   "Use specific keyword-rich queries; reformulate if results are poor.")
    parameters = {"query": "string - search query", "k": "int, optional - number of results (default 5)"}
    json_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Keyword-rich search query"},
            "k": {"type": "integer", "description": "Number of ranked results to return (default 5, max 10)"},
        },
        "required": ["query"],
    }

    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def run(self, query: str = "", k: int = config.SEARCH_TOP_K, **_: Any) -> ToolResult:
        if not str(query).strip():
            return ToolResult("Error: empty query", error=True)
        hits = self.retriever.search(str(query), int(k))
        if not hits:
            return ToolResult("No results.", data=[])
        lines = [f"[{i+1}] doc_id={h.doc_id} score={h.score:.3f} snippet={h.snippet!r}" for i, h in enumerate(hits)]
        return ToolResult("\n".join(lines), data=[h.to_dict() for h in hits], doc_ids=[h.doc_id for h in hits])
