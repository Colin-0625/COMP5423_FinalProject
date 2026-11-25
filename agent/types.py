"""Shared data structures for search hits, tool results, and structured agent state."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SearchHit:
    doc_id: str
    score: float
    snippet: str

    def to_dict(self) -> Dict[str, Any]:
        return {"doc_id": self.doc_id, "score": round(float(self.score), 4), "snippet": self.snippet}


@dataclass
class ToolResult:
    """Result returned by a tool execution."""
    text: str
    data: Optional[Any] = None
    doc_ids: List[str] = field(default_factory=list)
    error: bool = False


@dataclass
class AgentState:
    """Structured internal agent state maintained separately from raw conversation history."""
    question: str
    visited_docs: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)
    candidate_answer: Optional[str] = None
    loaded_skills: List[str] = field(default_factory=list)
    notes: str = ""
    opened_pages: Dict[str, List[int]] = field(default_factory=dict)
    seen_calls: Dict[str, str] = field(default_factory=dict)
    loaded_skill_bodies: Dict[str, str] = field(default_factory=dict)
    context_stats: List[Dict[str, int]] = field(default_factory=list)
    last_summary_step: int = 0
    step: int = 0
    tool_calls: int = 0
    total_tokens: int = 0
    messages: List[Dict[str, Any]] = field(default_factory=list)
    trajectory: List[Dict[str, Any]] = field(default_factory=list)

    def mark_visited(self, doc_ids: List[str]) -> None:
        for d in doc_ids:
            ds = str(d)
            if ds and ds not in self.visited_docs:
                self.visited_docs.append(ds)

    def mark_opened_page(self, doc_id: str, page: int) -> None:
        ds = str(doc_id)
        self.mark_visited([ds])
        pages = self.opened_pages.setdefault(ds, [])
        if int(page) not in pages:
            pages.append(int(page))
            pages.sort()

    def add_sub_query(self, query: str) -> None:
        q = " ".join(str(query or "").split())
        if q and q not in self.sub_queries:
            self.sub_queries.append(q)

    def add_evidence(
        self,
        doc_id: str,
        text: str,
        source_tool: str = "",
        score: float = 0.0,
        max_items: int = 8,
    ) -> bool:
        """Deduplicated, relevance-ranked evidence insertion."""
        clean = " ".join(str(text or "").split()).strip()
        if not clean:
            return False
        ds = str(doc_id)
        fp = (ds, clean[:120].lower())
        for item in self.evidence:
            if (str(item.get("doc_id", "")), str(item.get("text", ""))[:120].lower()) == fp:
                if float(score) > float(item.get("score", 0.0)):
                    item["score"] = round(float(score), 3)
                return False
        self.evidence.append({
            "doc_id": ds,
            "text": clean[:420],
            "source_tool": source_tool,
            "score": round(float(score), 3),
        })
        # Keep top max_items by score (stable: higher score first, newer breaks ties)
        if len(self.evidence) > max_items:
            self.evidence.sort(key=lambda e: float(e.get("score", 0.0)), reverse=True)
            del self.evidence[max_items:]
        return True

    def render_state_block(self, max_evidence: int = 6, max_steps: int = 12) -> str:
        """Render structured internal state into a compact prompt block for the LLM."""
        remaining = max(0, int(max_steps) - int(self.step) + 1)
        lines: List[str] = [
            f"[STRUCTURED SEARCH STATE | step={self.step}/{max_steps} | remaining_turns={remaining} | tool_calls={self.tool_calls}]"
        ]
        if self.sub_queries:
            lines.append("Sub-queries tried: " + " | ".join(f'"{q}"' for q in self.sub_queries[-8:]))
        if self.visited_docs:
            opened_str = ", ".join(
                f"{d}(p{','.join(str(p) for p in self.opened_pages[d])})" if d in self.opened_pages else d
                for d in self.visited_docs[-15:]
            )
            extra = f" (+{len(self.visited_docs) - 15} more)" if len(self.visited_docs) > 15 else ""
            lines.append(f"Visited docs ({len(self.visited_docs)}): {opened_str}{extra}")
        if self.loaded_skills:
            lines.append("Active loaded skills: " + ", ".join(self.loaded_skills))
        if self.candidate_answer:
            lines.append(f"Current candidate answer: {self.candidate_answer}")
        if self.evidence:
            lines.append("Selected evidence (deduplicated):")
            for idx, ev in enumerate(self.evidence[:max_evidence], start=1):
                lines.append(f"  [{idx}] doc_id={ev['doc_id']} ({ev.get('source_tool', '')}): {ev['text']}")
        if self.notes:
            lines.append(f"Progress notes: {self.notes}")

        # Early-stopping / budget convergence guidance
        if remaining <= 2:
            lines.append(
                "Budget hint: Very few steps remain. Consolidate verified evidence and output final_answer concisely."
            )
        elif self.candidate_answer and self.evidence:
            lines.append(
                "Convergence hint: A candidate answer is recorded with supporting evidence. "
                "Verify any remaining doubt quickly and emit final_answer to save tool calls and tokens."
            )
        return "\n".join(lines)

    def log(self, type_: str, **kw: Any) -> None:
        """Append one machine-readable trajectory step."""
        self.trajectory.append({"step": self.step, "type": type_, **kw})
