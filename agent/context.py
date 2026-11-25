"""Context management for the search agent.

Provides:
  - ContextManager: full-history pass-through baseline (used when context control is disabled).
  - StructuredContextManager: state-driven context management combining:
      1. Structured internal state (AgentState) maintained separately from raw history
      2. Deduplication (evidence deduplication, visited-document tagging, duplicate-call tracking)
      3. Turn-atomic sliding-window pruning and hard character budget (preserving OpenAI tool_call_id alignment
         and runtime-loaded skill instructions)
      4. Zero-token extractive evidence compression and optional LLM note summarization
         (routed through agent.call_llm so all tokens are accurately counted)
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Set, Tuple

import config
from agent.types import AgentState, ToolResult


_WORD_RE = re.compile(r"[A-Za-z0-9]{2,}")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_STOPWORDS = {
    "the", "and", "for", "that", "with", "from", "this", "was", "were", "are",
    "have", "has", "had", "not", "but", "what", "which", "who", "when", "where",
    "how", "why", "into", "about", "their", "they", "them", "been", "also",
    "same", "said", "full", "name", "known", "year", "later", "early", "first",
    "different", "role", "type", "many", "more", "most", "some", "such", "than",
}


def _extract_keywords(text: str) -> Set[str]:
    return {w.lower() for w in _WORD_RE.findall(str(text or "")) if w.lower() not in _STOPWORDS}


def _score_span(span: str, keywords: Set[str]) -> float:
    if not span or not keywords:
        return 0.0
    words = _extract_keywords(span)
    if not words:
        return 0.0
    overlap = len(words & keywords)
    if overlap == 0:
        return 0.0
    digit_bonus = 0.25 if re.search(r"\b\d{4}\b", span) else 0.0
    return overlap / (len(keywords) ** 0.5) + digit_bonus


def normalize_call_signature(tool_name: str, args: Dict[str, Any]) -> str:
    """Canonical key for detecting duplicate tool calls."""
    norm_args: Dict[str, Any] = {}
    for k, v in sorted((args or {}).items()):
        if isinstance(v, str):
            norm_args[k] = " ".join(v.strip().lower().split())
        else:
            norm_args[k] = v
    return f"{tool_name}:{json.dumps(norm_args, sort_keys=True, ensure_ascii=False)}"


def _msg_chars(msg: Dict[str, Any]) -> int:
    n = len(msg.get("content") or "")
    if msg.get("tool_calls"):
        n += len(json.dumps(msg["tool_calls"], ensure_ascii=False))
    return n


def _total_chars(messages: List[Dict[str, Any]]) -> int:
    return sum(_msg_chars(m) for m in messages)


def _group_into_turns(history: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Group conversation messages into atomic turns so tool responses stay paired with assistant tool_calls."""
    turns: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    for msg in history:
        if msg.get("role") == "assistant":
            if cur:
                turns.append(cur)
            cur = [dict(msg)]
        else:
            if not cur:
                cur = [dict(msg)]
            else:
                cur.append(dict(msg))
    if cur:
        turns.append(cur)
    return turns


class ContextManager:
    """Pass-through baseline that appends the full raw conversation history."""

    is_structured: bool = False

    def build_messages(self, state: AgentState, system_prompt: str, agent=None) -> List[Dict[str, Any]]:
        msgs: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}] + [dict(m) for m in state.messages]
        chars = _total_chars(msgs)
        state.context_stats.append({
            "step": state.step,
            "raw_chars": chars,
            "sent_chars": chars,
            "dropped_msgs": 0,
            "evidence_count": len(state.evidence),
            "visited_count": len(state.visited_docs),
        })
        return msgs

    def format_observation_for_history(
        self, state: AgentState, tool_name: str, args: Dict[str, Any], result: ToolResult
    ) -> str:
        return result.text

    def after_observation(
        self, state: AgentState, tool_name: str, args: Dict[str, Any], result: ToolResult, agent=None
    ) -> None:
        return None


class StructuredContextManager(ContextManager):
    """State-driven context manager with deduplication, turn-atomic pruning, and compression."""

    is_structured: bool = True

    def __init__(
        self,
        recent_turns: int = config.CTX_RECENT_TURNS,
        max_evidence: int = config.CTX_MAX_EVIDENCE,
        max_context_chars: int = config.CTX_MAX_CHARS,
        enable_llm_summary: bool = config.CTX_ENABLE_LLM_SUMMARY,
        max_steps: int = config.MAX_STEPS,
    ) -> None:
        self.recent_turns = max(1, int(recent_turns))
        self.max_evidence = max(2, int(max_evidence))
        self.max_context_chars = max(2500, int(max_context_chars))
        self.enable_llm_summary = bool(enable_llm_summary)
        self.max_steps = int(max_steps)

    def format_observation_for_history(
        self, state: AgentState, tool_name: str, args: Dict[str, Any], result: ToolResult
    ) -> str:
        """Annotate search results with [already opened]/[seen] markers to reduce redundant opens."""
        if tool_name == "search" and not result.error and isinstance(result.data, list):
            lines: List[str] = []
            for idx, hit in enumerate(result.data, start=1):
                did = str(hit.get("doc_id", ""))
                tag = " [already opened]" if did in state.opened_pages else (
                    " [seen]" if did in state.visited_docs else ""
                )
                lines.append(
                    f"[{idx}] doc_id={did}{tag} score={float(hit.get('score', 0.0)):.3f} "
                    f"snippet={hit.get('snippet', '')!r}"
                )
            if lines:
                return "\n".join(lines)
        return result.text

    def after_observation(
        self, state: AgentState, tool_name: str, args: Dict[str, Any], result: ToolResult, agent=None
    ) -> None:
        """Update structured AgentState and extract deduplicated key evidence (zero extra tokens)."""
        if result.error:
            return

        if tool_name == "search":
            q = str((args or {}).get("query", "")).strip()
            if q:
                state.add_sub_query(q)
        elif tool_name == "open_doc":
            did = str((args or {}).get("doc_id", "")).strip()
            try:
                page = int((args or {}).get("page", 0))
            except (TypeError, ValueError):
                page = 0
            if did:
                state.mark_opened_page(did, page)
        elif tool_name == "find_in_doc":
            kw = str((args or {}).get("keyword", "")).strip()
            if kw:
                state.add_sub_query(f"find:{kw}")
        elif tool_name == "load_skill":
            sname = str((args or {}).get("name", "")).strip()
            if sname and not str((args or {}).get("resource", "")).strip():
                state.loaded_skill_bodies[sname] = result.text

        keywords = _extract_keywords(state.question)
        for sq in state.sub_queries[-3:]:
            keywords |= _extract_keywords(sq)

        if tool_name == "open_doc":
            did = str((args or {}).get("doc_id", ""))
            body = result.text.split("\n", 1)[1] if "\n" in result.text else result.text
            candidates: List[Tuple[float, str]] = []
            for seg in _SENT_SPLIT_RE.split(body):
                seg = " ".join(seg.split()).strip()
                if len(seg) < 25:
                    continue
                sc = _score_span(seg, keywords)
                if sc > 0:
                    candidates.append((sc, seg))
            candidates.sort(key=lambda x: -x[0])
            for sc, seg in candidates[:2]:
                state.add_evidence(did, seg, source_tool="open_doc", score=sc + 0.5, max_items=self.max_evidence)

        elif tool_name == "find_in_doc":
            did = str((args or {}).get("doc_id", ""))
            for line in result.text.splitlines()[:2]:
                if line.startswith("[") and "No match" not in line:
                    sc = _score_span(line, keywords) + 0.8
                    state.add_evidence(did, line, source_tool="find_in_doc", score=sc, max_items=self.max_evidence)

        elif tool_name == "search" and isinstance(result.data, list):
            for hit in result.data[:2]:
                did = str(hit.get("doc_id", ""))
                snip = str(hit.get("snippet", ""))
                sc = _score_span(snip, keywords)
                if sc > 0:
                    state.add_evidence(did, snip, source_tool="search", score=sc, max_items=self.max_evidence)

        if self.enable_llm_summary and agent is not None:
            raw_chars = _total_chars(state.messages)
            if raw_chars > self.max_context_chars and (state.step - state.last_summary_step) >= 3:
                self._summarize_notes_with_llm(state, agent)

    def _summarize_notes_with_llm(self, state: AgentState, agent: Any) -> None:
        """Compress accumulated evidence and recent turns into concise state.notes via agent.call_llm."""
        recent_snippet = "\n".join(
            f"{m.get('role')}: {str(m.get('content') or '')[:500]}" for m in state.messages[-6:]
        )
        prompt = [
            {
                "role": "system",
                "content": (
                    "You compress a search agent's working notes into <=100 words of verified facts, "
                    "entity leads, and dead ends. Output plain text only."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {state.question}\n"
                    f"Previous notes: {state.notes or '(none)'}\n"
                    f"Recent trace:\n{recent_snippet}\n"
                    "Write updated concise progress notes:"
                ),
            },
        ]
        try:
            resp = agent.call_llm(prompt, state, max_tokens=1024)
            summary = " ".join(resp.content.split()).strip()
            if summary:
                state.notes = summary[:800]
            state.last_summary_step = state.step
            state.log(
                "context_summary",
                content=state.notes,
                tokens=resp.total_tokens,
                usage=resp.usage,
            )
        except Exception:
            return

    def build_messages(self, state: AgentState, system_prompt: str, agent=None) -> List[Dict[str, Any]]:
        """Assemble a bounded, protocol-valid prompt:
        [system_prompt] + [Question + Structured State (+ runtime loaded skills)] + [Pruned Recent Turns].
        """
        raw_full: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}] + [dict(m) for m in state.messages]
        raw_chars = _total_chars(raw_full)

        if not state.messages:
            state.context_stats.append({
                "step": state.step,
                "raw_chars": raw_chars,
                "sent_chars": raw_chars,
                "dropped_msgs": 0,
                "evidence_count": len(state.evidence),
                "visited_count": len(state.visited_docs),
            })
            return raw_full

        question_msg = state.messages[0]
        turns = _group_into_turns(state.messages[1:])

        state_parts = [state.render_state_block(max_evidence=self.max_evidence, max_steps=self.max_steps)]
        if state.loaded_skill_bodies:
            skill_blocks = [
                f"[ACTIVE SKILL: {sname}]\n{body[:2500]}"
                for sname, body in state.loaded_skill_bodies.items()
            ]
            state_parts.append("\n\n".join(skill_blocks))

        dropped_msgs = 0
        if len(turns) > self.recent_turns:
            older_turns = turns[:-self.recent_turns]
            kept_turns = turns[-self.recent_turns:]
            dropped_msgs = sum(len(t) for t in older_turns)
            pruned_obs = sum(
                1 for t in older_turns for m in t if m.get("role") in ("tool", "user")
            )
            state_parts.append(
                f"[CONTEXT PRUNING NOTICE: {dropped_msgs} earlier messages ({pruned_obs} tool observations) "
                f"were pruned from raw history; verified facts are preserved in Selected evidence above.]"
            )
        else:
            kept_turns = turns

        recent: List[Dict[str, Any]] = [dict(m) for t in kept_turns for m in t]
        # Strip any stale non-standard fields before sending to the API
        combined_user = f"{question_msg.get('content', '')}\n\n" + "\n\n".join(state_parts)

        # Enforce hard max_context_chars budget by compressing tool/user observation content (preserving tool_call_id pairs)
        fixed_chars = len(system_prompt) + len(combined_user)
        budget_for_recent = max(800, self.max_context_chars - fixed_chars)

        if _total_chars(recent) > budget_for_recent:
            for i in range(len(recent) - 1):
                if _total_chars(recent) <= budget_for_recent:
                    break
                if recent[i].get("role") in ("tool", "user") and len(recent[i].get("content") or "") > 360:
                    first_line = (recent[i].get("content") or "").splitlines()[0]
                    recent[i]["content"] = (
                        f"{first_line}\n[compressed for context budget — key facts saved in Structured Search State]"
                    )
            if _total_chars(recent) > budget_for_recent and recent:
                excess = _total_chars(recent) - budget_for_recent
                last_content = recent[-1].get("content") or ""
                target_len = max(400, len(last_content) - excess)
                if len(last_content) > target_len:
                    recent[-1]["content"] = last_content[:target_len] + "\n...[truncated to context budget]"

        out: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": combined_user},
        ]
        # Ensure clean role transition if recent starts with a stray user message
        if recent and recent[0].get("role") == "user":
            out[-1]["content"] += "\n\n" + str(recent[0].get("content") or "")
            recent = recent[1:]
        out.extend(recent)

        sent_chars = _total_chars(out)
        state.context_stats.append({
            "step": state.step,
            "raw_chars": raw_chars,
            "sent_chars": sent_chars,
            "dropped_msgs": dropped_msgs,
            "evidence_count": len(state.evidence),
            "visited_count": len(state.visited_docs),
        })
        return out
