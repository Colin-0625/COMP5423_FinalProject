"""Iterative search agent loop using native OpenAI Function Calling, structured context control,
and runtime Agent Skills loading.

All LLM calls route through `SearchAgent.call_llm` so `state.total_tokens` accurately records
the sum of `usage.total_tokens` across every API call made for a question.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import config
from agent.context import ContextManager, StructuredContextManager, normalize_call_signature
from agent.llm_client import LLMClient, LLMResponse
from agent.types import AgentState, ToolResult
from tools.base import ToolRegistry
from tools.skill_tool import LoadSkillTool, SkillLibrary


SYSTEM_PROMPT_TEMPLATE = """You are a multi-step search agent answering challenging factual questions using a fixed document collection.
You must gather and verify evidence from the document collection via the provided tools before answering.
{skill_section}
Guidelines:
1. Decompose complex questions into concise, keyword-rich `search` queries (never paste an entire long question into `search`).
2. Make at most ONE focused tool call per step so each action builds on the previous observation. Inspect promising documents with `open_doc` or jump to key clues with `find_in_doc`, and avoid repeating identical tool calls.
3. When you have verified the answer, stop calling tools and output ONLY the final short answer (e.g., an entity name, number, year, or short phrase — NOT a full sentence, no markdown formatting, no prefix like "Answer:"), because evaluation uses normalized Exact Match (EM).
4. Stop as soon as you have verified the answer to minimize tool calls and token usage.
"""

SKILL_SECTION_TEMPLATE = """
Available reusable skills (names and descriptions ONLY; full instructions are loaded at runtime via `load_skill`):
{skill_catalog}

Automatic skill selection policy:
- Decide automatically whether a skill helps based on the current question or search state.
- Call `load_skill` ONLY when a skill clearly applies (e.g., multi-clue riddle, ambiguous candidates, or poor/empty search hits).
- If the question is straightforward and no skill is needed, skip `load_skill` and search directly.
"""


def _parse_tool_args(raw_args: Any) -> Dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    if not raw_args:
        return {}
    s = str(raw_args).strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(s[start:end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return {}


def _clean_final_answer(ans: Any) -> str:
    if ans is None:
        return ""
    s = str(ans).strip()
    if not s:
        return ""
    if "\n" in s:
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        # 1. If any line explicitly starts with "Final Answer:" or "Answer:", prefer it
        for ln in reversed(lines):
            m = re.match(r"^(?:final\s+answer|answer)\s*[:：]\s*(.+)$", ln, flags=re.I)
            if m and m.group(1).strip():
                s = m.group(1).strip()
                break
        else:
            # 2. Drop trailing parenthetical notes like "(1908-2001)" when an earlier short answer exists
            non_note = [ln for ln in lines if not (ln.startswith("(") and ln.endswith(")"))]
            candidates = non_note or lines
            short_lines = [ln for ln in candidates if len(ln) <= 60 and not ln.endswith(".")]
            s = short_lines[-1] if short_lines else candidates[-1]
    s = re.sub(r"^(?:final\s+answer|answer)\s*[:：]\s*", "", s, flags=re.I).strip()
    s = re.sub(r"^\*\*(.+?)\*\*$", r"\1", s).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1].strip()
    return s


@dataclass
class AgentRunResult:
    question: str
    answer: str
    state: AgentState

    def to_prediction_record(self, qid: str) -> Dict[str, Any]:
        """Serialize run output into the required JSONL prediction schema."""
        return {
            "id": str(qid),
            "question": self.question,
            "answer": self.answer,
            "trajectory": self.state.trajectory,
            "tool_calls": self.state.tool_calls,
            "total_tokens": self.state.total_tokens,
        }


class SearchAgent:
    """Skill-augmented iterative search agent using native OpenAI tool calling."""

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        context_manager: Optional[ContextManager] = None,
        skill_library: Optional[SkillLibrary] = None,
        max_steps: int = config.MAX_STEPS,
        use_skills: bool = True,
        force_skill: Optional[str] = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.context_manager = context_manager or ContextManager()
        self.skill_library = skill_library
        self.max_steps = max_steps
        self.use_skills = use_skills
        self.force_skill = force_skill

    def build_system_prompt(self) -> str:
        if self.use_skills and self.skill_library is not None and self.registry.get("load_skill") is not None:
            skill_section = SKILL_SECTION_TEMPLATE.format(skill_catalog=self.skill_library.catalog())
        else:
            skill_section = ""
        return SYSTEM_PROMPT_TEMPLATE.format(skill_section=skill_section)

    def call_llm(
        self,
        messages: List[Dict[str, Any]],
        state: AgentState,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kw: Any,
    ) -> LLMResponse:
        """Central LLM call entry point ensuring all token usage is accumulated in state."""
        resp = self.llm.chat(messages, tools=tools, **kw)
        state.total_tokens += int(resp.total_tokens)
        return resp

    def run(self, question: str) -> AgentRunResult:
        state = AgentState(question=question)
        system_prompt = self.build_system_prompt()
        openai_tools = self.registry.openai_tools()
        state.messages.append({"role": "user", "content": f"Question: {question}"})

        # Optional fixed-skill mode for controlled experiments
        if self.use_skills and self.force_skill and self.registry.get("load_skill") is not None:
            call_id = "call_force_skill_0"
            args = {"name": self.force_skill}
            state.messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "load_skill", "arguments": json.dumps(args)},
                }],
            })
            obs_text = self._execute_tool(state, "load_skill", args)
            state.messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": "load_skill",
                "content": obs_text,
            })

        answer: Optional[str] = None
        next_temp: Optional[float] = None
        for step in range(1, self.max_steps + 1):
            state.step = step
            messages = self.context_manager.build_messages(state, system_prompt, agent=self)
            ctx_snap = state.context_stats[-1] if state.context_stats else None
            ctx_meta = (
                {
                    "raw_chars": ctx_snap["raw_chars"],
                    "sent_chars": ctx_snap["sent_chars"],
                    "dropped_msgs": ctx_snap["dropped_msgs"],
                }
                if ctx_snap is not None
                else None
            )
            call_kw = {"temperature": next_temp} if next_temp is not None else {}
            resp = self.call_llm(messages, state, tools=openai_tools, **call_kw)
            next_temp = None

            if resp.tool_calls:
                thought_text = (resp.content or resp.reasoning_content or "").strip()
                state.log(
                    "thought",
                    content=thought_text[:2000],
                    tokens=resp.total_tokens,
                    usage=resp.usage,
                    ctx=ctx_meta,
                )
                assistant_msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": resp.content or "",
                    "tool_calls": resp.tool_calls,
                }
                state.messages.append(assistant_msg)

                for tc in resp.tool_calls:
                    fn = tc.get("function") or {}
                    tool_name = str(fn.get("name", "")).strip()
                    args = _parse_tool_args(fn.get("arguments"))
                    obs_text = self._execute_tool(state, tool_name, args)
                    state.messages.append({
                        "role": "tool",
                        "tool_call_id": str(tc.get("id", "")),
                        "name": tool_name,
                        "content": obs_text,
                    })
                continue

            # No tool calls: model returned a final answer (or empty response if truncated)
            cleaned = _clean_final_answer(resp.content)
            if not cleaned:
                state.log(
                    "thought",
                    content="",
                    tokens=resp.total_tokens,
                    usage=resp.usage,
                    empty_response=True,
                    ctx=ctx_meta,
                )
                state.messages.append({"role": "assistant", "content": ""})
                state.messages.append({
                    "role": "user",
                    "content": "Keep your reasoning brief and immediately call ONE tool or output ONLY the concise final answer.",
                })
                next_temp = 0.3
                continue

            state.log(
                "thought",
                content=(resp.reasoning_content or "").strip()[:2000],
                tokens=resp.total_tokens,
                usage=resp.usage,
                ctx=ctx_meta,
            )
            state.messages.append({"role": "assistant", "content": resp.content})
            answer = cleaned
            state.candidate_answer = answer
            state.log("answer", content=answer)
            break

        # Forced final synthesis if step budget is exhausted (tools disabled so model must answer)
        if answer is None:
            state.step = self.max_steps + 1
            state.messages.append({
                "role": "user",
                "content": (
                    "Step limit reached. Based on the verified evidence gathered so far, "
                    "output ONLY the concise final answer (short entity name, year, number, or phrase — no explanation)."
                ),
            })
            messages = self.context_manager.build_messages(state, system_prompt, agent=self)
            resp = self.call_llm(messages, state, tools=None)
            answer = _clean_final_answer(resp.content) or _clean_final_answer(state.candidate_answer)
            state.candidate_answer = answer
            state.log("answer", content=answer, tokens=resp.total_tokens, usage=resp.usage, forced_stop=True)

        return AgentRunResult(question=question, answer=answer, state=state)

    def _execute_tool(self, state: AgentState, tool_name: str, args: Dict[str, Any]) -> str:
        state.tool_calls += 1
        state.log("tool_call", tool=tool_name, args=args)

        sig = normalize_call_signature(tool_name, args)
        deduplicated = False
        if self.context_manager.is_structured and sig in state.seen_calls:
            deduplicated = True
            prev_summary = state.seen_calls[sig]
            result = ToolResult(
                text=(
                    f"[Deduplicated call] You already ran {tool_name}({args}) earlier "
                    f"({prev_summary}). Check [STRUCTURED SEARCH STATE] above, try a different "
                    f"query/document/page, or provide your concise final answer."
                ),
                error=False,
            )
            obs_text = result.text
        else:
            result = self.registry.execute(tool_name, args)

            if tool_name == "load_skill" and not result.error:
                skill_name = str(args.get("name", "")).strip()
                if skill_name and skill_name not in state.loaded_skills:
                    state.loaded_skills.append(skill_name)
                state.log("skill", content=skill_name, tool=tool_name)

            # Format observation before marking new hits visited so only previously-seen docs are tagged
            obs_text = self.context_manager.format_observation_for_history(state, tool_name, args, result)

            if result.doc_ids:
                state.mark_visited(result.doc_ids)

            if not result.error:
                first_line = (result.text.splitlines()[0] if result.text else "ok")[:120]
                state.seen_calls[sig] = first_line

        state.log(
            "observation",
            tool=tool_name,
            content=result.text,
            error=result.error,
            deduplicated=deduplicated,
        )
        if not deduplicated:
            self.context_manager.after_observation(state, tool_name, args, result, agent=self)
        return obs_text


def build_agent(
    retriever_mode: str = "fusion",
    use_skills: bool = True,
    force_skill: Optional[str] = None,
    use_context_manager: bool = True,
    enable_llm_summary: bool = config.CTX_ENABLE_LLM_SUMMARY,
    max_steps: int = config.MAX_STEPS,
    llm_client: Optional[LLMClient] = None,
) -> SearchAgent:
    """Assemble the search agent with the unified retrieval tool, supporting tools, skills, and context control."""
    from retrieval.base import build_retriever_and_store
    from tools.doc_tools import FindInDocTool, OpenDocTool
    from tools.search import SearchTool

    retriever, store = build_retriever_and_store(mode=retriever_mode)
    registry = ToolRegistry()
    registry.register(SearchTool(retriever))
    registry.register(OpenDocTool(store))
    registry.register(FindInDocTool(store))

    skill_library: Optional[SkillLibrary] = None
    if use_skills:
        skill_library = SkillLibrary(config.SKILLS_DIR)
        registry.register(LoadSkillTool(skill_library))

    llm = llm_client or LLMClient()
    if use_context_manager:
        ctx: ContextManager = StructuredContextManager(
            max_steps=max_steps,
            enable_llm_summary=enable_llm_summary,
        )
    else:
        ctx = ContextManager()

    return SearchAgent(
        llm=llm,
        registry=registry,
        context_manager=ctx,
        skill_library=skill_library,
        max_steps=max_steps,
        use_skills=use_skills,
        force_skill=force_skill,
    )
