r"""Unit tests for structured state, context pruning, deduplication, and document pagination.

Run:
  python -m unittest tests/test_context_manager.py -v
"""
from __future__ import annotations

import json
import os
import re
import sys
import unittest
from typing import Any, Dict, List, Optional

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)

import config
from agent.context import ContextManager, StructuredContextManager
from agent.llm_client import LLMClient, LLMResponse
from agent.loop import SearchAgent
from agent.types import SearchHit
from retrieval.base import DocStore, Retriever
from tools.base import ToolRegistry
from tools.doc_tools import FindInDocTool, OpenDocTool
from tools.search import SearchTool
from tools.skill_tool import LoadSkillTool, SkillLibrary


class _LongDocStore(DocStore):
    def __init__(self) -> None:
        filler = "Background paragraph describing general historical context and miscellaneous facts. " * 45
        self.docs = {
            f"doc_{i}": {
                "doc_id": f"doc_{i}",
                "url": f"http://example.com/doc_{i}",
                "text": (
                    f"Header for doc_{i}.\n\n\n\n"
                    + filler
                    + f"\n\nKey fact {i}: Folke Karl Skoog discovered kinetin in 1955 at Wisconsin (doc_{i}).\n\n"
                    + filler
                    + f"\n\nDeep target marker TOKEN_PAGE_CHECK_{i} appears near the end of doc_{i}.\n"
                ),
            }
            for i in range(6)
        }

    def get(self, doc_id: str) -> Optional[Dict[str, str]]:
        return self.docs.get(doc_id)


class _DummyRetriever(Retriever):
    def __init__(self, store: _LongDocStore) -> None:
        self.store = store

    def search(self, query: str, k: int = 5) -> List[SearchHit]:
        hits = []
        for idx, (did, _d) in enumerate(self.store.docs.items()):
            hits.append(SearchHit(did, 10.0 - idx, f"Folke Karl Skoog snippet from {did}"))
        return hits[:k]


class _ScriptedLLM(LLMClient):
    """Plays back a fixed sequence of native tool-calling or final-answer responses."""

    def __init__(self, script: List[Dict[str, Any]]) -> None:
        super().__init__(api_key="test", base_url="http://test", model="scripted")
        self.script = list(script)
        self.idx = 0
        self.summary_calls = 0
        self.seen_messages: List[List[Dict[str, Any]]] = []

    def chat(self, messages: List[Dict[str, Any]], tools=None, **kw) -> LLMResponse:
        self.seen_messages.append(messages)
        if "working notes" in str(messages[0].get("content", "")).lower():
            self.summary_calls += 1
            return LLMResponse("Summary: Folke Karl Skoog discovered kinetin in 1955.", total_tokens=55)
        if self.idx < len(self.script):
            step = self.script[self.idx]
            self.idx += 1
        else:
            step = {"answer": "Folke Karl Skoog"}

        if "tool" in step:
            tc = {
                "id": f"call_{self.idx}",
                "type": "function",
                "function": {
                    "name": step["tool"],
                    "arguments": json.dumps(step.get("args", {})),
                },
            }
            return LLMResponse(content=step.get("thought", ""), total_tokens=100, tool_calls=[tc])
        return LLMResponse(content=str(step.get("answer", "")), total_tokens=100)


class TestContextManagementAndTools(unittest.TestCase):
    def _make_agent(self, script: List[Dict[str, Any]], use_ctx: bool = True, enable_llm_summary: bool = False):
        store = _LongDocStore()
        registry = ToolRegistry()
        registry.register(SearchTool(_DummyRetriever(store)))
        registry.register(OpenDocTool(store))
        registry.register(FindInDocTool(store))
        skill_lib = SkillLibrary(config.SKILLS_DIR)
        registry.register(LoadSkillTool(skill_lib))
        llm = _ScriptedLLM(script)
        ctx = (
            StructuredContextManager(
                recent_turns=2,
                max_evidence=6,
                max_context_chars=6000,
                enable_llm_summary=enable_llm_summary,
                max_steps=12,
            )
            if use_ctx
            else ContextManager()
        )
        agent = SearchAgent(
            llm=llm,
            registry=registry,
            context_manager=ctx,
            skill_library=skill_lib,
            max_steps=12,
            use_skills=True,
        )
        return agent, llm, store

    def test_1_state_independence_pruning_and_tool_call_pairing(self) -> None:
        script = [
            {"thought": "search 1", "tool": "search", "args": {"query": "Folke Karl Skoog kinetin", "k": 4}},
            {"thought": "open 0", "tool": "open_doc", "args": {"doc_id": "doc_0", "page": 0}},
            {"thought": "open 1", "tool": "open_doc", "args": {"doc_id": "doc_1", "page": 0}},
            {"thought": "open 2", "tool": "open_doc", "args": {"doc_id": "doc_2", "page": 0}},
            {"thought": "open 3", "tool": "open_doc", "args": {"doc_id": "doc_3", "page": 0}},
            {"thought": "open 4", "tool": "open_doc", "args": {"doc_id": "doc_4", "page": 0}},
            {"answer": "Folke Karl Skoog"},
        ]
        e4_agent, _, _ = self._make_agent(script, use_ctx=False)
        e5_agent, e5_llm, _ = self._make_agent(script, use_ctx=True)

        res_e4 = e4_agent.run("Who discovered kinetin?")
        res_e5 = e5_agent.run("Who discovered kinetin?")

        st = res_e5.state
        self.assertEqual(res_e5.answer, "Folke Karl Skoog")
        self.assertIn("doc_0", st.visited_docs)
        self.assertIn("doc_4", st.opened_pages)
        self.assertIn("Folke Karl Skoog kinetin", st.sub_queries)
        self.assertGreater(len(st.evidence), 0)
        self.assertGreater(len(st.messages), 8)

        last_e4 = res_e4.state.context_stats[-1]
        last_e5 = st.context_stats[-1]
        self.assertGreater(last_e4["sent_chars"], last_e5["sent_chars"])
        self.assertGreater(last_e5["raw_chars"], last_e5["sent_chars"])
        self.assertGreater(last_e5["dropped_msgs"], 0)
        self.assertLessEqual(last_e5["sent_chars"], 6500)

        # Protocol invariant: every role=="tool" message sent to the API must follow a matching assistant tool_call
        for msgs in e5_llm.seen_messages:
            pending_ids = set()
            for m in msgs:
                if m["role"] == "assistant" and m.get("tool_calls"):
                    pending_ids = {tc["id"] for tc in m["tool_calls"]}
                elif m["role"] == "tool":
                    self.assertIn(m["tool_call_id"], pending_ids)

    def test_3_runtime_skill_not_in_base_prompt_and_preserved_after_pruning(self) -> None:
        script = [
            {"thought": "load skill", "tool": "load_skill", "args": {"name": "multi-clue-decomposition"}},
            {"thought": "open 0", "tool": "open_doc", "args": {"doc_id": "doc_0", "page": 0}},
            {"thought": "open 1", "tool": "open_doc", "args": {"doc_id": "doc_1", "page": 0}},
            {"thought": "open 2", "tool": "open_doc", "args": {"doc_id": "doc_2", "page": 0}},
            {"thought": "open 3", "tool": "open_doc", "args": {"doc_id": "doc_3", "page": 0}},
            {"answer": "Folke Karl Skoog"},
        ]
        agent, _, _ = self._make_agent(script, use_ctx=True)
        base_prompt = agent.build_system_prompt()
        self.assertIn("multi-clue-decomposition", base_prompt)
        self.assertNotIn("NEVER paste the entire long question into", base_prompt)

        res = agent.run("Who discovered kinetin?")
        self.assertIn("multi-clue-decomposition", res.state.loaded_skills)
        self.assertIn("multi-clue-decomposition", res.state.loaded_skill_bodies)

        final_msgs = agent.context_manager.build_messages(res.state, base_prompt, agent=agent)
        combined = "\n".join(str(m.get("content") or "") for m in final_msgs)
        self.assertIn("[ACTIVE SKILL: multi-clue-decomposition]", combined)
        self.assertIn("NEVER paste the entire long question into", combined)

    def test_4_duplicate_call_counts_tool_call_but_saves_context(self) -> None:
        script = [
            {"thought": "open 0", "tool": "open_doc", "args": {"doc_id": "doc_0", "page": 0}},
            {"thought": "open 0 again", "tool": "open_doc", "args": {"doc_id": "doc_0", "page": 0}},
            {"answer": "Folke Karl Skoog"},
        ]
        agent, _, _ = self._make_agent(script, use_ctx=True)
        res = agent.run("Who discovered kinetin?")

        self.assertEqual(res.state.tool_calls, 2)
        obs_steps = [s for s in res.state.trajectory if s.get("type") == "observation"]
        self.assertEqual(len(obs_steps), 2)
        self.assertFalse(obs_steps[0]["deduplicated"])
        self.assertTrue(obs_steps[1]["deduplicated"])
        self.assertLess(len(obs_steps[1]["content"]), len(obs_steps[0]["content"]) // 3)

    def test_5_optional_llm_summary_counts_tokens(self) -> None:
        script = [
            {"thought": "open 0", "tool": "open_doc", "args": {"doc_id": "doc_0", "page": 0}},
            {"thought": "open 1", "tool": "open_doc", "args": {"doc_id": "doc_1", "page": 0}},
            {"thought": "open 2", "tool": "open_doc", "args": {"doc_id": "doc_2", "page": 0}},
            {"thought": "open 3", "tool": "open_doc", "args": {"doc_id": "doc_3", "page": 0}},
            {"answer": "Folke Karl Skoog"},
        ]
        agent, llm, _ = self._make_agent(script, use_ctx=True, enable_llm_summary=True)
        res = agent.run("Who discovered kinetin?")

        self.assertGreaterEqual(llm.summary_calls, 1)
        self.assertTrue(res.state.notes)
        expected_tokens = 5 * 100 + llm.summary_calls * 55
        self.assertEqual(res.state.total_tokens, expected_tokens)

    def test_6_find_in_doc_and_open_doc_page_alignment(self) -> None:
        store = _LongDocStore()
        finder = FindInDocTool(store)
        opener = OpenDocTool(store)

        res_find = finder.run(doc_id="doc_0", keyword="TOKEN_PAGE_CHECK_0")
        self.assertFalse(res_find.error)
        m = re.search(r"\(page (\d+)\)", res_find.text)
        self.assertIsNotNone(m, f"Expected page hint in: {res_find.text}")
        page_idx = int(m.group(1))
        self.assertGreater(page_idx, 0)

        res_open = opener.run(doc_id="doc_0", page=page_idx)
        self.assertFalse(res_open.error)
        self.assertIn("TOKEN_PAGE_CHECK_0", res_open.text)


if __name__ == "__main__":
    unittest.main()
