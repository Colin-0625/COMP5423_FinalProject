"""Unit tests for normalized EM metrics, Agent Skills validation, resource loading,
and automatic skill selection across distinct scenarios.

Run:
  python -m unittest tests/test_metrics_and_skills.py -v
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from typing import Dict, List, Optional

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)

import config
from agent.context import StructuredContextManager
from agent.llm_client import LLMClient, LLMResponse
from agent.loop import SearchAgent
from agent.types import SearchHit
from eval.metrics import evaluate_predictions, exact_match, normalize_answer
from retrieval.base import DocStore, Retriever
from skills.validate_skills import validate_skills
from tools.base import ToolRegistry
from tools.doc_tools import FindInDocTool, OpenDocTool
from tools.search import SearchTool
from tools.skill_tool import LoadSkillTool, SkillLibrary


class _InMemoryDocStore(DocStore):
    DOCS = {
        "d1": {
            "doc_id": "d1",
            "url": "http://example.com/1",
            "text": "The Eiffel Tower is a wrought-iron lattice tower in Paris, completed in 1889.",
        },
    }

    def get(self, doc_id: str) -> Optional[Dict[str, str]]:
        return self.DOCS.get(doc_id)


class _InMemoryRetriever(Retriever):
    def __init__(self, store: _InMemoryDocStore) -> None:
        self.store = store

    def search(self, query: str, k: int = 5) -> List[SearchHit]:
        q = set(query.lower().split())
        hits = []
        for d in self.store.DOCS.values():
            score = len(q & set(d["text"].lower().split()))
            if score:
                hits.append(SearchHit(d["doc_id"], float(score), d["text"][:300]))
        return sorted(hits, key=lambda h: -h.score)[:k]


class TestNormalizedEM(unittest.TestCase):
    def test_normalize_case_punct_articles_whitespace(self) -> None:
        self.assertEqual(normalize_answer("  The  Eiffel Tower! "), "eiffel tower")
        self.assertEqual(normalize_answer("A scientist's book, an apple."), "scientists book apple")
        self.assertEqual(normalize_answer("Folke Karl Skoog."), "folke karl skoog")

    def test_article_removal_respects_word_boundaries(self) -> None:
        self.assertEqual(normalize_answer("The Theater of Anemia"), "theater of anemia")
        self.assertEqual(normalize_answer("Another Theory"), "another theory")

    def test_exact_match_single_and_aliases(self) -> None:
        self.assertEqual(exact_match("The United States.", "United States"), 1.0)
        self.assertEqual(exact_match("1,889", "1889"), 1.0)
        self.assertEqual(exact_match("Paris", ["Lyon", "The Paris!"]), 1.0)
        self.assertEqual(exact_match("London", "Paris"), 0.0)
        self.assertEqual(exact_match("", "1889"), 0.0)

    def test_evaluate_predictions_summary(self) -> None:
        preds = [
            {"id": "1", "answer": "The 1889!", "tool_calls": 2, "total_tokens": 300},
            {"id": "2", "answer": "Wrong", "tool_calls": 4, "total_tokens": 700},
        ]
        gold = {"1": "1889", "2": "Gustave Eiffel"}
        res = evaluate_predictions(preds, gold)
        self.assertEqual(res["num_questions"], 2)
        self.assertEqual(res["em"], 0.5)
        self.assertEqual(res["em_percent"], 50.0)
        self.assertEqual(res["total_tool_calls"], 6)
        self.assertEqual(res["avg_tool_calls"], 3.0)
        self.assertEqual(res["total_tokens"], 1000)
        self.assertEqual(res["avg_tokens"], 500.0)


class _AutoSkillTestLLM(LLMClient):
    """Deterministic test stub simulating four distinct automatic skill-selection decisions."""

    def __init__(self) -> None:
        super().__init__(api_key="test", base_url="http://test", model="test-stub")
        self._call_counter = 0

    def _tc(self, name: str, args: Dict) -> List[Dict]:
        self._call_counter += 1
        return [{
            "id": f"call_{self._call_counter}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }]

    def chat(self, messages: List[Dict], tools=None, **kw) -> LLMResponse:
        full_text = "\n".join(str(m.get("content") or "") for m in messages)
        tool_obs = [str(m.get("content") or "") for m in messages if m.get("role") == "tool"]

        if "three indirect clues" in full_text.lower() and "[ACTIVE SKILL: multi-clue-decomposition]" not in full_text:
            return LLMResponse(
                content="Multiple indirect clues detected; loading multi-clue-decomposition skill.",
                total_tokens=90,
                tool_calls=self._tc("load_skill", {"name": "multi-clue-decomposition"}),
            )

        if "two engineers named" in full_text.lower() and "[ACTIVE SKILL: entity-disambiguation]" not in full_text:
            return LLMResponse(
                content="Competing similarly named entities; loading entity-disambiguation skill.",
                total_tokens=90,
                tool_calls=self._tc("load_skill", {"name": "entity-disambiguation"}),
            )

        if "obscure unindexed phrase" in full_text.lower():
            if not tool_obs:
                return LLMResponse(
                    content="Trying initial search.",
                    total_tokens=80,
                    tool_calls=self._tc("search", {"query": "xyzzy_nonexistent_term_999"}),
                )
            if "No results." in tool_obs[-1] and "[ACTIVE SKILL: search-recovery]" not in full_text:
                return LLMResponse(
                    content="Search returned No results; loading search-recovery skill to pivot.",
                    total_tokens=95,
                    tool_calls=self._tc("load_skill", {"name": "search-recovery"}),
                )

        if not any("doc_id=" in o for o in tool_obs):
            return LLMResponse(
                content="Searching for Eiffel Tower.",
                total_tokens=80,
                tool_calls=self._tc("search", {"query": "Eiffel Tower", "k": 2}),
            )
        return LLMResponse(content="1889", total_tokens=70)


class TestSkillsAndAutomaticSelection(unittest.TestCase):
    def test_validate_skills_passes_all_spec_checks(self) -> None:
        errors = validate_skills(config.SKILLS_DIR)
        self.assertEqual(errors, [], f"Skill validation errors: {errors}")

    def test_skill_library_and_resource_loading(self) -> None:
        lib = SkillLibrary(config.SKILLS_DIR)
        self.assertGreaterEqual(len(lib.names()), 3)
        self.assertIn("multi-clue-decomposition", lib.names())
        self.assertIn("entity-disambiguation", lib.names())
        self.assertIn("search-recovery", lib.names())

        tool = LoadSkillTool(lib)
        r_main = tool.run(name="multi-clue-decomposition")
        self.assertFalse(r_main.error)
        self.assertIn("references/clue-priority.md", r_main.text)

        r_ref = tool.run(name="multi-clue-decomposition", resource="references/clue-priority.md")
        self.assertFalse(r_ref.error)
        self.assertIn("Clue Priority Quick Reference", r_ref.text)

        # Ensure path traversal outside the skill directory is rejected
        r_bad = tool.run(name="multi-clue-decomposition", resource="../../config.py")
        self.assertTrue(r_bad.error)

    def test_four_distinct_automatic_skill_decisions(self) -> None:
        store = _InMemoryDocStore()
        registry = ToolRegistry()
        registry.register(SearchTool(_InMemoryRetriever(store)))
        registry.register(OpenDocTool(store))
        registry.register(FindInDocTool(store))
        lib = SkillLibrary(config.SKILLS_DIR)
        registry.register(LoadSkillTool(lib))

        agent = SearchAgent(
            llm=_AutoSkillTestLLM(),
            registry=registry,
            context_manager=StructuredContextManager(max_steps=8),
            skill_library=lib,
            max_steps=8,
            use_skills=True,
        )

        r1 = agent.run("A question with three indirect clues about the Eiffel Tower.")
        r2 = agent.run("Between two engineers named Eiffel, which year was the tower completed?")
        r3 = agent.run("Find the completion year using an obscure unindexed phrase.")
        r4 = agent.run("In what year was the Eiffel Tower completed?")

        self.assertEqual(r1.state.loaded_skills, ["multi-clue-decomposition"])
        self.assertEqual(r2.state.loaded_skills, ["entity-disambiguation"])
        self.assertEqual(r3.state.loaded_skills, ["search-recovery"])
        self.assertEqual(r4.state.loaded_skills, [])

        self.assertEqual(r1.state.tool_calls, 2)
        self.assertEqual(r4.state.tool_calls, 1)


if __name__ == "__main__":
    unittest.main()
