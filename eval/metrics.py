"""Evaluation metrics: normalized Exact Match (EM) and efficiency aggregation.

Normalized EM applies:
  - lowercasing
  - punctuation removal
  - English article removal (a, an, the)
  - whitespace normalization
"""
from __future__ import annotations

import re
import string
from typing import Iterable, List, Union


_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_ARTICLE_RE = re.compile(r"\b(a|an|the)\b")


def normalize_answer(text: str) -> str:
    s = str(text or "").lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLE_RE.sub(" ", s)
    return " ".join(s.split())


def exact_match(prediction: str, gold: Union[str, List[str]]) -> float:
    """Return 1.0 if normalized prediction matches gold (or any gold alias), else 0.0."""
    pred_norm = normalize_answer(prediction)
    golds = [gold] if isinstance(gold, str) else list(gold)
    return 1.0 if any(pred_norm == normalize_answer(g) for g in golds) else 0.0


def evaluate_predictions(records: Iterable[dict], gold_by_id: dict) -> dict:
    total = 0
    correct = 0.0
    total_tool_calls = 0
    total_tokens = 0
    for rec in records:
        qid = str(rec["id"])
        if qid not in gold_by_id:
            continue
        total += 1
        correct += exact_match(rec.get("answer", ""), gold_by_id[qid])
        total_tool_calls += int(rec.get("tool_calls", 0))
        total_tokens += int(rec.get("total_tokens", 0))
    em = (correct / total) if total else 0.0
    return {
        "num_questions": total,
        "em": round(em, 4),
        "em_percent": round(em * 100.0, 2),
        "total_tool_calls": total_tool_calls,
        "avg_tool_calls": round(total_tool_calls / total, 2) if total else 0.0,
        "total_tokens": total_tokens,
        "avg_tokens": round(total_tokens / total, 1) if total else 0.0,
    }
