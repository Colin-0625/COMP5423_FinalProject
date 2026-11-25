---
name: search-recovery
description: Use mid-search when previous search calls returned No results, irrelevant/off-topic documents, or repeated documents without advancing toward the answer, and you need to reformulate or pivot your retrieval strategy.
---

# Search Failure Diagnosis & Query Recovery

Follow this procedure whenever recent `search` calls fail to surface relevant evidence.

## 1. Diagnose Why the Previous Query Failed
Check your `Sub-queries tried` in `[STRUCTURED SEARCH STATE]` and identify the failure pattern:
- **Over-constrained**: Too many keywords or quoted phrases forced an empty or noisy match.
- **Paraphrase mismatch**: The question describes a concept indirectly, while corpus documents use canonical encyclopedia/news wording.
- **Wrong entity angle**: The chosen anchor term is too generic or ambiguous in a 100k-document web corpus.

## 2. Apply One Distinct Recovery Move (Never Repeat Identical Queries)
Choose the recovery tactic that best fits the diagnosis:
1. **Simplify & strip modifiers**: Reduce the query to 3–5 core nouns/names; remove adjectives, filler verbs, and speculative dates.
2. **Translate paraphrases to canonical terms**: Replace indirect descriptions with direct domain terms, synonyms, or standard titles (e.g., "highest national recognition" -> `"National Medal"` or `"Order of"`; "different continent" -> explicit country/region names if inferable).
3. **Pivot to an unused clue**: Drop the failing clue completely and search using a *different* independent constraint from the question.
4. **Mine bridge terms from opened docs**: If a partially related document mentions a linked institution, colleague, event, or URL slug, run `find_in_doc` or search that bridge term next.

## 3. Budget Discipline
- If 2 consecutive recovery queries still yield no new leads, avoid looping: synthesize the strongest verified candidate from `[STRUCTURED SEARCH STATE]` or inspect the highest-scoring unvisited hit before deciding.
