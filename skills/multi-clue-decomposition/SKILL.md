---
name: multi-clue-decomposition
description: Use when a question combines 3 or more indirect, multi-hop, or constraint-heavy clues (e.g., birth period, institution, award, competition, or linked historical events) that cannot be answered with a single broad search query.
---

# Multi-Clue Question Decomposition

Follow this procedure when a question stacks multiple indirect constraints describing an unnamed target entity or fact.

## 1. Extract and Rank Clues
Break the question into atomic factual constraints and classify each clue by search selectivity (see `references/clue-priority.md` if needed):
- **High-selectivity anchors (search first)**: unusual numbers/years, specific awards, rare diseases/molecules, named competitions, unique geographic or institutional pairings.
- **Medium-selectivity filters (verify second)**: broad decades, nationality/continent, general professions, approximate time ranges.
- **Low-selectivity prose (do NOT search directly)**: subjective phrasing ("known for their accessibility", "had a lot of friends") or heavily paraphrased riddles.

## 2. Anchor-First Query Strategy
- NEVER paste the entire long question into `search`.
- Issue a short, 4–8 keyword query combining the **two most distinctive concrete anchors**.
- If the question hides a specific term behind a paraphrase (e.g., "the disease that shares the same name as zodiac sign Cancer" -> `cancer`; "first decade of the 1900s" -> `1900s` or `1908`), translate the paraphrase into concrete Wikipedia/article keywords before searching.

## 3. Candidate Verification Loop
Once a plausible candidate entity appears in a search snippet:
1. Open the most promising document with `open_doc` or jump directly to specific clues with `find_in_doc`.
2. Check **every** concrete constraint from the question against the document text (not just one matching clue).
3. If one clue contradicts the candidate, discard it immediately and pivot to the next candidate or a secondary anchor clue.

## 4. Concise Final Answer
When all key clues align on one verified entity, emit `final_answer` using ONLY the exact requested short answer (e.g., full canonical name, exact year, or short title) without explanatory sentences.
