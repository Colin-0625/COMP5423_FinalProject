"""Reciprocal Rank Fusion (RRF) combining multiple retrieval categories behind ONE Retriever.

Because SearchTool wraps this single Retriever, the agent sees EXACTLY ONE retrieval tool,
and internal sub-retrievals do not count as extra agent tool_calls.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from agent.types import SearchHit
from retrieval.base import DocStore, Retriever
from retrieval.doc_store import make_snippet


class FusionRetriever(Retriever):
    """Weighted Reciprocal Rank Fusion over 2 or 3 retrieval categories."""

    def __init__(
        self,
        retrievers: Sequence[Tuple[str, Retriever, float]],
        store: DocStore,
        k_rrf: int = 60,
        candidate_multiplier: int = 4,
        min_candidates: int = 20,
    ):
        if not retrievers:
            raise ValueError("FusionRetriever requires at least one sub-retriever")
        self.retrievers = list(retrievers)
        self.store = store
        self.k_rrf = int(k_rrf)
        self.candidate_multiplier = int(candidate_multiplier)
        self.min_candidates = int(min_candidates)

    def search(self, query: str, k: int = 5) -> List[SearchHit]:
        cand_k = max(int(k) * self.candidate_multiplier, self.min_candidates)
        rrf_scores: Dict[str, float] = {}
        snippets: Dict[str, str] = {}

        for _name, ret, weight in self.retrievers:
            hits = ret.search(query, k=cand_k)
            for rank, hit in enumerate(hits, start=1):
                rrf_scores[hit.doc_id] = rrf_scores.get(hit.doc_id, 0.0) + float(weight) / (self.k_rrf + rank)
                if hit.doc_id not in snippets and hit.snippet:
                    snippets[hit.doc_id] = hit.snippet

        ranked = sorted(rrf_scores.items(), key=lambda kv: -kv[1])[: int(k)]
        out: List[SearchHit] = []
        for did, score in ranked:
            snip = snippets.get(did, "")
            if not snip:
                doc = self.store.get(did)
                snip = make_snippet(doc["text"], query) if doc else ""
            out.append(SearchHit(doc_id=did, score=float(score), snippet=snip))
        return out
