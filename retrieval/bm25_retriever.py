"""Category 1: Sparse Retrieval using BM25S over the full 100,195-document collection."""
from __future__ import annotations

import json
import os
import threading
from typing import List, Optional

import config
from agent.types import SearchHit
from data.loader import iter_corpus
from retrieval.base import DocStore, Retriever
from retrieval.doc_store import make_snippet


def build_bm25_index(
    corpus_dir: Optional[str] = None,
    index_dir: Optional[str] = None,
) -> str:
    """Build and persist a BM25S index over all documents."""
    import bm25s

    out_dir = index_dir or os.path.join(config.INDEX_DIR, "bm25")
    os.makedirs(out_dir, exist_ok=True)

    print("[bm25] loading corpus texts...")
    doc_ids: List[str] = []
    texts: List[str] = []
    for doc in iter_corpus(corpus_dir):
        doc_ids.append(doc["doc_id"])
        texts.append(doc["text"])
    print(f"[bm25] tokenizing {len(texts)} documents...")

    corpus_tokens = bm25s.tokenize(texts, stopwords="en", show_progress=True)
    retriever = bm25s.BM25(method="lucene")
    retriever.index(corpus_tokens, show_progress=True)
    retriever.save(out_dir)

    with open(os.path.join(out_dir, "doc_ids.json"), "w", encoding="utf-8") as f:
        json.dump(doc_ids, f)

    print(f"[bm25] saved BM25S index for {len(doc_ids)} documents to {out_dir}")
    return out_dir


class BM25Retriever(Retriever):
    """Sparse BM25S retriever backed by mmap index + SQLiteDocStore for snippets."""

    def __init__(self, store: DocStore, index_dir: Optional[str] = None):
        import bm25s

        self.store = store
        self.index_dir = index_dir or os.path.join(config.INDEX_DIR, "bm25")
        ids_path = os.path.join(self.index_dir, "doc_ids.json")
        if not os.path.isfile(ids_path):
            raise FileNotFoundError(
                f"BM25 index not found in {self.index_dir}. Run: python retrieval/build_indexes.py --step bm25"
            )
        with open(ids_path, "r", encoding="utf-8") as f:
            self.doc_ids: List[str] = json.load(f)
        self._retriever = bm25s.BM25.load(self.index_dir, mmap=True)
        self._lock = threading.Lock()

    def search(self, query: str, k: int = 5) -> List[SearchHit]:
        import bm25s

        q_tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
        k_eff = min(max(1, int(k)), len(self.doc_ids))
        with self._lock:
            results, scores = self._retriever.retrieve(q_tokens, k=k_eff, n_threads=1, show_progress=False)

        hits: List[SearchHit] = []
        for idx, score in zip(results[0], scores[0]):
            if float(score) <= 0.0:
                continue
            did = self.doc_ids[int(idx)]
            doc = self.store.get(did)
            snippet = make_snippet(doc["text"], query) if doc else ""
            hits.append(SearchHit(doc_id=did, score=float(score), snippet=snippet))
        return hits
