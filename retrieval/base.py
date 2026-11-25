"""Abstract interfaces and factory for retrievers and the document store."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import config
from agent.types import SearchHit


class Retriever(ABC):
    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[SearchHit]:
        """Return top-k hits ranked best-first. Implementations must be deterministic and thread-safe."""


class DocStore(ABC):
    @abstractmethod
    def get(self, doc_id: str) -> Optional[Dict[str, str]]:
        """Return {"doc_id", "text", "url"} for doc_id, or None if missing."""


def build_retriever_and_store(mode: str = "fusion") -> Tuple[Retriever, DocStore]:
    """Load the SQLite document store and requested retrieval backend.

    Args:
        mode: Retrieval mode:
            - "bm25"      : Sparse retrieval (BM25S)
            - "dense"     : Encoder-based dense retrieval (BAAI/bge-base-en-v1.5 + FAISS)
            - "llm_dense" : Instruction-aware LLM dense retrieval (Qwen/Qwen3-Embedding-0.6B + FAISS)
            - "fusion2"   : RRF fusion of BM25S + BGE
            - "fusion"    : RRF fusion of all available retrieval categories
    """
    from retrieval.bm25_retriever import BM25Retriever
    from retrieval.dense_retriever import DenseRetriever
    from retrieval.doc_store import SQLiteDocStore
    from retrieval.fusion import FusionRetriever
    from retrieval.llm_dense_retriever import LLMDenseRetriever

    store = SQLiteDocStore()
    mode = (mode or "fusion").lower()

    if mode == "bm25":
        return BM25Retriever(store), store
    if mode == "dense":
        return DenseRetriever(store), store
    if mode == "llm_dense":
        return LLMDenseRetriever(store), store
    if mode == "fusion2":
        subs = [
            ("bm25", BM25Retriever(store), 1.0),
            ("dense", DenseRetriever(store), 1.0),
        ]
        return FusionRetriever(subs, store), store
    if mode == "fusion":
        subs = []
        if os.path.isfile(os.path.join(config.INDEX_DIR, "bm25", "doc_ids.json")):
            subs.append(("bm25", BM25Retriever(store), 1.0))
        if os.path.isfile(os.path.join(config.INDEX_DIR, "dense_encoder", "index.faiss")):
            subs.append(("dense", DenseRetriever(store), 1.0))
        if os.path.isfile(os.path.join(config.INDEX_DIR, "dense_llm", "index.faiss")):
            subs.append(("llm_dense", LLMDenseRetriever(store), 1.2))
        if not subs:
            raise FileNotFoundError(
                f"No built retrieval indexes found in {config.INDEX_DIR}. "
                "Run: python retrieval/build_indexes.py --step all"
            )
        if len(subs) == 1:
            return subs[0][1], store
        return FusionRetriever(subs, store), store

    raise ValueError(f"Unknown retrieval mode: {mode!r}")
