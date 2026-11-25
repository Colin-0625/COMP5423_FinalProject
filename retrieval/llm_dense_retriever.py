"""Category 3: Instruction-aware Dense Retrieval using an LLM-based Embedding Model
(Qwen/Qwen3-Embedding-0.6B + FAISS).
"""
from __future__ import annotations

import os
from typing import Optional

import config
from retrieval.base import DocStore
from retrieval.dense_base import FaissDenseRetriever, build_dense_index


DEFAULT_LLM_EMBED_MODEL = os.environ.get("LLM_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
DEFAULT_TASK_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


def build_llm_dense_index(
    corpus_dir: Optional[str] = None,
    index_dir: Optional[str] = None,
    model_name: str = DEFAULT_LLM_EMBED_MODEL,
    batch_size: int = 32,
) -> str:
    out_dir = index_dir or os.path.join(config.INDEX_DIR, "dense_llm")
    return build_dense_index(
        model_name=model_name,
        index_dir=out_dir,
        corpus_dir=corpus_dir,
        max_seq_length=512,
        max_doc_chars=2400,
        batch_size=batch_size,
        shard_size=10000,
        doc_prefix="",
        use_fp16=True,
    )


class LLMDenseRetriever(FaissDenseRetriever):
    """Instruction-aware LLM-based dense retriever (Qwen3-Embedding-0.6B)."""

    def __init__(
        self,
        store: DocStore,
        index_dir: Optional[str] = None,
        task_instruction: str = DEFAULT_TASK_INSTRUCTION,
    ):
        self.task_instruction = task_instruction
        super().__init__(
            store=store,
            index_dir=index_dir or os.path.join(config.INDEX_DIR, "dense_llm"),
            query_prefix="",
            max_seq_length=512,
            use_fp16=True,
        )

    def format_query(self, query: str) -> str:
        return f"Instruct: {self.task_instruction}\nQuery: {query}"
