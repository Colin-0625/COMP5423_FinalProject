"""Category 2: Encoder-based Dense Retrieval (BAAI/bge-base-en-v1.5 + FAISS)."""
from __future__ import annotations

import os
from typing import Optional

import config
from retrieval.base import DocStore
from retrieval.dense_base import FaissDenseRetriever, build_dense_index


DEFAULT_ENCODER_MODEL = os.environ.get("DENSE_ENCODER_MODEL", "BAAI/bge-base-en-v1.5")
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def build_encoder_dense_index(
    corpus_dir: Optional[str] = None,
    index_dir: Optional[str] = None,
    model_name: str = DEFAULT_ENCODER_MODEL,
    batch_size: int = 128,
) -> str:
    out_dir = index_dir or os.path.join(config.INDEX_DIR, "dense_encoder")
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


class DenseRetriever(FaissDenseRetriever):
    """Encoder-based dense retriever (BGE v1.5)."""

    def __init__(self, store: DocStore, index_dir: Optional[str] = None):
        super().__init__(
            store=store,
            index_dir=index_dir or os.path.join(config.INDEX_DIR, "dense_encoder"),
            query_prefix=BGE_QUERY_PREFIX,
            max_seq_length=512,
            use_fp16=True,
        )
