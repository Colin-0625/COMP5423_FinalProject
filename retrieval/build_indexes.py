"""CLI script to build all persistent indexes over the 100,195-document corpus.

Usage:
  python retrieval/build_indexes.py --step docstore
  python retrieval/build_indexes.py --step bm25
  python retrieval/build_indexes.py --step dense --batch-size 128
  python retrieval/build_indexes.py --step llm_dense --batch-size 32
  python retrieval/build_indexes.py --step all
"""
from __future__ import annotations

import argparse
import os
import sys
import time

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)

from retrieval.bm25_retriever import build_bm25_index
from retrieval.dense_retriever import DEFAULT_ENCODER_MODEL, build_encoder_dense_index
from retrieval.doc_store import build_sqlite_doc_store
from retrieval.llm_dense_retriever import DEFAULT_LLM_EMBED_MODEL, build_llm_dense_index


def main() -> None:
    p = argparse.ArgumentParser(description="Build BrowseComp-Plus DocStore and 3-category retrieval indexes.")
    p.add_argument(
        "--step",
        required=True,
        choices=["docstore", "bm25", "dense", "llm_dense", "all"],
        help="Which index step to build.",
    )
    p.add_argument("--corpus-dir", default=None, help="Override corpus parquet directory.")
    p.add_argument("--encoder-model", default=DEFAULT_ENCODER_MODEL)
    p.add_argument("--llm-model", default=DEFAULT_LLM_EMBED_MODEL)
    p.add_argument("--dense-batch-size", type=int, default=128)
    p.add_argument("--llm-batch-size", type=int, default=32)
    args = p.parse_args()

    steps = ["docstore", "bm25", "dense", "llm_dense"] if args.step == "all" else [args.step]
    for step in steps:
        t0 = time.time()
        print(f"\n========== Building step: {step} ==========")
        if step == "docstore":
            build_sqlite_doc_store(corpus_dir=args.corpus_dir)
        elif step == "bm25":
            build_bm25_index(corpus_dir=args.corpus_dir)
        elif step == "dense":
            build_encoder_dense_index(
                corpus_dir=args.corpus_dir,
                model_name=args.encoder_model,
                batch_size=args.dense_batch_size,
            )
        elif step == "llm_dense":
            build_llm_dense_index(
                corpus_dir=args.corpus_dir,
                model_name=args.llm_model,
                batch_size=args.llm_batch_size,
            )
        print(f"========== Finished {step} in {time.time() - t0:.1f}s ==========")


if __name__ == "__main__":
    main()
