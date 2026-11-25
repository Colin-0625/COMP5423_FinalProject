"""Quick smoke test for real retrieval modes on validation.jsonl questions.

Usage:
  python retrieval/smoke_test.py --mode bm25 --num 3
  python retrieval/smoke_test.py --mode fusion --num 3
"""
from __future__ import annotations

import argparse
import os
import sys
import time

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)

import config
from data.loader import load_questions
from retrieval.base import build_retriever_and_store


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="bm25", choices=["bm25", "dense", "llm_dense", "fusion2", "fusion"])
    p.add_argument("--num", type=int, default=3)
    p.add_argument("--k", type=int, default=3)
    args = p.parse_args()

    t0 = time.time()
    retriever, store = build_retriever_and_store(mode=args.mode)
    print(f"[smoke] loaded mode={args.mode} in {time.time() - t0:.2f}s")

    val_path = os.path.join(config.DATA_DIR, "validation.jsonl")
    questions = load_questions(val_path)[: args.num]

    for q in questions:
        ts = time.time()
        hits = retriever.search(q["question"], k=args.k)
        dt = (time.time() - ts) * 1000
        print(f"\nQ[{q['id']}] ({dt:.1f}ms) gold={q.get('answer')!r}")
        for i, h in enumerate(hits, start=1):
            print(f"  [{i}] doc_id={h.doc_id} score={h.score:.4f} snippet={h.snippet[:140]!r}")


if __name__ == "__main__":
    main()
