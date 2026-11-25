"""Shared dense retrieval infrastructure (sharded FP16 GPU encoding + FAISS IndexFlatIP).

Used by:
  - Category 2: Encoder-based dense retrieval (BAAI/bge-base-en-v1.5)
  - Category 3: Instruction-aware LLM-based dense retrieval (Qwen/Qwen3-Embedding-0.6B)
"""
from __future__ import annotations

import json
import os
import threading
from typing import List, Optional

import numpy as np

import config
from agent.types import SearchHit
from data.loader import iter_corpus
from retrieval.base import DocStore, Retriever
from retrieval.doc_store import make_snippet


def _truncate_doc(text: str, max_chars: int) -> str:
    """Keep title + opening content within max_chars for dense encoding."""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def build_dense_index(
    model_name: str,
    index_dir: str,
    corpus_dir: Optional[str] = None,
    max_seq_length: int = 512,
    max_doc_chars: int = 2400,
    batch_size: int = 64,
    shard_size: int = 10000,
    doc_prefix: str = "",
    use_fp16: bool = True,
) -> str:
    """Encode all 100,195 documents in resumable shards and build a FAISS IndexFlatIP."""
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    os.makedirs(index_dir, exist_ok=True)
    shards_dir = os.path.join(index_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[dense:{model_name}] device={device} fp16={use_fp16 and device == 'cuda'} max_seq_len={max_seq_length}")

    model_kwargs = {}
    if use_fp16 and device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16

    model = SentenceTransformer(
        model_name,
        device=device,
        trust_remote_code=True,
        model_kwargs=model_kwargs,
    )
    model.max_seq_length = max_seq_length

    doc_ids: List[str] = []
    buf_texts: List[str] = []
    shard_idx = 0
    shard_paths: List[str] = []

    def _flush_shard(s_idx: int, texts: List[str]) -> str:
        sp = os.path.join(shards_dir, f"shard_{s_idx:04d}.npy")
        if os.path.isfile(sp):
            arr = np.load(sp, mmap_mode="r")
            if arr.shape[0] == len(texts):
                print(f"[dense:{model_name}] shard {s_idx:04d} cached ({len(texts)} docs), skipping.")
                return sp
        embs = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)
        np.save(sp, embs)
        print(f"[dense:{model_name}] saved shard {s_idx:04d} shape={embs.shape}")
        if device == "cuda":
            torch.cuda.empty_cache()
        return sp

    for doc in iter_corpus(corpus_dir):
        doc_ids.append(doc["doc_id"])
        txt = _truncate_doc(doc["text"], max_doc_chars)
        buf_texts.append(f"{doc_prefix}{txt}" if doc_prefix else txt)
        if len(buf_texts) >= shard_size:
            shard_paths.append(_flush_shard(shard_idx, buf_texts))
            buf_texts.clear()
            shard_idx += 1

    if buf_texts:
        shard_paths.append(_flush_shard(shard_idx, buf_texts))
        buf_texts.clear()

    # Build FAISS IndexFlatIP (exact cosine similarity on L2-normalized vectors)
    first = np.load(shard_paths[0])
    dim = int(first.shape[1])
    index = faiss.IndexFlatIP(dim)
    for sp in shard_paths:
        arr = np.load(sp).astype(np.float32)
        index.add(arr)

    faiss_path = os.path.join(index_dir, "index.faiss")
    faiss.write_index(index, faiss_path)

    with open(os.path.join(index_dir, "doc_ids.json"), "w", encoding="utf-8") as f:
        json.dump(doc_ids, f)

    meta = {
        "model_name": model_name,
        "num_docs": len(doc_ids),
        "dim": dim,
        "max_seq_length": max_seq_length,
        "max_doc_chars": max_doc_chars,
        "doc_prefix": doc_prefix,
    }
    with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[dense:{model_name}] built FAISS index ({len(doc_ids)} x {dim}) at {faiss_path}")
    return index_dir


class FaissDenseRetriever(Retriever):
    """Thread-safe FAISS + SentenceTransformer retriever."""

    def __init__(
        self,
        store: DocStore,
        index_dir: str,
        query_prefix: str = "",
        max_seq_length: int = 512,
        use_fp16: bool = True,
    ):
        import faiss
        import torch
        from sentence_transformers import SentenceTransformer

        self.store = store
        self.index_dir = index_dir
        self.query_prefix = query_prefix

        meta_path = os.path.join(index_dir, "meta.json")
        faiss_path = os.path.join(index_dir, "index.faiss")
        ids_path = os.path.join(index_dir, "doc_ids.json")
        if not (os.path.isfile(meta_path) and os.path.isfile(faiss_path) and os.path.isfile(ids_path)):
            raise FileNotFoundError(
                f"Dense index incomplete in {index_dir}. Run retrieval/build_indexes.py first."
            )

        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        with open(ids_path, "r", encoding="utf-8") as f:
            self.doc_ids: List[str] = json.load(f)

        self._index = faiss.read_index(faiss_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {"torch_dtype": torch.float16} if (use_fp16 and device == "cuda") else {}
        try:
            self._model = SentenceTransformer(
                self.meta["model_name"],
                device=device,
                trust_remote_code=True,
                model_kwargs=model_kwargs,
                local_files_only=True,
            )
        except Exception:
            self._model = SentenceTransformer(
                self.meta["model_name"],
                device=device,
                trust_remote_code=True,
                model_kwargs=model_kwargs,
            )
        self._model.max_seq_length = max_seq_length
        self._lock = threading.Lock()

    def format_query(self, query: str) -> str:
        return f"{self.query_prefix}{query}" if self.query_prefix else query

    def search(self, query: str, k: int = 5) -> List[SearchHit]:
        q_text = self.format_query(query)
        k_eff = min(max(1, int(k)), len(self.doc_ids))
        with self._lock:
            q_emb = self._model.encode(
                [q_text],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)
            scores, indices = self._index.search(q_emb, k_eff)

        hits: List[SearchHit] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            did = self.doc_ids[int(idx)]
            doc = self.store.get(did)
            snippet = make_snippet(doc["text"], query) if doc else ""
            hits.append(SearchHit(doc_id=did, score=float(score), snippet=snippet))
        return hits
