"""Dataset loading helpers for validation, test, and the 100,195-document collection."""
from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, Iterator, List, Optional

import config


def iter_corpus(corpus_dir: Optional[str] = None, batch_size: int = 2048) -> Iterator[Dict[str, str]]:
    """Stream {"doc_id", "text", "url"} from all BrowseComp-Plus Parquet files.

    Uses pyarrow batch iteration so peak RAM stays small over all 100,195 documents.
    """
    import pyarrow.parquet as pq

    cdir = corpus_dir or os.path.join(config.DATA_DIR, "corpus")
    files = sorted(glob.glob(os.path.join(cdir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {cdir}")

    for path in files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size, columns=["docid", "text", "url"]):
            docids = batch.column("docid").to_pylist()
            texts = batch.column("text").to_pylist()
            urls = batch.column("url").to_pylist()
            for did, txt, url in zip(docids, texts, urls):
                yield {
                    "doc_id": str(did),
                    "text": txt or "",
                    "url": url or "",
                }


def load_questions(path: str) -> List[Dict[str, Any]]:
    """Load validation or test questions from JSONL or JSON.

    Expected fields per record:
      - id (or query_id): str
      - question (or query): str
      - answer (validation only): str
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Question file not found: {path}")

    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return records

    if text.startswith("["):
        raw_items = json.loads(text)
    else:
        raw_items = [json.loads(line) for line in text.splitlines() if line.strip()]

    for idx, item in enumerate(raw_items):
        qid = str(item.get("id", item.get("query_id", idx)))
        question = str(item.get("question", item.get("query", ""))).strip()
        rec: Dict[str, Any] = {"id": qid, "question": question}
        if "answer" in item:
            rec["answer"] = item["answer"]
        records.append(rec)
    return records
