"""Thread-safe SQLite-backed DocStore over the 100,195-document BrowseComp-Plus collection,
plus query-aligned snippet extraction.
"""
from __future__ import annotations

import os
import re
import sqlite3
import threading
from typing import Dict, Optional

import config
from data.loader import iter_corpus
from retrieval.base import DocStore


_TOKEN_RE = re.compile(r"[A-Za-z0-9]{2,}")
_STOPWORDS = {
    "the", "and", "for", "that", "with", "from", "this", "was", "were", "are",
    "have", "has", "had", "not", "but", "what", "which", "who", "when", "where",
    "how", "why", "into", "about", "their", "they", "them", "been", "also",
}


def make_snippet(text: str, query: str = "", max_chars: int = config.SNIPPET_CHARS) -> str:
    """Return a clean snippet centered around the best query term match."""
    if not text:
        return ""
    # Strip YAML-like frontmatter header if present (e.g., ---\ntitle: ...\n---\n)
    body = text
    m_fm = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, flags=re.S)
    title_prefix = ""
    if m_fm:
        for line in m_fm.group(1).splitlines():
            if line.lower().startswith("title:"):
                title_prefix = line.split(":", 1)[1].strip() + " | "
                break
        body = text[m_fm.end():]

    clean_body = " ".join(body.split())
    avail = max(80, max_chars - len(title_prefix))
    if not query or len(clean_body) <= avail:
        return (title_prefix + clean_body[:avail]).strip()

    terms = [t.lower() for t in _TOKEN_RE.findall(query) if t.lower() not in _STOPWORDS]
    low = clean_body.lower()
    best_pos = -1
    for term in sorted(set(terms), key=len, reverse=True):
        pos = low.find(term)
        if pos != -1:
            best_pos = pos
            break

    if best_pos == -1:
        return (title_prefix + clean_body[:avail]).strip()

    start = max(0, best_pos - avail // 3)
    end = min(len(clean_body), start + avail)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(clean_body) else ""
    return (title_prefix + prefix + clean_body[start:end].strip() + suffix).strip()


def build_sqlite_doc_store(
    corpus_dir: Optional[str] = None,
    db_path: Optional[str] = None,
    batch_size: int = 2000,
) -> str:
    """Build SQLite DocStore from all Parquet files in corpus_dir."""
    db_path = db_path or os.path.join(config.INDEX_DIR, "docs.sqlite")
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    tmp_path = db_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    conn = sqlite3.connect(tmp_path)
    cur = conn.cursor()
    cur.execute("PRAGMA synchronous = OFF")
    cur.execute("PRAGMA journal_mode = MEMORY")
    cur.execute("CREATE TABLE docs (doc_id TEXT PRIMARY KEY, text TEXT, url TEXT)")

    buf = []
    count = 0
    for doc in iter_corpus(corpus_dir):
        buf.append((doc["doc_id"], doc["text"], doc["url"]))
        if len(buf) >= batch_size:
            cur.executemany("INSERT OR REPLACE INTO docs (doc_id, text, url) VALUES (?, ?, ?)", buf)
            conn.commit()
            count += len(buf)
            buf.clear()
            if count % 20000 == 0:
                print(f"[docstore] inserted {count} docs...")
    if buf:
        cur.executemany("INSERT OR REPLACE INTO docs (doc_id, text, url) VALUES (?, ?, ?)", buf)
        conn.commit()
        count += len(buf)

    cur.execute("PRAGMA journal_mode = DELETE")
    cur.close()
    conn.close()
    if os.path.exists(db_path):
        os.remove(db_path)
    import time
    for _ in range(5):
        try:
            os.replace(tmp_path, db_path)
            break
        except PermissionError:
            time.sleep(1.0)
    else:
        os.replace(tmp_path, db_path)
    print(f"[docstore] built {db_path} with {count} documents.")
    return db_path


class SQLiteDocStore(DocStore):
    """Read-only, thread-safe SQLite DocStore."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.path.join(config.INDEX_DIR, "docs.sqlite")
        if not os.path.isfile(self.db_path):
            raise FileNotFoundError(
                f"SQLite DocStore not found at {self.db_path}. Run: python retrieval/build_indexes.py --step docstore"
            )
        self._local = threading.local()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("PRAGMA query_only = ON")
            self._local.conn = conn
        return conn

    def get(self, doc_id: str) -> Optional[Dict[str, str]]:
        row = self._conn().execute(
            "SELECT doc_id, text, url FROM docs WHERE doc_id = ?",
            (str(doc_id),),
        ).fetchone()
        if row is None:
            return None
        return {"doc_id": row[0], "text": row[1], "url": row[2]}

    def count(self) -> int:
        return int(self._conn().execute("SELECT COUNT(*) FROM docs").fetchone()[0])
