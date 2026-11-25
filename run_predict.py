"""Batch prediction runner producing test_prediction.jsonl in the required JSONL schema.

Supports multi-threaded execution (--workers), checkpoint resume (--resume),
and ablation switches (--retriever, --no-skills, --force-skill, --no-context, --llm-summary).

Example:
  python run_predict.py --input data_files/test.jsonl --output ../test_prediction.jsonl --workers 4 --resume
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading
from typing import Dict, List, Optional, Set

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

import config
from agent.loop import SearchAgent, build_agent
from data.loader import load_questions


def _load_existing_ids(output_path: str) -> Set[str]:
    done: Set[str] = set()
    if not os.path.isfile(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "id" in obj:
                    done.add(str(obj["id"]))
            except json.JSONDecodeError:
                continue
    return done


def run_batch(
    questions: List[Dict],
    output_path: str,
    agent: SearchAgent,
    workers: int = 1,
    resume: bool = False,
) -> List[Dict]:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    done_ids = _load_existing_ids(output_path) if resume else set()
    pending = [q for q in questions if str(q["id"]) not in done_ids]

    if not resume and os.path.exists(output_path):
        os.remove(output_path)

    print(f"[run_predict] total={len(questions)} already_done={len(done_ids)} pending={len(pending)} workers={workers}")
    lock = threading.Lock()
    results_written: List[Dict] = []

    def _solve_one(item: Dict) -> Optional[Dict]:
        qid, qtext = str(item["id"]), item["question"]
        try:
            res = agent.run(qtext)
            record = res.to_prediction_record(qid)
        except Exception as e:
            print(f"[ERROR] qid={qid}: {type(e).__name__}: {e}", file=sys.stderr)
            return None
        with lock:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            results_written.append(record)
            print(
                f"[OK] id={qid} calls={record['tool_calls']} tokens={record['total_tokens']} "
                f"ans={record['answer']!r}"
            )
        return record

    if workers <= 1:
        for item in pending:
            _solve_one(item)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_solve_one, pending))

    # Re-order output records to match input question order
    if os.path.isfile(output_path):
        by_id: Dict[str, Dict] = {}
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                by_id[str(obj["id"])] = obj
        ordered = [by_id[str(q["id"])] for q in questions if str(q["id"]) in by_id]
        with open(output_path, "w", encoding="utf-8") as f:
            for rec in ordered:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return ordered
    return results_written


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Skill-Augmented Search Agent on a question file.")
    p.add_argument(
        "--input",
        type=str,
        default=os.path.join(config.DATA_DIR, "test.jsonl"),
        help="Path to input question JSONL file (default: data_files/test.jsonl).",
    )
    p.add_argument("--output", type=str, default="test_prediction.jsonl", help="Output JSONL path.")
    p.add_argument("--limit", type=int, default=0, help="Run only the first N questions (0 = all).")
    p.add_argument("--workers", type=int, default=1, help="Number of concurrent worker threads.")
    p.add_argument("--resume", action="store_true", help="Skip question IDs already present in --output.")
    p.add_argument("--retriever", type=str, default="fusion", choices=["bm25", "dense", "fusion", "fusion2", "llm_dense"])
    p.add_argument("--no-skills", action="store_true", help="Disable reusable skills.")
    p.add_argument("--force-skill", type=str, default=None, help="Force-load a single skill before step 1.")
    p.add_argument("--no-context", action="store_true", help="Disable structured context management.")
    p.add_argument("--llm-summary", action="store_true", help="Enable optional LLM note summarization.")
    p.add_argument("--max-steps", type=int, default=config.MAX_STEPS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    questions = load_questions(args.input)
    if args.limit > 0:
        questions = questions[: args.limit]

    agent = build_agent(
        retriever_mode=args.retriever,
        use_skills=not args.no_skills,
        force_skill=args.force_skill,
        use_context_manager=not args.no_context,
        enable_llm_summary=args.llm_summary,
        max_steps=args.max_steps,
    )
    run_batch(questions, args.output, agent, workers=args.workers, resume=args.resume)


if __name__ == "__main__":
    main()
