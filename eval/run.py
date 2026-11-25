"""Validation and ablation evaluation runner.

Example:
  python eval/run.py --input data_files/validation.jsonl --retriever fusion --tag full_system --workers 4 --resume
"""
from __future__ import annotations

import argparse
import json
import os
import sys

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)

import config
from agent.loop import build_agent
from data.loader import load_questions
from eval.metrics import evaluate_predictions
from run_predict import run_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the search agent on the validation set.")
    p.add_argument(
        "--input",
        type=str,
        default=os.path.join(config.DATA_DIR, "validation.jsonl"),
        help="Validation JSONL path with reference answers.",
    )
    p.add_argument("--out-dir", type=str, default=os.path.join(CODE_DIR, "eval_outputs"))
    p.add_argument("--tag", type=str, default="exp", help="Experiment identifier tag.")
    p.add_argument("--limit", type=int, default=0, help="Evaluate first N questions (0 = all).")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--retriever", type=str, default="fusion", choices=["bm25", "dense", "fusion", "fusion2", "llm_dense"])
    p.add_argument("--no-skills", action="store_true")
    p.add_argument("--force-skill", type=str, default=None)
    p.add_argument("--no-context", action="store_true")
    p.add_argument("--llm-summary", action="store_true", help="Enable optional LLM note summarization.")
    p.add_argument("--max-steps", type=int, default=config.MAX_STEPS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    questions = load_questions(args.input)
    if args.limit > 0:
        questions = questions[: args.limit]

    gold_by_id = {str(q["id"]): q["answer"] for q in questions if "answer" in q}
    if not gold_by_id:
        raise ValueError("Validation file has no 'answer' fields to score.")

    os.makedirs(args.out_dir, exist_ok=True)
    pred_path = os.path.join(args.out_dir, f"{args.tag}_predictions.jsonl")
    summary_path = os.path.join(args.out_dir, f"{args.tag}_summary.json")

    agent = build_agent(
        retriever_mode=args.retriever,
        use_skills=not args.no_skills,
        force_skill=args.force_skill,
        use_context_manager=not args.no_context,
        enable_llm_summary=args.llm_summary,
        max_steps=args.max_steps,
    )
    records = run_batch(questions, pred_path, agent, workers=args.workers, resume=args.resume)
    metrics = evaluate_predictions(records, gold_by_id)
    summary = {
        "tag": args.tag,
        "config": {
            "retriever": args.retriever,
            "use_skills": not args.no_skills,
            "force_skill": args.force_skill,
            "use_context_manager": not args.no_context,
            "enable_llm_summary": args.llm_summary,
            "max_steps": args.max_steps,
        },
        "metrics": metrics,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Evaluation Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved predictions -> {pred_path}")
    print(f"Saved summary     -> {summary_path}")


if __name__ == "__main__":
    main()
