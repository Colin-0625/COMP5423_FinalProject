"""Validate test_prediction.jsonl against the COMP5423 Valid System Submission rules.

Usage:
  python validate_submission.py --pred ../test_prediction.jsonl --expected 100
  python validate_submission.py --pred ../test_prediction.jsonl --input data_files/test.jsonl --expected 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Set

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.loader import load_questions


REQUIRED_FIELDS = {
    "id": str,
    "question": str,
    "answer": str,
    "trajectory": list,
    "tool_calls": int,
    "total_tokens": int,
}


def validate(pred_path: str, expected_count: int = 100, input_path: str = "") -> List[str]:
    errors: List[str] = []
    if not os.path.isfile(pred_path):
        return [f"File not found: {pred_path}"]

    seen_ids: Set[str] = set()
    rows = 0
    with open(pred_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {lineno}: invalid JSON ({e})")
                continue

            if not isinstance(obj, dict):
                errors.append(f"Line {lineno}: expected JSON object")
                continue

            for field, ftype in REQUIRED_FIELDS.items():
                if field not in obj:
                    errors.append(f"Line {lineno}: missing required field {field!r}")
                elif field in ("tool_calls", "total_tokens") and isinstance(obj[field], bool):
                    errors.append(f"Line {lineno}: field {field!r} must be int, not bool")
                elif not isinstance(obj[field], ftype):
                    errors.append(f"Line {lineno}: field {field!r} expected {ftype.__name__}, got {type(obj[field]).__name__}")

            qid = obj.get("id")
            if isinstance(qid, str):
                if qid in seen_ids:
                    errors.append(f"Line {lineno}: duplicate id {qid!r}")
                seen_ids.add(qid)

            if isinstance(obj.get("trajectory"), list):
                traj = obj["trajectory"]
                if len(traj) == 0:
                    errors.append(f"Line {lineno} (id={qid}): trajectory is empty")
                else:
                    # Reconcile embedded trajectory counts against top-level tool_calls and total_tokens
                    if any(isinstance(s, dict) and "type" in s for s in traj):
                        tc_in_traj = sum(1 for s in traj if isinstance(s, dict) and s.get("type") == "tool_call")
                        tok_in_traj = sum(int(s.get("tokens", 0)) for s in traj if isinstance(s, dict) and "tokens" in s)
                        if isinstance(obj.get("tool_calls"), int) and tc_in_traj != obj["tool_calls"]:
                            errors.append(
                                f"Line {lineno} (id={qid}): tool_calls={obj['tool_calls']} != trajectory tool_call steps ({tc_in_traj})"
                            )
                        if isinstance(obj.get("total_tokens"), int) and tok_in_traj != obj["total_tokens"]:
                            errors.append(
                                f"Line {lineno} (id={qid}): total_tokens={obj['total_tokens']} != sum of trajectory step tokens ({tok_in_traj})"
                            )
            if isinstance(obj.get("tool_calls"), int) and obj["tool_calls"] < 0:
                errors.append(f"Line {lineno} (id={qid}): tool_calls < 0")
            if isinstance(obj.get("total_tokens"), int) and obj["total_tokens"] < 0:
                errors.append(f"Line {lineno} (id={qid}): total_tokens < 0")

    if expected_count > 0 and rows != expected_count:
        errors.append(f"Record count mismatch: found {rows}, expected {expected_count}")

    if input_path:
        input_ids = {str(q["id"]) for q in load_questions(input_path)}
        missing = sorted(input_ids - seen_ids)
        extra = sorted(seen_ids - input_ids)
        if missing:
            errors.append(f"Missing {len(missing)} input IDs, e.g. {missing[:5]}")
        if extra:
            errors.append(f"Extra {len(extra)} unknown IDs, e.g. {extra[:5]}")

    return errors


def main() -> None:
    p = argparse.ArgumentParser(description="Validate test_prediction.jsonl schema and completeness.")
    p.add_argument("--pred", required=True, help="Path to prediction JSONL file.")
    p.add_argument("--input", default="", help="Optional test input JSONL to cross-check IDs.")
    p.add_argument("--expected", type=int, default=100, help="Expected number of records (default 100).")
    args = p.parse_args()

    errors = validate(args.pred, expected_count=args.expected, input_path=args.input)
    if errors:
        print(f"VALIDATION FAILED ({len(errors)} error(s)):")
        for err in errors[:30]:
            print("  -", err)
        sys.exit(1)
    print(f"VALIDATION PASSED: {args.pred} ({args.expected} valid records)")


if __name__ == "__main__":
    main()
