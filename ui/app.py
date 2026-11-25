"""Interactive terminal CLI for the Skill-Augmented Search Agent.

Displays final answer, selected skills, structured search state / evidence,
and the chronological tool-use trajectory.
"""
from __future__ import annotations

import argparse
import os
import sys

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)

from agent.loop import build_agent


def main() -> None:
    p = argparse.ArgumentParser(description="Interactive CLI for the Skill-Augmented Search Agent.")
    p.add_argument("--retriever", default="fusion", choices=["bm25", "dense", "llm_dense", "fusion2", "fusion"])
    p.add_argument("--no-skills", action="store_true")
    p.add_argument("--no-context", action="store_true")
    args = p.parse_args()

    agent = build_agent(
        retriever_mode=args.retriever,
        use_skills=not args.no_skills,
        use_context_manager=not args.no_context,
    )
    print("Skill-Augmented Search Agent CLI (type 'exit' to quit)\n")
    while True:
        try:
            q = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("exit", "quit"):
            break
        res = agent.run(q)
        st = res.state
        print("\n--- Final Answer ---")
        print(res.answer)
        print("\n--- Selected Skills ---")
        print(st.loaded_skills or "(none)")
        print("\n--- Structured Search State ---")
        print("Sub-queries:", st.sub_queries or "(none)")
        print("Visited docs:", st.visited_docs)
        print("Opened pages:", st.opened_pages or "(none)")
        print("Candidate answer:", st.candidate_answer)
        if st.evidence:
            print(f"Selected evidence ({len(st.evidence)}):")
            for idx, ev in enumerate(st.evidence, start=1):
                print(f"  [{idx}] doc_id={ev['doc_id']} ({ev.get('source_tool', '')}): {ev['text'][:160]}")
        if st.context_stats:
            last = st.context_stats[-1]
            print(
                f"Context control (last step): raw={last['raw_chars']} chars -> "
                f"sent={last['sent_chars']} chars (dropped_msgs={last['dropped_msgs']})"
            )
        print(f"\n--- Tool-Use Process ({st.tool_calls} calls, {st.total_tokens} tokens) ---")
        for step in st.trajectory:
            print(f"  [step {step.get('step')}] {step.get('type')}: {step.get('tool') or ''} {str(step.get('content', step.get('args', '')))[:160]}")
        print()


if __name__ == "__main__":
    main()
