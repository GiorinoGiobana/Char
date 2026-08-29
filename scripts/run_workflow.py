"""Command-line entry point for the paper workflow."""

from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import get_llm
from src.statesculptor import run_workflow


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the StateSculptor database workflow")
    parser.add_argument("--question", required=True, help="User request Q")
    parser.add_argument("--db", required=True, help="Path to the current SQLite database")
    parser.add_argument("--attachment", default="", help="Path to the non-executable attachment JSON")
    parser.add_argument("--provider", default=None, help="LLM provider; defaults to LLM_PROVIDER")
    parser.add_argument("--risk-threshold", type=float, default=0.01)
    parser.add_argument("--keep-sandbox", action="store_true")
    parser.add_argument("--output", default="", help="Optional output JSON path")
    args = parser.parse_args()

    llm = get_llm(args.provider)
    result = run_workflow(
        question=args.question,
        db_path=args.db,
        attachment_path=args.attachment,
        llm=llm,
        risk_threshold=args.risk_threshold,
        keep_sandbox=args.keep_sandbox,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
