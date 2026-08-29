"""Minimal example for running the StateSculptor workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import get_llm
from .workflow import construct_test_data, run_workflow


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a minimal StateSculptor example")
    parser.add_argument("--question", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--gold-sql", required=True)
    parser.add_argument("--modified-sql", default="")
    parser.add_argument("--provider", default=None)
    args = parser.parse_args()

    case = construct_test_data(args.db, args.gold_sql, args.modified_sql)
    attachment = Path(case["work_dir"]) / "attachment.json"
    attachment.write_text(json.dumps(case["attachment"], ensure_ascii=False), encoding="utf-8")
    result = run_workflow(args.question, case["database_path"], str(attachment), llm=get_llm(args.provider))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
