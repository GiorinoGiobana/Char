"""Run the few-shot experiment through the shared StateSculptor workflow."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.environment import setup_sandbox
from src.eval_utils import compare_query_results
from src.tools import run_sqlite_query
from experiments.prompting.agent_few_shot import app

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--output", default="results/exp_few_shot")
    args = parser.parse_args()
    dataset = json.loads((ROOT / "bird" / "dev" / "dev_modified.json").read_text(encoding="utf-8"))
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.jsonl").open("a", encoding="utf-8") as handle:
        for item in dataset[args.start:args.end]:
            db_path, recovery_path = setup_sandbox(item["question"])
            state = {"question": item["question"], "db_path": db_path, "recovery_json_path": recovery_path}
            merged = {}
            for output in app.stream(state):
                for value in output.values():
                    if isinstance(value, dict):
                        merged.update(value)
            sql = merged.get("final_sql", "")
            candidate = run_sqlite_query(db_path, sql) if sql else []
            gold = run_sqlite_query(db_path, item["SQL"])
            match, reason = compare_query_results(gold, candidate)
            record = {"question_id": item.get("question_id"), "prompt_mode": "few_shot", "result": merged, "is_match": match, "match_reason": reason}
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + chr(10))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
