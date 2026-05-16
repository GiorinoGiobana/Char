#!/usr/bin/env python3
"""
Experiment 1: Run all 1534 questions using ChatAnywhere API (GLM-4.7 model)
"""

import argparse
import json
import os
import sys
import time
import io
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import DEV_DATABASES_DIR, DEV_MODIFIED_JSON, setup_sandbox  # noqa: E402
from src.graph_agent_zero_shot import app  # noqa: E402
from src.eval_utils import compare_query_results, truncate_for_json  # noqa: E402
from src.tools import read_json_file, run_sqlite_query, write_json_file, safe_remove_sandbox  # noqa: E402


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_remove(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _load_dev_modified() -> list[dict]:
    with open(DEV_MODIFIED_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_final_state_from_stream(initial_state: dict) -> dict:
    """Workflow execution with timeout protection (cross-platform, Windows compatible)"""
    import threading
    import _thread

    def timeout_handler():
        _thread.interrupt_main()

    merged: dict = {}
    timer = None

    try:
        timer = threading.Timer(600.0, timeout_handler)
        timer.start()

        for output in app.stream(initial_state):
            for _node, value in output.items():
                if isinstance(value, dict):
                    merged.update(value)
    except KeyboardInterrupt:
        raise TimeoutError("Workflow execution interrupted")
    finally:
        if timer:
            timer.cancel()

    return merged


def _copy_recovery_json(src_path: str, dst_path: str) -> bool:
    if not os.path.exists(src_path):
        return False
    data = read_json_file(src_path)
    write_json_file(dst_path, data)
    return True


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: ChatAnywhere GLM-4.7 One-Shot")
    parser.add_argument("--out-dir", type=str, default="", help="Output directory (default results/exp1_chatanywhere_YYYYmmdd_HHMMSS）")
    parser.add_argument("--dry-run", action="store_true", help="Only generate environment and save, do not call LLM")
    parser.add_argument("--keep-temp", action="store_true", help="Keep sandbox DB and recovery.json in temp_env")
    parser.add_argument("--force", action="store_true", help="Do not skip completed questions (overwrite and re-run)")
    parser.add_argument("--eval-gold", action="store_true", help="Evaluate: align with gold SQL results (not passed to model)")
    parser.add_argument("--start", type=int, default=0, help="Start question index (even experiment starts from 0)")
    parser.add_argument("--end", type=int, default=None, help="End question index (exclusive), default all")
    parser.add_argument("--batch-rest", type=int, default=5, help="Rest seconds between batches")
    args = parser.parse_args()

    start = args.start
    data = _load_dev_modified()
    total_questions = len(data)
    end = args.end if args.end is not None else total_questions
    limit = end - start

    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.path.join("results", "exp_zero_shot")
    _ensure_dir(out_dir)

    recovery_out_dir = os.path.join(out_dir, "recovery_json")
    _ensure_dir(recovery_out_dir)
    meta_path = os.path.join(out_dir, "meta.json")
    results_path = os.path.join(out_dir, "results.jsonl")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "chatanywhere-glm-4.7",
            "total": total_questions,
            "limit": limit,
            "start": start,
            "end": end,
            "keep_temp": bool(args.keep_temp),
        }, f, ensure_ascii=False, indent=2)

    done_qids = set()
    if os.path.exists(results_path) and not args.force:
        try:
            with open(results_path, "r", encoding="utf-8-sig") as f_in:
                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    qid = obj.get("question_id")
                    if qid is not None:
                        done_qids.add(qid)
        except Exception:
            done_qids = set()

    print(f"[Start] ChatAnywhere GLM-4.7 One-Shot：Questions{start}-{end}，total{limit}")
    print(f"[Output] Output directory: {out_dir}")
    print(f"[Time] Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)

    slice_items = data[start:end]
    batch_num = 0

    with open(results_path, "a", encoding="utf-8") as f_out:
        for idx, item in enumerate(slice_items, start=start):
            question_id = item.get("question_id", idx)

            if question_id in done_qids and not args.force:
                print(f"⏭️  Skip questions {idx}: {question_id} (already processed)", flush=True)
                continue

            batch_num += 1
            t0 = time.time()

            print(f"🔍 Process questions {idx}: {question_id} - {item.get('question', '')[:50]}...", flush=True)

            db_id = item.get("db_id", "")
            question = item.get("question", "")
            gold_sql = item.get("SQL", "")

            record = {
                "index": idx,
                "question_id": question_id,
                "db_id": db_id,
                "question": question,
                "modified": item.get("modified", ""),
                "gold_sql": gold_sql,
                "status": "pending",
                "error": "",
                "workflow_type": "",
                "final_answer": "",
                "final_sql": "",
                "final_query_result": "",
                "repair_sql": "",
                "recovery_json_saved_to": "",
                "time_sec": 0,
                "eval_gold_enabled": args.eval_gold,
                "eval_gold_result": None,
                "eval_sandbox_gold_result": None,
                "eval_is_match": False,
                "eval_diff_summary": "",
                "eval_final_result": None,
                "eval_final_is_match": False,
                "eval_final_diff_summary": "",
            }

            t0 = time.time()
            try:
                original_gold_result = None
                if args.eval_gold:
                    if gold_sql:
                        try:
                            original_db_path = os.path.join(DEV_DATABASES_DIR, db_id, f"{db_id}.sqlite")
                            original_gold_result = run_sqlite_query(original_db_path, gold_sql)
                            record["eval_gold_result"] = original_gold_result
                        except Exception:
                            pass

                sandbox_dir = os.path.join("temp_env", f"sandbox_{question_id}")
                _ensure_dir(sandbox_dir)

                db_path, recovery_json_path = setup_sandbox(question)

                if args.dry_run:
                    record["status"] = "dry_run_complete"
                    record["db_path"] = db_path
                    record["recovery_json_path"] = recovery_json_path
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()
                    print(f"💨 Dry run completed {idx}: {question_id}", flush=True)
                    continue

                recovery_dst_path = os.path.join(recovery_out_dir, f"recovery_{question_id}.json")
                _copy_recovery_json(recovery_json_path, recovery_dst_path)
                record["recovery_json_saved_to"] = recovery_dst_path

                initial_state = {
                    "question": question,
                    "db_path": db_path,
                    "recovery_json_path": recovery_json_path,
                }

                try:
                    final_state = _collect_final_state_from_stream(initial_state)
                    record["workflow_type"] = final_state.get("workflow_type", "")
                    record["final_answer"] = final_state.get("final_answer", "")
                    record["final_sql"] = final_state.get("final_sql", "")
                    record["final_query_result"] = final_state.get("final_query_result", "")
                    record["repair_sql"] = final_state.get("repair_sql", "")
                    record["status"] = "completed"
                except Exception as e:
                    record["status"] = "error"
                    record["error"] = f"Workflow error: {repr(e)}"
                    print(f"💥 Workflow error {idx}: {question_id} - {repr(e)}", flush=True)

                    if not args.keep_temp:
                        safe_remove_sandbox(db_path, recovery_json_path)

                if args.eval_gold and original_gold_result is not None:
                    final_sql = record.get("final_sql", "")
                    if final_sql:
                        try:
                            final_result = run_sqlite_query(db_path, final_sql)
                            record["eval_final_result"] = final_result
                            match, diff = compare_query_results(original_gold_result, final_result)
                            record["eval_final_is_match"] = match
                            record["eval_final_diff_summary"] = diff if not match else ""
                        except Exception as e:
                            record["eval_final_diff_summary"] = f"Eval error: {repr(e)}"

            except Exception as e:
                record["status"] = "error"
                record["error"] = f"Processing error: {repr(e)}"
                print(f"💥 Process questions {idx} error: {repr(e)}", flush=True)

            if not args.keep_temp:
                safe_remove_sandbox(db_path, recovery_json_path)

            elapsed = time.time() - t0
            record["time_sec"] = elapsed

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()

            status_symbol = "✅" if record["status"] == "completed" else "❌"
            print(f"{status_symbol} Completed questions {idx}: {question_id} ({elapsed:.1f}s, {record.get('workflow_type', '')})", flush=True)

            if args.batch_rest > 0 and batch_num % 5 == 0:
                print(f"💤 Batch rest {args.batch_rest} seconds...", flush=True)
                time.sleep(args.batch_rest)

        print("-" * 60)
        print(f"✅ Experiment completed! Results saved to: {out_dir}")
        print(f"⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()