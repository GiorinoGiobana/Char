#!/usr/bin/env python3
"""
Few-Shot prompt experiment: Using GLM4.7 model, even questions (0-1534)
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import argparse
import json
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import DEV_DATABASES_DIR, DEV_MODIFIED_JSON, setup_sandbox
from src.graph_agent_glm_few_shot import app
from src.eval_utils import compare_query_results, truncate_for_json
from src.tools import read_json_file, run_sqlite_query, write_json_file, safe_remove_sandbox


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
        raise TimeoutError("Workflow execution timeout (240s)")
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
    parser = argparse.ArgumentParser(description="Few-Shot prompt experiment (GLM4.7 model)")
    parser.add_argument("--out-dir", type=str, default="", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Only generate environment and save, do not call LLM")
    parser.add_argument("--keep-temp", action="store_true", help="Keep sandbox DB and recovery.json in temp_env")
    parser.add_argument("--force", action="store_true", help="Do not skip completed questions (overwrite and re-run)")
    parser.add_argument("--eval-gold", action="store_true", help="Evaluate: align with gold SQL results")
    parser.add_argument("--start", type=int, default=0, help="Start question index")
    parser.add_argument("--end", type=int, default=1534, help="End question index (exclusive)")
    parser.add_argument("--batch-rest", type=int, default=5, help="Rest seconds between batches")
    args = parser.parse_args()

    start = args.start
    end = args.end

    data = _load_dev_modified()
    total_questions = len(data)

    indices = list(range(start, end))
    print(f"📝 Processing all questions: {start}-{end}")

    slice_items = [data[i] for i in indices]
    limit = len(slice_items)

    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.path.join("results", "exp_few_shot")
    _ensure_dir(out_dir)

    recovery_out_dir = os.path.join(out_dir, "recovery_json")
    _ensure_dir(recovery_out_dir)

    results_path = os.path.join(out_dir, "results.jsonl")
    meta_path = os.path.join(out_dir, "meta.json")

    meta = {
        "experiment": "exp_few_shot_glm",
        "model": "GLM-4.7",
        "prompt_mode": "few_shot",
        "dev_modified_json": DEV_MODIFIED_JSON,
        "start_index": start,
        "end_index": end,
        "all_indices": True,
        "total_questions": total_questions,
        "processed_questions": limit,
        "batch_rest": args.batch_rest,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": bool(args.dry_run),
        "keep_temp": bool(args.keep_temp),
        "force": bool(args.force),
        "eval_gold": bool(args.eval_gold),
    }
    write_json_file(meta_path, meta)

    done_qids: set = set()
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

    print(f"🚀 Start running Few-Shot experiment (GLM4.7 model, even questions{start}-{end}，total{limit}）")
    print(f"📁 Output directory: {out_dir}")
    print(f"📊 Total questions: {total_questions}, This batch: {limit}")
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)

    total = len(slice_items)

    with open(results_path, "a", encoding="utf-8") as f_out:
        for idx, item in zip(indices, slice_items):
            question_id = item.get("question_id")
            question = str(item.get("question", ""))
            db_path = ""
            recovery_json_path = ""
            already_written = False

            record: dict = {
                "index": idx,
                "question_id": question_id,
                "db_id": item.get("db_id"),
                "question": question,
                "modified": item.get("modified"),
                "gold_sql": item.get("SQL"),
                "status": "unknown",
                "error": "",
                "workflow_type": "",
                "final_answer": "",
                "final_sql": "",
                "final_query_result": "",
                "repair_sql": "",
                "recovery_json_saved_to": "",
                "time_sec": 0.0,
                "eval_gold_enabled": bool(args.eval_gold),
                "eval_gold_result": "",
                "eval_sandbox_gold_result": "",
                "eval_is_match": False,
                "eval_diff_summary": "",
                "eval_final_result": "",
                "eval_final_is_match": False,
                "eval_final_diff_summary": "",
            }

            t0 = time.time()
            try:
                if question_id in done_qids:
                    print(f"⏭️  Skip questions {idx}: {question_id} (already processed)", flush=True)
                    continue

                print(f"🔍 Process questions {idx}: {question_id} - {question[:50]}...", flush=True)

                original_gold_result = None
                if args.eval_gold:
                    gold_sql = item.get("SQL", "")
                    if gold_sql:
                        try:
                            db_id = item.get("db_id")
                            original_db_path = os.path.join(DEV_DATABASES_DIR, db_id, f"{db_id}.sqlite")
                            original_gold_result = run_sqlite_query(original_db_path, gold_sql)
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
                    already_written = True
                    print(f"💨 Dry run completed {idx}: {question_id}")
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
                except Exception as e:
                    record["status"] = "error"
                    record["error"] = f"Workflow error: {repr(e)}"
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()
                    print(f"💥 Workflow error {idx}: {question_id} - {repr(e)}")

                    if not args.keep_temp:
                        safe_remove_sandbox(db_path, recovery_json_path)
                        try:
                            os.rmdir(sandbox_dir)
                        except Exception:
                            pass
                    continue

                record.update({
                    "workflow_type": final_state.get("workflow_type", ""),
                    "final_answer": final_state.get("final_answer", ""),
                    "final_sql": final_state.get("final_sql", ""),
                    "final_query_result": final_state.get("final_query_result", ""),
                    "repair_sql": final_state.get("repair_sql", ""),
                    "status": "completed",
                })

                if args.eval_gold and original_gold_result is not None:
                    gold_sql = item.get("SQL", "")
                    if gold_sql:
                        try:
                            record["eval_gold_result"] = truncate_for_json(original_gold_result)

                            if isinstance(original_gold_result, str) and original_gold_result.startswith("Error:"):
                                record["eval_is_match"] = False
                                record["eval_diff_summary"] = "INVALID_GOLD_SQL"
                            else:
                                if record.get("workflow_type") == "CASE_2_NORMAL":
                                    record["eval_is_match"] = True
                                    record["eval_diff_summary"] = "NO_REPAIR_NEEDED"
                                else:
                                    try:
                                        sandbox_gold_result = run_sqlite_query(db_path, gold_sql)
                                        record["eval_sandbox_gold_result"] = truncate_for_json(sandbox_gold_result)

                                        is_match, diff_summary = compare_query_results(original_gold_result, sandbox_gold_result)
                                        record["eval_is_match"] = is_match
                                        record["eval_diff_summary"] = diff_summary
                                    except Exception as eval_e:
                                        record["eval_is_match"] = False
                                        record["eval_diff_summary"] = f"EVAL_ERROR: {repr(eval_e)}"

                                if record.get("workflow_type") != "CASE_2_NORMAL":
                                    final_sql = record.get("final_sql", "")
                                    if final_sql:
                                        try:
                                            final_result = run_sqlite_query(db_path, final_sql)
                                            record["eval_final_result"] = truncate_for_json(final_result)
                                            is_match, diff_summary = compare_query_results(original_gold_result, final_result)
                                            record["eval_final_is_match"] = is_match
                                            record["eval_final_diff_summary"] = diff_summary
                                        except Exception as final_e:
                                            record["eval_final_is_match"] = False
                                            record["eval_final_diff_summary"] = f"FINAL_EVAL_ERROR: {repr(final_e)}"
                        except Exception as outer_e:
                            record["eval_diff_summary"] = f"OUTER_EVAL_ERROR: {repr(outer_e)}"

                if not args.keep_temp:
                    safe_remove_sandbox(db_path, recovery_json_path)
                    try:
                        os.rmdir(sandbox_dir)
                    except Exception:
                        pass

                except Exception as e:
                record["status"] = "error"
                record["error"] = repr(e)
                print(f"💥 Process questions {idx} error: {repr(e)}")

            elapsed = time.time() - t0
            record["time_sec"] = elapsed

            if not already_written:
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()

            if not args.keep_temp:
                safe_remove_sandbox(db_path, recovery_json_path)

            status_symbol = {
                "completed": "✅",
                "timeout": "⏰",
                "error": "❌",
                "skipped_already_done": "⏭️",
                "dry_run_complete": "💨"
            }.get(record.get("status", ""), "❓")

            print(f"{status_symbol} Completed questions {idx}: {question_id} ({elapsed:.1f}s, {record.get('workflow_type', '')})")

        print("-" * 60)
        print(f"✅ Experiment completed! Results saved to: {out_dir}")
        print(f"⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()