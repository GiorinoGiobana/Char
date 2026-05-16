#!/usr/bin/env python3
"""
Experiment 1: Run all 1534 questions using GLM4.7 model
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import DEV_DATABASES_DIR, DEV_MODIFIED_JSON, setup_sandbox  # noqa: E402
from src.graph_agent_glm import app  # noqa: E402
from src.eval_utils import compare_query_results, truncate_for_json  # noqa: E402
from src.tools import read_json_file, run_sqlite_query, write_json_file  # noqa: E402


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
        timer = threading.Timer(120.0, timeout_handler)
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
    parser = argparse.ArgumentParser(description="Experiment 1 full runner with GLM4.7 model")
    parser.add_argument("--out-dir", type=str, default="", help="Output directory (default results/exp1_full_glm_YYYYmmdd_HHMMSS）")
    parser.add_argument("--dry-run", action="store_true", help="Only generate environment and save, do not call LLM")
    parser.add_argument("--keep-temp", action="store_true", help="Keep sandbox DB and recovery.json in temp_env")
    parser.add_argument("--force", action="store_true", help="Do not skip completed questions (overwrite and re-run)")
    parser.add_argument("--eval-gold", action="store_true", help="Evaluate: align with gold SQL results (not passed to model)")
    parser.add_argument("--start", type=int, default=0, help="Start question index")
    parser.add_argument("--end", type=int, default=None, help="End question index (exclusive), default all")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of questions per batch")
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
        out_dir = os.path.join("results", f"exp1_full_glm_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    _ensure_dir(out_dir)

    recovery_out_dir = os.path.join(out_dir, "recovery_json")
    _ensure_dir(recovery_out_dir)

    results_path = os.path.join(out_dir, "results.jsonl")
    meta_path = os.path.join(out_dir, "meta.json")

    meta = {
        "experiment": "exp1_full_glm",
        "model": "GLM-4.7",
        "dev_modified_json": DEV_MODIFIED_JSON,
        "start_index": start,
        "end_index": end,
        "total_questions": total_questions,
        "batch_size": args.batch_size,
        "batch_rest": args.batch_rest,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": bool(args.dry_run),
        "keep_temp": bool(args.keep_temp),
        "force": bool(args.force),
        "eval_gold": bool(args.eval_gold),
    }
    write_json_file(meta_path, meta)

    slice_items = data[start:end]

    done_qids: set = set()
    if os.path.exists(results_path) and not args.force:
        try:
            with open(results_path, "r", encoding="utf-8") as f_in:
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

    print(f"🚀 Start running Exp1 full experiment (GLM4.7 model, Questions{start}-{end-1}，total{limit}）")
    print(f"📁 Output directory: {out_dir}")
    print(f"📊 Total questions: {total_questions}, This batch: {limit}, Batch size: {args.batch_size}")
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)

    total = len(slice_items)
    batch_num = 0
    
    with open(results_path, "a", encoding="utf-8") as f_out:
        for idx, item in enumerate(slice_items, start=start):
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
                    record["status"] = "skipped_already_done"
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()
                    already_written = True
                    print(f"⏭️  Skip questions {idx}: {question_id} (already processed)")
                    continue

                print(f"🔍 Process questions {idx}: {question_id} - {question[:50]}...")

                # Save gold SQL result on original database as baseline before repair
                original_gold_result = None
                if args.eval_gold:
                    gold_sql = item.get("SQL", "")
                    if gold_sql:
                        try:
                            # Execute gold SQL on original database (not on sandbox database)
                            db_id = item.get("db_id")
                            original_db_path = os.path.join(DEV_DATABASES_DIR, db_id, f"{db_id}.sqlite")
                            original_gold_result = run_sqlite_query(original_db_path, gold_sql)
                        except Exception:
                            pass

                # Set up sandbox environment
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

                # Copy recovery.json to results directory
                recovery_dst_path = os.path.join(recovery_out_dir, f"recovery_{question_id}.json")
                _copy_recovery_json(recovery_json_path, recovery_dst_path)
                record["recovery_json_saved_to"] = recovery_dst_path

                # Run LangGraph workflow (with timeout protection)
                initial_state = {
                    "question": question,
                    "db_path": db_path,
                    "recovery_json_path": recovery_json_path,
                }

                try:
                    final_state = _collect_final_state_from_stream(initial_state)
                except TimeoutError:
                    record["status"] = "timeout"
                    record["error"] = "Workflow execution timeout (120s)"
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()
                    print(f"⏰ Workflow timeout {idx}: {question_id}")
                    
                    # Clean up temporary files
                    if not args.keep_temp:
                        _safe_remove(db_path)
                        _safe_remove(recovery_json_path)
                        try:
                            os.rmdir(sandbox_dir)
                        except Exception:
                            pass
                    continue
                except Exception as e:
                    record["status"] = "error"
                    record["error"] = f"Workflow error: {repr(e)}"
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()
                    print(f"💥 Workflow error {idx}: {question_id} - {repr(e)}")
                    
                    # Clean up temporary files
                    if not args.keep_temp:
                        _safe_remove(db_path)
                        _safe_remove(recovery_json_path)
                        try:
                            os.rmdir(sandbox_dir)
                        except Exception:
                            pass
                    continue
                
                # Extract results
                record.update({
                    "workflow_type": final_state.get("workflow_type", ""),
                    "final_answer": final_state.get("final_answer", ""),
                    "final_sql": final_state.get("final_sql", ""),
                    "final_query_result": final_state.get("final_query_result", ""),
                    "repair_sql": final_state.get("repair_sql", ""),
                    "status": "completed",
                })

                # Evaluation 1: Database repair capability (test with gold SQL)
                if args.eval_gold and original_gold_result is not None:
                    gold_sql = item.get("SQL", "")
                    if gold_sql:
                        try:
                            record["eval_gold_result"] = truncate_for_json(original_gold_result)
                            
                            # Check if Gold SQL executes successfully on original database
                            if isinstance(original_gold_result, str) and original_gold_result.startswith("Error:"):
                                # Gold SQL execution failed on original database, skip evaluation
                                record["eval_is_match"] = False
                                record["eval_diff_summary"] = "INVALID_GOLD_SQL"
                            else:
                                # CASE_2_NORMAL does not need repair, return true directly
                                if record.get("workflow_type") == "CASE_2_NORMAL":
                                    record["eval_is_match"] = True
                                    record["eval_diff_summary"] = "NO_REPAIR_NEEDED"
                                else:
                                    # Execute gold SQL on repaired database
                                    repaired_gold_result = run_sqlite_query(db_path, gold_sql)
                                    record["eval_sandbox_gold_result"] = truncate_for_json(repaired_gold_result)
                                    
                                    # Compare results (original vs repaired)
                                    is_match, diff_summary = compare_query_results(gold_sql, original_gold_result, repaired_gold_result)
                                    record["eval_is_match"] = bool(is_match)
                                    record["eval_diff_summary"] = diff_summary
                        except Exception as e:
                            record["eval_is_match"] = False
                            record["eval_diff_summary"] = f"GOLD_EVAL_ERROR: {repr(e)}"

                # Evaluation 2: Final query capability (test with LLM-generated final_sql)
                final_sql = record.get("final_sql", "")
                if final_sql:
                    try:
                        final_res_raw = run_sqlite_query(db_path, final_sql)
                        record["eval_final_result"] = truncate_for_json(final_res_raw)
                        
                        # Compare LLM-generated SQL result with gold SQL result
                        # Only compare when Gold SQL executes successfully on original database
                        gold_sql = item.get("SQL", "")
                        if gold_sql and original_gold_result is not None:
                            if isinstance(original_gold_result, str) and original_gold_result.startswith("Error:"):
                                # Gold SQL execution failed on original database, skip evaluation
                                record["eval_final_is_match"] = False
                                record["eval_final_diff_summary"] = "INVALID_GOLD_SQL"
                            else:
                                gold_res_raw = run_sqlite_query(db_path, gold_sql)
                                final_is_match, final_diff = compare_query_results(gold_sql, gold_res_raw, final_res_raw)
                                record["eval_final_is_match"] = bool(final_is_match)
                                record["eval_final_diff_summary"] = final_diff
                        else:
                            record["eval_final_is_match"] = False
                            record["eval_final_diff_summary"] = "NO_GOLD_SQL_OR_RESULT"
                    except Exception as e:
                        record["eval_final_is_match"] = False
                        record["eval_final_diff_summary"] = f"FINAL_EVAL_ERROR: {repr(e)}"
                else:
                    record["eval_final_is_match"] = False
                    record["eval_final_diff_summary"] = "MISSING_FINAL_SQL"

                # Clean up temporary files
                if not args.keep_temp:
                    _safe_remove(db_path)
                    _safe_remove(recovery_json_path)
                    try:
                        os.rmdir(sandbox_dir)
                    except Exception:
                        pass

                t1 = time.time()
                record["time_sec"] = t1 - t0
                
                # Output results
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()
                
                # Real-time progress output
                repair_success = "✅" if record.get("eval_is_match", False) else "❌"
                final_success = "✅" if record.get("eval_final_is_match", False) else "❌"
                print(f"   {idx:4d}: {repair_success}repair {final_success}query | Time: {record['time_sec']:.1f}s | {record['workflow_type']}")
                
                # Rest between batches (rest after every batch_size questions)
                batch_num += 1
                if batch_num % args.batch_size == 0 and batch_num < total:
                    completed = batch_num
                    remaining = total - completed
                    print(f"\n📦 Batch completed: {completed}/{total} ({completed/total*100:.1f}%)")
                    print(f"⏳ Rest {args.batch_rest}seconds before continuing...")
                    time.sleep(args.batch_rest)
                    print("-" * 60)

            except Exception as e:
                t1 = time.time()
                record["status"] = "error"
                record["error"] = repr(e)
                record["time_sec"] = t1 - t0
                
                if not already_written:
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()
                
                print(f"💥 Error {idx}: {question_id} - {repr(e)}")

    print("-" * 60)
    print(f"🎉 Exp1 full experiment (GLM4.7 model) completed! Results saved in: {out_dir}")
    print(f"⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
