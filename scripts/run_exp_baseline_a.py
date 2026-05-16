#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline A Experiment: Direct Query on Perturbed DB with Clues (No repair, query directly with clues)

Design points:
1. Router node unchanged, still determines CASE type
2. Worker node modified: prohibit generating/executing any DDL/DML, only SELECT allowed
3. Execute generated SELECT query directly on perturbed DB
4. Provide JSON clues, but prohibit using for repair

This tests whether the clues alone reveal enough information.
"""

import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import argparse
import json
import os
import time
import threading
import _thread
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import DEV_DATABASES_DIR, DEV_MODIFIED_JSON, setup_sandbox
from src.graph_agent_baseline_a import app
from src.eval_utils import compare_query_results, truncate_for_json
from src.tools import read_json_file, run_sqlite_query, write_json_file, safe_remove_sandbox


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_dev_modified() -> list[dict]:
    with open(DEV_MODIFIED_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_no_result_indices() -> set[int]:
    """Load question indices from no_result_indices.txt"""
    indices_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "no_result_indices.txt")
    no_result_set = set()
    if os.path.exists(indices_file):
        with open(indices_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        no_result_set.add(int(line))
                    except ValueError:
                        continue
    return no_result_set


def _collect_final_state_from_stream(initial_state: dict) -> dict:
    """Workflow execution with timeout protection (cross-platform, Windows compatible)"""
    merged: dict = {}
    timer = None

    def timeout_handler():
        _thread.interrupt_main()

    try:
        # Workflow timeout set to 3600 seconds (1 hour, practically unlimited)
        timer = threading.Timer(3600.0, timeout_handler)
        timer.start()

        for output in app.stream(initial_state):
            for _node, value in output.items():
                if isinstance(value, dict):
                    merged.update(value)
    except KeyboardInterrupt:
        raise TimeoutError("Workflow execution timeout (3600s)")
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
    parser = argparse.ArgumentParser(
        description="Baseline A: Direct Query on Perturbed DB with Clues (No repair, direct query with clues)"
    )
    parser.add_argument("--out-dir", type=str, default="", help="Output directory (default results/exp_baseline_a）")
    parser.add_argument("--dry-run", action="store_true", help="Only generate environment and save, do not call LLM")
    parser.add_argument("--keep-temp", action="store_true", help="Keep sandbox DB and recovery.json in temp_env")
    parser.add_argument("--force", action="store_true", help="Do not skip completed questions (overwrite and re-run)")
    parser.add_argument("--eval-gold", action="store_true", help="Evaluate: align with gold SQL results")
    parser.add_argument("--start", type=int, default=0, help="Start question index")
    parser.add_argument("--end", type=int, default=None, help="End question index (exclusive), default all")
    parser.add_argument("--batch-rest", type=int, default=5, help="Rest seconds between batches")
    parser.add_argument("--reverse", action="store_true", help="Process questions in reverse order (from end-1 to start)")
    parser.add_argument("--no-result-only", action="store_true", help="Only process questions in no_result_indices.txt")
    parser.add_argument("--indices-file", type=str, default=None, help="Read question ID list to process from file")
    args = parser.parse_args()

    data = _load_dev_modified()
    total_questions = len(data)
    start = args.start
    end = args.end if args.end is not None else total_questions
    
    # If --indices-file is specified, read question ID list from file
    if args.indices_file:
        with open(args.indices_file, 'r', encoding='utf-8') as f:
            target_indices = set(int(line.strip()) for line in f if line.strip())
        data = [item for item in data if item.get("question_id") in target_indices]
        total_questions = len(data)
        start = 0
        end = total_questions
    # If --no-result-only is specified, only keep questions in no_result_indices.txt
    elif args.no_result_only:
        no_result_indices = _load_no_result_indices()
        data = [item for item in data if item.get("question_id") in no_result_indices]
        total_questions = len(data)
        # Recalculate start and end positions in new data
        start = max(0, args.start)
        end = min(args.end if args.end is not None else total_questions, total_questions)
    
    limit = end - start

    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.path.join("results", "exp_baseline_a")
    _ensure_dir(out_dir)

    recovery_out_dir = os.path.join(out_dir, "recovery_json")
    _ensure_dir(recovery_out_dir)
    meta_path = os.path.join(out_dir, "meta.json")
    results_path = os.path.join(out_dir, "results.jsonl")

    # Save experiment metadata
    meta = {
        "experiment": "baseline_a_direct_query",
        "description": "Baseline A: Direct Query on Perturbed DB with Clues (No repair, direct query with clues). Router unchanged, Worker prohibits repair and only generates SELECT.",
        "model": "GLM-4.7-thinking (chatanywhere2)",
        "total": total_questions,
        "limit": limit,
        "start": start,
        "end": end,
        "keep_temp": bool(args.keep_temp),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json_file(meta_path, meta)

    # Load processed questions
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

    print("=" * 70)
    print("[Start] Baseline A: Direct Query on Perturbed DB with Clues")
    if args.reverse:
        print(f"[Range] Questions {end-1}-{start}(reverse), total {limit} ")
    else:
        print(f"[Range] Questions {start}-{end-1}, total {limit} ")
    print(f"[Output] Output directory: {out_dir}")
    print(f"[Feature] Router unchanged, Worker prohibits repair and only generates SELECT")
    print(f"[Time] Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    slice_items = data[start:end]
    if args.reverse:
        slice_items = list(reversed(slice_items))
    batch_num = 0

    with open(results_path, "a", encoding="utf-8") as f_out:
        for idx, item in enumerate(slice_items, start=start):
            question_id = item.get("question_id", idx)

            if question_id in done_qids and not args.force:
                print(f"[Skip] Questions {idx}: {question_id} (already processed)", flush=True)
                continue

            batch_num += 1
            t0 = time.time()

            print(f"\n[Process] Questions {idx}: {question_id} - {item.get('question', '')[:50]}...", flush=True)

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
                "baseline": "A",
                "baseline_name": "Direct Query on Perturbed DB with Clues"
            }

            try:
                # Execute Gold SQL on original database to get standard answer
                original_gold_result = None
                if args.eval_gold and gold_sql:
                    try:
                        original_db_path = os.path.join(DEV_DATABASES_DIR, db_id, f"{db_id}.sqlite")
                        if os.path.exists(original_db_path):
                            original_gold_result = run_sqlite_query(original_db_path, gold_sql)
                            record["eval_gold_result"] = original_gold_result
                    except Exception:
                        pass

                # Set up sandbox environment (create perturbed DB)
                sandbox_dir = os.path.join("temp_env", "exp_baseline_a")
                _ensure_dir(sandbox_dir)

                db_path, recovery_json_path = setup_sandbox(question, sandbox_dir)
                record["db_path"] = db_path

                if args.dry_run:
                    record["status"] = "dry_run_complete"
                    record["db_path"] = db_path
                    record["recovery_json_path"] = recovery_json_path
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()
                    print(f"[Dry run] Completed {idx}: {question_id}", flush=True)
                    if not args.keep_temp:
                        safe_remove_sandbox(db_path, recovery_json_path)
                    continue

                # Copy recovery.json to results directory
                recovery_dst_path = os.path.join(recovery_out_dir, f"recovery_{question_id}.json")
                _copy_recovery_json(recovery_json_path, recovery_dst_path)
                record["recovery_json_saved_to"] = recovery_dst_path

                # Build initial state
                initial_state = {
                    "question": question,
                    "db_path": db_path,
                    "recovery_json_path": recovery_json_path,
                }

                # Execute Baseline A workflow
                try:
                    final_state = _collect_final_state_from_stream(initial_state)
                    record["workflow_type"] = final_state.get("workflow_type", "")
                    record["final_answer"] = final_state.get("final_answer", "")
                    record["final_sql"] = final_state.get("final_sql", "")
                    record["final_query_result"] = final_state.get("final_query_result", "")
                    record["repair_sql"] = final_state.get("repair_sql", "")
                    record["status"] = "completed"

                    # Evaluate results
                    if args.eval_gold and original_gold_result is not None:
                        if gold_sql:
                            try:
                                record["eval_gold_result"] = truncate_for_json(original_gold_result)

                                if isinstance(original_gold_result, str) and original_gold_result.startswith("Error:"):
                                    record["eval_is_match"] = False
                                    record["eval_diff_summary"] = "INVALID_GOLD_SQL"
                                else:
                                    # Evaluate Gold SQL execution result on perturbed DB
                                    if record.get("workflow_type") == "CASE_2_NORMAL":
                                        record["eval_is_match"] = True
                                        record["eval_diff_summary"] = "NO_REPAIR_NEEDED"
                                    else:
                                        try:
                                            sandbox_gold_result = run_sqlite_query(db_path, gold_sql)
                                            record["eval_sandbox_gold_result"] = truncate_for_json(sandbox_gold_result)

                                            is_match, diff_summary = compare_query_results(gold_sql, original_gold_result, sandbox_gold_result)
                                            record["eval_is_match"] = is_match
                                            record["eval_diff_summary"] = diff_summary
                                        except Exception as eval_e:
                                            record["eval_is_match"] = False
                                            record["eval_diff_summary"] = f"EVAL_ERROR: {repr(eval_e)}"

                                    # Evaluate final_sql (SELECT generated by Baseline A)
                                    final_sql = record.get("final_sql", "")
                                    if final_sql:
                                        try:
                                            final_result = run_sqlite_query(db_path, final_sql)
                                            record["eval_final_result"] = truncate_for_json(final_result)
                                            is_match, diff_summary = compare_query_results(gold_sql, original_gold_result, final_result)
                                            record["eval_final_is_match"] = is_match
                                            record["eval_final_diff_summary"] = diff_summary
                                        except Exception as final_e:
                                            record["eval_final_is_match"] = False
                                            record["eval_final_diff_summary"] = f"FINAL_EVAL_ERROR: {repr(final_e)}"
                            except Exception as outer_e:
                                record["eval_diff_summary"] = f"OUTER_EVAL_ERROR: {repr(outer_e)}"

                except Exception as e:
                    record["status"] = "error"
                    record["error"] = f"Workflow error: {repr(e)}"
                    print(f"[Error] Workflow error {idx}: {question_id} - {repr(e)}", flush=True)

            except Exception as e:
                record["status"] = "error"
                record["error"] = f"Processing error: {repr(e)}"
                print(f"[Error] Process questions {idx} error: {repr(e)}", flush=True)

            elapsed = time.time() - t0
            record["time_sec"] = elapsed

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()

            status_symbol = {
                "completed": "[OK]",
                "timeout": "[TMO]",
                "error": "[ERR]",
                "dry_run_complete": "[DRY]"
            }.get(record.get("status", ""), "[?]")

            print(f"{status_symbol} Completed questions {idx}: {question_id} ({elapsed:.1f}s, {record.get('workflow_type', '')})", flush=True)

            if not args.keep_temp:
                sandbox_db = record.get("db_path", "")
                sandbox_recovery = os.path.join("temp_env", "exp_baseline_a", f"{question_id}_recovery.json") if question_id is not None else ""
                if sandbox_db:
                    safe_remove_sandbox(sandbox_db, sandbox_recovery)
                else:
                    safe_remove_sandbox(
                        os.path.join("temp_env", "exp_baseline_a", f"{question_id}_{db_id}.sqlite"),
                        sandbox_recovery
                    )

            if args.batch_rest > 0 and batch_num % 5 == 0:
                print(f"[Batch] Rest {args.batch_rest} seconds...", flush=True)
                time.sleep(args.batch_rest)

    print("\n" + "=" * 70)
    print(f"[Done] Baseline A experiment completed!")
    print(f"[Results] Results saved to: {out_dir}")
    print(f"[Time] End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
