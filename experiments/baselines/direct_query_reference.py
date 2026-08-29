#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline A: Direct Query on Perturbed DB with Clues (No repair, query directly with clues)

The purpose of this Baseline is to evaluate: without any database repair, directly using clues from recovery.json
to generate SELECT queries on the perturbed database, what performance can be achieved.

Workflow:
1. Read question (Q) + perturbed Schema + JSON clues
2. Skip all repair (DDL/DML) steps
3. Directly try to generate a DQL (SELECT query) based on clues on the current "incomplete/perturbed" database
4. Execute the query directly on the perturbed database and evaluate

Core metrics:
- Execution Accuracy (EX): whether execution results match Gold SQL results on complete database
- Execution Failure Rate: proportion of SQL execution errors
- Empty Result Rate: proportion of successful SQL execution but empty result set
"""

import argparse
import json
import os
import re
import sys
import time
import signal
import sqlite3
import io
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from contextlib import contextmanager

# Set stdout/stderr encoding to avoid Windows encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_llm
from src.environment import DEV_DATABASES_DIR, DEV_MODIFIED_JSON, setup_sandbox
from src.eval_utils import compare_query_results, truncate_for_json
from src.tools import read_json_file, run_sqlite_query, write_json_file, get_db_schema
from langchain_core.messages import SystemMessage, HumanMessage


# ==================== Baseline A Dedicated Prompt Templates ====================

BASELINE_A_SYSTEM_PROMPT = """You are an expert SQL Query Generator specialized in querying potentially corrupted databases.

Your task is to generate a SELECT query that answers the user's question using the CURRENT database state.

IMPORTANT CONSTRAINTS:
1. You are ONLY allowed to generate SELECT queries. DO NOT generate CREATE, ALTER, INSERT, UPDATE, DELETE, DROP, or any other DDL/DML statements.
2. The current database structure or data may be incomplete or corrupted (e.g., missing tables, missing columns, incorrect data).
3. You have access to recovery clues in JSON format that contain information about what is missing or corrupted.

Your Goal:
1. Analyze the user's question to understand what information is being requested.
2. Examine the provided database schema (which may be incomplete/corrupted).
3. Study the recovery JSON clues carefully - they contain valuable information about:
   - Missing tables and their data
   - Missing columns and their values
   - Corrupted data and correct values
4. Generate a SQLite SELECT query that:
   - Works with the CURRENT database state (even if incomplete)
   - Uses the clues to work around missing/corrupted data when possible
   - Attempts to answer the user's question as best as possible
   - Uses proper JOIN syntax, WHERE clauses, GROUP BY, ORDER BY, LIMIT as needed

Strategy Tips:
- If a table is missing but mentioned in clues, you cannot query it directly. Try to work with available tables.
- If a column is missing, avoid using it in SELECT, WHERE, or JOIN conditions.
- If data is corrupted (NULL/0 when it shouldn't be), consider using IS NOT NULL checks or alternative approaches.
- Use the clues to understand what data SHOULD be there, even if it's not currently available.

Output Format:
- Output ONLY ONE SELECT query inside ```sql``` code block.
- Do NOT output any explanations, comments, or natural language.
- Do NOT output multiple queries.
- The query should be executable on the current (potentially corrupted) database.

Example Output:
```sql
SELECT column1, column2 FROM available_table WHERE condition;
```
"""


def _extract_sql(content: str) -> str:
    """Helper to extract SQL from LLM response"""
    match = re.search(r"```sql\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return content.replace("```sql", "").replace("```", "").strip()


def _extract_select_sql(content: str) -> Optional[str]:
    """
    Extract SELECT SQL statement from LLM response
    Strictly limit to SELECT statements only
    """
    # First try to extract code block
    sql = _extract_sql(content)
    
    if not sql:
        return None
    
    # Clean SQL
    sql = sql.strip()
    
    # Check if it is a SELECT statement (only SELECT allowed)
    upper_sql = sql.upper()
    if not (upper_sql.startswith("SELECT") or upper_sql.startswith("WITH")):
        # If not SELECT, try to find SELECT in content
        select_match = re.search(r'(SELECT|WITH)\s+', content, re.IGNORECASE)
        if select_match:
            sql = content[select_match.start():].strip()
            # Remove possible subsequent non-SQL content
            sql = re.split(r'\n\n', sql)[0].strip()
        else:
            return None
    
    # Remove trailing semicolons and extra content
    sql = sql.rstrip(';').strip()
    
    # Validate: ensure not DDL/DML
    forbidden_keywords = ['CREATE', 'ALTER', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'TRUNCATE']
    first_word = sql.upper().split()[0] if sql.split() else ""
    if first_word in forbidden_keywords:
        return None
    
    return sql


class TimeoutException(Exception):
    """SQL execution timeout exception"""
    pass


@contextmanager
def time_limit(seconds: int):
    """Context manager: limit code block execution time"""
    def signal_handler(signum, frame):
        raise TimeoutException(f"Execution timed out after {seconds} seconds")
    
    # Windows does not support signal.SIGALRM, use threading approach
    import threading
    timer = threading.Timer(seconds, lambda: (_ for _ in ()).throw(TimeoutException("Timeout")))
    timer.start()
    try:
        yield
    finally:
        timer.cancel()


def execute_sql_with_timeout(db_path: str, sql: str, timeout: int = 120) -> Union[List[Any], str]:
    """
    Execute SQL query with timeout protection
    
    Args:
        db_path: Database path
        sql: SQL query statement
        timeout: Timeout in seconds, default 120
        
    Returns:
        Query result list or error message string
    """
    if not os.path.exists(db_path):
        return f"Error: Database file not found: {db_path}"
    
    result_container = []
    exception_container = []
    
    def execute_query():
        conn = None
        try:
            conn = sqlite3.connect(db_path, timeout=timeout)
            cursor = conn.cursor()
            cursor.execute(sql)
            
            if sql.strip().upper().startswith("SELECT") or sql.strip().upper().startswith("WITH"):
                results = cursor.fetchall()
                result_container.append(results)
            else:
                result_container.append("Error: Only SELECT queries are allowed in Baseline A")
        except Exception as e:
            exception_container.append(str(e))
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass
    
    # Use threading for timeout
    import threading
    query_thread = threading.Thread(target=execute_query)
    query_thread.daemon = True
    query_thread.start()
    query_thread.join(timeout=timeout)
    
    if query_thread.is_alive():
        return f"Error: Query execution timed out after {timeout} seconds"
    
    if exception_container:
        return f"Error: {exception_container[0]}"
    
    if result_container:
        return result_container[0]
    
    return "Error: Unknown error during query execution"


def generate_direct_query(
    question: str,
    db_path: str,
    recovery_json_path: str,
    max_retries: int = 2
) -> Dict[str, Any]:
    """
    Baseline A core function: directly generate query, do not repair database
    
    Args:
        question: User question
        db_path: Perturbed database path
        recovery_json_path: recovery.json path
        max_retries: Max retries when LLM generation fails
        
    Returns:
        Dictionary containing generation results
    """
    llm = get_llm("chatanywhere2")  # Using GLM-4.7 thinking mode
    schema = get_db_schema(db_path)
    
    # Prepare recovery.json context
    context = {}
    context_str = "{}"
    has_recovery_json = False
    
    if os.path.exists(recovery_json_path):
        try:
            context = read_json_file(recovery_json_path)
            context_str = json.dumps(context, indent=2)
            has_recovery_json = True
            
            # Truncate overly long context
            if len(context_str) > 5000:
                if 'data_payload' in context and isinstance(context['data_payload'], list):
                    original_len = len(context['data_payload'])
                    if original_len > 10:
                        truncated_payload = context['data_payload'][:10]
                        context_for_prompt = context.copy()
                        context_for_prompt['data_payload'] = truncated_payload
                        context_for_prompt['note'] = f"Data truncated. Showing first 10 of {original_len} rows."
                        context_str = json.dumps(context_for_prompt, indent=2)
                if len(context_str) > 10000:
                    context_str = context_str[:10000] + "... [TRUNCATED]"
        except Exception as e:
            context_str = f"Error reading recovery.json: {str(e)}"
    
    # Build Prompt
    prompt = f"""{BASELINE_A_SYSTEM_PROMPT}

User Question: {question}

Current Database Schema (may be incomplete/corrupted):
{schema}

Recovery JSON Clues (information about what's missing/corrupted):
{context_str}

Generate a SELECT query to answer the question using the current database state.
Remember: ONLY output ONE SELECT query in ```sql``` block. No other SQL types allowed.
"""
    
    generated_sql = None
    llm_response = ""
    generation_attempts = 0
    
    # Try generating SQL (with retries)
    for attempt in range(max_retries + 1):
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            llm_response = response.content
            generation_attempts += 1
            
            # Extract SQL
            generated_sql = _extract_select_sql(llm_response)
            
            if generated_sql:
                print(f"[Baseline A] Generated SQL (attempt {generation_attempts}):\n{generated_sql}\n")
                break
            else:
                print(f"[Baseline A] No valid SELECT SQL found in attempt {generation_attempts}")
                if attempt < max_retries:
                    prompt += "\n\nYour previous response did not contain a valid SELECT query. Please output ONLY a SELECT query inside ```sql``` block."
                    
        except Exception as e:
            print(f"[Baseline A] LLM invocation error in attempt {generation_attempts}: {e}")
            if attempt < max_retries:
                continue
    
    # If no SQL generated, return failure result
    if not generated_sql:
        return {
            "generated_sql": "",
            "execution_result": "Error: Failed to generate valid SELECT SQL",
            "execution_status": "generation_failed",
            "has_recovery_json": has_recovery_json,
            "llm_response": llm_response,
            "generation_attempts": generation_attempts
        }
    
    # Execute generated SQL (on perturbed database)
    print(f"[Baseline A] Executing SQL on perturbed database...")
    execution_result = execute_sql_with_timeout(db_path, generated_sql, timeout=120)
    
    # Determine execution status
    execution_status = "success"
    is_empty_result = False
    
    if isinstance(execution_result, str) and execution_result.startswith("Error:"):
        execution_status = "execution_error"
    elif isinstance(execution_result, list):
        if len(execution_result) == 0:
            execution_status = "empty_result"
            is_empty_result = True
    
    print(f"[Baseline A] Execution Status: {execution_status}")
    print(f"[Baseline A] Result Preview: {truncate_for_json(str(execution_result))[:200]}...")
    
    return {
        "generated_sql": generated_sql,
        "execution_result": execution_result,
        "execution_status": execution_status,
        "is_empty_result": is_empty_result,
        "has_recovery_json": has_recovery_json,
        "llm_response": llm_response,
        "generation_attempts": generation_attempts
    }


def evaluate_baseline_a(
    db_path: str,
    gold_sql: str,
    generated_sql: str,
    execution_result: Any,
    db_id: str
) -> Dict[str, Any]:
    """
    Evaluate Baseline A results
    
    Core metrics:
    1. Execution Accuracy (EX): Whether result matches Gold SQL on complete database
    2. Execution Failure: Whether SQL execution errors
    3. Empty Result: Whether SQL returns empty result
    """
    eval_result = {
        "ex_accuracy": 0,  # Execution Accuracy: 0 or 1
        "execution_failed": False,
        "empty_result": False,
        "gold_result": None,
        "comparison_summary": ""
    }
    
    # 1. Execute Gold SQL on original (complete) database to get standard answer
    original_db_path = os.path.join(DEV_DATABASES_DIR, db_id, f"{db_id}.sqlite")
    gold_result = None
    
    if os.path.exists(original_db_path) and gold_sql:
        try:
            gold_result = run_sqlite_query(original_db_path, gold_sql)
            eval_result["gold_result"] = truncate_for_json(gold_result)
        except Exception as e:
            print(f"[Baseline A Eval] Warning: Failed to execute gold SQL on original DB: {e}")
    
    # 2. Check execution status
    if isinstance(execution_result, str) and execution_result.startswith("Error:"):
        eval_result["execution_failed"] = True
        eval_result["comparison_summary"] = f"EXECUTION_ERROR: {execution_result}"
        return eval_result
    
    if isinstance(execution_result, list) and len(execution_result) == 0:
        eval_result["empty_result"] = True
    
    # 3. Calculate Execution Accuracy (compare with Gold result)
    if gold_result is not None:
        is_match, diff_summary = compare_query_results(gold_sql, gold_result, execution_result)
        eval_result["ex_accuracy"] = 1 if is_match else 0
        eval_result["comparison_summary"] = diff_summary
    else:
        eval_result["comparison_summary"] = "NO_GOLD_RESULT_AVAILABLE"
    
    return eval_result


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_remove(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _copy_recovery_json(src_path: str, dst_path: str) -> bool:
    if not os.path.exists(src_path):
        return False
    data = read_json_file(src_path)
    write_json_file(dst_path, data)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Baseline A: Direct Query on Perturbed DB with Clues (No repair, query directly with clues)"
    )
    parser.add_argument("--out-dir", type=str, default="", help="Output directory")
    parser.add_argument("--start", type=int, default=0, help="Start question index")
    parser.add_argument("--end", type=int, default=1535, help="End question index (exclusive)")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of questions per batch")
    parser.add_argument("--batch-rest", type=int, default=5, help="Rest seconds between batches")
    parser.add_argument("--force", action="store_true", help="Overwrite processed questions")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary files")
    parser.add_argument("--max-retries", type=int, default=2, help="Max retries when SQL generation fails")
    args = parser.parse_args()

    # Load data
    with open(DEV_MODIFIED_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_questions = len(data)
    start = args.start
    end = args.end if args.end is not None else total_questions
    limit = end - start

    # Set output directory
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.path.join("results", "baseline_a_direct_query")
    _ensure_dir(out_dir)

    recovery_out_dir = os.path.join(out_dir, "recovery_json")
    _ensure_dir(recovery_out_dir)

    results_path = os.path.join(out_dir, "results.jsonl")
    meta_path = os.path.join(out_dir, "meta.json")

    # Save metadata
    meta = {
        "experiment": "baseline_a_direct_query",
        "description": "Baseline A: Direct Query on Perturbed DB with Clues (No repair, query directly with clues). Skip all repair steps, generate SELECT directly on perturbed DB.",
        "model": "GLM-4.7-thinking (chatanywhere2)",
        "dev_modified_json": DEV_MODIFIED_JSON,
        "start_index": start,
        "end_index": end,
        "total_questions": total_questions,
        "batch_size": args.batch_size,
        "batch_rest": args.batch_rest,
        "max_retries": args.max_retries,
        "timeout_seconds": 120,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "force": bool(args.force),
        "keep_temp": bool(args.keep_temp),
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
    print(f"[Range] Questions {start}-{end-1}, total {limit} ")
    print(f"[Output] Output directory: {out_dir}")
    print(f"[Config] Timeout: 120s | Max retries: {args.max_retries}")
    print(f"[Feature] Skip all repairs, directly generate SELECT query based on clues on perturbed database")
    print(f"[Time] Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    slice_items = data[start:end]
    total = len(slice_items)
    batch_num = 0
    
    # Statistics metrics
    stats = {
        "total": 0,
        "execution_failed": 0,
        "empty_result": 0,
        "ex_accuracy_sum": 0
    }

    with open(results_path, "a", encoding="utf-8") as f_out:
        for item in slice_items:
            question_id = item.get("question_id")
            idx = question_id if question_id is not None else 0

            if question_id in done_qids and not args.force:
                print(f"[Skip] Question {idx}: question_id={question_id} already processed")
                continue

            batch_num += 1
            t0 = time.time()
            
            print(f"\n[Process] Question {idx}: {question_id}")
            print(f"       Q: {item.get('question', '')[:80]}...")
            
            db_id = item.get("db_id", "")
            question = item.get("question", "")
            gold_sql = item.get("SQL", "")
            
            record = {
                "index": idx,
                "question_id": question_id,
                "db_id": db_id,
                "question": question,
                "gold_sql": gold_sql,
                "baseline": "A",
                "baseline_name": "Direct Query on Perturbed DB with Clues"
            }
            
            already_written = False
            
            try:
                # Set up sandbox environment（Create perturbed database）
                sandbox_dir = os.path.join("temp_env", f"baseline_a_sandbox_{question_id}")
                _ensure_dir(sandbox_dir)
                
                db_path, recovery_json_path = setup_sandbox(question)

                # Copy recovery.json to results directory
                recovery_dst_path = os.path.join(recovery_out_dir, f"recovery_{question_id}.json")
                _copy_recovery_json(recovery_json_path, recovery_dst_path)
                record["recovery_json_saved_to"] = recovery_dst_path

                # ========== Baseline A core workflow ==========
                # 1. Directly generate query (no repair)
                baseline_result = generate_direct_query(
                    question=question,
                    db_path=db_path,
                    recovery_json_path=recovery_json_path,
                    max_retries=args.max_retries
                )
                
                record.update({
                    "generated_sql": baseline_result.get("generated_sql", ""),
                    "execution_result": truncate_for_json(baseline_result.get("execution_result", "")),
                    "execution_status": baseline_result.get("execution_status", "unknown"),
                    "is_empty_result": baseline_result.get("is_empty_result", False),
                    "has_recovery_json": baseline_result.get("has_recovery_json", False),
                    "generation_attempts": baseline_result.get("generation_attempts", 0),
                    "status": "completed"
                })

                # 2. Evaluate results
                eval_result = evaluate_baseline_a(
                    db_path=db_path,
                    gold_sql=gold_sql,
                    generated_sql=baseline_result.get("generated_sql", ""),
                    execution_result=baseline_result.get("execution_result", ""),
                    db_id=db_id
                )
                
                record.update({
                    "ex_accuracy": eval_result.get("ex_accuracy", 0),
                    "execution_failed": eval_result.get("execution_failed", False),
                    "empty_result": eval_result.get("empty_result", False),
                    "gold_result": eval_result.get("gold_result"),
                    "comparison_summary": eval_result.get("comparison_summary", "")
                })
                
                # Update statistics
                stats["total"] += 1
                if eval_result.get("execution_failed"):
                    stats["execution_failed"] += 1
                if eval_result.get("empty_result"):
                    stats["empty_result"] += 1
                stats["ex_accuracy_sum"] += eval_result.get("ex_accuracy", 0)

                t1 = time.time()
                record["time_sec"] = t1 - t0
                
                # Print status
                ex_acc = eval_result.get("ex_accuracy", 0)
                exec_status = "✓" if not eval_result.get("execution_failed") else "✗EXEC_FAIL"
                empty_status = "EMPTY" if eval_result.get("empty_result") else ""
                match_status = "MATCH" if ex_acc == 1 else "MISMATCH"
                
                print(f"       Result: [{exec_status}] [{empty_status}] [{match_status}] | EX={ex_acc} | Time: {t1-t0:.1f}s")
                
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()
                already_written = True

                if not args.keep_temp:
                    _safe_remove(db_path)
                    _safe_remove(recovery_json_path)

                # Batch rest
                if batch_num % args.batch_size == 0:
                    current_ex_rate = stats["ex_accuracy_sum"] / stats["total"] * 100 if stats["total"] > 0 else 0
                    fail_rate = stats["execution_failed"] / stats["total"] * 100 if stats["total"] > 0 else 0
                    empty_rate = stats["empty_result"] / stats["total"] * 100 if stats["total"] > 0 else 0
                    
                    print(f"\n{'='*70}")
                    print(f"[Batch] Progress: {batch_num}/{total} ({100.0*batch_num/total:.1f}%)")
                    print(f"[Stats] EX accuracy: {current_ex_rate:.1f}% | Execution failure rate: {fail_rate:.1f}% | Empty result rate: {empty_rate:.1f}%")
                    print(f"[Rest] Rest {args.batch_rest} seconds before continuing...")
                    print(f"{'='*70}")
                    time.sleep(args.batch_rest)

            except Exception as e:
                t1 = time.time()
                record["status"] = "error"
                record["error"] = repr(e)
                record["time_sec"] = t1 - t0

                if not already_written:
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()
                
                print(f"       [Error] {repr(e)}")

            if not args.keep_temp:
                _safe_remove(db_path)
                _safe_remove(recovery_json_path)

    # Final statistics
    print("\n" + "=" * 70)
    print("[Done] Baseline A Experiment completed!")
    print(f"[Results] Saved in: {out_dir}")
    
    if stats["total"] > 0:
        final_ex_rate = stats["ex_accuracy_sum"] / stats["total"] * 100
        final_fail_rate = stats["execution_failed"] / stats["total"] * 100
        final_empty_rate = stats["empty_result"] / stats["total"] * 100
        
        print(f"\n[Final statistics]")
        print(f"  - Total samples: {stats['total']}")
        print(f"  - Execution Accuracy (EX): {final_ex_rate:.2f}% ({stats['ex_accuracy_sum']}/{stats['total']})")
        print(f"  - Execution Failure Rate: {final_fail_rate:.2f}% ({stats['execution_failed']}/{stats['total']})")
        print(f"  - Empty Result Rate: {final_empty_rate:.2f}% ({stats['empty_result']}/{stats['total']})")
    
    print(f"[Time] End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
