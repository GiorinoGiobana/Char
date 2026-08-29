#!/usr/bin/env python3
"""
Ablation Experiment Plan 2: Based on full experiment, remove the SQL regeneration part

Key design:
1. Router generates initial SQL and executes it, gets error feedback (same as full experiment)
2. Pass error information to Worker (same as full experiment)
3. Worker combines error information + recover.json for repair (same as full experiment)
4. After repair, do not generate new SQL, directly use initial SQL on repaired DB (ablation point)
5. Evaluate repair success rate and initial SQL Text-to-SQL accuracy

Compared with full experiment: can evaluate the value of "regenerate query SQL after repair" step
"""

import json
import os
import sys
import re
import argparse
import time
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime

# Add project root directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_llm
from src.tools import run_sqlite_query, read_json_file, get_db_schema, write_json_file
from src.eval_utils import compare_query_results, truncate_for_json
from src.environment import DEV_DATABASES_DIR, DEV_MODIFIED_JSON, setup_sandbox
from langchain_core.messages import SystemMessage, HumanMessage


# Ablation Plan 2 Worker Prompt - only repair, do not generate query SQL
# But includes error feedback information (same as full experiment)
REUSE_SQL_CASE1_PROMPT = """You are a Database Architect (Ablation Experiment - Reuse SQL).
The table required for the query is MISSING.

Input Information:
1. Error Feedback from Router:
   - Initial SQL attempted: {initial_query_sql}
   - Execution Result/Error: {initial_query_result}

2. Recovery JSON Context:
   - case_type: "missing_table"
   - target_table: The name of the table to restore.
   - table_schema: The CREATE TABLE statement.
   - data_payload: A list of dictionaries containing the table data (may be a SUBSET of columns and/or rows).
   - payload_columns (optional): If present, it lists the intended columns included in each row of data_payload.

IMPORTANT: The data_payload contains ONLY a SUBSET of the original table data. You should ONLY insert the rows provided in data_payload, do NOT try to insert the complete original table.

Your Goal:
1. Analyze the Error Feedback to understand what went wrong with the initial query.
2. Read the Recovery JSON Context.
3. Generate SQLite `CREATE TABLE` statements based on `table_schema`.
4. Generate `INSERT INTO` statements using `data_payload` to restore the data.
   - Do NOT assume `data_payload` contains all columns from the original schema.
   - Always generate inserts with explicit column lists.
   - ONLY insert the rows from data_payload, do NOT add any additional rows.
5. Execute these statements to fix the DB.
6. After repairs succeed, output "REPAIR COMPLETED" - the initial query SQL will be reused for final query.

Mandatory step protocol (STRICT):
- You must output SQL in ```sql``` blocks only.
- First output ONLY repair SQL blocks (CREATE/INSERT/ALTER/UPDATE). No natural language.
- After repairs succeed, output exactly one line: REPAIR COMPLETED
- DO NOT generate a new SELECT query - the initial query SQL will be automatically reused.
"""

REUSE_SQL_CASE3_PROMPT = """You are a Data Forensic Specialist (Ablation Experiment - Reuse SQL).
The table structure is correct, but specific data values have been TAMPERED (e.g., set to NULL or 0).

Input Information:
1. Error Feedback from Router:
   - Initial SQL attempted: {initial_query_sql}
   - Execution Result/Error: {initial_query_result}

2. Recovery JSON Context:
   - case_type: "data_corruption"
   - target_table: The table name.
   - primary_key: The column to use for identifying rows (WHERE clause).
   - columns_to_fix: List of columns that were corrupted and need restoration.
   - data_payload: A list of dictionaries. Each dict contains the Primary Key value and the correct values for `columns_to_fix`.

Your Goal:
1. Analyze the Error Feedback to understand what data corruption caused the initial query to fail.
2. Read the Recovery JSON Context.
3. Iterate through `data_payload`. For each item:
   - Construct an `UPDATE` statement.
   - SET the columns in `columns_to_fix` to their correct values from the payload.
   - WHERE `primary_key` equals the payload's primary key value.
4. Execute the fixes.
5. After repairs succeed, output "REPAIR COMPLETED" - the initial query SQL will be reused for final query.

Mandatory step protocol (STRICT):
- You must output SQL in ```sql``` blocks only.
- First output ONLY repair SQL blocks (UPDATE/ALTER/CREATE/INSERT). No natural language.
- After repairs succeed, output exactly one line: REPAIR COMPLETED
- DO NOT generate a new SELECT query - the initial query SQL will be automatically reused.
"""

REUSE_SQL_CASE4_PROMPT = """You are a Schema Migration Specialist (Ablation Experiment - Reuse SQL).
A specific column required for the query is MISSING from the table.

Input Information:
1. Error Feedback from Router:
   - Initial SQL attempted: {initial_query_sql}
   - Execution Result/Error: {initial_query_result}

2. Recovery JSON Context:
   - case_type: "missing_column"
   - target_table: The table name.
   - primary_key: The column to use for matching rows.
   - columns_to_fix: List containing the name of the missing column.
   - data_payload: A list of dictionaries. Each dict contains the Primary Key value and the value of the missing column.

IMPORTANT: The data_payload contains ONLY a SUBSET of rows from the original table. You should ONLY update rows provided in data_payload, do NOT try to update all rows in the table.

Your Goal:
1. Analyze the Error Feedback to understand which column is missing.
2. Read the Recovery JSON Context.
3. Generate `ALTER TABLE ... ADD COLUMN` statement for the column in `columns_to_fix`.
4. Generate `UPDATE` statements to populate this new column using `data_payload` and `primary_key`.
   - Only update rows present in `data_payload`.
5. Execute the fixes.
6. After repairs succeed, output "REPAIR COMPLETED" - the initial query SQL will be reused for final query.

Mandatory step protocol (STRICT):
- You must output SQL in ```sql``` blocks only.
- First output ONLY repair SQL blocks (ALTER/UPDATE/CREATE/INSERT). No natural language.
- After repairs succeed, output exactly one line: REPAIR COMPLETED
- DO NOT generate a new SELECT query - the initial query SQL will be automatically reused.
"""

REUSE_SQL_CASE2_PROMPT = """You are a SQL Query Expert (Ablation Experiment - Reuse SQL).
The database is intact and working normally. No repairs are needed.

Input Information:
1. Initial SQL attempted: {initial_query_sql}
2. Execution Result: {initial_query_result}

Your Goal:
Simply output "REPAIR COMPLETED" to indicate the database is ready for queries.
The initial query SQL will be automatically reused for final query.
"""


def _extract_sql(content: str) -> str:
    """Helper to extract SQL from LLM response"""
    match = re.search(r"```sql\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return content.replace("```sql", "").replace("```", "").strip()


def _extract_sql_candidates(content: str) -> List[str]:
    """Extract all SQL candidates from content"""
    blocks = re.findall(r"```sql\s*(.*?)```", content, flags=re.IGNORECASE | re.DOTALL)
    blocks = [b.strip() for b in blocks if b.strip()]
    if blocks:
        return blocks

    upper = content.upper()
    keywords = ["SELECT ", "WITH ", "PRAGMA ", "CREATE ", "INSERT ", "UPDATE ", "ALTER "]
    first_pos = None
    for k in keywords:
        pos = upper.find(k)
        if pos != -1:
            first_pos = pos if first_pos is None else min(first_pos, pos)

    if first_pos is None:
        return []

    candidate = content[first_pos:].strip()
    candidate = candidate.replace("```", "").strip()
    return [candidate] if candidate else []


def _split_sql_statements(sql: str) -> List[str]:
    """Split SQL into individual statements"""
    statements: List[str] = []
    buf: List[str] = []
    quote: Optional[str] = None
    i = 0
    while i < len(sql):
        ch = sql[i]
        if quote is not None:
            buf.append(ch)
            if ch == quote:
                if quote == "'" and i + 1 < len(sql) and sql[i + 1] == "'":
                    buf.append(sql[i + 1])
                    i += 2
                    continue
                quote = None
            i += 1
            continue

        if ch in ("'", '"', "`"):
            quote = ch
            buf.append(ch)
            i += 1
            continue

        if ch == ";":
            stmt = "".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
            i += 1
            continue

        buf.append(ch)
        i += 1

    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


def generate_initial_query(question: str, db_path: str) -> tuple[str, Any]:
    """Generate initial query SQL and execute, get error feedback (same as full experiment)"""
    llm = get_llm("glm4.7")
    schema = get_db_schema(db_path)
    
    gen_sql_prompt = f"""Generate a SQLite query for the following question on schema {schema}:
Question: {question}
Output ONLY the SQL."""
    
    gen_sql_response = llm.invoke(gen_sql_prompt)
    initial_sql = _extract_sql(gen_sql_response.content)
    
    print(f"[ReuseSQL] Initial Try SQL:\n{initial_sql}\n")
    result = run_sqlite_query(db_path, initial_sql)
    
    return initial_sql, result


def router_node(question: str, db_path: str, recovery_json_path: str, 
                initial_result: Any) -> str:
    """
    Router node - determine workflow type based on initial query result and recover.json
    Same as full experiment
    """
    llm = get_llm("glm4.7")
    
    # Check if recovery JSON exists
    has_json = False
    if os.path.exists(recovery_json_path):
        try:
            meta = read_json_file(recovery_json_path)
            has_json = bool(meta.get("case_type"))
        except Exception:
            has_json = False
    
    # Prepare Metadata Hint
    recovery_metadata = "Unavailable"
    if has_json:
        try:
            with open(recovery_json_path, 'r') as f:
                meta = json.load(f)
                recovery_metadata = json.dumps({k: v for k, v in meta.items() if k != 'data_payload'})
        except:
            pass

    # LLM diagnosis
    result_str = str(initial_result)
    if len(result_str) > 2000:
        result_str = result_str[:2000] + "... [TRUNCATED]"
    
    # Use simplified routing prompt
    router_prompt = f"""You are an elite SQL Resilience Architect.
Analyze the user's SQL query execution on a potentially corrupted database.

Inputs:
1. User Query: {question}
2. Execution Error/Result: {result_str}
3. Recovery Context Available: {has_json} (True/False)
4. Recovery Context Metadata: {recovery_metadata}

Instructions:
- If `Recovery Context Available` is False, output: "CASE_2_NORMAL"
- If `Recovery Context Available` is True:
  - Check Metadata `case_type`:
    - "missing_table" → "CASE_1_MISSING_TABLE"
    - "missing_column" → "CASE_4_MISSING_COLUMN"
    - "data_corruption" → "CASE_3_CORRUPTED_DATA"
  - If Metadata is unavailable, analyze the Error/Result

Return ONLY the status string."""
    
    decision = llm.invoke([HumanMessage(content=router_prompt)]).content.strip()
    
    # Clean LLM output
    workflow_type = "CASE_2_NORMAL"
    if "CASE_1" in decision: workflow_type = "CASE_1_MISSING_TABLE"
    elif "CASE_3" in decision: workflow_type = "CASE_3_CORRUPTED_DATA"
    elif "CASE_4" in decision: workflow_type = "CASE_4_MISSING_COLUMN"
    
    # Strict Fallback Logic
    if has_json and workflow_type == "CASE_2_NORMAL":
        print("[ReuseSQL Router] Warning: LLM predicted Normal but Recovery Context exists. Forcing CASE_3_CORRUPTED_DATA.")
        workflow_type = "CASE_3_CORRUPTED_DATA"
    
    print(f"[ReuseSQL Router] Diagnosis: {workflow_type}")
    
    return workflow_type


def run_reuse_sql_workflow(question: str, db_path: str, recovery_json_path: str,
                           workflow_type: str, initial_sql: str, initial_result: Any) -> Dict[str, Any]:
    """
    Run reuse SQL repair experiment
    Difference from full experiment: only repair, do not generate new SQL
    """
    llm = get_llm("glm4.7")
    schema = get_db_schema(db_path)
    
    # Prepare context
    context = {}
    context_str = "{}"
    has_recovery_json = False
    if os.path.exists(recovery_json_path):
        try:
            context = read_json_file(recovery_json_path)
            context_str = json.dumps(context, indent=2)
            has_recovery_json = True
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
        except Exception:
            context_str = "{}"
    
    # Prepare error feedback information
    initial_result_str = str(initial_result)
    if len(initial_result_str) > 1000:
        initial_result_str = initial_result_str[:1000] + "... [TRUNCATED]"
    
    # Select corresponding prompt (includes error feedback, but requires only repair without generating query SQL)
    prompt_map = {
        "CASE_1_MISSING_TABLE": REUSE_SQL_CASE1_PROMPT,
        "CASE_2_NORMAL": REUSE_SQL_CASE2_PROMPT,
        "CASE_3_CORRUPTED_DATA": REUSE_SQL_CASE3_PROMPT,
        "CASE_4_MISSING_COLUMN": REUSE_SQL_CASE4_PROMPT
    }
    prompt_template = prompt_map.get(workflow_type, REUSE_SQL_CASE2_PROMPT)
    
    # Format prompt, pass in error feedback
    formatted_prompt = prompt_template.format(
        initial_query_sql=initial_sql,
        initial_query_result=initial_result_str
    )
    
    messages = [
        SystemMessage(content=formatted_prompt),
        HumanMessage(content=f"""
        Context (Recovery JSON): {context_str}
        DB Schema: {schema}
        Question: {question}
        """)
    ]
    
    repair_sql_log = []
    repair_completed = False
    
    # Execute repair (max 6 turns)
    for turn in range(6):
        response = llm.invoke(messages)
        content = response.content
        messages.append(response)
        
        # Check if repair is completed
        if "REPAIR COMPLETED" in content.upper():
            repair_completed = True
            break
        
        # Extract and execute SQL
        sql_candidates = _extract_sql_candidates(content)
        if not sql_candidates:
            messages.append(HumanMessage(content="Please output the SQL statement to repair the database (in ```sql``` block), or output REPAIR COMPLETED to indicate repair is done."))
            continue
        
        for sql in sql_candidates:
            for stmt in _split_sql_statements(sql):
                upper_stmt = stmt.strip().upper()
                is_repair = (upper_stmt.startswith("CREATE") or 
                            upper_stmt.startswith("INSERT") or 
                            upper_stmt.startswith("UPDATE") or 
                            upper_stmt.startswith("ALTER"))
                
                if is_repair:
                    print(f"[ReuseSQL] Executing Repair SQL:\n{stmt}\n")
                    repair_sql_log.append(stmt)
                    result = run_sqlite_query(db_path, stmt)
                    messages.append(HumanMessage(content=f"Repair Execution Result: {result}"))
    
    return {
        "repair_completed": repair_completed,
        "repair_sql": ";\n".join(repair_sql_log),
        "turns_used": turn + 1,
        "has_recovery_json": has_recovery_json
    }


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
    parser = argparse.ArgumentParser(description="Ablation Experiment 2: Reuse SQL (Based on Full Experiment)")
    parser.add_argument("--out-dir", type=str, default="", help="Output directory")
    parser.add_argument("--start", type=int, default=0, help="Start question index")
    parser.add_argument("--end", type=int, default=None, help="End question index (exclusive)")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of questions per batch")
    parser.add_argument("--batch-rest", type=int, default=5, help="Rest seconds between batches")
    parser.add_argument("--force", action="store_true", help="Overwrite processed questions")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary files")
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
        out_dir = os.path.join("results", f"ablation_reuse_sql_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    _ensure_dir(out_dir)

    recovery_out_dir = os.path.join(out_dir, "recovery_json")
    _ensure_dir(recovery_out_dir)

    results_path = os.path.join(out_dir, "results.jsonl")
    meta_path = os.path.join(out_dir, "meta.json")

    # Save metadata
    meta = {
        "experiment": "ablation_reuse_sql",
        "description": "Ablation Plan 2: Based on full experiment, do not generate new SQL after repair, directly reuse initial SQL",
        "model": "GLM-4.7",
        "dev_modified_json": DEV_MODIFIED_JSON,
        "start_index": start,
        "end_index": end,
        "total_questions": total_questions,
        "batch_size": args.batch_size,
        "batch_rest": args.batch_rest,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "force": bool(args.force),
        "keep_temp": bool(args.keep_temp),
    }
    write_json_file(meta_path, meta)

    # Load processed questions
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

    print(f"[Start] Ablation Experiment Plan 2 (Reuse SQL): Questions{start}-{end-1}，total{limit}")
    print(f"[Output] Output directory: {out_dir}")
    print(f"[Feature] Based on full experiment, do not generate new SQL after repair, directly reuse initial SQL")
    print(f"[Time] Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)

    slice_items = data[start:end]
    total = len(slice_items)
    batch_num = 0

    with open(results_path, "a", encoding="utf-8") as f_out:
        for idx, item in enumerate(slice_items, start=start):
            question_id = item.get("question_id")
            question = str(item.get("question", ""))
            db_id = item.get("db_id", "")
            gold_sql = item.get("SQL", "")
            already_written = False

            record: dict = {
                "index": idx,
                "question_id": question_id,
                "db_id": db_id,
                "question": question,
                "gold_sql": gold_sql,
                "status": "unknown",
                "error": "",
                "workflow_type": "",
                "initial_query_sql": "",
                "initial_query_result": "",
                "final_answer": "",
                "final_sql": "",  # Ablation point: reuse initial_query_sql
                "final_query_result": "",
                "repair_sql": "",
                "recovery_json_saved_to": "",
                "time_sec": 0.0,
                "eval_is_match": False,
                "eval_diff_summary": "",
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
                    print(f"[Skip] Question {idx}: {question_id} (already processed)")
                    continue

                print(f"[Process] Question {idx}: {question_id} - {question[:50]}...")

                # Save gold SQL result on original database as baseline before repair
                original_gold_result = None
                if gold_sql:
                    try:
                        original_db_path = os.path.join(DEV_DATABASES_DIR, db_id, f"{db_id}.sqlite")
                        original_gold_result = run_sqlite_query(original_db_path, gold_sql)
                    except Exception:
                        pass

                # Set up sandbox environment
                sandbox_dir = os.path.join("temp_env", f"sandbox_{question_id}")
                _ensure_dir(sandbox_dir)
                
                db_path, recovery_json_path = setup_sandbox(question)

                # Copy recovery.json to results directory
                recovery_dst_path = os.path.join(recovery_out_dir, f"recovery_{question_id}.json")
                _copy_recovery_json(recovery_json_path, recovery_dst_path)
                record["recovery_json_saved_to"] = recovery_dst_path

                # 1. Generate initial query SQL and get error feedback（Same as full experiment）
                initial_sql, initial_result = generate_initial_query(question, db_path)
                record["initial_query_sql"] = initial_sql
                record["initial_query_result"] = truncate_for_json(initial_result)
                
                # 2. Router node - determine workflow type based on error feedback and recover.json（Same as full experiment）
                workflow_type = router_node(question, db_path, recovery_json_path, initial_result)
                record["workflow_type"] = workflow_type
                
                # 3. Run reuse SQL repair (only repair, do not generate new SQL)
                try:
                    fix_result = run_reuse_sql_workflow(
                        question=question,
                        db_path=db_path,
                        recovery_json_path=recovery_json_path,
                        workflow_type=workflow_type,
                        initial_sql=initial_sql,
                        initial_result=initial_result
                    )
                    
                    record.update({
                        "repair_sql": fix_result.get("repair_sql", ""),
                        "repair_completed": fix_result.get("repair_completed", False),
                        "has_recovery_json": fix_result.get("has_recovery_json", False),
                        "status": "completed",
                    })

                    # Evaluation 1: Repair success rate（Same as full experiment）
                    if original_gold_result is not None and gold_sql:
                        record["eval_gold_result"] = truncate_for_json(original_gold_result)
                        
                        # Check if Gold SQL executes successfully on original database
                        if isinstance(original_gold_result, str) and original_gold_result.startswith("Error:"):
                            record["eval_is_match"] = False
                            record["eval_diff_summary"] = "INVALID_GOLD_SQL"
                        else:
                            # CASE_2_NORMAL does not need repair
                            if workflow_type == "CASE_2_NORMAL":
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

                    # Evaluation 2: Text-to-SQL accuracy (ablation point: reuse initial SQL, do not generate new SQL)
                    # Execute initial SQL on repaired database
                    if initial_sql:
                        try:
                            final_result = run_sqlite_query(db_path, initial_sql)
                            record["eval_final_result"] = truncate_for_json(final_result)
                            record["final_query_result"] = truncate_for_json(final_result)
                            
                            # Compare initial SQL result with gold SQL result
                            if gold_sql and original_gold_result is not None:
                                if isinstance(original_gold_result, str) and original_gold_result.startswith("Error:"):
                                    record["eval_final_is_match"] = False
                                    record["eval_final_diff_summary"] = "INVALID_GOLD_SQL"
                                else:
                                    gold_res_raw = run_sqlite_query(db_path, gold_sql)
                                    final_is_match, final_diff_summary = compare_query_results(
                                        gold_sql, gold_res_raw, final_result
                                    )
                                    record["eval_final_is_match"] = bool(final_is_match)
                                    record["eval_final_diff_summary"] = final_diff_summary
                            else:
                                record["eval_final_is_match"] = False
                                record["eval_final_diff_summary"] = "NO_GOLD_SQL_OR_RESULT"
                        except Exception as e:
                            record["eval_final_is_match"] = False
                            record["eval_final_diff_summary"] = f"FINAL_EVAL_ERROR: {repr(e)}"
                    else:
                        record["eval_final_is_match"] = False
                        record["eval_final_diff_summary"] = "MISSING_INITIAL_SQL"
                    
                    # Ablation point: final_sql reuses initial_sql
                    record["final_sql"] = initial_sql
                    record["final_answer"] = "REUSE_INITIAL_SQL"

                except Exception as e:
                    record["status"] = "error"
                    record["error"] = f"Workflow error: {repr(e)}"
                    print(f"[Error] Workflow error {idx}: {question_id} - {repr(e)}")

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
                repair_success = "[Success]" if record.get("eval_is_match", False) else "[Fail]"
                final_success = "[Success]" if record.get("eval_final_is_match", False) else "[Fail]"
                print(f"   {idx:4d}: {repair_success}repair {final_success}query | Time: {record['time_sec']:.1f}s | {record['workflow_type']}")
                
                # Rest between batches
                batch_num += 1
                if batch_num % args.batch_size == 0 and batch_num < total:
                    completed = batch_num
                    remaining = total - completed
                    print(f"\n[Batch] Batch completed: {completed}/{total} ({completed/total*100:.1f}%)")
                    print(f"[Rest] Rest {args.batch_rest}seconds before continuing...")
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
                
                print(f"[Error] Error {idx}: {question_id} - {repr(e)}")

    print("-" * 60)
    print(f"[Completed] Ablation Experiment Plan 2 (Reuse SQL) completed!Results saved in: {out_dir}")
    print(f"[Time] End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
