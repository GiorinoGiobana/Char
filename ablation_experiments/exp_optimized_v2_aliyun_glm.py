#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized V2: Worker performs initial query reasoning internally (max 1 turn)
- Router only determines CASE type
- Worker performs initial query reasoning and execution before repair
- Reduce context passing, lower cost
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_llm
from src.environment import DEV_DATABASES_DIR, DEV_MODIFIED_JSON, setup_sandbox
from src.eval_utils import compare_query_results, truncate_for_json
from src.tools import read_json_file, run_sqlite_query, write_json_file, get_db_schema
from langchain_core.messages import SystemMessage, HumanMessage

# Reuse original prompt templates
from src.prompts import (
    CASE1_PROMPT, CASE2_PROMPT, CASE3_PROMPT, CASE4_PROMPT
)

# Optimized Worker Prompt - includes initial query reasoning
OPTIMIZED_WORKER_PROMPT = """You are an expert Database Repair and SQL Generation Agent.

Your task is to:
1. First, generate and execute an initial query to understand the database state
2. Analyze any errors or issues
3. Repair the database if needed (using recovery.json)
4. Generate and execute the final query to answer the user's question

Rules:
- Use the provided recovery.json to fix database issues
- You have at most 1 turn to generate the initial query
- After fixing the database, generate the final query
- Output "FINAL ANSWER: <answer>" when done

Recovery JSON Context: {context_str}
DB Schema: {schema}
User Question: {question}

Think step by step:
1. Generate initial query
2. Execute it and check for errors
3. If errors, analyze and fix the database
4. Generate final query after fixes
5. Output FINAL ANSWER
"""


def _extract_sql(content: str) -> str:
    """Helper to extract SQL from LLM response"""
    match = re.search(r"```sql\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return content.replace("```sql", "").replace("```", "").strip()


def _extract_sql_candidates(content: str) -> List[str]:
    """Extract SQL candidates from content"""
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
    """Split SQL into statements"""
    statements = []
    buf = []
    quote = None
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


def router_node(question: str, db_path: str, recovery_json_path: str) -> str:
    """
    Router node - same as original experiment, LLM determines CASE type
    1. Generate initial SQL and execute
    2. Determine CASE type based on execution result and recovery.json
    """
    llm = get_llm("qwen3-coder-480b")
    schema = get_db_schema(db_path)
    
    # 1. Generate initial SQL and execute
    gen_sql_prompt = f"Generate a SQLite query for the following question on schema {schema}:\nQuestion: {question}\nOutput ONLY the SQL."
    gen_sql_response = llm.invoke(gen_sql_prompt)
    sql = _extract_sql(gen_sql_response.content)
    
    print(f"[Router] Initial Try SQL:\n{sql}\n")
    result = run_sqlite_query(db_path, sql)
    
    # 2. Check if recovery JSON exists
    has_json = False
    if os.path.exists(recovery_json_path):
        try:
            meta = read_json_file(recovery_json_path)
            has_json = bool(meta.get("case_type"))
        except Exception:
            has_json = False
    
    # 3. Prepare Metadata Hint
    recovery_metadata = "Unavailable"
    if has_json:
        try:
            with open(recovery_json_path, 'r') as f:
                meta = json.load(f)
                recovery_metadata = json.dumps({k: v for k, v in meta.items() if k != 'data_payload'})
        except:
            pass

    # 4. LLM diagnosis
    result_str = str(result)
    if len(result_str) > 2000:
        result_str = result_str[:2000] + "... [TRUNCATED]"
    
    # Use simplified Router Prompt
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
        print("[Router] Warning: LLM predicted Normal but Recovery Context exists. Forcing CASE_3_CORRUPTED_DATA.")
        workflow_type = "CASE_3_CORRUPTED_DATA"
    
    print(f"[Router] Diagnosis: {workflow_type}")
    
    return workflow_type


def optimized_worker(question: str, db_path: str, recovery_json_path: str,
                     workflow_type: str, max_initial_turns: int = 1) -> Dict[str, Any]:
    """
    Optimized Worker - performs initial query reasoning internally (max 1 turn)
    reducing context passing，reducing cost
    """
    llm = get_llm("glm4.7")
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
        except Exception:
            context_str = "{}"
    
    # Phase 1: Initial query reasoning (max 1 turn)
    initial_sql = ""
    initial_result = ""
    initial_query_executed = False
    
    # Build initial query generation prompt
    init_prompt = f"""Generate a SQLite query for this question:
Question: {question}
DB Schema: {schema}

Output ONLY the SQL query."""
    
    messages = [HumanMessage(content=init_prompt)]
    
    for turn in range(max_initial_turns):
        response = llm.invoke(messages)
        content = response.content
        messages.append(response)
        
        # Extract SQL
        sql_candidates = _extract_sql_candidates(content)
        if sql_candidates:
            initial_sql = sql_candidates[0]
            print(f"[OptV2] Initial Try SQL (turn {turn+1}):\n{initial_sql}\n")
            
            # Execute SQL
            result = run_sqlite_query(db_path, initial_sql)
            initial_result = str(result)
            initial_query_executed = True
            
            print(f"[OptV2] Initial Result: {truncate_for_json(initial_result)}\n")
            
            # If no error, proceed directly to final query generation
            if not (isinstance(result, str) and result.startswith("Error:")):
                break
            
            # Has error, continue to next turn (let LLM correct)
            if turn < max_initial_turns - 1:
                messages.append(HumanMessage(
                    content=f"Query failed with error: {result}\nPlease fix the query and try again."
                ))
        else:
            # No SQL generated, continue
            if turn < max_initial_turns - 1:
                messages.append(HumanMessage(
                    content="Please generate a SQL query to answer the question."
                ))
    
    # Phase 2: Select repair strategy based on CASE type
    if workflow_type == "CASE_2_NORMAL":
        # CASE_2 does not need repair, use initial query result directly
        return {
            "workflow_type": workflow_type,
            "final_answer": initial_result,
            "final_sql": initial_sql,
            "repair_sql": "",
            "final_query_result": initial_result,
            "has_recovery_json": False,
            "initial_query_sql": initial_sql,
            "initial_query_result": initial_result
        }
    
    # Phase 3: Repair database and generate final query
    # Select corresponding prompt
    prompt_map = {
        "CASE_1_MISSING_TABLE": CASE1_PROMPT,
        "CASE_3_CORRUPTED_DATA": CASE3_PROMPT,
        "CASE_4_MISSING_COLUMN": CASE4_PROMPT
    }
    prompt_template = prompt_map.get(workflow_type, CASE3_PROMPT)
    
    # Build repair phase prompt
    repair_prompt = f"""{prompt_template}

Initial Query Attempt:
- SQL: {initial_sql}
- Result: {initial_result}

Now repair the database using the recovery.json and generate the final query."""

    messages = [
        SystemMessage(content=repair_prompt),
        HumanMessage(content=f"""
        Context (Recovery JSON): {context_str}
        DB Schema: {schema}
        Question: {question}
        """)
    ]
    
    # Execute repair workflow (max 6 turns)
    current_messages = messages.copy()
    final_res = ""
    last_run_sql = ""
    last_query_result = ""
    repair_sql_log = []
    query_executed = False
    
    for turn in range(6):
        response = llm.invoke(current_messages)
        content = response.content
        current_messages.append(response)
        
        if "FINAL ANSWER:" in content:
            if not query_executed:
                current_messages.append(
                    HumanMessage(content="You must first provide and execute a query SQL (in ```sql``` block) before outputting FINAL ANSWER.")
                )
                continue
            final_res = content.split("FINAL ANSWER:")[1].strip()
            break
        
        sql_candidates = _extract_sql_candidates(content)
        if not sql_candidates:
            continue

        for sql in sql_candidates:
            for stmt in _split_sql_statements(sql):
                upper_stmt = stmt.strip().upper()
                is_query = upper_stmt.startswith("SELECT") or upper_stmt.startswith("WITH") or upper_stmt.startswith("PRAGMA")
                if is_query:
                    print(f"[OptV2] Executing Query SQL:\n{stmt}\n")
                    last_run_sql = stmt
                    result = run_sqlite_query(db_path, stmt)
                    last_query_result = str(result)
                    query_executed = True
                    current_messages.append(HumanMessage(content=f"Query Execution Result: {result}"))
                else:
                    print(f"[OptV2] Executing Repair SQL:\n{stmt}\n")
                    repair_sql_log.append(stmt)
                    result = run_sqlite_query(db_path, stmt)
                    current_messages.append(HumanMessage(content=f"Repair Execution Result: {result}"))
    
    return {
        "workflow_type": workflow_type,
        "final_answer": final_res,
        "final_sql": last_run_sql,
        "repair_sql": ";\n".join(repair_sql_log),
        "final_query_result": last_query_result,
        "has_recovery_json": has_recovery_json,
        "initial_query_sql": initial_sql,
        "initial_query_result": initial_result
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
    parser = argparse.ArgumentParser(description="Optimized Experiment V2: Worker handles initial query internally")
    parser.add_argument("--out-dir", type=str, default="", help="Output directory")
    parser.add_argument("--start", type=int, default=0, help="Start question index")
    parser.add_argument("--end", type=int, default=None, help="End question index (exclusive)")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of questions per batch")
    parser.add_argument("--batch-rest", type=int, default=5, help="Rest seconds between batches")
    parser.add_argument("--force", action="store_true", help="Overwrite processed questions")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary files")
    parser.add_argument("--max-initial-turns", type=int, default=1, help="Max initial query turns")
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
        out_dir = os.path.join("results", "exp_optimized_v2_test")
    _ensure_dir(out_dir)

    recovery_out_dir = os.path.join(out_dir, "recovery_json")
    _ensure_dir(recovery_out_dir)

    results_path = os.path.join(out_dir, "results.jsonl")
    meta_path = os.path.join(out_dir, "meta.json")

    # Save metadata
    meta = {
        "experiment": "exp_optimized_v2",
        "description": "Optimized V2: Worker performs initial query reasoning internally (max 1 turn)，reducing context passing",
        "model": "GLM-4.7",
        "dev_modified_json": DEV_MODIFIED_JSON,
        "start_index": start,
        "end_index": end,
        "total_questions": total_questions,
        "batch_size": args.batch_size,
        "batch_rest": args.batch_rest,
        "max_initial_turns": args.max_initial_turns,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "force": bool(args.force),
        "keep_temp": bool(args.keep_temp),
    }
    write_json_file(meta_path, meta)

    # Load processed questions
    done_qids = set()
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

    print(f"[Start] Optimized V2: Questions{start}-{end-1}，total{limit}")
    print(f"[Output] Output directory: {out_dir}")
    print(f"[Feature] Worker performs initial query reasoning internally (max{args.max_initial_turns}turns)")
    print(f"[Time] Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)

    slice_items = data[start:end]
    total = len(slice_items)
    batch_num = 0

    with open(results_path, "a", encoding="utf-8") as f_out:
        for idx, item in enumerate(slice_items, start=start):
            question_id = item.get("question_id", idx)
            
            if question_id in done_qids and not args.force:
                print(f"[Skip] Question {idx}: question_id={question_id} already processed")
                continue

            batch_num += 1
            t0 = time.time()
            
            print(f"\n[Process] Question {idx}: {question_id} - {item.get('question', '')[:60]}...")
            
            db_id = item.get("db_id", "")
            question = item.get("question", "")
            gold_sql = item.get("SQL", "")
            
            record = {
                "index": idx,
                "question_id": question_id,
                "db_id": db_id,
                "question": question,
                "gold_sql": gold_sql,
            }
            
            already_written = False
            
            try:
                # Set up sandbox environment
                sandbox_dir = os.path.join("temp_env", f"sandbox_{question_id}")
                _ensure_dir(sandbox_dir)
                
                db_path, recovery_json_path = setup_sandbox(question)

                # Copy recovery.json to results directory
                recovery_dst_path = os.path.join(recovery_out_dir, f"recovery_{question_id}.json")
                _copy_recovery_json(recovery_json_path, recovery_dst_path)
                record["recovery_json_saved_to"] = recovery_dst_path

                # 1. Simplified Router - only determine CASE type
                workflow_type = router_node(question, db_path, recovery_json_path)
                record["workflow_type"] = workflow_type
                print(f"[OptV2 Router] Diagnosis: {workflow_type}")
                
                # 2. Optimized Worker - performs initial query reasoning internally
                workflow_result = optimized_worker(
                    question=question,
                    db_path=db_path,
                    recovery_json_path=recovery_json_path,
                    workflow_type=workflow_type,
                    max_initial_turns=args.max_initial_turns
                )
                
                record.update({
                    "initial_query_sql": workflow_result.get("initial_query_sql", ""),
                    "initial_query_result": truncate_for_json(workflow_result.get("initial_query_result", "")),
                    "final_answer": workflow_result.get("final_answer", ""),
                    "final_sql": workflow_result.get("final_sql", ""),
                    "final_query_result": truncate_for_json(workflow_result.get("final_query_result", "")),
                    "repair_sql": workflow_result.get("repair_sql", ""),
                    "has_recovery_json": workflow_result.get("has_recovery_json", False),
                    "status": "completed",
                })

                # Evaluation logic (same as original experiment)
                # Evaluation 1: Repair success rate
                original_gold_result = None
                if gold_sql:
                    try:
                        original_db_path = os.path.join(DEV_DATABASES_DIR, db_id, f"{db_id}.sqlite")
                        if os.path.exists(original_db_path):
                            original_gold_result = run_sqlite_query(original_db_path, gold_sql)
                    except Exception as e:
                        print(f"[Warning] Cannot get original gold result: {e}")
                
                if original_gold_result is not None and gold_sql:
                    record["eval_gold_result"] = truncate_for_json(original_gold_result)
                    
                    if isinstance(original_gold_result, str) and original_gold_result.startswith("Error:"):
                        record["eval_is_match"] = False
                        record["eval_diff_summary"] = "INVALID_GOLD_SQL"
                    else:
                        if workflow_type == "CASE_2_NORMAL":
                            record["eval_is_match"] = True
                            record["eval_diff_summary"] = "NO_REPAIR_NEEDED"
                        else:
                            repaired_gold_result = run_sqlite_query(db_path, gold_sql)
                            record["eval_sandbox_gold_result"] = truncate_for_json(repaired_gold_result)
                            
                            is_match, diff_summary = compare_query_results(gold_sql, original_gold_result, repaired_gold_result)
                            record["eval_is_match"] = bool(is_match)
                            record["eval_diff_summary"] = diff_summary

                # Evaluation 2: Text-to-SQL accuracy
                final_sql = record.get("final_sql", "")
                if final_sql:
                    try:
                        final_res_raw = run_sqlite_query(db_path, final_sql)
                        record["eval_final_result"] = truncate_for_json(final_res_raw)
                        
                        if gold_sql and original_gold_result is not None:
                            if isinstance(original_gold_result, str) and original_gold_result.startswith("Error:"):
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

                t1 = time.time()
                record["time_sec"] = t1 - t0
                
                repair_status = "[Success]Repair" if record.get("has_recovery_json") else "[Fail]Repair"
                query_status = "[Success]Query" if record.get("eval_final_is_match") else "[Fail]Query"
                print(f"      {repair_status} {query_status} | Time: {t1-t0:.1f}s | {workflow_type}")
                
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()
                already_written = True

                # Batch rest
                if batch_num % args.batch_size == 0:
                    print(f"\n[Batch] Batch completed: {batch_num}/{total} ({100.0*batch_num/total:.1f}%)")
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
    print(f"[Done] Optimized V2 completed! Results saved in: {out_dir}")
    print(f"[Time] End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
