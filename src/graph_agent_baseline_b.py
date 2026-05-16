#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline B: Monolithic Text-to-DDL/DML (Complete repair+query in one pass)

Design points:
1. No explicit routing - no router node
2. No skill decomposition - single large prompt
3. No separate staged execution and reflection - complete anomaly detection, DDL/DML generation, final SQL generation in one pass
4. Same backbone, same token budget, same self-correction limit (6 turns)
5. Single-node workflow, directly call LLM to generate complete solution
6. Force SQL output on last turn to avoid empty SQL issue
7. Single-turn LLM call timeout protection to prevent hanging
8. If no SQL after 6 turns, append one forced SELECT generation turn (no timeout)

Key differences from Full Workflow:
- Full Workflow: router -> case-specific worker -> evaluator (modular)
- Baseline B: monolithic_worker (integrated, no routing, no skill decomposition)
"""

import json
import os
import re
import sqlite3
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import get_llm
from src.tools import run_sqlite_query, read_json_file, get_db_schema
from src.prompts_baseline_b import BASELINE_B_MONOLITHIC_PROMPT

MAX_TURNS = 6
SINGLE_TURN_TIMEOUT = 3600


class AgentState(TypedDict):
    question: str
    db_path: str
    recovery_json_path: str

    has_recovery_json: bool
    case_type: str

    final_answer: str
    final_sql: str
    repair_sql: str
    final_query_result: str


def _extract_sql(content: str) -> str:
    match = re.search(r"```sql\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return content.replace("```sql", "").replace("```", "").strip()


def _extract_sql_candidates(content: str) -> list[str]:
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


def _split_sql_statements(sql: str) -> list[str]:
    statements: list[str] = []
    buf: list[str] = []
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


def _invoke_llm_with_timeout(llm, messages, timeout=SINGLE_TURN_TIMEOUT):
    import threading
    result_container = []
    error_container = []

    def _call():
        try:
            result_container.append(llm.invoke(messages))
        except Exception as e:
            error_container.append(e)

    thread = threading.Thread(target=_call)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        return None, TimeoutError(f"LLM invocation timeout ({timeout}s)")

    if error_container:
        return None, error_container[0]

    if result_container:
        return result_container[0], None

    return None, RuntimeError("Unknown LLM error")


def _force_generate_select(llm, question: str, db_path: str, schema: str) -> str:
    """
    Last resort: force LLM to generate a SELECT query.
    No timeout, use short prompt focused on query generation.
    """
    force_prompt = f"""Given this database schema:

{schema}

Generate a single SELECT SQL query to answer this question:
{question}

Output ONLY the SQL query inside ```sql``` block. No explanation needed."""

    messages = [SystemMessage(content=force_prompt), HumanMessage(content="Generate the SQL query now.")]

    try:
        response = llm.invoke(messages)
        content = response.content
        sql_candidates = _extract_sql_candidates(content)
        for sql in sql_candidates:
            for stmt in _split_sql_statements(sql):
                upper_stmt = stmt.strip().upper()
                if upper_stmt.startswith("SELECT") or upper_stmt.startswith("WITH"):
                    return stmt
    except Exception:
        pass

    tables = re.findall(r'CREATE TABLE (\w+)', schema)
    if tables:
        return f"SELECT * FROM {tables[0]} LIMIT 1"
    return "SELECT 1"


def monolithic_worker(state: AgentState):
    """
    Baseline B core node: Monolithic Worker
    """
    import os as _os
    llm = get_llm(_os.getenv("LLM_PROVIDER", "chatanywhere"))
    question = state['question']
    db_path = state['db_path']
    recovery_path = state['recovery_json_path']

    schema = get_db_schema(db_path)

    context = {}
    context_str = "{}"
    has_recovery = False
    detected_case_type = "normal"

    if state.get('has_recovery_json', False) and os.path.exists(recovery_path):
        try:
            context = read_json_file(recovery_path)
            context_str = json.dumps(context, indent=2)
            has_recovery = True

            detected_case_type = context.get('case_type', 'normal')

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

    system_prompt = f"""{BASELINE_B_MONOLITHIC_PROMPT}

User Question: {question}

Current Database Schema:
{schema}
"""

    messages = [SystemMessage(content=system_prompt)]
    if has_recovery and context_str != "{}":
        messages.append(HumanMessage(
            content=f"Recovery JSON (contains information about database anomalies and how to fix them):\n{context_str}"
        ))
    else:
        messages.append(HumanMessage(
            content="No recovery JSON provided. The database appears to be normal. Please generate a SELECT query to answer the question."
        ))

    last_run_sql = ""
    last_query_result = ""
    repair_sql_log = []
    query_executed = False
    final_res = ""
    turns_used = 0

    for turn in range(MAX_TURNS):
        turns_used = turn + 1
        is_last_turn = (turn == MAX_TURNS - 1)
        print(f"[Baseline B Monolithic Worker] Turn {turns_used}/{MAX_TURNS}")

        if is_last_turn and not query_executed:
            messages.append(HumanMessage(
                content="THIS IS YOUR FINAL TURN. You MUST output a SELECT query inside ```sql``` block NOW. "
                        "Even if you are unsure, output your best attempt at a SELECT query. "
                        "Do NOT output any more repair SQL. Output ONLY a SELECT query."
            ))

        try:
            response, llm_error = _invoke_llm_with_timeout(llm, messages)

            if llm_error:
                error_str = str(llm_error)
                is_rate_limit = "429" in error_str or "rate" in error_str.lower() or "rate limit" in error_str or "rate limit" in error_str

                if is_rate_limit:
                    import time as _time
                    backoff = 30
                    print(f"[Baseline B] Rate limited (429) in turn {turns_used}, waiting {backoff}s...")
                    _time.sleep(backoff)
                    response, llm_error = _invoke_llm_with_timeout(llm, messages)
                    if llm_error:
                        print(f"[Baseline B] Still error after retry: {llm_error}")
                        if is_last_turn:
                            break
                        continue
                else:
                    print(f"[Baseline B] LLM error in turn {turns_used}: {llm_error}")
                    if is_last_turn:
                        break
                    continue

            content = response.content
            messages.append(response)

            if "FINAL ANSWER:" in content:
                if not query_executed:
                    if is_last_turn:
                        sql_candidates = _extract_sql_candidates(content)
                        for sql in sql_candidates:
                            for stmt in _split_sql_statements(sql):
                                upper_stmt = stmt.strip().upper()
                                if upper_stmt.startswith("SELECT") or upper_stmt.startswith("WITH"):
                                    try:
                                        result = run_sqlite_query(db_path, stmt)
                                        last_run_sql = stmt
                                        last_query_result = str(result)
                                        query_executed = True
                                    except Exception:
                                        pass
                                    break
                        if not query_executed and last_run_sql:
                            try:
                                result = run_sqlite_query(db_path, last_run_sql)
                                last_query_result = str(result)
                                query_executed = True
                            except Exception:
                                pass
                        final_res = content.split("FINAL ANSWER:")[1].strip().split('\n')[0]
                        break
                    else:
                        messages.append(HumanMessage(
                            content="You must execute a SELECT query first (in ```sql``` block) before outputting FINAL ANSWER."
                        ))
                        continue
                final_res = content.split("FINAL ANSWER:")[1].strip().split('\n')[0]
                print(f"[Baseline B] Final Answer: {final_res[:100]}...")
                break

            sql_candidates = _extract_sql_candidates(content)
            if not sql_candidates:
                print(f"[Baseline B] No SQL found in turn {turns_used}")
                if is_last_turn:
                    break
                messages.append(HumanMessage(
                    content="Your response did not contain SQL. Please output SQL inside ```sql``` blocks. First repair SQL (if needed), then a SELECT query."
                ))
                continue

            for sql in sql_candidates:
                for stmt in _split_sql_statements(sql):
                    upper_stmt = stmt.strip().upper()
                    is_query = upper_stmt.startswith("SELECT") or upper_stmt.startswith("WITH") or upper_stmt.startswith("PRAGMA")

                    if is_query:
                        print(f"[Baseline B] Executing Query SQL (turn {turns_used}):\n{stmt}\n")
                        last_run_sql = stmt
                        try:
                            result = run_sqlite_query(db_path, stmt)
                            last_query_result = str(result)
                            query_executed = True
                            print(f"[Baseline B] Query Result: {last_query_result[:200]}...")
                            messages.append(HumanMessage(content=f"Query Execution Result: {result}"))
                        except Exception as e:
                            print(f"[Baseline B] Query error (turn {turns_used}): {e}")
                            messages.append(HumanMessage(
                                content=f"The previous query failed with error: {e}\nPlease fix the query and try again."
                            ))
                    else:
                        print(f"[Baseline B] Executing Repair SQL (turn {turns_used}):\n{stmt}\n")
                        repair_sql_log.append(stmt)
                        try:
                            result = run_sqlite_query(db_path, stmt)
                            print(f"[Baseline B] Repair Result: {str(result)[:200]}...")
                            messages.append(HumanMessage(content=f"Repair Execution Result: {result}"))
                        except Exception as e:
                            print(f"[Baseline B] Repair error (turn {turns_used}): {e}")
                            messages.append(HumanMessage(
                                content=f"The repair SQL failed with error: {e}\nPlease fix and try again."
                            ))

            if not is_last_turn and query_executed:
                messages.append(HumanMessage(
                    content="Review the result. If it correctly answers the question, output FINAL ANSWER. Otherwise, output a revised query."
                ))

        except Exception as e:
            print(f"[Baseline B] Unexpected error in turn {turns_used}: {e}")
            if is_last_turn:
                break
            continue

    # === Fallback: if still no SQL after 6 turns, force an additional turn to generate SELECT ===
    if not last_run_sql:
        print(f"[Baseline B] No SQL after {turns_used} turns, forcing SELECT generation...")
        forced_sql = _force_generate_select(llm, question, db_path, schema)
        last_run_sql = forced_sql
        print(f"[Baseline B] Forced SQL: {forced_sql[:100]}...")
        try:
            result = run_sqlite_query(db_path, forced_sql)
            last_query_result = str(result)
            query_executed = True
        except Exception as e:
            print(f"[Baseline B] Forced SQL execution error: {e}")
            last_query_result = ""

    if not final_res:
        if last_query_result:
            final_res = last_query_result
        elif last_run_sql:
            try:
                result = run_sqlite_query(db_path, last_run_sql)
                final_res = str(result)
                last_query_result = str(result)
            except Exception as e:
                final_res = f"Query execution failed: {str(e)}"
                last_query_result = ""
        else:
            final_res = "No query generated"
            last_query_result = ""

    return {
        "final_answer": final_res,
        "final_sql": last_run_sql,
        "repair_sql": ";\n".join(repair_sql_log),
        "final_query_result": last_query_result if last_query_result else "",
        "case_type": detected_case_type,
        "turns_used": turns_used
    }


workflow = StateGraph(AgentState)

workflow.add_node("monolithic_worker", monolithic_worker)

workflow.set_entry_point("monolithic_worker")

workflow.add_edge("monolithic_worker", END)

app = workflow.compile()
