#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline A: Direct Query on Perturbed DB with Clues (No repair, query directly with clues)

Design points:
1. Router node unchanged, still determines CASE type
2. Worker node modified: prohibit generating/executing any DDL/DML, only SELECT allowed
3. Execute generated SELECT query directly on perturbed DB
4. Provide JSON clues, but prohibit using them for repair
5. Each CASE uses dedicated prompts
6. Support 6 turns of reasoning and FINAL_ANSWER detection

This tests whether the clues alone reveal enough information.
"""

import json
import os
import re
import sqlite3
from typing import TypedDict, Literal, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import get_llm
from src.tools import run_sqlite_query, read_json_file, get_db_schema
from src.prompts_baseline_a import (
    BASELINE_A_ROUTER_PROMPT,
    BASELINE_A_CASE1_PROMPT,
    BASELINE_A_CASE2_PROMPT,
    BASELINE_A_CASE3_PROMPT,
    BASELINE_A_CASE4_PROMPT,
)


class AgentState(TypedDict):
    question: str
    db_path: str
    recovery_json_path: str

    initial_query_result: str
    initial_query_sql: str
    has_recovery_json: bool

    workflow_type: Literal["CASE_1_MISSING_TABLE", "CASE_2_NORMAL", "CASE_3_CORRUPTED_DATA", "CASE_4_MISSING_COLUMN"]

    final_answer: str
    final_sql: str
    repair_sql: str
    final_query_result: str


def _extract_sql(content: str) -> str:
    """Helper to extract SQL from LLM response"""
    match = re.search(r"```sql\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return content.replace("```sql", "").replace("```", "").strip()


def _extract_select_sql(content: str) -> Optional[str]:
    """
    Extract SELECT SQL statement from LLM response
    Strictly limit to SELECT statements only, filter out all DDL/DML
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


def _baseline_a_worker_logic(state: AgentState, prompt_template: str, max_turns: int = 6):
    """
    Baseline A Worker logic:
    - Provide JSON clues
    - Prohibit generating/executing any DDL/DML
    - Only SELECT allowed
    - Execute directly on perturbed DB
    - Support multi-turn reasoning (default 6 turns)
    - Support FINAL_ANSWER detection
    """
    import os as _os
    llm = get_llm(_os.getenv("LLM_PROVIDER", "glm4.7"))
    question = state['question']
    db_path = state['db_path']
    recovery_path = state['recovery_json_path']

    schema = get_db_schema(db_path)

    # Prepare recovery.json context (provide clues, but not for repair)
    context = {}
    context_str = "{}"
    has_recovery = False

    if state.get('has_recovery_json', False) and os.path.exists(recovery_path):
        try:
            context = read_json_file(recovery_path)
            context_str = json.dumps(context, indent=2)
            has_recovery = True

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

    # Build Prompt - directly concatenate information
    system_prompt = f"""{prompt_template}

User Question: {question}

Current Database Schema:
{schema}
"""

    # Multi-turn reasoning (max 6 turns)
    messages = [SystemMessage(content=system_prompt)]
    if has_recovery and context_str != "{}":
        messages.append(HumanMessage(content=f"Recovery JSON Clues (information about what's missing/corrupted):\n{context_str}"))

    generated_sql = None
    final_query_result = None
    final_answer = None
    last_error = None
    query_executed = False

    for turn in range(max_turns):
        print(f"[Baseline A Worker] Turn {turn + 1}/{max_turns}")

        try:
            response = llm.invoke(messages)
            content = response.content
            messages.append(response)

            # Check for FINAL ANSWER
            if "FINAL ANSWER:" in content:
                if not query_executed:
                    # Query not yet executed, request to execute query first
                    messages.append(HumanMessage(
                        content="You must execute a SELECT query first (in ```sql``` block) before outputting FINAL ANSWER."
                    ))
                    continue
                # Extract FINAL ANSWER
                final_answer = content.split("FINAL ANSWER:")[1].strip().split('\n')[0]
                print(f"[Baseline A Worker] Final Answer: {final_answer[:100]}...")
                break

            # Extract SELECT SQL (strict filtering, only SELECT allowed)
            generated_sql = _extract_select_sql(content)

            if not generated_sql:
                print(f"[Baseline A Worker] No valid SELECT SQL found in turn {turn + 1}")
                if turn < max_turns - 1:
                    messages.append(HumanMessage(
                        content="Your response did not contain a valid SELECT query. Please output ONLY a SELECT query inside ```sql``` block."
                    ))
                continue

            print(f"[Baseline A Worker] Generated SELECT SQL (turn {turn + 1}):\n{generated_sql}\n")

            # Execute generated SELECT directly on perturbed DB (no repair)
            try:
                result = run_sqlite_query(db_path, generated_sql)
                final_query_result = str(result)
                query_executed = True
                print(f"[Baseline A Worker] Execution Result: {final_query_result[:200]}...")

                # Add execution result to conversation
                messages.append(HumanMessage(content=f"Query Execution Result: {final_query_result}"))

                # If this is the last turn, or the result looks reasonable, return
                if turn == max_turns - 1:
                    final_answer = final_query_result
                    break

                # Otherwise continue, let the model evaluate the result and decide if improvement is needed
                messages.append(HumanMessage(
                    content="Review the result. If it answers the question correctly, output FINAL ANSWER. Otherwise, output a revised SELECT query."
                ))

            except Exception as e:
                last_error = str(e)
                print(f"[Baseline A Worker] Execution error in turn {turn + 1}: {last_error}")

                # Execution failed, add error message and continue to next turn
                if turn < max_turns - 1:
                    messages.append(HumanMessage(
                        content=f"The previous query failed with error: {last_error}\nPlease fix the query and try again. Output ONLY a SELECT query."
                    ))
                continue

        except Exception as e:
            print(f"[Baseline A Worker] LLM invocation error in turn {turn + 1}: {e}")
            last_error = str(e)
            if turn < max_turns - 1:
                continue
            else:
                return {
                    "final_answer": f"LLM Error after {max_turns} turns: {last_error}",
                    "final_sql": generated_sql if generated_sql else "",
                    "repair_sql": "",
                    "final_query_result": "",
                    "turns_used": turn + 1
                }

    # If no final_answer obtained, use the last query result
    # Modification: ensure a SQL must be output at the end, even if possibly incorrect
    if final_answer is None:
        if final_query_result:
            final_answer = final_query_result
        elif generated_sql:
            # Has SQL but no execution result, try to execute
            try:
                result = run_sqlite_query(db_path, generated_sql)
                final_answer = str(result)
                final_query_result = str(result)
            except Exception as e:
                # Execution failed, but still return SQL
                final_answer = f"Query execution failed: {str(e)}"
                final_query_result = ""
        else:
            # No SQL generated, this should not happen
            # Force the model to generate a basic query
            final_answer = "No query generated"
            final_query_result = ""
    
    # Ensure final_sql is not empty
    if not generated_sql:
        # If no SQL generated, extract the first table name from schema
        try:
            schema = get_db_schema(db_path)
            # Parse schema to get table names
            import re
            tables = re.findall(r'CREATE TABLE (\w+)', schema)
            if tables:
                table_name = tables[0]
                generated_sql = f"SELECT * FROM {table_name} LIMIT 1"
            else:
                generated_sql = "SELECT 1"
        except:
            generated_sql = "SELECT 1"
    
    # Ensure final_answer is not empty
    if final_answer is None or final_answer == "No query generated":
        # Try to execute generated SQL
        try:
            result = run_sqlite_query(db_path, generated_sql)
            final_answer = str(result)
            final_query_result = str(result)
        except Exception as e:
            final_answer = f"Query: {generated_sql}"
            final_query_result = ""

    return {
        "final_answer": final_answer,
        "final_sql": generated_sql,
        "repair_sql": "",  # Baseline A does not execute any repair
        "final_query_result": final_query_result if final_query_result else "",
        "turns_used": turn + 1
    }


# ==================== Router node (unchanged)====================

def router_node(state: AgentState):
    """
    Router node - consistent with original implementation
    Determine CASE type, but do not change database state
    """
    import os as _os
    llm = get_llm(_os.getenv("LLM_PROVIDER", "glm4.7"))
    question = state['question']
    db_path = state['db_path']

    gen_sql_prompt = f"Generate a SQLite query for the following question on schema {get_db_schema(db_path)}:\nQuestion: {question}\nOutput ONLY the SQL."
    gen_sql_response = llm.invoke(gen_sql_prompt)
    sql = _extract_sql(gen_sql_response.content)

    print(f"[Router] Initial Try SQL:\n{sql}\n")
    result = run_sqlite_query(db_path, sql)

    has_json = False
    if state.get('recovery_json_path') and os.path.exists(state['recovery_json_path']):
        try:
            meta = read_json_file(state['recovery_json_path'])
            has_json = bool(meta.get("case_type"))
        except Exception:
            has_json = False

    recovery_metadata = "Unavailable"
    if has_json:
        try:
            with open(state['recovery_json_path'], 'r') as f:
                meta = json.load(f)
                recovery_metadata = json.dumps({k: v for k, v in meta.items() if k != 'data_payload'})
        except:
            pass

    result_str = str(result)
    if len(result_str) > 2000:
        result_str = result_str[:2000] + "... [TRUNCATED]"

    diagnosis_msg = BASELINE_A_ROUTER_PROMPT.format(
        question=question,
        result=result_str,
        has_recovery_json=has_json,
        recovery_metadata=recovery_metadata
    )

    decision = llm.invoke([HumanMessage(content=diagnosis_msg)]).content.strip()

    workflow_type = "CASE_2_NORMAL"
    if "CASE_1" in decision: workflow_type = "CASE_1_MISSING_TABLE"
    elif "CASE_3" in decision: workflow_type = "CASE_3_CORRUPTED_DATA"
    elif "CASE_4" in decision: workflow_type = "CASE_4_MISSING_COLUMN"

    if has_json and workflow_type == "CASE_2_NORMAL":
        print("[Router] Warning: LLM predicted Normal but Recovery Context exists. Forcing CASE_3_CORRUPTED_DATA.")
        workflow_type = "CASE_3_CORRUPTED_DATA"

    print(f"[Router] Diagnosis: {workflow_type}")

    return {
        "workflow_type": workflow_type,
        "initial_query_result": str(result),
        "initial_query_sql": sql,
        "has_recovery_json": has_json,
        "final_answer": "",
        "final_sql": ""
    }


# ==================== Worker node (each CASE uses dedicated prompts)====================

def case1_worker(state: AgentState):
    """Case 1: Missing Table - Baseline A handling"""
    print("[Worker] Handling Case 1: Missing Table (Baseline A - No Repair)...")
    return _baseline_a_worker_logic(state, BASELINE_A_CASE1_PROMPT)


def case2_worker(state: AgentState):
    """Case 2: Normal Database - Baseline A handling (using dedicated optimized prompts)"""
    print("[Worker] Handling Case 2: Normal Database (Baseline A)...")
    return _baseline_a_worker_logic(state, BASELINE_A_CASE2_PROMPT)


def case3_worker(state: AgentState):
    """Case 3: Corrupted Data - Baseline A handling"""
    print("[Worker] Handling Case 3: Corrupted Data (Baseline A - No Repair)...")
    return _baseline_a_worker_logic(state, BASELINE_A_CASE3_PROMPT)


def case4_worker(state: AgentState):
    """Case 4: Missing Column - Baseline A handling"""
    print("[Worker] Handling Case 4: Missing Column (Baseline A - No Repair)...")
    return _baseline_a_worker_logic(state, BASELINE_A_CASE4_PROMPT)


def route_logic(state: AgentState):
    return state['workflow_type']


# ==================== Build workflow ====================

workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("worker_case1", case1_worker)
workflow.add_node("worker_case2", case2_worker)
workflow.add_node("worker_case3", case3_worker)
workflow.add_node("worker_case4", case4_worker)

workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    route_logic,
    {
        "CASE_1_MISSING_TABLE": "worker_case1",
        "CASE_2_NORMAL": "worker_case2",
        "CASE_3_CORRUPTED_DATA": "worker_case3",
        "CASE_4_MISSING_COLUMN": "worker_case4"
    }
)

workflow.add_edge("worker_case1", END)
workflow.add_edge("worker_case2", END)
workflow.add_edge("worker_case3", END)
workflow.add_edge("worker_case4", END)

app = workflow.compile()
