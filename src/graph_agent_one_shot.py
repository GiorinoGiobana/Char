import json
import sqlite3
import os
import re
from typing import TypedDict, Literal, Optional, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import get_llm
from src.tools import run_sqlite_query, read_json_file, get_db_schema
from prompt_engineering.prompts_templates import ONE_SHOT_ROUTER_PROMPT, ONE_SHOT_CASE1_PROMPT, ONE_SHOT_CASE2_PROMPT, ONE_SHOT_CASE3_PROMPT, ONE_SHOT_CASE4_PROMPT

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

def _worker_logic(state: AgentState, prompt_template: str):
    import os as _os
    llm = get_llm(_os.getenv("LLM_PROVIDER", "chatanywhere2"))
    question = state['question']
    db_path = state['db_path']
    recovery_path = state['recovery_json_path']

    schema = get_db_schema(db_path)
    has_recovery = state.get('has_recovery_json', bool(recovery_path and os.path.exists(recovery_path)))
    recovery_metadata = "{}"

    if has_recovery and recovery_path and os.path.exists(recovery_path):
        recovery_data = read_json_file(recovery_path)
        if recovery_data:
            case_type = recovery_data.get("case_type", "unknown")
            recovery_metadata = json.dumps({"case_type": case_type}, ensure_ascii=False)

    if '{schema}' in prompt_template:
        system_prompt = prompt_template.format(
            question=question,
            schema=schema,
            has_recovery_json=str(has_recovery),
            recovery_metadata=recovery_metadata
        )
    else:
        system_prompt = prompt_template.format(
            question=question,
            has_recovery_json=str(has_recovery),
            recovery_metadata=recovery_metadata
        )

    messages = [SystemMessage(content=system_prompt)]

    if has_recovery and recovery_path and os.path.exists(recovery_path):
        recovery_data = read_json_file(recovery_path)
        if recovery_data:
            context_json = json.dumps(recovery_data, ensure_ascii=False)
            messages.append(HumanMessage(content=f"Recovery Context JSON:\n{context_json}"))

    try:
        response = llm.invoke(messages)
        content = response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        return {
            "final_answer": f"LLM Error: {str(e)}",
            "final_sql": "",
            "repair_sql": ""
        }

    sql_candidates = _extract_sql_candidates(content)
    if not sql_candidates:
        return {
            "final_answer": f"No SQL found in response: {content[:500]}",
            "final_sql": "",
            "repair_sql": ""
        }

    db_dir = os.path.dirname(db_path)
    temp_db_path = os.path.join(db_dir, f"temp_{os.path.basename(db_path)}")

    try:
        import shutil
        shutil.copy2(db_path, temp_db_path)

        repair_statements = []
        select_statements = []

        for i, sql in enumerate(sql_candidates):
            sql_clean = sql.strip()
            if not sql_clean:
                continue

            sql_upper = sql_clean.upper().strip()
            if sql_upper.startswith("SELECT") or sql_upper.startswith("WITH"):
                select_statements.append(sql_clean)
            else:
                repair_statements.append(sql_clean)

        for stmt in repair_statements:
            try:
                run_sqlite_query(temp_db_path, stmt)
            except Exception as e:
                return {
                    "final_answer": f"SQL Execution Error (Repair): {str(e)}\nSQL: {stmt}",
                    "final_sql": "",
                    "repair_sql": "\n".join(repair_statements)
                }

        final_answer = ""
        final_sql = ""

        if select_statements:
            for sel_sql in select_statements:
                try:
                    result = run_sqlite_query(temp_db_path, sel_sql)
                    final_sql = sel_sql

                    if "FINAL ANSWER:" in content:
                        answer_match = re.search(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", content, re.DOTALL)
                        if answer_match:
                            final_answer = answer_match.group(1).strip()
                        else:
                            final_answer = str(result)
                    else:
                        final_answer = str(result)

                    break
                except Exception as e:
                    continue

        if not final_answer:
            final_answer = "No valid result obtained"

        return {
            "final_answer": final_answer,
            "final_sql": final_sql,
            "repair_sql": "\n".join(repair_statements)
        }

    finally:
        if os.path.exists(temp_db_path):
            try:
                os.remove(temp_db_path)
            except:
                pass

def router_node(state: AgentState):
    import os as _os
    llm = get_llm(_os.getenv("LLM_PROVIDER", "chatanywhere2"))
    question = state['question']
    db_path = state['db_path']
    recovery_path = state['recovery_json_path']

    has_recovery = bool(recovery_path and os.path.exists(recovery_path))
    recovery_metadata = "{}"

    if has_recovery and recovery_path:
        recovery_data = read_json_file(recovery_path)
        if recovery_data:
            case_type = recovery_data.get("case_type", "unknown")
            recovery_metadata = json.dumps({"case_type": case_type}, ensure_ascii=False)

    prompt = ONE_SHOT_ROUTER_PROMPT.format(
        question=question,
        result=state.get('initial_query_result', ''),
        has_recovery_json=str(has_recovery),
        recovery_metadata=recovery_metadata
    )

    try:
        response = llm.invoke([SystemMessage(content=prompt)])
        content = response.content if hasattr(response, 'content') else str(response)
        content = content.strip()
    except Exception as e:
        content = "CASE_2_NORMAL"

    workflow_type = content if content.startswith("CASE_") else "CASE_2_NORMAL"

    return {
        "workflow_type": workflow_type,
        "has_recovery_json": has_recovery,
        "initial_query_result": state.get('initial_query_result', ''),
        "initial_query_sql": state.get('initial_query_sql', ''),
    }

def case1_node(state: AgentState):
    return _worker_logic(state, ONE_SHOT_CASE1_PROMPT)

def case3_node(state: AgentState):
    return _worker_logic(state, ONE_SHOT_CASE3_PROMPT)

def case4_node(state: AgentState):
    return _worker_logic(state, ONE_SHOT_CASE4_PROMPT)

def case2_node(state: AgentState):
    return _worker_logic(state, ONE_SHOT_CASE2_PROMPT)

def build_workflow():
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("case1", case1_node)
    workflow.add_node("case3", case3_node)
    workflow.add_node("case4", case4_node)
    workflow.add_node("case2", case2_node)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        lambda state: state["workflow_type"],
        {
            "CASE_1_MISSING_TABLE": "case1",
            "CASE_3_CORRUPTED_DATA": "case3",
            "CASE_4_MISSING_COLUMN": "case4",
            "CASE_2_NORMAL": "case2"
        }
    )

    workflow.add_edge("case1", END)
    workflow.add_edge("case3", END)
    workflow.add_edge("case4", END)
    workflow.add_edge("case2", END)

    return workflow.compile()
