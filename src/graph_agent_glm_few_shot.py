import json
import sqlite3
import os
import re
from typing import TypedDict, Literal, Optional, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import get_llm
from src.tools import run_sqlite_query, read_json_file, get_db_schema
from prompt_engineering.prompts_templates import get_prompt_template

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
    llm = get_llm(_os.getenv("LLM_PROVIDER", "chatanywhere"))
    question = state['question']
    db_path = state['db_path']
    recovery_path = state['recovery_json_path']

    schema = get_db_schema(db_path)

    context = {}
    context_str = "{}"
    if state.get('has_recovery_json', False) and os.path.exists(recovery_path):
        try:
            context = read_json_file(recovery_path)
            context_str = json.dumps(context, indent=2)
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

    messages = [
        SystemMessage(content=prompt_template),
        HumanMessage(content=f"""
        Context (Recovery JSON): {context_str}
        DB Schema: {schema}
        Question: {question}
        """)
    ]

    current_messages = messages.copy()
    final_res = ""
    last_run_sql = ""
    last_query_result = ""
    repair_sql_log = []
    query_executed = False

    for _ in range(6):
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
                    print(f"[Worker] Executing Query SQL:\n{stmt}\n")
                    last_run_sql = stmt
                    result = run_sqlite_query(db_path, stmt)
                    last_query_result = str(result)
                    query_executed = True
                    current_messages.append(HumanMessage(content=f"Query Execution Result: {result}"))
                else:
                    print(f"[Worker] Executing Repair SQL:\n{stmt}\n")
                    repair_sql_log.append(stmt)
                    result = run_sqlite_query(db_path, stmt)
                    current_messages.append(HumanMessage(content=f"Repair Execution Result: {result}"))

    return {
        "final_answer": final_res,
        "final_sql": last_run_sql,
        "repair_sql": ";\n".join(repair_sql_log),
        "final_query_result": last_query_result,
    }


def router_node(state: AgentState):
    import os as _os
    llm = get_llm(_os.getenv("LLM_PROVIDER", "chatanywhere"))
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

    router_prompt = get_prompt_template("few_shot", "router")
    diagnosis_msg = router_prompt.format(
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


def case1_worker(state: AgentState):
    print("[Worker] Handling Case 1: Missing Table...")
    prompt_template = get_prompt_template("few_shot", "case1")
    return _worker_logic(state, prompt_template)


def case3_worker(state: AgentState):
    print("[Worker] Handling Case 3: Corrupted Data...")
    prompt_template = get_prompt_template("few_shot", "case3")
    return _worker_logic(state, prompt_template)


def case4_worker(state: AgentState):
    print("[Worker] Handling Case 4: Missing Column...")
    prompt_template = get_prompt_template("few_shot", "case4")
    return _worker_logic(state, prompt_template)


def case2_worker(state: AgentState):
    print("[Worker] Handling Case 2: Normal Database...")
    prompt_template = get_prompt_template("few_shot", "case2")
    return _worker_logic(state, prompt_template)


def route_logic(state: AgentState):
    return state['workflow_type']


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