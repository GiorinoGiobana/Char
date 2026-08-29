import json
import os
import re
import sqlite3
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from src.config import get_llm
from src.tools import run_sqlite_query, read_json_file, get_db_schema
from .prompts import (
    STATE_INSPECTION_PROMPT,
    DQL_PROBE_PROMPT,
    ADJUSTMENT_PLANNING_PROMPT,
    POST_DQL_EXECUTION_PROMPT,
)
from .state import (
    StateSculptorState,
    StateCondition,
    ParsedEvidence,
    DiagnosticSummary,
    ProbeResult,
    AdjustmentPlan,
    VerificationResult,
    DQLExecutionResult,
)


def _extract_sql(content: str) -> str:
    match = re.search(r"```sql\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return content.replace("```sql", "").replace("```", "").strip()


def _extract_json(content: str) -> dict:
    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return {"raw_response": content}


def _call_llm_with_prompt(system_prompt: str, user_content: str, llm=None) -> str:
    if llm is None:
        llm = get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content)
    ]
    response = llm.invoke(messages)
    return response.content


class SandboxMaterializer:
    """Materialize an isolated writable copy while preserving the source database."""

    def __init__(self, source_db_path: str, temp_root: Optional[str] = None):
        self.source_db_path = source_db_path
        self.temp_root = temp_root
        self.sandbox_dir: Optional[str] = None
        self.sandbox_db_path: Optional[str] = None

    def materialize(self) -> str:
        if not os.path.isfile(self.source_db_path):
            raise FileNotFoundError(f"Database file not found: {self.source_db_path}")

        self.sandbox_dir = tempfile.mkdtemp(
            prefix="statesculptor_",
            dir=self.temp_root,
        )
        self.sandbox_db_path = os.path.join(self.sandbox_dir, "sandbox.sqlite")
        source_uri = Path(self.source_db_path).resolve().as_uri() + "?mode=ro"
        source_conn: Optional[sqlite3.Connection] = None
        sandbox_conn: Optional[sqlite3.Connection] = None
        materialized = False
        try:
            source_conn = sqlite3.connect(source_uri, uri=True)
            sandbox_conn = sqlite3.connect(self.sandbox_db_path)
            source_conn.backup(sandbox_conn)
            sandbox_conn.commit()
            materialized = True
        finally:
            if sandbox_conn is not None:
                sandbox_conn.close()
            if source_conn is not None:
                source_conn.close()
            if not materialized:
                self.cleanup()
        return self.sandbox_db_path

    def cleanup(self) -> None:
        if self.sandbox_dir and os.path.isdir(self.sandbox_dir):
            shutil.rmtree(self.sandbox_dir, ignore_errors=True)
        self.sandbox_dir = None
        self.sandbox_db_path = None


class UnifiedDQLExecutor:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def execute(self, sql_query: str, mode: str = "probe") -> dict:
        is_success = True
        error_message = None
        returned_rows = []

        import time
        start = time.time()

        result = run_sqlite_query(self.db_path, sql_query, timeout=30.0)

        elapsed_ms = (time.time() - start) * 1000

        if isinstance(result, str) and result.startswith("Error"):
            is_success = False
            error_message = result
        elif isinstance(result, list):
            returned_rows = [list(row) for row in result]
        else:
            returned_rows = []

        return {
            "mode": mode,
            "is_success": is_success,
            "error_message": error_message,
            "returned_rows": returned_rows if mode == "final" else returned_rows[:10],
            "total_row_count": len(returned_rows),
            "execution_time_ms": round(elapsed_ms, 2)
        }


class AdjustmentExecutor:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._backup_created = False

    def _create_backup(self):
        backup_path = self.db_path + ".statesculptor_backup"
        if not self._backup_created:
            shutil.copy2(self.db_path, backup_path)
            self._backup_created = True
        return backup_path

    def rollback(self):
        backup_path = self.db_path + ".statesculptor_backup"
        if self._backup_created and os.path.exists(backup_path):
            shutil.copy2(backup_path, self.db_path)

    def cleanup_backup(self):
        backup_path = self.db_path + ".statesculptor_backup"
        if os.path.exists(backup_path):
            os.remove(backup_path)
        self._backup_created = False

    def _classify_statement(self, stmt: str) -> str:
        upper = stmt.strip().upper()
        if upper.startswith("CREATE") or upper.startswith("ALTER") or upper.startswith("DROP"):
            return "DDL"
        elif upper.startswith("INSERT") or upper.startswith("UPDATE") or upper.startswith("DELETE"):
            return "DML"
        else:
            return "OTHER"

    @staticmethod
    def _strip_identifier(identifier: str) -> str:
        return identifier.strip().strip('`"[]')

    @staticmethod
    def _quote_identifier(identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'

    def _verify_ddl_postcondition(self, cursor: sqlite3.Cursor, stmt: str) -> Optional[str]:
        identifier = r'(`[^`]+`|"[^"]+"|\[[^\]]+\]|[A-Za-z_]\w*)'
        create_match = re.match(
            rf"\s*CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?{identifier}",
            stmt,
            re.IGNORECASE,
        )
        if create_match:
            table_name = self._strip_identifier(create_match.group(1))
            cursor.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table_name,),
            )
            if cursor.fetchone() is None:
                return f"CREATE TABLE postcondition failed for {table_name}"
            return None

        alter_match = re.match(
            rf"\s*ALTER\s+TABLE\s+{identifier}\s+ADD\s+COLUMN\s+{identifier}",
            stmt,
            re.IGNORECASE,
        )
        if alter_match:
            table_name = self._strip_identifier(alter_match.group(1))
            column_name = self._strip_identifier(alter_match.group(2))
            cursor.execute(f"PRAGMA table_info({self._quote_identifier(table_name)})")
            columns = {row[1] for row in cursor.fetchall()}
            if column_name not in columns:
                return (
                    "ALTER TABLE postcondition failed: "
                    f"{column_name} is absent from {table_name}"
                )
        return None

    def execute_ordered_plan(self, statements: list[str]) -> dict:
        self._create_backup()
        execution_log = []
        total_affected = 0
        ddl_count = 0
        dml_count = 0
        conn: Optional[sqlite3.Connection] = None

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            conn.execute("BEGIN")

            for index, stmt in enumerate(statements):
                statement_type = self._classify_statement(stmt)
                log_entry = {
                    "index": index,
                    "statement_type": statement_type,
                    "statement": stmt,
                    "statement_preview": stmt[:150] + "..." if len(stmt) > 150 else stmt,
                }
                execution_log.append(log_entry)

                if statement_type == "OTHER":
                    raise ValueError(
                        f"Unsupported adjustment statement at index {index}: {stmt[:80]}"
                    )

                try:
                    cursor.execute(stmt)
                except sqlite3.Error as error:
                    log_entry["status"] = "FAILED"
                    log_entry["result"] = str(error)[:200]
                    raise RuntimeError(f"Statement {index} failed: {error}") from error

                log_entry["status"] = "SUCCESS"
                log_entry["affected_rows"] = max(cursor.rowcount, 0)
                if statement_type == "DDL":
                    ddl_count += 1
                else:
                    dml_count += 1
                    total_affected += max(cursor.rowcount, 0)
                    if cursor.rowcount == 0 and statement_type == "DML" and stmt.strip().upper().startswith("INSERT"):
                        log_entry["status"] = "FAILED"
                        log_entry["result"] = "No rows affected"
                        raise RuntimeError(
                            f"DML postcondition failed at index {index}: no rows were affected"
                        )

            validation_errors = []
            for log_entry in execution_log:
                if log_entry["statement_type"] == "DDL":
                    ddl_error = self._verify_ddl_postcondition(
                        cursor,
                        log_entry["statement"],
                    )
                    if ddl_error:
                        validation_errors.append(ddl_error)

            cursor.execute("PRAGMA integrity_check")
            integrity_rows = cursor.fetchall()
            integrity_ok = bool(integrity_rows) and all(
                str(row[0]).lower() == "ok" for row in integrity_rows
            )
            if not integrity_ok:
                validation_errors.append(f"Integrity check failed: {integrity_rows}")

            if validation_errors:
                raise RuntimeError("; ".join(validation_errors))

            conn.commit()
            return {
                "all_success": True,
                "total_executed": len(statements),
                "total_affected": total_affected,
                "execution_log": execution_log,
                "ddl_count": ddl_count,
                "dml_count": dml_count,
                "validation_log": "DDL/DML postconditions and integrity check passed.",
            }
        except Exception as error:
            if conn is not None:
                conn.rollback()
            self.rollback()
            return {
                "all_success": False,
                "total_executed": len(execution_log),
                "total_affected": 0,
                "execution_log": execution_log,
                "ddl_count": ddl_count,
                "dml_count": dml_count,
                "validation_log": str(error),
            }
        finally:
            if conn is not None:
                conn.close()


def state_inspection_and_clue_parsing(state: StateSculptorState) -> dict:
    question = state["question"]
    db_path = state["db_path"]
    recovery_json_path = state["recovery_json_path"]
    has_recovery_json = state["has_recovery_json"]

    db_schema = get_db_schema(db_path)

    clue_json = "{}"
    if has_recovery_json and recovery_json_path:
        clue_data = read_json_file(recovery_json_path)
        if clue_data:
            clue_json = json.dumps(clue_data, ensure_ascii=False, indent=2)

    prompt = STATE_INSPECTION_PROMPT.format(
        question=question,
        db_schema=db_schema,
        clue_json=clue_json if clue_json != "{}" else "(No clue provided - database may be intact)"
    )

    response = _call_llm_with_prompt(prompt, "")

    diagnostic_data = _extract_json(response)

    parsed_evidence: ParsedEvidence = {
        "target_table": diagnostic_data.get("parsed_evidence", {}).get("target_table"),
        "target_column": diagnostic_data.get("parsed_evidence", {}).get("target_column"),
        "candidate_types": diagnostic_data.get("parsed_evidence", {}).get("candidate_types"),
        "evidence_records": diagnostic_data.get("parsed_evidence", {}).get("evidence_records", [])
    }

    condition_str = diagnostic_data.get("state_condition_hypothesis", "IS")
    try:
        state_condition = StateCondition(condition_str.upper())
    except ValueError:
        state_condition = StateCondition.INTACT_STATE

    diagnostic_summary: DiagnosticSummary = {
        "state_condition_hypothesis": state_condition,
        "diagnostic_summary": diagnostic_data.get("diagnostic_summary", ""),
        "parsed_evidence": parsed_evidence
    }

    return {"diagnostic_summary": diagnostic_summary}


def pre_adjustment_dql_probe(state: StateSculptorState) -> dict:
    question = state["question"]
    db_path = state["db_path"]
    diagnostic_summary: DiagnosticSummary = state["diagnostic_summary"]

    db_schema = get_db_schema(db_path)
    evidence_str = json.dumps(diagnostic_summary["parsed_evidence"], ensure_ascii=False)

    prompt = DQL_PROBE_PROMPT.format(
        question=question,
        db_schema=db_schema,
        diagnostic_summary=diagnostic_summary["diagnostic_summary"],
        parsed_evidence=evidence_str
    )

    response = _call_llm_with_prompt(prompt, "")
    tentative_query = _extract_sql(response)

    executor = UnifiedDQLExecutor(db_path)
    probe_trace = executor.execute(tentative_query, mode="probe")

    probe_result: ProbeResult = {
        "tentative_query": tentative_query,
        "probe_execution_trace": probe_trace
    }

    return {"probe_result": probe_result}


MAX_RETRIES = 2


def _strip_sql_identifier(identifier: str) -> str:
    return identifier.strip().strip('`"[]')


def _quote_sql_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _parse_update_statement(statement: str) -> Optional[dict]:
    identifier = r'(`[^`]+`|"[^"]+"|\[[^\]]+\]|[A-Za-z_]\w*)'
    match = re.match(
        rf"\s*UPDATE\s+(?P<table>{identifier})\s+SET\s+(?P<set>.+?)\s+WHERE\s+(?P<where>.+?)\s*;?\s*$",
        statement,
        re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return None

    assignments = match.group("set")
    columns = []
    assignment_pattern = re.compile(
        rf"(?:^|,)\s*(?P<column>{identifier})\s*=",
        re.IGNORECASE | re.DOTALL,
    )
    for assignment in assignment_pattern.finditer(assignments):
        column = _strip_sql_identifier(assignment.group("column"))
        if column not in columns:
            columns.append(column)

    if not columns:
        return None
    return {
        "table": _strip_sql_identifier(match.group("table")),
        "columns": columns,
        "where": match.group("where").strip().rstrip(";"),
    }


def _clue_target_columns(clue: dict) -> set[str]:
    columns = set()
    for key in ("target_column", "target_columns", "columns_to_fix"):
        value = clue.get(key)
        if isinstance(value, str) and value:
            columns.add(value.lower())
        elif isinstance(value, list):
            columns.update(str(column).lower() for column in value if column)

    correction_target = clue.get("correction_target")
    if isinstance(correction_target, dict):
        column = correction_target.get("column")
        if isinstance(column, str) and column and column != "*":
            columns.add(column.lower())
    return columns


def update_risk_control(
    db_path: str,
    clue: dict,
    statements: list[str],
    state_condition: str,
) -> dict:
    """Implement g for Data Update actions before sandbox writes occur."""

    if state_condition != StateCondition.DATA_UPDATE.value:
        return {"allowed": True, "checks": [], "reason": "Not a Data Update action."}

    expected_table = str(clue.get("target_table") or "").lower()
    expected_columns = _clue_target_columns(clue)
    checks = []

    if not expected_table or not expected_columns:
        return {
            "allowed": False,
            "checks": checks,
            "reason": "Data Update risk control requires a target table and target column in the clue.",
        }

    conn = sqlite3.connect(db_path)
    try:
        for index, statement in enumerate(statements):
            if not statement.strip().upper().startswith("UPDATE"):
                return {
                    "allowed": False,
                    "checks": checks,
                    "reason": f"Data Update action {index} is not a permitted UPDATE statement.",
                }

            parsed = _parse_update_statement(statement)
            if parsed is None:
                return {
                    "allowed": False,
                    "checks": checks,
                    "reason": f"Update action {index} cannot be parsed safely.",
                }

            table_name = parsed["table"]
            target_columns = {column.lower() for column in parsed["columns"]}
            check = {"index": index, "table": table_name, "columns": parsed["columns"]}
            checks.append(check)

            if table_name.lower() != expected_table:
                return {
                    "allowed": False,
                    "checks": checks,
                    "reason": f"Update action {index} targets {table_name}, not the clue table.",
                }
            if not target_columns.issubset(expected_columns):
                return {
                    "allowed": False,
                    "checks": checks,
                    "reason": f"Update action {index} modifies columns outside the clue.",
                }

            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({_quote_sql_identifier(table_name)})")
            existing_columns = {row[1].lower() for row in cursor.fetchall()}
            if not target_columns.issubset(existing_columns):
                return {
                    "allowed": False,
                    "checks": checks,
                    "reason": f"Update action {index} does not target an existing column.",
                }

            try:
                cursor.execute(f"SELECT COUNT(*) FROM {_quote_sql_identifier(table_name)}")
                total_rows = int(cursor.fetchone()[0])
                cursor.execute(
                    f"SELECT COUNT(*) FROM {_quote_sql_identifier(table_name)} WHERE {parsed['where']}"
                )
                matched_rows = int(cursor.fetchone()[0])
            except sqlite3.Error as error:
                return {
                    "allowed": False,
                    "checks": checks,
                    "reason": f"Update action {index} has an invalid WHERE predicate: {error}",
                }

            check.update({"matched_rows": matched_rows, "total_rows": total_rows})
            if total_rows == 0 or matched_rows / total_rows > 0.01:
                return {
                    "allowed": False,
                    "checks": checks,
                    "reason": (
                        f"Update action {index} exceeds the 1% row-impact limit "
                        f"({matched_rows}/{total_rows})."
                    ),
                }
    finally:
        conn.close()

    return {"allowed": True, "checks": checks, "reason": "All Data Update actions passed risk control."}


def adjustment_planning_execution_verification(state: StateSculptorState) -> dict:
    question = state["question"]
    db_path = state["db_path"]
    diagnostic_summary: DiagnosticSummary = state["diagnostic_summary"]
    probe_result: ProbeResult = state["probe_result"]

    evidence_str = json.dumps(diagnostic_summary["parsed_evidence"], ensure_ascii=False)
    probe_trace_str = json.dumps(probe_result["probe_execution_trace"], ensure_ascii=False)
    state_condition = diagnostic_summary["state_condition_hypothesis"].value if isinstance(diagnostic_summary["state_condition_hypothesis"], StateCondition) else str(diagnostic_summary["state_condition_hypothesis"])

    executor = AdjustmentExecutor(db_path)
    risk_clue = dict(diagnostic_summary["parsed_evidence"])
    recovery_json_path = state.get("recovery_json_path", "")
    if recovery_json_path and os.path.isfile(recovery_json_path):
        raw_clue = read_json_file(recovery_json_path)
        if raw_clue:
            risk_clue.update(raw_clue)

    generated_statements: list[str] = []
    last_failure = "No adjustment plan was produced."

    for attempt in range(MAX_RETRIES + 1):
        retry_hint = ""
        if attempt > 0:
            retry_hint = f"\n\n## RETRY #{attempt}\nPrevious attempt failed. Use the error information to revise your plan."

        prompt = ADJUSTMENT_PLANNING_PROMPT.format(
            question=question,
            parsed_evidence=evidence_str,
            probe_trace=probe_trace_str,
            state_condition=state_condition
        ) + retry_hint

        response = _call_llm_with_prompt(prompt, "")
        plan_data = _extract_json(response)

        adjustment_performed = plan_data.get("adjustment_performed", False)
        generated_statements = plan_data.get("generated_statements", [])

        probe_failed = not probe_result.get("probe_execution_trace", {}).get("is_success", True)
        is_intact_state = state_condition == "IS"

        if (not adjustment_performed or not generated_statements) and not (is_intact_state and not probe_failed):
            last_failure = (
                f"State={state_condition} requires adjustment, but the planner returned a no-op."
            )
            if attempt < MAX_RETRIES:
                probe_trace_str = json.dumps({
                    "original_trace": probe_result["probe_execution_trace"],
                    "retry_attempt": attempt,
                    "force_adjustment": True,
                    "error_details": last_failure,
                }, ensure_ascii=False)
                continue
            break

        if not adjustment_performed or not generated_statements:
            executor.cleanup_backup()

            return {
                "adjustment_plan": {
                    "adjustment_performed": False,
                    "generated_statements": [],
                    "plan_reasoning": "No adjustment needed (Intact State or No-op)",
                    "is_noop": True
                },
                "verification_result": {
                    "verification_label": True,
                    "verification_details": "No adjustment performed - database was already intact.",
                    "affected_rows": 0,
                    "schema_check_passed": True
                },
                "repair_sql": ""
            }

        risk_result = update_risk_control(
            db_path=db_path,
            clue=risk_clue,
            statements=generated_statements,
            state_condition=state_condition,
        )
        if not risk_result["allowed"]:
            last_failure = risk_result["reason"]
            if attempt < MAX_RETRIES:
                probe_trace_str = json.dumps({
                    "original_trace": probe_result["probe_execution_trace"],
                    "retry_attempt": attempt,
                    "risk_control": risk_result,
                    "error_details": last_failure,
                }, ensure_ascii=False)
                continue
            break

        exec_result = executor.execute_ordered_plan(generated_statements)

        if exec_result["all_success"]:
            verification_log = (
                f"All {exec_result['total_executed']} statements executed successfully. "
                f"DDL: {exec_result['ddl_count']}, DML: {exec_result['dml_count']}. "
                f"{exec_result['validation_log']}"
            )

            executor.cleanup_backup()

            verification_result: VerificationResult = {
                "verification_label": True,
                "verification_details": verification_log,
                "affected_rows": exec_result["total_affected"],
                "schema_check_passed": True
            }

            repair_sql_str = "\n".join(generated_statements)

            return {
                "adjustment_plan": {
                    "adjustment_performed": True,
                    "generated_statements": generated_statements,
                    "plan_reasoning": plan_data.get("plan_reasoning", ""),
                    "is_noop": False
                },
                "verification_result": verification_result,
                "repair_sql": repair_sql_str
            }
        else:
            last_failure = exec_result["validation_log"]
            probe_trace_str = json.dumps({
                "original_trace": probe_result["probe_execution_trace"],
                "retry_attempt": attempt,
                "execution_errors": [log for log in exec_result["execution_log"] if log.get("status") == "FAILED"],
                "verification_error": last_failure,
            }, ensure_ascii=False)

    executor.cleanup_backup()

    verification_result: VerificationResult = {
        "verification_label": False,
        "verification_details": f"Failed after {MAX_RETRIES + 1} attempts: {last_failure}",
        "affected_rows": 0,
        "schema_check_passed": False
    }

    return {
        "adjustment_plan": {
            "adjustment_performed": False,
            "generated_statements": generated_statements,
            "plan_reasoning": "All retries exhausted",
            "is_noop": False
        },
        "verification_result": verification_result,
        "repair_sql": "\n".join(generated_statements)
    }


def post_adjustment_dql_execution(state: StateSculptorState) -> dict:
    question = state["question"]
    db_path = state["db_path"]
    probe_result: ProbeResult = state["probe_result"]
    verification_result: VerificationResult = state["verification_result"]

    db_schema = get_db_schema(db_path)
    probe_query = probe_result.get("tentative_query", "")
    verification_log = verification_result.get("verification_details", "")

    prompt = POST_DQL_EXECUTION_PROMPT.format(
        question=question,
        db_schema=db_schema,
        probe_query=probe_query
    )

    response = _call_llm_with_prompt(prompt, "")
    final_query = _extract_sql(response)

    executor = UnifiedDQLExecutor(db_path)
    dql_exec_trace = executor.execute(final_query, mode="final")

    final_answer = ""
    if dql_exec_trace["is_success"] and dql_exec_trace["returned_rows"]:
        rows = dql_exec_trace["returned_rows"]
        if len(rows) == 1 and len(rows[0]) == 1:
            final_answer = str(rows[0][0])
        else:
            final_answer = json.dumps(rows, ensure_ascii=False)
    elif not dql_exec_trace["is_success"]:
        final_answer = f"Query failed: {dql_exec_trace['error_message']}"
    else:
        final_answer = "No results found"

    dql_execution_result: DQLExecutionResult = {
        "final_query": final_query,
        "final_answer": final_answer,
        "execution_success": dql_exec_trace["is_success"],
        "result_rows": dql_exec_trace["returned_rows"]
    }

    return {
        "dql_execution_result": dql_execution_result,
        "final_sql": final_query,
        "final_answer": final_answer
    }


