"""Unified StateSculptor database interaction workflow."""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .skills import (
    AdjustmentExecutor,
    SandboxMaterializer,
    UnifiedDQLExecutor,
    _extract_json,
    _extract_sql,
)
from ..environment import _generate_context_and_modify
from .state import StateCondition
from ..config import get_llm
from ..tools import get_db_schema, read_json_file


LLMCallable = Callable[[str], str]


@dataclass
class PaperClue:
    """Non-executable clue normalized from a request and attachment."""

    case_type: str = "intact_state"
    target_table: Optional[str] = None
    target_columns: list[str] = field(default_factory=list)
    required_table: Optional[str] = None
    required_columns: list[str] = field(default_factory=list)
    key_fields: dict[str, Any] = field(default_factory=dict)
    primary_key: Optional[str] = None
    candidate_types: dict[str, str] = field(default_factory=dict)
    table_description: dict[str, Any] = field(default_factory=dict)
    evidence_records: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_type": self.case_type,
            "target_table": self.target_table,
            "target_columns": self.target_columns,
            "required_table": self.required_table,
            "required_columns": self.required_columns,
            "key_fields": self.key_fields,
            "primary_key": self.primary_key,
            "candidate_types": self.candidate_types,
            "table_description": self.table_description,
            "evidence_records": self.evidence_records,
        }


@dataclass
class ProbeDiagnosis:
    query: str
    trace: dict[str, Any]
    condition: str
    early_risk: Optional[dict[str, Any]] = None
    planning_error: Optional[str] = None


def _invoke_llm(prompt: str, llm: Any = None) -> str:
    """Invoke an LLM; tests may also provide a regular callable."""
    if llm is None:
        llm = get_llm()
    if callable(llm) and not hasattr(llm, "invoke"):
        return str(llm(prompt))
    response = llm.invoke(prompt)
    return getattr(response, "content", str(response))


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _compact_clue_dict(clue: PaperClue, max_records: int = 20) -> dict[str, Any]:
    """Build a compact clue view for the LLM while retaining full evidence."""
    data = clue.as_dict()
    records = data.get("evidence_records") or []
    if len(records) > max_records:
        data["evidence_records"] = records[:max_records]
        data["evidence_truncated_for_prompt"] = True
        data["evidence_row_count"] = len(records)
    return data


def normalize_clue(data: dict[str, Any]) -> PaperClue:
    """Normalize legacy recovery.json fields while preserving table/column roles."""
    evidence = data.get("parsed_evidence") if isinstance(data.get("parsed_evidence"), dict) else data
    case_type = str(data.get("case_type") or data.get("state_condition") or "intact_state").lower()
    aliases = {"du": "data_update", "mc": "missing_column", "mt": "missing_table", "is": "intact_state", "data_corruption": "data_update"}
    case_type = aliases.get(case_type, case_type)
    target_table = evidence.get("target_table") or data.get("target_table")
    required_table = evidence.get("required_table") or data.get("required_table")
    if case_type == "missing_table" and required_table is None:
        required_table = target_table
    if case_type == "missing_table":
        target_table = None
    target_columns = _as_list(evidence.get("target_columns") or evidence.get("target_column") or data.get("columns_to_fix") or data.get("target_column"))
    required_columns = _as_list(evidence.get("required_columns") or data.get("payload_columns"))
    candidate = evidence.get("candidate_types") or data.get("candidate_type") or data.get("types") or {}
    if isinstance(candidate, str) and target_columns:
        candidate = {target_columns[0]: candidate}
    if not isinstance(candidate, dict):
        candidate = {}
    records = evidence.get("evidence_records") or data.get("data_payload") or []
    if not isinstance(records, list):
        records = []
    return PaperClue(
        case_type=case_type,
        target_table=target_table,
        target_columns=target_columns,
        required_table=required_table,
        required_columns=required_columns,
        key_fields=evidence.get("key_fields") or data.get("key_fields") or {},
        primary_key=evidence.get("primary_key") or data.get("primary_key"),
        candidate_types={str(k): str(v) for k, v in candidate.items()},
        table_description=evidence.get("table_description") or data.get("table_description") or {},
        evidence_records=[row for row in records if isinstance(row, dict)],
        raw=data,
    )


def build_clue(question: str, db_path: str, attachment_path: str = "", llm: Any = None) -> PaperClue:
    """Algorithm 1: read schema/attachment and normalize a clue with the LLM."""
    attachment = read_json_file(attachment_path) if attachment_path and os.path.isfile(attachment_path) else {}
    prompt_attachment = dict(attachment)
    attachment_records = prompt_attachment.get("evidence_records") or []
    if len(attachment_records) > 20:
        prompt_attachment["evidence_records"] = attachment_records[:20]
        prompt_attachment["evidence_truncated_for_prompt"] = True
        prompt_attachment["evidence_row_count"] = len(attachment_records)
    prompt = (
        "Extract a non-executable database clue from the request. Preserve distinct "
        "target_table/target_columns and required_table/required_columns fields. "
        "Return JSON only.\nQuestion:\n" + question + "\nSchema:\n" + get_db_schema(db_path)
        + "\nAttachment:\n" + json.dumps(prompt_attachment, ensure_ascii=False)
    )
    if not attachment:
        # No attachment denotes an intact state.
        return PaperClue()
    try:
        parsed = _extract_json(_invoke_llm(prompt, llm)) if llm is not None else attachment
    except Exception:
        parsed = {}
    # Attachment fields are authoritative state evidence and must not be overwritten.
    if attachment:
        parsed = dict(parsed) if isinstance(parsed, dict) else {}
        for key in ("case_type", "target_table", "target_columns", "required_table",
                    "required_columns", "candidate_types", "table_description", "primary_key",
                    "evidence_records"):
            if key in attachment:
                parsed[key] = attachment[key]
    return normalize_clue(parsed if isinstance(parsed, dict) else {})


def _condition_from_clue(clue: PaperClue) -> str:
    return {"data_update": "DU", "missing_column": "MC", "missing_table": "MT", "intact_state": "IS"}.get(clue.case_type, "IS")


def _count_probe_rows(trace: dict[str, Any], clue: Optional[PaperClue] = None) -> int:
    # For Data Update, attachment records directly count the request scope.
    if clue is not None and clue.evidence_records:
        return len(clue.evidence_records)
    value = trace.get("total_row_count")
    return int(value) if isinstance(value, (int, float)) else len(trace.get("returned_rows") or [])


def plan_actions(
    question: str,
    db_path: str,
    clue: PaperClue,
    risk_threshold: float = 0.01,
    llm: Any = None,
    use_probe: bool = True,
) -> tuple[ProbeDiagnosis, list[str]]:
    """Algorithm 2: generate and execute SELECT before deciding whether to generate A."""
    probe_prompt = (
        "Generate one read-only SELECT query answering the request. Return SQL only.\n"
        f"Request: {question}\nSchema:\n{get_db_schema(db_path)}\nClue:\n{json.dumps(_compact_clue_dict(clue), ensure_ascii=False)}"
    )
    if use_probe:
        probe_query = _extract_sql(_invoke_llm(probe_prompt, llm))
        trace = UnifiedDQLExecutor(db_path).execute(probe_query, mode="probe")
    else:
        probe_query = ""
        trace = {"mode": "probe", "is_success": True, "error_message": None,
                 "returned_rows": [], "total_row_count": 0, "execution_time_ms": 0.0,
                 "disabled": True}
    condition = _condition_from_clue(clue)
    early_risk = None
    if condition == "DU" and clue.target_table:
        import sqlite3
        with sqlite3.connect(db_path) as conn:
            total = int(conn.execute(f'SELECT COUNT(*) FROM "{clue.target_table.replace(chr(34), chr(34)*2)}"').fetchone()[0])
        matched = _count_probe_rows(trace, clue)
        early_risk = {"matched_rows": matched, "target_rows": total, "threshold": risk_threshold}
        if total > 0 and matched / total > risk_threshold:
            return ProbeDiagnosis(probe_query, trace, condition, early_risk), []
    if condition == "IS":
        return ProbeDiagnosis(probe_query, trace, condition, early_risk), []
    planning_prompt = (
        "Generate an ordered JSON action plan. Use only DDL/DML, never SELECT. "
        "For MC add the column before backfilling; for MT create the required table before inserts; "
        "for DU use key-grounded UPDATE statements.\n"
        f"Request: {question}\nCondition: {condition}\nClue: {json.dumps(_compact_clue_dict(clue), ensure_ascii=False)}\n"
        f"Probe trace: {json.dumps(trace, ensure_ascii=False)}"
    )
    data = _extract_json(_invoke_llm(planning_prompt, llm))
    actions = data.get("generated_statements") if isinstance(data, dict) else []
    # Non-intact states require repair actions; retry once when the first plan is empty.
    if condition in {"DU", "MC", "MT"} and not actions:
        retry_prompt = planning_prompt + "\nMANDATORY: the state is not intact. Return a non-empty generated_statements list now."
        retry_data = _extract_json(_invoke_llm(retry_prompt, llm))
        actions = retry_data.get("generated_statements") if isinstance(retry_data, dict) else []
    normalized_actions = [str(a).strip() for a in actions or [] if str(a).strip()]
    # Syntax and operation type are handled during generation; risk control checks UPDATE scope.
    unsupported = [a for a in normalized_actions if not _is_supported_action(a)]
    if unsupported:
        # Replace mixed or unsupported output with an evidence-driven deterministic plan.
        fallback_actions = _deterministic_actions(clue, db_path)
        if fallback_actions:
            normalized_actions = fallback_actions
        else:
            return ProbeDiagnosis(probe_query, trace, condition, early_risk, "Unsupported non-DDL/DML action generated."), []
    if _mt_actions_need_fallback(clue, normalized_actions) or _mc_actions_need_fallback(clue, normalized_actions):
        normalized_actions = _deterministic_actions(clue, db_path)
    elif not normalized_actions and condition in {"DU", "MC", "MT"}:
        normalized_actions = _deterministic_actions(clue, db_path)
    return ProbeDiagnosis(probe_query, trace, condition, early_risk), normalized_actions


def _is_supported_action(statement: str) -> bool:
    """Supported action space: CREATE, ALTER, INSERT, and UPDATE."""
    return statement.strip().upper().startswith(("CREATE ", "ALTER ", "INSERT ", "UPDATE "))


def _parse_update(statement: str) -> Optional[dict[str, str]]:
    match = re.match(r"\s*UPDATE\s+([`\"\[]?[\w .-]+[`\"\]]?)\s+SET\s+(.+?)\s+WHERE\s+(.+?)\s*;?\s*$", statement, re.I | re.S)
    if not match:
        return None
    cols = re.findall(r"(?:^|,)\s*[`\"\[]?([A-Za-z_]\w*)[`\"\]]?\s*=", match.group(2))
    return {"table": match.group(1).strip(' `"[]'), "where": match.group(3).rstrip(';').strip(), "columns": cols}


def risk_control(
    db_path: str,
    question: str,
    clue: PaperClue,
    statements: list[str],
    risk_threshold: float = 0.01,
    llm: Any = None,
) -> dict[str, Any]:
    """Algorithm 4: control the scope of valid UPDATE actions and delegate unknown cases to the LLM."""
    if _condition_from_clue(clue) != "DU":
        return {"allowed": True, "checks": [], "reason": "UPDATE risk policy is not applicable."}
    import sqlite3
    checks: list[dict[str, Any]] = []
    with sqlite3.connect(db_path) as conn:
        for index, statement in enumerate(statements):
            parsed = _parse_update(statement)
            if parsed is not None:
                safe_table = parsed["table"].replace('"', '""')
                count_error = None
                try:
                    total = int(conn.execute(f'SELECT COUNT(*) FROM "{safe_table}"').fetchone()[0])
                    matched = int(conn.execute(f'SELECT COUNT(*) FROM "{safe_table}" WHERE {parsed["where"]}').fetchone()[0])
                except sqlite3.Error as error:
                    total = matched = 0
                    count_error = str(error)
                check = {"index": index, "matched_rows": matched, "target_rows": total, "statement": statement}
                checks.append(check)
                if count_error:
                    fallback_prompt = (
                        "Assess whether this candidate database action is safe and request-consistent. "
                        "Return exactly ALLOW or REJECT. Unknown or suspicious actions must be REJECT.\n"
                        f"Request: {question}\nClue: {json.dumps(_compact_clue_dict(clue), ensure_ascii=False)}\nAction: {statement}\n"
                        f"Counting error: {count_error}"
                    )
                    verdict = _invoke_llm(fallback_prompt, llm).strip().upper() if llm is not None else "REJECT"
                    check["llm_verdict"] = verdict
                    if "ALLOW" not in verdict or "REJECT" in verdict:
                        return {"allowed": False, "checks": checks, "reason": "LLM risk assessor rejected an uncountable UPDATE."}
                if total > 0 and matched / total > risk_threshold:
                    return {"allowed": False, "checks": checks, "reason": "UPDATE exceeds the configured row-impact threshold."}
                continue
            fallback_prompt = (
                "Assess whether this candidate database action is safe and request-consistent. "
                "Return exactly ALLOW or REJECT. Unknown or suspicious actions must be REJECT.\n"
                f"Request: {question}\nClue: {json.dumps(_compact_clue_dict(clue), ensure_ascii=False)}\nAction: {statement}"
            )
            verdict = _invoke_llm(fallback_prompt, llm).strip().upper() if llm is not None else "REJECT"
            checks.append({"index": index, "statement": statement, "llm_verdict": verdict})
            if "REJECT" in verdict or "ALLOW" not in verdict:
                return {"allowed": False, "checks": checks, "reason": "LLM risk assessor rejected an unknown action."}
    return {"allowed": True, "checks": checks, "reason": "All UPDATE actions passed risk control."}


def _infer_condition(modified_sql: str) -> str:
    text = (modified_sql or "").strip().upper()
    if text.startswith("UPDATE"):
        return "DU"
    if "DROP COLUMN" in text or text.startswith("DROP FIELD"):
        return "MC"
    if text.startswith("DROP TABLE"):
        return "MT"
    return "IS"


def _parse_modified_target(modified_sql: str) -> tuple[str, str, str]:
    """Parse DROP TABLE, DROP FIELD, and DROP COLUMN targets used by the test-data builder."""
    text = (modified_sql or "").strip()
    table_identifier = r'(`[^`]+`|"[^"]+"|\[[^\]]+\]|[^\s;]+)'
    column_identifier = r'(`[^`]+`|"[^"]+"|\[[^\]]+\]|.+?)'

    match = re.match(
        rf"DROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?P<table>{table_identifier})\s*;?$",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        return "MT", match.group("table").strip('`"[]'), ""

    match = re.match(
        rf"DROP\s+FIELD\s+(?P<table>{table_identifier})\s*\.\s*(?P<column>{column_identifier})\s*;?$",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        return "MC", match.group("table").strip('`"[]'), match.group("column").strip('`"[]').strip()

    match = re.match(
        rf"ALTER\s+TABLE\s+(?P<table>{table_identifier})\s+DROP\s+COLUMN\s+(?P<column>{column_identifier})\s*;?$",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        return "MC", match.group("table").strip('`"[]'), match.group("column").strip('`"[]').strip()

    return "", "", ""


def _schema_metadata(schema_sql: str) -> dict[str, Any]:
    """Convert CREATE TABLE text into non-executable column/type metadata."""
    if not schema_sql:
        return {"columns": [], "types": {}}
    body = schema_sql[schema_sql.find("(") + 1 : schema_sql.rfind(")")]
    columns: list[str] = []
    types: dict[str, str] = {}
    for part in re.split(r',\s*(?=[`"\[]?[A-Za-z_])', body):
        token = part.strip()
        match = re.match(r'[`"\[]?([^`"\]\s]+)[`"\]]?\s+([A-Za-z]+)', token)
        if match and match.group(1).upper() not in {"PRIMARY", "FOREIGN", "UNIQUE", "CONSTRAINT", "CHECK"}:
            columns.append(match.group(1).strip())
            types[match.group(1).strip()] = match.group(2).upper()
    return {"columns": columns, "types": types}


def _read_table_metadata(db_path: str, table_name: str) -> dict[str, Any]:
    """Read exact SQLite column names and types without lossy CREATE TABLE parsing."""
    if not table_name:
        return {"columns": [], "types": {}}
    safe_name = table_name.replace('"', '""')
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(f'PRAGMA table_info("{safe_name}")').fetchall()
    columns = [str(row[1]) for row in rows]
    types = {str(row[1]): (str(row[2]) if row[2] else "TEXT") for row in rows}
    return {"columns": columns, "types": types}


def _query_relevant_columns(
    db_path: str,
    table_name: str,
    gold_sql: str,
    fallback: Optional[list[str]] = None,
) -> list[str]:
    """Select columns needed by gold SQL while retaining primary/join keys."""
    metadata = _read_table_metadata(db_path, table_name)
    columns = metadata.get("columns", [])
    if not columns:
        return list(fallback or [])
    sql_lower = (gold_sql or "").lower()
    selected = [
        str(column) for column in columns
        if str(column).lower() in sql_lower
    ]
    primary = None
    safe_name = table_name.replace(chr(34), chr(34) * 2)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(f'PRAGMA table_info("{safe_name}")').fetchall()
        primary = next((str(row[1]) for row in rows if row[5] > 0), None)
    if primary and primary not in selected:
        selected.insert(0, primary)
    return selected or list(fallback or columns[:1])


def _sql_literal(value: Any) -> str:
    """Encode a Python value as a SQLite literal."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def _deterministic_actions(clue: PaperClue, db_path: str) -> list[str]:
    """Generate a conservative DDL/DML fallback plan from non-executable evidence."""
    actions: list[str] = []
    if clue.case_type == "missing_table" and clue.required_table:
        table = clue.required_table.replace('"', '""')
        columns = list(clue.required_columns or clue.table_description.get("columns") or [])
        types = clue.table_description.get("types") or {}
        if not columns and clue.evidence_records:
            columns = list(clue.evidence_records[0].keys())
        if not columns:
            return []
        definitions = [f'"{str(col).replace(chr(34), chr(34) * 2)}" {types.get(col, "TEXT")}' for col in columns]
        actions.append(f'CREATE TABLE "{table}" ({", ".join(definitions)});')
        rows = clue.evidence_records
        if rows:
            names = ", ".join(f'"{str(col).replace(chr(34), chr(34) * 2)}"' for col in columns)
            # Generate INSERT statements in chunks to avoid oversized statements.
            for start in range(0, len(rows), 100):
                chunk = rows[start : start + 100]
                values = [
                    "(" + ", ".join(_sql_literal(row.get(col)) for col in columns) + ")"
                    for row in chunk
                ]
                actions.append(f'INSERT INTO "{table}" ({names}) VALUES {", ".join(values)};')
        return actions

    if clue.case_type == "missing_column" and clue.target_table and clue.target_columns:
        table = clue.target_table.replace('"', '""')
        column = clue.target_columns[0]
        col_type = clue.candidate_types.get(column, "TEXT")
        actions.append(f'ALTER TABLE "{table}" ADD COLUMN "{column.replace(chr(34), chr(34) * 2)}" {col_type};')
        key = clue.primary_key or ("rowid" if clue.evidence_records and "rowid" in clue.evidence_records[0] else None)
        rows = [row for row in clue.evidence_records if column in row]
        if key and rows:
            # Backfill values in CASE chunks instead of thousands of individual UPDATEs.
            for start in range(0, len(rows), 200):
                chunk = rows[start : start + 200]
                cases = " ".join(
                    f"WHEN {_sql_literal(row.get(key))} THEN {_sql_literal(row[column])}" for row in chunk
                )
                keys = ", ".join(_sql_literal(row.get(key)) for row in chunk)
                key_expr = "rowid" if key == "rowid" else f'"{key.replace(chr(34), chr(34) * 2)}"'
                actions.append(
                    f'UPDATE "{table}" SET "{column.replace(chr(34), chr(34) * 2)}" = '
                    f'CASE {key_expr} {cases} END WHERE {key_expr} IN ({keys});'
                )
        else:
            for row in rows:
                keys = [k for k in row if k != column]
                if not keys:
                    continue
                predicate = " AND ".join(
                    f'"{k.replace(chr(34), chr(34) * 2)}" IS {_sql_literal(row[k])}' if row[k] is None
                    else f'"{k.replace(chr(34), chr(34) * 2)}" = {_sql_literal(row[k])}' for k in keys
                )
                actions.append(f'UPDATE "{table}" SET "{column.replace(chr(34), chr(34) * 2)}" = {_sql_literal(row[column])} WHERE {predicate};')
        return actions

    if clue.case_type == "data_update" and clue.target_table and clue.target_columns:
        table = clue.target_table.replace('"', '""')
        for row in clue.evidence_records:
            keys = [k for k in row if k not in clue.target_columns]
            if not keys:
                continue
            assignments = ", ".join(
                f'"{c.replace(chr(34), chr(34) * 2)}" = {_sql_literal(row[c])}'
                for c in clue.target_columns if c in row
            )
            if not assignments:
                continue
            predicate = " AND ".join(
                f'"{k.replace(chr(34), chr(34) * 2)}" IS {_sql_literal(row[k])}' if row[k] is None
                else f'"{k.replace(chr(34), chr(34) * 2)}" = {_sql_literal(row[k])}' for k in keys
            )
            actions.append(f'UPDATE "{table}" SET {assignments} WHERE {predicate};')
        return actions
    return actions


def _mt_actions_need_fallback(clue: PaperClue, actions: list[str]) -> bool:
    """Check whether a Missing Table plan loses required columns/rows or structure."""
    if clue.case_type != "missing_table" or not clue.required_table:
        return False
    if len(clue.evidence_records) > 20:
        return True
    create = next((a for a in actions if a.strip().upper().startswith("CREATE TABLE")), "")
    expected = clue.required_columns or clue.table_description.get("columns") or []
    return not create or any(str(col).lower() not in create.lower() for col in expected)


def _mc_actions_need_fallback(clue: PaperClue, actions: list[str]) -> bool:
    """Missing Column plans must cover all original values in the attachment."""
    if clue.case_type != "missing_column" or not clue.target_columns:
        return False
    if len(clue.evidence_records) > 20:
        return True
    return not any(a.strip().upper().startswith("ALTER TABLE") for a in actions)


def construct_test_data(
    intact_db_path: str,
    gold_sql: str,
    modified_sql: str = "",
    condition: str = "",
    output_dir: str = "",
    max_attachment_rows: int = 0,
) -> dict[str, Any]:
    """Algorithm 6: construct the current state and non-executable attachment from BIRD."""
    if not os.path.isfile(intact_db_path):
        raise FileNotFoundError(intact_db_path)
    tau_raw = condition.upper() if condition else _infer_condition(modified_sql)
    tau = {"DATA_UPDATE": "DU", "MISSING_COLUMN": "MC", "MISSING_TABLE": "MT", "INTACT_STATE": "IS"}.get(tau_raw, tau_raw)
    work_dir = output_dir or tempfile.mkdtemp(prefix="statesculptor_testdata_")
    os.makedirs(work_dir, exist_ok=True)
    current_db = os.path.join(work_dir, "current.sqlite")
    shutil.copy2(intact_db_path, current_db)
    if tau == "IS" or not modified_sql.strip():
        return {"database_path": current_db, "attachment": {}, "condition": "IS", "work_dir": work_dir}
    # Preserve original types before destructive schema changes.
    original_types: dict[str, str] = {}
    parsed_kind, parsed_table, parsed_column = _parse_modified_target(modified_sql)
    if parsed_kind == "MC" and tau == "MC":
        table_name = parsed_table
        column_name = parsed_column
        try:
            with sqlite3.connect(current_db) as conn:
                for row in conn.execute(f'PRAGMA table_info("{table_name.replace(chr(34), chr(34)*2)}")'):
                    if not column_name or row[1].lower() == column_name.lower():
                        original_types[row[1]] = row[2] or "TEXT"
        except sqlite3.Error:
            pass
    # Read exact schema metadata before the change for recovery planning only.
    target_for_metadata = ""
    if parsed_kind in {"MT", "MC"}:
        target_for_metadata = parsed_table
    table_metadata = _read_table_metadata(current_db, target_for_metadata) if target_for_metadata else {}
    context = _generate_context_and_modify(current_db, modified_sql, gold_sql)
    attachment: dict[str, Any] = {"case_type": {"DU": "data_update", "MC": "missing_column", "MT": "missing_table"}.get(tau, "intact_state")}
    if tau == "MT":
        attachment["required_table"] = context.get("target_table")
        attachment["table_description"] = table_metadata or _schema_metadata(context.get("table_schema", ""))
        attachment["required_columns"] = _query_relevant_columns(
            intact_db_path, target_for_metadata, gold_sql,
            context.get("payload_columns", []) or attachment["table_description"].get("columns", []),
        )
    else:
        attachment["target_table"] = context.get("target_table")
        attachment["target_columns"] = context.get("columns_to_fix", [])
        if context.get("primary_key"):
            attachment["primary_key"] = context.get("primary_key")
        if tau == "MC":
            attachment["candidate_types"] = {
                str(column): original_types.get(str(column), "TEXT") for column in attachment["target_columns"]
            }
    evidence_records = context.get("data_payload", []) or []
    if tau == "MC" and target_for_metadata and attachment.get("target_columns"):
        # Missing Column requires all original values; limiting rows can change results.
        try:
            safe_table = target_for_metadata.replace(chr(34), chr(34) * 2)
            meta = _read_table_metadata(intact_db_path, target_for_metadata)
            primary_name = str(context.get("primary_key") or "")
            primary = next((c for c in meta.get("columns", []) if c.lower() == primary_name.lower()), None)
            cols = ([primary] if primary else []) + [c for c in attachment["target_columns"] if c in meta.get("columns", []) and c != primary]
            if cols:
                select_cols = ", ".join(f'"{str(col).replace(chr(34), chr(34) * 2)}"' for col in cols)
                with sqlite3.connect(intact_db_path) as source_conn:
                    source_conn.row_factory = sqlite3.Row
                    evidence_records = [dict(row) for row in source_conn.execute(f'SELECT {select_cols} FROM "{safe_table}"').fetchall()]
        except sqlite3.Error:
            pass
    if tau == "MT":
        # Missing Table retains all query-relevant rows; limiting rows can change results.
        required_cols = attachment.get("required_columns", []) or table_metadata.get("columns", [])
        if required_cols:
            try:
                safe_table = target_for_metadata.replace('"', '""')
                select_cols = ", ".join(f'"{str(col).replace(chr(34), chr(34) * 2)}"' for col in required_cols)
                with sqlite3.connect(intact_db_path) as source_conn:
                    source_conn.row_factory = sqlite3.Row
                    full_rows = [dict(row) for row in source_conn.execute(f'SELECT {select_cols} FROM "{safe_table}"').fetchall()]
                if len(full_rows) > len(evidence_records):
                    evidence_records = full_rows
            except sqlite3.Error:
                pass
    if max_attachment_rows > 0 and len(evidence_records) > max_attachment_rows:
        attachment["evidence_records"] = evidence_records[:max_attachment_rows]
        attachment["evidence_truncated"] = True
        attachment["evidence_row_count"] = len(evidence_records)
    else:
        attachment["evidence_records"] = evidence_records
    return {"database_path": current_db, "attachment": attachment, "condition": tau, "work_dir": work_dir}


def run_workflow(
    question: str,
    db_path: str,
    attachment_path: str = "",
    llm: Any = None,
    risk_threshold: float = 0.01,
    keep_sandbox: bool = False,
    use_probe: bool = True,
    use_final_query: bool = True,
    reuse_probe_query: bool = False,
) -> dict[str, Any]:
    """Run the complete four-stage workflow and return a structured result."""
    materializer = SandboxMaterializer(db_path)
    sandbox = materializer.materialize()
    try:
        clue = build_clue(question, sandbox, attachment_path, llm)
        diagnosis, actions = plan_actions(question, sandbox, clue, risk_threshold, llm, use_probe=use_probe)
        if diagnosis.planning_error:
            return {
                "question": question, "clue": clue.as_dict(), "condition": diagnosis.condition,
                "probe": {"query": diagnosis.query, "trace": diagnosis.trace},
                "status": "planning_failed", "error": diagnosis.planning_error,
                "actions": [], "sandbox_db_path": sandbox if keep_sandbox else None,
            }
        risk = risk_control(sandbox, question, clue, actions, risk_threshold, llm)
        if actions and not risk["allowed"]:
            return {
                "question": question, "clue": clue.as_dict(), "condition": diagnosis.condition,
                "probe": {"query": diagnosis.query, "trace": diagnosis.trace, "early_risk": diagnosis.early_risk},
                "status": "rejected_by_risk_control", "actions": actions,
                "risk_control": risk, "verification_passed": False,
                "sandbox_db_path": sandbox if keep_sandbox else None,
            }
        execution = {"all_success": True, "total_affected": 0, "execution_log": []}
        verification = True
        if actions and risk["allowed"]:
            execution = AdjustmentExecutor(sandbox).execute_ordered_plan(actions)
            verification = bool(execution.get("all_success"))
        elif actions and not risk["allowed"]:
            verification = False
        if diagnosis.early_risk and diagnosis.early_risk.get("target_rows", 0) > 0 and diagnosis.early_risk.get("matched_rows", 0) / diagnosis.early_risk["target_rows"] > risk_threshold:
            return {
                "question": question, "clue": clue.as_dict(), "condition": diagnosis.condition,
                "probe": {"query": diagnosis.query, "trace": diagnosis.trace, "early_risk": diagnosis.early_risk},
                "status": "rejected_by_early_risk_control", "actions": [],
                "sandbox_db_path": sandbox if keep_sandbox else None,
            }
        if not use_final_query:
            final_query = ""
            final_trace = {"mode": "final", "is_success": False, "error_message": "Final query disabled.",
                           "returned_rows": [], "total_row_count": 0, "execution_time_ms": 0.0,
                           "disabled": True}
        elif reuse_probe_query and diagnosis.query:
            final_query = diagnosis.query
            final_trace = UnifiedDQLExecutor(sandbox).execute(final_query, mode="final")
        else:
            final_prompt = f"Generate the final read-only SELECT query for the request. Return SQL only.\nRequest: {question}\nSchema:\n{get_db_schema(sandbox)}"
            final_query = _extract_sql(_invoke_llm(final_prompt, llm))
            final_trace = UnifiedDQLExecutor(sandbox).execute(final_query, mode="final")
        return {
            "question": question,
            "clue": clue.as_dict(),
            "condition": diagnosis.condition,
            "probe": {"query": diagnosis.query, "trace": diagnosis.trace, "early_risk": diagnosis.early_risk, "planning_error": diagnosis.planning_error},
            "actions": actions,
            "risk_control": risk,
            "execution": execution,
            "verification_passed": verification,
            "final_sql": final_query,
            "final_trace": final_trace,
            "sandbox_db_path": sandbox if keep_sandbox else None,
        }
    finally:
        if not keep_sandbox:
            materializer.cleanup()



