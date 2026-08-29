from typing import TypedDict, Literal, Optional, Any
from enum import Enum


class StateCondition(str, Enum):
    DATA_UPDATE = "DU"
    MISSING_COLUMN = "MC"
    MISSING_TABLE = "MT"
    INTACT_STATE = "IS"


class ParsedEvidence(TypedDict):
    target_table: Optional[str]
    target_column: Optional[str]
    candidate_types: Optional[str]
    evidence_records: list[dict]


class DiagnosticSummary(TypedDict):
    state_condition_hypothesis: StateCondition
    diagnostic_summary: str
    parsed_evidence: ParsedEvidence


class ProbeResult(TypedDict):
    tentative_query: str
    probe_execution_trace: dict


class AdjustmentPlan(TypedDict):
    adjustment_performed: bool
    generated_statements: list[str]
    plan_reasoning: str
    is_noop: bool


class VerificationResult(TypedDict):
    verification_label: bool
    verification_details: str
    affected_rows: int
    schema_check_passed: bool


class DQLExecutionResult(TypedDict):
    final_query: str
    final_answer: str
    execution_success: bool
    result_rows: list[Any]


class StateSculptorState(TypedDict):
    question: str
    db_path: str
    recovery_json_path: str
    has_recovery_json: bool

    diagnostic_summary: DiagnosticSummary

    probe_result: ProbeResult

    adjustment_plan: AdjustmentPlan
    verification_result: VerificationResult

    dql_execution_result: DQLExecutionResult

    final_answer: str
    final_sql: str
    repair_sql: str
