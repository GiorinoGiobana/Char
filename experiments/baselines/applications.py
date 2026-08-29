"""Independent implementations of Baseline A and Baseline B."""
from __future__ import annotations
import json
from typing import Any
from src.statesculptor.skills import AdjustmentExecutor, SandboxMaterializer, UnifiedDQLExecutor, _extract_json, _extract_sql
from src.statesculptor.workflow import build_clue, _invoke_llm
from src.tools import get_db_schema

class BaselineApplication:
    """Keep the distinct baseline constraints while reusing shared database execution."""
    def __init__(self, variant: str) -> None:
        self.variant = variant

    def stream(self, state: dict[str, Any]):
        if self.variant == "query_only":
            yield {"baseline": self._query_only(state)}
        elif self.variant == "monolithic":
            yield {"baseline": self._monolithic(state)}
        else:
            raise ValueError(f"Unknown baseline variant: {self.variant}")

    @staticmethod
    def _query_only(state: dict[str, Any]) -> dict[str, Any]:
        from experiments.baselines.prompts import BASELINE_A_CASE1_PROMPT, BASELINE_A_CASE2_PROMPT, BASELINE_A_CASE3_PROMPT, BASELINE_A_CASE4_PROMPT
        materializer = SandboxMaterializer(str(state["db_path"]))
        sandbox = materializer.materialize()
        try:
            clue = build_clue(str(state["question"]), sandbox, str(state.get("recovery_json_path", "")), None)
            prompts = {"MT": BASELINE_A_CASE1_PROMPT, "IS": BASELINE_A_CASE2_PROMPT, "DU": BASELINE_A_CASE3_PROMPT, "MC": BASELINE_A_CASE4_PROMPT}
            request = prompts.get(clue.case_type, BASELINE_A_CASE2_PROMPT) + "\n\nCurrent request:\n" + str(state["question"]) + "\nSchema:\n" + get_db_schema(sandbox) + "\nClue JSON:\n" + json.dumps(clue.as_dict(), ensure_ascii=False)
            query = _extract_sql(_invoke_llm(request, None))
            trace = UnifiedDQLExecutor(sandbox).execute(query, mode="final")
            return {"mode": "query_only", "clue": clue.as_dict(), "final_sql": query, "final_trace": trace, "actions": [], "verification_passed": True}
        finally:
            materializer.cleanup()

    @staticmethod
    def _monolithic(state: dict[str, Any]) -> dict[str, Any]:
        from experiments.baselines.monolithic_prompts import BASELINE_B_MONOLITHIC_PROMPT
        materializer = SandboxMaterializer(str(state["db_path"]))
        sandbox = materializer.materialize()
        try:
            clue = build_clue(str(state["question"]), sandbox, str(state.get("recovery_json_path", "")), None)
            request = BASELINE_B_MONOLITHIC_PROMPT + "\n\nCurrent request:\n" + str(state["question"]) + "\nSchema:\n" + get_db_schema(sandbox) + "\nClue JSON:\n" + json.dumps(clue.as_dict(), ensure_ascii=False)
            data = _extract_json(_invoke_llm(request, None))
            actions = [str(value).strip() for value in data.get("generated_statements", []) if str(value).strip()]
            execution = AdjustmentExecutor(sandbox).execute_ordered_plan(actions) if actions else {"all_success": True, "execution_log": []}
            query = _extract_sql(str(data.get("final_sql", "")))
            trace = UnifiedDQLExecutor(sandbox).execute(query, mode="final") if query else {"is_success": False, "error_message": "No final SQL", "returned_rows": []}
            return {"mode": "monolithic", "clue": clue.as_dict(), "actions": actions, "execution": execution, "final_sql": query, "final_trace": trace, "verification_passed": bool(execution.get("all_success", False))}
        finally:
            materializer.cleanup()
