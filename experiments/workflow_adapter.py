"""Compatibility adapter between experiment entry points and the StateSculptor workflow."""

from __future__ import annotations

from typing import Any, Callable

from src.config import get_llm
from src.statesculptor.workflow import run_workflow

class PromptAwareLLM:
    """Inject complete shot templates into the shared workflow without rewriting them."""
    def __init__(self, base_llm: Any, prompt_loader: Callable[[str], str] | None = None) -> None:
        self.base_llm = base_llm
        self.prompt_loader = prompt_loader
    def invoke(self, prompt: Any) -> Any:
        if not isinstance(prompt, str) or self.prompt_loader is None:
            return self.base_llm.invoke(prompt)
        return self.base_llm.invoke(self.prompt_loader(_detect_phase(prompt)) + "\n\n--- Current StateSculptor Context ---\n" + prompt)

def _detect_phase(prompt: str) -> str:
    lower = prompt.lower()
    if "extract a non-executable" in lower or "normalize" in lower:
        return "router"
    if "ordered json action plan" in lower or "generated_statements" in lower:
        return "case3"
    return "case2"

def make_llm(provider: str | None = None, prompt_loader: Callable[[str], str] | None = None) -> Any:
    base = get_llm(provider)
    return PromptAwareLLM(base, prompt_loader) if prompt_loader else base

def execute_unified(state: dict[str, Any], *, provider: str | None = None,
                    prompt_loader: Callable[[str], str] | None = None,
                    **workflow_options: Any) -> dict[str, Any]:
    """Map legacy experiment fields to the shared workflow and retain evaluation aliases."""
    result = run_workflow(str(state.get("question", "")), str(state.get("db_path", "")),
                          str(state.get("recovery_json_path", "") or ""),
                          llm=make_llm(provider, prompt_loader), **workflow_options)
    final_trace = result.get("final_trace", {})
    return {**result,
            "workflow_type": {"MT": "CASE_1_MISSING_TABLE", "IS": "CASE_2_NORMAL",
                              "DU": "CASE_3_CORRUPTED_DATA", "MC": "CASE_4_MISSING_COLUMN"}.get(
                                  result.get("condition"), result.get("condition", "")),
            "final_answer": final_trace.get("returned_rows", []),
            "final_query_result": final_trace.get("returned_rows", []),
            "repair_sql": ";\n".join(result.get("actions", []))}

class WorkflowApplication:
    """Provide the legacy app.stream(initial_state) interface over the shared workflow."""
    def __init__(self, *, variant: str = "full", provider: str | None = None,
                 prompt_loader: Callable[[str], str] | None = None) -> None:
        self.variant = variant
        self.provider = provider
        self.prompt_loader = prompt_loader
    def stream(self, state: dict[str, Any]):
        options: dict[str, Any] = {}
        if self.variant == "no_probe":
            options["use_probe"] = False
        elif self.variant == "no_final_query":
            options["use_final_query"] = False
        elif self.variant == "reuse_probe_query":
            options["reuse_probe_query"] = True
        yield {"statesculptor": execute_unified(state, provider=self.provider,
                                                 prompt_loader=self.prompt_loader, **options)}

def load_prompt_loader(style: str) -> Callable[[str], str]:
    """Return a loader for the complete original zero/one/few-shot templates."""
    from experiments.prompting.prompts_templates import get_prompt_template
    mode = style.lower().replace("-", "_")
    def loader(phase: str) -> str:
        case = {"router": "router", "case2": "case2", "case3": "case3"}.get(phase, "case2")
        return get_prompt_template(mode, case)
    return loader
