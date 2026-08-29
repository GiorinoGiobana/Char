"""Workflow ablations reported in the paper.

Each variant changes only one StateSculptor stage; all other logic remains shared.
"""
from __future__ import annotations
from typing import Any
from experiments.workflow_adapter import WorkflowApplication

def without_probe() -> WorkflowApplication:
    return WorkflowApplication(variant="no_probe")

def without_final_query() -> WorkflowApplication:
    return WorkflowApplication(variant="no_final_query")

def reuse_probe_query() -> WorkflowApplication:
    return WorkflowApplication(variant="reuse_probe_query")

def run(state: dict[str, Any], variant: str = "full") -> dict[str, Any]:
    app = WorkflowApplication(variant=variant)
    return next(iter(app.stream(state)))["statesculptor"]
