"""Ablation: remove post-adjustment DQL execution."""
from experiments.workflow_adapter import WorkflowApplication
app = WorkflowApplication(variant="no_final_query")
