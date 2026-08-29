"""Ablation: remove the pre-adjustment DQL probe."""
from experiments.workflow_adapter import WorkflowApplication
app = WorkflowApplication(variant="no_probe")
