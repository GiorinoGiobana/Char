"""Additional ablation: reuse the probe query after adjustment."""
from experiments.workflow_adapter import WorkflowApplication
app = WorkflowApplication(variant="reuse_probe_query")
