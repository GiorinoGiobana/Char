"""Run the shared StateSculptor workflow with the DeepSeek backbone."""
from experiments.backbones.agents.agent_deepseek import app

__all__ = ["app"]

if __name__ == "__main__":
    from experiments.backbones.run_full import main
    raise SystemExit(main())
