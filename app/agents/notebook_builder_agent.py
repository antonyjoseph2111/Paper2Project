from pathlib import Path

from app.agents.base import Agent
from app.models.schemas import DecisionConfig, JobRecord
from app.services.notebook_builder import build_colab_notebook


class NotebookBuilderAgent(Agent):
    name = "notebook_builder"

    def run(self, job: JobRecord, output_dir: Path, decision_config: DecisionConfig) -> str:
        return build_colab_notebook(job, output_dir, decision_config)
