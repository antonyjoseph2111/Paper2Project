from pathlib import Path

from app.agents.base import Agent
from app.models.schemas import ArtifactManifest, DecisionConfig, JobRecord
from app.services.code_generator import build_generated_project


class CodeGeneratorAgent(Agent):
    name = "code_generator"

    def run(self, job: JobRecord, output_dir: Path, decision_config: DecisionConfig) -> ArtifactManifest:
        self.memory.append(
            job,
            self.name,
            "internal",
            "template-generator",
            "system",
            self.load_prompt(),
        )
        files = build_generated_project(job, output_dir, decision_config)
        return ArtifactManifest(output_dir=str(output_dir), files=files, notebook_file="")
