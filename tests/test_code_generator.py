from pathlib import Path

import yaml

from app.models.schemas import (
    DatasetDecision,
    DecisionConfig,
    EvidenceField,
    JobRecord,
    ModelDecision,
    JobStatus,
    PaperAnalysis,
    ParsedPaper,
    TrainingDecision,
)
from app.services.code_generator import build_generated_project


def test_generated_project_reads_config_shape(tmp_path: Path) -> None:
    job = JobRecord(
        job_id="job-2",
        filename="paper.pdf",
        status=JobStatus.AWAITING_APPROVAL,
        parsed_paper=ParsedPaper(title="Paper"),
        analysis=PaperAnalysis(
            task=EvidenceField(value="classification"),
            domain=EvidenceField(value="NLP"),
            input_data_type=EvidenceField(value="text"),
            output_format=EvidenceField(value="label"),
            model_type=EvidenceField(value="transformer"),
            components=EvidenceField(value=["embedding"]),
            loss=EvidenceField(value="cross_entropy"),
            metrics=EvidenceField(value=["accuracy"]),
        ),
    )
    decision = DecisionConfig(
        dataset=DatasetDecision(selected="ag_news", source="huggingface"),
        model=ModelDecision(selected="distilbert-base-uncased", reason="baseline", editable=True),
        training=TrainingDecision(),
    )
    files = build_generated_project(job, tmp_path, decision)
    assert any(path.endswith("train.py") for path in files)
    config = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert config["dataset"]["selected"] == "ag_news"
