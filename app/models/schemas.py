from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(str, Enum):
    PARSED = "parsed"
    ANALYZED = "analyzed"
    PLANNED = "planned"
    AWAITING_APPROVAL = "awaiting_approval"
    GENERATED = "generated"
    FAILED = "failed"


class SectionChunk(BaseModel):
    name: str
    content: str


class ParsedPaper(BaseModel):
    title: str = ""
    problem: str = ""
    abstract: str = ""
    introduction: str = ""
    methodology_text: str = ""
    model_description: str = ""
    equations: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    sections: list[SectionChunk] = Field(default_factory=list)


class TrainingDetails(BaseModel):
    optimizer: str = "adamw"
    scheduler: str = "linear"
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 2e-5


class PaperAnalysis(BaseModel):
    task: str = "unknown"
    domain: str = "unknown"
    input_data_type: str = "unknown"
    output_format: str = "unknown"
    model_type: str = "unknown"
    components: list[str] = Field(default_factory=list)
    loss: str = "unknown"
    metrics: list[str] = Field(default_factory=list)
    training_details: TrainingDetails = Field(default_factory=TrainingDetails)
    ambiguities: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class DatasetRequirements(BaseModel):
    source: str = "huggingface"
    candidate_datasets: list[str] = Field(default_factory=list)
    split_strategy: str = "train/validation/test"
    notes: str = ""


class ModelStructure(BaseModel):
    backbone: str = ""
    head: str = ""
    notes: str = ""


class PipelinePlan(BaseModel):
    steps: list[str] = Field(default_factory=list)
    dataset_requirements: DatasetRequirements = Field(default_factory=DatasetRequirements)
    model_structure: ModelStructure = Field(default_factory=ModelStructure)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class DatasetDecision(BaseModel):
    selected: str
    alternatives: list[str] = Field(default_factory=list)
    editable: bool = True
    reason: str = ""


class ModelDecision(BaseModel):
    selected: str
    reason: str = ""
    editable: bool = True


class TrainingDecision(BaseModel):
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 2e-5
    seed: int = 42


class DecisionConfig(BaseModel):
    dataset: DatasetDecision
    model: ModelDecision
    training: TrainingDecision = Field(default_factory=TrainingDecision)
    approval_required: bool = True
    assumptions: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)


class ArtifactManifest(BaseModel):
    output_dir: str = ""
    files: list[str] = Field(default_factory=list)
    notebook_file: str = ""


class JobRecord(BaseModel):
    job_id: str
    filename: str
    status: JobStatus
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    parsed_paper: ParsedPaper | None = None
    analysis: PaperAnalysis | None = None
    plan: PipelinePlan | None = None
    decision_config: DecisionConfig | None = None
    artifacts: ArtifactManifest | None = None
    errors: list[str] = Field(default_factory=list)


class DecisionUpdateRequest(BaseModel):
    dataset_selected: str | None = None
    model_selected: str | None = None
    epochs: int | None = None
    batch_size: int | None = None
    learning_rate: float | None = None
    seed: int | None = None
