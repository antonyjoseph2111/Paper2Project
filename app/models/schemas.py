from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    PARSED = "parsed"
    ANALYZED = "analyzed"
    PLANNED = "planned"
    AWAITING_APPROVAL = "awaiting_approval"
    GENERATING = "generating"
    GENERATED = "generated"
    PARTIAL = "partial"
    FAILED = "failed"


class SectionChunk(BaseModel):
    name: str
    content: str
    chunk_id: str
    source_page_start: int | None = None
    source_page_end: int | None = None


class EvidenceField(BaseModel):
    value: Any
    confidence: float = 0.0
    source_section: str = ""
    assumed: bool = False


class ParsedPaper(BaseModel):
    title: str = ""
    problem: str = ""
    abstract: str = ""
    introduction: str = ""
    methodology_text: str = ""
    model_description: str = ""
    equations: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    figures: list[str] = Field(default_factory=list)
    sections: list[SectionChunk] = Field(default_factory=list)
    chunk_count: int = 0
    source_kind: str = "pdf"


class TrainingDetails(BaseModel):
    optimizer: str = "adamw"
    scheduler: str = "linear"
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    max_length: int = 128


class PaperAnalysis(BaseModel):
    task: EvidenceField = Field(default_factory=lambda: EvidenceField(value="unknown"))
    domain: EvidenceField = Field(default_factory=lambda: EvidenceField(value="unknown"))
    input_data_type: EvidenceField = Field(default_factory=lambda: EvidenceField(value="unknown"))
    output_format: EvidenceField = Field(default_factory=lambda: EvidenceField(value="unknown"))
    model_type: EvidenceField = Field(default_factory=lambda: EvidenceField(value="unknown"))
    components: EvidenceField = Field(default_factory=lambda: EvidenceField(value=[]))
    loss: EvidenceField = Field(default_factory=lambda: EvidenceField(value="unknown"))
    metrics: EvidenceField = Field(default_factory=lambda: EvidenceField(value=[]))
    training_details: TrainingDetails = Field(default_factory=TrainingDetails)
    ambiguities: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class DatasetRequirements(BaseModel):
    source: str = "synthetic"
    candidate_datasets: list[str] = Field(default_factory=list)
    split_strategy: str = "train/validation/test"
    notes: str = ""
    synthetic_fallback: str = ""


class ModelStructure(BaseModel):
    backbone: str = ""
    head: str = ""
    notes: str = ""
    reference_components: list[str] = Field(default_factory=list)


class PipelinePlan(BaseModel):
    steps: list[str] = Field(default_factory=list)
    dataset_requirements: DatasetRequirements = Field(default_factory=DatasetRequirements)
    model_structure: ModelStructure = Field(default_factory=ModelStructure)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    implementation_notes: list[str] = Field(default_factory=list)


class DatasetDecision(BaseModel):
    selected: str
    source: str = "synthetic"
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
    optimizer: str = "adamw"
    scheduler: str = "linear"
    loss: str = "cross_entropy"
    weight_decay: float = 0.0
    max_length: int = 128
    use_tensorboard: bool = True
    use_wandb: bool = False


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
    archive_file: str = ""


class AgentTurn(BaseModel):
    stage: str
    provider: str
    model: str
    role: str
    content: str
    created_at: datetime = Field(default_factory=utc_now)


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
    agent_memory: list[AgentTurn] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class DecisionUpdateRequest(BaseModel):
    dataset_selected: str | None = None
    dataset_source: str | None = None
    model_selected: str | None = None
    epochs: int | None = None
    batch_size: int | None = None
    learning_rate: float | None = None
    seed: int | None = None
    optimizer: str | None = None
    scheduler: str | None = None
    loss: str | None = None
    weight_decay: float | None = None
    max_length: int | None = None
    use_tensorboard: bool | None = None
    use_wandb: bool | None = None


class ProviderSpec(BaseModel):
    provider: str
    model: str


class LLMResponse(BaseModel):
    provider: str
    model: str
    content: str
    raw_payload: dict[str, Any] = Field(default_factory=dict)
