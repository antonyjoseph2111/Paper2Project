from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException

from app.agents.code_generator_agent import CodeGeneratorAgent
from app.agents.decision import DecisionAgent
from app.agents.notebook_builder_agent import NotebookBuilderAgent
from app.agents.paper_analyst import PaperAnalystAgent
from app.agents.planner import PlannerAgent
from app.core.config import settings
from app.models.schemas import DecisionUpdateRequest, JobRecord, JobStatus, utc_now
from app.services.pdf_parser import parse_pdf

logger = logging.getLogger(__name__)


class Paper2ProjectWorkflow:
    def __init__(self) -> None:
        self.jobs: dict[str, JobRecord] = {}
        self.paper_analyst = PaperAnalystAgent()
        self.planner = PlannerAgent()
        self.decision_agent = DecisionAgent()
        self.code_generator = CodeGeneratorAgent()
        self.notebook_builder = NotebookBuilderAgent()

    def create_job(self, filename: str, pdf_bytes: bytes) -> JobRecord:
        job_id = str(uuid4())
        job_dir = settings.artifact_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = job_dir / filename
        pdf_path.write_bytes(pdf_bytes)
        parsed = parse_pdf(pdf_path)
        analysis = self.paper_analyst.run(parsed)
        plan = self.planner.run(analysis)
        decision = self.decision_agent.run(plan)
        job = JobRecord(
            job_id=job_id,
            filename=filename,
            status=JobStatus.AWAITING_APPROVAL,
            parsed_paper=parsed,
            analysis=analysis,
            plan=plan,
            decision_config=decision,
        )
        self.jobs[job_id] = job
        logger.info("Created Paper2Project job %s", job_id)
        return job

    def get_job(self, job_id: str) -> JobRecord | None:
        return self.jobs.get(job_id)

    def update_decision(self, job_id: str, request: DecisionUpdateRequest) -> JobRecord:
        job = self.get_job_or_404(job_id)
        decision = job.decision_config
        if not decision:
            raise HTTPException(status_code=409, detail="Decision config not initialized.")

        if request.dataset_selected:
            decision.dataset.selected = request.dataset_selected
        if request.model_selected:
            decision.model.selected = request.model_selected
        if request.epochs is not None:
            decision.training.epochs = request.epochs
        if request.batch_size is not None:
            decision.training.batch_size = request.batch_size
        if request.learning_rate is not None:
            decision.training.learning_rate = request.learning_rate
        if request.seed is not None:
            decision.training.seed = request.seed

        job.updated_at = utc_now()
        self.jobs[job_id] = job
        return job

    def approve_and_generate(self, job_id: str) -> JobRecord:
        job = self.get_job_or_404(job_id)
        if not job.decision_config:
            raise HTTPException(status_code=409, detail="Decision config is required.")
        output_dir = settings.artifact_root / job_id / "generated_project"
        manifest = self.code_generator.run(job, output_dir, job.decision_config)
        notebook_path = self.notebook_builder.run(job, output_dir, job.decision_config)
        manifest.notebook_file = notebook_path
        manifest.files.append(notebook_path)
        job.artifacts = manifest
        job.status = JobStatus.GENERATED
        job.updated_at = utc_now()
        self.jobs[job_id] = job
        logger.info("Generated artifacts for job %s", job_id)
        return job

    def get_job_or_404(self, job_id: str) -> JobRecord:
        job = self.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
        return job


WORKFLOW = Paper2ProjectWorkflow()
