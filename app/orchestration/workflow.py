from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from shutil import make_archive
from uuid import uuid4

from fastapi import HTTPException

from app.agents.code_generator_agent import CodeGeneratorAgent
from app.agents.decision import DecisionAgent
from app.agents.notebook_builder_agent import NotebookBuilderAgent
from app.agents.paper_analyst import PaperAnalystAgent
from app.agents.planner import PlannerAgent
from app.core.config import settings
from app.models.schemas import (
    ArtifactManifest,
    DecisionUpdateRequest,
    JobRecord,
    JobStatus,
    utc_now,
)
from app.services.job_store import JobStore
from app.services.pdf_parser import parse_pdf
from app.services.source_enrichment import extract_arxiv_source, maybe_download_arxiv_source, maybe_fetch_grobid

logger = logging.getLogger(__name__)


class Paper2ProjectWorkflow:
    def __init__(self) -> None:
        self.store = JobStore()
        self.jobs: dict[str, JobRecord] = self.store.load_all()
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=settings.job_worker_threads)
        self.paper_analyst = PaperAnalystAgent()
        self.planner = PlannerAgent()
        self.decision_agent = DecisionAgent()
        self.code_generator = CodeGeneratorAgent()
        self.notebook_builder = NotebookBuilderAgent()

    def enqueue_job(self, filename: str, pdf_bytes: bytes) -> JobRecord:
        job_id = str(uuid4())
        job_dir = settings.artifact_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = job_dir / filename
        pdf_path.write_bytes(pdf_bytes)
        job = JobRecord(job_id=job_id, filename=filename, status=JobStatus.QUEUED)
        self._save_job(job)
        self.executor.submit(self._process_job, job_id, pdf_path)
        return job

    def _process_job(self, job_id: str, pdf_path: Path) -> None:
        job = self.get_job_or_404(job_id)
        job.status = JobStatus.PROCESSING
        self._save_job(job)
        arxiv_source_dir: Path | None = None
        grobid_tei_path: Path | None = None
        try:
            archive_path = maybe_download_arxiv_source(pdf_path)
            if archive_path:
                arxiv_source_dir = extract_arxiv_source(archive_path)
        except Exception as exc:
            job.errors.append(f"ArXiv source enrichment skipped: {exc}")
            self._save_job(job)
        try:
            grobid_tei_path = maybe_fetch_grobid(pdf_path)
        except Exception as exc:
            job.errors.append(f"Grobid enrichment skipped: {exc}")
            self._save_job(job)
        try:
            parsed = parse_pdf(pdf_path, grobid_tei_path=grobid_tei_path, arxiv_source_dir=arxiv_source_dir)
            job.parsed_paper = parsed
            job.status = JobStatus.PARSED
            self._save_job(job)
        except Exception as exc:
            self._record_error(job, f"PDF parsing failed: {exc}", JobStatus.FAILED)
            return

        try:
            analysis = self.paper_analyst.run(job, job.parsed_paper)
            job.analysis = analysis
            job.status = JobStatus.ANALYZED
            self._save_job(job)
        except Exception as exc:
            self._record_error(job, f"Paper analysis failed: {exc}", JobStatus.PARTIAL)

        try:
            if job.analysis:
                plan = self.planner.run(job, job.analysis)
                job.plan = plan
                job.status = JobStatus.PLANNED
                self._save_job(job)
        except Exception as exc:
            self._record_error(job, f"Planning failed: {exc}", JobStatus.PARTIAL)

        try:
            if job.plan:
                decision = self.decision_agent.run(job, job.plan)
                job.decision_config = decision
                job.status = JobStatus.AWAITING_APPROVAL
            else:
                job.status = JobStatus.PARTIAL
            self._save_job(job)
        except Exception as exc:
            self._record_error(job, f"Decision config generation failed: {exc}", JobStatus.PARTIAL)

    def enqueue_generation(self, job_id: str) -> JobRecord:
        job = self.get_job_or_404(job_id)
        if not job.decision_config:
            raise HTTPException(status_code=409, detail="Decision config is required before generation.")
        job.status = JobStatus.GENERATING
        self._save_job(job)
        self.executor.submit(self._generate_job, job_id)
        return job

    def _generate_job(self, job_id: str) -> None:
        job = self.get_job_or_404(job_id)
        if not job.decision_config:
            self._record_error(job, "No decision config available for generation.", JobStatus.FAILED)
            return
        output_dir = settings.artifact_root / job_id / "generated_project"
        try:
            manifest = self.code_generator.run(job, output_dir, job.decision_config)
            notebook_path = self.notebook_builder.run(job, output_dir, job.decision_config)
            manifest.notebook_file = notebook_path
            if notebook_path not in manifest.files:
                manifest.files.append(notebook_path)
            archive_base = str(output_dir.parent / "artifacts_bundle")
            manifest.archive_file = make_archive(archive_base, "zip", root_dir=output_dir)
            job.artifacts = manifest
            job.status = JobStatus.GENERATED
            self._save_job(job)
            logger.info("Generated artifacts for job %s", job_id)
        except Exception as exc:
            partial_manifest = ArtifactManifest(output_dir=str(output_dir), files=self._existing_files(output_dir))
            job.artifacts = partial_manifest
            self._record_error(job, f"Artifact generation failed: {exc}", JobStatus.PARTIAL)

    def get_job(self, job_id: str) -> JobRecord | None:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is not None:
                return deepcopy(job)
        stored = self.store.load(job_id)
        return deepcopy(stored) if stored is not None else None

    def get_job_or_404(self, job_id: str) -> JobRecord:
        job = self.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
        return job

    def update_decision(self, job_id: str, request: DecisionUpdateRequest) -> JobRecord:
        job = self.get_job_or_404(job_id)
        decision = job.decision_config
        if not decision:
            raise HTTPException(status_code=409, detail="Decision config not initialized.")

        if request.dataset_selected:
            decision.dataset.selected = request.dataset_selected
        if request.dataset_source:
            decision.dataset.source = request.dataset_source
        if request.model_selected:
            decision.model.selected = request.model_selected
        for field in [
            "epochs",
            "batch_size",
            "learning_rate",
            "seed",
            "optimizer",
            "scheduler",
            "loss",
            "weight_decay",
            "max_length",
            "use_tensorboard",
            "use_wandb",
        ]:
            value = getattr(request, field)
            if value is not None:
                setattr(decision.training, field, value)

        job.updated_at = utc_now()
        self._save_job(job)
        return job

    def _save_job(self, job: JobRecord) -> None:
        job.updated_at = utc_now()
        with self.lock:
            self.jobs[job.job_id] = deepcopy(job)
        self.store.save(job)

    def _record_error(self, job: JobRecord, message: str, status: JobStatus) -> None:
        logger.error(message)
        job.errors.append(message)
        job.status = status
        self._save_job(job)

    def _existing_files(self, output_dir: Path) -> list[str]:
        if not output_dir.exists():
            return []
        return [str(path) for path in output_dir.rglob("*") if path.is_file()]


WORKFLOW = Paper2ProjectWorkflow()
