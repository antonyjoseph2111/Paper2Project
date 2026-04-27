from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.api.deps import verify_api_key
from app.models.schemas import AgentTurn, ArtifactManifest, DecisionUpdateRequest, JobRecord
from app.orchestration.workflow import WORKFLOW

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.get("", response_model=list[JobRecord])
def list_jobs(limit: int = 25) -> list[JobRecord]:
    return WORKFLOW.list_jobs(limit=limit)


@router.post("", response_model=JobRecord)
async def create_job(file: UploadFile = File(...)) -> JobRecord:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="A PDF upload is required.")
    content = await file.read()
    return WORKFLOW.enqueue_job(file.filename, content)


@router.get("/{job_id}", response_model=JobRecord)
def get_job(job_id: str) -> JobRecord:
    return WORKFLOW.get_job_or_404(job_id)


@router.get("/{job_id}/trace", response_model=list[AgentTurn])
def get_job_trace(job_id: str) -> list[AgentTurn]:
    job = WORKFLOW.get_job_or_404(job_id)
    return job.agent_memory


@router.get("/{job_id}/decision")
def get_decision(job_id: str) -> dict:
    job = WORKFLOW.get_job_or_404(job_id)
    if not job.decision_config:
        raise HTTPException(status_code=409, detail="Decision config not available yet.")
    return job.decision_config.model_dump()


@router.patch("/{job_id}/decision", response_model=JobRecord)
def update_decision(job_id: str, request: DecisionUpdateRequest) -> JobRecord:
    return WORKFLOW.update_decision(job_id, request)


@router.post("/{job_id}/approve", response_model=JobRecord)
def approve_job(job_id: str) -> JobRecord:
    return WORKFLOW.enqueue_generation(job_id)


@router.get("/{job_id}/artifacts", response_model=ArtifactManifest)
def get_artifacts(job_id: str) -> ArtifactManifest:
    job = WORKFLOW.get_job_or_404(job_id)
    if not job.artifacts:
        raise HTTPException(status_code=409, detail="Artifacts not available yet.")
    return job.artifacts


@router.get("/{job_id}/artifacts/download")
def download_artifacts(job_id: str) -> FileResponse:
    job = WORKFLOW.get_job_or_404(job_id)
    if not job.artifacts or not job.artifacts.archive_file:
        raise HTTPException(status_code=409, detail="Artifact archive not available yet.")
    return FileResponse(job.artifacts.archive_file, filename=f"{job_id}_artifacts.zip", media_type="application/zip")
