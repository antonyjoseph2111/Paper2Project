from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.schemas import DecisionUpdateRequest, JobRecord
from app.orchestration.workflow import WORKFLOW

router = APIRouter()


@router.post("", response_model=JobRecord)
async def create_job(file: UploadFile = File(...)) -> JobRecord:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="A PDF upload is required.")
    content = await file.read()
    return WORKFLOW.create_job(file.filename, content)


@router.get("/{job_id}", response_model=JobRecord)
def get_job(job_id: str) -> JobRecord:
    job = WORKFLOW.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@router.get("/{job_id}/decision")
def get_decision(job_id: str) -> dict:
    job = WORKFLOW.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if not job.decision_config:
        raise HTTPException(status_code=409, detail="Decision config not available yet.")
    return job.decision_config.model_dump()


@router.patch("/{job_id}/decision", response_model=JobRecord)
def update_decision(job_id: str, request: DecisionUpdateRequest) -> JobRecord:
    return WORKFLOW.update_decision(job_id, request)


@router.post("/{job_id}/approve", response_model=JobRecord)
def approve_job(job_id: str) -> JobRecord:
    return WORKFLOW.approve_and_generate(job_id)
