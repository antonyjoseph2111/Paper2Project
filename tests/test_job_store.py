from pathlib import Path

from app.models.schemas import JobRecord, JobStatus
from app.services.job_store import JobStore


def test_job_store_roundtrip(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path)
    job = JobRecord(job_id="job-1", filename="paper.pdf", status=JobStatus.QUEUED)
    store.save(job)
    loaded = store.load("job-1")
    assert loaded is not None
    assert loaded.job_id == "job-1"
    assert loaded.filename == "paper.pdf"
