from __future__ import annotations

import json
from pathlib import Path

from app.core.config import settings
from app.models.schemas import JobRecord


class JobStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or settings.state_root
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, job_id: str) -> Path:
        return self.root / f"{job_id}.json"

    def save(self, job: JobRecord) -> None:
        self.path_for(job.job_id).write_text(job.model_dump_json(indent=2), encoding="utf-8")

    def load(self, job_id: str) -> JobRecord | None:
        path = self.path_for(job_id)
        if not path.exists():
            return None
        return JobRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def load_all(self) -> dict[str, JobRecord]:
        jobs: dict[str, JobRecord] = {}
        for path in self.root.glob("*.json"):
            try:
                job = JobRecord.model_validate_json(path.read_text(encoding="utf-8"))
                jobs[job.job_id] = job
            except Exception:
                continue
        return jobs
