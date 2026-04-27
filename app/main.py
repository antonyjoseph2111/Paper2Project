from fastapi import FastAPI

from app.api.routes.jobs import router as jobs_router
from app.core.logging import configure_logging


configure_logging()

app = FastAPI(
    title="Paper2Project",
    version="0.1.0",
    description="Multi-agent system for converting ML papers into reproducible PyTorch projects.",
)

app.include_router(jobs_router, prefix="/jobs", tags=["jobs"])


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}
