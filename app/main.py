from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.jobs import router as jobs_router
from app.core.config import settings
from app.core.logging import configure_logging


configure_logging()

app = FastAPI(
    title="Paper2Project",
    version="0.2.0",
    description="Multi-agent system for converting ML papers into reproducible PyTorch projects.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs_router, prefix="/jobs", tags=["jobs"])


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}
