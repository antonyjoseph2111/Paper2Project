from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes.jobs import router as jobs_router
from app.api.routes.ui import router as ui_router
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

app.include_router(ui_router)
app.include_router(jobs_router, prefix="/jobs", tags=["jobs"])
web_dir = Path(__file__).resolve().parent / "web"
if web_dir.exists():
    app.mount("/web", StaticFiles(directory=web_dir), name="web")


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}
