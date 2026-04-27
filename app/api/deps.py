from __future__ import annotations

from fastapi import Header, HTTPException

from app.core.config import settings


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not settings.require_api_key:
        return
    if not x_api_key or x_api_key not in settings.api_keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
