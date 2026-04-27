from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from app.models.schemas import ProviderSpec


class Settings(BaseSettings):
    app_name: str = "Paper2Project"
    artifact_root: Path = Path("artifacts")
    state_root: Path = Path("state")
    allowed_origins: Annotated[list[str], NoDecode] = ["*"]
    api_keys: Annotated[list[str], NoDecode] = []
    require_api_key: bool = False
    llm_roster: Annotated[list[str], NoDecode] = [
        "openai:gpt-4.1-mini",
        "anthropic:claude-3-5-sonnet-latest",
        "google:gemini-2.5-flash",
    ]
    llm_strategy: str = "fallback_chain"
    llm_timeout_seconds: float = 90.0
    llm_max_retries: int = 2
    llm_retry_backoff_seconds: float = 1.5
    llm_anthropic_max_tokens: int = 6000
    max_section_chunk_chars: int = 5000
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    openrouter_api_key: str = ""
    deepseek_api_key: str = ""
    groq_api_key: str = ""
    together_api_key: str = ""
    xai_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    llm_openai_base_url: str = "https://api.openai.com/v1"
    llm_openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_deepseek_base_url: str = "https://api.deepseek.com/v1"
    llm_groq_base_url: str = "https://api.groq.com/openai/v1"
    llm_together_base_url: str = "https://api.together.xyz/v1"
    llm_xai_base_url: str = "https://api.x.ai/v1"
    llm_ollama_chat_path: str = "/api/chat"
    llm_google_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    llm_anthropic_base_url: str = "https://api.anthropic.com/v1"
    grobid_url: str = ""
    arxiv_source_enabled: bool = True
    job_worker_threads: int = 4
    model_config = SettingsConfigDict(env_prefix="P2P_", extra="ignore", env_file=".env", env_file_encoding="utf-8")

    @field_validator("api_keys", "allowed_origins", "llm_roster", mode="before")
    @classmethod
    def split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                import json

                return json.loads(stripped)
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    def parsed_roster(self) -> list[ProviderSpec]:
        providers: list[ProviderSpec] = []
        for item in self.llm_roster:
            if ":" not in item:
                continue
            provider, model = item.split(":", 1)
            providers.append(ProviderSpec(provider=provider.strip(), model=model.strip()))
        return providers


settings = Settings()
settings.artifact_root.mkdir(parents=True, exist_ok=True)
settings.state_root.mkdir(parents=True, exist_ok=True)
