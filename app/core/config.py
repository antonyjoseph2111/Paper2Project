from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Paper2Project"
    artifact_root: Path = Path("artifacts")
    llm_provider: str = "stub"
    llm_model: str = "gpt-4.1-mini"
    model_config = SettingsConfigDict(env_prefix="P2P_", extra="ignore")


settings = Settings()
settings.artifact_root.mkdir(parents=True, exist_ok=True)
