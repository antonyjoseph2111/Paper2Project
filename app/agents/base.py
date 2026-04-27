from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class Agent(ABC):
    name: str

    @property
    def prompt_path(self) -> Path:
        return Path("app") / "prompts" / f"{self.name}.md"

    def load_prompt(self) -> str:
        return self.prompt_path.read_text(encoding="utf-8")

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError
