from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.models.schemas import JobRecord
from app.services.llm_client import MultiProviderLLMClient
from app.services.llm_memory import SharedAgentMemory

logger = logging.getLogger(__name__)


class Agent(ABC):
    name: str

    def __init__(self) -> None:
        self.llm = MultiProviderLLMClient()
        self.memory = SharedAgentMemory()

    @property
    def prompt_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / "prompts" / f"{self.name}.md"

    def load_prompt(self) -> str:
        return self.prompt_path.read_text(encoding="utf-8")

    def _extract_json_object(self, text: str) -> dict[str, Any] | None:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except json.JSONDecodeError:
                pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                return None
        return None

    def run_llm_json(self, job: JobRecord, payload: dict[str, Any]) -> dict[str, Any] | None:
        if not self.llm.has_any_provider():
            return None
        prompt = self.load_prompt()
        self.memory.append(job, self.name, "internal", "prompt-template", "system", prompt)
        self.memory.append(
            job,
            self.name,
            "internal",
            "structured-payload",
            "user",
            json.dumps(payload, ensure_ascii=False, indent=2)[:4000],
        )
        memory = self.memory.render(job, self.name)
        responses = self.llm.generate_json(self.name, prompt, payload, memory)
        parsed: dict[str, Any] | None = None
        for response in responses:
            self.memory.append(job, self.name, response.provider, response.model, "assistant", response.content)
            parsed = self._extract_json_object(response.content)
            if parsed is None:
                message = (
                    f"Provider {response.provider}/{response.model} returned non-parseable JSON for stage {self.name}."
                )
                logger.warning(message)
                job.errors.append(message)
                continue
        return parsed

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError
