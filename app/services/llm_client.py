from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from app.core.config import settings
from app.models.schemas import LLMResponse, ProviderSpec

logger = logging.getLogger(__name__)


class MultiProviderLLMClient:
    def __init__(self) -> None:
        self.timeout = settings.llm_timeout_seconds

    def has_any_provider(self) -> bool:
        return any(self._api_key_for(spec.provider) or spec.provider == "ollama" for spec in settings.parsed_roster())

    def available_specs(self) -> list[ProviderSpec]:
        specs: list[ProviderSpec] = []
        for spec in settings.parsed_roster():
            if spec.provider == "ollama" or self._api_key_for(spec.provider):
                specs.append(spec)
        return specs

    def generate_json(
        self,
        stage: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        shared_memory: str,
    ) -> list[LLMResponse]:
        message = json.dumps(user_payload, ensure_ascii=False, indent=2)
        responses: list[LLMResponse] = []
        prior_outputs = ""
        specs = self.available_specs()
        strategy = settings.llm_strategy.lower()
        for spec in specs:
            prompt = (
                f"{system_prompt}\n\n"
                f"Shared engineer memory:\n{shared_memory}\n\n"
                f"Prior model drafts:\n{prior_outputs or 'None yet.'}\n\n"
                f"Return JSON only for stage '{stage}'.\n\n"
                f"Input payload:\n{message}"
            )
            try:
                response = self._call_provider_with_retry(spec, prompt)
                responses.append(response)
                if strategy == "ensemble":
                    prior_outputs += f"\n[{spec.provider}:{spec.model}]\n{response.content}\n"
                elif strategy in {"fallback_chain", "first_success"}:
                    break
            except Exception as exc:
                logger.warning("LLM provider %s/%s failed for %s: %s", spec.provider, spec.model, stage, exc)
        return responses

    def _call_provider_with_retry(self, spec: ProviderSpec, prompt: str) -> LLMResponse:
        last_error: Exception | None = None
        for attempt in range(settings.llm_max_retries + 1):
            try:
                return self._call_provider(spec, prompt)
            except Exception as exc:
                last_error = exc
                if attempt >= settings.llm_max_retries:
                    break
                sleep_seconds = settings.llm_retry_backoff_seconds * (attempt + 1)
                logger.info(
                    "Retrying provider %s/%s after error on attempt %s: %s",
                    spec.provider,
                    spec.model,
                    attempt + 1,
                    exc,
                )
                time.sleep(sleep_seconds)
        assert last_error is not None
        raise last_error

    def _call_provider(self, spec: ProviderSpec, prompt: str) -> LLMResponse:
        provider = spec.provider.lower()
        if provider in {"openai", "openrouter", "deepseek", "groq", "together", "xai"}:
            return self._call_openai_compatible(spec, prompt)
        if provider == "anthropic":
            return self._call_anthropic(spec, prompt)
        if provider == "google":
            return self._call_google(spec, prompt)
        if provider == "ollama":
            return self._call_ollama(spec, prompt)
        raise ValueError(f"Unsupported provider: {provider}")

    def _call_openai_compatible(self, spec: ProviderSpec, prompt: str) -> LLMResponse:
        base_urls = {
            "openai": settings.llm_openai_base_url,
            "openrouter": settings.llm_openrouter_base_url,
            "deepseek": settings.llm_deepseek_base_url,
            "groq": settings.llm_groq_base_url,
            "together": settings.llm_together_base_url,
            "xai": settings.llm_xai_base_url,
        }
        api_key = self._api_key_for(spec.provider)
        url = f"{base_urls[spec.provider]}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": spec.model,
            "messages": [
                {"role": "system", "content": "You are a careful ML systems engineer. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        content = data["choices"][0]["message"]["content"]
        return LLMResponse(provider=spec.provider, model=spec.model, content=content, raw_payload=data)

    def _call_anthropic(self, spec: ProviderSpec, prompt: str) -> LLMResponse:
        headers = {
            "x-api-key": settings.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": spec.model,
            "max_tokens": settings.llm_anthropic_max_tokens,
            "system": "You are a careful ML systems engineer. Return JSON only.",
            "messages": [{"role": "user", "content": prompt}],
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{settings.llm_anthropic_base_url}/messages", headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        content = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
        return LLMResponse(provider=spec.provider, model=spec.model, content=content, raw_payload=data)

    def _call_google(self, spec: ProviderSpec, prompt: str) -> LLMResponse:
        params = {"key": settings.google_api_key}
        payload = {
            "contents": [{"parts": [{"text": f"You are a careful ML systems engineer. Return JSON only.\n\n{prompt}"}]}],
            "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
        }
        url = f"{settings.llm_google_base_url}/models/{spec.model}:generateContent"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, params=params, json=payload)
            response.raise_for_status()
            data = response.json()
        candidates = data.get("candidates", [])
        parts = candidates[0]["content"]["parts"] if candidates else []
        content = "".join(part.get("text", "") for part in parts)
        return LLMResponse(provider=spec.provider, model=spec.model, content=content, raw_payload=data)

    def _call_ollama(self, spec: ProviderSpec, prompt: str) -> LLMResponse:
        payload = {
            "model": spec.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": "You are a careful ML systems engineer. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            "format": "json",
        }
        url = settings.ollama_base_url.rstrip("/") + settings.llm_ollama_chat_path
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        content = data.get("message", {}).get("content", "")
        return LLMResponse(provider=spec.provider, model=spec.model, content=content, raw_payload=data)

    def _api_key_for(self, provider: str) -> str:
        mapping = {
            "openai": settings.openai_api_key,
            "anthropic": settings.anthropic_api_key,
            "google": settings.google_api_key,
            "openrouter": settings.openrouter_api_key,
            "deepseek": settings.deepseek_api_key,
            "groq": settings.groq_api_key,
            "together": settings.together_api_key,
            "xai": settings.xai_api_key,
        }
        return mapping.get(provider.lower(), "")
