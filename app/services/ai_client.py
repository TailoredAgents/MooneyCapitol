from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AITextResult:
    status: str
    text: str | None = None
    output_json: dict[str, Any] | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "completed"


class OpenAITextClient:
    """Small no-op-safe wrapper around OpenAI text generation.

    The rest of the app can call this client without caring whether AI features
    are disabled, the API key is missing, or OpenAI returns an error.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        enabled: bool = False,
        timeout_seconds: float = 12.0,
        responses_url: str = OPENAI_RESPONSES_URL,
    ):
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.enabled = enabled
        self.timeout_seconds = timeout_seconds
        self.responses_url = responses_url

    @classmethod
    def from_env(cls, *, enabled_env: str = "OPENAI_AI_FEATURES_ENABLED") -> "OpenAITextClient":
        return cls(enabled=_env_flag(enabled_env))

    def generate_text(
        self,
        *,
        model: str,
        instructions: str,
        input_text: str,
        max_output_tokens: int = 400,
        metadata: dict[str, str] | None = None,
    ) -> AITextResult:
        if not self.enabled:
            return AITextResult(status="disabled", error="OpenAI AI features are disabled")
        if not self.api_key:
            return AITextResult(status="disabled", error="OPENAI_API_KEY is not configured")

        payload: dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": input_text,
            "max_output_tokens": max_output_tokens,
        }
        if metadata:
            payload["metadata"] = metadata

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(
                    self.responses_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
            response.raise_for_status()
            output = response.json()
            return AITextResult(status="completed", text=_extract_output_text(output), output_json=output)
        except Exception as exc:
            return AITextResult(status="error", error=str(exc))


def _extract_output_text(output: dict[str, Any]) -> str | None:
    direct = output.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    parts: list[str] = []
    for item in output.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return "\n".join(parts) if parts else None


def openai_model_from_env(env_name: str, default: str) -> str:
    return os.getenv(env_name, default).strip() or default


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in TRUE_VALUES
