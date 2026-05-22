from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from app.observability.logging import get_logger
from app.services.kv_store import get_json, set_json


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
TRUE_VALUES = {"1", "true", "yes", "on"}
logger = get_logger("openai_client")


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
        quota = reserve_openai_budget(metadata=metadata)
        if not quota.allowed:
            return AITextResult(status="limited", error=quota.reason)

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


@dataclass(frozen=True)
class OpenAIQuotaDecision:
    allowed: bool
    reason: str | None = None
    daily_count: int = 0
    daily_limit: int = 0
    research_count: int = 0
    research_limit: int = 0


def reserve_openai_budget(metadata: dict[str, str] | None = None) -> OpenAIQuotaDecision:
    """Reserve one daily OpenAI request slot.

    Limits are intentionally request-count based. Exact token cost varies by
    model and output length, so this gives the operator a simple hard stop that
    prevents runaway background AI jobs.
    """

    daily_limit = _env_int("OPENAI_DAILY_REQUEST_LIMIT", 200)
    research_limit = _env_int("OPENAI_RESEARCH_DAILY_REQUEST_LIMIT", 30)
    feature = (metadata or {}).get("feature", "")
    is_research = feature == "ticker_research"
    today = datetime.now(timezone.utc).date().isoformat()
    key = f"openai_usage:{today}"

    if daily_limit <= 0 and (not is_research or research_limit <= 0):
        return OpenAIQuotaDecision(True, daily_limit=daily_limit, research_limit=research_limit)

    try:
        usage = get_json(key) or {}
        if not isinstance(usage, dict) or usage.get("date") != today:
            usage = {"date": today, "requests": 0, "research_requests": 0, "features": {}}
        requests = int(usage.get("requests") or 0)
        research_requests = int(usage.get("research_requests") or 0)
        if daily_limit > 0 and requests >= daily_limit:
            return OpenAIQuotaDecision(
                False,
                reason=f"OpenAI daily request limit reached ({requests}/{daily_limit})",
                daily_count=requests,
                daily_limit=daily_limit,
                research_count=research_requests,
                research_limit=research_limit,
            )
        if is_research and research_limit > 0 and research_requests >= research_limit:
            return OpenAIQuotaDecision(
                False,
                reason=f"OpenAI research daily request limit reached ({research_requests}/{research_limit})",
                daily_count=requests,
                daily_limit=daily_limit,
                research_count=research_requests,
                research_limit=research_limit,
            )
        usage["requests"] = requests + 1
        usage["research_requests"] = research_requests + (1 if is_research else 0)
        features = usage.get("features") if isinstance(usage.get("features"), dict) else {}
        if feature:
            features[feature] = int(features.get(feature) or 0) + 1
        usage["features"] = features
        set_json(key, usage)
        return OpenAIQuotaDecision(
            True,
            daily_count=int(usage["requests"]),
            daily_limit=daily_limit,
            research_count=int(usage["research_requests"]),
            research_limit=research_limit,
        )
    except Exception as exc:
        logger.warning("openai.budget_check_failed", err=str(exc))
        return OpenAIQuotaDecision(False, reason=f"OpenAI budget check failed: {exc}")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in TRUE_VALUES
