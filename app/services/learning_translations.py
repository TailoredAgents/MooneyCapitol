from __future__ import annotations

import json
from contextlib import AbstractContextManager
from datetime import date
from typing import Any, Callable

from app.core.config_store import CONFIG
from app.db.models import AIArtifact
from app.db.session import get_session
from app.observability.logging import get_logger
from app.services.ai_artifacts import complete_ai_artifact, create_ai_artifact, fail_ai_artifact
from app.services.ai_client import OpenAITextClient, openai_model_from_env


logger = get_logger("learning_translations")

LEARNING_TRANSLATION_TYPE = "learning_translation"
LEARNING_SOURCE_TYPE = "learning_report"
LEARNING_TRANSLATION_PROMPT_VERSION = "learning_translation_v1"

LEARNING_TRANSLATION_INSTRUCTIONS = """You translate MooneyCapitol nightly learning reports for the operator.
Use only the provided structured learning result.
Write 2-4 concise sentences in plain English.
Explain what the model learned, which features mattered, and any caution if training was skipped or weak.
Do not give trading advice, do not promise performance, and do not invent data."""


def learning_translation_input(*, trade_date: date | str, result: dict[str, Any]) -> dict[str, Any]:
    feature_importance = result.get("feature_importance") or []
    ranking = result.get("ranking_metrics") or {}
    sandbox = result.get("sandbox_enhanced") or {}
    regime = result.get("regime_specialists") or {}
    return {
        "date": str(trade_date),
        "status": result.get("status"),
        "model_type": result.get("model_type"),
        "rows": result.get("rows"),
        "positives": result.get("positives"),
        "reason": result.get("reason"),
        "source_breakdown": result.get("source_breakdown"),
        "label_breakdown": result.get("label_breakdown"),
        "top_features": feature_importance[:8],
        "ranking_metrics": {
            "taken_rows": ranking.get("taken_rows"),
            "top_5_taken_hit_rate": ranking.get("top_5_taken_hit_rate"),
            "top_10_taken_hit_rate": ranking.get("top_10_taken_hit_rate"),
            "median_taken_rank": ranking.get("median_taken_rank"),
        },
        "sandbox_enhanced": {
            "enabled": sandbox.get("enabled"),
            "real_samples": sandbox.get("real_samples"),
            "synthetic_samples": sandbox.get("synthetic_samples"),
            "enhancement_ratio": sandbox.get("enhancement_ratio"),
        },
        "regime_specialists": {
            "regimes_trained": regime.get("regimes_trained"),
            "total_samples": regime.get("total_samples"),
        },
    }


def generate_learning_report_translation(
    *,
    trade_date: date | str,
    result: dict[str, Any],
    client: OpenAITextClient | None = None,
    slack: Any | None = None,
    session_scope: Callable[[], AbstractContextManager] = get_session,
) -> int | None:
    client = client or OpenAITextClient.from_env()
    if not client.enabled:
        return None

    model = openai_model_from_env("OPENAI_LEARNING_TRANSLATION_MODEL", CONFIG.openai.learning_translation_model)
    input_json = learning_translation_input(trade_date=trade_date, result=result)

    artifact_id: int
    with session_scope() as session:
        artifact = create_ai_artifact(
            session,
            artifact_type=LEARNING_TRANSLATION_TYPE,
            source_type=LEARNING_SOURCE_TYPE,
            source_id=str(trade_date),
            symbol=None,
            model=model,
            prompt_version=LEARNING_TRANSLATION_PROMPT_VERSION,
            input_json=input_json,
            status="running",
        )
        artifact_id = int(artifact.id)

    response = client.generate_text(
        model=model,
        instructions=LEARNING_TRANSLATION_INSTRUCTIONS,
        input_text=json.dumps(input_json, sort_keys=True, default=str),
        max_output_tokens=220,
        metadata={"feature": LEARNING_TRANSLATION_TYPE, "date": str(trade_date)},
    )

    with session_scope() as session:
        artifact = session.get(AIArtifact, artifact_id)
        if artifact is None:
            logger.warning("learning_translation.artifact_missing", artifact_id=artifact_id, date=str(trade_date))
            return artifact_id
        if response.ok:
            complete_ai_artifact(artifact, output_text=response.text, output_json=response.output_json)
            if slack is not None and response.text:
                slack.post(f"Learning AI Summary - {trade_date}\n{response.text}")
        else:
            fail_ai_artifact(artifact, error=response.error or response.status)
    return artifact_id
