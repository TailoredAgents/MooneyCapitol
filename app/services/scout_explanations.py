from __future__ import annotations

import json
from contextlib import AbstractContextManager
from typing import Any, Callable

from app.core.config_store import CONFIG
from app.db.models import AIArtifact
from app.db.session import get_session
from app.observability.logging import get_logger
from app.services.ai_artifacts import complete_ai_artifact, create_ai_artifact, fail_ai_artifact
from app.services.ai_client import OpenAITextClient, openai_model_from_env


logger = get_logger("scout_explanations")

SCOUT_EXPLANATION_PROMPT_VERSION = "scout_explanation_v1"
SCOUT_EXPLANATION_TYPE = "scout_explanation"
SCOUT_SOURCE_TYPE = "alert"

SCOUT_EXPLANATION_INSTRUCTIONS = """You explain small-cap momentum scout alerts for an experienced trader.
Use only the provided structured alert data.
Write one concise sentence in plain English.
Do not give financial advice, do not promise outcomes, and do not add facts that are not in the input.
Mention the strongest concrete factors such as direction, R:R, RVOL, spread, L2, p2R, or warning notes when present."""


def scout_explanation_input(
    *,
    alert_id: int,
    alert_type: str,
    symbol: str,
    direction: str | None,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "alert_id": alert_id,
        "alert_type": alert_type,
        "symbol": symbol,
        "direction": direction,
        "entry": payload.get("entry"),
        "stop": payload.get("stop"),
        "target": payload.get("target") or payload.get("targets"),
        "rr": payload.get("rr"),
        "p2r": payload.get("p2r"),
        "spread": payload.get("spread"),
        "rvol": payload.get("rvol"),
        "l2": payload.get("l2"),
        "box_summary": payload.get("box_summary"),
        "note": payload.get("note"),
        "features": {
            key: payload.get("features", {}).get(key)
            for key in ("rvol_break", "l2_mean", "l2_persist", "spread_cents", "price_bucket", "time_bucket")
            if isinstance(payload.get("features"), dict) and key in payload.get("features", {})
        },
    }


def generate_scout_alert_explanation(
    *,
    alert_id: int,
    alert_type: str,
    symbol: str,
    direction: str | None,
    payload: dict[str, Any],
    client: OpenAITextClient | None = None,
    session_scope: Callable[[], AbstractContextManager] = get_session,
) -> int | None:
    client = client or OpenAITextClient.from_env()
    if not client.enabled:
        return None

    model = openai_model_from_env("OPENAI_SCOUT_EXPLANATION_MODEL", CONFIG.openai.scout_explanation_model)
    input_json = scout_explanation_input(
        alert_id=alert_id,
        alert_type=alert_type,
        symbol=symbol,
        direction=direction,
        payload=payload,
    )

    artifact_id: int
    with session_scope() as session:
        artifact = create_ai_artifact(
            session,
            artifact_type=SCOUT_EXPLANATION_TYPE,
            source_type=SCOUT_SOURCE_TYPE,
            source_id=alert_id,
            symbol=symbol,
            model=model,
            prompt_version=SCOUT_EXPLANATION_PROMPT_VERSION,
            input_json=input_json,
            status="running",
        )
        artifact_id = int(artifact.id)

    result = client.generate_text(
        model=model,
        instructions=SCOUT_EXPLANATION_INSTRUCTIONS,
        input_text=json.dumps(input_json, sort_keys=True, default=str),
        max_output_tokens=120,
        metadata={"feature": SCOUT_EXPLANATION_TYPE, "alert_type": alert_type, "symbol": symbol},
    )

    with session_scope() as session:
        artifact = session.get(AIArtifact, artifact_id)
        if artifact is None:
            logger.warning("scout_explanation.artifact_missing", artifact_id=artifact_id, alert_id=alert_id)
            return artifact_id
        if result.ok:
            complete_ai_artifact(artifact, output_text=result.text, output_json=result.output_json)
        else:
            fail_ai_artifact(artifact, error=result.error or result.status)
    return artifact_id
