from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.db.models import AIArtifact


def create_ai_artifact(
    session: Session,
    *,
    artifact_type: str,
    source_type: str,
    source_id: str | int | None = None,
    symbol: str | None = None,
    model: str,
    prompt_version: str,
    input_json: dict[str, Any] | None = None,
    status: str = "created",
) -> AIArtifact:
    artifact = AIArtifact(
        artifact_type=artifact_type,
        source_type=source_type,
        source_id=str(source_id) if source_id is not None else None,
        symbol=symbol.upper() if symbol else None,
        model=model,
        prompt_version=prompt_version,
        input_json=input_json,
        status=status,
        created_at=_now(),
        updated_at=_now(),
    )
    session.add(artifact)
    session.flush()
    return artifact


def complete_ai_artifact(
    artifact: AIArtifact,
    *,
    output_text: str | None,
    output_json: dict[str, Any] | None = None,
) -> AIArtifact:
    artifact.status = "completed"
    artifact.output_text = output_text
    artifact.output_json = output_json
    artifact.error = None
    artifact.updated_at = _now()
    return artifact


def fail_ai_artifact(artifact: AIArtifact, *, error: str) -> AIArtifact:
    artifact.status = "error"
    artifact.error = error
    artifact.updated_at = _now()
    return artifact


def latest_ai_artifact(
    session: Session,
    *,
    artifact_type: str,
    source_type: str | None = None,
    source_id: str | int | None = None,
    symbol: str | None = None,
) -> AIArtifact | None:
    stmt = select(AIArtifact).where(AIArtifact.artifact_type == artifact_type)
    if source_type is not None:
        stmt = stmt.where(AIArtifact.source_type == source_type)
    if source_id is not None:
        stmt = stmt.where(AIArtifact.source_id == str(source_id))
    if symbol is not None:
        stmt = stmt.where(AIArtifact.symbol == symbol.upper())
    return session.execute(stmt.order_by(desc(AIArtifact.created_at)).limit(1)).scalars().first()


def _now() -> datetime:
    return datetime.now(timezone.utc)
