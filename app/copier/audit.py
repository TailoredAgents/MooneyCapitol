from __future__ import annotations

from app.db.models import CopierAuditEvent
from app.db.session import get_session
from app.observability.logging import get_logger


logger = get_logger("copier.audit")


def record_audit_event(
    event_type: str,
    message: str,
    actor: str | None = None,
    target_account_id: int | None = None,
    payload: dict | None = None,
) -> None:
    """Persist an operator/system audit event when the DB schema is available."""
    try:
        with get_session() as session:
            session.add(
                CopierAuditEvent(
                    event_type=event_type,
                    actor=actor,
                    target_account_id=target_account_id,
                    message=message,
                    payload=payload,
                )
            )
    except Exception as exc:  # pragma: no cover - startup/migration guard
        logger.warning("copier.audit.write_failed", event_type=event_type, err=str(exc))

