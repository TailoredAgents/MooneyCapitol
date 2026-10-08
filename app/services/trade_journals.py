from __future__ import annotations

import json
from contextlib import AbstractContextManager
from typing import Any, Callable

from sqlalchemy import String, cast, desc, select

from app.core.config_store import CONFIG
from app.db.models import AIArtifact, CopyOrder, CopyTargetAccount, MasterExecution
from app.db.session import get_session
from app.observability.logging import get_logger
from app.services.ai_artifacts import complete_ai_artifact, create_ai_artifact, fail_ai_artifact, latest_ai_artifact
from app.services.ai_client import OpenAITextClient, openai_model_from_env
from app.security import redact_sensitive


logger = get_logger("trade_journals")

TRADE_JOURNAL_TYPE = "trade_journal"
TRADE_JOURNAL_SOURCE_TYPE = "copy_order"
TRADE_JOURNAL_PROMPT_VERSION = "trade_journal_v1"
TRADE_JOURNAL_FINAL_STATUSES = {
    "blocked",
    "cancelled",
    "expired",
    "filled",
    "partially_filled",
    "rejected",
    "submit_failed",
    "would_copy",
}

TRADE_JOURNAL_INSTRUCTIONS = """You write concise trade journal notes for a trading operator.
Use only the provided structured master execution and copied-order data.
Write 3-5 short bullet lines:
- setup/execution facts
- copy result
- latency/slippage quality when present
- clear issue to review if rejected, blocked, failed, or high slippage
Do not give financial advice, do not predict future performance, and do not invent missing P&L."""


def trade_journal_input(
    *,
    master: MasterExecution,
    order: CopyOrder,
    target_name: str | None,
    scout_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    master_notional = _notional(master.qty, master.price)
    copy_notional = _notional(order.qty, master.price)
    copy_filled_notional = _notional(order.filled_qty, order.avg_fill_price)
    slippage_bps = _slippage_bps(
        side=master.side,
        master_price=master.price,
        copy_avg_fill_price=order.avg_fill_price,
    )
    tags = trade_journal_tags(order=order, slippage_bps=slippage_bps, scout_context=scout_context)
    return redact_sensitive({
        "master_execution": {
            "id": master.id,
            "broker_execution_id": master.broker_execution_id,
            "broker_order_id": master.broker_order_id,
            "account_ref": master.account_ref,
            "symbol": master.symbol,
            "side": master.side,
            "qty": master.qty,
            "price": master.price,
            "notional": master_notional,
            "executed_at": _iso(master.executed_at),
            "received_at": _iso(master.received_at),
        },
        "copy_order": {
            "id": order.id,
            "target": target_name,
            "client_order_id": order.client_order_id,
            "broker_order_id": order.broker_order_id,
            "qty": order.qty,
            "notional": copy_notional,
            "status": order.status,
            "submitted_at": _iso(order.submitted_at),
            "accepted_at": _iso(order.accepted_at),
            "filled_at": _iso(order.filled_at),
            "filled_qty": order.filled_qty,
            "avg_fill_price": order.avg_fill_price,
            "filled_notional": copy_filled_notional,
            "latency_ms": order.latency_ms,
            "reject_reason": order.reject_reason,
            "slippage_bps": slippage_bps,
        },
        "scout_context": scout_context or {"source": "manual_no_alert"},
        "tags": tags,
    })


def trade_journal_tags(
    *,
    order: CopyOrder,
    slippage_bps: float | None,
    scout_context: dict[str, Any] | None = None,
) -> list[str]:
    tags: list[str] = []
    status = (order.status or "").lower()
    if scout_context:
        tags.append("scout_alert")
    else:
        tags.append("manual_no_alert")
    if status in {"filled", "partially_filled"}:
        tags.append("copied")
    if status in {"blocked", "rejected", "submit_failed", "cancelled", "expired"}:
        tags.append(status)
    if order.latency_ms is not None:
        tags.append("under_300ms" if float(order.latency_ms) < 300 else "over_300ms")
    if slippage_bps is not None and abs(float(slippage_bps)) >= 25:
        tags.append("high_slippage")
    return tags


def generate_trade_journal(
    *,
    master: MasterExecution,
    order: CopyOrder,
    target_name: str | None = None,
    scout_context: dict[str, Any] | None = None,
    client: OpenAITextClient | None = None,
    session_scope: Callable[[], AbstractContextManager] = get_session,
) -> int | None:
    client = client or OpenAITextClient.from_env()
    if not client.enabled:
        return None

    model = openai_model_from_env("OPENAI_TRADE_JOURNAL_MODEL", CONFIG.openai.trade_journal_model)
    input_json = trade_journal_input(master=master, order=order, target_name=target_name, scout_context=scout_context)

    artifact_id: int
    with session_scope() as session:
        existing = latest_ai_artifact(
            session,
            artifact_type=TRADE_JOURNAL_TYPE,
            source_type=TRADE_JOURNAL_SOURCE_TYPE,
            source_id=order.id,
        )
        if existing is not None:
            return int(existing.id)
        artifact = create_ai_artifact(
            session,
            artifact_type=TRADE_JOURNAL_TYPE,
            source_type=TRADE_JOURNAL_SOURCE_TYPE,
            source_id=order.id,
            symbol=master.symbol,
            model=model,
            prompt_version=TRADE_JOURNAL_PROMPT_VERSION,
            input_json=input_json,
            status="running",
        )
        artifact_id = int(artifact.id)

    response = client.generate_text(
        model=model,
        instructions=TRADE_JOURNAL_INSTRUCTIONS,
        input_text=json.dumps(input_json, sort_keys=True, default=str),
        max_output_tokens=240,
        metadata={"feature": TRADE_JOURNAL_TYPE, "symbol": master.symbol, "copy_order_id": str(order.id)},
    )

    with session_scope() as session:
        artifact = session.get(AIArtifact, artifact_id)
        if artifact is None:
            logger.warning("trade_journal.artifact_missing", artifact_id=artifact_id, copy_order_id=order.id)
            return artifact_id
        if response.ok:
            complete_ai_artifact(artifact, output_text=response.text, output_json=response.output_json)
        else:
            fail_ai_artifact(artifact, error=response.error or response.status)
    return artifact_id


def generate_pending_trade_journals(
    *,
    limit: int = 25,
    client: OpenAITextClient | None = None,
    session_scope: Callable[[], AbstractContextManager] = get_session,
) -> dict[str, int]:
    client = client or OpenAITextClient.from_env()
    if not client.enabled:
        return {"checked": 0, "created": 0, "skipped": 0}

    candidates = _load_journal_candidates(limit=limit, session_scope=session_scope)
    created = 0
    skipped = 0
    for master, order, target_name in candidates:
        artifact_id = generate_trade_journal(
            master=master,
            order=order,
            target_name=target_name,
            client=client,
            session_scope=session_scope,
        )
        if artifact_id:
            created += 1
        else:
            skipped += 1
    return {"checked": len(candidates), "created": created, "skipped": skipped}


def _load_journal_candidates(
    *,
    limit: int,
    session_scope: Callable[[], AbstractContextManager] = get_session,
) -> list[tuple[MasterExecution, CopyOrder, str | None]]:
    with session_scope() as session:
        journal_exists = (
            select(AIArtifact.id)
            .where(
                AIArtifact.artifact_type == TRADE_JOURNAL_TYPE,
                AIArtifact.source_type == TRADE_JOURNAL_SOURCE_TYPE,
                AIArtifact.source_id == cast(CopyOrder.id, String),
            )
            .exists()
        )
        stmt = (
            select(MasterExecution, CopyOrder, CopyTargetAccount.name)
            .join(CopyOrder, CopyOrder.master_execution_id == MasterExecution.id)
            .join(CopyTargetAccount, CopyOrder.target_account_id == CopyTargetAccount.id)
            .where(CopyOrder.status.in_(sorted(TRADE_JOURNAL_FINAL_STATUSES)))
            .where(~journal_exists)
            .order_by(desc(MasterExecution.executed_at), desc(CopyOrder.id))
            .limit(limit)
        )
        return session.execute(stmt).all()


def _notional(qty: float | int | None, price: float | int | None) -> float | None:
    if qty is None or price is None:
        return None
    return round(float(qty) * float(price), 4)


def _slippage_bps(
    *,
    side: str | None,
    master_price: float | int | None,
    copy_avg_fill_price: float | int | None,
) -> float | None:
    if copy_avg_fill_price is None or not master_price:
        return None
    direction = 1 if (side or "").upper() == "BUY" else -1
    return round(((float(copy_avg_fill_price) - float(master_price)) / float(master_price)) * 10_000 * direction, 4)


def _iso(value: Any) -> str | None:
    return value.isoformat() if value else None
