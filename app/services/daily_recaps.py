from __future__ import annotations

import json
from contextlib import AbstractContextManager
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Callable

from sqlalchemy import desc, select

from app.core.config_store import CONFIG
from app.db.models import AIArtifact, CopyOrder, CopyTargetAccount, MasterExecution
from app.db.session import get_session
from app.observability.logging import get_logger
from app.services.ai_artifacts import complete_ai_artifact, create_ai_artifact, fail_ai_artifact, latest_ai_artifact
from app.services.ai_client import OpenAITextClient, openai_model_from_env


logger = get_logger("daily_recaps")

DAILY_RECAP_TYPE = "daily_recap"
DAILY_RECAP_SOURCE_TYPE = "eod_report"
DAILY_RECAP_PROMPT_VERSION = "daily_recap_v1"

DAILY_RECAP_INSTRUCTIONS = """You write an end-of-day trading operations recap.
Use only the provided structured report data.
Write a concise operator-facing recap with:
1. headline for the day,
2. master/scout results,
3. copier health including latency and slippage,
4. learning observations when present,
5. concrete issues to check before the next session.
Do not give financial advice, do not predict future performance, and separate confirmed facts from interpretation."""


def daily_recap_input(
    *,
    trade_date: date | str,
    ledger_summary: dict[str, Any],
    copy_summary: dict[str, Any] | None = None,
    learning_translation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    top_setups = ledger_summary.get("by_setup") or []
    return {
        "date": str(trade_date),
        "ledger": {
            "total_trades": ledger_summary.get("total_trades"),
            "wins": ledger_summary.get("wins"),
            "win_rate": ledger_summary.get("win_rate"),
            "avg_r": ledger_summary.get("avg_r"),
            "expectancy": ledger_summary.get("expectancy"),
            "net_pnl": ledger_summary.get("net_pnl"),
            "top_setups": top_setups[:5],
            "by_hour": ledger_summary.get("by_hour") or {},
        },
        "copier": copy_summary or _empty_copy_summary(str(trade_date)),
        "learning": learning_translation,
    }


def generate_daily_recap(
    *,
    trade_date: date | str,
    ledger_summary: dict[str, Any],
    copy_summary: dict[str, Any] | None = None,
    learning_translation: dict[str, Any] | None = None,
    client: OpenAITextClient | None = None,
    slack: Any | None = None,
    session_scope: Callable[[], AbstractContextManager] = get_session,
) -> int | None:
    client = client or OpenAITextClient.from_env()
    if not client.enabled:
        return None

    with session_scope() as session:
        existing = latest_ai_artifact(
            session,
            artifact_type=DAILY_RECAP_TYPE,
            source_type=DAILY_RECAP_SOURCE_TYPE,
            source_id=str(trade_date),
        )
        if existing is not None:
            return int(existing.id)
        if copy_summary is None:
            copy_summary = copy_trading_summary(session=session, trade_date=trade_date)
        if learning_translation is None:
            learning_translation = latest_learning_translation(session=session, trade_date=trade_date)

    model = openai_model_from_env("OPENAI_DAILY_RECAP_MODEL", CONFIG.openai.daily_recap_model)
    input_json = daily_recap_input(
        trade_date=trade_date,
        ledger_summary=ledger_summary,
        copy_summary=copy_summary,
        learning_translation=learning_translation,
    )

    artifact_id: int
    with session_scope() as session:
        artifact = create_ai_artifact(
            session,
            artifact_type=DAILY_RECAP_TYPE,
            source_type=DAILY_RECAP_SOURCE_TYPE,
            source_id=str(trade_date),
            symbol=None,
            model=model,
            prompt_version=DAILY_RECAP_PROMPT_VERSION,
            input_json=input_json,
            status="running",
        )
        artifact_id = int(artifact.id)

    response = client.generate_text(
        model=model,
        instructions=DAILY_RECAP_INSTRUCTIONS,
        input_text=json.dumps(input_json, sort_keys=True, default=str),
        max_output_tokens=520,
        metadata={"feature": DAILY_RECAP_TYPE, "date": str(trade_date)},
    )

    with session_scope() as session:
        artifact = session.get(AIArtifact, artifact_id)
        if artifact is None:
            logger.warning("daily_recap.artifact_missing", artifact_id=artifact_id, date=str(trade_date))
            return artifact_id
        if response.ok:
            complete_ai_artifact(artifact, output_text=response.text, output_json=response.output_json)
            if slack is not None and response.text:
                slack.post(f"AI Daily Recap - {trade_date}\n{response.text}")
        else:
            fail_ai_artifact(artifact, error=response.error or response.status)
    return artifact_id


def copy_trading_summary(*, session: Any, trade_date: date | str) -> dict[str, Any]:
    start, end = _utc_window(trade_date)
    rows = session.execute(
        select(MasterExecution, CopyOrder, CopyTargetAccount.name)
        .join(CopyOrder, CopyOrder.master_execution_id == MasterExecution.id)
        .join(CopyTargetAccount, CopyOrder.target_account_id == CopyTargetAccount.id)
        .where(MasterExecution.executed_at >= start, MasterExecution.executed_at < end)
        .order_by(desc(MasterExecution.executed_at), desc(CopyOrder.id))
        .limit(500)
    ).all()

    statuses: dict[str, int] = {}
    latencies: list[float] = []
    slippages: list[float] = []
    issues: list[dict[str, Any]] = []
    filled = 0
    for master, order, target_name in rows:
        status = str(order.status or "unknown").lower()
        statuses[status] = statuses.get(status, 0) + 1
        if status in {"filled", "partially_filled"}:
            filled += 1
        if order.latency_ms is not None:
            latencies.append(float(order.latency_ms))
        slippage = _slippage_bps(master=master, order=order)
        if slippage is not None:
            slippages.append(slippage)
        if status in {"blocked", "rejected", "submit_failed", "cancelled", "expired"} or order.reject_reason:
            issues.append(
                {
                    "symbol": master.symbol,
                    "target": target_name,
                    "status": order.status,
                    "reason": order.reject_reason,
                    "latency_ms": order.latency_ms,
                }
            )

    return {
        "date": str(trade_date),
        "total_copy_orders": len(rows),
        "filled_copy_orders": filled,
        "status_counts": statuses,
        "latency": _latency_summary(latencies),
        "slippage_bps": _value_summary(slippages),
        "issues": issues[:10],
    }


def latest_learning_translation(*, session: Any, trade_date: date | str) -> dict[str, Any] | None:
    artifact = latest_ai_artifact(
        session,
        artifact_type="learning_translation",
        source_type="learning_report",
        source_id=str(trade_date),
    )
    if not artifact or artifact.status != "completed" or not artifact.output_text:
        return None
    return {
        "text": artifact.output_text,
        "model": artifact.model,
        "created_at": artifact.created_at.isoformat() if artifact.created_at else None,
    }


def _empty_copy_summary(trade_date: str) -> dict[str, Any]:
    return {
        "date": trade_date,
        "total_copy_orders": 0,
        "filled_copy_orders": 0,
        "status_counts": {},
        "latency": _latency_summary([]),
        "slippage_bps": _value_summary([]),
        "issues": [],
    }


def _latency_summary(values: list[float]) -> dict[str, Any]:
    summary = _value_summary(values)
    if not values:
        return {**summary, "under_300ms_rate": None}
    under = sum(1 for value in values if value <= 300)
    return {**summary, "under_300ms_rate": round(under / len(values), 4)}


def _value_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "max": None, "min": None}
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 4),
        "max": round(max(values), 4),
        "min": round(min(values), 4),
    }


def _slippage_bps(*, master: MasterExecution, order: CopyOrder) -> float | None:
    if order.avg_fill_price is None or not master.price:
        return None
    direction = 1 if (master.side or "").upper() == "BUY" else -1
    return round(((float(order.avg_fill_price) - float(master.price)) / float(master.price)) * 10_000 * direction, 4)


def _utc_window(value: date | str) -> tuple[datetime, datetime]:
    if isinstance(value, str):
        value = date.fromisoformat(value)
    start = datetime.combine(value, time.min, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start, end
