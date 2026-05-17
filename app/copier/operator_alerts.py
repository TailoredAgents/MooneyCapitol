from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from app.copier.reconciliation import PositionSyncResult, ReconciliationResult, StartupRecoveryResult


LIMIT_BLOCK_REASONS = {
    "max_notional_exceeded",
    "max_position_pct_exceeded",
    "max_daily_notional_exceeded",
    "max_daily_trades_exceeded",
    "max_orders_per_minute_exceeded",
    "short_copy_disabled",
}


@dataclass
class OperatorAlertThrottle:
    min_interval: timedelta = timedelta(minutes=15)
    last_sent: dict[str, datetime] = field(default_factory=dict)

    def allow(self, key: str, now: datetime | None = None) -> bool:
        now = now or datetime.now(tz=timezone.utc)
        last = self.last_sent.get(key)
        if last and now - last < self.min_interval:
            return False
        self.last_sent[key] = now
        return True


def readiness_alert_text(readiness: dict) -> str | None:
    blockers = readiness.get("blockers") or []
    if not blockers:
        return None
    lines = ["Copier readiness blockers detected:"]
    for item in blockers[:8]:
        lines.append(f"- {item.get('label', item.get('key', 'unknown'))}")
    if len(blockers) > 8:
        lines.append(f"- plus {len(blockers) - 8} more")
    lines.append("Open /dashboard -> Copier for details.")
    return "\n".join(lines)


def reconciliation_alert_text(result: ReconciliationResult) -> str | None:
    if not (result.errors or result.mismatches):
        return None
    return (
        "Copier reconciliation issue:\n"
        f"- checked: {result.checked}\n"
        f"- updated: {result.updated}\n"
        f"- mismatches: {result.mismatches}\n"
        f"- errors: {result.errors}\n"
        "Open /dashboard -> Copier -> Reconciliations."
    )


def position_sync_alert_text(result: PositionSyncResult) -> str | None:
    if not (result.errors or result.mismatches):
        return None
    return (
        "Copier position sync issue:\n"
        f"- targets checked: {result.targets_checked}\n"
        f"- symbols checked: {result.symbols_checked}\n"
        f"- mismatches: {result.mismatches}\n"
        f"- errors: {result.errors}\n"
        "Open /dashboard -> Copier -> Reconciliations."
    )


def recovery_alert_text(result: StartupRecoveryResult) -> str | None:
    if not (result.errors or result.mismatches or result.stale_open_orders):
        return None
    return (
        "Copier startup recovery warning:\n"
        f"- checked: {result.checked}\n"
        f"- updated: {result.updated}\n"
        f"- mismatches: {result.mismatches}\n"
        f"- errors: {result.errors}\n"
        f"- stale open orders: {result.stale_open_orders}\n"
        "Review Copier readiness and reconciliations before disabling the kill switch."
    )


def listener_error_alert_text(error: Exception | str) -> str:
    return f"Copier listener error: {error}\nOpen /dashboard -> Copier for runtime status."


def copy_limit_alert_text(results: list) -> str | None:
    blocked = [result for result in results if not result.submitted and result.reason in LIMIT_BLOCK_REASONS]
    if not blocked:
        return None
    lines = ["Copied trade blocked before broker submit:"]
    for result in blocked[:8]:
        lines.append(f"- {result.target}: {result.reason} qty={result.quantity}")
    if len(blocked) > 8:
        lines.append(f"- plus {len(blocked) - 8} more")
    lines.append("Open /dashboard -> Copier for details.")
    return "\n".join(lines)
