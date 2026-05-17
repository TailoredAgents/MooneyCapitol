from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.copier.event_capture import WebullEventCapture
from app.copier.factory import build_copy_targets
from app.copier.models import MasterExecutionEvent, WebullEquityOrder
from app.copier.runtime import _event_payloads, _is_fill_payload
from app.copier.service import CopyOrchestrator
from app.core.config import CopierConfig


class ReadOnlyTradingClient:
    def place_equity_order(self, account_id: str, order: WebullEquityOrder) -> dict:
        raise RuntimeError("read-only session cannot place broker orders")


class _NoopSlack:
    def post(self, text: str) -> None:
        return None


@dataclass
class ReadOnlySessionSummary:
    events: int = 0
    payloads: int = 0
    parsed_fills: int = 0
    ignored: int = 0
    parse_errors: int = 0
    duplicates: int = 0
    would_copy: int = 0
    blocked: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": self.events,
            "payloads": self.payloads,
            "parsed_fills": self.parsed_fills,
            "ignored": self.ignored,
            "parse_errors": self.parse_errors,
            "duplicates": self.duplicates,
            "would_copy": self.would_copy,
            "blocked": self.blocked,
            "errors": list(self.errors),
        }


class WebullReadOnlySessionRunner:
    def __init__(
        self,
        config: CopierConfig,
        *,
        output_path: Path,
        orchestrator: CopyOrchestrator | None = None,
    ) -> None:
        self.config = config
        self.capture = WebullEventCapture(output_path, self.config)
        self.orchestrator = orchestrator or CopyOrchestrator(
            slack=_NoopSlack(),
            background_persistence=False,
            background_alerts=False,
        )
        self.summary = ReadOnlySessionSummary()
        self._target_cache_signature: str | None = None
        self._target_cache: list | None = None

    def handle_event(self, *args: Any) -> dict[str, Any]:
        capture_record = self.capture.handle_event(*args)
        self.summary.events += 1
        event_report = {
            "sequence": capture_record["sequence"],
            "capture_status": capture_record["status"],
            "payloads": [],
        }
        payloads = list(_event_payloads(args))
        self.summary.payloads += len(payloads)
        if not payloads:
            self.summary.ignored += 1
            return event_report

        for payload in payloads:
            payload_report = self._handle_payload(payload)
            event_report["payloads"].append(payload_report)
        return event_report

    def _handle_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not _is_fill_payload(payload, self.config):
            self.summary.ignored += 1
            return {"status": "ignored", "reason": "not_fill_payload"}
        try:
            master = MasterExecutionEvent.from_webull_payload(payload)
        except ValueError as exc:
            self.summary.parse_errors += 1
            self.summary.errors.append(str(exc))
            return {"status": "parse_error", "error": str(exc)}

        self.summary.parsed_fills += 1
        if not self.config.enabled:
            self.summary.ignored += 1
            return {"status": "ignored", "reason": "copier_disabled", "execution_id": master.execution_id}

        results = self.orchestrator.plan_execution(master, self._targets())
        if not results:
            self.summary.duplicates += 1
            return {"status": "duplicate_or_no_targets", "execution_id": master.execution_id}

        would_copy = sum(1 for result in results if result.allowed)
        blocked = sum(1 for result in results if not result.allowed)
        self.summary.would_copy += would_copy
        self.summary.blocked += blocked
        return {
            "status": "planned",
            "execution_id": master.execution_id,
            "symbol": master.symbol,
            "side": master.side,
            "would_copy": would_copy,
            "blocked": blocked,
            "results": [
                {
                    "target": result.target,
                    "allowed": result.allowed,
                    "reason": result.reason,
                    "client_order_id": result.client_order_id,
                    "quantity": result.quantity,
                }
                for result in results
            ],
        }

    def _targets(self) -> list:
        signature = self.config.model_dump_json()
        if self._target_cache is not None and self._target_cache_signature == signature:
            return self._target_cache
        self._target_cache = build_copy_targets(
            self.config,
            client_factory=lambda target_cfg: ReadOnlyTradingClient(),
            allow_missing_accounts=True,
        )
        self._target_cache_signature = signature
        return self._target_cache


def read_only_session_config(config: CopierConfig, *, simulate_copying: bool = False) -> CopierConfig:
    updates: dict[str, Any] = {"mode": "read_only", "live_trading_enabled": False}
    if simulate_copying:
        updates.update({"enabled": True, "global_kill_switch": False})
    return config.model_copy(update=updates, deep=True)
