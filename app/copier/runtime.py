from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from app.copier.factory import build_copy_targets
from app.copier.models import MasterExecutionEvent, WebullCredentials
from app.copier.service import CopyOrchestrator
from app.copier.state import set_copier_status
from app.copier.webull_master import WebullMasterEventListener
from app.core.config import CopierConfig
from app.core.config_store import CONFIG, refresh_config
from app.observability.logging import get_logger


logger = get_logger("copier.runtime")

FILLED_STATUSES = {
    "FILLED",
    "FINAL_FILLED",
    "PARTIALLY_FILLED",
    "PARTIAL_FILLED",
    "PARTIAL_EXECUTED",
    "EXECUTED",
    "TRADED",
}


@dataclass(frozen=True)
class EventHandlingResult:
    processed: int
    submitted: int
    ignored: int
    reason: str | None = None


class WebullCopierRuntime:
    def __init__(
        self,
        orchestrator: CopyOrchestrator | None = None,
        target_builder: Callable[[CopierConfig | None], list] | None = None,
        config_provider: Callable[[], CopierConfig] | None = None,
        refresh_interval_seconds: float = 1.0,
    ) -> None:
        self.orchestrator = orchestrator or CopyOrchestrator()
        self.target_builder = target_builder or build_copy_targets
        self.config_provider = config_provider or (lambda: CONFIG.copier)
        self.refresh_interval = timedelta(seconds=refresh_interval_seconds)
        self._last_config_refresh = datetime.now(tz=timezone.utc)
        self._target_cache: list | None = None
        self._target_cache_signature: str | None = None

    def handle_webull_event(self, *args: Any) -> EventHandlingResult:
        self._refresh_config_if_due()
        config = self.config_provider()
        payloads = list(_event_payloads(args))
        if not payloads:
            return EventHandlingResult(processed=0, submitted=0, ignored=1, reason="no_payload")

        processed = 0
        submitted = 0
        ignored = 0
        for payload in payloads:
            if not _is_fill_payload(payload, config):
                ignored += 1
                continue
            try:
                master = MasterExecutionEvent.from_webull_payload(payload)
            except ValueError as exc:
                ignored += 1
                set_copier_status(last_error=str(exc))
                logger.warning("copier.event_parse_failed", err=str(exc), payload=payload)
                continue

            set_copier_status(
                state="master_execution_received",
                last_master_execution_at=master.executed_at.isoformat(),
                last_error=None,
            )
            if not config.enabled:
                ignored += 1
                logger.info("copier.event_ignored", reason="copier_disabled", execution_id=master.execution_id)
                continue
            if config.mode == "read_only":
                targets = self._targets_for_config(config)
                results = self.orchestrator.plan_execution(master, targets)
                processed += 1
                set_copier_status(
                    state="read_only_planned",
                    last_error=None,
                )
                logger.info(
                    "copier.read_only_planned",
                    execution_id=master.execution_id,
                    would_copy=sum(1 for result in results if result.allowed),
                    blocked=sum(1 for result in results if not result.allowed),
                )
                continue
            if config.mode == "live" and not config.live_trading_enabled:
                ignored += 1
                set_copier_status(last_error="live mode blocked because live_trading_enabled is false")
                logger.warning("copier.live_mode_blocked", execution_id=master.execution_id)
                continue

            targets = self._targets_for_config(config)
            event_received_at = datetime.now(tz=timezone.utc)
            results = self.orchestrator.copy_execution(master, targets)
            processed += 1
            submitted += sum(1 for result in results if result.submitted)
            if any(result.submitted for result in results):
                last_submit = max(
                    (result.submitted_at for result in results if result.submitted_at),
                    default=datetime.now(tz=timezone.utc),
                )
                set_copier_status(
                    state="copied",
                    last_copy_order_at=last_submit.isoformat(),
                    latency=_latency_summary(results, event_received_at),
                )
        return EventHandlingResult(processed=processed, submitted=submitted, ignored=ignored)

    def _refresh_config_if_due(self) -> None:
        now = datetime.now(tz=timezone.utc)
        if now - self._last_config_refresh < self.refresh_interval:
            return
        refresh_config()
        self._last_config_refresh = now

    def _targets_for_config(self, config: CopierConfig) -> list:
        signature = config.model_dump_json()
        if self._target_cache is not None and self._target_cache_signature == signature:
            return self._target_cache
        self._target_cache = self.target_builder(config)
        self._target_cache_signature = signature
        return self._target_cache

    def warm_targets(self, config: CopierConfig | None = None) -> None:
        config = config or self.config_provider()
        targets = self._targets_for_config(config)
        for target in targets:
            warm_up = getattr(target.client, "warm_up", None)
            if callable(warm_up):
                warm_up()


def build_master_listener(runtime: WebullCopierRuntime | None = None) -> WebullMasterEventListener:
    refresh_config()
    config = CONFIG.copier
    credentials = _master_credentials(config)
    account_id = config.master_account or os.getenv(config.master_account_env)
    if not account_id:
        raise ValueError(f"Missing Webull master account id: {config.master_account_env}")
    runtime = runtime or WebullCopierRuntime()
    return WebullMasterEventListener(
        credentials=credentials,
        account_ids=[account_id],
        on_event=runtime.handle_webull_event,
    )


def start_webull_copier() -> None:
    refresh_config()
    config = CONFIG.copier
    if not config.enabled:
        set_copier_status(state="disabled", master_connected=False)
        logger.info("copier.disabled")
        return
    if config.master_broker != "webull":
        raise ValueError(f"Unsupported master broker for copier: {config.master_broker}")
    runtime = WebullCopierRuntime()
    runtime.warm_targets(config)
    listener = build_master_listener(runtime)
    listener.subscribe()


def _master_credentials(config: CopierConfig) -> WebullCredentials:
    app_key = os.getenv(config.master_app_key_env)
    app_secret = os.getenv(config.master_app_secret_env)
    endpoint = os.getenv(config.master_endpoint_env)
    events_endpoint = os.getenv(config.master_events_endpoint_env)
    missing = [
        name
        for name, value in [
            (config.master_app_key_env, app_key),
            (config.master_app_secret_env, app_secret),
            (config.master_endpoint_env, endpoint),
        ]
        if not value
    ]
    if missing:
        raise ValueError(f"Missing Webull master environment values: {', '.join(missing)}")
    return WebullCredentials(
        app_key=app_key or "",
        app_secret=app_secret or "",
        endpoint=endpoint or "",
        events_endpoint=events_endpoint or None,
        account_id=config.master_account or os.getenv(config.master_account_env),
        environment=config.mode,
    )


def _event_payloads(args: Iterable[Any]) -> Iterable[dict]:
    for item in args:
        if isinstance(item, list):
            for child in item:
                if isinstance(child, dict):
                    yield from _expand_payload(child)
        elif isinstance(item, dict):
            yield from _expand_payload(item)


def _expand_payload(payload: dict) -> Iterable[dict]:
    data = payload.get("data")
    if isinstance(data, list):
        for child in data:
            if isinstance(child, dict):
                yield child
        return
    if isinstance(data, dict):
        yield data
        return
    event_payload = payload.get("payload")
    if isinstance(event_payload, dict):
        child = dict(event_payload)
        for key in ("id", "event_type", "position", "timestamp"):
            if key in payload and key not in child:
                child[key] = payload[key]
        yield child
        return
    yield payload


def _is_fill_payload(payload: dict, config: CopierConfig) -> bool:
    if config.equities_only:
        instrument_type = str(payload.get("instrument_type") or payload.get("instrumentType") or "EQUITY").upper()
        if instrument_type not in {"EQUITY", "STOCK"}:
            return False
    status = str(
        payload.get("scene_type")
        or payload.get("sceneType")
        or payload.get("status")
        or payload.get("order_status")
        or payload.get("orderStatus")
        or ""
    ).upper()
    if status and status not in FILLED_STATUSES:
        return False
    quantity = payload.get("filled_qty") or payload.get("filled_quantity") or payload.get("last_filled_qty")
    price = payload.get("avg_fill_price") or payload.get("filled_price") or payload.get("last_filled_price")
    return quantity not in (None, "", 0, "0") and price not in (None, "", 0, "0")


def _latency_summary(results: list, event_received_at: datetime) -> dict:
    broker_submit_ms = {}
    event_to_submit_started_ms = {}
    event_to_broker_response_ms = {}
    for result in results:
        if result.latency_ms is not None:
            broker_submit_ms[result.target] = result.latency_ms
        if result.submitted_at is not None:
            started_ms = (result.submitted_at - event_received_at).total_seconds() * 1000
            event_to_submit_started_ms[result.target] = started_ms
            if result.latency_ms is not None:
                event_to_broker_response_ms[result.target] = started_ms + result.latency_ms
    return {
        "broker_submit_ms": broker_submit_ms,
        "event_to_submit_started_ms": event_to_submit_started_ms,
        "event_to_broker_response_ms": event_to_broker_response_ms,
    }
