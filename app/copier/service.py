from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, time, timedelta, timezone
from typing import Any

from app.copier.engine import CopyEngine, CopyResult, CopyTarget
from app.copier.models import MasterExecutionEvent
from app.copier.operator_alerts import OperatorAlertThrottle, copy_limit_alert_text
from app.copier.risk import RiskUsage
from app.copier.repository import (
    load_copy_positions,
    load_copy_risk_usage,
    master_execution_exists,
    persist_copy_results,
    persist_master_execution,
)
from app.db.session import get_session
from app.observability.logging import get_logger


logger = get_logger("copier.service")


class CopyOrchestrator:
    def __init__(
        self,
        engine: CopyEngine | None = None,
        session_scope: Callable = get_session,
        require_persistence: bool = True,
        slack: Any | None = None,
        alert_throttle: OperatorAlertThrottle | None = None,
        db_dedupe_before_submit: bool = False,
        hydrate_state_before_submit: bool = False,
        background_persistence: bool = True,
        background_alerts: bool = True,
        persist_results: bool = True,
        log_latency: bool = True,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self.engine = engine or CopyEngine()
        self.session_scope = session_scope
        self.require_persistence = require_persistence
        self.slack = slack or _default_slack()
        self.alert_throttle = alert_throttle or OperatorAlertThrottle()
        self.db_dedupe_before_submit = db_dedupe_before_submit
        self.hydrate_state_before_submit = hydrate_state_before_submit
        self.background_persistence = background_persistence
        self.background_alerts = background_alerts
        self.persist_results = persist_results
        self.log_latency = log_latency
        self.executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="copier-bg")
        self._seen_execution_keys: set[tuple[str, str | None, str]] = set()
        self._target_positions: dict[str, dict[str, float]] = {}
        self._usage_events: dict[str, list[tuple[datetime, float]]] = {}

    def copy_execution(self, master: MasterExecutionEvent, targets: list[CopyTarget]) -> list[CopyResult]:
        key = (master.broker, master.account_id, master.execution_id)
        if key in self._seen_execution_keys:
            logger.info("copier.duplicate_execution_skipped", broker=master.broker, execution_id=master.execution_id)
            return []
        if self.db_dedupe_before_submit and self._exists_in_db(master, key):
            return []

        self._seen_execution_keys.add(key)
        targets_for_submit = self._targets_for_hot_path(targets)
        if self.hydrate_state_before_submit:
            targets_for_submit = self._targets_with_db_state(targets_for_submit)

        received_at = datetime.now(tz=timezone.utc)
        results = self.engine.copy_execution(master, targets_for_submit)
        self._record_hot_path_state(master, results)
        self._dispatch_alerts(results)
        self._dispatch_persistence(master, targets_for_submit, results)
        if self.log_latency:
            self._log_latency(master, results, received_at)
        return results

    def plan_execution(self, master: MasterExecutionEvent, targets: list[CopyTarget]) -> list[CopyResult]:
        key = (master.broker, master.account_id, master.execution_id)
        if key in self._seen_execution_keys:
            logger.info("copier.duplicate_execution_skipped", broker=master.broker, execution_id=master.execution_id)
            return []
        if self.db_dedupe_before_submit and self._exists_in_db(master, key):
            return []

        self._seen_execution_keys.add(key)
        targets_for_plan = self._targets_for_hot_path(targets)
        if self.hydrate_state_before_submit:
            targets_for_plan = self._targets_with_db_state(targets_for_plan)

        results = self.engine.plan_execution(master, targets_for_plan)
        self._dispatch_persistence(master, targets_for_plan, results)
        return results

    def record_master_only(self, master: MasterExecutionEvent) -> bool:
        key = (master.broker, master.account_id, master.execution_id)
        if key in self._seen_execution_keys:
            return False
        with self.session_scope() as session:
            if master_execution_exists(session, master):
                self._seen_execution_keys.add(key)
                return False
            persist_master_execution(session, master)
        self._seen_execution_keys.add(key)
        return True

    def _exists_in_db(self, master: MasterExecutionEvent, key: tuple[str, str | None, str]) -> bool:
        try:
            with self.session_scope() as session:
                if master_execution_exists(session, master):
                    self._seen_execution_keys.add(key)
                    logger.info(
                        "copier.duplicate_execution_skipped",
                        broker=master.broker,
                        account_id=master.account_id,
                        execution_id=master.execution_id,
                    )
                    return True
        except Exception as exc:
            logger.warning(
                "copier.dedupe_check_failed",
                broker=master.broker,
                account_id=master.account_id,
                execution_id=master.execution_id,
                err=str(exc),
            )
            if self.require_persistence:
                raise
        return False

    def _targets_for_hot_path(self, targets: list[CopyTarget]) -> list[CopyTarget]:
        now = datetime.now(tz=timezone.utc)
        return [
            replace(
                target,
                risk_usage=self._usage_for_target(target.name, now),
                positions=dict(self._target_positions.get(target.name, target.positions or {})),
            )
            for target in targets
        ]

    def _targets_with_db_state(self, targets: list[CopyTarget]) -> list[CopyTarget]:
        try:
            with self.session_scope() as session:
                target_names = [target.name for target in targets]
                usage_by_name = load_copy_risk_usage(session, target_names)
                positions_by_name = load_copy_positions(session, target_names)
        except Exception as exc:
            logger.warning("copier.target_state_failed", err=str(exc))
            if self.require_persistence:
                raise
            return targets
        hydrated = [
            replace(
                target,
                risk_usage=usage_by_name.get(target.name, target.risk_usage),
                positions=positions_by_name.get(target.name, target.positions or {}),
            )
            for target in targets
        ]
        for target in hydrated:
            self._target_positions[target.name] = dict(target.positions or {})
        return hydrated

    def _record_hot_path_state(self, master: MasterExecutionEvent, results: list[CopyResult]) -> None:
        for result in results:
            if not result.submitted or result.quantity <= 0:
                continue
            symbol = master.symbol.upper()
            target_positions = self._target_positions.setdefault(result.target, {})
            signed_qty = result.quantity if master.side == "BUY" else -result.quantity
            target_positions[symbol] = target_positions.get(symbol, 0.0) + signed_qty
            self._usage_events.setdefault(result.target, []).append(
                (result.submitted_at or datetime.now(tz=timezone.utc), result.quantity * master.price)
            )

    def _usage_for_target(self, target_name: str, now: datetime) -> RiskUsage:
        events = self._usage_events.get(target_name, [])
        day_start = datetime.combine(now.date(), time.min, tzinfo=timezone.utc)
        minute_start = now - timedelta(minutes=1)
        active_events = [(ts, notional) for ts, notional in events if ts >= day_start]
        self._usage_events[target_name] = active_events
        minute_events = [(ts, notional) for ts, notional in active_events if ts >= minute_start]
        return RiskUsage(
            daily_notional=sum(notional for _, notional in active_events),
            daily_orders=len(active_events),
            minute_orders=len(minute_events),
        )

    def _dispatch_persistence(
        self,
        master: MasterExecutionEvent,
        targets: list[CopyTarget],
        results: list[CopyResult],
    ) -> None:
        if not self.persist_results:
            return
        if self.background_persistence:
            self.executor.submit(self._persist_results, master, targets, results)
            return
        self._persist_results(master, targets, results)

    def _persist_results(
        self,
        master: MasterExecutionEvent,
        targets: list[CopyTarget],
        results: list[CopyResult],
    ) -> None:
        try:
            targets_by_name = {target.name: target for target in targets}
            with self.session_scope() as session:
                persist_copy_results(session, master, targets_by_name, results)
        except Exception as exc:
            logger.warning(
                "copier.persist_failed",
                broker=master.broker,
                account_id=master.account_id,
                execution_id=master.execution_id,
                err=str(exc),
            )
            if self.require_persistence and not self.background_persistence:
                raise

    def _dispatch_alerts(self, results: list[CopyResult]) -> None:
        if self.background_alerts:
            self.executor.submit(self._alert_limit_blocks, results)
            return
        self._alert_limit_blocks(results)

    def _log_latency(
        self,
        master: MasterExecutionEvent,
        results: list[CopyResult],
        received_at: datetime,
    ) -> None:
        submitted_latencies = [result.latency_ms for result in results if result.latency_ms is not None]
        if not submitted_latencies:
            return
        logger.info(
            "copier.hot_path_latency",
            broker=master.broker,
            execution_id=master.execution_id,
            max_submit_ms=max(submitted_latencies),
            submitted=sum(1 for result in results if result.submitted),
            elapsed_ms=(datetime.now(tz=timezone.utc) - received_at).total_seconds() * 1000,
        )

    def _alert_limit_blocks(self, results: list[CopyResult]) -> None:
        text = copy_limit_alert_text(results)
        if text and self.alert_throttle.allow("copy_limit_block"):
            self.slack.post(text)


class _NoopSlack:
    def post(self, text: str) -> None:
        logger.info("copier.slack_unavailable", text=text)


def _default_slack():
    try:
        from app.adapters.slack import SlackAdapter

        return SlackAdapter()
    except Exception as exc:
        logger.warning("copier.slack_adapter_unavailable", err=str(exc))
        return _NoopSlack()
