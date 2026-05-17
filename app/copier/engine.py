from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Protocol

from app.copier.ids import copy_client_order_id
from app.copier.models import MasterExecutionEvent, WebullEquityOrder
from app.copier.risk import RiskPolicy, RiskUsage, validate_copy
from app.copier.sizing import SizingPolicy, size_child_order


class TradingClient(Protocol):
    def place_equity_order(self, account_id: str, order: WebullEquityOrder) -> dict: ...


@dataclass(frozen=True)
class CopyTarget:
    name: str
    account_id: str
    client: TradingClient
    sizing: SizingPolicy
    risk: RiskPolicy
    master_equity: float | None = None
    target_equity: float | None = None
    risk_usage: RiskUsage = RiskUsage()
    positions: dict[str, float] | None = None


@dataclass(frozen=True)
class CopyResult:
    target: str
    allowed: bool
    submitted: bool
    reason: str | None
    client_order_id: str | None = None
    quantity: float = 0.0
    order: WebullEquityOrder | None = None
    response: dict | None = None
    error: str | None = None
    submitted_at: datetime | None = None
    latency_ms: float | None = None


class CopyEngine:
    def copy_execution(self, master: MasterExecutionEvent, targets: list[CopyTarget]) -> list[CopyResult]:
        results: list[CopyResult] = []
        for idx, target in enumerate(targets):
            planned = self._plan_target(master, target, idx)
            if not planned.allowed:
                results.append(planned)
                continue

            started = perf_counter()
            submitted_at = datetime.now(tz=timezone.utc)
            try:
                response = target.client.place_equity_order(target.account_id, planned.order)
            except Exception as exc:
                results.append(
                    CopyResult(
                        target=target.name,
                        allowed=True,
                        submitted=False,
                        reason="broker_submit_failed",
                        client_order_id=planned.client_order_id,
                        quantity=planned.quantity,
                        order=planned.order,
                        error=str(exc),
                        submitted_at=submitted_at,
                        latency_ms=(perf_counter() - started) * 1000,
                    )
                )
                continue
            results.append(
                CopyResult(
                    target=target.name,
                    allowed=True,
                    submitted=True,
                    reason=None,
                    client_order_id=planned.client_order_id,
                    quantity=planned.quantity,
                    order=planned.order,
                    response=response,
                    submitted_at=submitted_at,
                    latency_ms=(perf_counter() - started) * 1000,
                )
            )
        return results

    def plan_execution(self, master: MasterExecutionEvent, targets: list[CopyTarget]) -> list[CopyResult]:
        results: list[CopyResult] = []
        for idx, target in enumerate(targets):
            planned = self._plan_target(master, target, idx)
            if planned.allowed:
                planned = CopyResult(
                    target=planned.target,
                    allowed=True,
                    submitted=False,
                    reason="read_only",
                    client_order_id=planned.client_order_id,
                    quantity=planned.quantity,
                    order=planned.order,
                )
            results.append(planned)
        return results

    def _plan_target(self, master: MasterExecutionEvent, target: CopyTarget, sequence: int) -> CopyResult:
        quantity = size_child_order(
            master,
            target.sizing,
            master_equity=target.master_equity,
            target_equity=target.target_equity,
        )
        client_order_id = copy_client_order_id(master.execution_id, target.name, sequence=sequence)
        decision = validate_copy(
            master,
            quantity,
            target.risk,
            target_equity=target.target_equity,
            usage=target.risk_usage,
            target_position_qty=_target_position_qty(target, master.symbol),
        )
        if not decision.allowed:
            return CopyResult(
                target=target.name,
                allowed=False,
                submitted=False,
                reason=decision.reason,
                client_order_id=client_order_id,
                quantity=quantity,
            )

        order = WebullEquityOrder(
            symbol=master.symbol,
            side=master.side,
            quantity=quantity,
            client_order_id=client_order_id,
            order_type="MARKET",
        )
        return CopyResult(
            target=target.name,
            allowed=True,
            submitted=False,
            reason=None,
            client_order_id=client_order_id,
            quantity=quantity,
            order=order,
        )


def _target_position_qty(target: CopyTarget, symbol: str) -> float:
    positions = target.positions or {}
    return float(positions.get(symbol.upper(), 0.0) or 0.0)
