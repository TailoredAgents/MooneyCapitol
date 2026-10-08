from __future__ import annotations

from collections.abc import AsyncIterator

from app.v2.brokers.base import (
    BrokerCapabilities,
    BrokerCapabilityUnavailable,
    BrokerSubmissionDisabled,
    FuturesBrokerAdapter,
    OrderRequest,
)


class RithmicAdapterPlaceholder(FuturesBrokerAdapter):
    """Deliberately disabled execution-facing seam.

    Direct R|Protocol observation lives in ``rithmic_protocol`` and the
    dedicated capture service.  It is intentionally not wired into this
    mutation-capable interface.
    """

    def __init__(self) -> None:
        self._connected = False

    async def connect(self) -> None:
        raise BrokerCapabilityUnavailable("Rithmic execution transport is hard-disabled")

    async def disconnect(self) -> None:
        self._connected = False

    async def list_accounts(self):
        raise BrokerCapabilityUnavailable("use the isolated read-only capture observer")

    async def account_events(self, account_id: str) -> AsyncIterator:
        raise BrokerCapabilityUnavailable("use the isolated read-only capture observer")
        yield  # pragma: no cover

    async def working_orders(self, account_id: str):
        raise BrokerCapabilityUnavailable("use the isolated read-only capture observer")

    async def positions(self, account_id: str):
        raise BrokerCapabilityUnavailable("use the isolated read-only capture observer")

    async def execution_replay(self, account_id: str) -> AsyncIterator:
        raise BrokerCapabilityUnavailable("use the isolated read-only capture observer")
        yield  # pragma: no cover

    async def submit(self, request: OrderRequest) -> str:
        raise BrokerSubmissionDisabled("V2 broker submission is disabled")

    async def modify(self, broker_order_id: str, request: OrderRequest) -> None:
        raise BrokerSubmissionDisabled("V2 broker modification is disabled")

    async def cancel(self, broker_order_id: str) -> None:
        raise BrokerSubmissionDisabled("V2 broker cancellation is disabled")

    async def submit_bracket(self, entry: OrderRequest, stop: OrderRequest, target: OrderRequest):
        raise BrokerSubmissionDisabled("V2 broker bracket submission is disabled")

    async def cancel_all(self, account_id: str) -> None:
        raise BrokerSubmissionDisabled("V2 broker cancellation is disabled")

    async def account_risk(self, account_id: str):
        raise BrokerCapabilityUnavailable("use the isolated read-only capture observer")

    async def contract_reference(self, contract_id: str):
        raise BrokerCapabilityUnavailable("use the optional read-only Ticker observer")

    def capabilities(self) -> BrokerCapabilities:
        return BrokerCapabilities(
            supported=frozenset(),
            details={
                "status": "execution_disabled",
                "submission": "disabled",
                "observation": "separate_capture_service",
            },
        )

    def health(self):
        return {"ok": False, "ready": False, "connected": False, "submission_enabled": False}
