from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Mapping

from app.v2.domain.broker_observation import (
    BrokerAccountPnlObservation,
    BrokerBracketObservation,
    BrokerContractReference,
    BrokerExecutionObservation,
    BrokerOrderObservation,
    BrokerPlant,
    BrokerPositionObservation,
    BrokerRmsObservation,
    ObservedBrokerAccount,
    PlantHealth,
)
from app.v2.domain.models import AccountRiskSnapshot, BrokerAccount, FuturesContract, NormalizedOrderEvent, PositionSnapshot


class BrokerCapability(str, Enum):
    ACCOUNT_LIFECYCLE = "ACCOUNT_LIFECYCLE"
    WORKING_ORDERS = "WORKING_ORDERS"
    POSITIONS = "POSITIONS"
    EXECUTION_REPLAY = "EXECUTION_REPLAY"
    SUBMIT = "SUBMIT"
    MODIFY = "MODIFY"
    CANCEL = "CANCEL"
    BRACKET = "BRACKET"
    CANCEL_ALL = "CANCEL_ALL"
    ACCOUNT_RISK = "ACCOUNT_RISK"
    CONTRACT_REFERENCE = "CONTRACT_REFERENCE"
    SYSTEM_DISCOVERY = "SYSTEM_DISCOVERY"
    ACCOUNT_DISCOVERY = "ACCOUNT_DISCOVERY"
    ORDER_STREAM = "ORDER_STREAM"
    ORDER_SNAPSHOT = "ORDER_SNAPSHOT"
    ORDER_HISTORY = "ORDER_HISTORY"
    FILL_HISTORY = "FILL_HISTORY"
    POSITION_STREAM = "POSITION_STREAM"
    POSITION_SNAPSHOT = "POSITION_SNAPSHOT"
    PNL_STREAM = "PNL_STREAM"
    PNL_SNAPSHOT = "PNL_SNAPSHOT"
    RMS_OBSERVATION = "RMS_OBSERVATION"
    BRACKET_STREAM = "BRACKET_STREAM"
    BRACKET_SNAPSHOT = "BRACKET_SNAPSHOT"
    REFERENCE_SEARCH = "REFERENCE_SEARCH"


@dataclass(frozen=True)
class BrokerCapabilities:
    supported: frozenset[BrokerCapability]
    details: Mapping[str, str]

    def has(self, capability: BrokerCapability) -> bool:
        return capability in self.supported


@dataclass(frozen=True)
class OrderRequest:
    account_id: str
    contract_id: str
    side: str
    quantity: int
    order_type: str
    client_intent_id: str

    @property
    def client_order_id(self) -> str:
        """Compatibility alias; Rithmic does not provide server idempotency for it."""
        return self.client_intent_id


class BrokerCapabilityUnavailable(RuntimeError):
    pass


class BrokerSubmissionDisabled(RuntimeError):
    pass


class ReadOnlyBrokerObserver(ABC):
    """Observation-only boundary. Implementations expose no broker mutation methods."""

    @abstractmethod
    async def connect(self, required_plants: frozenset[BrokerPlant]) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    async def discover_accounts(self) -> tuple[ObservedBrokerAccount, ...]: ...

    @abstractmethod
    def order_events(self, account_id: str) -> AsyncIterator[BrokerOrderObservation]: ...

    @abstractmethod
    def execution_events(self, account_id: str) -> AsyncIterator[BrokerExecutionObservation]: ...

    @abstractmethod
    def position_events(self, account_id: str) -> AsyncIterator[BrokerPositionObservation]: ...

    @abstractmethod
    def pnl_events(self, account_id: str) -> AsyncIterator[BrokerAccountPnlObservation]: ...

    @abstractmethod
    def rms_events(self, account_id: str) -> AsyncIterator[BrokerRmsObservation]: ...

    @abstractmethod
    def bracket_events(self, account_id: str) -> AsyncIterator[BrokerBracketObservation]: ...

    @abstractmethod
    async def reference_data(self, symbol: str, exchange: str) -> BrokerContractReference: ...

    @abstractmethod
    def plant_health(self) -> Mapping[BrokerPlant, PlantHealth]: ...


class FuturesBrokerAdapter(ABC):
    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    async def list_accounts(self) -> tuple[BrokerAccount, ...]: ...

    @abstractmethod
    def account_events(self, account_id: str) -> AsyncIterator[NormalizedOrderEvent]: ...

    @abstractmethod
    async def working_orders(self, account_id: str) -> tuple[NormalizedOrderEvent, ...]: ...

    @abstractmethod
    async def positions(self, account_id: str) -> tuple[PositionSnapshot, ...]: ...

    @abstractmethod
    def execution_replay(self, account_id: str) -> AsyncIterator[NormalizedOrderEvent]: ...

    @abstractmethod
    async def submit(self, request: OrderRequest) -> str: ...

    @abstractmethod
    async def modify(self, broker_order_id: str, request: OrderRequest) -> None: ...

    @abstractmethod
    async def cancel(self, broker_order_id: str) -> None: ...

    @abstractmethod
    async def submit_bracket(self, entry: OrderRequest, stop: OrderRequest, target: OrderRequest) -> tuple[str, ...]: ...

    @abstractmethod
    async def cancel_all(self, account_id: str) -> None: ...

    @abstractmethod
    async def account_risk(self, account_id: str) -> AccountRiskSnapshot: ...

    @abstractmethod
    async def contract_reference(self, contract_id: str) -> FuturesContract: ...

    @abstractmethod
    def capabilities(self) -> BrokerCapabilities: ...

    @abstractmethod
    def health(self) -> Mapping[str, object]: ...
