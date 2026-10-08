from datetime import datetime, timezone
from decimal import Decimal

import pytest

from app.v2.brokers.base import BrokerCapability, OrderRequest
from app.v2.domain.broker_observation import (
    BrokerAccountIdentity,
    BrokerOrderObservation,
    BrokerPlant,
    CommandOutcome,
    ObservationSource,
    ObservedOrderState,
    PlantHealth,
    decimal_from_wire,
)


NOW = datetime(2026, 10, 7, tzinfo=timezone.utc)


def test_wire_decimals_preserve_missing_and_reject_invalid_values():
    assert decimal_from_wire(None, field_name="balance") is None
    assert decimal_from_wire("", field_name="balance") is None
    assert decimal_from_wire("100.2500", field_name="balance") == Decimal("100.2500")
    with pytest.raises(ValueError):
        decimal_from_wire("not-money", field_name="balance")


def test_broker_identity_requires_all_opaque_components():
    identity = BrokerAccountIdentity("fcm", "ib", "account")
    assert identity.opaque_key == ("fcm", "ib", "account")
    with pytest.raises(ValueError):
        BrokerAccountIdentity("", "ib", "account")


def test_unknown_completion_is_preserved_without_inventing_a_fill():
    observation = BrokerOrderObservation(
        event_id="event-1",
        connection_generation="generation-1",
        source=ObservationSource.LIVE,
        account=BrokerAccountIdentity("fcm", "ib", "account"),
        template_id=351,
        received_at=NOW,
        basket_id="basket-1",
        normalized_state=ObservedOrderState.COMPLETED_UNKNOWN,
        raw_notify_type="COMPLETE",
        raw_status="complete",
        completion_reason="undocumented-value",
        command_outcome=CommandOutcome.NOT_APPLICABLE,
        quantity=2**33,
    )
    assert observation.normalized_state is ObservedOrderState.COMPLETED_UNKNOWN
    assert observation.quantity == 2**33


def test_required_plants_are_independently_ready():
    order = PlantHealth(BrokerPlant.ORDER, True, "g-order", True, True, True, False, NOW, NOW)
    pnl = PlantHealth(BrokerPlant.PNL, True, "g-pnl", True, True, False, False, NOW, NOW)
    assert order.ready
    assert not pnl.ready


def test_v2_order_request_names_local_identity_as_intent():
    request = OrderRequest("account", "contract", "BUY", 1, "MARKET", "intent-1")
    assert request.client_intent_id == "intent-1"
    assert request.client_order_id == "intent-1"


def test_read_only_capabilities_are_distinct_from_mutations():
    assert BrokerCapability.ORDER_STREAM is not BrokerCapability.SUBMIT
    assert BrokerCapability.BRACKET_STREAM is not BrokerCapability.BRACKET
