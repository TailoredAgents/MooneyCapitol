import asyncio
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Mapping

import pytest

from app.v2.market_data import (
    AvailabilityMode,
    BarPayload,
    BboPayload,
    HistoricalMarketDataRequest,
    MarketDataCapability,
    MarketDataEntitlement,
    MarketDataEntitlementError,
    MarketDataError,
    MarketEventKind,
    ProviderContractBinding,
    TradePayload,
    UnsupportedMarketDataCapability,
    datetime_to_epoch_ns,
)
from app.v2.providers.massive_futures import MassiveFuturesProvider


NOW = datetime(2026, 10, 5, 16, tzinfo=timezone.utc)
START = datetime(2026, 9, 18, 13, 30, tzinfo=timezone.utc)
END = START + timedelta(minutes=2)
NQ_ID = "CME:NQ:2026-12-18"
ES_ID = "CME:ES:2026-12-18"
NQ = ProviderContractBinding(NQ_ID, "NQ", "NQZ6", "XCME")
ES = ProviderContractBinding(ES_ID, "ES", "ESZ6", "XCME")
HISTORICAL_CAPABILITIES = frozenset(
    {
        MarketDataCapability.CONTRACT_REFERENCE,
        MarketDataCapability.SCHEDULES,
        MarketDataCapability.BARS,
        MarketDataCapability.TRADES,
        MarketDataCapability.BBO,
        MarketDataCapability.HISTORICAL_REPLAY,
    }
)


class FakeTransport:
    def __init__(self, routes: Mapping[str, Any]):
        self.routes = dict(routes)
        self.calls: list[tuple[str, dict[str, str]]] = []

    async def get_json(self, url: str, *, params: Mapping[str, str]):
        self.calls.append((url, dict(params)))
        response = self.routes[url]
        if callable(response):
            response = response(dict(params))
        return response


def _response(results, *, request_id="req-1", next_url=None):
    payload = {"status": "OK", "request_id": request_id, "results": results}
    if next_url is not None:
        payload["next_url"] = next_url
    return payload


def _entitlement(**changes):
    values = dict(enabled=True, capabilities=HISTORICAL_CAPABILITIES)
    values.update(changes)
    return MarketDataEntitlement(**values)


def _provider(routes, *, entitlement=None, bindings=(NQ, ES), max_pages=10_000):
    transport = FakeTransport(routes)
    provider = MassiveFuturesProvider(
        transport=transport,
        bindings=bindings,
        entitlement=entitlement or _entitlement(),
        clock=lambda: NOW,
        max_pages=max_pages,
    )
    return provider, transport


def _collect(provider, request):
    async def run():
        return [event async for event in provider.replay(request)]

    return asyncio.run(run())


def _market_routes():
    start_ns = datetime_to_epoch_ns(START)
    routes = {}
    for symbol, base_price in (("NQZ6", "20500.25"), ("ESZ6", "6000.25")):
        routes[f"https://api.massive.com/futures/v1/aggs/{symbol}"] = _response(
            [
                {
                    "ticker": symbol,
                    "window_start": start_ns,
                    "session_end_date": "2026-09-18",
                    "open": base_price,
                    "high": str(Decimal(base_price) + Decimal("1.00")),
                    "low": str(Decimal(base_price) - Decimal("0.50")),
                    "close": str(Decimal(base_price) + Decimal("0.25")),
                    "volume": 12,
                    "transactions": 7,
                }
            ],
            request_id=f"bar-{symbol}",
        )
        routes[f"https://api.massive.com/futures/v1/trades/{symbol}"] = _response(
            [
                {
                    "ticker": symbol,
                    "timestamp": start_ns + 30_000_000_123,
                    "session_end_date": "2026-09-18",
                    "price": base_price,
                    "size": 3,
                    "sequence_number": 41,
                    "report_sequence": 39,
                    "channel": 312,
                }
            ],
            request_id=f"trade-{symbol}",
        )
        routes[f"https://api.massive.com/futures/v1/quotes/{symbol}"] = _response(
            [
                {
                    "ticker": symbol,
                    "timestamp": start_ns + 20_000_000_999,
                    "session_end_date": "2026-09-18",
                    "bid_price": str(Decimal(base_price) - Decimal("0.25")),
                    "bid_size": 4,
                    "ask_price": base_price,
                    "ask_size": 5,
                    "bid_timestamp": start_ns + 19_000_000_111,
                    "ask_timestamp": start_ns + 20_000_000_999,
                    "sequence_number": 31,
                    "report_sequence": 30,
                    "channel": 312,
                }
            ],
            request_id=f"quote-{symbol}",
        )
    return routes


def test_multi_contract_historical_replay_is_typed_finalized_and_deterministic():
    provider, transport = _provider(_market_routes())
    request = HistoricalMarketDataRequest(
        contract_ids=(NQ_ID, ES_ID),
        start=START,
        end=END,
        kinds=frozenset({MarketEventKind.BAR, MarketEventKind.TRADE, MarketEventKind.BBO}),
        bar_resolutions=("1min",),
    )

    events = _collect(provider, request)

    assert len(events) == 6
    assert [event.source_timestamp_ns for event in events] == sorted(event.source_timestamp_ns for event in events)
    assert {event.contract_id for event in events} == {NQ_ID, ES_ID}
    assert {type(event.payload) for event in events} == {BarPayload, TradePayload, BboPayload}
    assert all(event.availability_mode == AvailabilityMode.FINALIZED_HISTORICAL for event in events)
    assert all(event.is_final and event.revision == 0 and event.lineage.revision == 0 for event in events)
    assert all(not event.eligible_at(END) for event in events)
    assert all(event.eligible_at(END, allow_finalized_historical=True) for event in events)

    nq_trade = next(event.payload for event in events if event.contract_id == NQ_ID and event.kind == MarketEventKind.TRADE)
    assert nq_trade.price == Decimal("20500.25")
    assert nq_trade.occurred_at_ns % 1_000 == 123  # precision survives beyond datetime microseconds
    nq_bar_event = next(event for event in events if event.contract_id == NQ_ID and event.kind == MarketEventKind.BAR)
    assert nq_bar_event.source_timestamp == nq_bar_event.payload.ends_at
    assert nq_bar_event.source_timestamp_ns == nq_bar_event.payload.window_end_ns

    paths = {url.removeprefix("https://api.massive.com") for url, _ in transport.calls}
    assert paths == {
        "/futures/v1/aggs/NQZ6",
        "/futures/v1/aggs/ESZ6",
        "/futures/v1/trades/NQZ6",
        "/futures/v1/trades/ESZ6",
        "/futures/v1/quotes/NQZ6",
        "/futures/v1/quotes/ESZ6",
    }
    assert all("apiKey" not in params for _, params in transport.calls)


def test_reference_and_schedule_use_exact_contract_facts_and_preserve_events():
    routes = {
        "https://api.massive.com/futures/v1/contracts": _response(
            [
                {
                    "ticker": "ESZ6",
                    "product_code": "ES",
                    "trading_venue": "XCME",
                    "type": "single",
                    "date": "2026-09-18",
                    "first_trade_date": "2024-01-02",
                    "last_trade_date": "2026-12-18",
                    "settlement_date": "2026-12-18",
                    "trade_tick_size": 0.25,
                    "settlement_tick_size": 0.25,
                    "spread_tick_size": 0.05,
                }
            ]
        ),
        "https://api.massive.com/futures/v1/schedules": _response(
            [
                {
                    "event": "close",
                    "timestamp": "2026-09-18T21:00:00+00:00",
                    "session_end_date": "2026-09-18",
                    "product_code": "ES",
                    "trading_venue": "XCME",
                },
                {
                    "event": "open",
                    "timestamp": "2026-09-17T22:00:00+00:00",
                    "session_end_date": "2026-09-18",
                    "product_code": "ES",
                    "trading_venue": "XCME",
                },
                {
                    "event": "pre_open",
                    "timestamp": "2026-09-17T21:45:00+00:00",
                    "session_end_date": "2026-09-18",
                    "product_code": "ES",
                    "trading_venue": "XCME",
                },
            ]
        ),
    }
    provider, transport = _provider(routes, bindings=(ES,))

    reference = asyncio.run(provider.contract_reference(ES_ID, as_of=date(2026, 9, 18)))
    schedule = asyncio.run(provider.schedule(ES_ID, date(2026, 9, 18)))

    assert reference.contract_id == ES_ID and reference.provider_symbol == "ESZ6"
    assert reference.trade_tick_size == Decimal("0.25")
    assert reference.settlement_date == date(2026, 12, 18)
    assert [item.event for item in schedule.events] == ["pre_open", "open", "close"]
    assert schedule.trade_date == date(2026, 9, 18)
    reference_call = transport.calls[0]
    assert reference_call[0].endswith("/futures/v1/contracts")
    assert reference_call[1]["ticker"] == "ESZ6"
    assert reference_call[1]["date"] == "2026-09-18"
    assert transport.calls[1][0].endswith("/futures/v1/schedules")


def test_entitlement_is_fail_closed_and_rest_streaming_is_never_advertised():
    disabled, _ = _provider({}, entitlement=MarketDataEntitlement(), bindings=(NQ,))
    assert disabled.capabilities() == frozenset()
    with pytest.raises(MarketDataEntitlementError, match="disabled"):
        asyncio.run(disabled.contract_reference(NQ_ID))

    bars_only = _entitlement(
        capabilities=frozenset({MarketDataCapability.BARS, MarketDataCapability.HISTORICAL_REPLAY})
    )
    provider, _ = _provider({}, entitlement=bars_only, bindings=(NQ,))
    request = HistoricalMarketDataRequest(
        (NQ_ID,), START, END, frozenset({MarketEventKind.BBO})
    )
    with pytest.raises(MarketDataEntitlementError, match="BBO"):
        _collect(provider, request)
    assert MarketDataCapability.LIVE_STREAMING not in provider.capabilities()
    with pytest.raises(UnsupportedMarketDataCapability, match="does not support"):
        provider.stream((NQ_ID,), frozenset({MarketEventKind.TRADE}))


def test_delayed_entitlement_rejects_unavailable_window_before_transport_call():
    entitlement = _entitlement(delivery_delay=timedelta(minutes=10))
    provider, transport = _provider({}, entitlement=entitlement, bindings=(NQ,))
    request = HistoricalMarketDataRequest(
        (NQ_ID,), NOW - timedelta(minutes=5), NOW, frozenset({MarketEventKind.TRADE})
    )
    with pytest.raises(MarketDataEntitlementError, match="finalized"):
        _collect(provider, request)
    assert transport.calls == []


def test_safe_pagination_is_locally_sorted_and_external_next_url_is_rejected():
    start_ns = datetime_to_epoch_ns(START)
    initial = "https://api.massive.com/futures/v1/trades/NQZ6"
    second = f"{initial}?cursor=safe"
    later = {
        "ticker": "NQZ6",
        "timestamp": start_ns + 20,
        "session_end_date": "2026-09-18",
        "price": 20500.25,
        "size": 1,
        "sequence_number": 2,
    }
    earlier = {**later, "timestamp": start_ns + 10, "sequence_number": 1, "price": 20500.0}
    provider, transport = _provider(
        {
            initial: _response([later], next_url=second),
            second: _response([earlier, later], request_id="req-2"),
        },
        bindings=(NQ,),
    )
    request = HistoricalMarketDataRequest(
        (NQ_ID,), START, END, frozenset({MarketEventKind.TRADE})
    )

    events = _collect(provider, request)

    assert [event.sequence for event in events] == [1, 2]
    assert transport.calls[1] == (second, {})

    unsafe_provider, unsafe_transport = _provider(
        {initial: _response([], next_url="https://attacker.example/futures/v1/trades/NQZ6")},
        bindings=(NQ,),
    )
    with pytest.raises(MarketDataError, match="escaped"):
        _collect(unsafe_provider, request)
    assert len(unsafe_transport.calls) == 1


def test_bar_overlapping_cutoff_is_not_exposed_as_complete():
    start_ns = datetime_to_epoch_ns(START)
    endpoint = "https://api.massive.com/futures/v1/aggs/NQZ6"
    provider, _ = _provider(
        {
            endpoint: _response(
                [
                    {
                        "ticker": "NQZ6",
                        "window_start": start_ns + 90_000_000_000,
                        "session_end_date": "2026-09-18",
                        "open": 20500,
                        "high": 20501,
                        "low": 20499,
                        "close": 20500.25,
                        "volume": 1,
                        "transactions": 1,
                    }
                ]
            )
        },
        bindings=(NQ,),
    )
    request = HistoricalMarketDataRequest(
        (NQ_ID,), START, END, frozenset({MarketEventKind.BAR}), ("1min",)
    )

    assert _collect(provider, request) == []


def test_bare_contract_binding_and_variable_bar_resolution_are_rejected():
    with pytest.raises(ValueError, match="EXCHANGE:PRODUCT"):
        ProviderContractBinding("NQ", "NQ", "NQZ6", "XCME")
    with pytest.raises(ValueError, match="product"):
        ProviderContractBinding(ES_ID, "NQ", "NQZ6", "XCME")
    with pytest.raises(ValueError, match="EXCHANGE:PRODUCT"):
        HistoricalMarketDataRequest(
            ("not-an-exact-contract",), START, END, frozenset({MarketEventKind.TRADE})
        )
    with pytest.raises(ValueError, match="fixed"):
        HistoricalMarketDataRequest(
            (NQ_ID,), START, END, frozenset({MarketEventKind.BAR}), ("1session",)
        )
