from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from app.v2.capture.config import CaptureConfig
from app.v2.capture.contracts import (
    CapturePlant,
    ObservedAccount,
    PlantHealth,
    RecoveryResult,
)
from app.v2.capture.journal import InMemoryCaptureJournal
from app.v2.capture.main import app
from app.v2.capture.reference_api import (
    ReferenceQueryConfig,
    ReferenceQueryController,
    ReferenceQueryUnavailable,
    validated_code,
    validated_search_text,
    validated_tick_size_type,
)
from app.v2.capture.service import RithmicCaptureService


TOKEN = "synthetic-reference-query-token-00000001"


class DurableJournal(InMemoryCaptureJournal):
    durable = True


class QueryObserver:
    def __init__(self) -> None:
        self.sink = None
        self.calls: list[tuple[str, ...]] = []
        self.health = {
            plant: PlantHealth(
                plant=plant,
                connected=True,
                authenticated=True,
                generation_id=f"{plant.value.lower()}-generation",
            )
            for plant in (CapturePlant.ORDER, CapturePlant.PNL, CapturePlant.TICKER)
        }

    async def start(self, sink) -> None:
        self.sink = sink

    async def stop(self) -> None:
        return None

    async def discover_accounts(self):
        return (ObservedAccount("allowed-account"),)

    async def prepare_reconciliation(self, account_id, generations) -> None:
        del account_id, generations

    async def subscribe_account(self, account_id, plant) -> None:
        self.calls.append(("subscribe", account_id, plant.value))

    async def reconcile_account(self, account_id, generations):
        del generations
        return RecoveryResult(clean=True, checkpoint=f"clean-{account_id}")

    async def apply_buffered(self, account_id, events) -> None:
        del account_id, events

    async def finalize_reconciliation(self, account_id, generations):
        del generations
        return RecoveryResult(clean=True, checkpoint=f"clean-{account_id}")

    async def abort_recovery(self, generations) -> None:
        del generations

    def plant_health(self):
        return dict(self.health)

    async def search_symbols(self, search_text, *, exchange=None, product_code=None):
        self.calls.append(("search", search_text, exchange or "", product_code or ""))
        return (
            {
                "symbol": "MNQZ6",
                "exchange": "CME",
                "symbol_name": "Micro E-mini Nasdaq-100",
                "product_code": "MNQ",
                "instrument_type": "Future",
                "expiration_date": "20261218",
                "account_id": "must-not-leak",
                "user_msg": ["must-not-leak"],
            },
            {"symbol": "MNQH7", "exchange": "CME"},
        )

    async def reference_data(self, symbol, exchange):
        self.calls.append(("reference", symbol, exchange))
        return {
            "symbol": symbol,
            "exchange": exchange,
            "trading_symbol": symbol,
            "tick_size_type": "SIMPLE",
            "minimum_quoted_price_change": Decimal("0.25"),
            "single_point_value": Decimal("2"),
            "is_tradable": "1",
            "password": "must-not-leak",
        }

    async def tick_size_table(self, tick_size_type):
        self.calls.append(("ticks", tick_size_type))
        return (
            {
                "tick_size_type": tick_size_type,
                "minimum_feed_price_change": Decimal("0.25"),
                "first_price": Decimal("0"),
                "last_price": Decimal("1000000"),
                "first_price_operator": "GREATER_THAN_OR_EQUAL_TO",
                "last_price_operator": "LESS_THAN",
                "fcm_id": "must-not-leak",
            },
        )


def capture_config(*, ticker: bool = True) -> CaptureConfig:
    plants = {CapturePlant.ORDER, CapturePlant.PNL}
    if ticker:
        plants.add(CapturePlant.TICKER)
    return CaptureConfig(
        connectivity_enabled=True,
        environment="TEST",
        account_allowlist=frozenset({"allowed-account"}),
        enabled_plants=frozenset(plants),
        observer_factory="app.v2.brokers.rithmic_protocol.adapter:create_observer",
        journal_factory="app.v2.capture.persistence:create_journal",
        poll_seconds=0.01,
        reconcile_timeout_seconds=1,
        bindings_configured=True,
    )


def query_config(*, enabled: bool = True, token: str = TOKEN) -> ReferenceQueryConfig:
    return ReferenceQueryConfig.from_mapping(
        {
            "RITHMIC_REFERENCE_QUERY_ENABLED": "1" if enabled else "0",
            "RITHMIC_REFERENCE_QUERY_TOKEN": token,
            "RITHMIC_REFERENCE_QUERY_TIMEOUT_SECONDS": "2",
            "RITHMIC_REFERENCE_QUERY_MAX_RESULTS": "50",
        }
    )


def test_query_token_is_never_retained_and_uses_constant_time_digest_compare(monkeypatch):
    calls: list[tuple[bytes, bytes]] = []

    def compare(left: bytes, right: bytes) -> bool:
        calls.append((left, right))
        return left == right

    monkeypatch.setattr("app.v2.capture.reference_api.hmac.compare_digest", compare)
    config = query_config()

    assert TOKEN not in repr(config)
    assert config.authorize(TOKEN)
    assert not config.authorize("wrong")
    assert calls and all(len(left) == len(right) == 32 for left, right in calls)
    assert not query_config(token="too-short").available
    assert not query_config(enabled=False).available


def test_reference_inputs_are_ascii_bounded_and_reject_control_or_path_syntax():
    assert validated_search_text(" MNQ Z6 ") == "MNQ Z6"
    assert validated_code("CME", required=True) == "CME"
    assert validated_tick_size_type("SIMPLE-1") == "SIMPLE-1"
    for value in ("", "../MNQ", "MNQ?password=x", "MNQ\nZ6", "M" * 65):
        with pytest.raises(ValueError):
            validated_tick_size_type(value)


def test_capture_service_requires_configured_authenticated_ticker_for_queries():
    async def scenario():
        observer = QueryObserver()
        service = RithmicCaptureService(
            capture_config(ticker=True), observer=observer, journal=DurableJournal()
        )
        await service.start()
        assert await service.wait_until_ready(timeout=1)
        assert service.ticker_reference_available
        rows = await service.search_reference_symbols(
            "MNQ", exchange="CME", timeout_seconds=1
        )
        assert rows[0]["symbol"] == "MNQZ6"

        observer.health[CapturePlant.TICKER] = PlantHealth(
            plant=CapturePlant.TICKER,
            connected=True,
            authenticated=False,
            generation_id="ticker-generation",
        )
        await asyncio.sleep(0.03)
        assert not service.ticker_reference_available
        with pytest.raises(ConnectionError):
            await service.contract_reference("MNQZ6", "CME", timeout_seconds=1)
        await service.stop()

        no_ticker = RithmicCaptureService(
            capture_config(ticker=False), observer=QueryObserver(), journal=DurableJournal()
        )
        await no_ticker.start()
        assert await no_ticker.wait_until_ready(timeout=1)
        assert not no_ticker.ticker_reference_available
        await no_ticker.stop()

    asyncio.run(scenario())


def test_controller_filters_sensitive_or_protocol_internal_fields_and_bounds_results():
    async def scenario():
        observer = QueryObserver()
        service = RithmicCaptureService(
            capture_config(), observer=observer, journal=DurableJournal()
        )
        await service.start()
        assert await service.wait_until_ready(timeout=1)
        controller = ReferenceQueryController(service, query_config())

        search = await controller.search(
            "MNQ", exchange="CME", product_code="MNQ", limit=1
        )
        assert search["count"] == 1 and search["truncated"] is True
        assert search["results"][0]["symbol"] == "MNQZ6"
        assert "account_id" not in search["results"][0]
        assert "user_msg" not in search["results"][0]

        reference = await controller.reference("MNQZ6", "CME")
        assert reference["minimum_quoted_price_change"] == "0.25"
        assert "password" not in reference

        ticks = await controller.tick_sizes("SIMPLE")
        assert ticks["results"][0]["minimum_feed_price_change"] == "0.25"
        assert "fcm_id" not in ticks["results"][0]
        await service.stop()

    asyncio.run(scenario())


def test_http_reference_routes_are_authenticated_generic_and_get_only():
    async def scenario():
        observer = QueryObserver()
        service = RithmicCaptureService(
            capture_config(), observer=observer, journal=DurableJournal()
        )
        await service.start()
        assert await service.wait_until_ready(timeout=1)
        app.state.capture_service = service
        app.state.reference_queries = ReferenceQueryController(service, query_config())
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://capture") as client:
            missing = await client.get("/reference/symbols", params={"q": "MNQ"})
            assert missing.status_code == 401
            assert missing.json() == {"detail": "unauthorized"}

            wrong = await client.get(
                "/reference/symbols",
                params={"q": "MNQ"},
                headers={"X-Rithmic-Reference-Token": "wrong"},
            )
            assert wrong.status_code == 401

            response = await client.get(
                "/reference/symbols",
                params={"q": "MNQ", "limit": 1},
                headers={"X-Rithmic-Reference-Token": TOKEN},
            )
            assert response.status_code == 200
            assert response.headers["cache-control"] == "no-store"
            assert response.json()["results"][0]["symbol"] == "MNQZ6"

            reference = await client.get(
                "/reference/contracts/MNQZ6",
                params={"exchange": "CME"},
                headers={"X-Rithmic-Reference-Token": TOKEN},
            )
            assert reference.status_code == 200

            ticks = await client.get(
                "/reference/tick-sizes/SIMPLE",
                headers={"X-Rithmic-Reference-Token": TOKEN},
            )
            assert ticks.status_code == 200

            rejected = await client.get(
                "/reference/symbols",
                params={"q": "MNQ?password=secret"},
                headers={"X-Rithmic-Reference-Token": TOKEN},
            )
            assert rejected.status_code == 422
            assert "password=secret" not in rejected.text

            mutation = await client.post(
                "/reference/symbols",
                headers={"X-Rithmic-Reference-Token": TOKEN},
            )
            assert mutation.status_code == 405
        await service.stop()

    asyncio.run(scenario())


def test_http_surface_is_unavailable_when_operator_has_not_explicitly_enabled_it():
    async def scenario():
        class Surface:
            ticker_reference_available = True

        app.state.reference_queries = ReferenceQueryController(
            Surface(), query_config(enabled=False)
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://capture") as client:
            response = await client.get(
                "/reference/symbols",
                params={"q": "MNQ"},
                headers={"X-Rithmic-Reference-Token": TOKEN},
            )
        assert response.status_code == 503
        assert response.json() == {"detail": "reference service unavailable"}

    asyncio.run(scenario())


def test_reference_query_deployment_defaults_remain_disabled_and_secret_backed():
    render = Path("render.yaml").read_text(encoding="utf-8")
    service = render[render.index("name: mooney-rithmic-capture") :]
    assert "RITHMIC_REFERENCE_QUERY_ENABLED" in service
    assert 'value: "0"' in service
    token_block = service[service.index("RITHMIC_REFERENCE_QUERY_TOKEN") :]
    assert "sync: false" in token_block.split("- key:", 1)[0]
    assert "RITHMIC_REFERENCE_QUERY_TOKEN=" in Path(".env.example").read_text(
        encoding="utf-8"
    )
    assert "X-Rithmic-Reference-Token" in Path(
        "docs/RITHMIC_READ_ONLY_RUNBOOK.md"
    ).read_text(encoding="utf-8")
