from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, AsyncIterator, Protocol
from urllib.parse import parse_qsl, quote, urljoin, urlparse

from app.v2.market_data import (
    AvailabilityMode,
    BarPayload,
    BboPayload,
    ContractReference,
    DataQuality,
    FuturesMarketDataProvider,
    FuturesSchedule,
    HistoricalMarketDataRequest,
    MarketDataCapability,
    MarketDataEntitlement,
    MarketDataEntitlementError,
    MarketDataError,
    MarketDataEvent,
    MarketDataLineage,
    MarketEventKind,
    ProviderContractBinding,
    ScheduleEvent,
    TradePayload,
    UnsupportedMarketDataCapability,
    datetime_to_epoch_ns,
    epoch_ns_to_datetime,
    fixed_resolution_ns,
    market_event_sort_key,
)


MASSIVE_FUTURES_BASE_URL = "https://api.massive.com"
MASSIVE_SOURCE = "massive-futures-rest/v1"
MASSIVE_SCHEMA_VERSION = "futures-v1"


class AsyncJsonTransport(Protocol):
    """Authentication, HTTP lifecycle, and retry policy are injected."""

    async def get_json(self, url: str, *, params: Mapping[str, str]) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class _ProviderRow:
    value: Mapping[str, Any]
    request_id: str | None
    retrieved_at: datetime
    endpoint: str


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MassiveFuturesProvider(FuturesMarketDataProvider):
    """Historical Massive Futures REST adapter.

    The adapter does not own credentials and performs no implicit network or
    demo fallback. A caller must provide both an authenticated transport and an
    explicit entitlement profile.
    """

    SUPPORTED_CAPABILITIES = frozenset(
        {
            MarketDataCapability.CONTRACT_REFERENCE,
            MarketDataCapability.SCHEDULES,
            MarketDataCapability.TRADES,
            MarketDataCapability.BARS,
            MarketDataCapability.BBO,
            MarketDataCapability.HISTORICAL_REPLAY,
        }
    )

    def __init__(
        self,
        *,
        transport: AsyncJsonTransport,
        bindings: Sequence[ProviderContractBinding],
        entitlement: MarketDataEntitlement,
        clock: Callable[[], datetime] = _utc_now,
        base_url: str = MASSIVE_FUTURES_BASE_URL,
        max_pages: int = 10_000,
    ) -> None:
        if transport is None:
            raise ValueError("an authenticated Massive transport is required")
        if not bindings:
            raise ValueError("at least one exact-contract binding is required")
        unsupported = set(entitlement.capabilities) - set(self.SUPPORTED_CAPABILITIES)
        if unsupported:
            names = ", ".join(sorted(item.value for item in unsupported))
            raise MarketDataEntitlementError(f"Massive REST adapter cannot grant: {names}")
        if max_pages < 1:
            raise ValueError("max_pages must be positive")

        parsed_base = urlparse(base_url.rstrip("/"))
        if (
            parsed_base.scheme != "https"
            or not parsed_base.hostname
            or parsed_base.username
            or parsed_base.password
            or parsed_base.query
            or parsed_base.fragment
        ):
            raise ValueError("Massive base URL must be a credential-free HTTPS origin")

        by_id: dict[str, ProviderContractBinding] = {}
        provider_symbols: set[str] = set()
        for binding in bindings:
            if binding.contract_id in by_id:
                raise ValueError(f"duplicate canonical contract binding: {binding.contract_id}")
            normalized_symbol = binding.provider_symbol.upper()
            if normalized_symbol in provider_symbols:
                raise ValueError(f"provider symbol is bound more than once: {binding.provider_symbol}")
            by_id[binding.contract_id] = binding
            provider_symbols.add(normalized_symbol)

        self._transport = transport
        self._bindings = by_id
        self._entitlement = entitlement
        self._clock = clock
        self._base_url = base_url.rstrip("/")
        self._base = parsed_base
        self._max_pages = max_pages

    def capabilities(self) -> frozenset[MarketDataCapability]:
        if not self._entitlement.enabled:
            return frozenset()
        return frozenset(self._entitlement.capabilities & self.SUPPORTED_CAPABILITIES)

    async def contract_reference(self, contract_id: str, *, as_of: date | None = None) -> ContractReference:
        self._entitlement.require({MarketDataCapability.CONTRACT_REFERENCE})
        binding = self._binding(contract_id)
        params = {
            "ticker": binding.provider_symbol,
            "product_code": binding.product_code,
            "type": "single",
            "limit": "2",
            "sort": "product_code.asc",
        }
        if as_of is not None:
            params["date"] = as_of.isoformat()
        endpoint = "/futures/v1/contracts"
        rows = await self._collect(endpoint, params)
        matching = [row for row in rows if self._reference_matches(row.value, binding)]
        if len(matching) != 1:
            raise MarketDataError(
                f"Massive reference lookup for {contract_id} returned {len(matching)} exact matches"
            )
        row = matching[0].value
        contract_type = row.get("type")
        if contract_type not in (None, "single"):
            raise MarketDataError("Massive exact-contract reference unexpectedly describes a combo")
        venue = _required_str(row, "trading_venue")
        if binding.trading_venue and venue.upper() != binding.trading_venue.upper():
            raise MarketDataError("Massive contract venue does not match its configured binding")
        return ContractReference(
            contract_id=binding.contract_id,
            product_code=binding.product_code,
            provider_symbol=binding.provider_symbol,
            trading_venue=venue,
            first_trade_date=_optional_date(row.get("first_trade_date"), "first_trade_date"),
            last_trade_date=_optional_date(row.get("last_trade_date"), "last_trade_date"),
            settlement_date=_optional_date(row.get("settlement_date"), "settlement_date"),
            trade_tick_size=_required_decimal(row, "trade_tick_size"),
            settlement_tick_size=_optional_decimal(row.get("settlement_tick_size"), "settlement_tick_size"),
            spread_tick_size=_optional_decimal(row.get("spread_tick_size"), "spread_tick_size"),
            active_as_of=_optional_date(row.get("date"), "date"),
            source=MASSIVE_SOURCE,
        )

    async def schedule(self, contract_id: str, trade_date: date) -> FuturesSchedule:
        self._entitlement.require({MarketDataCapability.SCHEDULES})
        binding = self._binding(contract_id)
        params = {
            "product_code": binding.product_code,
            "session_end_date": trade_date.isoformat(),
            "limit": "1000",
            "sort": "timestamp.asc",
        }
        if binding.trading_venue:
            params["trading_venue"] = binding.trading_venue
        endpoint = "/futures/v1/schedules"
        rows = await self._collect(endpoint, params)
        events: dict[tuple[datetime, str], ScheduleEvent] = {}
        venue: str | None = None
        retrieved_at: datetime | None = None
        for provider_row in rows:
            row = provider_row.value
            if _required_str(row, "product_code").upper() != binding.product_code.upper():
                raise MarketDataError("Massive schedule product does not match the requested contract")
            if _required_date(row, "session_end_date") != trade_date:
                raise MarketDataError("Massive schedule trade date is outside the requested session")
            row_venue = _required_str(row, "trading_venue")
            if binding.trading_venue and row_venue.upper() != binding.trading_venue.upper():
                raise MarketDataError("Massive schedule venue does not match its configured binding")
            if venue is not None and venue.upper() != row_venue.upper():
                raise MarketDataError("Massive schedule response mixes trading venues")
            venue = row_venue
            timestamp = _required_iso_datetime(row, "timestamp")
            event = _required_str(row, "event").lower()
            normalized = ScheduleEvent(event=event, timestamp=timestamp)
            events[(timestamp, event)] = normalized
            if retrieved_at is None or provider_row.retrieved_at > retrieved_at:
                retrieved_at = provider_row.retrieved_at
        if not events or venue is None or retrieved_at is None:
            raise MarketDataError(f"Massive returned no schedule facts for {contract_id} on {trade_date}")
        ordered = tuple(sorted(events.values(), key=lambda item: (item.timestamp, item.event)))
        return FuturesSchedule(
            contract_id=binding.contract_id,
            product_code=binding.product_code,
            trade_date=trade_date,
            trading_venue=venue,
            events=ordered,
            source=MASSIVE_SOURCE,
            retrieved_at=retrieved_at,
        )

    async def replay(self, request: HistoricalMarketDataRequest) -> AsyncIterator[MarketDataEvent]:
        required = {MarketDataCapability.HISTORICAL_REPLAY}
        kind_capabilities = {
            MarketEventKind.BAR: MarketDataCapability.BARS,
            MarketEventKind.TRADE: MarketDataCapability.TRADES,
            MarketEventKind.BBO: MarketDataCapability.BBO,
        }
        required.update(kind_capabilities[kind] for kind in request.kinds)
        self._entitlement.require(required)
        now = self._now()
        self._entitlement.require_historical_window(request.start, request.end, now=now)
        for contract_id in request.contract_ids:
            self._binding(contract_id)

        events: dict[str, MarketDataEvent] = {}
        for contract_id in request.contract_ids:
            binding = self._binding(contract_id)
            if MarketEventKind.BAR in request.kinds:
                for resolution in request.bar_resolutions:
                    for event in await self._bars(binding, request, resolution):
                        _add_unique(events, event)
            if MarketEventKind.TRADE in request.kinds:
                for event in await self._trades(binding, request):
                    _add_unique(events, event)
            if MarketEventKind.BBO in request.kinds:
                for event in await self._quotes(binding, request):
                    _add_unique(events, event)

        for event in sorted(events.values(), key=market_event_sort_key):
            yield event

    def stream(
        self, contract_ids: tuple[str, ...], kinds: frozenset[MarketEventKind]
    ) -> AsyncIterator[MarketDataEvent]:
        del contract_ids, kinds
        raise UnsupportedMarketDataCapability("Massive REST adapter does not support live streaming")

    async def _bars(
        self,
        binding: ProviderContractBinding,
        request: HistoricalMarketDataRequest,
        resolution: str,
    ) -> list[MarketDataEvent]:
        path_symbol = quote(binding.provider_symbol, safe="")
        endpoint = f"/futures/v1/aggs/{path_symbol}"
        start_ns = datetime_to_epoch_ns(request.start)
        end_ns = datetime_to_epoch_ns(request.end)
        params = {
            "resolution": resolution,
            "window_start.gte": str(start_ns),
            "window_start.lt": str(end_ns),
            "limit": "50000",
            "sort": "window_start.asc",
        }
        rows = await self._collect(endpoint, params)
        duration_ns = fixed_resolution_ns(resolution)
        events: list[MarketDataEvent] = []
        for provider_row in rows:
            row = provider_row.value
            self._require_ticker(row, binding)
            window_start_ns = _required_int(row, "window_start")
            if not start_ns <= window_start_ns < end_ns:
                raise MarketDataError("Massive aggregate lies outside the requested start-time range")
            window_end_ns = window_start_ns + duration_ns
            # A bar ending after the cutoff is not yet knowable and must not leak.
            if window_end_ns > end_ns:
                continue
            payload = BarPayload(
                resolution=resolution,
                starts_at=epoch_ns_to_datetime(window_start_ns),
                ends_at=epoch_ns_to_datetime(window_end_ns),
                window_start_ns=window_start_ns,
                window_end_ns=window_end_ns,
                trade_date=_required_date(row, "session_end_date"),
                open=_required_decimal(row, "open"),
                high=_required_decimal(row, "high"),
                low=_required_decimal(row, "low"),
                close=_required_decimal(row, "close"),
                volume=_required_int(row, "volume"),
                transactions=_required_int(row, "transactions"),
            )
            event_id = f"massive:{binding.provider_symbol}:bar:{resolution}:{window_start_ns}"
            events.append(
                self._event(
                    binding=binding,
                    kind=MarketEventKind.BAR,
                    payload=payload,
                    source_timestamp_ns=window_end_ns,
                    sequence=None,
                    event_id=event_id,
                    provider_row=provider_row,
                )
            )
        return events

    async def _trades(
        self, binding: ProviderContractBinding, request: HistoricalMarketDataRequest
    ) -> list[MarketDataEvent]:
        path_symbol = quote(binding.provider_symbol, safe="")
        endpoint = f"/futures/v1/trades/{path_symbol}"
        start_ns = datetime_to_epoch_ns(request.start)
        end_ns = datetime_to_epoch_ns(request.end)
        rows = await self._collect(
            endpoint,
            {
                "timestamp.gte": str(start_ns),
                "timestamp.lt": str(end_ns),
                "limit": "50000",
                "sort": "timestamp.asc",
            },
        )
        events: list[MarketDataEvent] = []
        for provider_row in rows:
            row = provider_row.value
            self._require_ticker(row, binding)
            timestamp_ns = _required_int(row, "timestamp")
            _require_in_range(timestamp_ns, start_ns, end_ns, "trade")
            sequence = _required_int(row, "sequence_number")
            payload = TradePayload(
                occurred_at=epoch_ns_to_datetime(timestamp_ns),
                occurred_at_ns=timestamp_ns,
                trade_date=_required_date(row, "session_end_date"),
                price=_required_decimal(row, "price"),
                size=_required_int(row, "size"),
                report_sequence=_optional_int(row.get("report_sequence"), "report_sequence"),
                channel=_optional_int(row.get("channel"), "channel"),
            )
            event_id = f"massive:{binding.provider_symbol}:trade:{timestamp_ns}:{sequence}"
            events.append(
                self._event(
                    binding=binding,
                    kind=MarketEventKind.TRADE,
                    payload=payload,
                    source_timestamp_ns=timestamp_ns,
                    sequence=sequence,
                    event_id=event_id,
                    provider_row=provider_row,
                )
            )
        return events

    async def _quotes(
        self, binding: ProviderContractBinding, request: HistoricalMarketDataRequest
    ) -> list[MarketDataEvent]:
        path_symbol = quote(binding.provider_symbol, safe="")
        endpoint = f"/futures/v1/quotes/{path_symbol}"
        start_ns = datetime_to_epoch_ns(request.start)
        end_ns = datetime_to_epoch_ns(request.end)
        rows = await self._collect(
            endpoint,
            {
                "timestamp.gte": str(start_ns),
                "timestamp.lt": str(end_ns),
                "limit": "49999",
                "sort": "timestamp.asc",
            },
        )
        events: list[MarketDataEvent] = []
        for provider_row in rows:
            row = provider_row.value
            self._require_ticker(row, binding)
            timestamp_ns = _required_int(row, "timestamp")
            _require_in_range(timestamp_ns, start_ns, end_ns, "quote")
            sequence = _required_int(row, "sequence_number")
            payload = BboPayload(
                occurred_at=epoch_ns_to_datetime(timestamp_ns),
                occurred_at_ns=timestamp_ns,
                trade_date=_required_date(row, "session_end_date"),
                bid_price=_optional_decimal(row.get("bid_price"), "bid_price"),
                bid_size=_optional_int(row.get("bid_size"), "bid_size"),
                ask_price=_optional_decimal(row.get("ask_price"), "ask_price"),
                ask_size=_optional_int(row.get("ask_size"), "ask_size"),
                bid_timestamp_ns=_optional_int(row.get("bid_timestamp"), "bid_timestamp"),
                ask_timestamp_ns=_optional_int(row.get("ask_timestamp"), "ask_timestamp"),
                report_sequence=_optional_int(row.get("report_sequence"), "report_sequence"),
                channel=_optional_int(row.get("channel"), "channel"),
            )
            event_id = f"massive:{binding.provider_symbol}:bbo:{timestamp_ns}:{sequence}"
            events.append(
                self._event(
                    binding=binding,
                    kind=MarketEventKind.BBO,
                    payload=payload,
                    source_timestamp_ns=timestamp_ns,
                    sequence=sequence,
                    event_id=event_id,
                    provider_row=provider_row,
                )
            )
        return events

    def _event(
        self,
        *,
        binding: ProviderContractBinding,
        kind: MarketEventKind,
        payload: BarPayload | TradePayload | BboPayload,
        source_timestamp_ns: int,
        sequence: int | None,
        event_id: str,
        provider_row: _ProviderRow,
    ) -> MarketDataEvent:
        lineage = MarketDataLineage(
            provider="massive",
            provider_symbol=binding.provider_symbol,
            endpoint=provider_row.endpoint,
            schema_version=MASSIVE_SCHEMA_VERSION,
            request_id=provider_row.request_id,
            retrieved_at=provider_row.retrieved_at,
            revision=0,
        )
        return MarketDataEvent(
            contract_id=binding.contract_id,
            kind=kind,
            source=MASSIVE_SOURCE,
            source_timestamp=epoch_ns_to_datetime(source_timestamp_ns),
            received_timestamp=provider_row.retrieved_at,
            sequence=sequence,
            payload=payload,
            quality=DataQuality(),
            available_timestamp=provider_row.retrieved_at,
            availability_mode=AvailabilityMode.FINALIZED_HISTORICAL,
            revision=0,
            is_final=True,
            source_timestamp_ns=source_timestamp_ns,
            provider_event_id=event_id,
            lineage=lineage,
        )

    async def _collect(self, endpoint: str, params: Mapping[str, str]) -> list[_ProviderRow]:
        current_url = self._safe_url(endpoint)
        current_params: Mapping[str, str] = dict(params)
        visited: set[str] = set()
        output: list[_ProviderRow] = []
        pages = 0
        while True:
            if current_url in visited:
                raise MarketDataError("Massive pagination cycle detected")
            visited.add(current_url)
            pages += 1
            if pages > self._max_pages:
                raise MarketDataError("Massive pagination exceeded the configured page ceiling")

            response = await self._transport.get_json(current_url, params=current_params)
            retrieved_at = self._now()
            if not isinstance(response, Mapping) or response.get("status") != "OK":
                raise MarketDataError("Massive returned a non-OK or malformed response")
            request_id_value = response.get("request_id")
            request_id = str(request_id_value) if request_id_value is not None else None
            results = response.get("results", [])
            if not isinstance(results, list):
                raise MarketDataError("Massive response results must be a list")
            for value in results:
                if not isinstance(value, Mapping):
                    raise MarketDataError("Massive response contains a non-object result")
                output.append(_ProviderRow(dict(value), request_id, retrieved_at, endpoint))

            next_url = response.get("next_url")
            if next_url in (None, ""):
                return output
            if not isinstance(next_url, str):
                raise MarketDataError("Massive next_url must be a string")
            current_url = self._safe_url(next_url)
            current_params = {}

    def _safe_url(self, value: str) -> str:
        absolute = urljoin(f"{self._base_url}/", value)
        parsed = urlparse(absolute)
        try:
            parsed_port = parsed.port
            base_port = self._base.port
        except ValueError as exc:
            raise MarketDataError("Massive pagination URL contained an invalid port") from exc
        if (
            parsed.scheme != self._base.scheme
            or parsed.hostname is None
            or parsed.hostname.lower() != self._base.hostname.lower()
            or parsed_port != base_port
            or parsed.username
            or parsed.password
            or parsed.fragment
            or not parsed.path.startswith("/futures/")
        ):
            raise MarketDataError("Massive pagination URL escaped the configured futures API origin")
        if any(key.lower() == "apikey" for key, _ in parse_qsl(parsed.query, keep_blank_values=True)):
            raise MarketDataError("Massive pagination URL unexpectedly contained credential material")
        return absolute

    def _binding(self, contract_id: str) -> ProviderContractBinding:
        try:
            return self._bindings[contract_id]
        except KeyError as exc:
            raise MarketDataError(f"no Massive exact-contract binding for {contract_id}") from exc

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("provider clock must return a timezone-aware datetime")
        return value

    @staticmethod
    def _reference_matches(row: Mapping[str, Any], binding: ProviderContractBinding) -> bool:
        ticker = row.get("ticker")
        product_code = row.get("product_code")
        return (
            isinstance(ticker, str)
            and ticker.upper() == binding.provider_symbol.upper()
            and isinstance(product_code, str)
            and product_code.upper() == binding.product_code.upper()
        )

    @staticmethod
    def _require_ticker(row: Mapping[str, Any], binding: ProviderContractBinding) -> None:
        if _required_str(row, "ticker").upper() != binding.provider_symbol.upper():
            raise MarketDataError("Massive event ticker does not match the requested exact contract")


def _add_unique(events: dict[str, MarketDataEvent], event: MarketDataEvent) -> None:
    if not event.provider_event_id:
        raise MarketDataError("normalized provider event lacks a stable identity")
    previous = events.get(event.provider_event_id)
    if previous is not None:
        same_observation = (
            previous.contract_id == event.contract_id
            and previous.kind == event.kind
            and previous.source_timestamp_ns == event.source_timestamp_ns
            and previous.sequence == event.sequence
            and previous.revision == event.revision
            and previous.payload == event.payload
        )
        if not same_observation:
            raise MarketDataError("Massive returned conflicting revisions without revision metadata")
        # Page boundaries can repeat the final row. Keep the first occurrence so
        # request lineage remains deterministic without treating it as a new fact.
        return
    events[event.provider_event_id] = event


def _required_str(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise MarketDataError(f"Massive field {field} must be a non-empty string")
    return value.strip()


def _required_int(row: Mapping[str, Any], field: str) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise MarketDataError(f"Massive field {field} must be an integer")
    return value


def _optional_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise MarketDataError(f"Massive field {field} must be an integer when present")
    return value


def _to_decimal(value: Any, field: str) -> Decimal:
    if value is None or isinstance(value, bool):
        raise MarketDataError(f"Massive field {field} must be numeric")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise MarketDataError(f"Massive field {field} must be numeric") from exc
    if not result.is_finite():
        raise MarketDataError(f"Massive field {field} must be finite")
    return result


def _required_decimal(row: Mapping[str, Any], field: str) -> Decimal:
    return _to_decimal(row.get(field), field)


def _optional_decimal(value: Any, field: str) -> Decimal | None:
    if value is None:
        return None
    return _to_decimal(value, field)


def _required_date(row: Mapping[str, Any], field: str) -> date:
    value = _required_str(row, field)
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise MarketDataError(f"Massive field {field} must be an ISO date") from exc


def _optional_date(value: Any, field: str) -> date | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise MarketDataError(f"Massive field {field} must be an ISO date when present")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise MarketDataError(f"Massive field {field} must be an ISO date") from exc


def _required_iso_datetime(row: Mapping[str, Any], field: str) -> datetime:
    value = _required_str(row, field)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MarketDataError(f"Massive field {field} must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise MarketDataError(f"Massive field {field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _require_in_range(value: int, start: int, end: int, kind: str) -> None:
    if not start <= value < end:
        raise MarketDataError(f"Massive {kind} lies outside the requested half-open interval")
