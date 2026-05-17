from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal


OrderSide = Literal["BUY", "SELL"]
OrderType = Literal["MARKET", "LIMIT"]
TimeInForce = Literal["DAY", "GTC"]
TradingSession = Literal["CORE", "EXTENDED"]


@dataclass(frozen=True)
class WebullCredentials:
    app_key: str
    app_secret: str
    endpoint: str
    account_id: str | None = None
    region_id: str = "us"
    environment: str = "test"


@dataclass(frozen=True)
class WebullEquityOrder:
    symbol: str
    side: OrderSide
    quantity: float
    client_order_id: str
    order_type: OrderType = "MARKET"
    time_in_force: TimeInForce = "DAY"
    trading_session: TradingSession = "CORE"
    limit_price: float | None = None
    market: str = "US"

    def to_webull_payload(self) -> dict:
        payload = {
            "combo_type": "NORMAL",
            "client_order_id": self.client_order_id,
            "symbol": self.symbol.upper(),
            "instrument_type": "EQUITY",
            "market": self.market,
            "order_type": self.order_type,
            "quantity": _format_decimal(self.quantity),
            "support_trading_session": self.trading_session,
            "side": self.side,
            "time_in_force": self.time_in_force,
            "entrust_type": "QTY",
        }
        if self.order_type == "LIMIT":
            if self.limit_price is None:
                raise ValueError("limit_price is required for LIMIT orders")
            payload["limit_price"] = _format_decimal(self.limit_price)
        return payload


@dataclass(frozen=True)
class MasterExecutionEvent:
    broker: str
    account_id: str | None
    execution_id: str
    order_id: str | None
    client_order_id: str | None
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    executed_at: datetime
    raw_payload: dict

    @classmethod
    def from_webull_payload(cls, payload: dict) -> "MasterExecutionEvent":
        execution_id = _first_value(payload, ["execution_id", "exec_id", "fill_id", "event_id", "order_id", "client_order_id"])
        symbol = _first_value(payload, ["symbol", "ticker"])
        side = str(_first_value(payload, ["side", "order_side"]) or "").upper()
        quantity = _first_value(payload, ["filled_qty", "filled_quantity", "last_filled_qty", "quantity", "qty"])
        price = _first_value(payload, ["avg_fill_price", "filled_price", "last_filled_price", "price"])
        executed_at = _parse_ts(_first_value(payload, ["executed_at", "filled_at", "updated_at", "timestamp", "ts"]))
        if not execution_id:
            raise ValueError("Webull event payload is missing execution/order identifier")
        if not symbol:
            raise ValueError("Webull event payload is missing symbol")
        if side not in {"BUY", "SELL"}:
            raise ValueError(f"Unsupported Webull side: {side}")
        if quantity is None or float(quantity) <= 0:
            raise ValueError("Webull event payload is missing positive filled quantity")
        if price is None or float(price) <= 0:
            raise ValueError("Webull event payload is missing positive fill price")
        return cls(
            broker="webull",
            account_id=_first_value(payload, ["account_id", "accountId"]),
            execution_id=str(execution_id),
            order_id=_optional_str(_first_value(payload, ["order_id", "orderId"])),
            client_order_id=_optional_str(_first_value(payload, ["client_order_id", "clientOrderId"])),
            symbol=str(symbol).upper(),
            side=side,  # type: ignore[arg-type]
            quantity=float(quantity),
            price=float(price),
            executed_at=executed_at,
            raw_payload=payload,
        )


def _format_decimal(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.8f}".rstrip("0").rstrip(".")


def _first_value(payload: dict, keys: list[str]):
    for key in keys:
        if key in payload and payload[key] not in (None, ""):
            return payload[key]
    return None


def _optional_str(value) -> str | None:
    return str(value) if value not in (None, "") else None


def _parse_ts(value) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        # Accept seconds or milliseconds.
        timestamp = float(value) / 1000 if float(value) > 10_000_000_000 else float(value)
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    if isinstance(value, str) and value:
        normalized = value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(tz=timezone.utc)
