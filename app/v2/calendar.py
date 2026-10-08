from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo

from app.v2.domain.models import FuturesContract


EASTERN = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class MaintenanceBreak:
    start_at: datetime
    end_at: datetime


@dataclass(frozen=True)
class SessionSchedule:
    trade_date: date
    opens_at: datetime | None
    closes_at: datetime | None
    maintenance_break_after: MaintenanceBreak | None
    holiday: bool = False
    early_close: bool = False
    available_contract_ids: frozenset[str] = field(default_factory=frozenset)
    source: str = "cme-normal-hours"

    def is_open_at(self, timestamp: datetime) -> bool:
        if self.holiday or self.opens_at is None or self.closes_at is None:
            return False
        value = _aware(timestamp).astimezone(EASTERN)
        return self.opens_at <= value < self.closes_at

    def contract_is_available(self, contract: FuturesContract) -> bool:
        if not contract.is_available_on(self.trade_date):
            return False
        return not self.available_contract_ids or contract.contract_id in self.available_contract_ids


@dataclass(frozen=True)
class SessionOverride:
    trade_date: date
    opens_at: datetime | None
    closes_at: datetime | None
    holiday: bool = False
    early_close: bool = False
    available_contract_ids: frozenset[str] = field(default_factory=frozenset)
    source: str = "provider-override"


class TradingCalendar(Protocol):
    def schedule_for(self, trade_date: date) -> SessionSchedule: ...
    def trade_date_at(self, timestamp: datetime) -> date | None: ...


class CmeEquityIndexCalendar:
    """Operational calendar foundation, not a strategy-regime classifier.

    Normal NQ/MNQ hours are represented as 18:00 ET on the prior calendar day
    through 17:00 ET on the trade date. Holiday and early-close facts must be
    supplied as provider overrides; they are intentionally not guessed here.
    """

    def __init__(self, overrides: list[SessionOverride] | None = None) -> None:
        self._overrides = {item.trade_date: item for item in overrides or []}

    def schedule_for(self, trade_date: date) -> SessionSchedule:
        override = self._overrides.get(trade_date)
        if override is not None:
            maintenance = _maintenance_after(override.closes_at)
            return SessionSchedule(
                trade_date=trade_date,
                opens_at=override.opens_at,
                closes_at=override.closes_at,
                maintenance_break_after=maintenance,
                holiday=override.holiday,
                early_close=override.early_close,
                available_contract_ids=override.available_contract_ids,
                source=override.source,
            )
        if trade_date.weekday() >= 5:
            return SessionSchedule(trade_date, None, None, None, holiday=True, source="weekend")
        prior = trade_date - timedelta(days=1)
        opens = datetime.combine(prior, time(18, 0), tzinfo=EASTERN)
        closes = datetime.combine(trade_date, time(17, 0), tzinfo=EASTERN)
        return SessionSchedule(trade_date, opens, closes, _maintenance_after(closes))

    def trade_date_at(self, timestamp: datetime) -> date | None:
        value = _aware(timestamp).astimezone(EASTERN)
        candidates = (value.date(), value.date() + timedelta(days=1))
        for candidate in candidates:
            schedule = self.schedule_for(candidate)
            if schedule.is_open_at(value):
                return candidate
        return None

    def add_override(self, override: SessionOverride) -> None:
        self._overrides[override.trade_date] = override


def _maintenance_after(closes_at: datetime | None) -> MaintenanceBreak | None:
    if closes_at is None:
        return None
    return MaintenanceBreak(closes_at, closes_at + timedelta(hours=1))


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("session timestamps must be timezone-aware")
    return value
