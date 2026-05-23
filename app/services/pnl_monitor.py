from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from typing import Any, Protocol

from sqlalchemy import and_, desc, select

from app.copier.reconciliation import normalize_positions
from app.copier.webull_client import WebullTradingClient
from app.core.config import CopierConfig
from app.core.config_store import CONFIG
from app.db.models import AccountSnapshot, PnlAlert, PositionSnapshot, TradingSession
from app.db.session import get_session
from app.observability.logging import get_logger


logger = get_logger("pnl_monitor")


class BalancePositionClient(Protocol):
    def get_account_balance(self, account_id: str) -> dict[str, Any]: ...
    def get_account_positions(self, account_id: str) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class PnlRiskLimits:
    max_daily_loss: float = 1_000.0
    max_daily_loss_pct: float = 0.05
    max_drawdown: float = 1_500.0
    max_total_exposure: float = 25_000.0
    warning_fraction: float = 0.70


@dataclass(frozen=True)
class MonitoredAccount:
    name: str
    account_ref: str
    account_type: str
    client: BalancePositionClient
    broker: str = "webull"


@dataclass
class AccountStatus:
    account_ref: str
    account_name: str
    account_type: str
    timestamp: datetime
    cash_balance: float | None
    equity_value: float | None
    total_value: float
    buying_power: float | None
    unrealized_pnl: float
    realized_pnl_today: float
    total_pnl_today: float
    max_drawdown_today: float
    max_profit_today: float
    position_count: int
    total_exposure: float
    risk_level: str = "normal"
    alerts: list[str] = field(default_factory=list)
    raw_balance: dict[str, Any] | None = None

    @property
    def is_healthy(self) -> bool:
        return self.risk_level in {"normal", "warning"}


@dataclass(frozen=True)
class NormalizedBalance:
    cash_balance: float | None
    equity_value: float | None
    total_value: float
    buying_power: float | None
    day_trades_used: int | None = None
    day_trades_remaining: int | None = None
    raw_payload: dict[str, Any] | None = None


class PnlMonitor:
    def __init__(
        self,
        *,
        session_scope: Callable = get_session,
        config_provider: Callable[[], CopierConfig] | None = None,
        accounts_provider: Callable[[], list[MonitoredAccount]] | None = None,
        risk_limits: PnlRiskLimits | None = None,
        slack: Any | None = None,
    ) -> None:
        self.session_scope = session_scope
        self.config_provider = config_provider or (lambda: CONFIG.copier)
        self.accounts_provider = accounts_provider
        self.risk_limits = risk_limits or PnlRiskLimits()
        self.slack = slack

    def collect_once(self) -> list[AccountStatus]:
        statuses: list[AccountStatus] = []
        for account in self._monitored_accounts():
            try:
                status = self.collect_account(account)
            except Exception as exc:
                logger.warning("pnl_monitor.account_failed", account=account.name, err=str(exc))
                continue
            statuses.append(status)
        return statuses

    async def collect_once_async(self) -> list[AccountStatus]:
        return self.collect_once()

    def collect_account(self, account: MonitoredAccount) -> AccountStatus:
        raw_balance = account.client.get_account_balance(account.account_ref)
        balance = normalize_account_balance(raw_balance)
        raw_positions = account.client.get_account_positions(account.account_ref)
        positions = normalize_positions(raw_positions)
        now = datetime.now(tz=timezone.utc)
        market_session = classify_market_session(now)

        with self.session_scope() as session:
            trading_session = ensure_trading_session(session, account, balance.total_value, now)
            realized_pnl_today = float(trading_session.realized_pnl or 0.0)
            total_pnl_today = balance.total_value - float(trading_session.starting_value or balance.total_value)
            unrealized_pnl = _positions_unrealized_pnl(positions)
            total_exposure = sum(abs(float(position.market_value or 0.0)) for position in positions)
            position_count = sum(1 for position in positions if abs(position.qty) > 0)
            max_drawdown = min(float(trading_session.max_drawdown or 0.0), total_pnl_today)
            max_profit = max(float(trading_session.max_profit or 0.0), total_pnl_today)
            risk_level, alerts = assess_risk(
                total_pnl_today=total_pnl_today,
                max_drawdown=max_drawdown,
                total_exposure=total_exposure,
                starting_value=float(trading_session.starting_value or balance.total_value),
                limits=self.risk_limits,
            )
            status = AccountStatus(
                account_ref=account.account_ref,
                account_name=account.name,
                account_type=account.account_type,
                timestamp=now,
                cash_balance=balance.cash_balance,
                equity_value=balance.equity_value,
                total_value=balance.total_value,
                buying_power=balance.buying_power,
                unrealized_pnl=unrealized_pnl,
                realized_pnl_today=realized_pnl_today,
                total_pnl_today=total_pnl_today,
                max_drawdown_today=max_drawdown,
                max_profit_today=max_profit,
                position_count=position_count,
                total_exposure=total_exposure,
                risk_level=risk_level,
                alerts=alerts,
                raw_balance=raw_balance,
            )
            persist_pnl_snapshot(session, account, status, balance, positions, market_session)
            trading_session.unrealized_pnl = unrealized_pnl
            trading_session.total_pnl = total_pnl_today
            trading_session.max_drawdown = max_drawdown
            trading_session.max_profit = max_profit
            if risk_level in {"critical", "emergency"}:
                persist_pnl_alert(session, account, status, self.risk_limits)
                self._post_alert(status)
            return status

    def get_latest_status(self, account_ref: str) -> AccountStatus | None:
        with self.session_scope() as session:
            row = (
                session.execute(
                    select(AccountSnapshot)
                    .where(AccountSnapshot.account_ref == account_ref)
                    .order_by(desc(AccountSnapshot.snapshot_time))
                    .limit(1)
                )
                .scalars()
                .first()
            )
            if row is None:
                return None
            return status_from_snapshot(row)

    def _monitored_accounts(self) -> list[MonitoredAccount]:
        if self.accounts_provider is not None:
            return self.accounts_provider()
        return build_monitored_accounts(self.config_provider())

    def _post_alert(self, status: AccountStatus) -> None:
        if self.slack is None:
            return
        text = (
            f"P&L risk alert: {status.account_name} {status.risk_level}\n"
            f"Today P&L: ${status.total_pnl_today:.2f}\n"
            f"Drawdown: ${status.max_drawdown_today:.2f}\n"
            f"Alerts: {', '.join(status.alerts)}"
        )
        post = getattr(self.slack, "post", None)
        if callable(post):
            post(text)


def build_monitored_accounts(config: CopierConfig) -> list[MonitoredAccount]:
    accounts: list[MonitoredAccount] = []
    master_ref = config.master_account or os.getenv(config.master_account_env)
    if master_ref:
        try:
            accounts.append(
                MonitoredAccount(
                    name="master",
                    account_ref=master_ref,
                    account_type="master",
                    client=WebullTradingClient.from_env(
                        app_key_env=config.master_app_key_env,
                        app_secret_env=config.master_app_secret_env,
                        endpoint_env=config.master_endpoint_env,
                        account_id_env=config.master_account_env,
                        environment=config.mode,
                    ),
                )
            )
        except Exception as exc:
            logger.warning("pnl_monitor.master_client_unavailable", err=str(exc))
    for target in config.targets:
        account_ref = target.account_ref or os.getenv(target.account_id_env)
        if not account_ref:
            continue
        try:
            accounts.append(
                MonitoredAccount(
                    name=target.name,
                    account_ref=account_ref,
                    account_type="copy",
                    client=WebullTradingClient.from_env(
                        app_key_env=target.api_key_env,
                        app_secret_env=target.api_secret_env,
                        endpoint_env=target.endpoint_env,
                        account_id_env=target.account_id_env,
                        environment=target.environment,
                    ),
                )
            )
        except Exception as exc:
            logger.warning("pnl_monitor.target_client_unavailable", target=target.name, err=str(exc))
    return accounts


def normalize_account_balance(payload: dict[str, Any]) -> NormalizedBalance:
    data = _balance_data(payload)
    total_value = _first_float(
        data,
        [
            "total_value",
            "totalValue",
            "total_net_liquidation_value",
            "totalNetLiquidationValue",
            "net_liquidation",
            "netLiquidation",
            "net_liquidation_value",
            "netLiquidationValue",
            "net_account_value",
            "netAccountValue",
            "account_value",
            "accountValue",
            "equity",
            "equity_value",
            "equityValue",
            "total_equity",
            "totalEquity",
        ],
    )
    equity_value = _first_float(
        data,
        [
            "equity_value",
            "equityValue",
            "equity",
            "total_equity",
            "totalEquity",
            "total_net_liquidation_value",
            "totalNetLiquidationValue",
            "net_liquidation",
            "netLiquidation",
            "net_liquidation_value",
            "netLiquidationValue",
            "net_account_value",
            "netAccountValue",
        ],
    )
    cash_balance = _first_float(
        data,
        ["cash_balance", "cashBalance", "total_cash_balance", "totalCashBalance", "cash", "cashAvailable"],
    )
    buying_power = _first_float(data, ["buying_power", "buyingPower", "dayTradingBuyingPower", "overnightBuyingPower"])
    if total_value is None:
        total_value = equity_value if equity_value is not None else cash_balance
    if total_value is None:
        raise ValueError(f"Webull balance payload is missing total/equity value: {payload}")
    return NormalizedBalance(
        cash_balance=cash_balance,
        equity_value=equity_value,
        total_value=float(total_value),
        buying_power=buying_power,
        day_trades_used=_first_int(data, ["day_trades_used", "dayTradesUsed"]),
        day_trades_remaining=_first_int(data, ["day_trades_remaining", "dayTradesRemaining"]),
        raw_payload=payload,
    )


def assess_risk(
    *,
    total_pnl_today: float,
    max_drawdown: float,
    total_exposure: float,
    starting_value: float,
    limits: PnlRiskLimits,
) -> tuple[str, list[str]]:
    alerts: list[str] = []
    if total_pnl_today <= -limits.max_daily_loss:
        alerts.append(f"daily loss limit breached: ${total_pnl_today:.2f}")
        return "emergency", alerts
    loss_pct = abs(total_pnl_today / starting_value) if starting_value > 0 and total_pnl_today < 0 else 0.0
    if loss_pct >= limits.max_daily_loss_pct:
        alerts.append(f"daily loss percent breached: {loss_pct:.2%}")
        return "emergency", alerts
    if max_drawdown <= -limits.max_drawdown:
        alerts.append(f"drawdown limit breached: ${max_drawdown:.2f}")
        return "critical", alerts
    if total_exposure >= limits.max_total_exposure:
        alerts.append(f"total exposure limit breached: ${total_exposure:.2f}")
        return "critical", alerts
    if total_pnl_today <= -(limits.max_daily_loss * limits.warning_fraction):
        alerts.append(f"approaching daily loss limit: ${total_pnl_today:.2f}")
        return "warning", alerts
    if max_drawdown <= -(limits.max_drawdown * limits.warning_fraction):
        alerts.append(f"approaching drawdown limit: ${max_drawdown:.2f}")
        return "warning", alerts
    return "normal", alerts


def ensure_trading_session(session, account: MonitoredAccount, starting_value: float, now: datetime) -> TradingSession:
    today = now.date()
    row = (
        session.execute(
            select(TradingSession).where(
                TradingSession.account_ref == account.account_ref,
                TradingSession.trade_date == today,
            )
        )
        .scalars()
        .first()
    )
    if row is not None:
        return row
    row = TradingSession(
        account_ref=account.account_ref,
        account_name=account.name,
        trade_date=today,
        session_start=now,
        active=True,
        starting_value=starting_value,
    )
    session.add(row)
    session.flush()
    return row


def persist_pnl_snapshot(
    session,
    account: MonitoredAccount,
    status: AccountStatus,
    balance: NormalizedBalance,
    positions,
    market_session: str,
) -> None:
    snapshot = AccountSnapshot(
        account_ref=account.account_ref,
        account_name=account.name,
        account_type=account.account_type,
        broker=account.broker,
        snapshot_time=status.timestamp,
        market_session=market_session,
        cash_balance=balance.cash_balance,
        equity_value=balance.equity_value,
        total_value=balance.total_value,
        buying_power=balance.buying_power,
        day_trades_used=balance.day_trades_used,
        day_trades_remaining=balance.day_trades_remaining,
        unrealized_pnl=status.unrealized_pnl,
        realized_pnl_today=status.realized_pnl_today,
        total_pnl_today=status.total_pnl_today,
        max_drawdown_today=status.max_drawdown_today,
        max_profit_today=status.max_profit_today,
        position_count=status.position_count,
        total_exposure=status.total_exposure,
        risk_level=status.risk_level,
        raw_payload=balance.raw_payload,
    )
    session.add(snapshot)
    for position in positions:
        market_value = float(position.market_value or 0.0)
        avg_price = position.avg_price
        cost_basis = float(avg_price or 0.0) * float(position.qty)
        unrealized = market_value - cost_basis
        current_price = market_value / position.qty if position.qty else None
        session.add(
            PositionSnapshot(
                account_ref=account.account_ref,
                account_name=account.name,
                symbol=position.symbol,
                snapshot_time=status.timestamp,
                qty=position.qty,
                avg_price=avg_price,
                current_price=current_price,
                market_value=market_value,
                cost_basis=cost_basis,
                unrealized_pnl=unrealized,
                unrealized_pnl_pct=(unrealized / cost_basis) if cost_basis else None,
                side="short" if position.qty < 0 else "long",
                raw_payload=position.raw_payload,
            )
        )


def persist_pnl_alert(session, account: MonitoredAccount, status: AccountStatus, limits: PnlRiskLimits) -> None:
    message = "; ".join(status.alerts) or f"P&L risk level {status.risk_level}"
    existing = (
        session.execute(
            select(PnlAlert)
            .where(
                PnlAlert.account_ref == account.account_ref,
                PnlAlert.severity == status.risk_level,
                PnlAlert.acknowledged.is_(False),
            )
            .limit(1)
        )
        .scalars()
        .first()
    )
    if existing is not None:
        return
    session.add(
        PnlAlert(
            account_ref=account.account_ref,
            account_name=account.name,
            account_type=account.account_type,
            alert_time=status.timestamp,
            alert_type="risk_threshold",
            severity=status.risk_level,
            threshold_value=limits.max_daily_loss,
            actual_value=status.total_pnl_today,
            account_value=status.total_value,
            unrealized_pnl=status.unrealized_pnl,
            realized_pnl=status.realized_pnl_today,
            message=message,
            raw_context={"alerts": status.alerts, "total_exposure": status.total_exposure},
        )
    )


def status_from_snapshot(row: AccountSnapshot) -> AccountStatus:
    return AccountStatus(
        account_ref=row.account_ref,
        account_name=row.account_name,
        account_type=row.account_type,
        timestamp=row.snapshot_time,
        cash_balance=row.cash_balance,
        equity_value=row.equity_value,
        total_value=row.total_value,
        buying_power=row.buying_power,
        unrealized_pnl=row.unrealized_pnl,
        realized_pnl_today=row.realized_pnl_today,
        total_pnl_today=row.total_pnl_today,
        max_drawdown_today=row.max_drawdown_today,
        max_profit_today=row.max_profit_today,
        position_count=row.position_count,
        total_exposure=row.total_exposure,
        risk_level=row.risk_level,
        alerts=[],
        raw_balance=row.raw_payload,
    )


def latest_account_snapshots(session) -> list[AccountSnapshot]:
    rows = session.execute(select(AccountSnapshot).order_by(desc(AccountSnapshot.snapshot_time))).scalars().all()
    latest: dict[str, AccountSnapshot] = {}
    for row in rows:
        latest.setdefault(row.account_ref, row)
    return list(latest.values())


def latest_position_snapshots(session, account_ref: str) -> list[PositionSnapshot]:
    rows = (
        session.execute(
            select(PositionSnapshot)
            .where(PositionSnapshot.account_ref == account_ref)
            .order_by(desc(PositionSnapshot.snapshot_time))
        )
        .scalars()
        .all()
    )
    latest: dict[str, PositionSnapshot] = {}
    for row in rows:
        if abs(float(row.qty or 0.0)) > 0:
            latest.setdefault(row.symbol, row)
    return list(latest.values())


def classify_market_session(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    t = dt.astimezone().time()
    if time(4, 0) <= t < time(9, 30):
        return "premarket"
    if time(9, 30) <= t <= time(16, 0):
        return "regular"
    if time(16, 0) < t <= time(20, 0):
        return "afterhours"
    return "closed"


def _positions_unrealized_pnl(positions) -> float:
    total = 0.0
    for position in positions:
        market_value = float(position.market_value or 0.0)
        avg_price = float(position.avg_price or 0.0)
        total += market_value - (avg_price * float(position.qty))
    return total


def _payload_data(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("account", "balance", "summary"):
            child = data.get(key)
            if isinstance(child, dict):
                return {**data, **child}
        return data
    return payload


def _balance_data(payload: dict[str, Any]) -> dict[str, Any]:
    data = _payload_data(payload)
    currency_assets = data.get("account_currency_assets") or data.get("accountCurrencyAssets")
    if isinstance(currency_assets, list):
        asset = next((item for item in currency_assets if isinstance(item, dict) and item.get("currency") == "USD"), None)
        if asset is None:
            asset = next((item for item in currency_assets if isinstance(item, dict)), None)
        if isinstance(asset, dict):
            return {**asset, **data}
    return data


def _first_float(payload: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _first_int(payload: dict[str, Any], keys: list[str]) -> int | None:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


_PNL_MONITOR: PnlMonitor | None = None


def get_pnl_monitor() -> PnlMonitor:
    global _PNL_MONITOR
    if _PNL_MONITOR is None:
        _PNL_MONITOR = PnlMonitor()
    return _PNL_MONITOR
