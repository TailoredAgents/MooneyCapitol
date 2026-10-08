from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable
from uuid import uuid4

from .bindings import ExternalBindingRegistry
from .constants import PROTOCOL_TEMPLATE_VERSION, Plant, Template
from .errors import UnauthorizedAccount
from .normalization import BrokerAccountKey
from .transport import ReadOnlyOutboundPolicy


@dataclass(frozen=True)
class LoginCredentials:
    username: str = field(repr=False)
    password: str = field(repr=False)

    def __post_init__(self) -> None:
        if not self.username or not self.password:
            raise ValueError("Rithmic username and password are required")


class AccountAllowlist:
    """Exact FCM/IB/account allowlist; broker identifiers remain opaque."""

    def __init__(self, accounts: Iterable[BrokerAccountKey]) -> None:
        self._accounts = frozenset(accounts)
        if not self._accounts:
            raise ValueError("at least one explicitly allowlisted account is required")

    def allows(self, account: BrokerAccountKey) -> bool:
        return account in self._accounts

    def require(self, account: BrokerAccountKey) -> None:
        if account not in self._accounts:
            raise UnauthorizedAccount("account is not explicitly allowlisted")

    def filter(self, accounts: Iterable[BrokerAccountKey]) -> tuple[BrokerAccountKey, ...]:
        return tuple(account for account in accounts if account in self._accounts)

    @property
    def count(self) -> int:
        return len(self._accounts)


class ReadOnlyMessageFactory:
    """Creates only observation/discovery messages from external bindings."""

    def __init__(
        self,
        registry: ExternalBindingRegistry,
        *,
        account_allowlist: AccountAllowlist | None = None,
        template_version: str = PROTOCOL_TEMPLATE_VERSION,
        policy: ReadOnlyOutboundPolicy | None = None,
    ) -> None:
        self.registry = registry
        self.account_allowlist = account_allowlist
        self.template_version = template_version
        self.policy = policy or ReadOnlyOutboundPolicy()

    def _build(self, template: Template, **fields: Any) -> Any:
        self.policy.assert_allowed(int(template))
        return self.registry.build(int(template), **fields)

    @staticmethod
    def correlation_id(prefix: str) -> str:
        return f"{prefix}-{uuid4().hex}"

    def _account_fields(self, account: BrokerAccountKey) -> dict[str, str]:
        if self.account_allowlist is None:
            raise UnauthorizedAccount("account-scoped requests require an explicit allowlist")
        self.account_allowlist.require(account)
        return {"fcm_id": account.fcm_id, "ib_id": account.ib_id, "account_id": account.account_id}

    def system_info(self, *, correlation_id: str | None = None) -> Any:
        return self._build(
            Template.SYSTEM_INFO_REQUEST,
            user_msg=[correlation_id or self.correlation_id("system-info")],
        )

    def gateway_info(self, system_name: str, *, correlation_id: str | None = None) -> Any:
        return self._build(
            Template.GATEWAY_INFO_REQUEST,
            system_name=system_name,
            user_msg=[correlation_id or self.correlation_id("gateway-info")],
        )

    def login(
        self,
        credentials: LoginCredentials,
        *,
        plant: Plant,
        system_name: str,
        app_name: str,
        app_version: str,
        correlation_id: str | None = None,
    ) -> Any:
        if plant not in {Plant.ORDER, Plant.PNL, Plant.TICKER}:
            raise ValueError("read-only runtime supports Order, PnL and optional Ticker plants")
        return self._build(
            Template.LOGIN_REQUEST,
            template_version=self.template_version,
            user_msg=[correlation_id or self.correlation_id("login")],
            user=credentials.username,
            password=credentials.password,
            app_name=app_name,
            app_version=app_version,
            system_name=system_name,
            infra_type=int(plant),
        )

    def logout(self, *, correlation_id: str | None = None) -> Any:
        return self._build(
            Template.LOGOUT_REQUEST,
            user_msg=[correlation_id or self.correlation_id("logout")],
        )

    def heartbeat(self, *, correlation_id: str | None = None) -> Any:
        return self._build(
            Template.HEARTBEAT_REQUEST,
            user_msg=[correlation_id or self.correlation_id("heartbeat")],
        )

    def login_info(self, *, correlation_id: str | None = None) -> Any:
        return self._build(
            Template.LOGIN_INFO_REQUEST,
            user_msg=[correlation_id or self.correlation_id("login-info")],
        )

    def account_list(
        self,
        *,
        fcm_id: str,
        ib_id: str,
        user_type: int = 3,
        correlation_id: str | None = None,
    ) -> Any:
        if user_type not in {1, 2, 3}:
            raise ValueError("account-list user_type must be FCM, IB, or TRADER")
        return self._build(
            Template.ACCOUNT_LIST_REQUEST,
            user_msg=[correlation_id or self.correlation_id("accounts")],
            fcm_id=fcm_id,
            ib_id=ib_id,
            user_type=user_type,
        )

    def user_info(
        self,
        *,
        fcm_id: str,
        ib_id: str,
        user: str,
        correlation_id: str | None = None,
    ) -> Any:
        if not fcm_id or not ib_id or not user:
            raise ValueError("user-info request requires FCM, IB, and user identities")
        return self._build(
            Template.USER_INFO_REQUEST,
            user_msg=[correlation_id or self.correlation_id("user-info")],
            fcm_id=fcm_id,
            ib_id=ib_id,
            user=user,
        )

    def playback_account_users(
        self,
        *,
        start_index: int,
        finish_index: int | None = None,
        playback: str = "accounts",
        correlation_id: str | None = None,
    ) -> Any:
        if playback not in {"accounts", "users", "all"}:
            raise ValueError("account/user playback must be accounts, users, or all")
        if not 0 <= start_index <= 2_147_483_647:
            raise ValueError(
                "account/user playback start_index must fit a non-negative proto int32"
            )
        if finish_index is not None and finish_index > 2_147_483_647:
            raise ValueError(
                "account/user playback finish_index must fit proto int32"
            )
        if finish_index is not None and finish_index < start_index:
            raise ValueError("account/user playback finish_index precedes start_index")
        fields: dict[str, Any] = {
            "user_msg": [correlation_id or self.correlation_id("account-users")],
            "playback": playback,
            "start_index": start_index,
        }
        if finish_index is not None:
            fields["finish_index"] = finish_index
        return self._build(Template.PLAYBACK_ACCOUNT_USERS_REQUEST, **fields)

    def subscribe_orders(self, account: BrokerAccountKey, *, correlation_id: str | None = None) -> Any:
        return self._build(
            Template.ORDER_UPDATES_REQUEST,
            user_msg=[correlation_id or self.correlation_id("orders-sub")],
            **self._account_fields(account),
        )

    def subscribe_brackets(self, account: BrokerAccountKey, *, correlation_id: str | None = None) -> Any:
        return self._build(
            Template.BRACKET_UPDATES_REQUEST,
            user_msg=[correlation_id or self.correlation_id("brackets-sub")],
            **self._account_fields(account),
        )

    def subscribe_pnl(self, account: BrokerAccountKey, *, correlation_id: str | None = None) -> Any:
        return self._build(
            Template.PNL_UPDATES_REQUEST,
            user_msg=[correlation_id or self.correlation_id("pnl-sub")],
            request=1,
            rms_updates_only=False,
            **self._account_fields(account),
        )

    def subscribe_rms(self, account: BrokerAccountKey, *, correlation_id: str | None = None) -> Any:
        return self._build(
            Template.RMS_UPDATES_REQUEST,
            user_msg=[correlation_id or self.correlation_id("rms-sub")],
            request="subscribe",
            **self._account_fields(account),
        )

    def show_orders(self, account: BrokerAccountKey, *, correlation_id: str | None = None) -> Any:
        return self._build(
            Template.SHOW_ORDERS_REQUEST,
            user_msg=[correlation_id or self.correlation_id("orders")],
            **self._account_fields(account),
        )

    def replay_executions(
        self,
        account: BrokerAccountKey,
        *,
        start_index: int | None = None,
        finish_index: int | None = None,
        correlation_id: str | None = None,
    ) -> Any:
        fields: dict[str, Any] = self._account_fields(account)
        fields["user_msg"] = [correlation_id or self.correlation_id("executions")]
        if start_index is not None:
            fields["start_index"] = start_index
        if finish_index is not None:
            fields["finish_index"] = finish_index
        return self._build(Template.REPLAY_EXECUTIONS_REQUEST, **fields)

    def show_order_history(
        self,
        account: BrokerAccountKey,
        basket_id: str,
        *,
        correlation_id: str | None = None,
    ) -> Any:
        if not basket_id:
            raise ValueError("basket_id is required")
        return self._build(
            Template.SHOW_ORDER_HISTORY_REQUEST,
            user_msg=[correlation_id or self.correlation_id("order-history")],
            basket_id=basket_id,
            **self._account_fields(account),
        )

    def order_history_summary(
        self,
        account: BrokerAccountKey,
        trade_date: str,
        *,
        correlation_id: str | None = None,
    ) -> Any:
        if len(trade_date) != 8 or not trade_date.isdigit():
            raise ValueError("order-history date must be CCYYMMDD")
        return self._build(
            Template.ORDER_HISTORY_SUMMARY_REQUEST,
            user_msg=[correlation_id or self.correlation_id("order-history-summary")],
            date=trade_date,
            **self._account_fields(account),
        )

    def fill_history(
        self,
        account: BrokerAccountKey,
        *,
        start_index: int,
        finish_index: int,
        index_format: str = "trade_date",
        correlation_id: str | None = None,
    ) -> Any:
        if index_format not in {"ssboe", "trade_date"}:
            raise ValueError("fill-history index_format must be ssboe or trade_date")
        return self._build(
            Template.FILL_HISTORY_REQUEST,
            user_msg=[correlation_id or self.correlation_id("fills")],
            index_format=index_format,
            start_index=start_index,
            finish_index=finish_index,
            # A current-day account recovery is deliberately bounded. Enabling
            # server flow control would require RequestFlowControl continuation
            # and request-key aliasing; this read-only phase instead requests a
            # complete multipart response and waits for its terminal rp_code.
            flow_control="disabled",
            **self._account_fields(account),
        )

    def show_brackets(self, account: BrokerAccountKey, *, correlation_id: str | None = None) -> Any:
        return self._build(
            Template.SHOW_BRACKETS_REQUEST,
            user_msg=[correlation_id or self.correlation_id("brackets")],
            **self._account_fields(account),
        )

    def show_bracket_stops(
        self, account: BrokerAccountKey, *, correlation_id: str | None = None
    ) -> Any:
        return self._build(
            Template.SHOW_BRACKET_STOPS_REQUEST,
            user_msg=[correlation_id or self.correlation_id("bracket-stops")],
            **self._account_fields(account),
        )

    def pnl_snapshot(self, account: BrokerAccountKey, *, correlation_id: str | None = None) -> Any:
        return self._build(
            Template.PNL_SNAPSHOT_REQUEST,
            user_msg=[correlation_id or self.correlation_id("pnl-snapshot")],
            **self._account_fields(account),
        )

    def account_rms(
        self,
        account: BrokerAccountKey,
        *,
        user_type: int = 3,
        correlation_id: str | None = None,
    ) -> Any:
        fields = self._account_fields(account)
        # The official request is FCM/IB/user-type scoped rather than account
        # scoped. Requiring an allowlisted account still gates this query.
        fields.pop("account_id")
        return self._build(
            Template.ACCOUNT_RMS_REQUEST,
            user_msg=[correlation_id or self.correlation_id("account-rms")],
            user_type=user_type,
            **fields,
        )

    def product_rms(
        self, account: BrokerAccountKey, *, correlation_id: str | None = None
    ) -> Any:
        return self._build(
            Template.PRODUCT_RMS_REQUEST,
            user_msg=[correlation_id or self.correlation_id("product-rms")],
            **self._account_fields(account),
        )

    def search_symbols(
        self,
        search_text: str,
        *,
        exchange: str | None = None,
        product_code: str | None = None,
        instrument_type: int = 1,
        contains: bool = True,
        correlation_id: str | None = None,
    ) -> Any:
        fields: dict[str, Any] = {
            "user_msg": [correlation_id or self.correlation_id("symbol-search")],
            "search_text": search_text,
            "instrument_type": instrument_type,
            "pattern": 2 if contains else 1,
        }
        if exchange:
            fields["exchange"] = exchange
        if product_code:
            fields["product_code"] = product_code
        return self._build(Template.SEARCH_SYMBOLS_REQUEST, **fields)

    def reference_data(
        self, symbol: str, exchange: str, *, correlation_id: str | None = None
    ) -> Any:
        return self._build(
            Template.REFERENCE_DATA_REQUEST,
            user_msg=[correlation_id or self.correlation_id("reference")],
            symbol=symbol,
            exchange=exchange,
        )

    def tick_size_table(self, tick_size_type: str, *, correlation_id: str | None = None) -> Any:
        return self._build(
            Template.TICK_SIZE_TABLE_REQUEST,
            user_msg=[correlation_id or self.correlation_id("tick-table")],
            tick_size_type=tick_size_type,
        )
