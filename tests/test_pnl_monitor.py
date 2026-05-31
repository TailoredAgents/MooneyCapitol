from contextlib import contextmanager

from app.core.config import CopyTargetAccountConfig
from app.services.pnl_monitor import (
    MonitoredAccount,
    PnlMonitor,
    PnlRiskLimits,
    assess_risk,
    normalize_account_balance,
    target_display_name,
)


class FakeClient:
    def get_account_balance(self, account_id):
        return {
            "data": {
                "cashBalance": "5000",
                "netLiquidation": "10000",
                "buyingPower": "40000",
            }
        }

    def get_account_positions(self, account_id):
        return {
            "data": {
                "positions": [
                    {
                        "symbol": "AAPL",
                        "positionQty": "10",
                        "avgPrice": "90",
                        "marketValue": "1000",
                    }
                ]
            }
        }


class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def scalars(self):
        return self

    def first(self):
        return self.rows[0] if self.rows else None

    def all(self):
        return self.rows


class FakeSession:
    def __init__(self):
        self.added = []
        self.flushed = 0

    def execute(self, stmt):
        return FakeResult([])

    def add(self, row):
        if getattr(row, "id", None) is None:
            row.id = len(self.added) + 1
        self.added.append(row)

    def flush(self):
        self.flushed += 1


def test_normalize_account_balance_accepts_webull_aliases():
    balance = normalize_account_balance(
        {"data": {"cashBalance": "5000", "netLiquidation": "10000", "buyingPower": "40000"}}
    )

    assert balance.cash_balance == 5000
    assert balance.total_value == 10000
    assert balance.buying_power == 40000


def test_normalize_account_balance_accepts_webull_openapi_balance_shape():
    balance = normalize_account_balance(
        {
            "total_asset_currency": "USD",
            "total_net_liquidation_value": "165.27",
            "total_cash_balance": "165.27",
            "account_currency_assets": [
                {
                    "currency": "USD",
                    "cash_balance": "165.27",
                    "buying_power": "661.08",
                    "maintenance_margin": "0.00",
                }
            ],
        }
    )

    assert balance.total_value == 165.27
    assert balance.equity_value == 165.27
    assert balance.cash_balance == 165.27
    assert balance.buying_power == 661.08


def test_normalize_account_balance_accepts_margin_asset_shape():
    balance = normalize_account_balance(
        {
            "total_asset_currency": "USD",
            "total_asset": "25000.50",
            "total_cash_balance": "12000.25",
            "account_currency_assets": [
                {
                    "currency": "USD",
                    "cash_balance": "12000.25",
                    "day_buying_power": "100000.00",
                    "overnight_buying_power": "50000.00",
                    "margin_power": "75000.00",
                }
            ],
        }
    )

    assert balance.total_value == 25000.50
    assert balance.equity_value == 25000.50
    assert balance.cash_balance == 12000.25
    assert balance.buying_power == 100000.00


def test_target_display_name_prefers_env_then_config(monkeypatch):
    target = CopyTargetAccountConfig(
        name="personal",
        display_name="Austin Dugger's Account",
        display_name_env="WEBULL_PERSONAL_DISPLAY_NAME",
        endpoint_env="WEBULL_PERSONAL_API_ENDPOINT",
        api_key_env="WEBULL_PERSONAL_APP_KEY",
        api_secret_env="WEBULL_PERSONAL_APP_SECRET",
        account_id_env="WEBULL_PERSONAL_ACCOUNT_ID",
    )

    assert target_display_name(target) == "Austin Dugger's Account"

    monkeypatch.setenv("WEBULL_PERSONAL_DISPLAY_NAME", "Austin Live Account")

    assert target_display_name(target) == "Austin Live Account"


def test_target_display_name_uses_default_display_name_for_stale_personal_config():
    target = CopyTargetAccountConfig(
        name="personal",
        endpoint_env="WEBULL_PERSONAL_API_ENDPOINT",
        api_key_env="WEBULL_PERSONAL_APP_KEY",
        api_secret_env="WEBULL_PERSONAL_APP_SECRET",
        account_id_env="WEBULL_PERSONAL_ACCOUNT_ID",
    )

    assert target_display_name(target) == "Austin Dugger's Account"


def test_assess_risk_flags_loss_limit():
    level, alerts = assess_risk(
        total_pnl_today=-1200,
        max_drawdown=-1200,
        total_exposure=1000,
        starting_value=10_000,
        limits=PnlRiskLimits(max_daily_loss=1000),
    )

    assert level == "emergency"
    assert "daily loss limit" in alerts[0]


def test_pnl_monitor_collects_real_client_data_without_mocking_balances():
    session = FakeSession()

    @contextmanager
    def session_scope():
        yield session

    monitor = PnlMonitor(
        session_scope=session_scope,
        accounts_provider=lambda: [
            MonitoredAccount(name="personal", account_ref="acct-1", account_type="copy", client=FakeClient())
        ],
    )

    statuses = monitor.collect_once()

    assert len(statuses) == 1
    assert statuses[0].account_ref == "acct-1"
    assert statuses[0].total_value == 10000
    assert statuses[0].position_count == 1
    assert statuses[0].total_exposure == 1000
    assert any(row.__class__.__name__ == "AccountSnapshot" for row in session.added)
    assert any(row.__class__.__name__ == "PositionSnapshot" for row in session.added)
