from datetime import datetime, timezone

from app.api.routes.pnl import _snapshot_response, _summary_bucket
from app.core.config import CopierConfig, CopyTargetAccountConfig
from app.db.models import AccountSnapshot, PnlAlert


def _snapshot(
    *,
    account_ref,
    account_name,
    account_type,
    total_value,
    total_pnl_today=0.0,
    cash_balance=0.0,
    buying_power=0.0,
    total_exposure=0.0,
    position_count=0,
):
    return AccountSnapshot(
        account_ref=account_ref,
        account_name=account_name,
        account_type=account_type,
        snapshot_time=datetime(2026, 5, 23, tzinfo=timezone.utc),
        total_value=total_value,
        total_pnl_today=total_pnl_today,
        cash_balance=cash_balance,
        equity_value=total_value,
        buying_power=buying_power,
        unrealized_pnl=0.0,
        realized_pnl_today=0.0,
        max_drawdown_today=0.0,
        max_profit_today=0.0,
        total_exposure=total_exposure,
        position_count=position_count,
        risk_level="normal",
    )


def test_pnl_summary_bucket_totals_accounts_and_alerts():
    master = _snapshot(
        account_ref="master-1",
        account_name="master",
        account_type="master",
        total_value=30_000,
        total_pnl_today=250,
        buying_power=120_000,
        total_exposure=5_000,
        position_count=2,
    )
    personal = _snapshot(
        account_ref="copy-1",
        account_name="personal",
        account_type="copy",
        total_value=165.27,
        total_pnl_today=1.25,
        buying_power=661.08,
        total_exposure=20,
        position_count=1,
    )
    other = _snapshot(
        account_ref="copy-2",
        account_name="future-copy",
        account_type="copy",
        total_value=10_000,
        total_pnl_today=-50,
        buying_power=40_000,
        total_exposure=1_000,
    )
    alerts = [
        PnlAlert(account_ref="copy-1", account_name="personal", alert_type="loss", severity="warning", message="watch"),
        PnlAlert(account_ref="not-in-bucket", account_name="other", alert_type="loss", severity="warning", message="skip"),
    ]

    bucket = _summary_bucket([master, personal, other], alerts)

    assert bucket["account_count"] == 3
    assert bucket["total_value"] == 40165.27
    assert bucket["total_pnl_today"] == 201.25
    assert bucket["buying_power"] == 160661.08
    assert bucket["total_exposure"] == 6020
    assert bucket["position_count"] == 3
    assert bucket["active_alerts"] == 1
    assert bucket["last_refresh"] == "2026-05-23T00:00:00+00:00"


def test_snapshot_response_uses_configured_copy_account_display_name(monkeypatch):
    from app.api.routes import pnl

    original = pnl.CONFIG.copier
    try:
        pnl.CONFIG.copier = CopierConfig(
            targets=[
                CopyTargetAccountConfig(
                    name="personal",
                    display_name="Austin Dugger's Account",
                    endpoint_env="WEBULL_PERSONAL_API_ENDPOINT",
                    api_key_env="WEBULL_PERSONAL_APP_KEY",
                    api_secret_env="WEBULL_PERSONAL_APP_SECRET",
                    account_id_env="WEBULL_PERSONAL_ACCOUNT_ID",
                )
            ]
        )
        row = _snapshot(
            account_ref="IEG8KBJUE5TU4K637T57ADR7C8",
            account_name="personal",
            account_type="copy",
            total_value=165.27,
        )

        response = _snapshot_response(row)

        assert response.account_name == "Austin Dugger's Account"
    finally:
        pnl.CONFIG.copier = original
