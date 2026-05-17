from datetime import datetime, timezone

from app.copier.engine import CopyResult
from app.copier.readonly_session import WebullReadOnlySessionRunner, read_only_session_config
from app.core.config import CopierConfig, CopyTargetAccountConfig


class FakeOrchestrator:
    def __init__(self):
        self.calls = []

    def plan_execution(self, master, targets):
        self.calls.append((master, targets))
        return [
            CopyResult(
                target="personal",
                allowed=True,
                submitted=False,
                reason="read_only",
                client_order_id="mc123",
                quantity=2,
            )
        ]


def _config(enabled=True):
    return CopierConfig(
        enabled=enabled,
        mode="live",
        live_trading_enabled=True,
        global_kill_switch=False,
        master_account="master",
        master_equity=30_000,
        targets=[
            CopyTargetAccountConfig(
                name="personal",
                enabled=True,
                account_ref="copy-acct",
                endpoint_env="WEBULL_PERSONAL_API_ENDPOINT",
                api_key_env="WEBULL_PERSONAL_APP_KEY",
                api_secret_env="WEBULL_PERSONAL_APP_SECRET",
                account_id_env="WEBULL_PERSONAL_ACCOUNT_ID",
                equity=10_000,
                sizing_mode="percent_equity",
            )
        ],
    )


def _payload(status="FILLED"):
    return {
        "account_id": "master",
        "order_id": "order-1",
        "symbol": "AAPL",
        "side": "BUY",
        "status": status,
        "filled_qty": "3",
        "avg_fill_price": "12.50",
        "filled_at": datetime(2026, 5, 16, 14, 30, tzinfo=timezone.utc).isoformat(),
        "instrument_type": "EQUITY",
    }


def test_read_only_session_forces_safe_mode_flags():
    config = read_only_session_config(_config(), simulate_copying=True)

    assert config.mode == "read_only"
    assert config.live_trading_enabled is False
    assert config.enabled is True
    assert config.global_kill_switch is False


def test_read_only_session_records_would_copy_decision(tmp_path):
    orchestrator = FakeOrchestrator()
    runner = WebullReadOnlySessionRunner(
        read_only_session_config(_config()),
        output_path=tmp_path / "session.jsonl",
        orchestrator=orchestrator,
    )

    report = runner.handle_event(_payload())

    assert report["payloads"][0]["status"] == "planned"
    assert report["payloads"][0]["would_copy"] == 1
    assert runner.summary.parsed_fills == 1
    assert runner.summary.would_copy == 1
    assert runner.summary.blocked == 0
    assert orchestrator.calls[0][0].execution_id == "order-1"
    assert orchestrator.calls[0][1][0].name == "personal"


def test_read_only_session_ignores_non_fill_payload(tmp_path):
    runner = WebullReadOnlySessionRunner(
        read_only_session_config(_config()),
        output_path=tmp_path / "session.jsonl",
        orchestrator=FakeOrchestrator(),
    )

    report = runner.handle_event(_payload(status="NEW"))

    assert report["payloads"][0]["status"] == "ignored"
    assert runner.summary.ignored == 1
    assert runner.summary.parsed_fills == 0


def test_read_only_session_respects_disabled_config(tmp_path):
    orchestrator = FakeOrchestrator()
    runner = WebullReadOnlySessionRunner(
        read_only_session_config(_config(enabled=False)),
        output_path=tmp_path / "session.jsonl",
        orchestrator=orchestrator,
    )

    report = runner.handle_event(_payload())

    assert report["payloads"][0]["reason"] == "copier_disabled"
    assert runner.summary.ignored == 1
    assert orchestrator.calls == []
