from app.copier.preflight import run_webull_preflight
from app.core.config import CopierConfig, CopyTargetAccountConfig


class FakeWebullClient:
    def __init__(self, credentials):
        self.credentials = credentials
        self.warmed = False

    def warm_up(self):
        self.warmed = True

    def get_account_list(self):
        return [{"account_id": self.credentials.account_id}]

    def get_account_positions(self, account_id):
        return [{"symbol": "AAPL", "quantity": "2"}]

    def get_account_balance(self, account_id):
        return {"account_id": account_id, "net_liquidation": "10000"}


def _config():
    return CopierConfig(
        enabled=True,
        mode="read_only",
        global_kill_switch=True,
        master_account="master-acct",
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
                equity=2_000,
                sizing_mode="percent_equity",
            )
        ],
    )


def test_webull_preflight_passes_with_valid_config_and_network(monkeypatch):
    monkeypatch.setenv("WEBULL_MASTER_API_ENDPOINT", "https://api.example")
    monkeypatch.setenv("WEBULL_MASTER_EVENTS_ENDPOINT", "events.example")
    monkeypatch.setenv("WEBULL_MASTER_APP_KEY", "master-key")
    monkeypatch.setenv("WEBULL_MASTER_APP_SECRET", "master-secret")
    monkeypatch.setenv("WEBULL_PERSONAL_API_ENDPOINT", "https://api.example")
    monkeypatch.setenv("WEBULL_PERSONAL_APP_KEY", "copy-key")
    monkeypatch.setenv("WEBULL_PERSONAL_APP_SECRET", "copy-secret")

    report = run_webull_preflight(_config(), client_factory=FakeWebullClient)

    assert report["ready"] is True
    keys = {check["key"] for check in report["checks"]}
    assert "master:master.account_match" in keys
    assert "target:personal.account_match" in keys
    assert report["accounts"][0]["account"].endswith("acct")


def test_webull_preflight_reports_missing_credentials_without_network(monkeypatch):
    monkeypatch.delenv("WEBULL_MASTER_API_ENDPOINT", raising=False)
    monkeypatch.delenv("WEBULL_MASTER_EVENTS_ENDPOINT", raising=False)
    monkeypatch.delenv("WEBULL_MASTER_APP_KEY", raising=False)
    monkeypatch.delenv("WEBULL_MASTER_APP_SECRET", raising=False)

    report = run_webull_preflight(_config(), include_network=False)

    assert report["ready"] is False
    blocker_keys = {check["key"] for check in report["blockers"]}
    assert "master.credentials" in blocker_keys
    assert "target:personal.credentials" in blocker_keys


def test_webull_preflight_blocks_account_mismatch(monkeypatch):
    class MismatchedClient(FakeWebullClient):
        def get_account_list(self):
            return [{"account_id": "different-acct"}]

    monkeypatch.setenv("WEBULL_MASTER_API_ENDPOINT", "https://api.example")
    monkeypatch.setenv("WEBULL_MASTER_EVENTS_ENDPOINT", "events.example")
    monkeypatch.setenv("WEBULL_MASTER_APP_KEY", "master-key")
    monkeypatch.setenv("WEBULL_MASTER_APP_SECRET", "master-secret")
    monkeypatch.setenv("WEBULL_PERSONAL_API_ENDPOINT", "https://api.example")
    monkeypatch.setenv("WEBULL_PERSONAL_APP_KEY", "copy-key")
    monkeypatch.setenv("WEBULL_PERSONAL_APP_SECRET", "copy-secret")

    report = run_webull_preflight(_config(), client_factory=MismatchedClient)

    assert report["ready"] is False
    blocker_keys = {check["key"] for check in report["blockers"]}
    assert "master:master.account_match" in blocker_keys
    assert "target:personal.account_match" in blocker_keys
