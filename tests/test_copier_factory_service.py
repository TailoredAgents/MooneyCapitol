from contextlib import contextmanager
from datetime import datetime, timezone

from app.copier.engine import CopyResult
from app.copier.factory import build_copy_targets
from app.copier.models import MasterExecutionEvent
from app.copier.service import CopyOrchestrator
from app.core.config import CopierConfig, CopyTargetAccountConfig


class FakeClient:
    def place_equity_order(self, account_id, order):
        return {"order_id": "child-1"}


class FakeSlack:
    def __init__(self):
        self.posts = []

    def post(self, text):
        self.posts.append(text)


def _master():
    return MasterExecutionEvent(
        broker="webull",
        account_id="master",
        execution_id="exec-1",
        order_id="order-1",
        client_order_id="master-client",
        symbol="AAPL",
        side="BUY",
        quantity=10,
        price=25.0,
        executed_at=datetime(2026, 5, 16, 14, 30, tzinfo=timezone.utc),
        raw_payload={},
    )


def test_build_copy_targets_uses_configured_webull_accounts():
    config = CopierConfig(
        enabled=True,
        global_kill_switch=False,
        master_equity=30_000,
        copy_shorts=True,
        targets=[
            CopyTargetAccountConfig(
                name="personal",
                account_ref="copy-acct",
                endpoint_env="WEBULL_PERSONAL_API_ENDPOINT",
                api_key_env="WEBULL_PERSONAL_APP_KEY",
                api_secret_env="WEBULL_PERSONAL_APP_SECRET",
                account_id_env="WEBULL_PERSONAL_ACCOUNT_ID",
                enabled=True,
                equity=10_000,
                sizing_mode="percent_equity",
                sizing_value=0,
                min_notional=25,
                max_notional_per_trade=1_000,
                max_position_pct=0.10,
                max_daily_notional=5_000,
                max_daily_trades=40,
                allowlist=["aapl"],
            )
        ],
    )

    targets = build_copy_targets(config, client_factory=lambda target_cfg: FakeClient())

    assert len(targets) == 1
    target = targets[0]
    assert target.name == "personal"
    assert target.account_id == "copy-acct"
    assert target.sizing.mode == "percent_equity"
    assert target.sizing.min_notional == 25
    assert target.risk.enabled is True
    assert target.risk.global_kill_switch is False
    assert target.risk.max_position_pct == 0.10
    assert target.risk.max_daily_notional == 5_000
    assert target.risk.max_daily_trades == 40
    assert target.risk.max_orders_per_minute == config.max_orders_per_minute
    assert target.risk.allowlist == {"AAPL"}
    assert target.master_equity == 30_000
    assert target.target_equity == 10_000


def test_copy_orchestrator_persists_engine_results(monkeypatch):
    persisted = {}

    class FakeEngine:
        def copy_execution(self, master, targets):
            return [
                CopyResult(
                    target="personal",
                    allowed=False,
                    submitted=False,
                    reason="global_kill_switch",
                )
            ]

    @contextmanager
    def fake_session_scope():
        yield "session"

    def fake_persist(session, master, targets_by_name, results):
        persisted["session"] = session
        persisted["master"] = master
        persisted["targets"] = targets_by_name
        persisted["results"] = results

    monkeypatch.setattr("app.copier.service.persist_copy_results", fake_persist)
    orchestrator = CopyOrchestrator(
        engine=FakeEngine(),
        session_scope=fake_session_scope,
        require_persistence=False,
        slack=FakeSlack(),
        background_persistence=False,
        background_alerts=False,
    )

    target = build_copy_targets(
        CopierConfig(
            enabled=False,
            targets=[
                CopyTargetAccountConfig(
                    name="personal",
                    account_ref="copy-acct",
                    endpoint_env="WEBULL_PERSONAL_API_ENDPOINT",
                    api_key_env="WEBULL_PERSONAL_APP_KEY",
                    api_secret_env="WEBULL_PERSONAL_APP_SECRET",
                    account_id_env="WEBULL_PERSONAL_ACCOUNT_ID",
                )
            ],
        ),
        client_factory=lambda target_cfg: FakeClient(),
    )
    results = orchestrator.copy_execution(_master(), target)

    assert results[0].reason == "global_kill_switch"
    assert persisted["session"] == "session"
    assert persisted["master"].execution_id == "exec-1"
    assert list(persisted["targets"]) == ["personal"]
    assert persisted["results"] == results


def test_copy_orchestrator_plans_read_only_results_without_hot_path_position_update(monkeypatch):
    persisted = {}
    observed = {}

    class FakeEngine:
        def plan_execution(self, master, targets):
            observed["positions_before"] = targets[0].positions
            return [
                CopyResult(
                    target="personal",
                    allowed=True,
                    submitted=False,
                    reason="read_only",
                    client_order_id="mc-read-only",
                    quantity=2,
                )
            ]

    @contextmanager
    def fake_session_scope():
        yield "session"

    def fake_persist(session, master, targets_by_name, results):
        persisted["session"] = session
        persisted["results"] = results

    monkeypatch.setattr("app.copier.service.persist_copy_results", fake_persist)
    orchestrator = CopyOrchestrator(
        engine=FakeEngine(),
        session_scope=fake_session_scope,
        require_persistence=False,
        slack=FakeSlack(),
        background_persistence=False,
        background_alerts=False,
    )
    target = build_copy_targets(
        CopierConfig(
            enabled=True,
            global_kill_switch=False,
            targets=[
                CopyTargetAccountConfig(
                    name="personal",
                    account_ref="copy-acct",
                    endpoint_env="WEBULL_PERSONAL_API_ENDPOINT",
                    api_key_env="WEBULL_PERSONAL_APP_KEY",
                    api_secret_env="WEBULL_PERSONAL_APP_SECRET",
                    account_id_env="WEBULL_PERSONAL_ACCOUNT_ID",
                )
            ],
        ),
        client_factory=lambda target_cfg: FakeClient(),
    )

    results = orchestrator.plan_execution(_master(), target)
    second_results = orchestrator.plan_execution(_master(), target)

    assert results[0].reason == "read_only"
    assert second_results == []
    assert observed["positions_before"] == {}
    assert persisted["session"] == "session"
    assert persisted["results"] == results
    assert orchestrator._target_positions == {}


def test_copy_orchestrator_posts_limit_block_alert(monkeypatch):
    class FakeEngine:
        def copy_execution(self, master, targets):
            return [
                CopyResult(
                    target="personal",
                    allowed=False,
                    submitted=False,
                    reason="max_daily_notional_exceeded",
                    quantity=2,
                )
            ]

    @contextmanager
    def fake_session_scope():
        yield "session"

    monkeypatch.setattr("app.copier.service.persist_copy_results", lambda *args, **kwargs: None)
    slack = FakeSlack()
    orchestrator = CopyOrchestrator(
        engine=FakeEngine(),
        session_scope=fake_session_scope,
        require_persistence=False,
        slack=slack,
        background_persistence=False,
        background_alerts=False,
    )
    target = build_copy_targets(
        CopierConfig(
            enabled=False,
            targets=[
                CopyTargetAccountConfig(
                    name="personal",
                    account_ref="copy-acct",
                    endpoint_env="WEBULL_PERSONAL_API_ENDPOINT",
                    api_key_env="WEBULL_PERSONAL_APP_KEY",
                    api_secret_env="WEBULL_PERSONAL_APP_SECRET",
                    account_id_env="WEBULL_PERSONAL_ACCOUNT_ID",
                )
            ],
        ),
        client_factory=lambda target_cfg: FakeClient(),
    )

    orchestrator.copy_execution(_master(), target)

    assert "max_daily_notional_exceeded" in slack.posts[0]


def test_copy_orchestrator_hydrates_target_positions(monkeypatch):
    observed = {}

    class FakeEngine:
        def copy_execution(self, master, targets):
            observed["positions"] = targets[0].positions
            return []

    @contextmanager
    def fake_session_scope():
        yield "session"

    monkeypatch.setattr("app.copier.service.master_execution_exists", lambda session, master: False)
    monkeypatch.setattr("app.copier.service.load_copy_risk_usage", lambda session, names: {})
    monkeypatch.setattr("app.copier.service.load_copy_positions", lambda session, names: {"personal": {"AAPL": 4.0}})
    monkeypatch.setattr("app.copier.service.persist_copy_results", lambda *args, **kwargs: None)
    orchestrator = CopyOrchestrator(
        engine=FakeEngine(),
        session_scope=fake_session_scope,
        require_persistence=False,
        slack=FakeSlack(),
        hydrate_state_before_submit=True,
        background_persistence=False,
        background_alerts=False,
    )
    target = build_copy_targets(
        CopierConfig(
            enabled=True,
            global_kill_switch=False,
            targets=[
                CopyTargetAccountConfig(
                    name="personal",
                    account_ref="copy-acct",
                    endpoint_env="WEBULL_PERSONAL_API_ENDPOINT",
                    api_key_env="WEBULL_PERSONAL_APP_KEY",
                    api_secret_env="WEBULL_PERSONAL_APP_SECRET",
                    account_id_env="WEBULL_PERSONAL_ACCOUNT_ID",
                )
            ],
        ),
        client_factory=lambda target_cfg: FakeClient(),
    )

    orchestrator.copy_execution(_master(), target)

    assert observed["positions"] == {"AAPL": 4.0}


def test_copy_orchestrator_does_not_read_db_before_hot_path_submit():
    observed = {}

    class FakeEngine:
        def copy_execution(self, master, targets):
            observed["submitted"] = True
            return []

    @contextmanager
    def failing_session_scope():
        raise RuntimeError("database should not be used before submit")
        yield

    orchestrator = CopyOrchestrator(
        engine=FakeEngine(),
        session_scope=failing_session_scope,
        require_persistence=False,
        slack=FakeSlack(),
        background_persistence=False,
        background_alerts=False,
    )
    target = build_copy_targets(
        CopierConfig(
            enabled=True,
            global_kill_switch=False,
            targets=[
                CopyTargetAccountConfig(
                    name="personal",
                    account_ref="copy-acct",
                    endpoint_env="WEBULL_PERSONAL_API_ENDPOINT",
                    api_key_env="WEBULL_PERSONAL_APP_KEY",
                    api_secret_env="WEBULL_PERSONAL_APP_SECRET",
                    account_id_env="WEBULL_PERSONAL_ACCOUNT_ID",
                )
            ],
        ),
        client_factory=lambda target_cfg: FakeClient(),
    )

    orchestrator.copy_execution(_master(), target)

    assert observed["submitted"] is True
