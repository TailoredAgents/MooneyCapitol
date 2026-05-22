from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timezone
from types import SimpleNamespace

from app.db.models import AIArtifact
from app.services.daily_recaps import (
    DAILY_RECAP_PROMPT_VERSION,
    copy_trading_summary,
    daily_recap_input,
    generate_daily_recap,
)


class FakeAIClient:
    def __init__(self, *, enabled=True, text="Quiet day: copier latency stayed healthy and no issues need review."):
        self.enabled = enabled
        self.text = text
        self.calls = []

    def generate_text(self, **kwargs):
        self.calls.append(kwargs)
        return type("Result", (), {"ok": True, "text": self.text, "output_json": {"id": "resp_daily"}})()


class FakeSlack:
    def __init__(self):
        self.posts = []

    def post(self, text):
        self.posts.append(text)


class FakeScalarResult:
    def __init__(self, rows):
        self.rows = rows

    def all(self):
        return self.rows

    def first(self):
        return self.rows[0] if self.rows else None


class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def all(self):
        return self.rows

    def scalars(self):
        return FakeScalarResult(self.rows)


class FakeSession:
    def __init__(self, store, execute_rows=None):
        self.store = store
        self.execute_rows = execute_rows or []
        self.pending = None

    def add(self, row):
        self.pending = row

    def flush(self):
        row = self.pending
        if row.id is None:
            row.id = len(self.store) + 1
        self.store[row.id] = row

    def get(self, model, row_id):
        assert model is AIArtifact
        return self.store.get(row_id)

    def execute(self, stmt):
        return FakeResult(self.execute_rows)


def fake_session_scope(store, execute_rows=None):
    @contextmanager
    def scope():
        yield FakeSession(store, execute_rows=execute_rows)

    return scope


def _ledger_summary():
    return {
        "total_trades": 2,
        "wins": 1,
        "win_rate": 50.0,
        "avg_r": 0.4,
        "expectancy": 0.4,
        "net_pnl": 125.5,
        "by_setup": [{"symbol": "ABCD", "pnl": 180.0}, {"symbol": "WXYZ", "pnl": -54.5}],
        "by_hour": {"14": 125.5},
    }


def test_daily_recap_input_combines_ledger_copier_and_learning():
    payload = daily_recap_input(
        trade_date=date(2026, 5, 22),
        ledger_summary=_ledger_summary(),
        copy_summary={"total_copy_orders": 2, "latency": {"mean": 82.0}},
        learning_translation={"text": "RVOL mattered most."},
    )

    assert payload["date"] == "2026-05-22"
    assert payload["ledger"]["net_pnl"] == 125.5
    assert payload["ledger"]["top_setups"][0]["symbol"] == "ABCD"
    assert payload["copier"]["total_copy_orders"] == 2
    assert payload["learning"]["text"] == "RVOL mattered most."


def test_copy_trading_summary_calculates_latency_slippage_and_issues():
    master = SimpleNamespace(symbol="ABCD", side="BUY", price=2.0, executed_at=datetime(2026, 5, 22, tzinfo=timezone.utc))
    order = SimpleNamespace(
        status="filled",
        latency_ms=82.4,
        avg_fill_price=2.01,
        reject_reason=None,
    )
    failed_order = SimpleNamespace(
        status="submit_failed",
        latency_ms=325.0,
        avg_fill_price=None,
        reject_reason="broker unavailable",
    )
    rows = [
        (master, order, "personal"),
        (SimpleNamespace(symbol="WXYZ", side="SELL", price=5.0, executed_at=master.executed_at), failed_order, "personal"),
    ]

    summary = copy_trading_summary(session=FakeSession({}, execute_rows=rows), trade_date=date(2026, 5, 22))

    assert summary["total_copy_orders"] == 2
    assert summary["filled_copy_orders"] == 1
    assert summary["status_counts"] == {"filled": 1, "submit_failed": 1}
    assert summary["latency"]["under_300ms_rate"] == 0.5
    assert summary["slippage_bps"]["mean"] == 50.0
    assert summary["issues"][0]["reason"] == "broker unavailable"


def test_generate_daily_recap_stores_artifact_and_posts_slack():
    store = {}
    client = FakeAIClient()
    slack = FakeSlack()

    artifact_id = generate_daily_recap(
        trade_date=date(2026, 5, 22),
        ledger_summary=_ledger_summary(),
        copy_summary={"total_copy_orders": 1, "latency": {"mean": 82.4}},
        learning_translation={"text": "L2 persistence mattered."},
        client=client,  # type: ignore[arg-type]
        slack=slack,
        session_scope=fake_session_scope(store),
    )

    artifact = store[artifact_id]
    assert artifact.artifact_type == "daily_recap"
    assert artifact.source_type == "eod_report"
    assert artifact.source_id == "2026-05-22"
    assert artifact.prompt_version == DAILY_RECAP_PROMPT_VERSION
    assert artifact.status == "completed"
    assert artifact.output_text.startswith("Quiet day")
    assert client.calls[0]["model"] == "gpt-5.4"
    assert slack.posts and "AI Daily Recap - 2026-05-22" in slack.posts[0]


def test_generate_daily_recap_noops_when_disabled():
    store = {}
    client = FakeAIClient(enabled=False)

    artifact_id = generate_daily_recap(
        trade_date=date(2026, 5, 22),
        ledger_summary=_ledger_summary(),
        client=client,  # type: ignore[arg-type]
        session_scope=fake_session_scope(store),
    )

    assert artifact_id is None
    assert store == {}
    assert client.calls == []
