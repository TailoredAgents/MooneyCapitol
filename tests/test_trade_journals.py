from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone

from app.db.models import AIArtifact, CopyOrder, MasterExecution
from app.services.trade_journals import (
    TRADE_JOURNAL_PROMPT_VERSION,
    generate_trade_journal,
    trade_journal_input,
    trade_journal_tags,
)


class FakeAIClient:
    def __init__(self, *, enabled=True, text="- ABCD copied successfully with 82 ms latency."):
        self.enabled = enabled
        self.text = text
        self.calls = []

    def generate_text(self, **kwargs):
        self.calls.append(kwargs)
        return type("Result", (), {"ok": True, "text": self.text, "output_json": {"id": "resp_trade"}})()


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

    def scalars(self):
        return FakeScalarResult(self.rows)


class FakeSession:
    def __init__(self, store):
        self.store = store
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
        return FakeResult([])


def fake_session_scope(store):
    @contextmanager
    def scope():
        yield FakeSession(store)

    return scope


def _dt():
    return datetime(2026, 5, 22, 14, 30, tzinfo=timezone.utc)


def _master():
    return MasterExecution(
        id=11,
        broker="webull",
        account_ref="master",
        broker_execution_id="exec-1",
        broker_order_id="order-1",
        symbol="ABCD",
        side="BUY",
        qty=100.0,
        price=2.0,
        asset_class="equity",
        executed_at=_dt(),
        received_at=_dt(),
    )


def _order(status="filled"):
    return CopyOrder(
        id=22,
        master_execution_id=11,
        target_account_id=3,
        broker="webull",
        client_order_id="mc-22",
        broker_order_id="child-1",
        symbol="ABCD",
        side="BUY",
        qty=50.0,
        order_type="market",
        time_in_force="day",
        status=status,
        submitted_at=_dt(),
        accepted_at=_dt(),
        filled_at=_dt() if status == "filled" else None,
        filled_qty=50.0 if status == "filled" else None,
        avg_fill_price=2.01 if status == "filled" else None,
        reject_reason=None,
        latency_ms=82.4,
    )


def test_trade_journal_input_separates_master_copy_and_tags():
    payload = trade_journal_input(master=_master(), order=_order(), target_name="personal")

    assert payload["master_execution"]["symbol"] == "ABCD"
    assert payload["master_execution"]["notional"] == 200.0
    assert payload["copy_order"]["target"] == "personal"
    assert payload["copy_order"]["notional"] == 100.0
    assert payload["copy_order"]["filled_notional"] == 100.5
    assert payload["copy_order"]["slippage_bps"] == 50.0
    assert "copied" in payload["tags"]
    assert "under_300ms" in payload["tags"]
    assert "high_slippage" in payload["tags"]


def test_trade_journal_tags_marks_failed_orders_for_review():
    order = _order(status="submit_failed")
    order.reject_reason = "broker unavailable"
    order.latency_ms = 325.0

    tags = trade_journal_tags(order=order, slippage_bps=None)

    assert "manual_no_alert" in tags
    assert "submit_failed" in tags
    assert "over_300ms" in tags


def test_generate_trade_journal_stores_completed_artifact():
    store = {}
    client = FakeAIClient()

    artifact_id = generate_trade_journal(
        master=_master(),
        order=_order(),
        target_name="personal",
        client=client,  # type: ignore[arg-type]
        session_scope=fake_session_scope(store),
    )

    artifact = store[artifact_id]
    assert artifact.artifact_type == "trade_journal"
    assert artifact.source_type == "copy_order"
    assert artifact.source_id == "22"
    assert artifact.symbol == "ABCD"
    assert artifact.prompt_version == TRADE_JOURNAL_PROMPT_VERSION
    assert artifact.status == "completed"
    assert artifact.output_text.startswith("- ABCD copied")
    assert client.calls[0]["model"] == "gpt-5.4-mini"


def test_generate_trade_journal_noops_when_disabled():
    store = {}
    client = FakeAIClient(enabled=False)

    artifact_id = generate_trade_journal(
        master=_master(),
        order=_order(),
        target_name="personal",
        client=client,  # type: ignore[arg-type]
        session_scope=fake_session_scope(store),
    )

    assert artifact_id is None
    assert store == {}
    assert client.calls == []
