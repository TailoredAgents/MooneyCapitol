from __future__ import annotations

from contextlib import contextmanager

from app.db.models import AIArtifact
from app.services.scout_explanations import (
    SCOUT_EXPLANATION_PROMPT_VERSION,
    generate_scout_alert_explanation,
    scout_explanation_input,
)


class FakeAIClient:
    def __init__(self, *, enabled=True, text="ABCD is primed long on strong RVOL and L2 confirmation."):
        self.enabled = enabled
        self.text = text
        self.calls = []

    def generate_text(self, **kwargs):
        self.calls.append(kwargs)
        return type("Result", (), {"ok": True, "text": self.text, "output_json": {"id": "resp_test"}})()


class FakeSession:
    def __init__(self, store):
        self.store = store

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


def fake_session_scope(store):
    @contextmanager
    def scope():
        yield FakeSession(store)

    return scope


def test_scout_explanation_input_uses_trade_relevant_fields():
    payload = {
        "entry": "1.20",
        "stop": "1.10",
        "target": "1.45",
        "rr": "2.50",
        "p2r": "0.72",
        "spread": "0.7c",
        "l2": "74% ask",
        "features": {"rvol_break": 3.2, "unused": 99},
    }

    result = scout_explanation_input(
        alert_id=9,
        alert_type="primed",
        symbol="ABCD",
        direction="long",
        payload=payload,
    )

    assert result["symbol"] == "ABCD"
    assert result["entry"] == "1.20"
    assert result["l2"] == "74% ask"
    assert result["features"] == {"rvol_break": 3.2}


def test_generate_scout_alert_explanation_stores_completed_artifact():
    store = {}
    client = FakeAIClient()

    artifact_id = generate_scout_alert_explanation(
        alert_id=22,
        alert_type="trigger",
        symbol="ABCD",
        direction="long",
        payload={"entry": "1.20", "rr": "2.40", "rvol": "3.1x"},
        client=client,  # type: ignore[arg-type]
        session_scope=fake_session_scope(store),
    )

    artifact = store[artifact_id]
    assert artifact.artifact_type == "scout_explanation"
    assert artifact.source_id == "22"
    assert artifact.symbol == "ABCD"
    assert artifact.prompt_version == SCOUT_EXPLANATION_PROMPT_VERSION
    assert artifact.status == "completed"
    assert artifact.output_text.startswith("ABCD is primed")
    assert client.calls[0]["model"] == "gpt-5.4-mini"


def test_generate_scout_alert_explanation_noops_when_client_disabled():
    store = {}
    client = FakeAIClient(enabled=False)

    artifact_id = generate_scout_alert_explanation(
        alert_id=22,
        alert_type="trigger",
        symbol="ABCD",
        direction="long",
        payload={},
        client=client,  # type: ignore[arg-type]
        session_scope=fake_session_scope(store),
    )

    assert artifact_id is None
    assert store == {}
    assert client.calls == []
