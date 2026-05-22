from __future__ import annotations

from contextlib import contextmanager
from datetime import date

from app.db.models import AIArtifact
from app.services.learning_translations import (
    LEARNING_TRANSLATION_PROMPT_VERSION,
    generate_learning_report_translation,
    learning_translation_input,
)


class FakeAIClient:
    def __init__(self, *, enabled=True, text="L2 persistence and RVOL were the strongest learning signals tonight."):
        self.enabled = enabled
        self.text = text
        self.calls = []

    def generate_text(self, **kwargs):
        self.calls.append(kwargs)
        return type("Result", (), {"ok": True, "text": self.text, "output_json": {"id": "resp_learning"}})()


class FakeSlack:
    def __init__(self):
        self.posts = []

    def post(self, text):
        self.posts.append(text)


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


def test_learning_translation_input_summarizes_key_model_fields():
    payload = learning_translation_input(
        trade_date=date(2026, 5, 22),
        result={
            "status": "trained",
            "model_type": "xgboost",
            "rows": 42,
            "positives": 12,
            "feature_importance": [{"feature": "l2_mean", "importance": 0.4}],
            "ranking_metrics": {"taken_rows": 5, "top_5_taken_hit_rate": 0.8},
            "sandbox_enhanced": {"enabled": False},
        },
    )

    assert payload["date"] == "2026-05-22"
    assert payload["model_type"] == "xgboost"
    assert payload["top_features"] == [{"feature": "l2_mean", "importance": 0.4}]
    assert payload["ranking_metrics"]["top_5_taken_hit_rate"] == 0.8


def test_generate_learning_report_translation_stores_artifact_and_posts_slack():
    store = {}
    client = FakeAIClient()
    slack = FakeSlack()

    artifact_id = generate_learning_report_translation(
        trade_date=date(2026, 5, 22),
        result={"status": "trained", "model_type": "xgboost", "rows": 42},
        client=client,  # type: ignore[arg-type]
        slack=slack,
        session_scope=fake_session_scope(store),
    )

    artifact = store[artifact_id]
    assert artifact.artifact_type == "learning_translation"
    assert artifact.source_type == "learning_report"
    assert artifact.source_id == "2026-05-22"
    assert artifact.prompt_version == LEARNING_TRANSLATION_PROMPT_VERSION
    assert artifact.status == "completed"
    assert artifact.output_text.startswith("L2 persistence")
    assert client.calls[0]["model"] == "gpt-5.4-mini"
    assert slack.posts and "Learning AI Summary - 2026-05-22" in slack.posts[0]


def test_generate_learning_report_translation_noops_when_disabled():
    store = {}
    client = FakeAIClient(enabled=False)
    slack = FakeSlack()

    artifact_id = generate_learning_report_translation(
        trade_date=date(2026, 5, 22),
        result={"status": "trained"},
        client=client,  # type: ignore[arg-type]
        slack=slack,
        session_scope=fake_session_scope(store),
    )

    assert artifact_id is None
    assert store == {}
    assert slack.posts == []
