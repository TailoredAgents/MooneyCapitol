from __future__ import annotations

from app.core.config import AppConfig, OpenAIConfig
from app.db.models import AIArtifact, Base
from app.services.ai_artifacts import complete_ai_artifact, create_ai_artifact, fail_ai_artifact
from app.services.ai_client import OpenAITextClient, _extract_output_text, openai_model_from_env, reserve_openai_budget


class FakeSession:
    def __init__(self):
        self.added = []
        self.flushed = False

    def add(self, row):
        self.added.append(row)

    def flush(self):
        self.flushed = True


def test_openai_config_defaults_are_disabled():
    cfg = AppConfig()

    assert isinstance(cfg.openai, OpenAIConfig)
    assert cfg.openai.ai_features_enabled is False
    assert cfg.openai.research_enabled is False
    assert cfg.openai.daily_request_limit == 200
    assert cfg.openai.research_daily_request_limit == 30
    assert cfg.openai.scout_explanation_model == "gpt-5.4-mini"
    assert cfg.openai.research_model == "gpt-5.5"


def test_ai_artifacts_model_is_registered_with_metadata():
    table = Base.metadata.tables["ai_artifacts"]

    assert table.c.artifact_type.type.length == 64
    assert table.c.source_type.type.length == 64
    assert table.c.source_id.type.length == 128
    assert table.c.symbol.type.length == 16
    assert table.c.output_text.nullable is True


def test_ai_artifact_repository_creates_and_updates_rows():
    session = FakeSession()

    artifact = create_ai_artifact(
        session,  # type: ignore[arg-type]
        artifact_type="scout_explanation",
        source_type="alert",
        source_id=123,
        symbol="abcd",
        model="gpt-5.4-mini",
        prompt_version="scout_explanation_v1",
        input_json={"symbol": "ABCD"},
    )

    assert session.added == [artifact]
    assert session.flushed is True
    assert artifact.source_id == "123"
    assert artifact.symbol == "ABCD"
    assert artifact.status == "created"

    complete_ai_artifact(artifact, output_text="ABCD is primed long.", output_json={"id": "resp_1"})
    assert artifact.status == "completed"
    assert artifact.output_text == "ABCD is primed long."
    assert artifact.error is None

    fail_ai_artifact(artifact, error="timeout")
    assert artifact.status == "error"
    assert artifact.error == "timeout"


def test_openai_client_is_noop_when_disabled_or_missing_key():
    disabled = OpenAITextClient(enabled=False, api_key="test")
    missing_key = OpenAITextClient(enabled=True, api_key="")

    assert disabled.generate_text(model="gpt-5.4-mini", instructions="x", input_text="y").status == "disabled"
    result = missing_key.generate_text(model="gpt-5.4-mini", instructions="x", input_text="y")
    assert result.status == "disabled"
    assert "OPENAI_API_KEY" in (result.error or "")


def test_openai_client_from_env_respects_feature_flag(monkeypatch):
    monkeypatch.setenv("OPENAI_AI_FEATURES_ENABLED", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_SCOUT_EXPLANATION_MODEL", "custom-model")

    client = OpenAITextClient.from_env()

    assert client.enabled is True
    assert client.api_key == "test-key"
    assert openai_model_from_env("OPENAI_SCOUT_EXPLANATION_MODEL", "gpt-5.4-mini") == "custom-model"


def test_openai_budget_reserves_daily_and_feature_usage(monkeypatch):
    store = {}
    monkeypatch.setenv("OPENAI_DAILY_REQUEST_LIMIT", "2")
    monkeypatch.setenv("OPENAI_RESEARCH_DAILY_REQUEST_LIMIT", "1")
    monkeypatch.setattr("app.services.ai_client.get_json", lambda key: store.get(key))
    monkeypatch.setattr("app.services.ai_client.set_json", lambda key, value: store.__setitem__(key, value))

    first = reserve_openai_budget(metadata={"feature": "ticker_research"})
    second = reserve_openai_budget(metadata={"feature": "ticker_research"})
    third = reserve_openai_budget(metadata={"feature": "daily_recap"})

    assert first.allowed is True
    assert first.daily_count == 1
    assert first.research_count == 1
    assert second.allowed is False
    assert "research daily request limit" in (second.reason or "")
    assert third.allowed is True
    assert third.daily_count == 2


def test_openai_client_returns_limited_when_budget_is_exhausted(monkeypatch):
    monkeypatch.setattr(
        "app.services.ai_client.reserve_openai_budget",
        lambda metadata=None: type("Decision", (), {"allowed": False, "reason": "limit reached"})(),
    )
    client = OpenAITextClient(enabled=True, api_key="test")

    result = client.generate_text(model="gpt-5.4-mini", instructions="x", input_text="y")

    assert result.status == "limited"
    assert result.error == "limit reached"


def test_extract_output_text_supports_responses_shapes():
    assert _extract_output_text({"output_text": " direct "}) == "direct"

    nested = {
        "output": [
            {"content": [{"type": "output_text", "text": "first"}, {"type": "output_text", "text": "second"}]},
        ]
    }
    assert _extract_output_text(nested) == "first\nsecond"

    assert _extract_output_text({"output": []}) is None
