from __future__ import annotations

import asyncio
from contextlib import contextmanager

from app.adapters.polygon_client import PolygonClient
from app.db.models import AIArtifact
from app.services.ticker_research import (
    TICKER_RESEARCH_PROMPT_VERSION,
    collect_ticker_research_sources,
    generate_ticker_research_for_alert,
    ticker_research_input,
)


class FakeAIClient:
    def __init__(self, *, enabled=True, text="- ABCD has one current Polygon news item from Test Publisher."):
        self.enabled = enabled
        self.text = text
        self.calls = []

    def generate_text(self, **kwargs):
        self.calls.append(kwargs)
        return type("Result", (), {"ok": True, "text": self.text, "output_json": {"id": "resp_research"}})()


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


class FakePolygon:
    async def get_ticker_details(self, symbol):
        return {
            "ticker": symbol.upper(),
            "name": "ABCD Holdings Inc.",
            "primary_exchange": "XNAS",
            "market_cap": 12_500_000,
            "sic_description": "Biotechnology",
        }

    async def get_ticker_news(self, symbol, limit=5):
        return [
            {
                "id": "news-1",
                "title": "ABCD announces financing",
                "publisher": {"name": "Test Publisher"},
                "published_utc": "2026-05-22T12:00:00Z",
                "article_url": "https://example.com/abcd",
                "description": "ABCD announced a financing and strategic update.",
                "tickers": [symbol.upper()],
                "insights": [{"ticker": symbol.upper(), "sentiment": "negative", "sentiment_reasoning": "Dilution risk"}],
            }
        ]

    async def close(self):
        return None


def fake_session_scope(store):
    @contextmanager
    def scope():
        yield FakeSession(store)

    return scope


def test_collect_ticker_research_sources_normalizes_polygon_sources():
    sources = asyncio.run(collect_ticker_research_sources(symbol="abcd", polygon=FakePolygon()))  # type: ignore[arg-type]

    assert sources["symbol"] == "ABCD"
    assert sources["ticker_details"]["name"] == "ABCD Holdings Inc."
    assert sources["news"][0]["title"] == "ABCD announces financing"
    assert sources["news"][0]["publisher"] == "Test Publisher"
    assert sources["news"][0]["insights"][0]["sentiment_reasoning"] == "Dilution risk"
    assert sources["warnings"] == []


def test_polygon_reference_methods_use_current_endpoints(monkeypatch):
    calls = []

    async def fake_request(self, method, path, params=None):
        calls.append((method, path, params))
        if path.endswith("/ABCD"):
            return {"results": {"ticker": "ABCD"}}
        return {"results": [{"title": "ABCD news"}]}

    monkeypatch.setattr(PolygonClient, "_request", fake_request)
    client = PolygonClient(api_key="test")
    async def run():
        try:
            details = await client.get_ticker_details("abcd")
            news = await client.get_ticker_news("abcd", limit=3)
            return details, news
        finally:
            await client.close()

    details, news = asyncio.run(run())

    assert details == {"ticker": "ABCD"}
    assert news == [{"title": "ABCD news"}]
    assert calls[0][1] == "/v3/reference/tickers/ABCD"
    assert calls[1][1] == "/v2/reference/news"
    assert calls[1][2]["ticker"] == "ABCD"
    assert calls[1][2]["limit"] == 3


def test_ticker_research_input_labels_source_quality():
    payload = ticker_research_input(
        alert_id=22,
        alert_type="trigger",
        symbol="ABCD",
        direction="long",
        payload={"entry": "2.00", "rvol": "4.2x"},
        sources={"ticker_details": {"name": "ABCD"}, "news": [{"title": "News"}], "warnings": ["no_filings"]},
    )

    assert payload["alert"]["symbol"] == "ABCD"
    assert payload["alert"]["entry"] == "2.00"
    assert payload["source_quality"]["has_ticker_details"] is True
    assert payload["source_quality"]["news_count"] == 1
    assert payload["source_quality"]["warnings"] == ["no_filings"]


def test_generate_ticker_research_stores_completed_artifact():
    store = {}
    client = FakeAIClient()

    artifact_id = generate_ticker_research_for_alert(
        alert_id=22,
        alert_type="trigger",
        symbol="ABCD",
        direction="long",
        payload={"entry": "2.00", "rvol": "4.2x"},
        sources={"ticker_details": {"name": "ABCD"}, "news": [{"title": "News"}], "warnings": []},
        client=client,  # type: ignore[arg-type]
        session_scope=fake_session_scope(store),
    )

    artifact = store[artifact_id]
    assert artifact.artifact_type == "ticker_research"
    assert artifact.source_type == "alert"
    assert artifact.source_id == "22"
    assert artifact.symbol == "ABCD"
    assert artifact.prompt_version == TICKER_RESEARCH_PROMPT_VERSION
    assert artifact.status == "completed"
    assert artifact.output_text.startswith("- ABCD")
    assert client.calls[0]["model"] == "gpt-5.5"


def test_generate_ticker_research_noops_when_disabled():
    store = {}
    client = FakeAIClient(enabled=False)

    artifact_id = generate_ticker_research_for_alert(
        alert_id=22,
        alert_type="trigger",
        symbol="ABCD",
        direction="long",
        payload={},
        sources={"ticker_details": None, "news": [], "warnings": ["no_recent_polygon_news"]},
        client=client,  # type: ignore[arg-type]
        session_scope=fake_session_scope(store),
    )

    assert artifact_id is None
    assert store == {}
    assert client.calls == []
