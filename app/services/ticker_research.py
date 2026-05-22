from __future__ import annotations

import asyncio
import json
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from typing import Any, Callable

from app.adapters.polygon_client import PolygonClient
from app.core.config_store import CONFIG
from app.db.models import AIArtifact
from app.db.session import get_session
from app.observability.logging import get_logger
from app.services.ai_artifacts import complete_ai_artifact, create_ai_artifact, fail_ai_artifact, latest_ai_artifact
from app.services.ai_client import OpenAITextClient, openai_model_from_env


logger = get_logger("ticker_research")

TICKER_RESEARCH_TYPE = "ticker_research"
TICKER_RESEARCH_SOURCE_TYPE = "alert"
TICKER_RESEARCH_PROMPT_VERSION = "ticker_research_v1"

TICKER_RESEARCH_INSTRUCTIONS = """You write source-aware ticker and catalyst research for a small-cap momentum trader.
Use only the provided alert data and source records.
Write 3-5 concise bullet lines covering:
- known company/news context from the sources,
- likely catalyst if sources support one,
- source-backed risk flags such as offering/dilution, reverse split, halt/news uncertainty, low-float volatility, or no current source data,
- source timestamp or publisher when available.
Do not give trading advice, do not predict outcomes, and do not invent facts. If source data is missing, say that clearly."""


async def collect_ticker_research_sources(
    *,
    symbol: str,
    polygon: PolygonClient | None = None,
    news_limit: int = 5,
) -> dict[str, Any]:
    client = polygon or PolygonClient()
    close_client = polygon is None
    fetched_at = datetime.now(timezone.utc).isoformat()
    result: dict[str, Any] = {
        "symbol": symbol.upper(),
        "fetched_at": fetched_at,
        "provider": "polygon",
        "source_docs": {
            "ticker_overview": "https://polygon.io/docs/rest/stocks/tickers/ticker-overview",
            "news": "https://polygon.io/docs/rest/stocks/news",
        },
        "ticker_details": None,
        "news": [],
        "warnings": [],
    }
    try:
        details = await client.get_ticker_details(symbol)
        result["ticker_details"] = _normalize_ticker_details(details)
    except Exception as exc:
        result["warnings"].append(f"ticker_details_unavailable: {exc}")
    try:
        news = await client.get_ticker_news(symbol, limit=news_limit)
        result["news"] = [_normalize_news_item(row) for row in news[:news_limit]]
        if not result["news"]:
            result["warnings"].append("no_recent_polygon_news")
    except Exception as exc:
        result["warnings"].append(f"news_unavailable: {exc}")
    finally:
        if close_client:
            await client.close()
    if result["ticker_details"] is None:
        result["warnings"].append("ticker_details_missing")
    return result


def ticker_research_input(
    *,
    alert_id: int,
    alert_type: str,
    symbol: str,
    direction: str | None,
    payload: dict[str, Any],
    sources: dict[str, Any],
) -> dict[str, Any]:
    return {
        "alert": {
            "id": alert_id,
            "type": alert_type,
            "symbol": symbol.upper(),
            "direction": direction,
            "entry": payload.get("entry"),
            "stop": payload.get("stop"),
            "target": payload.get("target") or payload.get("targets"),
            "rr": payload.get("rr"),
            "p2r": payload.get("p2r"),
            "rvol": payload.get("rvol"),
            "spread": payload.get("spread"),
            "l2": payload.get("l2"),
            "note": payload.get("note"),
        },
        "sources": sources,
        "source_quality": {
            "has_ticker_details": bool(sources.get("ticker_details")),
            "news_count": len(sources.get("news") or []),
            "warnings": sources.get("warnings") or [],
        },
    }


def generate_ticker_research_for_alert(
    *,
    alert_id: int,
    alert_type: str,
    symbol: str,
    direction: str | None,
    payload: dict[str, Any],
    sources: dict[str, Any] | None = None,
    client: OpenAITextClient | None = None,
    session_scope: Callable[[], AbstractContextManager] = get_session,
) -> int | None:
    client = client or OpenAITextClient.from_env(enabled_env="OPENAI_RESEARCH_ENABLED")
    if not client.enabled:
        return None

    with session_scope() as session:
        existing = latest_ai_artifact(
            session,
            artifact_type=TICKER_RESEARCH_TYPE,
            source_type=TICKER_RESEARCH_SOURCE_TYPE,
            source_id=alert_id,
        )
        if existing is not None:
            return int(existing.id)

    source_data = sources or asyncio.run(collect_ticker_research_sources(symbol=symbol))
    model = openai_model_from_env("OPENAI_RESEARCH_MODEL", CONFIG.openai.research_model)
    input_json = ticker_research_input(
        alert_id=alert_id,
        alert_type=alert_type,
        symbol=symbol,
        direction=direction,
        payload=payload,
        sources=source_data,
    )

    artifact_id: int
    with session_scope() as session:
        artifact = create_ai_artifact(
            session,
            artifact_type=TICKER_RESEARCH_TYPE,
            source_type=TICKER_RESEARCH_SOURCE_TYPE,
            source_id=alert_id,
            symbol=symbol,
            model=model,
            prompt_version=TICKER_RESEARCH_PROMPT_VERSION,
            input_json=input_json,
            status="running",
        )
        artifact_id = int(artifact.id)

    response = client.generate_text(
        model=model,
        instructions=TICKER_RESEARCH_INSTRUCTIONS,
        input_text=json.dumps(input_json, sort_keys=True, default=str),
        max_output_tokens=360,
        metadata={"feature": TICKER_RESEARCH_TYPE, "alert_type": alert_type, "symbol": symbol.upper()},
    )

    with session_scope() as session:
        artifact = session.get(AIArtifact, artifact_id)
        if artifact is None:
            logger.warning("ticker_research.artifact_missing", artifact_id=artifact_id, alert_id=alert_id)
            return artifact_id
        if response.ok:
            complete_ai_artifact(artifact, output_text=response.text, output_json=response.output_json)
        else:
            fail_ai_artifact(artifact, error=response.error or response.status)
    return artifact_id


def _normalize_ticker_details(details: dict[str, Any] | None) -> dict[str, Any] | None:
    if not details:
        return None
    return {
        "ticker": details.get("ticker"),
        "name": details.get("name"),
        "market": details.get("market"),
        "locale": details.get("locale"),
        "primary_exchange": details.get("primary_exchange"),
        "type": details.get("type"),
        "active": details.get("active"),
        "currency_name": details.get("currency_name"),
        "cik": details.get("cik"),
        "market_cap": details.get("market_cap"),
        "share_class_shares_outstanding": details.get("share_class_shares_outstanding"),
        "weighted_shares_outstanding": details.get("weighted_shares_outstanding"),
        "sic_description": details.get("sic_description"),
        "homepage_url": details.get("homepage_url"),
        "list_date": details.get("list_date"),
    }


def _normalize_news_item(row: dict[str, Any]) -> dict[str, Any]:
    publisher = row.get("publisher") if isinstance(row.get("publisher"), dict) else {}
    insights = row.get("insights") if isinstance(row.get("insights"), list) else []
    return {
        "id": row.get("id"),
        "title": row.get("title"),
        "publisher": publisher.get("name"),
        "published_utc": row.get("published_utc"),
        "article_url": row.get("article_url"),
        "description": row.get("description"),
        "tickers": row.get("tickers") or [],
        "insights": [
            {
                "ticker": item.get("ticker"),
                "sentiment": item.get("sentiment"),
                "sentiment_reasoning": item.get("sentiment_reasoning"),
            }
            for item in insights[:3]
            if isinstance(item, dict)
        ],
    }
