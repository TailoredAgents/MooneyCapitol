from __future__ import annotations

from app.api.routes import shadow
from app.services.shadow_trader import evaluate_shadow_decision


def test_shadow_decision_would_take_high_quality_trigger(monkeypatch):
    monkeypatch.setenv("SHADOW_TRADER_MIN_P2R", "0.70")
    monkeypatch.setenv("SHADOW_TRADER_MIN_RR", "2.00")
    payload = {
        "entry": "4.25",
        "stop": "4.05",
        "primary_target": 4.75,
        "rr": "2.50",
        "p2r": "0.82",
    }

    decision = evaluate_shadow_decision(payload, "trigger")

    assert decision["would_take"] is True
    assert decision["decision"] == "would_take"
    assert decision["confidence"] == "medium"
    assert decision["suggested_size_pct"] == 0.05
    assert decision["entry_price"] == 4.25


def test_shadow_decision_skips_below_learned_threshold():
    payload = {
        "entry": "4.25",
        "stop": "4.05",
        "rr": "3.00",
        "p2r": "0.90",
        "note": "below learned threshold",
    }

    decision = evaluate_shadow_decision(payload, "trigger")

    assert decision["would_take"] is False
    assert decision["decision"] == "skip"
    assert "below learned threshold" in decision["reason"]


def test_shadow_decisions_route_is_registered():
    paths = {route.path for route in shadow.router.routes}

    assert "/shadow/decisions" in paths


def test_shadow_route_returns_service_payload(monkeypatch):
    monkeypatch.setattr(
        shadow,
        "list_shadow_decisions",
        lambda limit: {"items": [], "count": 0, "summary": {"total": 0}, "limit": limit},
    )

    payload = shadow.get_shadow_decisions(limit=12)

    assert payload["count"] == 0
    assert payload["limit"] == 12
