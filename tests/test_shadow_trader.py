from __future__ import annotations

import app.services.shadow_trader as shadow_trader_module
from app.api.routes import shadow
from app.services.shadow_trader import evaluate_shadow_decision


def test_shadow_decision_would_take_high_quality_trigger(monkeypatch):
    monkeypatch.setenv("SHADOW_TRADER_MIN_P2R", "0.70")
    monkeypatch.setenv("SHADOW_TRADER_MIN_RR", "2.00")
    monkeypatch.setenv("SHADOW_TRADER_SIZE_PCT", "0.05")
    monkeypatch.setattr(shadow_trader_module, "_predict_size", lambda features: None)
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


def test_shadow_uses_predicted_size_when_model_available(monkeypatch):
    monkeypatch.setattr(shadow_trader_module, "_predict_size", lambda features: 0.15)
    payload = {
        "entry": "5.00",
        "stop": "4.80",
        "rr": "2.50",
        "p2r": "0.82",
        "features": {"rvol_break": 2.5, "l2_mean": 0.75},
    }

    decision = evaluate_shadow_decision(payload, "trigger")

    assert decision["would_take"] is True
    assert decision["suggested_size_pct"] == 0.15


def test_shadow_falls_back_to_env_size_when_no_model(monkeypatch):
    monkeypatch.setenv("SHADOW_TRADER_SIZE_PCT", "0.05")
    monkeypatch.setattr(shadow_trader_module, "_predict_size", lambda features: None)
    payload = {
        "entry": "5.00",
        "stop": "4.80",
        "rr": "2.50",
        "p2r": "0.82",
    }

    decision = evaluate_shadow_decision(payload, "trigger")

    assert decision["would_take"] is True
    assert decision["suggested_size_pct"] == 0.05


def test_shadow_size_is_none_when_not_taking(monkeypatch):
    monkeypatch.setattr(shadow_trader_module, "_predict_size", lambda features: 0.20)
    payload = {
        "entry": "5.00",
        "stop": "4.80",
        "rr": "1.50",  # below min RR
        "p2r": "0.82",
    }

    decision = evaluate_shadow_decision(payload, "trigger")

    assert decision["would_take"] is False
    assert decision["suggested_size_pct"] is None
