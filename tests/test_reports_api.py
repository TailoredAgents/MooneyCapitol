from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace

from app.api.routes import reports


class FakeLedger:
    def daily_summary(self, trade_date):
        return {
            "date": str(trade_date),
            "total_trades": 0,
            "wins": 0,
            "win_rate": 0.0,
            "avg_r": 0.0,
            "expectancy": 0.0,
            "net_pnl": 0.0,
            "by_setup": [],
            "by_hour": {},
        }


def test_get_eod_report_includes_latest_ai_recap(monkeypatch):
    monkeypatch.setattr(reports, "get_ledger_service", lambda: FakeLedger())
    monkeypatch.setattr(reports, "_latest_daily_recap", lambda trade_date: {"text": "Quiet day."})

    payload = reports.get_eod_report(date=date(2026, 5, 22))

    assert payload["date"] == "2026-05-22"
    assert payload["ai_recap"] == {"text": "Quiet day."}


def test_get_eod_ai_recap_returns_empty_state(monkeypatch):
    monkeypatch.setattr(reports, "_latest_daily_recap", lambda trade_date: None)

    payload = reports.get_eod_ai_recap(date=date(2026, 5, 22))

    assert payload == {"date": "2026-05-22", "ai_recap": None}


def test_latest_daily_recap_serializes_completed_artifact(monkeypatch):
    created_at = datetime(2026, 5, 22, 21, 0, tzinfo=timezone.utc)
    artifact = SimpleNamespace(
        status="completed",
        output_text="Copier stayed under target latency.",
        model="gpt-5.4",
        created_at=created_at,
    )

    class FakeSession:
        pass

    class FakeScope:
        def __enter__(self):
            return FakeSession()

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(reports, "get_session", lambda: FakeScope())
    monkeypatch.setattr(reports, "latest_ai_artifact", lambda *args, **kwargs: artifact)

    result = reports._latest_daily_recap(date(2026, 5, 22))

    assert result == {
        "text": "Copier stayed under target latency.",
        "model": "gpt-5.4",
        "created_at": created_at.isoformat(),
    }
