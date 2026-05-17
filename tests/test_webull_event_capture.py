import json

from app.copier.event_capture import WebullEventCapture, analyze_capture_file, analyze_webull_event_args, summarize_capture
from app.core.config import CopierConfig


def _payload(status="FILLED"):
    return {
        "account_id": "master",
        "order_id": "order-1",
        "symbol": "AAPL",
        "side": "BUY",
        "status": status,
        "filled_qty": "3",
        "avg_fill_price": "12.50",
        "filled_at": "2026-05-16T14:30:00+00:00",
        "instrument_type": "EQUITY",
    }


def test_analyze_webull_event_args_parses_fill_payload():
    record = analyze_webull_event_args(("topic", "event", _payload(), None), CopierConfig())

    assert record["status"] == "parsed"
    assert record["payload_count"] == 1
    assert record["analyses"][0]["normalized"]["symbol"] == "AAPL"
    assert record["analyses"][0]["normalized"]["quantity"] == 3.0


def test_analyze_webull_event_args_ignores_non_fill_payload():
    record = analyze_webull_event_args((_payload(status="NEW"),), CopierConfig())

    assert record["status"] == "ignored"
    assert record["analyses"][0]["is_fill_payload"] is False


def test_webull_event_capture_writes_jsonl(tmp_path):
    output = tmp_path / "capture.jsonl"
    capture = WebullEventCapture(output, CopierConfig())

    capture.handle_event(_payload())
    capture.handle_event(_payload(status="NEW"))

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["sequence"] == 1
    assert json.loads(lines[1])["status"] == "ignored"


def test_analyze_capture_file_accepts_jsonl_and_summarizes(tmp_path):
    output = tmp_path / "capture.jsonl"
    capture = WebullEventCapture(output, CopierConfig())
    capture.handle_event(_payload())
    capture.handle_event(_payload(status="NEW"))

    records = analyze_capture_file(output, CopierConfig())
    summary = summarize_capture(records)

    assert summary["total"] == 2
    assert summary["parsed"] == 1
    assert summary["ignored"] == 1
    assert summary["symbols"] == {"AAPL": 1}
