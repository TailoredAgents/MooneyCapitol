from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from app.copier.models import MasterExecutionEvent
from app.copier.runtime import _event_payloads, _is_fill_payload
from app.core.config import CopierConfig


class WebullEventCapture:
    def __init__(self, output_path: Path, config: CopierConfig) -> None:
        self.output_path = output_path
        self.config = config
        self.count = 0
        self._lock = Lock()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def handle_event(self, *args: Any) -> dict[str, Any]:
        record = analyze_webull_event_args(args, self.config)
        with self._lock:
            self.count += 1
            record["sequence"] = self.count
            with self.output_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        return record


def analyze_webull_event_args(args: tuple[Any, ...] | list[Any], config: CopierConfig) -> dict[str, Any]:
    payloads = list(_event_payloads(args))
    analyses = [_analyze_payload(payload, config) for payload in payloads]
    return {
        "received_at": datetime.now(tz=timezone.utc).isoformat(),
        "status": _overall_status(analyses, payloads),
        "raw_args": [_safe_jsonable(arg) for arg in args],
        "payload_count": len(payloads),
        "payloads": [_safe_jsonable(payload) for payload in payloads],
        "analyses": analyses,
    }


def analyze_capture_file(path: Path, config: CopierConfig) -> list[dict[str, Any]]:
    rows = _load_capture_rows(path)
    results: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        if isinstance(row, dict) and "raw_args" in row:
            result = analyze_webull_event_args(tuple(row.get("raw_args") or []), config)
        elif isinstance(row, dict):
            result = analyze_webull_event_args((row,), config)
        else:
            result = {
                "received_at": datetime.now(tz=timezone.utc).isoformat(),
                "status": "unsupported_row",
                "raw_args": [_safe_jsonable(row)],
                "payload_count": 0,
                "payloads": [],
                "analyses": [],
            }
        result["sequence"] = idx
        results.append(result)
    return results


def summarize_capture(records: list[dict[str, Any]]) -> dict[str, Any]:
    parsed = 0
    ignored = 0
    parse_errors = 0
    no_payload = 0
    symbols: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for record in records:
        status = str(record.get("status") or "unknown")
        statuses[status] = statuses.get(status, 0) + 1
        if status == "parsed":
            parsed += 1
        elif status == "no_payload":
            no_payload += 1
        elif status == "parse_error":
            parse_errors += 1
        elif status in {"ignored", "mixed"}:
            ignored += 1
        for analysis in record.get("analyses") or []:
            normalized = analysis.get("normalized") or {}
            symbol = normalized.get("symbol")
            if symbol:
                symbols[symbol] = symbols.get(symbol, 0) + 1
    return {
        "total": len(records),
        "parsed": parsed,
        "ignored": ignored,
        "parse_errors": parse_errors,
        "no_payload": no_payload,
        "statuses": statuses,
        "symbols": symbols,
    }


def _analyze_payload(payload: dict[str, Any], config: CopierConfig) -> dict[str, Any]:
    is_fill = _is_fill_payload(payload, config)
    result: dict[str, Any] = {
        "is_fill_payload": is_fill,
        "status": "ignored" if not is_fill else "fill_detected",
        "payload_keys": sorted(str(key) for key in payload.keys()),
    }
    if not is_fill:
        return result
    try:
        event = MasterExecutionEvent.from_webull_payload(payload)
    except ValueError as exc:
        result.update({"status": "parse_error", "error": str(exc)})
        return result
    normalized = asdict(event)
    normalized["executed_at"] = event.executed_at.isoformat()
    result.update({"status": "parsed", "normalized": normalized})
    return result


def _overall_status(analyses: list[dict[str, Any]], payloads: list[dict[str, Any]]) -> str:
    if not payloads:
        return "no_payload"
    statuses = {str(item.get("status")) for item in analyses}
    if "parse_error" in statuses:
        return "parse_error"
    if "parsed" in statuses:
        return "parsed" if statuses <= {"parsed"} else "mixed"
    return "ignored"


def _load_capture_rows(path: Path) -> list[Any]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    if isinstance(parsed, list):
        return parsed
    return [parsed]


def _safe_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_jsonable(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    return repr(value)
