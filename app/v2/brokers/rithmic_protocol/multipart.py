from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .fields import FieldView, is_exact_success_response, optional_text


@dataclass(frozen=True)
class ResponseProgress:
    correlation_id: str
    template_id: int
    intermediate_codes: tuple[str, ...]
    terminal_codes: tuple[str, ...]
    terminal: bool
    success: bool | None
    row_count: int


@dataclass
class _PendingResponse:
    expected_template_ids: frozenset[int]
    rows: int = 0
    terminal: bool = False
    intermediate_codes: list[str] = field(default_factory=list)
    terminal_codes: list[str] = field(default_factory=list)
    tainted: bool = False


def response_correlation(message: Mapping[str, Any] | Any) -> str | None:
    view = FieldView(message)
    # user_msg is the client-owned opaque correlation echoed by Rithmic. A
    # server request_key may be present and differ; it must not shadow the key
    # under which the local waiter was registered.
    user_messages = view.strings("user_msg")
    if user_messages:
        return user_messages[-1]
    request_key = optional_text(view.get("request_key"))
    if request_key:
        return request_key
    return None


class MultipartResponseTracker:
    """Tracks official intermediate ``rq_handler_rp_code`` and terminal ``rp_code``."""

    def __init__(self) -> None:
        self._pending: dict[str, _PendingResponse] = {}

    def begin(self, correlation_id: str, expected_template_ids: set[int] | frozenset[int]) -> None:
        if not correlation_id:
            raise ValueError("correlation_id is required")
        if correlation_id in self._pending:
            raise ValueError("correlation_id is already pending")
        self._pending[correlation_id] = _PendingResponse(frozenset(expected_template_ids))

    def consume(
        self,
        template_id: int,
        message: Mapping[str, Any] | Any,
        *,
        correlation_id: str | None = None,
    ) -> ResponseProgress | None:
        key = correlation_id or response_correlation(message)
        if not key or key not in self._pending:
            return None
        pending = self._pending[key]
        if int(template_id) not in pending.expected_template_ids:
            return None
        view = FieldView(message)
        intermediate = view.strings("rq_handler_rp_code")
        terminal_codes = view.strings("rp_code")
        if intermediate:
            pending.rows += 1
            pending.intermediate_codes.extend(intermediate)
            if not is_exact_success_response(intermediate):
                pending.tainted = True
        if terminal_codes:
            pending.terminal = True
            pending.terminal_codes.extend(terminal_codes)
        progress = ResponseProgress(
            correlation_id=key,
            template_id=int(template_id),
            intermediate_codes=tuple(pending.intermediate_codes),
            terminal_codes=tuple(pending.terminal_codes),
            terminal=pending.terminal,
            success=(
                is_exact_success_response(pending.terminal_codes) and not pending.tainted
                if pending.terminal_codes
                else None
            ),
            row_count=pending.rows,
        )
        if pending.terminal:
            del self._pending[key]
        return progress

    def abandon(self, correlation_id: str) -> None:
        self._pending.pop(correlation_id, None)

    @property
    def pending_count(self) -> int:
        return len(self._pending)
