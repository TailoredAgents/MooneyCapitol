from __future__ import annotations

import re
from typing import Any, Mapping


_SENSITIVE_KEY = re.compile(
    r"(?:"
    r"(?:^|[_-])(?:secret|token|password|passwd|credential|api[_-]?key|private[_-]?key)(?:$|[_-])"
    r"|(?:^|[_-])(?:rithmic[_-]?)?(?:user|username|user_name|user_id)$"
    r"|(?:^|[_-])(?:fcm|fcm_id|ib|ib_id)$"
    r"|(?:^|[_-])(?:account|account_id|account_ref|broker_account_id)$"
    r"|(?:^|[_-])(?:basket_id|original_basket_id|parent_basket_id|linked_basket_ids)$"
    r"|(?:^|[_-])(?:exchange_order_id|ticker_plant_exchange_order_id|tp_exchange_order_id)$"
    r"|(?:^|[_-])fill_id$"
    # Generic application ``tags`` are non-secret classification labels used
    # by the legacy journal. Redact only broker/protocol tag identities.
    r"|(?:^|[_-])(?:user_tag|rithmic_tag|broker_tag|order_tag)$"
    r"|(?:^|[_-])(?:first_name|last_name|full_name|email|email_address|phone|phone_number|"
    r"address|mac_addr)(?:$|[_-])"
    r"|^(?:payload|payload_json|raw_payload|raw_payload_json|raw_protocol_payload|raw_message|"
    r"decoded_payload|decoded_protocol_payload|decoded_message|protocol_payload|wire_payload|"
    r"raw_frame|raw_frame_b64|wire_frame|protobuf_frame|serialized_frame|frame_bytes)$"
    r")",
    re.I,
)

_INLINE_SENSITIVE_VALUE = re.compile(
    r"(?i)\b("
    r"password|passwd|secret|token|credential|api[_-]?key|"
    r"rithmic[_-]?(?:user|username)|user_name|fcm_id|ib_id|account_id|account_ref|"
    r"broker_account_id|basket_id|exchange_order_id|fill_id|user_tag|email|phone|address"
    r")\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_URL_CREDENTIALS = re.compile(r"(?i)\b(wss?://)[^\s/@:]+:[^\s/@]+@")


def _redact_text(value: str) -> str:
    value = _INLINE_SENSITIVE_VALUE.sub(lambda match: f"{match.group(1)}=[REDACTED]", value)
    return _URL_CREDENTIALS.sub(r"\1[REDACTED]@", value)


def _sensitive_key(value: object) -> bool:
    # Protobuf-to-dict tooling can emit either snake_case field names or JSON
    # camelCase names.  Normalize both before applying one centralized policy.
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(value))
    return bool(_SENSITIVE_KEY.search(normalized))


def redact_sensitive(value: Any) -> Any:
    """Recursively redact broker identifiers, credentials, PII, and raw payloads."""
    if isinstance(value, Mapping):
        return {
            key: "[REDACTED]" if _sensitive_key(key) else redact_sensitive(child)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive(item) for item in value)
    if isinstance(value, str):
        return _redact_text(value)
    return value


def redact_log_event(_logger, _method_name: str, event_dict: dict) -> dict:
    return redact_sensitive(event_dict)
