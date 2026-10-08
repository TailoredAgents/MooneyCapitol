from __future__ import annotations

from app.security import redact_log_event, redact_sensitive


def test_rithmic_identifiers_pii_credentials_and_payloads_are_redacted():
    event = {
        "rithmic_username": "synthetic-user",
        "password": "synthetic-password",
        "fcm_id": "synthetic-fcm",
        "ib_id": "synthetic-ib",
        "broker_account_id": "synthetic-account",
        "basket_id": "synthetic-basket",
        "original_basket_id": "synthetic-original",
        "linked_basket_ids": ["synthetic-linked"],
        "exchange_order_id": "synthetic-exchange-order",
        "tp_exchange_order_id": "synthetic-ticker-order",
        "fill_id": "synthetic-fill",
        "user_tag": "synthetic-tag",
        "email_address": "person.invalid@example.invalid",
        "phone_number": "555-0100",
        "postal_address": "synthetic street",
        "raw_payload": {"unrecognized_vendor_field": "sensitive"},
        "decoded_message": {"another_vendor_field": "sensitive"},
        "rawProtocolPayload": b"synthetic-wire-frame",
        "raw_frame_b64": "c3ludGhldGljLXdpcmUtZnJhbWU=",
        "wireFrame": b"synthetic-wire-frame",
        "protobuf_frame": b"synthetic-wire-frame",
        "exchangeOrderId": "synthetic-camel-case-order",
        "linkedBasketIds": ["synthetic-camel-case-linked"],
        "firstName": "Synthetic",
        "last_name": "Person",
    }

    redacted = redact_sensitive(event)

    assert set(redacted.values()) == {"[REDACTED]"}


def test_redaction_recurses_without_hiding_safe_operational_metadata():
    event = {
        "plant": "ORDER",
        "state": "READY",
        "payload_fingerprint": "sha256:synthetic",
        "nested": [
            {"account_id": "synthetic-account", "template_id": 351},
            ({"user_tag": "synthetic-tag"},),
        ],
    }

    redacted = redact_sensitive(event)

    assert redacted["plant"] == "ORDER"
    assert redacted["state"] == "READY"
    assert redacted["payload_fingerprint"] == "sha256:synthetic"
    assert redacted["nested"][0] == {
        "account_id": "[REDACTED]",
        "template_id": 351,
    }
    assert redacted["nested"][1] == ({"user_tag": "[REDACTED]"},)
    assert event["nested"][0]["account_id"] == "synthetic-account"


def test_generic_application_tags_remain_while_broker_tags_are_redacted():
    redacted = redact_sensitive(
        {
            "tags": ["copied", "under_300ms"],
            "rithmic_tag": "synthetic-rithmic-tag",
            "brokerTag": "synthetic-broker-tag",
            "order_tag": "synthetic-order-tag",
        }
    )

    assert redacted["tags"] == ["copied", "under_300ms"]
    assert redacted["rithmic_tag"] == "[REDACTED]"
    assert redacted["brokerTag"] == "[REDACTED]"
    assert redacted["order_tag"] == "[REDACTED]"


def test_labeled_values_and_url_credentials_embedded_in_log_text_are_redacted():
    text = (
        "login password=synthetic-password account_id='synthetic-account' "
        "fill_id=synthetic-fill "
        "wss://synthetic-user:synthetic-password@example.invalid/path"
    )

    redacted = redact_sensitive(text)

    assert "synthetic-password" not in redacted
    assert "synthetic-account" not in redacted
    assert "synthetic-fill" not in redacted
    assert "synthetic-user" not in redacted
    assert "password=[REDACTED]" in redacted
    assert "account_id=[REDACTED]" in redacted
    assert "fill_id=[REDACTED]" in redacted
    assert "wss://[REDACTED]@example.invalid/path" in redacted


def test_structured_log_processor_uses_central_redaction():
    event = {
        "event": "rithmic_connected",
        "rithmic_user": "synthetic-user",
        "event_facts": {"basket_id": "synthetic-basket", "template_id": 351},
    }

    redacted = redact_log_event(None, "info", event)

    assert redacted == {
        "event": "rithmic_connected",
        "rithmic_user": "[REDACTED]",
        "event_facts": {"basket_id": "[REDACTED]", "template_id": 351},
    }
