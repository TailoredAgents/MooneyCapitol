from datetime import datetime, timedelta, timezone

from app.copier.operator_alerts import (
    OperatorAlertThrottle,
    copy_limit_alert_text,
    listener_error_alert_text,
    position_sync_alert_text,
    readiness_alert_text,
    reconciliation_alert_text,
    recovery_alert_text,
)
from app.copier.engine import CopyResult
from app.copier.reconciliation import PositionSyncResult, ReconciliationResult, StartupRecoveryResult


def test_operator_alert_throttle_limits_repeated_keys():
    throttle = OperatorAlertThrottle(min_interval=timedelta(minutes=15))
    now = datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc)

    assert throttle.allow("readiness", now=now) is True
    assert throttle.allow("readiness", now=now + timedelta(minutes=5)) is False
    assert throttle.allow("readiness", now=now + timedelta(minutes=16)) is True


def test_readiness_alert_text_lists_blockers():
    text = readiness_alert_text(
        {
            "blockers": [
                {"key": "master_account", "label": "Webull master account id is configured"},
                {"key": "enabled_targets", "label": "At least one copy target is enabled"},
            ]
        }
    )

    assert "Copier readiness blockers detected" in text
    assert "Webull master account id" in text
    assert "At least one copy target" in text


def test_readiness_alert_text_returns_none_without_blockers():
    assert readiness_alert_text({"blockers": []}) is None


def test_reconciliation_alert_text_only_for_issues():
    assert reconciliation_alert_text(ReconciliationResult(checked=1, updated=1, mismatches=0, errors=0)) is None

    text = reconciliation_alert_text(ReconciliationResult(checked=2, updated=0, mismatches=1, errors=1))

    assert "Copier reconciliation issue" in text
    assert "mismatches: 1" in text
    assert "errors: 1" in text


def test_recovery_alert_text_for_startup_warnings():
    text = recovery_alert_text(
        StartupRecoveryResult(checked=3, updated=1, mismatches=0, errors=0, stale_open_orders=2)
    )

    assert "Copier startup recovery warning" in text
    assert "stale open orders: 2" in text


def test_position_sync_alert_text_for_mismatches():
    text = position_sync_alert_text(PositionSyncResult(targets_checked=1, symbols_checked=2, mismatches=1, errors=0))

    assert "Copier position sync issue" in text
    assert "mismatches: 1" in text


def test_listener_error_alert_text():
    assert "socket closed" in listener_error_alert_text("socket closed")


def test_copy_limit_alert_text_lists_limit_blocks():
    text = copy_limit_alert_text(
        [
            CopyResult(
                target="personal",
                allowed=False,
                submitted=False,
                reason="max_daily_notional_exceeded",
                quantity=10,
            )
        ]
    )

    assert "Copied trade blocked" in text
    assert "personal: max_daily_notional_exceeded" in text


def test_copy_limit_alert_text_ignores_non_limit_blocks():
    text = copy_limit_alert_text(
        [CopyResult(target="personal", allowed=False, submitted=False, reason="target_disabled")]
    )

    assert text is None


def test_copy_limit_alert_text_lists_short_blocks():
    text = copy_limit_alert_text(
        [CopyResult(target="personal", allowed=False, submitted=False, reason="short_copy_disabled", quantity=5)]
    )

    assert "short_copy_disabled" in text
