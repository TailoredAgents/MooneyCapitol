from app.copier.engine import CopyResult
from app.copier.repository import copy_order_status_for_result, load_copy_positions


class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def all(self):
        return self.rows


class FakeSession:
    def __init__(self, rows):
        self.rows = rows

    def execute(self, stmt):
        return FakeResult(self.rows)


def test_load_copy_positions_estimates_active_target_exposure():
    session = FakeSession(
        [
            ("personal", "AAPL", "BUY", 10.0, None, "submitted"),
            ("personal", "AAPL", "SELL", 4.0, 4.0, "filled"),
            ("personal", "TSLA", "BUY", 3.0, 2.0, "partially_filled"),
            ("future", "AAPL", "BUY", 1.0, None, "submitted"),
        ]
    )

    positions = load_copy_positions(session, ["personal", "future"])

    assert positions["personal"]["AAPL"] == 6.0
    assert positions["personal"]["TSLA"] == 2.0
    assert positions["future"]["AAPL"] == 1.0


def test_copy_order_status_for_read_only_and_blocked_results():
    assert copy_order_status_for_result(
        CopyResult(target="personal", allowed=True, submitted=False, reason="read_only")
    ) == "would_copy"
    assert copy_order_status_for_result(
        CopyResult(target="personal", allowed=False, submitted=False, reason="target_disabled")
    ) == "blocked"
    assert copy_order_status_for_result(
        CopyResult(target="personal", allowed=True, submitted=False, reason="broker_submit_failed")
    ) == "submit_failed"
    assert copy_order_status_for_result(
        CopyResult(target="personal", allowed=True, submitted=True, reason=None)
    ) == "submitted"
