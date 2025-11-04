from datetime import datetime, timedelta

from app.workers.budget import PingBudget


def test_ping_budget_limits():
    budget = PingBudget(limit=2, window_seconds=60)
    now = datetime.utcnow()
    assert budget.allow(now) is True
    assert budget.allow(now + timedelta(seconds=1)) is True
    assert budget.allow(now + timedelta(seconds=2)) is False
    # after window expires
    assert budget.allow(now + timedelta(seconds=61)) is True
