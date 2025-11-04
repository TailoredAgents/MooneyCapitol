import pytest

from app.core.rr import project_rr


def test_project_rr_long():
    assert project_rr("long", entry=4.12, stop=4.06, target=4.22) == pytest.approx((4.22 - 4.12) / (4.12 - 4.06))


def test_project_rr_short():
    assert project_rr("short", entry=7.5, stop=7.7, target=7.2) == pytest.approx((7.5 - 7.2) / (7.7 - 7.5))


def test_project_rr_invalid():
    assert project_rr("long", entry=4.0, stop=4.1, target=4.2) is None
    assert project_rr("short", entry=4.0, stop=3.9, target=4.1) is None
