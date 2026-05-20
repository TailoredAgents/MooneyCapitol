from pathlib import Path


def test_dashboard_exposes_pnl_tab_and_existing_pnl_api_calls():
    html = Path("app/templates/dashboard.html").read_text(encoding="utf-8")

    assert 'data-tab="pnl"' in html
    assert 'id="tab-pnl"' in html
    assert "Refresh From Webull" in html
    assert "api('/pnl/summary')" in html
    assert "api('/pnl/refresh'" in html
    assert "/pnl/accounts/" in html


def test_dashboard_exposes_launch_readiness_tab_and_api_call():
    html = Path("app/templates/dashboard.html").read_text(encoding="utf-8")

    assert 'data-tab="launch"' in html
    assert 'id="tab-launch"' in html
    assert "Refresh Readiness" in html
    assert "api('/launch/readiness')" in html
    assert "renderLaunchReadiness" in html
    assert "Read-Only Validation History" in html
    assert 'id="launch-readonly-body"' in html
    assert "read_only_history" in html
