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


def test_dashboard_v3_separates_trader_and_owner_navigation():
    html = Path("app/templates/dashboard.html").read_text(encoding="utf-8")

    assert "<h1>Mooney Capitol</h1>" in html
    assert "status-indicators" not in html
    assert 'id="ws-status"' not in html
    assert "Operator Console" in html
    assert "V3 Operator Console" not in html
    assert "Trader Workspace" in html
    assert "Developer Setup" in html
    assert "Owner Setup" not in html
    assert "Copier Settings" in html
    assert "Copier Setup" not in html
    assert "Launch Readiness" in html
    assert 'class="tab-panel owner-panel"' in html
    assert "Developer setup controls can affect live trading" not in html
    assert "Emergency Stop" in html
