from pathlib import Path


def test_dashboard_exposes_pnl_tab_and_existing_pnl_api_calls():
    html = Path("app/templates/dashboard.html").read_text(encoding="utf-8")

    assert 'data-tab="pnl"' in html
    assert 'id="tab-pnl"' in html
    assert "Refresh From Webull" in html
    assert "Account Monitor" in html
    assert "Account P&L" not in html
    assert "api('/pnl/summary')" in html
    assert "api('/pnl/refresh'" in html
    assert "/pnl/accounts/" in html
    assert 'id="pnl-total-exposure"' in html
    assert 'id="pnl-last-refresh"' in html
    assert "moneySpan(summary.total_pnl_today)" in html
    assert "No account snapshots yet" in html
    assert "No active risk alerts" in html


def test_dashboard_exposes_launch_readiness_tab_and_api_call():
    html = Path("app/templates/dashboard.html").read_text(encoding="utf-8")

    assert 'data-tab="launch"' in html
    assert 'id="tab-launch"' in html
    assert "Refresh Readiness" in html
    assert "Pre-launch checklist for credentials, account data, worker health, and copier safety." in html
    assert "Last Checked" in html
    assert "Fix First" in html
    assert 'id="launch-fix-first-list"' in html
    assert "Readiness API" in html
    assert "System Snapshot" in html
    assert "No validation decisions yet" in html
    assert "api('/launch/readiness')" in html
    assert "renderLaunchReadiness" in html
    assert "Read-Only Validation History" in html
    assert 'id="launch-readonly-body"' in html
    assert "read_only_history" in html


def test_pages_include_png_favicon_link():
    dashboard = Path("app/templates/dashboard.html").read_text(encoding="utf-8")
    login = Path("app/templates/login.html").read_text(encoding="utf-8")

    assert 'href="/static/favicon-32.png"' in dashboard
    assert 'href="/static/favicon-32.png"' in login
    assert 'href="/static/favicon.png"' in dashboard
    assert 'href="/static/apple-touch-icon.png"' in login


def test_dashboard_v3_separates_trader_and_owner_navigation():
    html = Path("app/templates/dashboard.html").read_text(encoding="utf-8")

    assert "<h1>Mooney Capital</h1>" in html
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


def test_copier_settings_uses_control_console_layout():
    html = Path("app/templates/dashboard.html").read_text(encoding="utf-8")

    assert "Control copy mode, target accounts, and emergency stop behavior." in html
    assert "Copier State" in html
    assert "Launch Blockers" in html
    assert "Copy Controls" in html
    assert "Save Copy Settings" in html
    assert "Master Equity Override" in html
    assert "Usually pulled from Webull" in html
    assert "Copy Account" in html
    assert "Equity Source / Override" in html
    assert 'id="readiness-blocker-count"' in html
    assert 'id="readiness-warning-count"' in html
    assert 'id="readiness-passing-count"' in html
    assert "These settings can affect real orders." not in html


def test_scout_page_has_trader_cockpit_status_and_lane_counts():
    html = Path("app/templates/dashboard.html").read_text(encoding="utf-8")

    assert "Live Scout" in html
    assert "scout-status-grid" in html
    assert 'id="scout-scanner-status"' in html
    assert 'id="scout-ws-status"' in html
    assert 'id="scout-symbol-count"' in html
    assert 'id="scout-data-mode"' in html
    assert 'id="scout-last-update"' in html
    assert 'id="armed-count"' in html
    assert 'id="primed-count"' in html
    assert 'id="active-count"' in html
    assert "Selected Symbol Depth" in html
    assert "Select an alert to view market depth" in html
    assert "AI Research" in html
    assert 'id="scout-ai-research"' in html
    assert "AI explanation" in html
    assert "item.ai_explanation" in html
    assert "Catalyst research" in html
    assert "item.ticker_research" in html
    assert "renderLane(armedList, lanes.armed, 'No armed boxes', 'armed')" in html


def test_trade_monitor_has_summary_cards_and_latency_chips():
    html = Path("app/templates/dashboard.html").read_text(encoding="utf-8")

    assert "Trade Monitor" in html
    assert "Trade Activity" in html
    assert "Trade Results" not in html
    assert 'id="trade-total-count"' in html
    assert 'id="trade-copied-count"' in html
    assert 'id="trade-blocked-count"' in html
    assert 'id="trade-avg-latency"' in html
    assert 'id="trade-under-300"' in html
    assert "renderTradeSummary(items)" in html
    assert "latencyChip(item.copy_latency_ms)" in html
    assert "item.ai_journal" in html
    assert "No trade activity yet" in html
    assert "Daily AI Recap" in html
    assert "renderDailyRecap" in html
    assert "/reports/eod/ai" in html


def test_launch_page_displays_ai_learning_translation_when_available():
    html = Path("app/templates/dashboard.html").read_text(encoding="utf-8")

    assert "summary.learning_translation" in html
    assert "AI Learning Summary" in html


def test_reports_api_exposes_ai_daily_recap_route():
    source = Path("app/api/routes/reports.py").read_text(encoding="utf-8")

    assert "/reports/eod/ai" in source
    assert "ai_recap" in source


def test_dashboard_has_mobile_friendly_navigation_and_summary_grids():
    html = Path("app/templates/dashboard.html").read_text(encoding="utf-8")

    assert "@media (max-width: 720px)" in html
    assert ".nav-group:first-of-type .tabs" in html
    assert "grid-template-columns: repeat(3, minmax(0, 1fr));" in html
    assert ".developer-nav" in html
    assert "display: none;" in html
    assert "@media (hover: none)" in html
    assert "box-shadow: inset -16px 0 16px -18px" in html
