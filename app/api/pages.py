from __future__ import annotations

from urllib.parse import parse_qs

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.api.auth import (
    clear_dashboard_session_cookie,
    is_dashboard_authenticated,
    login_credentials_match,
    set_dashboard_session_cookie,
)


router = APIRouter()
PUBLIC_SITE_URL = "https://mooneytrading.com"

env = Environment(
    loader=FileSystemLoader("app/templates"), autoescape=select_autoescape(["html", "xml"])
)


def _render_login(error: str | None = None, status_code: int = 200) -> HTMLResponse:
    template = env.get_template("login.html")
    return HTMLResponse(template.render(error=error), status_code=status_code)


def _render_public_policy(
    *,
    title: str,
    description: str,
    eyebrow: str,
    updated: str,
    path: str,
    sections: list[dict[str, object]],
) -> HTMLResponse:
    template = env.get_template("public_policy.html")
    canonical_url = f"{PUBLIC_SITE_URL}{path}"
    return HTMLResponse(
        template.render(
            title=title,
            description=description,
            eyebrow=eyebrow,
            updated=updated,
            canonical_url=canonical_url,
            active_path=path,
            sections=sections,
        )
    )


@router.get("/", response_class=HTMLResponse)
def public_home():
    template = env.get_template("public_home.html")
    return template.render()


@router.get("/robots.txt", include_in_schema=False)
def robots_txt():
    body = "\n".join(
        [
            "User-agent: *",
            "Allow: /",
            "Disallow: /dashboard",
            "Disallow: /login",
            "Disallow: /health",
            "Disallow: /config",
            "Disallow: /watchlist",
            "Disallow: /setups",
            "Disallow: /reports",
            "Disallow: /learning",
            "Disallow: /copier",
            "Disallow: /launch",
            "Disallow: /pnl",
            "Disallow: /slack",
            "Disallow: /slash",
            "Disallow: /ws",
            f"Sitemap: {PUBLIC_SITE_URL}/sitemap.xml",
            "",
        ]
    )
    return PlainTextResponse(body)


@router.get("/sitemap.xml", include_in_schema=False)
def sitemap_xml():
    urls = [
        f"{PUBLIC_SITE_URL}/",
        f"{PUBLIC_SITE_URL}/platform",
        f"{PUBLIC_SITE_URL}/faq",
        f"{PUBLIC_SITE_URL}/risk-disclosure",
        f"{PUBLIC_SITE_URL}/privacy",
    ]
    url_entries = "\n".join(
        f"""  <url>
    <loc>{url}</loc>
    <changefreq>weekly</changefreq>
    <priority>{'1.0' if url.endswith('/') else '0.6'}</priority>
  </url>"""
        for url in urls
    )
    body = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
{url_entries}
</urlset>
"""
    return Response(content=body, media_type="application/xml")


@router.get("/platform", response_class=HTMLResponse)
def platform_overview():
    return _render_public_policy(
        title="Platform Overview",
        description="A plain-language overview of the Mooney Trading private trading operations platform.",
        eyebrow="Platform overview",
        updated="May 23, 2026",
        path="/platform",
        sections=[
            {
                "heading": "What Mooney Trading is",
                "items": [
                    "Mooney Trading is a private trading operations platform built around a human trader, monitored account data, and internal operator controls.",
                    "The system combines a live scout, copy execution oversight, account monitoring, latency tracking, and review reports in one authenticated internal console.",
                    "The public site explains the platform at a high level and does not provide public account access, public trading signals, or investor onboarding.",
                ],
            },
            {
                "heading": "Trader workflow",
                "items": [
                    "The scout organizes potential setups into readiness lanes so the trader can review market context faster.",
                    "The trader remains responsible for decisions on the master account.",
                    "Configured copy accounts are intended to mirror exposure by account percentage when copying is enabled and live-trading gates are satisfied.",
                ],
            },
            {
                "heading": "Operator workflow",
                "items": [
                    "The internal console separates trader-facing views from developer and launch-readiness controls.",
                    "Operators can monitor account value, positions, fills, copied order status, latency, slippage, and system readiness.",
                    "Operational controls, credentials, account identifiers, readiness checks, and execution settings stay inside authenticated access.",
                ],
            },
            {
                "heading": "Learning and review",
                "items": [
                    "The learning layer uses alerts, setups, trades, fills, and outcomes to support explainable review.",
                    "AI summaries are read-only helpers for explanations, research context, trade journals, daily recaps, and learning summaries.",
                    "Model outputs support human review and should not be treated as financial advice, guarantees, or autonomous public recommendations.",
                ],
            },
            {
                "heading": "Launch boundaries",
                "items": [
                    "Before real-money operation, brokerage credentials, market data, Slack, Sentry, OpenAI, database, and deployment settings must be configured and validated.",
                    "Live trading requires account-level validation, read-only checks, working equity snapshots, copier readiness, risk monitoring, and operator review.",
                    "Any future public-facing product would require separate legal, regulatory, brokerage, privacy, and operational review.",
                ],
            },
        ],
    )


@router.get("/faq", response_class=HTMLResponse)
def public_faq():
    return _render_public_policy(
        title="FAQ",
        description="Common questions about the Mooney Trading private trading operations platform.",
        eyebrow="Common questions",
        updated="May 23, 2026",
        path="/faq",
        sections=[
            {
                "heading": "Is Mooney Trading open to the public?",
                "items": [
                    "No. Mooney Trading is currently a private trading operations platform for authorized internal users.",
                    "The public site is informational and does not provide account access, signup, public trading signals, or investor onboarding.",
                    "The private internal console remains behind authenticated access and is not linked from the public website.",
                ],
            },
            {
                "heading": "What does the platform do?",
                "items": [
                    "The platform supports a trading workflow with live market scouting, account monitoring, copied-order oversight, latency tracking, and review reports.",
                    "It is built to help operators see the state of the trading system, account snapshots, copied executions, and post-trade review context in one place.",
                    "It is not presented as a public advisory service or a public investment product.",
                ],
            },
            {
                "heading": "Does the system trade by itself?",
                "items": [
                    "The current design keeps the human trader responsible for master-account decisions.",
                    "Copy execution is controlled by operator settings, launch-readiness gates, account validation, and risk monitoring.",
                    "AI features are read-only helpers for explanation, research context, trade journals, recaps, and learning summaries.",
                ],
            },
            {
                "heading": "Why does the public site mention risk?",
                "items": [
                    "Trading involves real financial risk, including possible loss of principal.",
                    "Software, brokers, market data, external APIs, and network connections can fail or behave unexpectedly.",
                    "The risk and privacy pages are included so public visitors understand the boundaries before any future product direction is considered.",
                ],
            },
            {
                "heading": "What would need to happen before a public product?",
                "items": [
                    "A public product would require separate legal, regulatory, brokerage, privacy, security, support, and operational review.",
                    "Customer onboarding, billing, account permissions, disclosures, agreements, and data handling would need to be designed before launch.",
                    "Nothing on the current public site should be treated as an offer to manage money, provide advice, or give public access to trading activity.",
                ],
            },
        ],
    )


@router.get("/risk-disclosure", response_class=HTMLResponse)
def risk_disclosure():
    return _render_public_policy(
        title="Risk Disclosure",
        description="Important risk information for visitors reviewing Mooney Trading.",
        eyebrow="Trading risk disclosure",
        updated="May 23, 2026",
        path="/risk-disclosure",
        sections=[
            {
                "heading": "Trading risk",
                "items": [
                    "Trading securities involves risk, including the possible loss of principal.",
                    "Market prices can move quickly, liquidity can change without warning, and execution quality can vary by broker, symbol, order type, and market condition.",
                    "Past trades, alerts, reports, simulations, or examples do not guarantee future results.",
                ],
            },
            {
                "heading": "No financial advice",
                "items": [
                    "Mooney Trading is a private trading operations technology platform.",
                    "Public information on this site is general and informational only.",
                    "Nothing on this site is investment advice, a recommendation, a trading signal for the public, or an instruction to buy or sell any security.",
                ],
            },
            {
                "heading": "No public copy-trading offer",
                "items": [
                    "This site does not offer public account management, advisory services, investor access, or a public copy-trading product.",
                    "The private internal console is reserved for authorized internal users.",
                    "Any future product, service, or investor-facing program would need separate legal, regulatory, brokerage, and operational review before launch.",
                ],
            },
            {
                "heading": "Technology limitations",
                "items": [
                    "Software can fail, data can be delayed or incorrect, external APIs can be unavailable, and copied orders can experience latency, slippage, rejection, or partial fills.",
                    "Monitoring tools, AI summaries, model outputs, and learning reports are decision-support tools, not guarantees of correctness or profitability.",
                    "Operators remain responsible for reviewing credentials, brokerage permissions, account settings, risk controls, and live-trading readiness before using real funds.",
                ],
            },
        ],
    )


@router.get("/privacy", response_class=HTMLResponse)
def privacy_policy():
    return _render_public_policy(
        title="Privacy Notice",
        description="High-level privacy notice for the Mooney Trading public site.",
        eyebrow="Privacy notice",
        updated="May 23, 2026",
        path="/privacy",
        sections=[
            {
                "heading": "Public site",
                "items": [
                    "The public site is informational and does not include a public account sign-up flow.",
                    "Visitors can view the homepage, risk disclosure, privacy notice, and public assets without accessing internal trading tools.",
                    "The public site should not be used to submit brokerage credentials, account numbers, private trading information, or personal financial data.",
                ],
            },
            {
                "heading": "Internal operator console",
                "items": [
                    "The internal console is separate from the public site and is intended only for authorized operators.",
                    "Internal activity can involve operational records such as account snapshots, alerts, fills, copied order records, latency measurements, AI summaries, and audit logs.",
                    "Brokerage credentials and API keys belong in Render environment variables or approved secret storage, not in public pages, screenshots, chat messages, or source code.",
                ],
            },
            {
                "heading": "Service providers",
                "items": [
                    "The deployed system can rely on infrastructure, monitoring, brokerage, market-data, Slack, Sentry, OpenAI, and database providers configured by the operator.",
                    "Those providers may process operational data needed to run, monitor, secure, or troubleshoot the system.",
                    "Provider access should be reviewed before any future public product or customer onboarding flow is launched.",
                ],
            },
            {
                "heading": "Contact and changes",
                "items": [
                    "This notice is a practical project notice, not a substitute for a lawyer-reviewed privacy policy.",
                    "Before collecting customer information, accepting investor interest, or offering a public service, this page should be replaced or reviewed by qualified counsel.",
                    "Material changes to the public site or internal data flows should be reflected here before launch.",
                ],
            },
        ],
    )


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if is_dashboard_authenticated(request):
        return RedirectResponse("/dashboard", status_code=303)
    return _render_login()


@router.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request):
    raw_body = (await request.body()).decode("utf-8")
    form = parse_qs(raw_body, keep_blank_values=True)
    username = form.get("username", [""])[0]
    password = form.get("password", [""])[0]

    if not login_credentials_match(username, password):
        return _render_login("Invalid username or password.", status_code=401)

    response = RedirectResponse("/dashboard", status_code=303)
    set_dashboard_session_cookie(response, request)
    return response


@router.post("/logout")
def logout(request: Request):
    response = RedirectResponse("/login", status_code=303)
    clear_dashboard_session_cookie(response, request)
    return response


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    if not is_dashboard_authenticated(request):
        return RedirectResponse("/login", status_code=303)
    template = env.get_template("dashboard.html")
    return template.render()

