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
    body = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>{PUBLIC_SITE_URL}/</loc>
    <changefreq>weekly</changefreq>
    <priority>1.0</priority>
  </url>
</urlset>
"""
    return Response(content=body, media_type="application/xml")


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

