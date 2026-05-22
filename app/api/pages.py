from __future__ import annotations

from urllib.parse import parse_qs

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.api.auth import (
    clear_dashboard_session_cookie,
    is_dashboard_authenticated,
    login_credentials_match,
    set_dashboard_session_cookie,
)


router = APIRouter()

env = Environment(
    loader=FileSystemLoader("app/templates"), autoescape=select_autoescape(["html", "xml"])
)


def _render_login(error: str | None = None, status_code: int = 200) -> HTMLResponse:
    template = env.get_template("login.html")
    return HTMLResponse(template.render(error=error), status_code=status_code)


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

