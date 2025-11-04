from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape


router = APIRouter()

env = Environment(
    loader=FileSystemLoader("app/templates"), autoescape=select_autoescape(["html", "xml"])
)


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    template = env.get_template("dashboard.html")
    return template.render()

