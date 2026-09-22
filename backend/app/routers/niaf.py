"""
NIAF-Bench pages.

A standalone surface, deliberately independent of the Distill product pages and
of the MCP server: no auth dependency, its own template directory, its own URL
prefix. This is the placeholder home page that the benchmark UI grows into.
"""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

BENCH_NAME = "NIAF-Bench"

router = APIRouter(prefix="/niaf", tags=["niaf-bench"])

templates_path = Path(__file__).parent.parent / "templates" / "niaf"
templates = Jinja2Templates(directory=str(templates_path))


@router.get("", response_class=HTMLResponse)
async def niaf_home(request: Request):
    """NIAF-Bench home page."""
    # request-first signature: supported by the pinned Starlette and by
    # current releases, which dropped the context-first form the other
    # page routes still use.
    return templates.TemplateResponse(
        request,
        "home.html",
        {"bench_name": BENCH_NAME},
    )
