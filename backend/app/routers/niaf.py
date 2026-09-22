"""
NIAF-Bench pages.

A standalone surface, deliberately independent of the Distill product pages and
of the MCP server: no auth dependency, no database access, its own static
directory, its own URL prefix.

The page is served as a static file rather than a Jinja template on purpose.
It is entirely self-contained — every figure renders client-side from the
`DATA` object inside it — so there is nothing to interpolate, and its CSS and
JavaScript braces cannot be mistaken for template syntax. When real results
land, the page should fetch a scored run artifact from its own endpoint rather
than have one baked in at render time; that keeps the runner/scoring split
intact.
"""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(prefix="/niaf", tags=["niaf-bench"])

STATIC_DIR = Path(__file__).parent.parent / "static" / "niaf"
HOME_PAGE = STATIC_DIR / "index.html"


@router.get("", response_class=FileResponse)
async def niaf_home() -> FileResponse:
    """NIAF-Bench home page."""
    return FileResponse(HOME_PAGE, media_type="text/html")
