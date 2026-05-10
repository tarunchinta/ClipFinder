"""Page routes for server-side rendered pages."""

from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from app.auth.users import current_user_optional
from app.models.user import User
from app.config import get_settings

router = APIRouter()
settings = get_settings()

# Setup Jinja2 templates
templates_path = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_path))


@router.get("/", response_class=HTMLResponse, tags=["pages"])
async def landing_page(
    request: Request,
    user: User = Depends(current_user_optional),
):
    """
    Landing page with Google OAuth CTA.
    
    If user is already authenticated, shows dashboard link instead.
    """
    return templates.TemplateResponse(
        "landing.html",
        {
            "request": request,
            "user": user,
        }
    )


@router.get("/dashboard", response_class=HTMLResponse, tags=["pages"])
async def dashboard_page(
    request: Request,
    user: User = Depends(current_user_optional),
):
    """
    Dashboard page with Google Drive folder picker.
    
    Redirects to landing if not authenticated.
    """
    if not user:
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/", status_code=302)
    
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "user": user,
            "google_client_id": settings.google_client_id,
            "google_api_key": settings.google_api_key,
            "google_access_token": user.google_access_token or "",
        }
    )


@router.get("/search", response_class=HTMLResponse, tags=["pages"])
async def search_page(
    request: Request,
    user: User = Depends(current_user_optional),
):
    """
    Search page for querying indexed files.
    
    Redirects to landing if not authenticated.
    """
    if not user:
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/", status_code=302)
    
    return templates.TemplateResponse(
        "search.html",
        {
            "request": request,
            "user": user,
        }
    )



