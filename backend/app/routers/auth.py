"""Authentication routes."""

import secrets
import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse

from app.auth.users import fastapi_users, current_active_user, get_user_manager, UserManager
from app.auth.backend import auth_backend, get_jwt_strategy
from app.auth.google import google_oauth_client
from app.schemas.user import UserRead, UserCreate, UserUpdate
from app.models.user import User
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()


@router.get("/google/login", tags=["auth"])
async def google_login(request: Request):
    """
    Redirect to Google OAuth login.
    
    This is a convenience endpoint that fetches the authorization URL
    and performs an automatic redirect instead of returning JSON.
    Generates and stores OAuth state for CSRF protection.
    
    Uses access_type=offline to get a refresh token for long-lived access.
    Uses prompt=consent to always get a refresh token (even on re-login).
    """
    # Generate a random state for CSRF protection
    state = secrets.token_urlsafe(32)
    
    authorization_url = await google_oauth_client.get_authorization_url(
        redirect_uri=settings.google_redirect_uri,
        state=state,
        scope=google_oauth_client.base_scopes,
        extras_params={
            "access_type": "offline",  # Required to get refresh token
            "prompt": "consent",  # Force consent to always get refresh token
        },
    )

    # Store state in a cookie for validation on callback
    response = RedirectResponse(url=authorization_url, status_code=302)
    response.set_cookie(
        key="oauth_state",
        value=state,
        max_age=600,  # 10 minutes
        httponly=True,
        samesite="lax",
        secure=False,  # Set True in production with HTTPS
    )
    return response


@router.get("/google/callback", tags=["auth"])
async def google_callback(
    request: Request,
    user_manager: UserManager = Depends(get_user_manager),
):
    """
    Custom OAuth callback that handles state validation and user creation.
    """
    # Get the authorization code from query params
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    error = request.query_params.get("error")
    
    if error:
        logger.error(f"OAuth error: {error}")
        return RedirectResponse(url="/?error=oauth_error", status_code=302)
    
    if not code:
        logger.error("No authorization code received")
        return RedirectResponse(url="/?error=no_code", status_code=302)
    
    # Validate state from cookie
    stored_state = request.cookies.get("oauth_state")
    if not stored_state or stored_state != state:
        logger.error(f"State mismatch: stored={stored_state}, received={state}")
        return RedirectResponse(url="/?error=invalid_state", status_code=302)
    
    try:
        # Exchange code for tokens
        oauth2_token = await google_oauth_client.get_access_token(
            code=code,
            redirect_uri=settings.google_redirect_uri,
        )
        
        access_token = oauth2_token["access_token"]
        refresh_token = oauth2_token.get("refresh_token")
        expires_at = oauth2_token.get("expires_at")
        
        # Debug: Log token info
        logger.info(f"OAuth token response - has refresh_token: {refresh_token is not None}, expires_at: {expires_at}")
        if not refresh_token:
            logger.warning("No refresh token received from Google! Make sure access_type=offline is set")
        
        # Get user info from Google
        account_id, account_email = await google_oauth_client.get_id_email(access_token)
        
        logger.info(f"OAuth success for: {account_email}")
        
        # Create or update user via user manager
        user = await user_manager.oauth_callback(
            oauth_name="google",
            access_token=access_token,
            account_id=account_id,
            account_email=account_email,
            expires_at=expires_at,
            refresh_token=refresh_token,
            request=request,
            associate_by_email=True,
            is_verified_by_default=True,
        )
        
        logger.info(f"User authenticated: {user.email}")
        
        # Generate JWT token
        jwt_strategy = get_jwt_strategy()
        token = await jwt_strategy.write_token(user)
        
        # Redirect to dashboard with auth cookie
        response = RedirectResponse(url="/dashboard", status_code=302)
        response.set_cookie(
            key="clipfinder_auth",
            value=token,
            max_age=settings.jwt_lifetime_seconds,
            httponly=True,
            samesite="lax",
            secure=False,  # Set True in production with HTTPS
        )
        # Clear the oauth_state cookie
        response.delete_cookie("oauth_state")
        
        return response
        
    except Exception as e:
        logger.exception(f"OAuth callback error: {e}")
        return RedirectResponse(url="/?error=oauth_failed", status_code=302)

# Include fastapi-users auth routes
# JWT login/logout (for API access)
router.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/jwt",
    tags=["auth"],
)

# NOTE: Google OAuth routes are handled by custom /google/login and /google/callback above
# This avoids the JSON response issue and gives us full control over state management

# User registration (optional, mainly OAuth)
router.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    tags=["auth"],
)

# User management
router.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/users",
    tags=["users"],
)


@router.get("/me", response_model=UserRead, tags=["auth"])
async def get_current_user(user: User = Depends(current_active_user)):
    """Get current authenticated user."""
    return user


@router.post("/logout", tags=["auth"])
async def logout(request: Request):
    """Logout and clear auth cookie."""
    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie("clipfinder_auth")
    return response



