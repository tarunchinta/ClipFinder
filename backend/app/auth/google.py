"""Google OAuth2 client configuration."""

from httpx_oauth.clients.google import GoogleOAuth2

from app.config import get_settings

settings = get_settings()

# Google OAuth2 client with Drive read-only scope
google_oauth_client = GoogleOAuth2(
    client_id=settings.google_client_id,
    client_secret=settings.google_client_secret,
    scopes=[
        "openid",
        "email", 
        "profile",
        # Read-only access to Google Drive files
        "https://www.googleapis.com/auth/drive.readonly",
    ],
)



