"""User manager and FastAPI Users setup."""

import logging
from datetime import datetime
from typing import Optional
import uuid

from fastapi import Depends, Request
from fastapi_users import BaseUserManager, FastAPIUsers, UUIDIDMixin
from fastapi_users.db import SQLAlchemyUserDatabase
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_async_session
from app.models.user import User, OAuthAccount
from app.auth.backend import auth_backend

logger = logging.getLogger(__name__)
settings = get_settings()


async def get_user_db(session: AsyncSession = Depends(get_async_session)):
    """Get user database adapter."""
    yield SQLAlchemyUserDatabase(session, User, OAuthAccount)


class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    """Custom user manager with OAuth handling."""
    
    reset_password_token_secret = settings.jwt_secret
    verification_token_secret = settings.jwt_secret

    async def on_after_register(self, user: User, request: Optional[Request] = None):
        """Called after a new user registers."""
        print(f"User {user.id} ({user.email}) has registered.")

    async def on_after_login(
        self,
        user: User,
        request: Optional[Request] = None,
        response=None,
    ):
        """Called after successful login."""
        # Update last login timestamp
        user.last_login_at = datetime.utcnow()
        print(f"User {user.id} ({user.email}) logged in.")

    async def on_after_forgot_password(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        """Called after password reset requested."""
        print(f"User {user.id} forgot password. Reset token: {token}")

    async def on_after_request_verify(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        """Called after verification requested."""
        print(f"Verification requested for user {user.id}. Token: {token}")

    async def oauth_callback(
        self,
        oauth_name: str,
        access_token: str,
        account_id: str,
        account_email: str,
        expires_at: Optional[int] = None,
        refresh_token: Optional[str] = None,
        request: Optional[Request] = None,
        *,
        associate_by_email: bool = False,
        is_verified_by_default: bool = True,
    ) -> User:
        """
        Handle OAuth callback - create or update user.
        
        Override to store Google tokens for Drive API access.
        """
        user = await super().oauth_callback(
            oauth_name,
            access_token,
            account_id,
            account_email,
            expires_at,
            refresh_token,
            request,
            associate_by_email=associate_by_email,
            is_verified_by_default=is_verified_by_default,
        )
        
        # Store Google tokens for Drive API access
        if oauth_name == "google":
            update_dict = {
                "google_access_token": access_token,
            }
            if refresh_token:
                update_dict["google_refresh_token"] = refresh_token
                logger.info(f"Saving refresh token for user {user.email}")
            else:
                logger.warning(f"No refresh token to save for user {user.email}")
            if expires_at:
                update_dict["google_token_expires_at"] = datetime.fromtimestamp(expires_at)
            
            # Update user in database with all token fields
            await self.user_db.update(user, update_dict)
            logger.info(f"Updated Google tokens for user {user.email}")
            
            # Update local user object
            user.google_access_token = access_token
            if refresh_token:
                user.google_refresh_token = refresh_token
            if expires_at:
                user.google_token_expires_at = datetime.fromtimestamp(expires_at)
        
        return user


async def get_user_manager(user_db=Depends(get_user_db)):
    """Dependency for getting user manager."""
    yield UserManager(user_db)


# Create FastAPIUsers instance
fastapi_users = FastAPIUsers[User, uuid.UUID](get_user_manager, [auth_backend])

# User dependencies
current_active_user = fastapi_users.current_user(active=True)
current_user_optional = fastapi_users.current_user(active=True, optional=True)



