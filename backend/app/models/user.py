"""User model for authentication."""

from datetime import datetime
from typing import List
import uuid

from fastapi_users.db import SQLAlchemyBaseUserTableUUID, SQLAlchemyBaseOAuthAccountTableUUID
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, Mapped, mapped_column

from app.database import Base


class OAuthAccount(SQLAlchemyBaseOAuthAccountTableUUID, Base):
    """OAuth account linked to a user (e.g., Google)."""
    pass


class User(SQLAlchemyBaseUserTableUUID, Base):
    """
    User model with OAuth support.
    
    Inherits from fastapi-users base which provides:
    - id (UUID)
    - email (str, unique)
    - hashed_password (str, nullable for OAuth-only users)
    - is_active (bool)
    - is_superuser (bool)
    - is_verified (bool)
    """
    
    # OAuth accounts linked to this user
    oauth_accounts: Mapped[List[OAuthAccount]] = relationship(
        "OAuthAccount", lazy="joined"
    )
    
    # Additional fields for Distill
    display_name: Mapped[str] = mapped_column(String(255), nullable=True)
    avatar_url: Mapped[str] = mapped_column(String(500), nullable=True)
    
    # Usage tracking
    search_count: Mapped[int] = mapped_column(Integer, default=0)
    clips_indexed: Mapped[int] = mapped_column(Integer, default=0)
    
    # Subscription status
    is_subscribed: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    last_login_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=True
    )
    
    # Google Drive token (encrypted in production)
    google_access_token: Mapped[str] = mapped_column(String(2000), nullable=True)
    google_refresh_token: Mapped[str] = mapped_column(String(2000), nullable=True)
    google_token_expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)



