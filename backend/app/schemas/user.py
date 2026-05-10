"""User schemas for API requests and responses."""

from datetime import datetime
from typing import Optional
import uuid

from fastapi_users import schemas
from pydantic import BaseModel


class UserRead(schemas.BaseUser[uuid.UUID]):
    """Schema for reading user data."""
    
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    search_count: int = 0
    clips_indexed: int = 0
    is_subscribed: bool = False
    created_at: Optional[datetime] = None


class UserCreate(schemas.BaseUserCreate):
    """Schema for creating a new user."""
    
    display_name: Optional[str] = None


class UserUpdate(schemas.BaseUserUpdate):
    """Schema for updating user data."""
    
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None


class UserPublic(BaseModel):
    """Public user info (safe to expose)."""
    
    id: uuid.UUID
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    search_count: int = 0
    clips_indexed: int = 0
    is_subscribed: bool = False
    
    class Config:
        from_attributes = True



