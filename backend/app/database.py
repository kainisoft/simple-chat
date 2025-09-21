"""
Database configuration and models for the chatbot application
"""

import uuid
from datetime import datetime
from typing import Optional, List
from sqlalchemy import create_engine, Column, String, Text, DateTime, Boolean, ForeignKey, Enum, DECIMAL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.mysql import CHAR
import enum

from .config import settings

# Create async engine
async_engine = create_async_engine(
    settings.database_url,
    echo=settings.log_level.upper() == "DEBUG",
    pool_pre_ping=True,
    pool_recycle=3600,
    pool_size=10,
    max_overflow=20
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Base class for all models
Base = declarative_base()

class MessageRole(enum.Enum):
    """Enum for message roles"""
    USER = "user"
    ASSISTANT = "assistant"

class Conversation(Base):
    """Conversation model"""
    __tablename__ = "conversations"
    
    id = Column(CHAR(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(255), nullable=False, default="New Conversation")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    user_id = Column(String(255), default="default_user", nullable=False)
    
    # Relationships
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    conversation_metadata = relationship("ConversationMetadata", back_populates="conversation", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, title={self.title})>"

class Message(Base):
    """Message model"""
    __tablename__ = "messages"
    
    id = Column(CHAR(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(CHAR(36), ForeignKey("conversations.id"), nullable=False)
    content = Column(Text, nullable=False)
    role = Column(Enum(MessageRole, values_callable=lambda obj: [e.value for e in obj]), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    processing_time = Column(DECIMAL(8, 4), nullable=True)
    model_info = Column(String(255), nullable=True)
    gpu_used = Column(Boolean, default=False, nullable=False)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    
    def __repr__(self):
        return f"<Message(id={self.id}, role={self.role.value}, conversation_id={self.conversation_id})>"

class ConversationMetadata(Base):
    """Conversation metadata model for additional properties"""
    __tablename__ = "conversation_metadata"
    
    id = Column(CHAR(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(CHAR(36), ForeignKey("conversations.id"), nullable=False)
    metadata_key = Column(String(100), nullable=False)
    metadata_value = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="conversation_metadata")
    
    def __repr__(self):
        return f"<ConversationMetadata(conversation_id={self.conversation_id}, key={self.metadata_key})>"

# Database dependency
async def get_db() -> AsyncSession:
    """Get database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# Database initialization
async def init_database():
    """Initialize database tables"""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# Database health check
async def check_database_health() -> bool:
    """Check if database is accessible"""
    try:
        async with AsyncSessionLocal() as session:
            await session.execute("SELECT 1")
            return True
    except Exception:
        return False
