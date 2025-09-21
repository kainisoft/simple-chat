"""
Pydantic models for request/response validation
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator


class ChatRequest(BaseModel):
    """Chat request model with validation"""
    message: str = Field(
        ..., 
        min_length=1, 
        max_length=2000,
        description="User input message for the chatbot"
    )
    conversation_id: Optional[str] = Field(
        None,
        description="Optional conversation ID for context tracking"
    )
    temperature: Optional[float] = Field(
        None,
        ge=0.1,
        le=2.0,
        description="Temperature for response generation (0.1-2.0)"
    )
    max_length: Optional[int] = Field(
        None,
        ge=10,
        le=1000,
        description="Maximum length of generated response"
    )
    
    @validator('message')
    def validate_message(cls, v):
        if not v.strip():
            raise ValueError('Message cannot be empty or only whitespace')
        return v.strip()


class ChatResponse(BaseModel):
    """Chat response model"""
    response: str = Field(..., description="Chatbot generated response")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_info: str = Field(..., description="Model name used for generation")
    conversation_id: Optional[str] = Field(None, description="Conversation ID if provided")
    processing_time: float = Field(..., description="Response generation time in seconds")
    gpu_used: bool = Field(..., description="Whether GPU was used for inference")

    class Config:
        protected_namespaces = ()
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    gpu_memory_used: Optional[float] = Field(None, description="GPU memory usage in MB")
    cpu_usage: float = Field(..., description="CPU usage percentage")
    memory_usage: float = Field(..., description="RAM usage percentage")

    class Config:
        protected_namespaces = ()
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ModelInfo(BaseModel):
    """Model information response"""
    model_name: str = Field(..., description="Name of the loaded model")
    model_size: str = Field(..., description="Model size description")
    device: str = Field(..., description="Device the model is running on")
    memory_usage: float = Field(..., description="Model memory usage in MB")
    parameters: int = Field(..., description="Number of model parameters")
    quantized: bool = Field(..., description="Whether the model is quantized")

    class Config:
        protected_namespaces = ()
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# New models for conversation management
class ConversationCreate(BaseModel):
    """Model for creating a new conversation"""
    title: Optional[str] = Field(default="New Conversation", max_length=255)
    user_id: Optional[str] = Field(default="default_user", max_length=255)


class ConversationResponse(BaseModel):
    """Model for conversation response"""
    id: str = Field(..., description="Conversation ID")
    title: str = Field(..., description="Conversation title")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    message_count: int = Field(..., description="Number of messages in conversation")
    last_message_at: Optional[datetime] = Field(None, description="Timestamp of last message")
    last_user_message: Optional[str] = Field(None, description="Preview of last user message")
    last_assistant_message: Optional[str] = Field(None, description="Preview of last assistant message")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ConversationUpdate(BaseModel):
    """Model for updating conversation"""
    title: Optional[str] = Field(None, max_length=255, description="New conversation title")


class MessageResponse(BaseModel):
    """Model for message response"""
    id: str = Field(..., description="Message ID")
    conversation_id: str = Field(..., description="Conversation ID")
    content: str = Field(..., description="Message content")
    role: str = Field(..., description="Message role (user/assistant)")
    created_at: datetime = Field(..., description="Creation timestamp")
    processing_time: Optional[float] = Field(None, description="Processing time for assistant messages")
    model_info: Optional[str] = Field(None, description="Model information")
    gpu_used: Optional[bool] = Field(None, description="Whether GPU was used")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ConversationListResponse(BaseModel):
    """Model for conversation list response"""
    conversations: List[ConversationResponse] = Field(..., description="List of conversations")
    total: int = Field(..., description="Total number of conversations")
    offset: int = Field(..., description="Current offset")
    limit: int = Field(..., description="Current limit")


class MessageListResponse(BaseModel):
    """Model for message list response"""
    messages: List[MessageResponse] = Field(..., description="List of messages")
    conversation_id: str = Field(..., description="Conversation ID")
    total: int = Field(..., description="Total number of messages")
    offset: int = Field(..., description="Current offset")
    limit: int = Field(..., description="Current limit")
