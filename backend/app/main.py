"""
High-Performance Chatbot API
Optimized for AMD Ryzen 9 8945HX + NVIDIA RTX 5070
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from .config import settings
from .models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ErrorResponse,
    ModelInfo,
    ConversationCreate,
    ConversationResponse,
    ConversationUpdate,
    ConversationListResponse,
    MessageResponse,
    MessageListResponse
)
from .chatbot_service import chatbot_service
from .database_service import database_service
from .database import init_database, check_database_health, MessageRole

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown"""
    # Startup
    logger.info("Starting Chatbot API...")
    logger.info(f"Hardware: AMD Ryzen 9 8945HX + NVIDIA RTX 5070")
    logger.info(f"Model: {settings.chatbot_model_name}")

    # Initialize database
    try:
        await init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise RuntimeError("Database initialization failed")

    # Load model
    success = await chatbot_service.load_model()
    if not success:
        logger.error("Failed to load model during startup")
        raise RuntimeError("Model loading failed")

    # Warmup model
    await chatbot_service.warmup()

    logger.info("Chatbot API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Chatbot API...")
    chatbot_service.cleanup()
    logger.info("Chatbot API shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="High-Performance Chatbot API",
    description="Optimized chatbot API for AMD Ryzen 9 8945HX + NVIDIA RTX 5070",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Dependency for API key validation (if enabled)
async def validate_api_key(request: Request):
    """Validate API key if enabled"""
    if settings.api_key_enabled:
        api_key = request.headers.get("X-API-Key")
        if not api_key or api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
    return True

# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if settings.log_level.upper() == "DEBUG" else None
        ).dict()
    )

# API Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "High-Performance Chatbot API",
        "hardware": "AMD Ryzen 9 8945HX + NVIDIA RTX 5070",
        "model": settings.chatbot_model_name,
        "status": "running"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        system_info = chatbot_service.get_system_info()
        db_healthy = await check_database_health()

        overall_status = "healthy"
        if not system_info["model_loaded"]:
            overall_status = "degraded"
        if not db_healthy:
            overall_status = "degraded" if overall_status == "healthy" else "unhealthy"

        return HealthResponse(
            status=overall_status,
            model_loaded=system_info["model_loaded"],
            gpu_available=system_info["gpu_available"],
            gpu_memory_used=system_info.get("gpu_memory_used"),
            cpu_usage=system_info["cpu_usage"],
            memory_usage=system_info["memory_usage"]
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            gpu_available=False,
            cpu_usage=0.0,
            memory_usage=0.0
        )

@app.post("/chat", response_model=ChatResponse)
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window}minute")
async def chat(
    request: Request,
    chat_request: ChatRequest,
    _: bool = Depends(validate_api_key)
):
    """Main chat endpoint with GPU-accelerated inference and conversation storage"""
    try:
        if not chatbot_service.model_loaded:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please try again later."
            )

        # Create conversation if not provided
        conversation_id = chat_request.conversation_id
        if not conversation_id:
            conversation = await database_service.create_conversation()
            conversation_id = conversation.id

        # Store user message in database
        await database_service.add_message(
            conversation_id=conversation_id,
            content=chat_request.message,
            role=MessageRole.USER
        )

        # Generate response
        result = await chatbot_service.generate_response(
            message=chat_request.message,
            conversation_id=conversation_id,
            temperature=chat_request.temperature,
            max_length=chat_request.max_length
        )

        # Store assistant response in database
        await database_service.add_message(
            conversation_id=conversation_id,
            content=result["response"],
            role=MessageRole.ASSISTANT,
            processing_time=result["processing_time"],
            model_info=result["model_info"],
            gpu_used=result["gpu_used"]
        )

        return ChatResponse(
            response=result["response"],
            model_info=result["model_info"],
            conversation_id=conversation_id,
            processing_time=result["processing_time"],
            gpu_used=result["gpu_used"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate", response_model=ChatResponse)
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window}minute")
async def generate(
    request: Request,
    chat_request: ChatRequest,
    _: bool = Depends(validate_api_key)
):
    """Alternative generation endpoint (alias for /chat)"""
    return await chat(request, chat_request, _)

@app.get("/model/info", response_model=ModelInfo)
async def model_info(_: bool = Depends(validate_api_key)):
    """Get detailed model information"""
    try:
        if not chatbot_service.model_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")

        system_info = chatbot_service.get_system_info()
        model_info = system_info["model_info"]

        return ModelInfo(
            model_name=model_info["name"],
            model_size="Small" if "small" in model_info["name"].lower() else "Medium",
            device=model_info["device"],
            memory_usage=system_info.get("gpu_memory_used", 0.0),
            parameters=model_info["parameters"],
            quantized=model_info["quantized"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system/status")
async def system_status(_: bool = Depends(validate_api_key)):
    """Get comprehensive system status"""
    try:
        return chatbot_service.get_system_info()
    except Exception as e:
        logger.error(f"System status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Conversation Management Endpoints
@app.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    conversation: ConversationCreate,
    _: bool = Depends(validate_api_key)
):
    """Create a new conversation"""
    try:
        new_conversation = await database_service.create_conversation(
            title=conversation.title,
            user_id=conversation.user_id
        )

        return ConversationResponse(
            id=new_conversation.id,
            title=new_conversation.title,
            created_at=new_conversation.created_at,
            updated_at=new_conversation.updated_at,
            message_count=0,
            last_message_at=None,
            last_user_message=None,
            last_assistant_message=None
        )
    except Exception as e:
        logger.error(f"Create conversation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations", response_model=ConversationListResponse)
async def get_conversations(
    user_id: str = "default_user",
    limit: int = 50,
    offset: int = 0,
    _: bool = Depends(validate_api_key)
):
    """Get list of conversations"""
    try:
        conversations = await database_service.get_conversations(
            user_id=user_id,
            limit=limit,
            offset=offset
        )

        conversation_responses = [
            ConversationResponse(**conv) for conv in conversations
        ]

        return ConversationListResponse(
            conversations=conversation_responses,
            total=len(conversation_responses),
            offset=offset,
            limit=limit
        )
    except Exception as e:
        logger.error(f"Get conversations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations/{conversation_id}/messages", response_model=MessageListResponse)
async def get_conversation_messages(
    conversation_id: str,
    limit: int = 100,
    offset: int = 0,
    _: bool = Depends(validate_api_key)
):
    """Get messages for a conversation"""
    try:
        messages = await database_service.get_conversation_messages(
            conversation_id=conversation_id,
            limit=limit,
            offset=offset
        )

        message_responses = [
            MessageResponse(
                id=msg.id,
                conversation_id=msg.conversation_id,
                content=msg.content,
                role=msg.role.value,
                created_at=msg.created_at,
                processing_time=float(msg.processing_time) if msg.processing_time else None,
                model_info=msg.model_info,
                gpu_used=msg.gpu_used
            ) for msg in messages
        ]

        return MessageListResponse(
            messages=message_responses,
            conversation_id=conversation_id,
            total=len(message_responses),
            offset=offset,
            limit=limit
        )
    except Exception as e:
        logger.error(f"Get conversation messages error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    _: bool = Depends(validate_api_key)
):
    """Delete a conversation"""
    try:
        success = await database_service.delete_conversation(conversation_id)
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"message": "Conversation deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete conversation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/conversations/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: str,
    update_data: ConversationUpdate,
    _: bool = Depends(validate_api_key)
):
    """Update conversation title"""
    try:
        if update_data.title:
            success = await database_service.update_conversation_title(
                conversation_id, update_data.title
            )
            if not success:
                raise HTTPException(status_code=404, detail="Conversation not found")

        # Get updated conversation
        conversation = await database_service.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Get conversation metadata
        conversations = await database_service.get_conversations(limit=1, offset=0)
        conv_data = next((c for c in conversations if c['id'] == conversation_id), None)

        if not conv_data:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return ConversationResponse(**conv_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update conversation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Development server
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=1,  # Single worker for GPU sharing
        reload=settings.api_reload,
        log_level=settings.log_level.lower()
    )
