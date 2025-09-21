"""
Database service layer for conversation and message management
"""

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy import select, desc, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .database import Conversation, Message, ConversationMetadata, MessageRole, AsyncSessionLocal
import logging

logger = logging.getLogger(__name__)

class DatabaseService:
    """Service class for database operations"""
    
    async def create_conversation(
        self, 
        title: str = "New Conversation", 
        user_id: str = "default_user"
    ) -> Conversation:
        """Create a new conversation"""
        async with AsyncSessionLocal() as session:
            conversation = Conversation(
                id=str(uuid.uuid4()),
                title=title,
                user_id=user_id
            )
            session.add(conversation)
            await session.commit()
            await session.refresh(conversation)
            logger.info(f"Created new conversation: {conversation.id}")
            return conversation
    
    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID"""
        async with AsyncSessionLocal() as session:
            stmt = select(Conversation).where(
                and_(Conversation.id == conversation_id, Conversation.is_active == True)
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()
    
    async def get_conversations(
        self, 
        user_id: str = "default_user", 
        limit: int = 50, 
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get list of conversations with metadata"""
        async with AsyncSessionLocal() as session:
            # Get conversations with message count and last message info
            stmt = select(
                Conversation.id,
                Conversation.title,
                Conversation.created_at,
                Conversation.updated_at,
                func.count(Message.id).label('message_count'),
                func.max(Message.created_at).label('last_message_at')
            ).select_from(
                Conversation
            ).outerjoin(
                Message, Conversation.id == Message.conversation_id
            ).where(
                and_(
                    Conversation.user_id == user_id,
                    Conversation.is_active == True
                )
            ).group_by(
                Conversation.id,
                Conversation.title,
                Conversation.created_at,
                Conversation.updated_at
            ).order_by(
                desc(func.coalesce(func.max(Message.created_at), Conversation.updated_at))
            ).limit(limit).offset(offset)
            
            result = await session.execute(stmt)
            conversations = []
            
            for row in result:
                # Get last user and assistant messages
                last_messages_stmt = select(Message).where(
                    Message.conversation_id == row.id
                ).order_by(desc(Message.created_at)).limit(2)
                
                last_messages_result = await session.execute(last_messages_stmt)
                last_messages = last_messages_result.scalars().all()
                
                last_user_message = None
                last_assistant_message = None
                
                for msg in last_messages:
                    if msg.role == MessageRole.USER and not last_user_message:
                        last_user_message = msg.content
                    elif msg.role == MessageRole.ASSISTANT and not last_assistant_message:
                        last_assistant_message = msg.content
                
                conversations.append({
                    'id': row.id,
                    'title': row.title,
                    'created_at': row.created_at,
                    'updated_at': row.updated_at,
                    'message_count': row.message_count or 0,
                    'last_message_at': row.last_message_at,
                    'last_user_message': last_user_message,
                    'last_assistant_message': last_assistant_message
                })
            
            return conversations
    
    async def get_conversation_messages(
        self, 
        conversation_id: str, 
        limit: int = 100, 
        offset: int = 0
    ) -> List[Message]:
        """Get messages for a conversation"""
        async with AsyncSessionLocal() as session:
            stmt = select(Message).where(
                Message.conversation_id == conversation_id
            ).order_by(Message.created_at).limit(limit).offset(offset)
            
            result = await session.execute(stmt)
            return result.scalars().all()
    
    async def add_message(
        self,
        conversation_id: str,
        content: str,
        role: MessageRole,
        processing_time: Optional[float] = None,
        model_info: Optional[str] = None,
        gpu_used: bool = False
    ) -> Message:
        """Add a message to a conversation"""
        async with AsyncSessionLocal() as session:
            message = Message(
                id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                content=content,
                role=role,
                processing_time=processing_time,
                model_info=model_info,
                gpu_used=gpu_used
            )
            session.add(message)
            
            # Update conversation's updated_at timestamp
            conversation_stmt = select(Conversation).where(Conversation.id == conversation_id)
            conversation_result = await session.execute(conversation_stmt)
            conversation = conversation_result.scalar_one_or_none()
            
            if conversation:
                conversation.updated_at = datetime.utcnow()
                
                # Auto-generate title from first user message if it's still "New Conversation"
                if conversation.title == "New Conversation" and role == MessageRole.USER:
                    # Use first 50 characters of the message as title
                    new_title = content[:50].strip()
                    if len(content) > 50:
                        new_title += "..."
                    conversation.title = new_title
            
            await session.commit()
            await session.refresh(message)
            logger.info(f"Added message to conversation {conversation_id}: {role.value}")
            return message
    
    async def delete_conversation(self, conversation_id: str) -> bool:
        """Soft delete a conversation"""
        async with AsyncSessionLocal() as session:
            stmt = select(Conversation).where(Conversation.id == conversation_id)
            result = await session.execute(stmt)
            conversation = result.scalar_one_or_none()
            
            if conversation:
                conversation.is_active = False
                await session.commit()
                logger.info(f"Deleted conversation: {conversation_id}")
                return True
            return False
    
    async def update_conversation_title(self, conversation_id: str, title: str) -> bool:
        """Update conversation title"""
        async with AsyncSessionLocal() as session:
            stmt = select(Conversation).where(Conversation.id == conversation_id)
            result = await session.execute(stmt)
            conversation = result.scalar_one_or_none()
            
            if conversation:
                conversation.title = title
                conversation.updated_at = datetime.utcnow()
                await session.commit()
                logger.info(f"Updated conversation title: {conversation_id}")
                return True
            return False
    
    async def get_conversation_context(
        self, 
        conversation_id: str, 
        max_messages: int = 20
    ) -> List[Dict[str, str]]:
        """Get conversation context for AI model (recent messages)"""
        async with AsyncSessionLocal() as session:
            stmt = select(Message).where(
                Message.conversation_id == conversation_id
            ).order_by(desc(Message.created_at)).limit(max_messages)
            
            result = await session.execute(stmt)
            messages = result.scalars().all()
            
            # Reverse to get chronological order and format for AI model
            context = []
            for message in reversed(messages):
                context.append({
                    'role': message.role.value,
                    'content': message.content
                })
            
            return context

# Global database service instance
database_service = DatabaseService()
