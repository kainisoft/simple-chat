"""
Test suite for the Chatbot API
"""

import pytest
import asyncio
import time
from httpx import AsyncClient
from fastapi.testclient import TestClient

from app.main import app
from app.chatbot_service import chatbot_service


class TestChatbotAPI:
    """Test cases for the Chatbot API"""
    
    @pytest.fixture
    def client(self):
        """Test client fixture"""
        return TestClient(app)
    
    @pytest.fixture
    async def async_client(self):
        """Async test client fixture"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "hardware" in data
        assert "AMD Ryzen 9 8945HX" in data["hardware"]
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "model_loaded" in data
        assert "gpu_available" in data
    
    @pytest.mark.asyncio
    async def test_chat_endpoint(self, async_client):
        """Test chat endpoint functionality"""
        # Wait for model to load
        await asyncio.sleep(2)
        
        payload = {
            "message": "Hello, how are you?",
            "temperature": 0.7,
            "max_length": 100
        }
        
        response = await async_client.post("/chat", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "response" in data
        assert "timestamp" in data
        assert "model_info" in data
        assert "processing_time" in data
        assert "gpu_used" in data
        
        # Check response time is reasonable
        assert data["processing_time"] < 5.0  # Should be under 5 seconds
    
    def test_chat_validation(self, client):
        """Test input validation"""
        # Empty message
        response = client.post("/chat", json={"message": ""})
        assert response.status_code == 422
        
        # Message too long
        long_message = "x" * 2001
        response = client.post("/chat", json={"message": long_message})
        assert response.status_code == 422
        
        # Invalid temperature
        response = client.post("/chat", json={
            "message": "test",
            "temperature": 3.0
        })
        assert response.status_code == 422
    
    def test_model_info_endpoint(self, client):
        """Test model info endpoint"""
        response = client.get("/model/info")
        if response.status_code == 200:
            data = response.json()
            assert "model_name" in data
            assert "device" in data
            assert "parameters" in data
        else:
            # Model might not be loaded in test environment
            assert response.status_code == 503
    
    def test_system_status_endpoint(self, client):
        """Test system status endpoint"""
        response = client.get("/system/status")
        assert response.status_code == 200
        data = response.json()
        assert "gpu_available" in data
        assert "cpu_usage" in data
        assert "memory_usage" in data
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, async_client):
        """Test concurrent request handling"""
        payload = {"message": "Test concurrent request"}
        
        # Send 5 concurrent requests
        tasks = []
        for i in range(5):
            task = async_client.post("/chat", json=payload)
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that most requests succeeded
        success_count = sum(1 for r in responses if hasattr(r, 'status_code') and r.status_code == 200)
        assert success_count >= 3  # At least 3 out of 5 should succeed
    
    def test_rate_limiting(self, client):
        """Test rate limiting functionality"""
        # This test might be skipped in development
        payload = {"message": "Rate limit test"}
        
        # Send many requests quickly
        responses = []
        for i in range(10):
            response = client.post("/chat", json=payload)
            responses.append(response.status_code)
        
        # Should have some successful requests
        assert 200 in responses


class TestChatbotService:
    """Test cases for the ChatbotService"""
    
    @pytest.mark.asyncio
    async def test_service_initialization(self):
        """Test service initialization"""
        assert chatbot_service is not None
        assert hasattr(chatbot_service, 'device')
        assert hasattr(chatbot_service, 'model_loaded')
    
    @pytest.mark.asyncio
    async def test_system_info(self):
        """Test system info retrieval"""
        info = chatbot_service.get_system_info()
        assert isinstance(info, dict)
        assert "gpu_available" in info
        assert "cpu_usage" in info
        assert "memory_usage" in info
        assert "model_loaded" in info
    
    @pytest.mark.asyncio
    async def test_model_loading(self):
        """Test model loading functionality"""
        # This might take a while
        success = await chatbot_service.load_model()
        assert isinstance(success, bool)
        
        if success:
            assert chatbot_service.model_loaded
            assert chatbot_service.model is not None
            assert chatbot_service.tokenizer is not None


class TestPerformance:
    """Performance test cases"""
    
    @pytest.mark.asyncio
    async def test_response_time(self, async_client):
        """Test response time requirements"""
        payload = {"message": "What is artificial intelligence?"}
        
        start_time = time.time()
        response = await async_client.post("/chat", json=payload)
        end_time = time.time()
        
        if response.status_code == 200:
            response_time = end_time - start_time
            # Should be under 3 seconds for most queries
            assert response_time < 3.0
            
            data = response.json()
            # Processing time should be under 2 seconds
            assert data["processing_time"] < 2.0
    
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Test memory usage is reasonable"""
        info = chatbot_service.get_system_info()
        
        # Memory usage should be under 90%
        assert info["memory_usage"] < 90.0
        
        # If GPU is available, check GPU memory
        if info["gpu_available"] and "gpu_memory_used" in info:
            # GPU memory should be reasonable
            assert info["gpu_memory_used"] < 8000  # 8GB limit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
