# API Reference

## Base URL
```
http://localhost:8000
```

## Authentication
Optional API key authentication via `X-API-Key` header.

## Endpoints

### POST /chat
Generate a chatbot response.

**Request Body:**
```json
{
  "message": "string (required, 1-2000 chars)",
  "conversation_id": "string (optional)",
  "temperature": "float (optional, 0.1-2.0)",
  "max_length": "integer (optional, 10-1000)"
}
```

**Response:**
```json
{
  "response": "string",
  "timestamp": "ISO 8601 datetime",
  "model_info": "string",
  "conversation_id": "string or null",
  "processing_time": "float (seconds)",
  "gpu_used": "boolean"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is artificial intelligence?",
    "temperature": 0.7,
    "max_length": 200
  }'
```

### POST /generate
Alias for `/chat` endpoint with identical functionality.

### GET /health
System health check.

**Response:**
```json
{
  "status": "healthy|degraded|unhealthy",
  "timestamp": "ISO 8601 datetime",
  "model_loaded": "boolean",
  "gpu_available": "boolean",
  "gpu_memory_used": "float (MB) or null",
  "cpu_usage": "float (percentage)",
  "memory_usage": "float (percentage)"
}
```

### GET /model/info
Get model information.

**Response:**
```json
{
  "model_name": "string",
  "model_size": "string",
  "device": "string",
  "memory_usage": "float (MB)",
  "parameters": "integer",
  "quantized": "boolean"
}
```

### GET /system/status
Comprehensive system status.

**Response:**
```json
{
  "model_loaded": "boolean",
  "device": "string",
  "gpu_available": "boolean",
  "cpu_usage": "float",
  "memory_usage": "float",
  "gpu_memory_used": "float",
  "gpu_memory_total": "float",
  "gpu_utilization": "float",
  "gpu_temperature": "float",
  "model_info": {
    "name": "string",
    "device": "string",
    "quantized": "boolean",
    "parameters": "integer"
  }
}
```

## Error Responses

All endpoints return error responses in this format:

```json
{
  "error": "string",
  "detail": "string or null",
  "timestamp": "ISO 8601 datetime"
}
```

### HTTP Status Codes
- `200` - Success
- `400` - Bad Request (validation error)
- `401` - Unauthorized (invalid API key)
- `429` - Too Many Requests (rate limit exceeded)
- `500` - Internal Server Error
- `503` - Service Unavailable (model not loaded)

## Rate Limiting
- Default: 100 requests per minute per IP
- Configurable via `RATE_LIMIT_REQUESTS` and `RATE_LIMIT_WINDOW`
- Rate limit headers included in responses:
  - `X-RateLimit-Limit`
  - `X-RateLimit-Remaining`
  - `X-RateLimit-Reset`

## Request/Response Examples

### Simple Chat
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'
```

### Chat with Parameters
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me about machine learning",
    "conversation_id": "conv-123",
    "temperature": 0.8,
    "max_length": 300
  }'
```

### With API Key
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"message": "Hello!"}'
```
