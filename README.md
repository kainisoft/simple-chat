# AI Chatbot Project

A high-performance full-stack AI chatbot application optimized for AMD Ryzen 9 8945HX + NVIDIA RTX 5070. Features a FastAPI backend powered by Microsoft DialoGPT with GPU acceleration, React frontend, and comprehensive Docker deployment.

<img width="1149" height="830" alt="Screenshot_20250921_140432" src="https://github.com/user-attachments/assets/80ad158e-5ae7-47cc-a21b-8bc6b2d6dc92" />

## Technology Stack

- **Backend**: Python 3.10, FastAPI, PyTorch, Transformers (Hugging Face), SQLAlchemy
- **Frontend**: React 18, Node.js 18, pnpm package manager
- **Database**: MariaDB 10.11 with Alembic migrations
- **AI/ML**: Microsoft DialoGPT, CUDA 12.9, GPU optimization for RTX 5070
- **Infrastructure**: Docker, Docker Compose, Nginx
- **Monitoring**: Prometheus, Grafana (production)
- **Development**: Hot reload, comprehensive testing, structured logging

## Project Structure

```
./
├── .gitignore                  # Comprehensive gitignore for all technologies
├── README.md                   # This file
├── docker-compose.yml          # Production environment (GPU-optimized)
├── docker-compose.dev.yml      # Development environment
├── docker-compose.cpu.yml      # CPU-only environment
├── backend/                    # FastAPI backend
│   ├── app/                    # Application source code
│   │   ├── __init__.py
│   │   ├── main.py            # FastAPI application with lifespan management
│   │   ├── chatbot_service.py # AI chatbot logic with GPU optimization
│   │   ├── config.py          # Hardware-optimized configuration
│   │   ├── models.py          # Pydantic models for API
│   │   ├── database.py        # Database models and connection
│   │   └── database_service.py # Database operations
│   ├── database/              # Database initialization
│   │   └── init/              # SQL initialization scripts
│   │       └── 01-create-schema.sql
│   ├── docs/                  # API documentation
│   │   ├── API_REFERENCE.md
│   │   └── DEPLOYMENT.md
│   ├── tests/                 # Backend tests
│   ├── models/                # Cached AI models (gitignored)
│   ├── logs/                  # Application logs (gitignored)
│   ├── Dockerfile             # Production Docker image (CUDA 12.9)
│   ├── Dockerfile.cpu         # CPU-only Docker image
│   ├── requirements.txt       # Python dependencies
│   ├── .env.example          # Environment template
│   ├── .gitignore            # Backend-specific gitignore
│   ├── benchmark.py          # Performance benchmarking
│   ├── test_simple_api.py    # API integration tests
│   ├── validate_setup.py     # Setup validation script
│   └── README.md             # Backend documentation
└── frontend/                  # React frontend
    ├── public/                # Static assets
    │   └── index.html
    ├── src/                   # React source code
    │   ├── App.js            # Main React component
    │   ├── App.css           # Component styling
    │   ├── index.js          # React entry point
    │   └── index.css         # Global styles
    ├── Dockerfile            # Production Docker image
    ├── Dockerfile.dev        # Development Docker image
    ├── nginx.conf            # Nginx configuration for production
    ├── package.json          # Node.js dependencies (pnpm)
    ├── .env.example         # Frontend environment template
    ├── .env.development     # Development environment
    ├── .env.production      # Production environment
    ├── .dockerignore        # Docker ignore patterns
    └── README.md            # Frontend documentation
```

## Quick Start

### Prerequisites

- **Docker CE 20.10+** with NVIDIA Container Toolkit (for GPU support)
- **Node.js 18+** and **pnpm** (for frontend development)
- **NVIDIA RTX 5070** with CUDA 12.9+ support (optional, falls back to CPU)
- **32GB RAM** recommended for optimal performance
- **AMD Ryzen 9 8945HX** or equivalent multi-core CPU

### Full Stack Development (Recommended)

1. **Start all services with Docker Compose:**
```bash
# Development mode (with hot reload)
docker compose -f docker-compose.dev.yml up

# Production mode (GPU-optimized)
docker compose up

# CPU-only mode (for systems without GPU)
docker compose -f docker-compose.cpu.yml up
```

2. **Access the application:**
- **Frontend**: http://localhost:3000
- **API Base URL**: http://localhost:8000
- **Interactive API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Backend Development (API Server)

1. **Environment setup:**
```bash
cd backend
cp .env.example .env
# Edit .env with your configuration
```

2. **Start backend only:**
```bash
docker compose -f docker-compose.dev.yml up chatbot-api
```

### Frontend Development (React App)

1. **Install dependencies:**
```bash
cd frontend
pnpm install
```

2. **Environment setup:**
```bash
cp .env.example .env.development
# Configure API URL and other settings
```

3. **Start development server:**
```bash
pnpm start
```

### Database Setup

The MariaDB database is automatically initialized with Docker Compose:
- **Database**: `chatbot_db`
- **User**: `chatbot_user`
- **Port**: 3306 (mapped to host)
- **Migrations**: Handled by Alembic automatically

## Docker Compose Configurations

### Development (`docker-compose.dev.yml`)
- **Hot reload**: Enabled for both backend and frontend
- **Debug access**: Backend debug port exposed (5678)
- **Volume mounts**: Source code mounted for live development
- **Database**: MariaDB with persistent data
- **Optimized for**: Development workflow and debugging

### Production (`docker-compose.yml`)
- **GPU acceleration**: NVIDIA RTX 5070 with CUDA 12.9 support
- **Performance**: Optimized resource allocation (8GB memory, 16 CPU cores)
- **Monitoring**: Prometheus metrics and health checks
- **Security**: Non-root containers, security options
- **Persistence**: Model cache and logs mounted
- **Restart policies**: Automatic recovery from failures

### CPU-only (`docker-compose.cpu.yml`)
- **Compatibility**: For systems without NVIDIA GPU
- **PyTorch**: CPU-optimized PyTorch installation
- **Resources**: Reduced memory and CPU requirements
- **Performance**: Optimized for AMD Ryzen 9 8945HX multi-core processing

## Hardware Optimization

### NVIDIA RTX 5070 + CUDA 12.9
- **GPU Memory**: 12GB GDDR6X with optimized allocation
- **CUDA Compute**: sm_89 architecture support
- **Memory Management**: Configured for 80% GPU memory usage
- **PyTorch**: CUDA-enabled with optimized memory allocation

### AMD Ryzen 9 8945HX
- **CPU Cores**: 16 threads optimized for AI workloads
- **Memory**: 32GB RAM with efficient model loading
- **Threading**: OMP and PyTorch threads configured for maximum performance
- **Cache**: Optimized model caching and loading strategies

## API Endpoints

### Core Endpoints
- `GET /` - API information and status
- `GET /health` - Comprehensive health check with system metrics
- `POST /chat` - Send message to chatbot (with conversation support)
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation

### Conversation Management
- `POST /conversations` - Create new conversation
- `GET /conversations` - List conversations with pagination
- `GET /conversations/{id}` - Get specific conversation
- `PUT /conversations/{id}` - Update conversation metadata
- `DELETE /conversations/{id}` - Delete conversation
- `GET /conversations/{id}/messages` - Get conversation messages

### Model Information
- `GET /model/info` - Current model information and capabilities
- `GET /metrics` - Prometheus metrics for monitoring

## Environment Configuration

### Backend Environment Variables

Key variables in `backend/.env.example`:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Model Configuration
CHATBOT_MODEL_NAME=microsoft/DialoGPT-medium
CHATBOT_MODEL_CACHE_DIR=./models
USE_GPU=true
GPU_MEMORY_FRACTION=0.8
MAX_LENGTH=512
TEMPERATURE=0.7

# Performance (AMD Ryzen 9 8945HX + RTX 5070)
BATCH_SIZE=4
MAX_CONCURRENT_REQUESTS=16
TORCH_THREADS=16
OMP_NUM_THREADS=16

# Database
DATABASE_URL=mysql+aiomysql://user:pass@host:3306/db
DATABASE_HOST=localhost
DATABASE_PORT=3306

# Security & Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
API_KEY_ENABLED=false

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Frontend Environment Variables

Key variables in `frontend/.env.example`:

```bash
# API Configuration
REACT_APP_API_URL=http://localhost:8000

# Development
NODE_ENV=development
PORT=3000
BROWSER=none

# Hot Reload (Docker)
CHOKIDAR_USEPOLLING=true
WATCHPACK_POLLING=true
```

## Development Workflow

### Backend Development
1. **Code Changes:**
   - Modify files in `backend/app/`
   - Hot reload automatically applies changes
   - View logs: `docker compose -f docker-compose.dev.yml logs -f chatbot-api`

2. **Database Changes:**
   - Modify models in `backend/app/database.py`
   - Generate migration: `alembic revision --autogenerate -m "description"`
   - Apply migration: `alembic upgrade head`

3. **Testing:**
   ```bash
   cd backend
   python -m pytest tests/ -v
   python test_simple_api.py  # Integration tests
   python benchmark.py        # Performance benchmarks
   ```

### Frontend Development
1. **Code Changes:**
   - Modify files in `frontend/src/`
   - React development server provides instant hot reload
   - API requests automatically proxy to backend

2. **Testing:**
   ```bash
   cd frontend
   pnpm test              # Run test suite
   pnpm test --coverage   # With coverage report
   pnpm build             # Production build test
   ```

### Full Stack Testing
```bash
# Start all services
docker compose -f docker-compose.dev.yml up -d

# Run backend validation
cd backend && python validate_setup.py

# Test API endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, chatbot!"}'
```

## Performance Metrics

### Hardware-Optimized Performance
- **Model Load Time**: ~3-5 seconds (DialoGPT-medium)
- **API Response Time**: ~200-500ms (GPU) / ~800ms-1.2s (CPU)
- **Memory Usage**: ~6-8GB (with model loaded)
- **GPU Memory**: ~4-6GB (RTX 5070)
- **Concurrent Requests**: Up to 16 simultaneous connections

### Benchmarking
```bash
cd backend
python benchmark.py  # Comprehensive performance testing
```

## Project Management

### Git Workflow
- **Comprehensive .gitignore**: Covers Python, Node.js, AI/ML models, Docker, IDEs
- **Ignored Files**: Models, logs, environment files, build artifacts
- **Tracked Files**: Source code, configuration templates, documentation

### File Organization
- **Models**: Automatically cached in `backend/models/` (gitignored)
- **Logs**: Structured logging in `backend/logs/` (gitignored)
- **Data**: Database persistence via Docker volumes
- **Environment**: Template files provided, actual files gitignored

## Troubleshooting

### Common Issues

1. **Docker/GPU Issues:**
   ```bash
   # Check Docker installation
   docker --version
   docker compose --version

   # Verify NVIDIA runtime
   docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu22.04 nvidia-smi

   # Check GPU availability
   nvidia-smi
   ```

2. **Container won't start:**
   - Verify Docker CE 20.10+ and NVIDIA Container Toolkit
   - Check port availability: `netstat -tulpn | grep :8000`
   - Review logs: `docker compose logs chatbot-api`

3. **Database connection issues:**
   ```bash
   # Check database health
   docker compose exec chatbot-db mysql -u chatbot_user -p chatbot_db

   # Reset database
   docker compose down -v
   docker compose up chatbot-db
   ```

4. **Frontend can't reach API:**
   - Ensure backend is running: `curl http://localhost:8000/health`
   - Check environment variables in `frontend/.env.development`
   - Verify CORS configuration in backend

5. **Permission denied:**
   ```bash
   # Add user to docker group
   sudo usermod -aG docker $USER
   newgrp docker

   # Fix file permissions
   sudo chown -R $USER:$USER backend/models backend/logs
   ```

6. **Model loading issues:**
   - Check internet connection for model download
   - Verify disk space: `df -h`
   - Clear model cache: `rm -rf backend/models/*`

7. **Performance issues:**
   - Monitor resources: `docker stats`
   - Check GPU utilization: `nvidia-smi -l 1`
   - Review configuration in `backend/.env`

### Validation Scripts

```bash
# Backend setup validation
cd backend && python validate_setup.py

# API integration test
cd backend && python test_simple_api.py

# Performance benchmark
cd backend && python benchmark.py
```

## Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** following the project structure
4. **Test thoroughly**:
   - Run backend tests: `cd backend && python -m pytest`
   - Run frontend tests: `cd frontend && pnpm test`
   - Test Docker builds: `docker compose build`
5. **Commit changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Submit a pull request**

### Development Guidelines
- Follow existing code style and structure
- Add tests for new features
- Update documentation as needed
- Ensure Docker builds work on both GPU and CPU configurations
- Test with both development and production Docker Compose files

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Hugging Face** for the Transformers library and DialoGPT model
- **FastAPI** for the high-performance web framework
- **React** team for the frontend framework
- **NVIDIA** for CUDA and GPU acceleration support
