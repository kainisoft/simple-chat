"""
Configuration module for the Chatbot API
Optimized for AMD Ryzen 9 8945HX + NVIDIA RTX 5070
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with hardware optimization"""
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")
    api_reload: bool = Field(default=False, env="API_RELOAD")
    
    # Model Configuration
    chatbot_model_name: str = Field(default="microsoft/DialoGPT-small", env="CHATBOT_MODEL_NAME")
    chatbot_model_cache_dir: str = Field(default="./models", env="CHATBOT_MODEL_CACHE_DIR")
    use_gpu: bool = Field(default=True, env="USE_GPU")
    gpu_memory_fraction: float = Field(default=0.8, env="GPU_MEMORY_FRACTION")
    max_length: int = Field(default=512, env="MAX_LENGTH")
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    top_p: float = Field(default=0.9, env="TOP_P")
    top_k: int = Field(default=50, env="TOP_K")
    
    # Performance Settings (Optimized for 32-core Ryzen 9 8945HX)
    batch_size: int = Field(default=4, env="BATCH_SIZE")
    max_concurrent_requests: int = Field(default=16, env="MAX_CONCURRENT_REQUESTS")
    torch_threads: int = Field(default=16, env="TORCH_THREADS")
    omp_num_threads: int = Field(default=16, env="OMP_NUM_THREADS")
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    
    # Security
    api_key_enabled: bool = Field(default=False, env="API_KEY_ENABLED")
    api_key: Optional[str] = Field(default=None, env="API_KEY")
    
    # Health Check
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")

    # Database Configuration
    database_url: str = Field(
        default="mysql+aiomysql://chatbot_user:chatbot_password@localhost:3306/chatbot_db",
        env="DATABASE_URL"
    )
    database_host: str = Field(default="localhost", env="DATABASE_HOST")
    database_port: int = Field(default=3306, env="DATABASE_PORT")
    database_name: str = Field(default="chatbot_db", env="DATABASE_NAME")
    database_user: str = Field(default="chatbot_user", env="DATABASE_USER")
    database_password: str = Field(default="chatbot_password", env="DATABASE_PASSWORD")
    
    # CUDA Settings for RTX 5070
    cuda_visible_devices: str = Field(default="0", env="CUDA_VISIBLE_DEVICES")
    pytorch_cuda_alloc_conf: str = Field(
        default="max_split_size_mb:512", 
        env="PYTORCH_CUDA_ALLOC_CONF"
    )
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "protected_namespaces": ('settings_',)
    }


# Global settings instance
settings = Settings()

# Set environment variables for optimal performance
os.environ["TORCH_NUM_THREADS"] = str(settings.torch_threads)
os.environ["OMP_NUM_THREADS"] = str(settings.omp_num_threads)
os.environ["CUDA_VISIBLE_DEVICES"] = settings.cuda_visible_devices
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = settings.pytorch_cuda_alloc_conf
