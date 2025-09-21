# Deployment Guide

## Production Deployment

### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA Container Toolkit
- 8GB+ GPU memory
- 16GB+ system RAM

### Environment Setup

1. **Copy environment file:**
```bash
cp .env.example .env
```

2. **Configure production settings:**
```env
# Production API settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
API_RELOAD=false

# Security
API_KEY_ENABLED=true
API_KEY=your-secure-api-key-here

# Performance
USE_GPU=true
GPU_MEMORY_FRACTION=0.8
BATCH_SIZE=4
MAX_CONCURRENT_REQUESTS=16

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Docker Deployment

#### GPU-Accelerated (Recommended)
```bash
# Build and deploy
docker-compose up -d --build

# Check status
docker-compose ps
docker-compose logs -f chatbot-api
```

#### CPU-Only Fallback
```bash
# For systems without GPU
docker-compose -f docker-compose.cpu.yml up -d --build
```

### Reverse Proxy Setup

#### Nginx Configuration
```nginx
upstream chatbot_api {
    server localhost:8000;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://chatbot_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeout settings for AI inference
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

#### Traefik Configuration
```yaml
version: '3.8'
services:
  chatbot-api:
    # ... existing configuration
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.chatbot.rule=Host(`your-domain.com`)"
      - "traefik.http.routers.chatbot.entrypoints=websecure"
      - "traefik.http.routers.chatbot.tls.certresolver=letsencrypt"
      - "traefik.http.services.chatbot.loadbalancer.server.port=8000"
```

### SSL/TLS Setup

#### Let's Encrypt with Certbot
```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### Monitoring Setup

#### Enable Monitoring Stack
```bash
# Deploy with Prometheus and Grafana
docker-compose --profile monitoring up -d

# Access Grafana at https://your-domain.com:3000
# Default: admin/admin
```

#### Custom Metrics
The API exposes Prometheus metrics at `/metrics`:
- Request count and duration
- GPU utilization
- Memory usage
- Model inference time

### Scaling Considerations

#### Horizontal Scaling
```yaml
# docker-compose.scale.yml
version: '3.8'
services:
  chatbot-api:
    # ... existing config
    deploy:
      replicas: 3
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

#### Load Balancer Configuration
```nginx
upstream chatbot_cluster {
    least_conn;
    server chatbot-api-1:8000;
    server chatbot-api-2:8000;
    server chatbot-api-3:8000;
}
```

### Security Hardening

#### Container Security
```yaml
# Security-hardened compose
services:
  chatbot-api:
    # ... existing config
    security_opt:
      - no-new-privileges:true
      - seccomp:unconfined
    read_only: true
    tmpfs:
      - /tmp
      - /var/tmp
    cap_drop:
      - ALL
    cap_add:
      - CHOWN
      - SETGID
      - SETUID
```

#### Firewall Configuration
```bash
# UFW rules
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw deny 8000/tcp   # Block direct API access
sudo ufw enable
```

### Backup and Recovery

#### Model and Data Backup
```bash
# Backup models and logs
docker run --rm -v chatbot-api_models:/data -v $(pwd):/backup \
  alpine tar czf /backup/models-backup.tar.gz /data

# Restore
docker run --rm -v chatbot-api_models:/data -v $(pwd):/backup \
  alpine tar xzf /backup/models-backup.tar.gz -C /
```

#### Database Backup (if using)
```bash
# Example for PostgreSQL
docker-compose exec postgres pg_dump -U user dbname > backup.sql
```

### Health Checks and Monitoring

#### Automated Health Checks
```bash
#!/bin/bash
# health-check.sh
HEALTH_URL="http://localhost:8000/health"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $HEALTH_URL)

if [ $RESPONSE -eq 200 ]; then
    echo "API is healthy"
    exit 0
else
    echo "API is unhealthy (HTTP $RESPONSE)"
    exit 1
fi
```

#### Log Aggregation
```yaml
# ELK Stack integration
logging:
  driver: "json-file"
  options:
    max-size: "100m"
    max-file: "3"
    labels: "service=chatbot-api"
```

### Performance Optimization

#### GPU Memory Management
```env
# Optimize for multiple models
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True

# Memory fraction per container
GPU_MEMORY_FRACTION=0.4  # For 3 containers on 8GB GPU
```

#### CPU Optimization
```env
# Maximize CPU utilization
TORCH_THREADS=32
OMP_NUM_THREADS=32
MKL_NUM_THREADS=32
```

### Troubleshooting

#### Common Issues
1. **GPU not accessible**: Check NVIDIA Container Toolkit
2. **Out of memory**: Reduce `GPU_MEMORY_FRACTION`
3. **Slow responses**: Verify CPU governor is set to 'performance'
4. **Container crashes**: Check resource limits and logs

#### Debug Commands
```bash
# Check GPU in container
docker-compose exec chatbot-api nvidia-smi

# Monitor resource usage
docker stats

# View detailed logs
docker-compose logs --tail=100 -f chatbot-api
```
