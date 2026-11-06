# EV Autonomous Trucking Perception System - Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the EV Autonomous Trucking Perception System in production environments. The system is designed for containerized deployment with Docker and Kubernetes, supporting both cloud and edge deployments.

## Prerequisites

### System Requirements
- **Operating System**: Ubuntu 20.04+ or macOS 12+
- **Python**: 3.9+
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **CUDA**: 11.8+ (for GPU acceleration)
- **Memory**: 8GB+ RAM
- **Storage**: 50GB+ available space

### Hardware Requirements
- **CPU**: 4+ cores (8+ recommended)
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Neural Engine**: Apple Silicon with Neural Engine (for optimization)
- **Memory**: 8GB+ RAM (16GB+ recommended)

## Quick Start

### 1. Clone Repository
```bash
git clone <repository-url>
cd ev
```

### 2. Install Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install additional ML dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorrt-llm
```

### 3. Run System
```bash
# Run with default configuration
python src/main.py

# Run with custom configuration
python src/main.py --config configs/production.yaml

# Run in simulation mode
python src/main.py --simulation
```

## Docker Deployment

### 1. Build Docker Image
```bash
# Build the Docker image
docker build -t ev-perception:latest -f docker/Dockerfile .

# Build with specific tag
docker build -t ev-perception:v1.0.0 -f docker/Dockerfile .
```

### 2. Run with Docker Compose
```bash
# Start all services
docker-compose -f docker/docker-compose.yml up -d

# Start specific services
docker-compose -f docker/docker-compose.yml up -d ev-perception redis

# View logs
docker-compose -f docker/docker-compose.yml logs -f ev-perception
```

### 3. Production Deployment
```bash
# Production deployment with resource limits
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.prod.yml up -d

# Scale services
docker-compose -f docker/docker-compose.yml up -d --scale ev-perception=3
```

## Kubernetes Deployment

### 1. Create Namespace
```bash
kubectl create namespace ev-perception
```

### 2. Apply Configurations
```bash
# Apply base configuration
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml

# Apply application deployment
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

### 3. Verify Deployment
```bash
# Check pod status
kubectl get pods -n ev-perception

# Check service status
kubectl get services -n ev-perception

# Check ingress status
kubectl get ingress -n ev-perception
```

## Configuration

### 1. Environment Variables
```bash
# Set environment variables
export PYTHONPATH=/app
export CUDA_VISIBLE_DEVICES=0
export LOG_LEVEL=INFO
export CONFIG_PATH=/app/configs/production.yaml
```

### 2. Configuration Files
```yaml
# configs/production.yaml
system_name: "EV Autonomous Trucking Perception System"
version: "1.0.0"
device: "cuda"

sensors:
  lidar:
    num_sensors: 4
    enabled: true
    lines: 128
    range_meters: 100.0
    frequency_hz: 10.0

  camera:
    num_sensors: 8
    enabled: true
    resolution: "8MP"
    fps: 30

  radar:
    num_sensors: 6
    enabled: true
    frequency_ghz: 77.0
    range_meters: 200.0

perception:
  num_classes: 20
  num_features: 64
  batch_size: 1

safety:
  max_latency_ms: 32.0
  min_confidence: 0.8
  asil_level: "ASIL-D"

performance_targets:
  end_to_end_latency_ms: 32.0
  throughput_boost: 4.0
  training_efficiency: 3.0
```

### 3. Security Configuration
```yaml
# configs/security.yaml
security:
  zero_trust:
    enabled: true
    continuous_verification: true
    behavioral_analysis: true
    threat_detection: true

  access_control:
    enabled: true
    role_based: true
    multi_factor_auth: true

  encryption:
    enabled: true
    algorithm: "AES-256"
    key_rotation: true
```

## Monitoring and Observability

### 1. Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ev-perception'
    static_configs:
      - targets: ['ev-perception:9090']
    scrape_interval: 5s
    metrics_path: '/metrics'
```

### 2. Grafana Dashboards
```bash
# Import dashboards
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @grafana/dashboards/ev-perception.json
```

### 3. Health Checks
```bash
# Check system health
curl http://localhost:8080/health

# Check metrics
curl http://localhost:9090/metrics

# Check logs
docker logs ev-perception-system
```

## Performance Optimization

### 1. GPU Acceleration
```bash
# Enable CUDA
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"

# Run with GPU
docker run --gpus all -it ev-perception:latest
```

### 2. Neural Engine Optimization
```bash
# Enable Neural Engine (macOS)
export MPS_ENABLE=1
export MPS_FORCE_CPU=0

# Run with Neural Engine
python src/main.py --device neural_engine
```

### 3. Memory Optimization
```bash
# Set memory limits
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1

# Run with memory optimization
python src/main.py --memory-optimization
```

## Security Hardening

### 1. Container Security
```bash
# Run with security options
docker run --security-opt no-new-privileges:true \
  --cap-drop ALL \
  --read-only \
  -v /app/data:/app/data:rw \
  ev-perception:latest
```

### 2. Network Security
```bash
# Create secure network
docker network create --driver bridge ev-network

# Run with network isolation
docker run --network ev-network ev-perception:latest
```

### 3. Access Control
```bash
# Set up authentication
export AUTH_ENABLED=true
export AUTH_PROVIDER=oauth2
export AUTH_CLIENT_ID=your-client-id
export AUTH_CLIENT_SECRET=your-client-secret
```

## Troubleshooting

### 1. Common Issues

#### CUDA Issues
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

#### Memory Issues
```bash
# Check memory usage
docker stats ev-perception-system

# Increase memory limits
docker run --memory=8g ev-perception:latest
```

#### Performance Issues
```bash
# Check CPU usage
docker exec ev-perception-system top

# Check GPU usage
nvidia-smi
```

### 2. Log Analysis
```bash
# View application logs
docker logs ev-perception-system

# View system logs
journalctl -u docker

# View performance logs
docker exec ev-perception-system cat /app/logs/performance.log
```

### 3. Debug Mode
```bash
# Run in debug mode
python src/main.py --debug --verbose

# Enable debug logging
export LOG_LEVEL=DEBUG
python src/main.py
```

## Production Checklist

### Pre-deployment
- [ ] System requirements met
- [ ] Dependencies installed
- [ ] Configuration validated
- [ ] Security settings applied
- [ ] Monitoring configured
- [ ] Backup strategy implemented

### Deployment
- [ ] Docker image built
- [ ] Container deployed
- [ ] Services started
- [ ] Health checks passing
- [ ] Metrics collected
- [ ] Logs monitored

### Post-deployment
- [ ] Performance validated
- [ ] Security verified
- [ ] Monitoring active
- [ ] Alerts configured
- [ ] Documentation updated
- [ ] Team trained

## Maintenance

### 1. Regular Updates
```bash
# Update dependencies
pip install -r requirements.txt --upgrade

# Update Docker image
docker pull ev-perception:latest
docker-compose pull
docker-compose up -d
```

### 2. Backup and Recovery
```bash
# Backup data
docker exec ev-perception-system tar -czf /backup/data.tar.gz /app/data

# Restore data
docker exec ev-perception-system tar -xzf /backup/data.tar.gz -C /app/data
```

### 3. Scaling
```bash
# Scale horizontally
docker-compose up -d --scale ev-perception=3

# Scale vertically
docker run --cpus=4 --memory=8g ev-perception:latest
```

## Support

### 1. Documentation
- [Architecture Guide](architecture.md)
- [API Reference](api.md)
- [Safety Framework](safety.md)
- [Troubleshooting Guide](troubleshooting.md)

### 2. Community
- GitHub Issues: [Repository Issues](https://github.com/your-repo/issues)
- Discussion Forum: [Community Forum](https://github.com/your-repo/discussions)
- Documentation: [Project Wiki](https://github.com/your-repo/wiki)

### 3. Professional Support
- Email: support@your-company.com
- Phone: +1-555-0123
- Hours: 24/7 for critical issues
