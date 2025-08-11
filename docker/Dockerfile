# EV Autonomous Trucking Perception System Dockerfile
# Optimized for production deployment with AWS ECS

FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python
RUN ln -s /usr/bin/python3.9 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional CUDA dependencies
RUN pip install --no-cache-dir \
    cupy-cuda11x \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/

# Create necessary directories
RUN mkdir -p /app/models /app/logs /app/data

# Set environment variables for the application
ENV PYTHONPATH=/app
ENV CONFIG_PATH=/app/configs/default.yaml
ENV MODEL_PATH=/app/models
ENV LOG_PATH=/app/logs
ENV DATA_PATH=/app/data

# Expose ports for monitoring and API
EXPOSE 8000 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Default command
CMD ["python", "src/main.py", "--config", "/app/configs/default.yaml"]
