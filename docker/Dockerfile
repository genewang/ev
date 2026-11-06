# EV Autonomous Trucking Perception System Dockerfile
# Optimized for production deployment with ASIL-D compliance

FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Install additional ML dependencies
RUN pip install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    torchaudio==2.0.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Install TensorRT-LLM
RUN pip install --no-cache-dir tensorrt-llm==0.5.0

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY tests/ ./tests/

# Create necessary directories
RUN mkdir -p /app/models /app/logs /app/data /app/cache

# Set permissions
RUN chmod +x /app/src/main.py

# Create non-root user for security
RUN useradd -m -u 1000 evuser && \
    chown -R evuser:evuser /app
USER evuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# Expose ports
EXPOSE 8080 9090

# Set default command
CMD ["python", "src/main.py", "--config", "configs/default.yaml"]