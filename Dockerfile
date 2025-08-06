# 多阶段构建 - GPU版本
FROM nvidia/cuda:12.9.0-devel-ubuntu20.04 as builder

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Shanghai

# 安装系统依赖，预先配置时区
RUN apt-get update && \
    ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1-dev \
    python3.8 \
    python3.8-dev \
    python3.8-venv \
    python3-pip \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# 创建Python虚拟环境
RUN python3.8 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 升级pip
RUN pip install --upgrade pip

# 安装PyTorch with CUDA support
RUN pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir uvicorn jieba requests

# 复制源代码
COPY . .

# 下载模型到缓存（可选，首次启动会下载）
RUN echo "# Models will be downloaded on first run" > /app/.model_cache_info

# 运行阶段
FROM nvidia/cuda:12.9.0-runtime-ubuntu20.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Shanghai

# 安装运行时依赖，预先配置时区
RUN apt-get update && \
    ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get install -y \
    ffmpeg \
    libsndfile1 \
    curl \
    git \
    python3.8 \
    python3.8-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 从构建阶段复制虚拟环境
COPY --from=builder /opt/venv /opt/venv

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

# 创建工作目录
WORKDIR /app

# 复制代码和虚拟环境
COPY --from=builder /app /app

# 创建模型缓存目录
RUN mkdir -p /root/.cache/modelscope/hub

# 创建tmp目录
RUN mkdir -p /app/tmp

# 暴露端口
EXPOSE 5001

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5001/ || exit 1

# 启动命令
CMD ["python", "src/api_optimized.py"]