# 多阶段构建 - GPU版本
FROM nvidia/cuda:12.9.0-devel-ubuntu20.04 as builder

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH=/opt/conda/bin:$PATH

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1-dev \
    python3.8 \
    python3.8-dev \
    python3.8-venv \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# 安装Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -ya

# 创建虚拟环境并安装PyTorch
RUN /opt/conda/bin/conda create -n asr_env python=3.8 -y && \
    /opt/conda/bin/conda activate asr_env && \
    conda install pytorch==2.3.1 torchaudio==2.3.1 pytorch-cuda=12.9 -c pytorch -c nvidia -y

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖（在conda环境中）
RUN /opt/conda/bin/conda activate asr_env && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir uvicorn jieba requests

# 复制源代码
COPY . .

# 下载模型到缓存（可选，首次启动会下载）
RUN echo "# Models will be downloaded on first run" > /app/.model_cache_info

# 运行阶段
FROM nvidia/cuda:12.9.0-runtime-ubuntu20.04

# 安装运行时依赖
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 从构建阶段复制conda环境
COPY --from=builder /opt/conda /opt/conda

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH=/opt/conda/bin:$PATH
ENV CONDA_DEFAULT_ENV=asr_env
SHELL ["/bin/bash", "-c", "source activate asr_env && exec $0 $@"]

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
CMD ["python", "api_optimized.py"]