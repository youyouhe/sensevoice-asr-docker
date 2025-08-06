# SenseVoice ASR Docker 部署指南
作者: youyouhe

## 📋 系统要求

### 硬件要求
- **GPU**: NVIDIA GPU (计算能力 3.5+)
- **显存**: 建议 8GB+ (模型加载需要)
- **内存**: 建议 16GB+ RAM
- **存储**: 建议 20GB+ 可用空间

### 软件要求
- **操作系统**: Ubuntu 20.04+ / CentOS 7+ / Docker Desktop
- **NVIDIA驱动**: 450.80.02+ (建议 520+)
- **Docker**: 20.10+ (推荐 24.0+)
- **Docker Compose**: 1.29.0+

## 🚀 快速开始

### 1. 环境准备

```bash
# 检查NVIDIA驱动
nvidia-smi

# 检查Docker安装
docker --version
docker-compose --version  # 或 docker compose version
```

### 2. 克隆项目
```bash
git clone <your-repo-url>
cd SenseVoice
```

### 3. 一键部署
```bash
# 赋予执行权限
chmod +x docker-manager.sh

# 一键部署（构建+启动+测试）
./docker-manager.sh deploy
```

### 4. 手动部署（可选）
```bash
# 检查环境
./docker-manager.sh check

# 构建镜像
./docker-manager.sh build

# 启动服务
./docker-manager.sh start

# 运行测试
./docker-manager.sh test
```

## 📖 使用说明

### API测试
```bash
# 使用curl测试
curl -X POST http://localhost:5001/asr \
  -F "file=@7.wav" \
  -F "lang=zh" | jq -r '.data'

# 或使用容器内测试脚本
docker-compose exec asr-service python /app/test_asr.py /app/7.wav
```

### API文档
- **Swagger UI**: http://localhost:5001/docs
- **ReDoc**: http://localhost:5001/redoc

### 上传音频文件
```bash
# 方法1：挂载目录运行
docker-compose run --rm -v $(pwd)/audio:/app/audio asr-service python /app/test_asr.py /app/audio/your_file.wav

# 方法2：使用curl上传
curl -X POST http://localhost:5001/asr \
  -F "file=@/path/to/your/audio.wav" \
  -F "lang=zh" | jq -r '.data'
```

## ⚙️ 配置选项

### 环境变量
创建 `.env` 文件来自定义配置：
```bash
# 服务配置
PORT=5001
HOST=0.0.0.0

# GPU配置
CUDA_VISIBLE_DEVICES=0
NVIDIA_VISIBLE_DEVICES=all

# 模型配置
MODEL_CACHE_SIZE=10GB
DISABLE_MODEL_UPDATE=true
```

### 资源限制
编辑 `docker-compose.yml` 中的 deploy 部分：
```yaml
deploy:
  resources:
    limits:
      memory: 16G
    reservations:
      memory: 8G
```

## 🔧 管理命令

### 服务管理
```bash
# 启动服务
./docker-manager.sh start

# 停止服务
./docker-manager.sh stop

# 重启服务
./docker-manager.sh restart

# 查看状态
./docker-manager.sh status

# 查看日志
./docker-manager.sh logs
```

### 镜像管理
```bash
# 重新构建镜像
./docker-manager.sh build

# 清理资源
./docker-manager.sh cleanup

# 查看Docker资源使用
docker stats
```

### 测试命令
```bash
# 运行API测试
./docker-manager.sh test

# 进入容器调试
docker-compose exec asr-service bash

# 检查GPU使用情况
nvidia-smi
```

## 📊 性能优化

### GPU优化
1. **指定GPU**：
   ```bash
   # 使用特定GPU
   export CUDA_VISIBLE_DEVICES=0
   ./docker-manager.sh start
   ```

2. **多GPU配置**：
   ```yaml
   # docker-compose.yml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             device_ids: ['0', '1']
             capabilities: [gpu]
   ```

### 模型缓存
模型会自动缓存到 `/root/.cache/modelscope/hub`，通过Docker volume持久化：
```bash
# 查看模型缓存
docker volume inspect sensevoice-model-cache

# 清理模型缓存（谨慎操作）
docker volume rm sensevoice-model-cache
```

### 性能监控
```bash
# 实时监控
docker stats sensevoice-asr

# GPU监控
watch -n 1 nvidia-smi

# 服务健康检查
curl -f http://localhost:5001/
```

## 🐛 故障排除

### 常见问题

1. **GPU无法识别**
   ```bash
   # 检查NVIDIA运行时
   docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi
   
   # 检查容器GPU支持
   docker info | grep -i runtime
   ```

2. **模型下载失败**
   ```bash
   # 检查网络连接
   docker-compose exec asr-service curl -I https://modelscope.cn
   
   # 清理缓存重试
   ./docker-manager.sh cleanup
   ./docker-manager.sh deploy
   ```

3. **内存不足**
   ```bash
   # 增加内存限制
   # 编辑 docker-compose.yml，增加 memory 限制
   ```

4. **端口冲突**
   ```bash
   # 检查端口占用
   netstat -tlnp | grep :5001
   
   # 修改端口（编辑 docker-compose.yml）
   ```

### 日志调试
```bash
# 查看所有日志
docker-compose logs

# 查看特定服务日志
docker-compose logs -f asr-service

# 实时跟踪日志
docker-compose logs --tail=100 -f asr-service
```

## 🔒 安全考虑

### 生产环境建议
1. **网络安全**：
   - 使用Nginx反向代理
   - 配置HTTPS证书
   - 限制访问IP

2. **资源安全**：
   - 设置资源限制
   - 使用非root用户
   - 定期更新镜像

3. **数据安全**：
   - 加密敏感数据
   - 定期备份模型缓存
   - 监控访问日志

### 权限配置
```yaml
# docker-compose.yml 安全配置
user: "1000:1000"  # 非root用户
read_only: true      # 只读文件系统
tmpfs:
  - /tmp             # 临时文件系统
```

## 📈 扩展部署

### 多实例部署
```yaml
# docker-compose.scale.yml
services:
  asr-service-1:
    <<: *asr-service
    ports:
      - "5001:5001"
  
  asr-service-2:
    <<: *asr-service
    ports:
      - "5002:5001"
```

### Kubernetes部署
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sensevoice-asr
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sensevoice-asr
  template:
    metadata:
      labels:
        app: sensevoice-asr
    spec:
      containers:
      - name: asr
        image: sensevoice-asr:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
        ports:
        - containerPort: 5001
```

## 🤝 技术支持

如遇问题，请检查：
1. 系统要求是否满足
2. NVIDIA驱动是否正确安装
3. Docker版本是否兼容
4. 网络连接是否正常

更多文档和更新，请参考项目GitHub页面。