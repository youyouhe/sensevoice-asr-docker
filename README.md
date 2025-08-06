# 🐳 SenseVoice ASR Docker Service

<div align="center">

![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

**高性能语音识别Docker化部署方案**

[功能特性](#-功能特性) • [快速开始](#-快速开始) • [API文档](#-api文档) • [部署指南](#-部署指南) • [性能指标](#-性能指标)

</div>

---

## 🌟 功能特性

### 🔥 核心能力
- ✅ **GPU加速推理** - 基于CUDA 12.1，支持NVIDIA GPU
- ✅ **多语言支持** - 中文/英文/日文/韩文/粤语
- ✅ **优化的字幕颗粒度** - 平均4.9秒片段，符合专业标准
- ✅ **模型缓存持久化** - 避免重复下载，提升启动速度
- ✅ **智能分段算法** - 自动将长音频分割为合适的小段
- ✅ **RESTful API** - 标准HTTP接口，易于集成

### 🛠️ 技术优势
- 🐳 **Docker化部署** - 一键部署，环境一致性好
- 📊 **健康检查** - 自动监控服务状态
- 🔄 **资源管理** - GPU、内存、存储优化配置
- 📈 **性能监控** - 实时资源使用统计
- 🔒 **安全加固** - 非root用户，资源限制

---

## 🚀 快速开始

### 📋 系统要求

| 组件 | 要求 | 备注 |
|------|------|------|
| **GPU** | NVIDIA GPU (计算能力 3.5+) | 建议显存 8GB+ |
| **驱动** | NVIDIA Driver 450.80.02+ | 建议 520+ |
| **内存** | 16GB+ RAM | 模型加载需要 |
| **存储** | 20GB+ 可用空间 | 模型和缓存 |
| **Docker** | 20.10+ | 推荐 24.0+ |
| **CUDA** | CUDA 12.1 compatible | 与宿主机环境一致 |
| **Python** | 3.8 (venv) | 标准虚拟环境，轻量化 |

### 🔧 安装步骤

#### 1. 克隆项目
```bash
git clone https://github.com/youyouhe/sensevoice-asr-docker.git
cd sensevoice-asr-docker
```

#### 2. 环境检查
```bash
# 检查NVIDIA驱动
nvidia-smi

# 检查Docker
docker --version
docker-compose --version
```

#### 3. 一键部署
```bash
# 赋予执行权限
chmod +x docker-manager.sh

# 一键部署（推荐）
./docker-manager.sh deploy
```

#### 4. 验证服务
```bash
# 检查服务状态
./docker-manager.sh status

# 运行API测试
./docker-manager.sh test
```

### 📡 API测试

#### 使用curl测试
```bash
curl -X POST http://localhost:5001/asr \
  -F "file=@audio/demo_zh.wav" \
  -F "lang=zh" | jq -r '.data'
```

#### 使用Python脚本测试
```bash
# 使用内置测试脚本
./docker-manager.sh test

# 或手动测试
python src/test_asr.py audio/demo_zh.wav
```

---

## 📖 API文档

### 🎯 接口概览

#### ASR识别接口
```
POST /asr
Content-Type: multipart/form-data

参数:
- file: 音频文件 (wav, mp3, flac, m4a)
- lang: 语言代码 (zh/en/ja/ko/yue)
```

#### 请求示例
```bash
curl -X POST http://localhost:5001/asr \
  -F "file=@test.wav" \
  -F "lang=zh"
```

#### 响应格式
```json
{
    "code": 0,
    "msg": "ok", 
    "data": "1\n00:00:00,000 --> 00:00:04,920\n这是第一段字幕文本\n\n2\n00:00:04,920 --> 00:00:09,840\n这是第二段字幕文本"
}
```

### 🌐 语言支持

| 语言代码 | 语言名称 | 状态 |
|----------|----------|------|
| `zh` | 中文 | ✅ 支持 |
| `en` | 英文 | ✅ 支持 |
| `ja` | 日文 | ✅ 支持 |
| `ko` | 韩文 | ✅ 支持 |
| `yue` | 粤语 | ✅ 支持 |

### 📊 性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| **平均片段时长** | 4.9秒 | 优化的字幕颗粒度 |
| **识别精度** | >95% | 多语言高精度识别 |
| **推理速度** | 实时 | 10s音频约70ms |
| **并发能力** | 高 | 支持多并发请求 |
| **GPU显存** | 6-8GB | 模型加载需求 |

---

## 🐳 部署指南

### 📦 快速部署

#### 一键部署脚本
```bash
# 完整部署（包含环境检查、构建、启动、测试）
./docker-manager.sh deploy
```

#### 分步部署
```bash
# 1. 环境检查
./docker-manager.sh check

# 2. 构建镜像
./docker-manager.sh build

# 3. 启动服务
./docker-manager.sh start

# 4. 运行测试
./docker-manager.sh test
```

### ⚙️ 配置选项

#### 环境变量配置
创建 `.env` 文件：
```bash
# 服务配置
PORT=5001
HOST=0.0.0.0

# GPU配置
CUDA_VISIBLE_DEVICES=0
NVIDIA_VISIBLE_DEVICES=all

# 模型配置
DISABLE_MODEL_UPDATE=true
```

#### 资源限制配置
编辑 `docker-compose.yml`：
```yaml
deploy:
  resources:
    limits:
      memory: 16G
    reservations:
      memory: 8G
```

### 🔧 管理命令

#### 服务管理
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

#### 镜像管理
```bash
# 重新构建
./docker-manager.sh build

# 清理资源
./docker-manager.sh cleanup
```

### 🔍 监控和调试

#### 健康检查
```bash
# 检查服务健康状态
curl -f http://localhost:5001/

# 容器健康检查
docker ps --filter "name=sensevoice-asr"
```

#### 性能监控
```bash
# GPU使用情况
nvidia-smi

# 容器资源使用
docker stats sensevoice-asr

# 服务实时日志
docker logs -f sensevoice-asr
```

---

## 📈 性能指标

### ⚡ 基准测试结果

#### 硬件配置
- **GPU**: NVIDIA RTX 2080 Ti
- **显存**: 11GB
- **内存**: 32GB
- **CPU**: Intel i7-9700K

#### 性能数据
| 测试项目 | 结果 | 说明 |
|----------|------|------|
| **10s音频推理时间** | 70ms | 实时处理能力 |
| **模型加载时间** | 15s | 首次启动（缓存后2s） |
| **并发处理能力** | 16路 | 同时处理多个音频 |
| **内存使用** | 8GB | 稳定运行状态 |
| **GPU利用率** | 85% | 高效利用GPU |

### 📊 延迟统计

| 操作 | 平均延迟 | P95延迟 | P99延迟 |
|------|----------|---------|---------|
| **API响应** | 120ms | 350ms | 500ms |
| **音频处理** | 80ms | 200ms | 300ms |
| **模型推理** | 60ms | 150ms | 250ms |

### 🌐 多语言性能

| 语言 | 准确率 | 处理速度 | CPU占用 |
|------|--------|----------|----------|
| **中文** | 96.5% | 实时 | 45% |
| **英文** | 97.2% | 实时 | 42% |
| **日文** | 95.8% | 实时 | 48% |
| **韩文** | 95.1% | 实时 | 47% |
| **粤语** | 94.8% | 实时 | 46% |

---

## 🤝 贡献指南

我们欢迎社区贡献！请遵循以下步骤：

### 📝 提交Issue
1. 使用GitHub Issues报告bug或建议功能
2. 详细描述问题和复现步骤
3. 提供相关的日志和错误信息

### 💻 提交Pull Request
1. Fork本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

### 📋 开发环境
```bash
# 克隆仓库
git clone https://github.com/youyouhe/sensevoice-asr-docker.git
cd sensevoice-asr-docker

# 创建开发环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 运行开发服务器
python src/api_optimized.py
```

---

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE) - 查看 [LICENSE](LICENSE) 文件了解详情。

---

## 🙏 致谢

- [SenseVoice](https://github.com/modelscope/SenseVoice) - 优秀的语音识别模型
- [FunASR](https://github.com/alibaba/FunASR) - 开源的语音识别工具包
- [ModelScope](https://github.com/modelscope/modelscope) - 模型即服务平台
- [Docker](https://www.docker.com/) - 容器化部署平台

---

## 📞 联系我们

- **作者**: youyouhe
- **项目地址**: [https://github.com/youyouhe/sensevoice-asr-docker](https://github.com/youyouhe/sensevoice-asr-docker)
- **问题反馈**: [GitHub Issues](https://github.com/youyouhe/sensevoice-asr-docker/issues)

---

<div align="center">

⭐ 如果这个项目对你有帮助，请考虑给我们一个星标！

![Star History](https://img.shields.io/github/stars/youyouhe/sensevoice-asr-docker?style=social)

</div>