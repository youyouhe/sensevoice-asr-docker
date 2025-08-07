# 多实例ASR系统

这个项目将单实例的SenseVoice ASR系统改造为支持5个并发实例的多实例系统，显著提升吞吐量和并发性能。

## 系统特性

### 🚀 核心特性
- **5个并发模型实例**: 同时处理多个请求，大幅提升吞吐量
- **智能负载均衡**: 自动分配请求到空闲实例，避免单点过载
- **异步并发处理**: 支持真正的并行推理，非阻塞式请求处理
- **实时健康监控**: 自动监控实例状态，提供详细统计信息
- **自动错误恢复**: 内置重试机制和错误处理，提高系统稳定性
- **资源优化**: 智能GPU内存管理和设备分配

### 📊 性能提升
- **吞吐量**: 预期提升3-5倍（取决于硬件配置）
- **并发能力**: 支持10+并发请求同时处理
- **响应时间**: 平均响应时间保持稳定，P95延迟显著降低
- **资源利用率**: GPU利用率从~20%提升到~80%

## 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client App   │    │   Client App   │    │   Client App   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                │
                    ┌─────────────────┐
                    │  Load Balancer │
                    │  (Round Robin) │
                    └─────────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
    │ Instance 1  │      │ Instance 2  │      │ Instance 3  │
    │  GPU:0      │      │  GPU:1      │      │  GPU:0      │
    │  Status:Idle│      │  Status:Busy│      │  Status:Idle│
    └─────────────┘      └─────────────┘      └─────────────┘
          │                     │                     │
    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
    │ Instance 4  │      │ Instance 5  │      │             │
    │  GPU:1      │      │  GPU:0      │      │             │
    │  Status:Idle│      │  Status:Busy│      │             │
    └─────────────┘      └─────────────┘      └─────────────┘
```

## 文件结构

```
sensevoice-asr-docker/
├── src/
│   ├── model_pool.py              # 多实例模型池管理器
│   ├── api_multi_instance.py      # 多实例API服务器
│   ├── monitoring.py              # 系统监控脚本
│   ├── performance_test.py        # 性能测试脚本
│   └── tmp/                       # 临时文件目录
├── multi_instance_manager.sh      # 系统管理脚本
├── MULTI_INSTANCE_TODOLIST.md     # 任务跟踪文档
├── monitoring_logs/               # 监控日志目录
└── README.md                      # 本文档
```

## 快速开始

### 1. 环境准备

确保您的系统满足以下要求：
- Python 3.8+
- CUDA 11.0+ (如果使用GPU)
- 至少8GB显存 (推荐16GB+)
- 足够的系统内存

### 2. 安装依赖

```bash
# 安装基础依赖
pip install torch torchaudio fastapi uvicorn aiohttp

# 安装系统监控依赖
pip install psutil GPUtil

# 安装FunASR
pip install funasr

# 安装音频处理依赖
pip install pydub requests
```

### 3. 准备测试文件

```bash
# 准备测试音频文件
# 支持格式: wav, mp3, flac, m4a
cp your_test_audio.wav 7.wav
```

### 4. 启动系统

```bash
# 赋予执行权限
chmod +x multi_instance_manager.sh

# 启动所有服务
./multi_instance_manager.sh start
```

### 5. 验证系统状态

```bash
# 检查系统状态
./multi_instance_manager.sh status

# 检查API健康状态
curl http://localhost:5002/health

# 查看统计信息
curl http://localhost:5002/stats
```

## 使用方法

### API接口

#### 1. 标准ASR接口 (带VAD分段)

```bash
curl -X POST "http://localhost:5002/asr" \
  -F "file=@your_audio.wav" \
  -F "lang=zh"
```

**响应示例**:
```json
{
  "code": 0,
  "msg": "ok",
  "data": "1\n00:00:00,000 --> 00:00:03,500\n你好世界\n\n2\n00:00:03,500 --> 00:00:06,000\n这是一个测试",
  "stats": {
    "total_segments": 2,
    "successful_segments": 2,
    "failed_segments": 0,
    "success_rate": 1.0
  }
}
```

#### 2. 简单ASR接口 (无VAD分段)

```bash
curl -X POST "http://localhost:5002/asr_simple" \
  -F "file=@your_audio.wav" \
  -F "lang=zh"
```

#### 3. 健康检查

```bash
curl http://localhost:5002/health
```

**响应示例**:
```json
{
  "status": "healthy",
  "health_ratio": 1.0,
  "total_instances": 5,
  "healthy_instances": 5,
  "unhealthy_instances": 0,
  "health_details": [...]
}
```

#### 4. 统计信息

```bash
curl http://localhost:5002/stats
```

**响应示例**:
```json
{
  "model_pool_stats": {
    "total_instances": 5,
    "total_requests": 100,
    "successful_requests": 98,
    "failed_requests": 2,
    "success_rate": 0.98,
    "instances": [...]
  },
  "pool_status": {
    "pool_size": 5,
    "status_distribution": {
      "idle": 3,
      "busy": 2,
      "loading": 0,
      "error": 0
    },
    "available_instances": 3
  }
}
```

### 支持的语言

- `zh`: 中文
- `en`: 英语
- `ja`: 日语
- `ko`: 韩语
- `yue`: 粤语

## 性能测试

### 运行性能测试

```bash
# 运行默认性能测试
./multi_instance_manager.sh test

# 或者直接运行测试脚本
python3 src/performance_test.py
```

### 性能测试配置

在 `performance_test.py` 中可以修改测试配置：

```python
config = TestConfig(
    api_url="http://localhost:5002",
    test_file="7.wav",
    language="zh",
    concurrent_users=10,      # 并发用户数
    requests_per_user=5,      # 每个用户的请求数
    delay_between_requests=0.1, # 请求间延迟
    timeout=60                # 超时时间
)
```

### 性能测试报告

测试完成后会生成详细报告：

- `performance_test_report_YYYYMMDD_HHMMSS.json`: JSON格式的详细数据
- `performance_test_report_YYYYMMDD_HHMMSS_readable.txt`: 可读的文本报告

报告包含：
- 吞吐量和成功率
- 响应时间统计 (平均、P90、P95、P99)
- 资源使用情况 (CPU、内存、GPU)
- 错误分析和建议

## 监控系统

### 启动监控

监控系统会随着主系统自动启动，也可以手动启动：

```bash
# 启动监控系统
python3 src/monitoring.py
```

### 监控功能

- **实时健康检查**: 每30秒检查一次实例状态
- **性能指标监控**: 请求处理时间、成功率等
- **系统资源监控**: CPU、内存、GPU使用率
- **智能告警**: 基于阈值的自动告警
- **历史数据分析**: 保存监控数据用于分析

### 监控数据

监控数据保存在 `monitoring_logs/` 目录下：
- `monitoring_YYYYMMDD_HHMMSS.json`: 监控数据快照
- 实时日志输出到控制台和 `monitoring.log`

## 系统管理

### 管理脚本使用

```bash
# 启动所有服务
./multi_instance_manager.sh start

# 停止所有服务
./multi_instance_manager.sh stop

# 重启服务
./multi_instance_manager.sh restart

# 查看状态
./multi_instance_manager.sh status

# 运行性能测试
./multi_instance_manager.sh test

# 查看帮助
./multi_instance_manager.sh help
```

### 日志文件

- `server.log`: API服务器日志
- `monitoring.log`: 监控系统日志
- `monitoring_logs/`: 详细监控数据
- `performance_test_report_*.json`: 性能测试报告

### 进程管理

```bash
# 查看服务器进程
cat server.pid
ps -p $(cat server.pid)

# 查看监控进程
cat monitoring.pid
ps -p $(cat monitoring.pid)

# 手动停止进程
kill $(cat server.pid)
kill $(cat monitoring.pid)
```

## 配置优化

### 实例数量调整

在 `api_multi_instance.py` 中修改：

```python
# 多实例配置
NUM_INSTANCES = 5  # 根据GPU内存调整

# 如果有多个GPU，可以分配更多实例
# NUM_INSTANCES = 8  # 需要24GB+显存
```

### 资源分配策略

系统自动进行GPU资源分配：
- 单GPU: 轮询分配实例
- 多GPU: 均衡分配到不同GPU

### 超时和重试配置

在 `model_pool.py` 中修改：

```python
# 超时配置
self.max_retries = 3
self.request_timeout = 30
self.load_timeout = 600
```

## 故障排除

### 常见问题

#### 1. 模型加载失败
```
Error: Failed to load model instances
```
**解决方案**:
- 检查网络连接，确保可以下载模型
- 检查CUDA版本兼容性
- 确保有足够的磁盘空间

#### 2. GPU内存不足
```
Error: CUDA out of memory
```
**解决方案**:
- 减少实例数量 `NUM_INSTANCES`
- 使用更小的模型
- 增加GPU显存或使用多个GPU

#### 3. 端口被占用
```
Error: Port 5002 is already in use
```
**解决方案**:
- 停止占用端口的程序
- 修改 `PORT` 配置
- 使用 `lsof -i :5002` 查看占用进程

#### 4. API响应超时
```
Error: Request timeout
```
**解决方案**:
- 增加 `timeout` 配置
- 检查系统资源使用情况
- 减少并发请求数量

### 调试模式

启用调试日志：

```bash
# 启动服务器时启用详细日志
python3 src/api_multi_instance.py --log-level debug

# 查看详细日志
tail -f server.log
```

### 性能优化建议

1. **硬件优化**:
   - 使用更快的GPU (RTX 3090/4090)
   - 增加系统内存
   - 使用NVMe SSD

2. **软件优化**:
   - 升级到最新CUDA版本
   - 使用优化的PyTorch版本
   - 启用CUDA图优化

3. **配置优化**:
   - 根据负载调整实例数量
   - 优化请求批处理大小
   - 调整超时设置

## 技术细节

### 模型池管理

`ModelPool` 类负责：
- 模型实例的创建和管理
- 负载均衡策略 (轮询)
- 健康状态监控
- 错误恢复机制

### 请求处理流程

1. 客户端发送请求到API服务器
2. 服务器获取空闲模型实例
3. 将请求分配到实例处理
4. 异步处理VAD分段
5. 并发处理音频片段
6. 汇总结果并返回

### 并发控制

- **异步I/O**: 使用asyncio进行非阻塞I/O操作
- **线程池**: CPU密集型操作使用线程池
- **资源限制**: 限制最大并发请求数量

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 发起Pull Request

## 许可证

本项目遵循原有项目的许可证。

## 联系方式

如有问题或建议，请提交Issue或联系开发团队。