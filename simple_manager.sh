#!/bin/bash

# 简化版多实例ASR系统启动脚本
# 适合首次启动和测试

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_blue() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# 停止现有服务
stop_existing_services() {
    log_info "停止现有服务..."
    
    # 停止服务器
    if [ -f server.pid ]; then
        SERVER_PID=$(cat server.pid)
        if ps -p $SERVER_PID > /dev/null; then
            log_info "停止现有服务器 (PID: $SERVER_PID)"
            kill $SERVER_PID
            wait $SERVER_PID 2>/dev/null || true
        fi
        rm -f server.pid
    fi
    
    # 停止监控
    if [ -f monitoring.pid ]; then
        MONITOR_PID=$(cat monitoring.pid)
        if ps -p $MONITOR_PID > /dev/null; then
            log_info "停止现有监控 (PID: $MONITOR_PID)"
            kill $MONITOR_PID
            wait $MONITOR_PID 2>/dev/null || true
        fi
        rm -f monitoring.pid
    fi
}

# 检查Python环境
check_python() {
    log_info "检查Python环境..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安装"
        exit 1
    fi
    
    log_info "Python3 已安装"
}

# 创建必要的目录
create_directories() {
    log_info "创建必要的目录..."
    mkdir -p src/tmp
    mkdir -p monitoring_logs
}

# 创建简化版API服务器
create_simple_api() {
    log_info "创建简化版API服务器..."
    
    cat > src/api_simple.py << 'EOF'
#!/usr/bin/env python3
# 简化版单实例API服务器（用于测试）

import os, re
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from typing_extensions import Annotated
from typing import List
import torchaudio
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from io import BytesIO
from pathlib import Path
import time
from datetime import timedelta
from funasr import AutoModel
import torch
import shutil
from pydub import AudioSegment
import json
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TMPDIR=Path(os.path.dirname(__file__)+"/tmp").as_posix()
Path(TMPDIR).mkdir(exist_ok=True)

HOST='0.0.0.0'
PORT=5003

# 全局变量
model = None
vad_model = None

async def initialize_models():
    """初始化模型"""
    global model, vad_model
    
    logger.info("Loading models...")
    
    # 加载ASR模型
    model = AutoModel(
        model="iic/SenseVoiceSmall",
        punc_model="ct-punc",
        disable_update=True,
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    
    # 加载VAD模型
    vad_model = AutoModel(
        model="fsmn-vad",
        max_single_segment_time=3000,
        max_end_silence_time=500,
        disable_update=True,
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    
    logger.info("Models loaded successfully!")

def ms_to_time_string(*, ms=0, seconds=None):
    if seconds is None:
        td = timedelta(milliseconds=ms)
    else:
        td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    time_string = f"{hours}:{minutes}:{seconds},{milliseconds}"
    return format_time(time_string, ',')

def format_time(s_time="", separate=','):
    if not s_time.strip():
        return f'00:00:00{separate}000'
    hou, min, sec,ms = 0, 0, 0,0
    tmp = s_time.strip().split(':')
    if len(tmp) >= 3:
        hou,min,sec = tmp[-3].strip(),tmp[-2].strip(),tmp[-1].strip()
    elif len(tmp) == 2:
        min,sec = tmp[0].strip(),tmp[1].strip()
    elif len(tmp) == 1:
        sec = tmp[0].strip()
    if re.search(r',|\.', str(sec)):
        t = re.split(r',|\.', str(sec))
        sec = t[0].strip()
        ms=t[1].strip()
    else:
        ms = 0
    hou = f'{int(hou):02}'[-2:]
    min = f'{int(min):02}'[-2:]
    sec = f'{int(sec):02}'
    ms = f'{int(ms):03}'[-3:]
    return f"{hou}:{min}:{sec}{separate}{ms}"

def remove_unwanted_characters(text: str) -> str:
    allowed_characters = re.compile(r'[^\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af'
                                    r'a-zA-Z0-9\s.,!@#$%^&*()_+\-=\[\]{};\'"\\|<>/?，。！｛｝【】；''""《》、（）￥]+')
    return re.sub(allowed_characters, '', text)

def create_optimal_segments(segments, max_duration=8000):
    optimal_segments = []
    for seg in segments[0]['value']:
        start_time = seg[0]
        end_time = seg[1]
        duration = end_time - start_time
        if duration > max_duration:
            num_segments = max(2, int(duration / max_duration) + 1)
            segment_duration = duration // num_segments
            for i in range(num_segments):
                seg_start = start_time + i * segment_duration
                seg_end = start_time + (i + 1) * segment_duration
                if i == num_segments - 1:
                    seg_end = end_time
                optimal_segments.append([seg_start, seg_end])
        else:
            optimal_segments.append([start_time, end_time])
    return optimal_segments

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """启动时初始化模型"""
    logger.info("Starting up simple ASR server...")
    try:
        await initialize_models()
        logger.info("Simple ASR server started successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        raise e

@app.get("/", response_class=HTMLResponse)
async def root():
    return f"""
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset=utf-8>
            <title>Simple ASR API</title>
        </head>
        <body>
            <h2>Simple ASR API</h2>
            <p>API 地址为 http://{HOST}:{PORT}/asr</p>
            <p>简化版单实例ASR服务器（用于测试）</p>
        </body>
    </html>
    """

@app.get("/health")
async def health():
    """健康检查端点"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "vad_loaded": vad_model is not None,
        "timestamp": time.time()
    }

@app.post("/asr")
async def asr(file: UploadFile, lang: str = Form(...)):
    """ASR处理端点"""
    print(f'{lang=},{file.filename=}')
    if lang not in ['zh','ja','en','ko','yue']:
        return {"code":1,"msg":f'不支持的语言代码:{lang}'}
    
    # 检查模型是否已初始化
    if model is None or vad_model is None:
        return {"code": 500, "msg": "Models not initialized"}
    
    # 创建临时文件
    temp_file_path = f"{TMPDIR}/{file.filename}"
    try:
        with open(temp_file_path, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
        
        # 使用VAD进行语音活动检测
        segments = vad_model.generate(input=temp_file_path)
        audiodata = AudioSegment.from_file(temp_file_path)
        
        # 优化分段策略
        optimal_segments = create_optimal_segments(segments, max_duration=6000)
        
        srts = []
        for i, seg in enumerate(optimal_segments):
            chunk = audiodata[seg[0]:seg[1]]
            filename = f"{TMPDIR}/{seg[0]}-{seg[1]}.wav"
            chunk.export(filename)
            
            # 处理音频
            res = model.generate(
                input=filename,
                language=lang,
                use_itn=True
            )
            text = remove_unwanted_characters(rich_transcription_postprocess(res[0]["text"]))
            print(f'Segment {i+1}: {text}')
            srts.append(f'{len(srts)+1}\n{ms_to_time_string(ms=seg[0])} --> {ms_to_time_string(ms=seg[1])}\n{text.strip()}')
        
        return {"code":0,"msg":"ok","data":"\n\n".join(srts)}
        
    except Exception as e:
        logger.error(f"ASR processing error: {str(e)}")
        return {"code": 500, "msg": f"Processing error: {str(e)}"}
    finally:
        # 清理临时文件
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass

if __name__=='__main__':
    import uvicorn
    
    print("=== Simple ASR Server Starting ===")
    print(f"Server running on http://{HOST}:{PORT}")
    
    uvicorn.run("api_simple:app", host=HOST, port=PORT, log_level="info")
EOF
    
    chmod +x src/api_simple.py
    log_info "简化版API服务器创建完成"
}

# 启动简化版服务器
start_simple_server() {
    log_info "启动简化版ASR服务器..."
    
    # 检查端口是否被占用
    if lsof -i :5003 >/dev/null 2>&1; then
        log_warn "端口5003已被占用，尝试使用端口5004"
        PORT=5004
        # 修改API文件中的端口
        sed -i "s/PORT=5003/PORT=5004/" src/api_simple.py
    fi
    
    # 后台启动服务器，添加错误处理
    log_info "启动服务器进程..."
    python3 src/api_simple.py > simple_server.log 2>&1 &
    SIMPLE_PID=$!
    
    # 保存PID
    echo $SIMPLE_PID > simple_server.pid
    
    log_info "服务器启动中... (PID: $SIMPLE_PID)"
    
    # 等待服务器启动
    log_info "等待服务器启动...（模型下载可能需要几分钟）"
    
    max_wait=600  # 10分钟
    wait_count=0
    
    while [ $wait_count -lt $max_wait ]; do
        # 动态获取端口
        CURRENT_PORT=$(grep "PORT=" src/api_simple.py | cut -d'=' -f2)
        
        if curl -s "http://localhost:${CURRENT_PORT}/health" >/dev/null 2>&1; then
            log_info "服务器启动成功"
            break
        fi
        
        # 每10秒显示一次进度
        if [ $((wait_count % 10)) -eq 0 ]; then
            elapsed=$((wait_count))
            log_info "等待服务器启动... (${elapsed}s / ${max_wait}s)"
            
            # 检查进程是否还在运行
            if ! ps -p $SIMPLE_PID > /dev/null; then
                log_error "服务器进程已停止，请检查日志: simple_server.log"
                exit 1
            fi
            
            # 显示模型下载进度
            if [ -d ~/.cache/modelscope ]; then
                cache_size=$(du -sh ~/.cache/modelscope/ 2>/dev/null | cut -f1)
                log_info "模型缓存大小: $cache_size"
            fi
        fi
        
        sleep 1
        wait_count=$((wait_count + 1))
    done
    
    # 最终检查
    CURRENT_PORT=$(grep "PORT=" src/api_simple.py | cut -d'=' -f2)
    if curl -s "http://localhost:${CURRENT_PORT}/health" >/dev/null 2>&1; then
        log_info "服务器启动成功"
        log_blue "API地址: http://localhost:${CURRENT_PORT}"
        log_blue "健康检查: http://localhost:${CURRENT_PORT}/health"
    else
        log_error "服务器启动失败，请检查日志: simple_server.log"
        kill $SIMPLE_PID 2>/dev/null
        exit 1
    fi
}

# 测试API
test_api() {
    log_info "测试API..."
    
    CURRENT_PORT=$(grep "PORT=" src/api_simple.py | cut -d'=' -f2)
    
    # 检查健康状态
    if curl -s "http://localhost:${CURRENT_PORT}/health" >/dev/null 2>&1; then
        log_info "健康检查通过"
        
        # 显示健康状态
        health_response=$(curl -s "http://localhost:${CURRENT_PORT}/health")
        log_info "健康状态: $health_response"
    else
        log_error "健康检查失败"
        return 1
    fi
    
    # 如果有测试文件，进行ASR测试
    if [ -f "src/tmp/7_2.wav" ]; then
        log_info "进行ASR测试 (使用7_2.wav)..."
        response=$(curl -s -X POST "http://localhost:${CURRENT_PORT}/asr" \
            -F "file=@src/tmp/7_2.wav" \
            -F "lang=zh")
        
        if [ $? -eq 0 ]; then
            log_info "ASR测试成功"
            echo "响应预览:"
            echo "$response" | head -c 200
            echo ""
        else
            log_error "ASR测试失败"
        fi
    else
        log_warn "测试文件 src/tmp/7_2.wav 不存在，跳过ASR测试"
    fi
}

# 主函数
main() {
    case "${1:-start}" in
        "start")
            log_info "启动简化版ASR系统..."
            stop_existing_services
            check_python
            create_directories
            create_simple_api
            start_simple_server
            test_api
            log_info "简化版系统启动完成"
            ;;
        "stop")
            stop_existing_services
            ;;
        "status")
            if [ -f simple_server.pid ]; then
                SIMPLE_PID=$(cat simple_server.pid)
                if ps -p $SIMPLE_PID > /dev/null; then
                    log_info "简化版服务器运行中 (PID: $SIMPLE_PID)"
                    
                    CURRENT_PORT=$(grep "PORT=" src/api_simple.py | cut -d'=' -f2)
                    if curl -s "http://localhost:${CURRENT_PORT}/health" >/dev/null 2>&1; then
                        log_info "服务器健康检查通过"
                    else
                        log_warn "服务器健康检查失败"
                    fi
                else
                    log_error "简化版服务器未运行"
                fi
            else
                log_error "简化版服务器未运行"
            fi
            ;;
        "test")
            test_api
            ;;
        *)
            log_info "用法: $0 [start|stop|status|test]"
            log_info "  start  - 启动简化版服务器"
            log_info "  stop   - 停止服务器"
            log_info "  status - 查看状态"
            log_info "  test   - 测试API"
            ;;
    esac
}

# 运行主函数
main "$@"