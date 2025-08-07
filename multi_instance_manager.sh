#!/bin/bash

# 多实例ASR系统启动脚本
# 启动多实例ASR服务器和监控系统

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

# 检查Python环境
check_python() {
    log_info "检查Python环境..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安装"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log_info "Python版本: $PYTHON_VERSION"
    
    # 检查必需的包
    REQUIRED_PACKAGES="torch fastapi uvicorn aiohttp psutil GPUtil"
    for package in $REQUIRED_PACKAGES; do
        if ! python3 -c "import $package" 2>/dev/null; then
            log_warn "缺少包: $package"
            log_info "安装中: pip install $package"
            pip install $package
        fi
    done
}

# 检查CUDA环境
check_cuda() {
    log_info "检查CUDA环境..."
    
    if command -v nvidia-smi &> /dev/null; then
        log_info "NVIDIA GPU检测到"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
    else
        log_warn "未检测到NVIDIA GPU，将使用CPU模式"
    fi
}

# 检查FunASR环境
check_funasr() {
    log_info "检查FunASR环境..."
    
    if ! python3 -c "import funasr" 2>/dev/null; then
        log_error "FunASR未安装"
        log_info "请安装FunASR: pip install funasr"
        exit 1
    fi
    
    log_info "FunASR已安装"
}

# 创建必要的目录
create_directories() {
    log_info "创建必要的目录..."
    
    # 创建临时目录
    mkdir -p src/tmp
    
    # 创建日志目录
    mkdir -p monitoring_logs
    
    # 创建模型缓存目录
    mkdir -p ~/.cache/modelscope/hub
}

# 检查测试文件
check_test_file() {
    log_info "检查测试文件..."
    
    if [ ! -f "7.wav" ]; then
        log_warn "测试文件 7.wav 不存在"
        log_info "请提供一个测试音频文件，或使用其他文件名"
        read -p "输入测试音频文件路径: " test_file
        if [ -f "$test_file" ]; then
            ln -sf "$test_file" 7.wav
            log_info "已创建测试文件链接"
        else
            log_error "测试文件不存在: $test_file"
            exit 1
        fi
    fi
}

# 启动多实例ASR服务器
start_server() {
    log_info "启动多实例ASR服务器..."
    
    # 检查端口是否被占用
    if lsof -i :5002 >/dev/null 2>&1; then
        log_error "端口5002已被占用"
        exit 1
    fi
    
    # 后台启动服务器
    nohup python3 src/api_multi_instance.py > server.log 2>&1 &
    SERVER_PID=$!
    
    # 保存PID
    echo $SERVER_PID > server.pid
    
    log_info "服务器启动中... (PID: $SERVER_PID)"
    
    # 等待服务器启动（模型下载可能需要较长时间）
    log_info "等待服务器启动...（模型下载可能需要几分钟）"
    
    # 渐进式检查，最多等待5分钟
    max_wait=300  # 5分钟
    wait_count=0
    
    while [ $wait_count -lt $max_wait ]; do
        if curl -s http://localhost:5002/health >/dev/null 2>&1; then
            log_info "服务器启动成功"
            break
        fi
        
        # 每10秒显示一次进度
        if [ $((wait_count % 10)) -eq 0 ]; then
            elapsed=$((wait_count))
            log_info "等待服务器启动... (${elapsed}s / ${max_wait}s)"
            
            # 检查进程是否还在运行
            if ! ps -p $SERVER_PID > /dev/null; then
                log_error "服务器进程已停止，请检查日志: server.log"
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
    if curl -s http://localhost:5002/health >/dev/null 2>&1; then
        log_info "服务器启动成功"
        log_blue "API地址: http://localhost:5002"
        log_blue "健康检查: http://localhost:5002/health"
        log_blue "统计信息: http://localhost:5002/stats"
    else
        log_error "服务器启动失败，请检查日志: server.log"
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
}

# 启动监控系统
start_monitoring() {
    log_info "启动监控系统..."
    
    # 后台启动监控
    nohup python3 src/monitoring.py > monitoring.log 2>&1 &
    MONITOR_PID=$!
    
    # 保存PID
    echo $MONITOR_PID > monitoring.pid
    
    log_info "监控系统启动中... (PID: $MONITOR_PID)"
    
    # 等待监控系统启动
    sleep 5
    
    if ps -p $MONITOR_PID > /dev/null; then
        log_info "监控系统启动成功"
        log_blue "监控日志: monitoring.log"
    else
        log_error "监控系统启动失败，请检查日志: monitoring.log"
    fi
}

# 显示系统状态
show_status() {
    log_info "系统状态检查..."
    
    # 检查服务器状态
    if [ -f server.pid ]; then
        SERVER_PID=$(cat server.pid)
        if ps -p $SERVER_PID > /dev/null; then
            log_info "ASR服务器运行中 (PID: $SERVER_PID)"
            
            # 获取健康状态
            if curl -s http://localhost:5002/health >/dev/null 2>&1; then
                log_info "服务器健康检查通过"
            else
                log_warn "服务器健康检查失败"
            fi
        else
            log_error "ASR服务器未运行"
        fi
    fi
    
    # 检查监控系统状态
    if [ -f monitoring.pid ]; then
        MONITOR_PID=$(cat monitoring.pid)
        if ps -p $MONITOR_PID > /dev/null; then
            log_info "监控系统运行中 (PID: $MONITOR_PID)"
        else
            log_error "监控系统未运行"
        fi
    fi
}

# 运行性能测试
run_performance_test() {
    log_info "运行性能测试..."
    
    # 检查服务器是否运行
    if ! curl -s http://localhost:5002/health >/dev/null 2>&1; then
        log_error "服务器未运行，无法运行性能测试"
        exit 1
    fi
    
    # 运行测试
    python3 src/performance_test.py
}

# 停止所有服务
stop_services() {
    log_info "停止所有服务..."
    
    # 停止服务器
    if [ -f server.pid ]; then
        SERVER_PID=$(cat server.pid)
        if ps -p $SERVER_PID > /dev/null; then
            log_info "停止ASR服务器 (PID: $SERVER_PID)"
            kill $SERVER_PID
            wait $SERVER_PID 2>/dev/null
        fi
        rm -f server.pid
    fi
    
    # 停止监控系统
    if [ -f monitoring.pid ]; then
        MONITOR_PID=$(cat monitoring.pid)
        if ps -p $MONITOR_PID > /dev/null; then
            log_info "停止监控系统 (PID: $MONITOR_PID)"
            kill $MONITOR_PID
            wait $MONITOR_PID 2>/dev/null
        fi
        rm -f monitoring.pid
    fi
    
    log_info "所有服务已停止"
}

# 显示使用帮助
show_help() {
    echo "多实例ASR系统管理脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  start       启动所有服务"
    echo "  stop        停止所有服务"
    echo "  restart     重启所有服务"
    echo "  status      显示系统状态"
    echo "  test        运行性能测试"
    echo "  help        显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 start    # 启动所有服务"
    echo "  $0 status   # 检查系统状态"
    echo "  $0 test     # 运行性能测试"
    echo "  $0 stop     # 停止所有服务"
}

# 主函数
main() {
    case "${1:-start}" in
        "start")
            log_info "启动多实例ASR系统..."
            check_python
            check_cuda
            check_funasr
            create_directories
            check_test_file
            start_server
            start_monitoring
            show_status
            log_info "系统启动完成"
            ;;
        "stop")
            stop_services
            ;;
        "restart")
            log_info "重启多实例ASR系统..."
            stop_services
            sleep 2
            check_python
            check_cuda
            check_funasr
            create_directories
            check_test_file
            start_server
            start_monitoring
            show_status
            log_info "系统重启完成"
            ;;
        "status")
            show_status
            ;;
        "test")
            run_performance_test
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            log_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"