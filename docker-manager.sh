#!/bin/bash

# SenseVoice ASR Docker构建和运行脚本
# 作者: youyouhe
# 版本: 1.0

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Docker是否安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker 未安装，请先安装Docker"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose 未安装，请先安装Docker Compose"
        exit 1
    fi
    
    print_success "Docker 环境检查通过"
}

# 检查NVIDIA GPU驱动
check_gpu() {
    if ! nvidia-smi &> /dev/null; then
        print_error "未检测到NVIDIA GPU或驱动，请确保安装了正确的GPU驱动"
        exit 1
    fi
    
    # 检查CUDA版本兼容性
    local driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
    print_info "检测到NVIDIA GPU，驱动版本: $driver_version"
    
    # 检查GPU数量
    local gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits)
    print_info "检测到 $gpu_count 个GPU"
}

# 清理旧的构建
cleanup() {
    print_info "清理旧的构建..."
    
    # 停止并删除旧容器
    docker-compose down -v --remove-orphans 2>/dev/null || true
    
    # 删除旧镜像
    docker rmi sensevoice-asr:latest 2>/dev/null || true
    
    # 清理未使用的镜像和缓存
    docker system prune -f 2>/dev/null || true
    
    print_success "清理完成"
}

# 构建Docker镜像
build_image() {
    print_info "开始构建Docker镜像..."
    
    # 使用优化的requirements文件
    cp requirements.docker.txt requirements.txt
    
    # 构建镜像
    if docker-compose build --no-cache; then
        print_success "Docker镜像构建成功"
    else
        print_error "Docker镜像构建失败"
        exit 1
    fi
    
    # 恢复原始requirements文件
    if [ -f requirements.txt.backup ]; then
        mv requirements.txt.backup requirements.txt
    fi
}

# 启动服务
start_service() {
    print_info "启动ASR服务..."
    
    # 启动服务
    if docker-compose up -d; then
        print_success "ASR服务启动成功"
    else
        print_error "ASR服务启动失败"
        exit 1
    fi
    
    # 等待服务就绪
    print_info "等待服务就绪..."
    sleep 10
    
    # 检查服务健康状态
    if check_health; then
        print_success "服务健康检查通过"
        print_info "API地址: http://localhost:5001"
        print_info "API文档: http://localhost:5001/docs"
    else
        print_error "服务健康检查失败"
        docker-compose logs asr-service
        exit 1
    fi
}

# 健康检查
check_health() {
    local max_attempts=12
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:5001/ &> /dev/null; then
            return 0
        fi
        
        print_info "等待服务启动... ($attempt/$max_attempts)"
        sleep 5
        ((attempt++))
    done
    
    return 1
}

# 运行测试
run_test() {
    print_info "运行API测试..."
    
    # 创建测试音频文件（如果没有的话）
    if [ ! -f "7.wav" ]; then
        print_warning "未找到7.wav测试文件，跳过测试"
        return 0
    fi
    
    # 运行测试
    if docker-compose exec asr-service python /app/test_asr.py /app/7.wav; then
        print_success "API测试通过"
    else
        print_error "API测试失败"
        return 1
    fi
}

# 显示服务状态
show_status() {
    print_info "=== 服务状态 ==="
    docker-compose ps
    
    print_info "\n=== 容器日志 ==="
    docker-compose logs --tail=20 asr-service
    
    print_info "\n=== 资源使用 ==="
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    fi
}

# 显示帮助信息
show_help() {
    echo "SenseVoice ASR Docker 管理脚本"
    echo ""
    echo "用法: $0 [命令]"
    echo ""
    echo "可用命令:"
    echo "  build     构建Docker镜像"
    echo "  start     启动ASR服务"
    echo "  stop      停止ASR服务"
    echo "  restart   重启ASR服务"
    echo "  test      运行API测试"
    echo "  status    显示服务状态"
    echo "  logs      查看服务日志"
    echo "  cleanup   清理所有资源"
    echo "  help      显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 build      # 构建镜像"
    echo "  $0 start      # 启动服务"
    echo "  $0 test       # 运行测试"
    echo ""
}

# 主函数
main() {
    case "${1:-help}" in
        "check")
            check_docker
            check_gpu
            ;;
        "build")
            check_docker
            check_gpu
            cleanup
            build_image
            ;;
        "start")
            check_docker
            start_service
            ;;
        "stop")
            docker-compose down
            print_success "服务已停止"
            ;;
        "restart")
            docker-compose restart
            print_success "服务已重启"
            ;;
        "test")
            run_test
            ;;
        "status")
            show_status
            ;;
        "logs")
            docker-compose logs -f asr-service
            ;;
        "cleanup")
            cleanup
            ;;
        "deploy")
            check_docker
            check_gpu
            cleanup
            build_image
            start_service
            sleep 5
            run_test
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# 执行主函数
main "$@"