#!/bin/bash

# 智能清理脚本 - 保留有用的缓存，只清理有问题部分
# 作者: youyouhe

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# 智能清理函数
smart_cleanup() {
    print_info "开始智能清理..."
    
    # 1. 停止容器（但不删除卷）
    print_info "停止运行中的容器..."
    docker-compose down --remove-orphans 2>/dev/null || true
    
    # 2. 只删除有问题的镜像（保留成功的层）
    print_info "检查并清理有问题的镜像..."
    
    # 检查当前镜像是否存在问题
    if docker images sensevoice-asr:latest &>/dev/null; then
        # 尝试创建临时容器检查镜像是否正常
        if ! docker run --rm --entrypoint="" sensevoice-asr:latest python --version &>/dev/null; then
            print_warning "检测到镜像问题，将重新构建"
            docker rmi sensevoice-asr:latest 2>/dev/null || true
        else
            print_success "当前镜像正常，保留使用"
            return 0
        fi
    fi
    
    # 3. 清理dangling镜像（无标签的镜像）
    DANGLING_IMAGES=$(docker images -f "dangling=true" -q)
    if [ -n "$DANGLING_IMAGES" ]; then
        print_info "清理dangling镜像..."
        echo "$DANGLING_IMAGES" | xargs -r docker rmi 2>/dev/null || true
    fi
    
    # 4. 清理未使用的build cache（保留有用的）
    print_info "清理未使用的build cache..."
    docker builder prune -f 2>/dev/null || true
    
    # 5. 清理停止的容器（不影响运行中的）
    docker container prune -f 2>/dev/null || true
    
    print_success "智能清理完成！保留了有用的缓存和镜像"
}

# 快速重建（不清理缓存）
fast_rebuild() {
    print_info "开始快速重建..."
    
    # 只停止容器，不删除任何缓存
    docker-compose down --remove-orphans 2>/dev/null || true
    
    # 直接构建，利用Docker的层缓存
    docker-compose build --no-cache=false
    
    print_success "快速重建完成！"
}

# 完全重建（清理所有）
full_rebuild() {
    print_info "开始完全重建..."
    
    # 停止容器并删除卷（包括模型缓存）
    docker-compose down -v --remove-orphans 2>/dev/null || true
    
    # 删除镜像
    docker rmi sensevoice-asr:latest 2>/dev/null || true
    
    # 清理所有缓存
    docker system prune -f 2>/dev/null || true
    
    print_success "完全重建完成！这将会重新下载所有依赖和模型。"
}

# 检查构建状态
check_build_status() {
    print_info "检查构建状态..."
    
    # 检查镜像是否存在
    if docker images sensevoice-asr:latest &>/dev/null; then
        print_success "✅ 镜像已存在: sensevoice-asr:latest"
        
        # 检查容器是否运行
        if docker ps --format "table {{.Names}}" | grep -q "sensevoice-asr"; then
            print_success "✅ 容器正在运行"
            
            # 检查健康状态
            if curl -f http://localhost:5001/ &>/dev/null; then
                print_success "✅ 服务健康"
            else
                print_warning "⚠️  服务可能有问题"
            fi
        else
            print_warning "⚠️  容器未运行"
        fi
    else
        print_warning "⚠️  镜像不存在，需要构建"
    fi
    
    # 显示缓存使用情况
    echo ""
    print_info "Docker缓存使用情况:"
    docker system df 2>/dev/null || echo "无法获取缓存信息"
}

# 主函数
case "${1:-help}" in
    "smart")
        smart_cleanup
        ;;
    "fast")
        fast_rebuild
        ;;
    "full")
        full_rebuild
        ;;
    "status")
        check_build_status
        ;;
    "help"|*)
        echo "智能构建管理脚本"
        echo ""
        echo "用法: $0 [命令]"
        echo ""
        echo "可用命令:"
        echo "  smart   - 智能清理（保留有用缓存）"
        echo "  fast    - 快速重建（利用层缓存）"
        echo "  full    - 完全重建（清理所有）"
        echo "  status  - 检查构建状态"
        echo "  help    - 显示帮助信息"
        echo ""
        echo "推荐使用场景："
        echo "  - 首次构建失败 → smart"
        echo "  - 代码修改 → fast"
        echo "  - 严重问题 → full"
        ;;
esac