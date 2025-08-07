"""
多实例ASR系统监控脚本
实时监控模型实例状态、性能指标和负载均衡情况
"""

import asyncio
import json
import time
import requests
import psutil
import GPUtil
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiInstanceMonitor:
    """多实例ASR系统监控器"""
    
    def __init__(self, api_url: str = "http://localhost:5002", monitoring_interval: int = 30):
        """
        初始化监控器
        
        Args:
            api_url: API服务器地址
            monitoring_interval: 监控间隔（秒）
        """
        self.api_url = api_url.rstrip('/')
        self.monitoring_interval = monitoring_interval
        self.is_running = False
        
        # 监控数据历史
        self.health_history = []
        self.stats_history = []
        self.performance_history = []
        
        # 监控阈值
        self.thresholds = {
            "min_healthy_ratio": 0.8,      # 最小健康实例比例
            "max_memory_usage": 0.9,       # 最大内存使用率
            "max_gpu_memory_usage": 0.9,   # 最大GPU内存使用率
            "min_success_rate": 0.95,      # 最小成功率
            "max_response_time": 10.0      # 最大响应时间（秒）
        }
        
        # 创建监控日志目录
        self.log_dir = Path("monitoring_logs")
        self.log_dir.mkdir(exist_ok=True)
    
    async def get_health_status(self) -> Dict[str, Any]:
        """获取API健康状态"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Health check failed: HTTP {response.status_code}")
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            return {"error": str(e)}
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取API统计信息"""
        try:
            response = requests.get(f"{self.api_url}/stats", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Stats check failed: HTTP {response.status_code}")
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            logger.error(f"Stats check error: {str(e)}")
            return {"error": str(e)}
    
    def get_system_resources(self) -> Dict[str, Any]:
        """获取系统资源使用情况"""
        try:
            # CPU和内存使用情况
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # GPU使用情况（如果可用）
            gpu_info = []
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    gpu_info.append({
                        "id": gpu.id,
                        "name": gpu.name,
                        "load": gpu.load * 100,
                        "memory_used": gpu.memoryUsed,
                        "memory_total": gpu.memoryTotal,
                        "memory_usage": gpu.memoryUtil * 100,
                        "temperature": gpu.temperature
                    })
            except Exception as e:
                logger.warning(f"GPU monitoring failed: {str(e)}")
            
            return {
                "timestamp": time.time(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "gpus": gpu_info
            }
        except Exception as e:
            logger.error(f"System resources check error: {str(e)}")
            return {"error": str(e)}
    
    def analyze_health(self, health_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析健康状态"""
        if "error" in health_data:
            return {"status": "error", "reason": health_data["error"]}
        
        analysis = {
            "status": "healthy",
            "warnings": [],
            "recommendations": []
        }
        
        # 检查健康实例比例
        healthy_ratio = health_data.get("health_ratio", 0)
        if healthy_ratio < self.thresholds["min_healthy_ratio"]:
            analysis["status"] = "degraded"
            analysis["warnings"].append(f"Low healthy instance ratio: {healthy_ratio:.2%}")
            analysis["recommendations"].append("Check model instances and restart if necessary")
        
        # 检查不健康的实例
        unhealthy_instances = health_data.get("unhealthy_instances", 0)
        if unhealthy_instances > 0:
            analysis["warnings"].append(f"{unhealthy_instances} unhealthy instances detected")
            analysis["recommendations"].append("Review instance logs for errors")
        
        # 详细健康检查
        health_details = health_data.get("health_details", [])
        for detail in health_details:
            if isinstance(detail, dict) and not detail.get("healthy", True):
                analysis["warnings"].append(f"Instance {detail.get('instance_id', 'unknown')} is unhealthy")
        
        return analysis
    
    def analyze_performance(self, stats_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析性能指标"""
        if "error" in stats_data:
            return {"status": "error", "reason": stats_data["error"]}
        
        analysis = {
            "status": "good",
            "warnings": [],
            "recommendations": []
        }
        
        # 分析模型池统计
        pool_stats = stats_data.get("model_pool_stats", {})
        total_requests = pool_stats.get("total_requests", 0)
        successful_requests = pool_stats.get("successful_requests", 0)
        failed_requests = pool_stats.get("failed_requests", 0)
        
        if total_requests > 0:
            success_rate = successful_requests / total_requests
            if success_rate < self.thresholds["min_success_rate"]:
                analysis["status"] = "poor"
                analysis["warnings"].append(f"Low success rate: {success_rate:.2%}")
                analysis["recommendations"].append("Check error logs and system resources")
        
        # 分析实例状态
        instances = pool_stats.get("instances", [])
        busy_instances = sum(1 for inst in instances if inst.get("status") == "busy")
        error_instances = sum(1 for inst in instances if inst.get("status") == "error")
        
        if error_instances > 0:
            analysis["warnings"].append(f"{error_instances} instances in error state")
            analysis["recommendations"].append("Restart error instances or check logs")
        
        # 检查负载分布
        if len(instances) > 0:
            avg_requests = sum(inst.get("request_count", 0) for inst in instances) / len(instances)
            for inst in instances:
                request_count = inst.get("request_count", 0)
                if request_count > avg_requests * 2:
                    analysis["warnings"].append(f"Instance {inst.get('instance_id')} has high load: {request_count} requests")
        
        return analysis
    
    def analyze_system_resources(self, resources_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析系统资源使用情况"""
        if "error" in resources_data:
            return {"status": "error", "reason": resources_data["error"]}
        
        analysis = {
            "status": "normal",
            "warnings": [],
            "recommendations": []
        }
        
        # CPU使用率检查
        cpu_percent = resources_data.get("cpu_percent", 0)
        if cpu_percent > 80:
            analysis["warnings"].append(f"High CPU usage: {cpu_percent:.1f}%")
            analysis["recommendations"].append("Consider reducing concurrent requests or scaling up")
        
        # 内存使用率检查
        memory_percent = resources_data.get("memory_percent", 0)
        if memory_percent > self.thresholds["max_memory_usage"] * 100:
            analysis["warnings"].append(f"High memory usage: {memory_percent:.1f}%")
            analysis["recommendations"].append("Free up memory or restart services")
        
        # GPU使用率检查
        gpus = resources_data.get("gpus", [])
        for gpu in gpus:
            gpu_memory_usage = gpu.get("memory_usage", 0)
            if gpu_memory_usage > self.thresholds["max_gpu_memory_usage"] * 100:
                analysis["warnings"].append(f"High GPU memory usage on GPU {gpu.get('id')}: {gpu_memory_usage:.1f}%")
                analysis["recommendations"].append("Consider reducing model instances or batch sizes")
            
            gpu_temp = gpu.get("temperature", 0)
            if gpu_temp > 80:
                analysis["warnings"].append(f"High GPU temperature on GPU {gpu.get('id')}: {gpu_temp}°C")
                analysis["recommendations"].append("Check cooling system and reduce load")
        
        return analysis
    
    def save_monitoring_data(self, data: Dict[str, Any], data_type: str):
        """保存监控数据到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.log_dir / f"{data_type}_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved {data_type} data to {filename}")
        except Exception as e:
            logger.error(f"Failed to save {data_type} data: {str(e)}")
    
    async def monitor_cycle(self) -> Dict[str, Any]:
        """执行一次监控周期"""
        logger.info("Starting monitoring cycle...")
        
        # 并发获取所有监控数据
        health_task = self.get_health_status()
        stats_task = self.get_stats()
        resources_data = self.get_system_resources()
        
        # 等待API数据获取完成
        health_data, stats_data = await asyncio.gather(health_task, stats_task)
        
        # 分析数据
        health_analysis = self.analyze_health(health_data)
        performance_analysis = self.analyze_performance(stats_data)
        resources_analysis = self.analyze_system_resources(resources_data)
        
        # 组合监控结果
        monitoring_result = {
            "timestamp": datetime.now().isoformat(),
            "health_data": health_data,
            "stats_data": stats_data,
            "resources_data": resources_data,
            "health_analysis": health_analysis,
            "performance_analysis": performance_analysis,
            "resources_analysis": resources_analysis,
            "overall_status": self._determine_overall_status(health_analysis, performance_analysis, resources_analysis)
        }
        
        # 保存监控数据
        self.save_monitoring_data(monitoring_result, "monitoring")
        
        # 添加到历史记录
        self.health_history.append(health_data)
        self.stats_history.append(stats_data)
        self.performance_history.append(monitoring_result)
        
        # 限制历史记录大小
        max_history = 100
        if len(self.health_history) > max_history:
            self.health_history = self.health_history[-max_history:]
        if len(self.stats_history) > max_history:
            self.stats_history = self.stats_history[-max_history:]
        if len(self.performance_history) > max_history:
            self.performance_history = self.performance_history[-max_history:]
        
        # 输出监控结果
        self._log_monitoring_result(monitoring_result)
        
        return monitoring_result
    
    def _determine_overall_status(self, health_analysis: Dict, performance_analysis: Dict, resources_analysis: Dict) -> str:
        """确定整体状态"""
        status_scores = {
            "healthy": 0,
            "good": 0,
            "normal": 0,
            "degraded": 1,
            "poor": 2,
            "error": 3
        }
        
        scores = [
            status_scores.get(health_analysis.get("status", "error"), 3),
            status_scores.get(performance_analysis.get("status", "error"), 3),
            status_scores.get(resources_analysis.get("status", "error"), 3)
        ]
        
        max_score = max(scores)
        if max_score == 0:
            return "healthy"
        elif max_score == 1:
            return "degraded"
        elif max_score == 2:
            return "poor"
        else:
            return "critical"
    
    def _log_monitoring_result(self, result: Dict[str, Any]):
        """输出监控结果"""
        overall_status = result["overall_status"]
        logger.info(f"Monitoring Cycle - Overall Status: {overall_status.upper()}")
        
        # 输出健康分析结果
        health_analysis = result["health_analysis"]
        if health_analysis["warnings"]:
            logger.warning(f"Health Warnings: {', '.join(health_analysis['warnings'])}")
        
        # 输出性能分析结果
        performance_analysis = result["performance_analysis"]
        if performance_analysis["warnings"]:
            logger.warning(f"Performance Warnings: {', '.join(performance_analysis['warnings'])}")
        
        # 输出资源分析结果
        resources_analysis = result["resources_analysis"]
        if resources_analysis["warnings"]:
            logger.warning(f"Resource Warnings: {', '.join(resources_analysis['warnings'])}")
        
        # 输出建议
        all_recommendations = []
        for analysis in [health_analysis, performance_analysis, resources_analysis]:
            all_recommendations.extend(analysis["recommendations"])
        
        if all_recommendations:
            logger.info(f"Recommendations: {', '.join(all_recommendations)}")
    
    async def start_monitoring(self):
        """启动监控"""
        logger.info("Starting multi-instance ASR monitoring...")
        self.is_running = True
        
        try:
            while self.is_running:
                await self.monitor_cycle()
                await asyncio.sleep(self.monitoring_interval)
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Monitoring error: {str(e)}")
        finally:
            self.is_running = False
    
    def stop_monitoring(self):
        """停止监控"""
        logger.info("Stopping monitoring...")
        self.is_running = False
    
    def get_recent_stats(self, hours: int = 1) -> Dict[str, Any]:
        """获取最近的统计信息"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_health = [data for data in self.health_history if datetime.fromisoformat(data.get("timestamp", "")) > cutoff_time]
        recent_stats = [data for data in self.stats_history if datetime.fromisoformat(data.get("timestamp", "")) > cutoff_time]
        recent_performance = [data for data in self.performance_history if datetime.fromisoformat(data.get("timestamp", "")) > cutoff_time]
        
        return {
            "time_range": f"Last {hours} hours",
            "health_records": len(recent_health),
            "stats_records": len(recent_stats),
            "performance_records": len(recent_performance),
            "recent_health": recent_health[-10:] if recent_health else [],
            "recent_stats": recent_stats[-10:] if recent_stats else [],
            "recent_performance": recent_performance[-10:] if recent_performance else []
        }
    
    def generate_report(self) -> str:
        """生成监控报告"""
        if not self.performance_history:
            return "No monitoring data available"
        
        latest = self.performance_history[-1]
        report = f"""
Multi-Instance ASR Monitoring Report
=====================================
Generated: {latest['timestamp']}
Overall Status: {latest['overall_status'].upper()}

Health Status:
- Healthy Instances: {latest['health_data'].get('healthy_instances', 0)}/{latest['health_data'].get('total_instances', 0)}
- Health Ratio: {latest['health_data'].get('health_ratio', 0):.2%}
- Status: {latest['health_analysis'].get('status', 'unknown')}

Performance Metrics:
- Total Requests: {latest['stats_data'].get('model_pool_stats', {}).get('total_requests', 0)}
- Successful Requests: {latest['stats_data'].get('model_pool_stats', {}).get('successful_requests', 0)}
- Failed Requests: {latest['stats_data'].get('model_pool_stats', {}).get('failed_requests', 0)}
- Success Rate: {latest['stats_data'].get('model_pool_stats', {}).get('success_rate', 0):.2%}

System Resources:
- CPU Usage: {latest['resources_data'].get('cpu_percent', 0):.1f}%
- Memory Usage: {latest['resources_data'].get('memory_percent', 0):.1f}%
- Available GPUs: {len(latest['resources_data'].get('gpus', []))}

Warnings and Recommendations:
"""
        
        # 添加警告和建议
        for analysis_type in ['health_analysis', 'performance_analysis', 'resources_analysis']:
            analysis = latest.get(analysis_type, {})
            if analysis.get('warnings'):
                report += f"\n{analysis_type.replace('_', ' ').title()} Warnings:\n"
                for warning in analysis['warnings']:
                    report += f"  - {warning}\n"
            
            if analysis.get('recommendations'):
                report += f"\n{analysis_type.replace('_', ' ').title()} Recommendations:\n"
                for recommendation in analysis['recommendations']:
                    report += f"  - {recommendation}\n"
        
        return report

async def main():
    """主函数"""
    monitor = MultiInstanceMonitor(api_url="http://localhost:5002", monitoring_interval=30)
    
    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\nGenerating final report...")
        report = monitor.generate_report()
        print(report)
        
        # 保存最终报告
        with open("final_monitoring_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        print("Final report saved to final_monitoring_report.txt")

if __name__ == "__main__":
    asyncio.run(main())