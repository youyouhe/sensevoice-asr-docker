"""
多实例ASR性能测试脚本
测试并发性能、吞吐量和GPU内存使用情况
"""

import asyncio
import aiohttp
import time
import json
import os
import psutil
import GPUtil
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import concurrent.futures
import statistics
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """测试结果"""
    request_id: int
    start_time: float
    end_time: float
    duration: float
    status_code: int
    success: bool
    error_message: Optional[str] = None
    response_time: Optional[float] = None
    processing_time: Optional[float] = None

@dataclass
class TestConfig:
    """测试配置"""
    api_url: str = "http://localhost:5002"
    test_file: str = "7.wav"  # 测试音频文件
    language: str = "zh"
    concurrent_users: int = 10
    requests_per_user: int = 5
    delay_between_requests: float = 0.1
    timeout: int = 60

class PerformanceTester:
    """性能测试器"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.results: List[TestResult] = []
        self.resource_monitor = ResourceMonitor()
        self.test_start_time = 0
        self.test_end_time = 0
        
    async def run_single_request(self, session: aiohttp.ClientSession, request_id: int) -> TestResult:
        """执行单个请求"""
        start_time = time.time()
        
        try:
            # 检查测试文件是否存在
            if not os.path.exists(self.config.test_file):
                return TestResult(
                    request_id=request_id,
                    start_time=start_time,
                    end_time=time.time(),
                    duration=0,
                    status_code=0,
                    success=False,
                    error_message=f"Test file not found: {self.config.test_file}"
                )
            
            # 准备文件上传
            data = aiohttp.FormData()
            data.add_field('lang', self.config.language)
            
            # 读取文件内容
            with open(self.config.test_file, 'rb') as f:
                file_data = f.read()
                data.add_field('file', file_data, filename=os.path.basename(self.config.test_file))
            
            # 发送请求
            response_start = time.time()
            async with session.post(
                f"{self.config.api_url}/asr",
                data=data,
                timeout=self.config.timeout
            ) as response:
                response_end = time.time()
                
                # 读取响应
                response_data = await response.text()
                
                end_time = time.time()
                duration = end_time - start_time
                
                if response.status == 200:
                    try:
                        result_json = json.loads(response_data)
                        success = result_json.get('code') == 0
                        error_msg = result_json.get('msg') if not success else None
                    except json.JSONDecodeError:
                        success = False
                        error_msg = "Invalid JSON response"
                else:
                    success = False
                    error_msg = f"HTTP {response.status}"
                
                return TestResult(
                    request_id=request_id,
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    status_code=response.status,
                    success=success,
                    error_message=error_msg,
                    response_time=response_end - response_start,
                    processing_time=duration - (response_end - response_start)
                )
                
        except asyncio.TimeoutError:
            return TestResult(
                request_id=request_id,
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time,
                status_code=0,
                success=False,
                error_message="Request timeout"
            )
        except Exception as e:
            return TestResult(
                request_id=request_id,
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time,
                status_code=0,
                success=False,
                error_message=str(e)
            )
    
    async def run_user_simulation(self, user_id: int) -> List[TestResult]:
        """模拟单个用户的请求"""
        user_results = []
        
        for req_id in range(self.config.requests_per_user):
            request_id = user_id * self.config.requests_per_user + req_id
            
            # 创建HTTP会话
            async with aiohttp.ClientSession() as session:
                result = await self.run_single_request(session, request_id)
                user_results.append(result)
                
                # 请求间延迟
                if req_id < self.config.requests_per_user - 1:
                    await asyncio.sleep(self.config.delay_between_requests)
        
        return user_results
    
    async def run_concurrent_test(self) -> Dict[str, Any]:
        """运行并发测试"""
        logger.info(f"Starting concurrent test with {self.config.concurrent_users} users, "
                   f"{self.config.requests_per_user} requests per user")
        
        # 启动资源监控
        monitor_thread = threading.Thread(target=self.resource_monitor.start_monitoring)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        self.test_start_time = time.time()
        
        # 创建并发任务
        tasks = []
        for user_id in range(self.config.concurrent_users):
            task = asyncio.create_task(self.run_user_simulation(user_id))
            tasks.append(task)
        
        # 等待所有任务完成
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        self.test_end_time = time.time()
        
        # 停止资源监控
        self.resource_monitor.stop_monitoring()
        
        # 收集结果
        self.results = []
        for user_results in all_results:
            if isinstance(user_results, list):
                self.results.extend(user_results)
            elif isinstance(user_results, Exception):
                logger.error(f"User simulation error: {str(user_results)}")
        
        # 分析结果
        return self.analyze_results()
    
    def analyze_results(self) -> Dict[str, Any]:
        """分析测试结果"""
        if not self.results:
            return {"error": "No test results available"}
        
        # 基本统计
        total_requests = len(self.results)
        successful_requests = sum(1 for r in self.results if r.success)
        failed_requests = total_requests - successful_requests
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        
        # 响应时间统计
        response_times = [r.duration for r in self.results if r.success]
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            median_response_time = statistics.median(response_times)
            
            # 百分位数
            sorted_times = sorted(response_times)
            p90 = sorted_times[int(len(sorted_times) * 0.9)]
            p95 = sorted_times[int(len(sorted_times) * 0.95)]
            p99 = sorted_times[int(len(sorted_times) * 0.99)]
        else:
            avg_response_time = min_response_time = max_response_time = median_response_time = 0
            p90 = p95 = p99 = 0
        
        # 吞吐量计算
        test_duration = self.test_end_time - self.test_start_time
        throughput = successful_requests / test_duration if test_duration > 0 else 0
        
        # 获取资源监控数据
        resource_stats = self.resource_monitor.get_statistics()
        
        # 错误分析
        error_messages = {}
        for result in self.results:
            if not result.success and result.error_message:
                error_messages[result.error_message] = error_messages.get(result.error_message, 0) + 1
        
        # 生成报告
        report = {
            "test_summary": {
                "start_time": datetime.fromtimestamp(self.test_start_time).isoformat(),
                "end_time": datetime.fromtimestamp(self.test_end_time).isoformat(),
                "duration": test_duration,
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "success_rate": success_rate,
                "throughput_requests_per_second": throughput
            },
            "performance_metrics": {
                "avg_response_time": avg_response_time,
                "min_response_time": min_response_time,
                "max_response_time": max_response_time,
                "median_response_time": median_response_time,
                "p90_response_time": p90,
                "p95_response_time": p95,
                "p99_response_time": p99
            },
            "resource_usage": resource_stats,
            "error_analysis": error_messages,
            "configuration": {
                "api_url": self.config.api_url,
                "test_file": self.config.test_file,
                "language": self.config.language,
                "concurrent_users": self.config.concurrent_users,
                "requests_per_user": self.config.requests_per_user,
                "delay_between_requests": self.config.delay_between_requests,
                "timeout": self.config.timeout
            }
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any], filename: str = None):
        """保存测试报告"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_test_report_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Test report saved to {filename}")
        
        # 同时保存可读版本
        readable_filename = filename.replace('.json', '_readable.txt')
        with open(readable_filename, 'w', encoding='utf-8') as f:
            f.write(self.generate_readable_report(report))
        
        logger.info(f"Readable report saved to {readable_filename}")
    
    def generate_readable_report(self, report: Dict[str, Any]) -> str:
        """生成可读的测试报告"""
        readable = f"""
Multi-Instance ASR Performance Test Report
===========================================

Test Configuration:
- API URL: {report['configuration']['api_url']}
- Test File: {report['configuration']['test_file']}
- Language: {report['configuration']['language']}
- Concurrent Users: {report['configuration']['concurrent_users']}
- Requests per User: {report['configuration']['requests_per_user']}
- Delay Between Requests: {report['configuration']['delay_between_requests']}s
- Timeout: {report['configuration']['timeout']}s

Test Summary:
- Start Time: {report['test_summary']['start_time']}
- End Time: {report['test_summary']['end_time']}
- Duration: {report['test_summary']['duration']:.2f} seconds
- Total Requests: {report['test_summary']['total_requests']}
- Successful Requests: {report['test_summary']['successful_requests']}
- Failed Requests: {report['test_summary']['failed_requests']}
- Success Rate: {report['test_summary']['success_rate']:.2%}
- Throughput: {report['test_summary']['throughput_requests_per_second']:.2f} requests/second

Performance Metrics:
- Average Response Time: {report['performance_metrics']['avg_response_time']:.2f}s
- Min Response Time: {report['performance_metrics']['min_response_time']:.2f}s
- Max Response Time: {report['performance_metrics']['max_response_time']:.2f}s
- Median Response Time: {report['performance_metrics']['median_response_time']:.2f}s
- 90th Percentile: {report['performance_metrics']['p90_response_time']:.2f}s
- 95th Percentile: {report['performance_metrics']['p95_response_time']:.2f}s
- 99th Percentile: {report['performance_metrics']['p99_response_time']:.2f}s

Resource Usage:
"""
        
        # 添加资源使用信息
        resource_usage = report.get('resource_usage', {})
        if resource_usage:
            readable += f"- Average CPU Usage: {resource_usage.get('avg_cpu_percent', 0):.1f}%\n"
            readable += f"- Average Memory Usage: {resource_usage.get('avg_memory_percent', 0):.1f}%\n"
            
            gpu_stats = resource_usage.get('gpu_stats', [])
            if gpu_stats:
                readable += "- GPU Usage:\n"
                for gpu in gpu_stats:
                    readable += f"  - GPU {gpu.get('id')}: {gpu.get('avg_memory_usage', 0):.1f}% memory, {gpu.get('avg_load', 0):.1f}% load\n"
        
        # 添加错误分析
        error_analysis = report.get('error_analysis', {})
        if error_analysis:
            readable += "\nError Analysis:\n"
            for error, count in error_analysis.items():
                readable += f"- {error}: {count} occurrences\n"
        
        return readable

class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self):
        self.is_running = False
        self.cpu_history = []
        self.memory_history = []
        self.gpu_history = []
        self.monitoring_interval = 1.0
        
    def start_monitoring(self):
        """开始监控"""
        self.is_running = True
        self.cpu_history = []
        self.memory_history = []
        self.gpu_history = []
        
        while self.is_running:
            try:
                # CPU使用率
                cpu_percent = psutil.cpu_percent(interval=None)
                self.cpu_history.append(cpu_percent)
                
                # 内存使用率
                memory = psutil.virtual_memory()
                self.memory_history.append(memory.percent)
                
                # GPU使用率
                try:
                    gpus = GPUtil.getGPUs()
                    for gpu in gpus:
                        self.gpu_history.append({
                            'id': gpu.id,
                            'memory_usage': gpu.memoryUtil * 100,
                            'load': gpu.load * 100,
                            'temperature': gpu.temperature
                        })
                except Exception as e:
                    logger.warning(f"GPU monitoring failed: {str(e)}")
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {str(e)}")
                time.sleep(self.monitoring_interval)
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_running = False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {}
        
        if self.cpu_history:
            stats['avg_cpu_percent'] = statistics.mean(self.cpu_history)
            stats['max_cpu_percent'] = max(self.cpu_history)
            stats['min_cpu_percent'] = min(self.cpu_history)
        
        if self.memory_history:
            stats['avg_memory_percent'] = statistics.mean(self.memory_history)
            stats['max_memory_percent'] = max(self.memory_history)
            stats['min_memory_percent'] = min(self.memory_history)
        
        if self.gpu_history:
            gpu_stats = {}
            for gpu_data in self.gpu_history:
                gpu_id = gpu_data['id']
                if gpu_id not in gpu_stats:
                    gpu_stats[gpu_id] = {
                        'memory_usage': [],
                        'load': [],
                        'temperature': []
                    }
                
                gpu_stats[gpu_id]['memory_usage'].append(gpu_data['memory_usage'])
                gpu_stats[gpu_id]['load'].append(gpu_data['load'])
                gpu_stats[gpu_id]['temperature'].append(gpu_data['temperature'])
            
            # 计算平均值
            stats['gpu_stats'] = []
            for gpu_id, data in gpu_stats.items():
                stats['gpu_stats'].append({
                    'id': gpu_id,
                    'avg_memory_usage': statistics.mean(data['memory_usage']),
                    'max_memory_usage': max(data['memory_usage']),
                    'avg_load': statistics.mean(data['load']),
                    'max_load': max(data['load']),
                    'avg_temperature': statistics.mean(data['temperature']),
                    'max_temperature': max(data['temperature'])
                })
        
        return stats

async def run_performance_tests():
    """运行性能测试"""
    # 创建测试配置
    config = TestConfig(
        api_url="http://localhost:5002",
        test_file="7.wav",  # 确保这个文件存在
        language="zh",
        concurrent_users=10,
        requests_per_user=5,
        delay_between_requests=0.1,
        timeout=60
    )
    
    # 创建测试器
    tester = PerformanceTester(config)
    
    # 运行测试
    logger.info("Starting performance test...")
    report = await tester.run_concurrent_test()
    
    # 保存报告
    tester.save_report(report)
    
    # 输出摘要
    print("\n" + "="*50)
    print("PERFORMANCE TEST SUMMARY")
    print("="*50)
    print(f"Success Rate: {report['test_summary']['success_rate']:.2%}")
    print(f"Throughput: {report['test_summary']['throughput_requests_per_second']:.2f} req/s")
    print(f"Avg Response Time: {report['performance_metrics']['avg_response_time']:.2f}s")
    print(f"P95 Response Time: {report['performance_metrics']['p95_response_time']:.2f}s")
    
    return report

if __name__ == "__main__":
    # 运行性能测试
    asyncio.run(run_performance_tests())