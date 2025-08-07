"""
多实例模型管理器
支持多个SenseVoice模型实例并发推理，提升吞吐量
"""

import asyncio
import threading
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import torch
from funasr import AutoModel
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import uuid

# 配置日志 - 输出到控制台，DEBUG级别
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # 输出到控制台
    ]
)
logger = logging.getLogger(__name__)

class InstanceStatus(Enum):
    """模型实例状态"""
    IDLE = "idle"           # 空闲
    BUSY = "busy"           # 忙碌
    LOADING = "loading"     # 加载中
    ERROR = "error"         # 错误

@dataclass
class QueueTask:
    """队列任务信息"""
    task_id: str
    audio_data: str
    language: str
    kwargs: dict
    future: asyncio.Future
    create_time: float = time.time()

@dataclass
class ModelInstance:
    """模型实例信息"""
    instance_id: int
    model: Optional[AutoModel] = None
    device: str = "cuda:0"
    status: InstanceStatus = InstanceStatus.LOADING
    last_used: float = 0
    request_count: int = 0
    error_count: int = 0
    load_time: float = 0
    
    def update_last_used(self):
        """更新最后使用时间"""
        self.last_used = time.time()
        
    def mark_busy(self):
        """标记为忙碌"""
        self.status = InstanceStatus.BUSY
        self.request_count += 1
        self.update_last_used()
        
    def mark_idle(self):
        """标记为空闲"""
        self.status = InstanceStatus.IDLE
        self.update_last_used()

class ModelPool:
    """模型池管理器"""
    
    def __init__(self, 
                 num_instances: int = 5,
                 model_name: str = "iic/SenseVoiceSmall",
                 devices: Optional[List[str]] = None,
                 load_timeout: int = 300):
        """
        初始化模型池
        
        Args:
            num_instances: 实例数量
            model_name: 模型名称
            devices: 设备列表，如果为None则自动分配
            load_timeout: 加载超时时间（秒）
        """
        self.num_instances = num_instances
        self.model_name = model_name
        self.load_timeout = load_timeout
        
        # 设备分配策略
        if devices is None:
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                devices = []
                for i in range(num_instances):
                    # 轮询分配GPU
                    gpu_id = i % num_gpus
                    devices.append(f"cuda:{gpu_id}")
            else:
                devices = ["cpu"] * num_instances
        
        self.devices = devices
        
        # 创建模型实例
        self.instances: List[ModelInstance] = []
        for i in range(num_instances):
            instance = ModelInstance(
                instance_id=i,
                device=devices[i]
            )
            self.instances.append(instance)
        
        # 线程安全的锁
        self._lock = threading.Lock()
        
        # 任务队列
        self.queue_capacity = 5000
        self.task_queue = deque()
        self.queue_lock = threading.Lock()
        
        # 队列处理状态
        self.is_processing_queue = False
        self.queue_processor_task = None
        
        # 负载均衡统计
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        logger.info(f"Model pool initialized with {num_instances} instances")
        logger.info(f"Devices: {devices}")
    
    async def load_all_models(self) -> bool:
        """异步加载所有模型实例"""
        logger.info("Starting to load all model instances...")
        
        async def load_single_instance(instance: ModelInstance) -> bool:
            """加载单个模型实例"""
            try:
                logger.info(f"Loading model instance {instance.instance_id} on {instance.device}")
                start_time = time.time()
                
                # 在单独的线程中加载模型
                loop = asyncio.get_event_loop()
                model = await loop.run_in_executor(
                    None, 
                    self._load_model_sync,
                    instance.device
                )
                
                instance.model = model
                instance.status = InstanceStatus.IDLE
                instance.load_time = time.time() - start_time
                instance.update_last_used()
                
                logger.info(f"Instance {instance.instance_id} loaded successfully in {instance.load_time:.2f}s")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load instance {instance.instance_id}: {str(e)}")
                instance.status = InstanceStatus.ERROR
                instance.error_count += 1
                return False
        
        # 并发加载所有实例
        tasks = [load_single_instance(instance) for instance in self.instances]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_loads = sum(1 for result in results if result is True)
        logger.info(f"Model loading completed: {successful_loads}/{self.num_instances} instances loaded successfully")
        
        return successful_loads > 0
    
    def _load_model_sync(self, device: str) -> AutoModel:
        """同步加载模型（在单独线程中执行）"""
        return AutoModel(
            model=self.model_name,
            punc_model="ct-punc",
            disable_update=True,
            device=device
        )
    
    def get_idle_instance(self) -> Optional[ModelInstance]:
        """获取空闲实例（使用轮询负载均衡）"""
        with self._lock:
            idle_instances = [inst for inst in self.instances if inst.status == InstanceStatus.IDLE]
            
            if not idle_instances:
                return None
            
            # 简单轮询：选择最近使用时间最早的实例
            selected_instance = min(idle_instances, key=lambda x: x.last_used)
            selected_instance.mark_busy()
            
            self.total_requests += 1
            return selected_instance
    
    def release_instance(self, instance_id: int):
        """释放实例"""
        with self._lock:
            for instance in self.instances:
                if instance.instance_id == instance_id:
                    instance.mark_idle()
                    self.successful_requests += 1
                    logger.debug(f"Instance {instance_id} released")
                    break
    
    def handle_instance_error(self, instance_id: int, error: Exception):
        """处理实例错误"""
        with self._lock:
            for instance in self.instances:
                if instance.instance_id == instance_id:
                    instance.status = InstanceStatus.ERROR
                    instance.error_count += 1
                    self.failed_requests += 1
                    logger.error(f"Instance {instance_id} error: {str(error)}")
                    break
    
    def get_instance_stats(self) -> Dict:
        """获取实例统计信息"""
        with self._lock:
            stats = {
                "total_instances": self.num_instances,
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": self.successful_requests / max(1, self.total_requests),
                "instances": []
            }
            
            for instance in self.instances:
                instance_stats = {
                    "instance_id": instance.instance_id,
                    "device": instance.device,
                    "status": instance.status.value,
                    "request_count": instance.request_count,
                    "error_count": instance.error_count,
                    "load_time": instance.load_time,
                    "last_used": instance.last_used
                }
                stats["instances"].append(instance_stats)
            
            return stats
    
    def get_pool_status(self) -> Dict:
        """获取池状态"""
        with self._lock:
            status_counts = {}
            for status in InstanceStatus:
                count = sum(1 for inst in self.instances if inst.status == status)
                status_counts[status.value] = count
            
            return {
                "pool_size": self.num_instances,
                "status_distribution": status_counts,
                "available_instances": status_counts.get("idle", 0)
            }
    
    async def health_check(self) -> Dict:
        """健康检查"""
        async def check_instance(instance: ModelInstance) -> Dict:
            """检查单个实例健康状态"""
            try:
                if instance.model is None:
                    return {"instance_id": instance.instance_id, "healthy": False, "reason": "Model not loaded"}
                
                # 简单的健康检查：尝试进行一次小规模推理
                # 这里可以添加更复杂的健康检查逻辑
                return {
                    "instance_id": instance.instance_id,
                    "healthy": True,
                    "status": instance.status.value,
                    "device": instance.device
                }
            except Exception as e:
                return {
                    "instance_id": instance.instance_id,
                    "healthy": False,
                    "reason": str(e)
                }
        
        tasks = [check_instance(instance) for instance in self.instances]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        healthy_count = sum(1 for result in results if isinstance(result, dict) and result.get("healthy", False))
        
        return {
            "total_instances": self.num_instances,
            "healthy_instances": healthy_count,
            "unhealthy_instances": self.num_instances - healthy_count,
            "health_details": results
        }
    
    def shutdown(self):
        """关闭模型池，释放资源"""
        logger.info("Shutting down model pool...")
        with self._lock:
            for instance in self.instances:
                if instance.model is not None:
                    # 释放模型资源
                    del instance.model
                    instance.model = None
                    instance.status = InstanceStatus.ERROR
            logger.info("Model pool shutdown complete")
    
    # ============ 队列管理方法 ============
    
    async def enqueue_task(self, audio_data: str, language: str, **kwargs) -> asyncio.Future:
        """
        将任务加入队列
        
        Args:
            audio_data: 音频文件路径
            language: 语言代码
            **kwargs: 其他参数
            
        Returns:
            asyncio.Future: 用于获取结果的Future对象
        """
        # 创建Future对象
        future = asyncio.Future()
        
        # 创建队列任务
        task = QueueTask(
            task_id=str(uuid.uuid4()),
            audio_data=audio_data,
            language=language,
            kwargs=kwargs,
            future=future
        )
        
        # 加入队列
        with self.queue_lock:
            if len(self.task_queue) >= self.queue_capacity:
                error_msg = f"Queue full: {len(self.task_queue)}/{self.queue_capacity}"
                logger.error(error_msg)
                future.set_exception(RuntimeError(error_msg))
                return future
            
            self.task_queue.append(task)
            logger.info(f"Task {task.task_id} enqueued. Queue size: {len(self.task_queue)}")
        
        # 如果队列处理器没有运行，启动它
        if not self.is_processing_queue:
            self.queue_processor_task = asyncio.create_task(self._process_queue())
        
        return future
    
    async def _process_queue(self):
        """队列处理循环"""
        self.is_processing_queue = True
        logger.info("Queue processor started")
        
        try:
            while True:
                task = None
                
                # 从队列中获取任务
                with self.queue_lock:
                    if self.task_queue:
                        task = self.task_queue.popleft()
                        logger.info(f"Dequeued task {task.task_id}. Remaining: {len(self.task_queue)}")
                
                if task is None:
                    # 队列为空，等待新任务
                    await asyncio.sleep(0.1)
                    continue
                
                try:
                    # 处理任务
                    logger.info(f"Processing task {task.task_id}")
                    result = await self._process_queued_task(task)
                    task.future.set_result(result)
                    logger.info(f"Task {task.task_id} completed successfully")
                    
                except Exception as e:
                    logger.error(f"Task {task.task_id} failed: {str(e)}")
                    task.future.set_exception(e)
                    
        except asyncio.CancelledError:
            logger.info("Queue processor cancelled")
        except Exception as e:
            logger.error(f"Queue processor error: {str(e)}")
        finally:
            self.is_processing_queue = False
            logger.info("Queue processor stopped")
    
    async def _process_queued_task(self, task: QueueTask) -> str:
        """
        处理队列中的任务
        
        Args:
            task: 队列任务
            
        Returns:
            str: 处理结果
        """
        # 阻塞等待逻辑 - 不限制重试次数，只要队列没满就一直等待
        logger.info(f"Starting blocking wait for task {task.task_id}")
        
        while True:
            try:
                # 获取空闲实例
                instance = self.get_idle_instance()
                if instance is None:
                    # 没有空闲实例，等待一段时间后重试
                    logger.debug(f"No idle instance available for task {task.task_id}, waiting...")
                    await asyncio.sleep(0.5)  # 固定等待间隔
                    continue
                
                logger.info(f"Got instance {instance.instance_id} for task {task.task_id}")
                
                try:
                    # 处理请求
                    result = await self._process_with_instance_queued(instance, task)
                    logger.info(f"Task {task.task_id} completed successfully on instance {instance.instance_id}")
                    
                    # 释放实例
                    self.release_instance(instance.instance_id)
                    
                    return result
                    
                except Exception as e:
                    # 处理实例错误
                    self.handle_instance_error(instance.instance_id, e)
                    logger.error(f"Task {task.task_id} failed on instance {instance.instance_id}: {str(e)}")
                    # 实例处理失败，继续等待其他实例
                    await asyncio.sleep(1.0)  # 失败后等待稍长时间
                    
            except Exception as e:
                logger.error(f"Task {task.task_id} encountered error: {str(e)}")
                # 其他错误，继续等待
                await asyncio.sleep(1.0)
    
    async def _process_with_instance_queued(self, instance: ModelInstance, task: QueueTask) -> str:
        """
        使用特定实例处理队列任务
        
        Args:
            instance: 模型实例
            task: 队列任务
            
        Returns:
            str: 处理结果
        """
        # 在单独的线程中执行推理
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._inference_sync,
            instance.model,
            task.audio_data,
            task.language,
            task.kwargs
        )
        
        return result
    
    def get_queue_status(self) -> Dict:
        """获取队列状态"""
        with self.queue_lock:
            return {
                "queue_capacity": self.queue_capacity,
                "queue_size": len(self.task_queue),
                "queue_utilization": len(self.task_queue) / self.queue_capacity,
                "is_processing": self.is_processing_queue
            }
    
    async def stop_queue_processor(self):
        """停止队列处理器"""
        if self.queue_processor_task:
            self.queue_processor_task.cancel()
            try:
                await self.queue_processor_task
            except asyncio.CancelledError:
                pass
            self.queue_processor_task = None
    
    def _inference_sync(self, model, audio_data, language: str, kwargs_dict) -> str:
        """同步推理（在单独线程中执行）"""
        logger.info(f"Starting inference for language: {language}, file: {audio_data}")
        
        # 设置默认参数
        inference_kwargs = {
            "language": language,
            "use_itn": kwargs_dict.get("use_itn", True),
            **kwargs_dict
        }
        
        logger.info(f"Inference kwargs: {inference_kwargs}")
        
        try:
            # 执行推理
            logger.info("Calling model.generate()")
            results = model.generate(input=audio_data, **inference_kwargs)
            logger.info(f"Model.generate() returned: {type(results)}, len: {len(results) if results else 0}")
            
            # 提取文本结果
            if results and len(results) > 0:
                result_text = results[0].get("text", "")
                logger.info(f"Inference SUCCESS: {result_text}")
                return result_text
            else:
                logger.warning("Inference returned empty results")
                return ""
        except Exception as e:
            logger.error(f"Inference FAILED: {str(e)}")
            raise

class ASRRequestHandler:
    """ASR请求处理器"""
    
    def __init__(self, model_pool: ModelPool):
        self.model_pool = model_pool
        self.max_retries = 3
        self.request_timeout = 30  # 秒
    
    async def process_request(self, audio_data, language: str = "zh", **kwargs) -> Tuple[bool, str]:
        """处理ASR请求 - 使用队列"""
        logger.info(f"Processing ASR request for language: {language}, file: {audio_data}")
        
        try:
            # 将任务加入队列
            future = await self.model_pool.enqueue_task(audio_data, language, **kwargs)
            
            # 等待结果
            result = await future
            logger.info(f"Request completed successfully")
            return True, result
            
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            return False, str(e)
    
    