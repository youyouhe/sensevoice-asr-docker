# Set the device with environment, default is cuda:0
# export SENSEVOICE_DEVICE=cuda:1

import os, re
import uuid
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from typing_extensions import Annotated
from typing import List
from enum import Enum
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
import asyncio
import logging

# 导入多实例管理器
from model_pool import ModelPool, ASRRequestHandler

# 配置日志 - 输出到控制台，DEBUG级别
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # 输出到控制台
    ]
)
logger = logging.getLogger(__name__)

TMPDIR=Path(os.path.dirname(__file__)+"/tmp").as_posix()
Path(TMPDIR).mkdir(exist_ok=True)

# 服务器配置
HOST='0.0.0.0'
PORT=5002  # 使用不同端口避免冲突

# 多实例配置
NUM_INSTANCES = 10  # 5个并发实例
MODEL_NAME = "iic/SenseVoiceSmall"

# 全局变量
model_pool = None
request_handler = None
vad_model = None

async def initialize_models():
    """初始化所有模型"""
    global model_pool, request_handler, vad_model
    
    logger.info("Initializing multi-instance model pool...")
    
    # 创建模型池
    model_pool = ModelPool(
        num_instances=NUM_INSTANCES,
        model_name=MODEL_NAME,
        load_timeout=600  # 10分钟超时
    )
    
    # 异步加载所有模型实例
    load_success = await model_pool.load_all_models()
    if not load_success:
        raise RuntimeError("Failed to load model instances")
    
    # 创建请求处理器
    request_handler = ASRRequestHandler(model_pool)
    
    # 加载VAD模型（单实例即可）
    logger.info("Loading VAD model...")
    vad_model = AutoModel(
        model="fsmn-vad",
        max_single_segment_time=3000,    # 最大片段长度：3秒
        max_end_silence_time=500,        # 结束静音时间：500ms
        disable_update=True,
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    
    logger.info("All models initialized successfully!")

def get_model_pool():
    """获取模型池（如果未初始化则先初始化）"""
    global model_pool
    if model_pool is None:
        # 在同步环境中需要特殊处理
        logger.warning("Model pool not initialized, this should not happen in production")
    return model_pool

def get_request_handler():
    """获取请求处理器"""
    global request_handler
    return request_handler

def get_vad_model():
    """获取VAD模型"""
    global vad_model
    return vad_model

'''
格式化毫秒或秒为符合srt格式的 2位小时:2位分:2位秒,3位毫秒 形式
print(ms_to_time_string(ms=12030))
-> 00:00:12,030
'''
def ms_to_time_string(*, ms=0, seconds=None):
    # 计算小时、分钟、秒和毫秒
    if seconds is None:
        td = timedelta(milliseconds=ms)
    else:
        td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000

    time_string = f"{hours}:{minutes}:{seconds},{milliseconds}"
    return format_time(time_string, ',')

# 将不规范的 时:分:秒,|.毫秒格式为  aa:bb:cc,ddd形式
# eg  001:01:2,4500  01:54,14 等做处理
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
    # 保留中文、日文、韩文、英文、数字和常见符号，去除其他字符
    allowed_characters = re.compile(r'[^\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af'
                                    r'a-zA-Z0-9\s.,!@#$%^&*()_+\-=\[\]{};\'"\\|<>/?，。！｛｝【】；''""《》、（）￥]+')
    return re.sub(allowed_characters, '', text)

def create_optimal_segments(segments, max_duration=8000):
    """优化分段：将过长的片段分割为更合适的小段"""
    optimal_segments = []
    
    for seg in segments[0]['value']:
        start_time = seg[0]
        end_time = seg[1]
        duration = end_time - start_time
        
        # 如果片段长度超过最大时长，进行分割
        if duration > max_duration:
            # 计算需要分割成几段
            num_segments = max(2, int(duration / max_duration) + 1)
            segment_duration = duration // num_segments
            
            for i in range(num_segments):
                seg_start = start_time + i * segment_duration
                seg_end = start_time + (i + 1) * segment_duration
                if i == num_segments - 1:  # 最后一段
                    seg_end = end_time
                optimal_segments.append([seg_start, seg_end])
        else:
            optimal_segments.append([start_time, end_time])
    
    return optimal_segments

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """启动时初始化模型"""
    logger.info("Starting up multi-instance ASR server...")
    try:
        await initialize_models()
        logger.info("Multi-instance ASR server started successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        raise e

@app.on_event("shutdown")
async def shutdown_event():
    """关闭时清理资源"""
    logger.info("Shutting down multi-instance ASR server...")
    if model_pool:
        model_pool.shutdown()
    logger.info("Server shutdown complete")

@app.get("/", response_class=HTMLResponse)
async def root():
    return f"""
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset=utf-8>
            <title>Multi-Instance ASR API</title>
        </head>
        <body>
            <h2>Multi-Instance ASR API</h2>
            <p>API 地址为 http://{HOST}:{PORT}/asr</p>
            <p>多实例特性：</p>
            <ul>
                <li>{NUM_INSTANCES}个并发模型实例</li>
                <li>智能负载均衡</li>
                <li>自动错误恢复</li>
                <li>实时健康监控</li>
                <li>更高吞吐量和并发性能</li>
            </ul>
            <p><a href="/health">健康检查</a></p>
            <p><a href="/stats">统计信息</a></p>
        </body>
    </html>
    """

@app.get("/health")
async def health():
    """健康检查端点"""
    if model_pool is None:
        return JSONResponse(status_code=503, content={"status": "not_initialized"})
    
    health_result = await model_pool.health_check()
    
    # 判断整体健康状态
    healthy_ratio = health_result["healthy_instances"] / health_result["total_instances"]
    
    if healthy_ratio >= 0.8:  # 80%以上实例健康
        status_code = 200
        overall_status = "healthy"
    elif healthy_ratio >= 0.5:  # 50%以上实例健康
        status_code = 200
        overall_status = "degraded"
    else:
        status_code = 503
        overall_status = "unhealthy"
    
    return {
        "status": overall_status,
        "health_ratio": healthy_ratio,
        **health_result
    }

@app.get("/stats")
async def stats():
    """获取统计信息"""
    if model_pool is None:
        return {"error": "Model pool not initialized"}
    
    return {
        "model_pool_stats": model_pool.get_instance_stats(),
        "pool_status": model_pool.get_pool_status(),
        "queue_status": model_pool.get_queue_status(),
        "timestamp": time.time()
    }

@app.post("/asr")
async def asr(file: UploadFile, lang: str = Form(...)):
    """ASR处理端点 - 使用多实例并行处理"""
    print(f'{lang=},{file.filename=}')
    logger.info(f"ASR request received - Language: {lang}, Filename: {file.filename}")
    
    if lang not in ['zh','ja','en','ko','yue']:
        logger.error(f"Unsupported language: {lang}")
        return {"code":1,"msg":f'不支持的语言代码:{lang}'}
    
    # 检查模型池是否已初始化
    if model_pool is None or request_handler is None:
        logger.error("Model pool not initialized")
        return {"code": 500, "msg": "Model pool not initialized"}
    
    # 创建临时文件
    temp_file_path = f"{TMPDIR}/{file.filename}"
    try:
        logger.info(f"Saving uploaded file to: {temp_file_path}")
        ## 将上传的文件保存到临时路径
        with open(temp_file_path, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
        
        logger.info("Running VAD model for voice activity detection")
        # 使用VAD进行语音活动检测
        segments = vad_model.generate(input=temp_file_path)
        logger.info(f"VAD detected {len(segments)} segments")
        
        audiodata = AudioSegment.from_file(temp_file_path)
        logger.info(f"Audio loaded - duration: {len(audiodata)}ms")
        
        # 优化分段策略
        optimal_segments = create_optimal_segments(segments, max_duration=6000)  # 最大6秒
        logger.info(f"Optimized to {len(optimal_segments)} segments for processing")
        
        # 跟踪创建的段文件，用于后续清理
        segment_files = []
        
        # 并行处理所有音频片段
        async def process_segment_async(segment_data):
            """异步处理单个音频片段"""
            i, seg = segment_data
            logger.info(f"Processing segment {i+1}/{len(optimal_segments)}: {seg[0]}-{seg[1]}ms")
            
            try:
                chunk = audiodata[seg[0]:seg[1]]
                # 使用UUID避免文件名冲突
                filename = f"{TMPDIR}/segment_{i}_{seg[0]}_{seg[1]}_{uuid.uuid4().hex[:8]}.wav"
                chunk.export(filename)
                segment_files.append(filename)  # 记录文件路径
                logger.info(f"Segment {i+1} exported to: {filename}")
                
                # 使用多实例处理器
                logger.info(f"Sending segment {i+1} to model pool")
                success, result = await request_handler.process_request(
                    audio_data=filename,
                    language=lang,
                    use_itn=True
                )
                
                if success:
                    text = remove_unwanted_characters(rich_transcription_postprocess(result))
                    logger.info(f"Segment {i+1} SUCCESS: {text}")
                    print(f'Segment {i+1}: {text}')
                    return {
                        "index": i,
                        "start_time": seg[0],
                        "end_time": seg[1],
                        "text": text.strip(),
                        "success": True
                    }
                else:
                    logger.error(f"Segment {i+1} FAILED: {result}")
                    print(f'Segment {i+1} failed: {result}')
                    return {
                        "index": i,
                        "start_time": seg[0],
                        "end_time": seg[1],
                        "text": "",
                        "success": False,
                        "error": result
                    }
                    
            except Exception as e:
                logger.error(f"Segment {i+1} ERROR: {str(e)}")
                print(f'Segment {i+1} error: {str(e)}')
                return {
                    "index": i,
                    "start_time": seg[0],
                    "end_time": seg[1],
                    "text": "",
                    "success": False,
                    "error": str(e)
                }
        
        # 并发处理所有片段
        segment_tasks = [process_segment_async((i, seg)) for i, seg in enumerate(optimal_segments)]
        segment_results = await asyncio.gather(*segment_tasks, return_exceptions=True)
        
        # 过滤和处理结果
        valid_results = []
        for result in segment_results:
            if isinstance(result, dict):
                valid_results.append(result)
        
        # 按索引排序并生成SRT
        valid_results.sort(key=lambda x: x["index"])
        srts = []
        for result in valid_results:
            if result["success"] and result["text"].strip():
                srt_entry = f'{len(srts)+1}\n{ms_to_time_string(ms=result["start_time"])} --> {ms_to_time_string(ms=result["end_time"])}\n{result["text"]}'
                srts.append(srt_entry)
        
        success_count = sum(1 for r in valid_results if r["success"])
        total_count = len(valid_results)
        
        return {
            "code": 0, 
            "msg": "ok", 
            "data": "\n\n".join(srts),
            "stats": {
                "total_segments": total_count,
                "successful_segments": success_count,
                "failed_segments": total_count - success_count,
                "success_rate": success_count / total_count if total_count > 0 else 0
            }
        }
        
    except Exception as e:
        logger.error(f"ASR processing error: {str(e)}")
        return {"code": 500, "msg": f"Processing error: {str(e)}"}
    finally:
        # 清理临时文件
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Cleaned up main temp file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up main temp file {temp_file_path}: {str(e)}")
        
        # 清理段文件
        for segment_file in segment_files:
            if os.path.exists(segment_file):
                try:
                    os.remove(segment_file)
                    logger.debug(f"Cleaned up segment file: {segment_file}")
                except Exception as e:
                    logger.warning(f"Failed to clean up segment file {segment_file}: {str(e)}")
        
        if segment_files:
            logger.info(f"Cleaned up {len(segment_files)} segment files")

@app.post("/asr_simple")
async def asr_simple(file: UploadFile, lang: str = Form(...)):
    """简单的ASR处理端点 - 不进行VAD分段，直接处理整个音频文件"""
    print(f'{lang=},{file.filename=}')
    if lang not in ['zh','ja','en','ko','yue']:
        return {"code":1,"msg":f'不支持的语言代码:{lang}'}
    
    # 检查模型池是否已初始化
    if model_pool is None or request_handler is None:
        return {"code": 500, "msg": "Model pool not initialized"}
    
    # 创建临时文件
    temp_file_path = f"{TMPDIR}/{file.filename}"
    try:
        ## 将上传的文件保存到临时路径
        with open(temp_file_path, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
        
        # 使用多实例处理器直接处理整个文件
        success, result = await request_handler.process_request(
            audio_data=temp_file_path,
            language=lang,
            use_itn=True
        )
        
        if success:
            text = remove_unwanted_characters(rich_transcription_postprocess(result))
            return {
                "code": 0,
                "msg": "ok",
                "data": text.strip()
            }
        else:
            return {
                "code": 500,
                "msg": f"ASR processing failed: {result}"
            }
            
    except Exception as e:
        logger.error(f"Simple ASR processing error: {str(e)}")
        return {"code": 500, "msg": f"Processing error: {str(e)}"}
    finally:
        # 清理临时文件
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass

def test_api_with_wav(wav_path, lang="zh"):
    """Test API with a local WAV file"""
    import requests
    
    # Check if file exists
    if not os.path.exists(wav_path):
        print(f"Error: File {wav_path} not found")
        return
    
    # Prepare the request
    url = f"http://{HOST}:{PORT}/asr"
    
    try:
        with open(wav_path, 'rb') as f:
            files = {'file': (os.path.basename(wav_path), f, 'audio/wav')}
            data = {'lang': lang}
            
            print(f"Sending request to {url}...")
            print(f"File: {wav_path}, Language: {lang}")
            
            response = requests.post(url, files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print("API Response:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print(f"Error: HTTP {response.status_code}")
                print(response.text)
                
    except Exception as e:
        print(f"Error testing API: {e}")

if __name__=='__main__':
    import uvicorn
    import sys
    
    # Check if we want to test instead of running server
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        wav_file = "7.wav"  # Default test file
        if len(sys.argv) > 2:
            wav_file = sys.argv[2]
        test_api_with_wav(wav_file)
    else:
        # Multi-instance server startup
        print("=== Multi-Instance ASR Server Starting ===")
        print(f"Features:")
        print(f"- {NUM_INSTANCES} concurrent model instances")
        print(f"- Intelligent load balancing")
        print(f"- Automatic error recovery")
        print(f"- Real-time health monitoring")
        print(f"- Higher throughput and concurrency")
        print(f"Server running on http://{HOST}:{PORT}")
        print(f"Health check: http://{HOST}:{PORT}/health")
        print(f"Statistics: http://{HOST}:{PORT}/stats")
        
        uvicorn.run("api_multi_instance:app", host=HOST, port=PORT, log_level="info")
