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
PORT=5004

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
