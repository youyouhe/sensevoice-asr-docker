# Set the device with environment, default is cuda:0
# export SENSEVOICE_DEVICE=cuda:1

import os, re
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from typing_extensions import Annotated
from typing import List
from enum import Enum
import torchaudio
# from model import SenseVoiceSmall  # 延后导入，避免重复加载模型
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

TMPDIR=Path(os.path.dirname(__file__)+"/tmp").as_posix()
Path(TMPDIR).mkdir(exist_ok=True)
device="cuda:0" if torch.cuda.is_available() else "cpu"

HOST='0.0.0.0'
PORT=5001  # 使用不同端口避免冲突

# Model cache paths
CACHE_DIR = Path.home() / ".cache" / "modelscope" / "hub"
MODELS = {
    "sensevoice": "iic/SenseVoiceSmall",
    "punc": "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
    "vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
}

def check_model_cache(model_name):
    """Check if model is already cached"""
    model_path = CACHE_DIR / "models" / MODELS[model_name]
    config_file = model_path / "configuration.json"
    return config_file.exists()

def load_models():
    """Load all required models with cache checking"""
    print("Checking model caches...")
    
    # Check and load SenseVoice model
    if check_model_cache("sensevoice"):
        print(f"SenseVoice model found in cache, loading...")
    else:
        print(f"SenseVoice model not found in cache, downloading...")
    
    model = AutoModel(
        model=MODELS["sensevoice"],
        punc_model="ct-punc", 
        disable_update=True,
        device=device
    )
    
    # Check and load VAD model - 优化配置
    if check_model_cache("vad"):
        print(f"VAD model found in cache, loading...")
    else:
        print(f"VAD model not found in cache, downloading...")
    
    vm = AutoModel(
        model="fsmn-vad",
        # 优化VAD参数，获得更细粒度的分段
        max_single_segment_time=3000,    # 最大片段长度：5秒（原来20秒）
        max_end_silence_time=500,        # 结束静音时间：500ms（原来250ms）
        disable_update=True,
        device=device
    )
    
    print("All models loaded successfully!")
    return model, vm

# 全局变量，存储已加载的模型
_loaded_models = None

def get_models():
    """获取已加载的模型，如果未加载则先加载"""
    global _loaded_models
    if _loaded_models is None:
        print("Loading models...")
        _loaded_models = load_models()
    return _loaded_models

# 加载模型（在启动时）
model, vm = get_models()

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

@app.get("/", response_class=HTMLResponse)
async def root():
    return f"""
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset=utf-8>
            <title>Optimized ASR API</title>
        </head>
        <body>
            <h2>Optimized ASR API</h2>
            <p>API 地址为 http://{HOST}:{PORT}/asr</p>
            <p>优化特性：</p>
            <ul>
                <li>更细粒度的字幕分段（3-8秒）</li>
                <li>更好的VAD参数配置</li>
                <li>智能片段分割算法</li>
            </ul>
        </body>
    </html>
    """

@app.post("/asr")
async def asr(file: UploadFile, lang: str = Form(...)):
    print(f'{lang=},{file.filename=}')
    if lang not in ['zh','ja','en','ko','yue']:
        return {"code":1,"msg":f'不支持的语言代码:{lang}'}
    # 创建一个临时文件路径
    temp_file_path = f"{TMPDIR}/{file.filename}"
    ## 将上传的文件保存到临时路径
    with open(temp_file_path, "wb") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
    
    # 使用VAD进行语音活动检测
    segments = vm.generate(input=temp_file_path)
    audiodata = AudioSegment.from_file(temp_file_path)    
    
    # 优化分段策略
    optimal_segments = create_optimal_segments(segments, max_duration=6000)  # 最大6秒
    
    srts=[]
    for i, seg in enumerate(optimal_segments):
        chunk=audiodata[seg[0]:seg[1]]
        filename=f"{TMPDIR}/{seg[0]}-{seg[1]}.wav"
        chunk.export(filename)
        res = model.generate(
            input=filename,
            language=lang,  # "zh", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True
        )
        text = remove_unwanted_characters(rich_transcription_postprocess(res[0]["text"]))
        print(f'Segment {i+1}: {text}')
        srts.append(f'{len(srts)+1}\n{ms_to_time_string(ms=seg[0])} --> {ms_to_time_string(ms=seg[1])}\n{text.strip()}')
    
    return {"code":0,"msg":"ok","data":"\n\n".join(srts)}

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
        # Only load models when starting the server
        print("=== Optimized ASR Server Starting ===")
        print("Features:")
        print("- VAD max_single_segment_time: 5 seconds (was 20s)")
        print("- VAD max_end_silence_time: 500ms (was 250ms)")
        print("- Smart segment splitting for better subtitle timing")
        print(f"Server running on http://{HOST}:{PORT}")
        uvicorn.run("api_optimized:app", host=HOST,port=PORT, log_level="info")
