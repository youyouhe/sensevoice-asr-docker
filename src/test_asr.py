#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
独立的ASR API测试脚本
不加载模型，直接测试API接口
"""

import requests
import json
import sys
import os

def test_asr_api(audio_file, language="zh", server_url="http://localhost:5001"):
    """
    测试ASR API
    
    Args:
        audio_file (str): 音频文件路径
        language (str): 语言代码 (zh/en/ja/ko/yue)
        server_url (str): 服务器地址
    """
    
    # 检查文件是否存在
    if not os.path.exists(audio_file):
        print(f"❌ 错误: 音频文件 '{audio_file}' 不存在")
        return False
    
    # 构建API URL
    api_url = f"{server_url}/asr"
    
    print("=== ASR API 测试 ===")
    print(f"音频文件: {audio_file}")
    print(f"语言: {language}")
    print(f"服务器: {api_url}")
    print("-" * 50)
    
    try:
        # 准备请求
        with open(audio_file, 'rb') as f:
            files = {'file': (os.path.basename(audio_file), f, 'audio/wav')}
            data = {'lang': language}
            
            print("📤 正在发送请求...")
            
            # 发送POST请求
            response = requests.post(api_url, files=files, data=data)
            
            print(f"📡 HTTP状态码: {response.status_code}")
            
            if response.status_code == 200:
                # 解析JSON响应
                result = response.json()
                
                if result.get('code') == 0:
                    print("✅ 识别成功!")
                    print("📝 识别结果:")
                    print("-" * 50)
                    
                    # 输出SRT字幕
                    srt_content = result.get('data', '')
                    print(srt_content)
                    print("-" * 50)
                    
                    # 统计信息
                    segments = srt_content.split('\n\n')
                    print(f"📊 统计信息:")
                    print(f"   总片段数: {len(segments)}")
                    
                    # 分析片段时长分布
                    durations = []
                    for segment in segments:
                        if '-->' in segment:
                            lines = segment.split('\n')
                            if len(lines) >= 2:
                                time_line = lines[1]
                                start_time, end_time = time_line.split(' --> ')
                                durations.append(get_duration_ms(end_time) - get_duration_ms(start_time))
                    
                    if durations:
                        avg_duration = sum(durations) / len(durations)
                        min_duration = min(durations)
                        max_duration = max(durations)
                        print(f"   平均片段时长: {avg_duration/1000:.1f}秒")
                        print(f"   最短片段时长: {min_duration/1000:.1f}秒")
                        print(f"   最长片段时长: {max_duration/1000:.1f}秒")
                    
                    return True
                else:
                    print(f"❌ 识别失败: {result.get('msg', '未知错误')}")
                    return False
            else:
                print(f"❌ HTTP错误: {response.status_code}")
                print(f"错误信息: {response.text}")
                return False
                
    except requests.exceptions.ConnectionError:
        print(f"❌ 连接失败: 无法连接到服务器 {server_url}")
        print("请确保服务器正在运行")
        return False
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False

def get_duration_ms(time_str):
    """
    将SRT时间字符串转换为毫秒数
    格式: "00:00:12,030" -> 12030
    """
    try:
        time_part, ms_part = time_str.split(',')
        hours, minutes, seconds = time_part.split(':')
        
        total_ms = (int(hours) * 3600 + 
                   int(minutes) * 60 + 
                   int(seconds)) * 1000 + int(ms_part)
        return total_ms
    except:
        return 0

def main():
    """主函数"""
    
    # 解析命令行参数
    if len(sys.argv) < 2:
        print("用法: python test_asr.py <音频文件> [语言] [服务器地址]")
        print("示例: python test_asr.py 7.wav zh http://localhost:5001")
        print("")
        print("支持的语言代码:")
        print("  zh - 中文")
        print("  en - 英文") 
        print("  yue - 粤语")
        print("  ja - 日文")
        print("  ko - 韩文")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    language = sys.argv[2] if len(sys.argv) > 2 else "zh"
    server_url = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:5001"
    
    # 执行测试
    success = test_asr_api(audio_file, language, server_url)
    
    if success:
        print("\n🎉 测试完成!")
    else:
        print("\n💥 测试失败!")
        sys.exit(1)

if __name__ == "__main__":
    main()