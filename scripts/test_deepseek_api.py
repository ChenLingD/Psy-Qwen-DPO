"""验证 DeepSeek API 能调通 + 看 V4-Flash 的实际响应格式"""
import os
import json
import requests

API_KEY = os.environ.get("DEEPSEEK_API_KEY")
assert API_KEY, "未找到 DEEPSEEK_API_KEY 环境变量"

# DeepSeek API 兼容 OpenAI 格式
URL = "https://api.deepseek.com/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# 简单 ping
payload = {
    "model": "deepseek-chat",  # ← 这个 model 名字需要确认，可能是 deepseek-chat / deepseek-v4-flash
    "messages": [
        {"role": "user", "content": "请回复一个字：好"}
    ],
    "temperature": 0,
    "max_tokens": 10,
}

print("[Test] sending request...")
response = requests.post(URL, headers=HEADERS, json=payload, timeout=30)
print(f"[Test] status: {response.status_code}")
print(f"[Test] response: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")