# -*- coding: utf-8 -*-
"""
test_dashscope_api.py

用于测试阿里云百炼 DashScope OpenAI 兼容接口是否可以正常调用 deepseek-v4-pro。

运行方式：
python test_dashscope_api.py

运行前需要在项目根目录配置 .env：

LLM_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
LLM_API_KEY=你的APIKey
LLM_MODEL=deepseek-v4-pro
LLM_ENABLE_THINKING=true
LLM_REASONING_EFFORT=high
"""

import os
import requests
from dotenv import load_dotenv


def str_to_bool(value: str) -> bool:
    """
    将字符串转换为布尔值。

    Args:
        value:
            字符串，例如 true / false / 1 / 0。

    Returns:
        bool:
            转换后的布尔值。
    """

    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def main():
    """
    测试 DashScope deepseek-v4-pro 调用。
    """

    load_dotenv()

    api_url = os.getenv("LLM_API_URL")
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL", "deepseek-v4-pro")

    enable_thinking = str_to_bool(os.getenv("LLM_ENABLE_THINKING", "true"))
    reasoning_effort = os.getenv("LLM_REASONING_EFFORT", "high")

    if not api_url:
        raise ValueError("缺少 LLM_API_URL，请检查 .env 文件。")

    if not api_key:
        raise ValueError("缺少 LLM_API_KEY，请检查 .env 文件。")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "你是谁？请用一句话回答。",
            }
        ],
        "temperature": 0.2,
        "max_tokens": 500,

        # DashScope / deepseek-v4-pro 相关参数
        # enable_thinking 是非 OpenAI 标准参数，HTTP 调用时直接放在 body 顶层即可。
        "enable_thinking": enable_thinking,

        # reasoning_effort 是 OpenAI 标准参数，deepseek-v4-pro 支持 high / max。
        "reasoning_effort": reasoning_effort,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print("正在请求模型：", model)
    print("请求地址：", api_url)

    response = requests.post(
        api_url,
        json=payload,
        headers=headers,
        timeout=90,
    )

    print("HTTP 状态码：", response.status_code)

    if response.status_code != 200:
        print("请求失败，返回内容：")
        print(response.text)
        return

    data = response.json()

    answer = data["choices"][0]["message"]["content"]

    print("\n模型回答：")
    print(answer)


if __name__ == "__main__":
    main()