"""
llm_client.py

这个文件负责生成系统回答。

当前项目支持两种模式：

1. 未配置大模型 API：
   返回检索结果摘要，不生成完整自然语言答案。

2. 配置大模型 API：
   将检索到的论文片段作为上下文，交给大模型生成回答。

目前你的项目还没有配置大模型 API，
所以默认会走 generate_answer_without_api。
"""

import os
from typing import Dict, List

import requests


def build_prompt(question: str, contexts: List[Dict]) -> str:
    """
    根据用户问题和检索片段构造提示词。

    Args:
        question:
            用户问题。

        contexts:
            检索到的论文片段。

    Returns:
        prompt:
            给大模型使用的提示词。
    """

    context_text = "\n\n".join(
        [
            f"[来源 {i + 1}] 文件：{item['source_file']}，第 {item['page_number']} 页\n"
            f"{item['text']}"
            for i, item in enumerate(contexts)
        ]
    )

    prompt = f"""
你是一个图像去雾与图像恢复方向的论文阅读助手。

请只根据下面给出的论文片段回答用户问题。
如果论文片段中没有足够信息，请明确说明“当前知识库中没有足够依据”。

【用户问题】
{question}

【论文片段】
{context_text}

【回答要求】
1. 回答要清晰、准确、简洁。
2. 尽量按要点回答。
3. 如果涉及来源，请指出来自哪个文件或页码。
4. 不要编造论文片段中没有的信息。
"""
    return prompt.strip()


def generate_answer_without_api(question: str, contexts: List[Dict]) -> str:
    """
    没有配置大模型 API 时的备用回答。

    这版不再只展示 Top1，而是展示 Top3 摘要。
    这样可以降低 Top1 偶尔排序不理想带来的影响。

    Args:
        question:
            用户问题。

        contexts:
            检索到的论文片段列表。

    Returns:
        answer:
            系统回答文本。
    """

    if not contexts:
        return (
            "没有检索到相关论文片段。\n\n"
            "建议你尝试：\n"
            "1. 使用英文关键词提问；\n"
            "2. 换一个更具体的问题；\n"
            "3. 检查论文是否已经成功构建索引。"
        )

    # 只取前 3 个结果做摘要，避免系统回答区域过长
    top_contexts = contexts[:3]

    lines = [
        "当前版本尚未配置大模型 API，因此系统先返回“检索结果摘要”。",
        "",
        f"已检索到 {len(contexts)} 个相关论文片段，Top-3 摘要如下：",
        "",
    ]

    for i, item in enumerate(top_contexts, start=1):
        preview = item["text"][:300].replace("\n", " ").strip()

        lines.append(
            f"{i}. 来源：{item['source_file']}，第 {item['page_number']} 页，"
            f"相似度：{item['score']:.4f}\n"
            f"   片段预览：{preview}..."
        )

    lines.append("")
    lines.append("你可以在下方查看完整 Top-K 检索结果。")

    return "\n\n".join(lines)


def generate_answer_with_api(question: str, contexts: List[Dict]) -> str:
    """
    调用大模型 API 生成回答。

    如果没有配置 API，则自动使用 generate_answer_without_api。

    需要配置环境变量：
    - LLM_API_URL
    - LLM_API_KEY
    - LLM_MODEL

    Args:
        question:
            用户问题。

        contexts:
            检索到的论文片段。

    Returns:
        answer:
            大模型回答，或备用回答。
    """

    api_url = os.getenv("LLM_API_URL")
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL")

    if not api_url or not api_key or not model:
        return generate_answer_without_api(question, contexts)

    prompt = build_prompt(question, contexts)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是一个严谨的科研论文问答助手。",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.2,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            api_url,
            json=payload,
            headers=headers,
            timeout=60,
        )

        response.raise_for_status()
        data = response.json()

        return data["choices"][0]["message"]["content"]

    except Exception as exc:
        fallback_answer = generate_answer_without_api(question, contexts)

        return (
            "大模型 API 调用失败，已返回检索结果摘要。\n\n"
            f"错误信息：{exc}\n\n"
            f"{fallback_answer}"
        )