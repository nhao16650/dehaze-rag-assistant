"""
llm_client.py

这个文件负责和大模型 API 通信。

当前版本支持：
1. 从 .env 文件读取 API 配置；
2. 调用 OpenAI-compatible Chat Completions 接口；
3. 基于检索到的论文片段生成自然语言回答；
4. 支持中文问题改写为英文检索 Query；
5. 如果 API 没配置或调用失败，自动退回检索结果摘要。

为什么使用 OpenAI-compatible 格式？
很多大模型服务商都兼容类似格式：
- api_url
- api_key
- model
- messages
这样后续切换模型时，只需要改 .env，不需要大改代码。
"""

import os
import re
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

from dehaze_rag.prompt_templates import (
    build_query_rewrite_prompt,
    build_rag_answer_prompt,
)


# 读取项目根目录下的 .env 文件
# 如果没有 .env，也不会报错，只是读不到 API 配置
load_dotenv()


def is_llm_configured() -> bool:
    """
    判断是否已经配置大模型 API。

    Returns:
        True:
            LLM_API_URL、LLM_API_KEY、LLM_MODEL 都存在。

        False:
            任意一个缺失，都认为没有配置。
    """

    api_url = os.getenv("LLM_API_URL")
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL")

    return bool(api_url and api_key and model)


def contains_chinese(text: str) -> bool:
    """
    判断文本中是否包含中文字符。

    Args:
        text:
            输入文本。

    Returns:
        True:
            包含中文。

        False:
            不包含中文。
    """

    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _extract_content_from_response(data: Dict) -> str:
    """
    从 OpenAI-compatible API 返回结果中提取回答内容。

    常见返回格式：
    {
        "choices": [
            {
                "message": {
                    "content": "回答内容"
                }
            }
        ]
    }

    Args:
        data:
            API 返回的 JSON 字典。

    Returns:
        content:
            大模型回答文本。

    Raises:
        RuntimeError:
            当返回格式不符合预期时抛出错误。
    """

    try:
        choices = data["choices"]
        first_choice = choices[0]
        message = first_choice["message"]
        content = message["content"]
        return str(content).strip()
    except Exception as exc:
        raise RuntimeError(f"无法从 API 返回结果中解析回答内容：{data}") from exc


def call_chat_completion(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 1200,
) -> str:
    """
    调用大模型 Chat Completions API。

    Args:
        messages:
            对话消息列表，例如：
            [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."}
            ]

        temperature:
            生成随机性。
            RAG 问答建议低一点，例如 0.1 ~ 0.3，
            因为科研问答更看重准确性，不需要太发散。

        max_tokens:
            最大输出长度。
            如果回答被截断，可以适当调大。

    Returns:
        大模型生成的文本。

    Raises:
        RuntimeError:
            API 未配置、网络失败、鉴权失败或返回格式异常。
    """

    api_url = os.getenv("LLM_API_URL")
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL")

    if not api_url or not api_key or not model:
        raise RuntimeError(
            "未配置大模型 API。请在项目根目录创建 .env，并填写 "
            "LLM_API_URL、LLM_API_KEY、LLM_MODEL。"
        )

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
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
            timeout=90,
        )
        response.raise_for_status()
        data = response.json()
        return _extract_content_from_response(data)

    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(
            f"大模型 API HTTP 请求失败：{exc}\n"
            f"响应内容：{getattr(exc.response, 'text', '')}"
        ) from exc

    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"大模型 API 网络请求失败：{exc}\n"
            "请检查网络、代理、API 地址和 API Key。"
        ) from exc


def rewrite_query_for_retrieval(question: str) -> str:
    """
    将中文问题改写为适合英文论文检索的英文 Query。

    只有当：
    1. 问题包含中文；
    2. 已配置大模型 API；
    才进行改写。

    如果没有配置 API，或者改写失败，就返回原问题，保证系统不会崩。

    Args:
        question:
            用户原始问题。

    Returns:
        retrieval_query:
            用于向量检索的问题。
            中文问题通常会被改写为英文。
    """

    question = question.strip()

    if not question:
        return question

    # 英文问题不需要改写
    if not contains_chinese(question):
        return question

    # 没配置 API 时无法改写，直接返回原问题
    if not is_llm_configured():
        return question

    prompt = build_query_rewrite_prompt(question)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful academic search query rewriting assistant.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    try:
        rewritten = call_chat_completion(
            messages=messages,
            temperature=0.0,
            max_tokens=200,
        )

        # 清理可能出现的引号、编号等多余内容
        rewritten = rewritten.strip().strip('"').strip("'").strip()

        # 如果模型返回异常短的内容，就保守使用原问题
        if len(rewritten) < 3:
            return question

        return rewritten

    except Exception:
        # 改写失败不能影响主流程
        return question


def generate_answer_without_api(
    question: str,
    contexts: List[Dict],
    answer_language: str = "中文",
) -> str:
    """
    没有配置大模型 API 时的备用回答。

    这个函数不会生成真正的自然语言总结，
    只返回检索结果摘要，保证系统在没有 API 时也能运行。

    Args:
        question:
            用户问题。

        contexts:
            检索到的论文片段。

        answer_language:
            回答语言。这里主要用于提示文本语言。

    Returns:
        answer:
            备用回答。
    """

    if not contexts:
        if answer_language == "English":
            return (
                "No relevant paper excerpts were retrieved.\n\n"
                "Suggestions:\n"
                "1. Try an English query.\n"
                "2. Use more specific keywords.\n"
                "3. Check whether the paper index has been built."
            )

        return (
            "没有检索到相关论文片段。\n\n"
            "建议你尝试：\n"
            "1. 使用英文关键词提问；\n"
            "2. 换一个更具体的问题；\n"
            "3. 检查论文是否已经成功构建索引。"
        )

    top_contexts = contexts[:3]

    if answer_language == "English":
        lines = [
            "The LLM API is not configured, so the system returns a retrieval summary.",
            "",
            f"Retrieved {len(contexts)} relevant paper excerpts. Top-3 results:",
            "",
        ]

        for i, item in enumerate(top_contexts, start=1):
            preview = item["text"][:300].replace("\n", " ").strip()
            lines.append(
                f"{i}. Source: {item['source_file']}, page {item['page_number']}, "
                f"score: {item['score']:.4f}\n"
                f"   Preview: {preview}..."
            )

        lines.append("")
        lines.append("You can check the full Top-K retrieved excerpts below.")
        return "\n\n".join(lines)

    lines = [
        "当前版本没有成功调用大模型 API，因此系统先返回“检索结果摘要”。",
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


def generate_answer_with_api(
    question: str,
    contexts: List[Dict],
    answer_language: str = "中文",
) -> str:
    """
    基于检索片段调用大模型生成回答。

    Args:
        question:
            用户原始问题。

        contexts:
            检索到的论文片段。

        answer_language:
            回答语言：
            - 中文
            - English

    Returns:
        answer:
            大模型生成回答。
            如果 API 未配置或调用失败，则返回备用检索摘要。
    """

    if not contexts:
        return generate_answer_without_api(
            question=question,
            contexts=contexts,
            answer_language=answer_language,
        )

    if not is_llm_configured():
        return generate_answer_without_api(
            question=question,
            contexts=contexts,
            answer_language=answer_language,
        )

    prompt = build_rag_answer_prompt(
        question=question,
        contexts=contexts,
        answer_language=answer_language,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "你是一个严谨的科研论文问答助手。"
                "你必须基于用户提供的论文片段回答问题，不能编造。"
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    try:
        return call_chat_completion(
            messages=messages,
            temperature=0.2,
            max_tokens=1200,
        )

    except Exception as exc:
        fallback_answer = generate_answer_without_api(
            question=question,
            contexts=contexts,
            answer_language=answer_language,
        )

        return (
            "大模型 API 调用失败，已返回检索结果摘要。\n\n"
            f"错误信息：{exc}\n\n"
            f"{fallback_answer}"
        )