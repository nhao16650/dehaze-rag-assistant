"""
prompt_templates.py

这个文件专门管理提示词模板。

为什么要单独放提示词？
1. 让 llm_client.py 不要堆太多字符串，代码更清晰；
2. 后续做 Agent 时，可以继续扩展：
   - 论文总结 Prompt
   - 创新点提取 Prompt
   - 数据集与指标提取 Prompt
   - 汇报提纲生成 Prompt
3. 面试时可以说：项目中对 Prompt 进行了模块化管理。
"""

from typing import Dict, List


def format_contexts(
    contexts: List[Dict],
    max_chars_per_context: int = 1200,
) -> str:
    """
    将检索到的论文片段格式化为大模型可读的上下文。

    Args:
        contexts:
            QueryEngine 检索到的论文片段列表。
            每个 item 通常包含：
            - source_file: 来源文件
            - page_number: 页码
            - score: 相似度
            - text: 片段正文

        max_chars_per_context:
            每个片段最多放多少字符。
            这样可以避免上下文过长，导致 API token 太多。

    Returns:
        格式化后的上下文字符串。
    """

    if not contexts:
        return "当前没有检索到任何论文片段。"

    blocks = []

    for i, item in enumerate(contexts, start=1):
        source_file = item.get("source_file", "unknown")
        page_number = item.get("page_number", "unknown")
        score = item.get("score", 0.0)

        text = item.get("text", "")
        text = text.replace("\n", " ").strip()

        if len(text) > max_chars_per_context:
            text = text[:max_chars_per_context] + "..."

        block = (
            f"[来源 {i}]\n"
            f"文件：{source_file}\n"
            f"页码：第 {page_number} 页\n"
            f"相似度：{score:.4f}\n"
            f"片段内容：\n{text}"
        )

        blocks.append(block)

    return "\n\n".join(blocks)


def build_rag_answer_prompt(
    question: str,
    contexts: List[Dict],
    answer_language: str = "中文",
) -> str:
    """
    构造 RAG 问答提示词。

    这个 Prompt 的核心目标：
    1. 让大模型只基于检索片段回答；
    2. 避免编造论文里没有的信息；
    3. 要求回答时带上来源文件和页码；
    4. 支持中文或英文回答。

    Args:
        question:
            用户原始问题。

        contexts:
            检索到的论文片段。

        answer_language:
            回答语言，可选：
            - 中文
            - English

    Returns:
        prompt:
            给大模型的用户提示词。
    """

    context_text = format_contexts(contexts)

    if answer_language == "English":
        language_instruction = (
            "Please answer in English. Keep the answer clear, concise, "
            "and grounded in the provided paper excerpts."
        )
    else:
        language_instruction = (
            "请使用中文回答。回答要清晰、准确、简洁，并且必须基于给定论文片段。"
        )

    prompt = f"""
你是一个严谨的图像去雾与图像恢复方向论文阅读助手。

你的任务是：根据【论文片段】回答【用户问题】。

请严格遵守以下规则：
1. 只能依据给定的论文片段回答，不要编造论文片段中没有的信息。
2. 如果论文片段中没有足够依据，请明确说明“当前知识库中没有足够依据”。
3. 回答中尽量指出依据来自哪个文件和第几页。
4. 如果多个片段表达相近，请综合总结，不要简单复制原文。
5. {language_instruction}

【用户问题】
{question}

【论文片段】
{context_text}

【请开始回答】
""".strip()

    return prompt


def build_query_rewrite_prompt(question: str) -> str:
    """
    构造“中文问题改写为英文检索 Query”的提示词。

    为什么要做 query rewrite？
    你的论文大多是英文，当前 Embedding 模型也更适合英文检索。
    如果用户用中文提问，直接检索英文论文，命中率可能较低。
    因此先让大模型把中文问题改写成适合检索英文论文的英文问题。

    Args:
        question:
            用户原始问题，可能是中文，也可能是中英混合。

    Returns:
        prompt:
            给大模型的改写提示词。
    """

    prompt = f"""
你是一个科研论文检索助手。

请将下面的用户问题改写成一个适合检索英文图像去雾论文的英文查询句。

要求：
1. 只输出英文查询句，不要解释。
2. 保留关键术语，例如 Dark Channel Prior、PSNR、SSIM、Transformer、dehazing、dataset。
3. 不要添加用户没有提到的具体论文名称。
4. 查询句要简洁，适合用于向量检索。

用户问题：
{question}

英文检索 Query：
""".strip()

    return prompt