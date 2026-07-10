# -*- coding: utf-8 -*-
"""
app.py

Gradio 网页界面入口。

当前版本功能：
1. 支持用户输入中文或英文问题；
2. 支持中文问题自动改写为英文检索 Query；
3. 支持调用大模型 API 生成自然语言回答；
4. 支持选择回答语言：中文 / English；
5. 支持展示实际检索 Query；
6. 支持展示 Top-K 相关论文片段、来源文件、页码和相似度；
7. 支持网页上传 PDF；
8. 支持上传 PDF 后自动重建知识库；
9. 支持 RAG + Agent 论文阅读助手。

本版本新增：
- PaperReadingAgent
- 论文名输入
- 论文阅读任务选择
- 一键生成论文阅读报告
"""

import os

from dotenv import load_dotenv


# ============================================================
# 1. 读取 .env 配置
# ============================================================

load_dotenv()


# ============================================================
# 2. 代理与 Gradio 环境设置
# ============================================================

os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
os.environ["no_proxy"] = "127.0.0.1,localhost,::1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

clear_proxy = os.getenv("CLEAR_PROXY_FOR_GRADIO", "0")

if clear_proxy == "1":
    for proxy_key in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]:
        os.environ.pop(proxy_key, None)


# ============================================================
# 3. 导入第三方库和项目模块
# ============================================================

import gradio as gr

from dehaze_rag.knowledge_base import upload_and_rebuild_knowledge_base
from dehaze_rag.llm_client import is_llm_configured
from dehaze_rag.paper_agent import PaperReadingAgent
from dehaze_rag.query_engine import QueryEngine
from dehaze_rag.report_exporter import save_markdown_report


# ============================================================
# 4. 初始化 RAG 问答引擎和论文阅读 Agent
# ============================================================

engine = QueryEngine()
paper_agent = PaperReadingAgent(engine)


# ============================================================
# 5. 工具函数：API 状态
# ============================================================

def get_api_status_text() -> str:
    """
    返回当前大模型 API 配置状态。

    Returns:
        str:
            用于网页展示的状态文本。
    """

    if is_llm_configured():
        return (
            "状态：已检测到大模型 API 配置。"
            "系统将尝试基于检索片段生成自然语言回答。"
        )

    return (
        "状态：未检测到大模型 API 配置。"
        "系统将返回检索结果摘要。"
        "如需启用大模型回答，请在项目根目录创建 .env 文件并填写 API 配置。"
    )


# ============================================================
# 6. 工具函数：构建检索片段 Markdown
# ============================================================

def build_reference_markdown(contexts) -> str:
    """
    将检索到的论文片段转换为 Markdown 文本。

    Args:
        contexts:
            QueryEngine 返回的相关论文片段列表。

    Returns:
        str:
            Markdown 格式检索结果。
    """

    if not contexts:
        return "没有检索到相关论文片段。"

    reference_blocks = []

    for i, item in enumerate(contexts, start=1):
        source_file = item.get("source_file", "unknown")
        page_number = item.get("page_number", "unknown")
        score = item.get("score", 0.0)
        text = item.get("text", "")

        snippet = text.replace("\n", " ").strip()
        snippet = snippet[:1000]

        block = (
            f"### 相关片段 {i}\n\n"
            f"- 来源文件：{source_file}\n"
            f"- 页码：第 {page_number} 页\n"
            f"- 相似度：{score:.4f}\n\n"
            f"片段内容：\n\n"
            f"{snippet}..."
        )

        reference_blocks.append(block)

    return "\n\n---\n\n".join(reference_blocks)


# ============================================================
# 7. 上传 PDF 并重建知识库
# ============================================================

def handle_upload_and_rebuild(uploaded_files):
    """
    处理网页上传 PDF 并重建知识库。

    Args:
        uploaded_files:
            Gradio 上传的 PDF 文件列表。

    Returns:
        status_text:
            展示给用户的状态信息。
    """

    result = upload_and_rebuild_knowledge_base(uploaded_files)

    if not result.get("success"):
        return "知识库更新失败：\n" + result.get("message", "未知错误")

    try:
        engine.reload_index()
    except Exception as exc:
        return (
            "知识库文件已重建，但问答引擎刷新失败。\n"
            f"错误信息：{exc}\n\n"
            "你可以尝试重启 app.py。"
        )

    message = result.get("message", "知识库更新完成。")

    return (
        "知识库更新成功。\n\n"
        f"{message}\n\n"
        "现在可以直接在下方输入问题进行问答。"
    )


# ============================================================
# 8. 普通论文问答函数
# ============================================================

def answer_question(
    question: str,
    answer_language: str,
    top_k: int,
    use_query_rewrite: bool,
):
    """
    Gradio 点击“开始问答”后调用的函数。

    Args:
        question:
            用户输入的问题。

        answer_language:
            回答语言。

        top_k:
            检索返回片段数量。

        use_query_rewrite:
            是否启用中文问题改写为英文检索 Query。

    Returns:
        answer:
            系统回答。

        retrieval_query_info:
            实际检索 Query 信息。

        references:
            检索到的相关论文片段。
    """

    question = question.strip()

    if not question:
        return (
            "请输入问题后再点击“开始问答”。",
            "",
            "",
        )

    result = engine.ask(
        question=question,
        top_k=int(top_k),
        answer_language=answer_language,
        use_query_rewrite=use_query_rewrite,
    )

    answer = result.get("answer", "")
    retrieval_query = result.get("retrieval_query", "")
    contexts = result.get("contexts", [])

    retrieval_query_info = (
        "### 用户原始问题\n\n"
        f"{question}\n\n"
        "### 实际用于向量检索的 Query\n\n"
        f"{retrieval_query}"
    )

    references = build_reference_markdown(contexts)

    return answer, retrieval_query_info, references


# ============================================================
# 9. Agent 论文阅读函数
# ============================================================

def run_paper_agent(
    paper_name: str,
    agent_task_type: str,
    agent_answer_language: str,
    agent_top_k: int,
):
    """
    运行 RAG + Agent 论文阅读任务。

    本版本新增：
    1. 生成 Agent 输出；
    2. 生成检索 Query 信息；
    3. 生成相关片段；
    4. 自动保存 Markdown 报告；
    5. 返回报告下载路径。

    Args:
        paper_name:
            论文名或方法名，例如 GridFormer。

        agent_task_type:
            Agent 任务类型。

        agent_answer_language:
            回答语言。

        agent_top_k:
            每个任务检索的片段数量。

    Returns:
        agent_answer:
            Agent 输出结果。

        agent_query_info:
            Agent 实际检索 Query。

        agent_references:
            Agent 检索到的相关论文片段。

        report_file:
            Markdown 报告文件路径，用于 Gradio 下载。
    """

    paper_name = paper_name.strip()

    if not paper_name:
        return (
            "请先输入论文名或方法名，例如 GridFormer、DCP、DehazeFormer。",
            "",
            "",
            None,
        )

    if agent_task_type == "生成完整论文阅读报告":
        result = paper_agent.run_full_report(
            paper_name=paper_name,
            answer_language=agent_answer_language,
            top_k=int(agent_top_k),
        )
    else:
        result = paper_agent.run_task(
            paper_name=paper_name,
            task_type=agent_task_type,
            answer_language=agent_answer_language,
            top_k=int(agent_top_k),
        )

    answer = result.get("answer", "")
    retrieval_query = result.get("retrieval_query", "")
    contexts = result.get("contexts", [])

    query_info = (
        "### Agent 任务\n\n"
        f"{agent_task_type}\n\n"
        "### Agent 生成的问题\n\n"
        f"{result.get('question', '')}\n\n"
        "### 实际用于向量检索的 Query\n\n"
        f"{retrieval_query}"
    )

    references = build_reference_markdown(contexts)

    # 自动保存 Markdown 报告，方便下载和后续放入 GitHub 示例。
    report_file = save_markdown_report(
        paper_name=paper_name,
        task_type=agent_task_type,
        answer=answer,
        query_info=query_info,
        references=references,
    )

    return answer, query_info, references, report_file


# ============================================================
# 10. 构建 Gradio 网页界面
# ============================================================

with gr.Blocks(
    title="图像去雾论文 RAG 智能问答系统",
) as demo:

    gr.Markdown(
        """
# 图像去雾论文 RAG 智能问答系统

本系统用于辅助阅读图像去雾与图像恢复方向论文。

当前版本支持：

- 中文或英文问题输入；
- 中文问题自动改写为英文检索 Query；
- 基于检索片段调用大模型生成回答；
- 展示来源文件、页码、相似度和相关论文片段；
- 上传 PDF 并自动重建本地知识库；
- RAG + Agent 论文阅读助手。
"""
    )

    api_status_box = gr.Textbox(
        label="系统状态",
        value=get_api_status_text(),
        lines=2,
        interactive=False,
    )

    with gr.Accordion("上传 PDF 并更新知识库", open=False):
        pdf_upload = gr.File(
            label="上传论文 PDF，可一次选择多个文件",
            file_types=[".pdf"],
            file_count="multiple",
        )

        rebuild_button = gr.Button("保存 PDF 并重建知识库")

        rebuild_status = gr.Textbox(
            label="知识库更新状态",
            lines=6,
            interactive=False,
        )

        rebuild_button.click(
            fn=handle_upload_and_rebuild,
            inputs=[pdf_upload],
            outputs=[rebuild_status],
        )

    gr.Markdown("## RAG + Agent 论文阅读助手")

    with gr.Accordion("论文阅读 Agent", open=True):
        paper_name_input = gr.Textbox(
            label="请输入论文名或方法名",
            placeholder="例如：GridFormer、DCP、DehazeFormer",
            lines=1,
        )

        with gr.Row():
            agent_task_radio = gr.Radio(
                choices=[
                    "一键总结论文",
                    "提取网络结构",
                    "提取方法创新点",
                    "提取数据集与评价指标",
                    "提取局限性与不足",
                    "生成论文阅读笔记",
                    "生成汇报PPT大纲",
                    "生成完整论文阅读报告",
                ],
                value="一键总结论文",
                label="选择论文阅读任务",
            )

            agent_language_radio = gr.Radio(
                choices=["中文", "English"],
                value="中文",
                label="回答语言",
            )

        agent_top_k_slider = gr.Slider(
            minimum=1,
            maximum=10,
            value=5,
            step=1,
            label="每个任务检索片段数量 Top-K",
        )

        agent_button = gr.Button("运行论文阅读 Agent")

        agent_answer_output = gr.Markdown(
            label="Agent 输出结果",
        )

        agent_query_output = gr.Markdown(
            label="Agent 检索 Query",
        )

        agent_reference_output = gr.Markdown(
            label="Agent 检索到的相关论文片段",
        )

        agent_report_file = gr.File(
            label="下载 Agent 生成的 Markdown 报告",
        )

        agent_button.click(
            fn=run_paper_agent,
            inputs=[
                paper_name_input,
                agent_task_radio,
                agent_language_radio,
                agent_top_k_slider,
            ],
            outputs=[
                agent_answer_output,
                agent_query_output,
                agent_reference_output,
                agent_report_file,
            ],
        )

    gr.Markdown("## 普通论文问答")

    question_input = gr.Textbox(
        label="请输入问题",
        placeholder="例如：暗通道先验的核心思想是什么？",
        lines=3,
    )

    with gr.Row():
        answer_language_radio = gr.Radio(
            choices=["中文", "English"],
            value="中文",
            label="回答语言",
        )

        top_k_slider = gr.Slider(
            minimum=1,
            maximum=10,
            value=5,
            step=1,
            label="检索片段数量 Top-K",
        )

    use_query_rewrite_checkbox = gr.Checkbox(
        value=True,
        label="启用中文问题改写为英文检索 Query",
    )

    submit_button = gr.Button("开始问答")

    answer_output = gr.Textbox(
        label="系统回答",
        lines=12,
    )

    retrieval_query_output = gr.Markdown(
        label="实际检索 Query",
    )

    reference_output = gr.Markdown(
        label="检索到的相关论文片段",
    )

    submit_button.click(
        fn=answer_question,
        inputs=[
            question_input,
            answer_language_radio,
            top_k_slider,
            use_query_rewrite_checkbox,
        ],
        outputs=[
            answer_output,
            retrieval_query_output,
            reference_output,
        ],
    )


# ============================================================
# 11. 启动网页服务
# ============================================================

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        inbrowser=True,
        show_error=True,
    )