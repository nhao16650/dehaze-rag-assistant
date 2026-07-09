"""
app.py

Gradio 网页界面入口。

这个文件负责：
1. 启动一个本地网页；
2. 接收用户输入的问题；
3. 调用 QueryEngine 做 RAG 检索；
4. 在网页中展示系统回答和相关论文片段。

注意：
有些 Windows 电脑开过 VPN / 代理 / Clash 后，
即使关闭代理，Python 程序访问 127.0.0.1 也可能被代理影响，
从而导致 Gradio 报 502 Bad Gateway。

所以本文件开头会清除代理环境变量，
并告诉程序访问 localhost / 127.0.0.1 时不要走代理。
"""

import os

# ============================================================
# 1. 清理代理环境变量
# ============================================================

# 清除可能残留的代理环境变量
# 这些变量如果存在，可能会让 Python 请求 localhost 时也走代理
for proxy_key in [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
]:
    os.environ.pop(proxy_key, None)

# 访问本机地址时，不走代理
os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
os.environ["no_proxy"] = "127.0.0.1,localhost,::1"

# 关闭 Gradio 统计请求，减少额外联网
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"


# ============================================================
# 2. 导入依赖
# ============================================================

import gradio as gr

from dehaze_rag.query_engine import QueryEngine


# ============================================================
# 3. 初始化 RAG 问答引擎
# ============================================================

# 注意：
# 这里会加载：
# 1. chunks.json 文本块；
# 2. Embedding 模型；
# 3. FAISS / NumPy 向量库。
#
# 所以第一次启动网页时会稍微慢一点。
engine = QueryEngine()


# ============================================================
# 4. 网页调用函数
# ============================================================

def answer_question(question: str, top_k: int):
    """
    Gradio 点击按钮后调用的函数。

    Args:
        question:
            用户输入的问题。

        top_k:
            检索返回的片段数量。

    Returns:
        answer:
            系统回答区域显示的内容。

        references:
            下方 Markdown 区域显示的相关论文片段。
    """

    # 去掉用户输入前后的空格
    question = question.strip()

    # 如果用户没有输入问题，直接提示
    if not question:
        return "请输入问题后再点击“开始问答”。", ""

    # 调用 RAG 问答引擎
    result = engine.ask(question, top_k=top_k)

    # 系统回答
    answer = result["answer"]

    # 检索到的相关论文片段
    contexts = result["contexts"]

    # 如果没有相关片段，返回空引用
    if not contexts:
        return answer, "没有检索到相关论文片段。"

    # 构造 Markdown 格式的参考片段
    reference_blocks = []

    for i, item in enumerate(contexts, start=1):
        # 只显示每个片段前 900 个字符，避免网页太长
        snippet = item["text"][:900].replace("\n", " ").strip()

        block = f"""
### 相关片段 {i}

- **来源文件**：{item['source_file']}
- **页码**：第 {item['page_number']} 页
- **相似度**：{item['score']:.4f}

**片段内容：**

{snippet}...
"""
        reference_blocks.append(block.strip())

    # 用分割线把多个片段分开
    references = "\n\n---\n\n".join(reference_blocks)

    return answer, references


# ============================================================
# 5. 构建 Gradio 网页界面
# ============================================================

with gr.Blocks(title="图像去雾论文 RAG 智能问答系统") as demo:
    # 页面标题和说明
    gr.Markdown(
        """
# 图像去雾论文 RAG 智能问答系统

该系统用于辅助阅读图像去雾与图像恢复方向论文。

你可以提问：

- What is the main idea of Dark Channel Prior?
- What datasets are commonly used in image dehazing?
- What are PSNR and SSIM used for?
- Why are Transformers suitable for image restoration tasks?

说明：  
当前版本如果没有配置大模型 API，会优先返回检索到的相关论文片段。
        """
    )

    # 用户问题输入框
    question_input = gr.Textbox(
        label="请输入问题",
        placeholder="例如：What is the main idea of Dark Channel Prior?",
        lines=3,
    )

    # Top-K 滑块
    top_k_slider = gr.Slider(
        minimum=1,
        maximum=10,
        value=5,
        step=1,
        label="检索片段数量 Top-K",
    )

    # 提交按钮
    submit_button = gr.Button("开始问答")

    # 系统回答区域
    answer_output = gr.Textbox(
        label="系统回答",
        lines=8,
    )

    # 相关片段展示区域
    reference_output = gr.Markdown(
        label="检索到的相关论文片段",
    )

    # 按钮点击事件
    submit_button.click(
        fn=answer_question,
        inputs=[question_input, top_k_slider],
        outputs=[answer_output, reference_output],
    )


# ============================================================
# 6. 启动网页服务
# ============================================================

if __name__ == "__main__":
    # 启动 Gradio 本地网页服务
    #
    # server_name="127.0.0.1"：
    #   只允许本机访问。
    #
    # server_port=7861：
    #   使用 7861 端口，避免 7860 被占用。
    #
    # share=False：
    #   不生成公网链接。
    #
    # inbrowser=True：
    #   启动后自动打开浏览器。
    #
    # show_error=True：
    #   如果网页内部出错，显示详细错误。
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        inbrowser=True,
        show_error=True,
    )