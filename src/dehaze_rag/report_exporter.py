# -*- coding: utf-8 -*-
"""
report_exporter.py

这个文件负责导出论文阅读 Agent 生成的报告。

为什么要单独做导出模块？
1. app.py 只负责网页交互；
2. paper_agent.py 只负责任务拆解和调用 RAG；
3. report_exporter.py 专门负责把结果保存成文件。

这样项目结构更清晰，也更工程化。

当前支持：
1. 将 Agent 输出保存为 Markdown 文件；
2. 自动创建 reports/ 目录；
3. 自动清理文件名，避免非法字符；
4. 文件名中加入时间戳，避免覆盖旧报告。

注意：
reports/ 目录中的报告可以选择上传 GitHub，也可以不上传。
如果报告是你自己生成的示例，可以放到 docs/examples/。
如果报告包含论文原文大段内容，建议不要上传。
"""

import re
from datetime import datetime
from pathlib import Path


def sanitize_filename(name: str) -> str:
    """
    清理文件名，避免出现 Windows 不支持的特殊字符。

    Args:
        name:
            原始文件名，例如 GridFormer。

    Returns:
        safe_name:
            清理后的安全文件名。
    """

    name = name.strip()

    if not name:
        name = "paper_report"

    # Windows 文件名不能包含这些字符：
    # \\ / : * ? " < > |
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)

    # 多个空格合并成一个下划线
    name = re.sub(r"\s+", "_", name)

    return name


def build_markdown_report(
    paper_name: str,
    task_type: str,
    answer: str,
    query_info: str,
    references: str,
) -> str:
    """
    构造 Markdown 报告内容。

    Args:
        paper_name:
            论文名或方法名。

        task_type:
            Agent 任务类型。

        answer:
            Agent 生成的回答。

        query_info:
            实际检索 Query 信息。

        references:
            检索到的相关论文片段。

    Returns:
        markdown:
            Markdown 格式报告。
    """

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    markdown = f"""# {paper_name} 论文阅读报告

生成时间：{now}

任务类型：{task_type}

---

## Agent 输出结果

{answer}

---

## 检索 Query

{query_info}

---

## 相关论文片段

{references}

---

## 说明

本报告由 Dehaze RAG Assistant 自动生成。  
回答内容基于本地论文知识库检索片段和大模型生成结果，仅供文献阅读和研究辅助使用。
"""

    return markdown


def save_markdown_report(
    paper_name: str,
    task_type: str,
    answer: str,
    query_info: str,
    references: str,
    output_dir: str = "reports",
) -> str:
    """
    将 Agent 结果保存为 Markdown 文件。

    Args:
        paper_name:
            论文名或方法名。

        task_type:
            Agent 任务类型。

        answer:
            Agent 生成的回答。

        query_info:
            检索 Query 信息。

        references:
            相关论文片段。

        output_dir:
            输出目录，默认 reports/。

    Returns:
        file_path:
            保存后的 Markdown 文件路径字符串。
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    safe_paper_name = sanitize_filename(paper_name)
    safe_task_type = sanitize_filename(task_type)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    file_name = f"{safe_paper_name}_{safe_task_type}_{timestamp}.md"
    file_path = output_path / file_name

    markdown = build_markdown_report(
        paper_name=paper_name,
        task_type=task_type,
        answer=answer,
        query_info=query_info,
        references=references,
    )

    file_path.write_text(markdown, encoding="utf-8")

    return str(file_path)