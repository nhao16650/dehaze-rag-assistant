# -*- coding: utf-8 -*-
"""
paper_agent.py

论文阅读 Agent 模块。

本版本重点升级：
1. Agent 任务会向 QueryEngine 传入 focus_keywords；
2. Agent 任务会开启 strict_source_filter；
3. 这样可以避免分析 DCP 时检索到 GridFormer、Mulsormer 等其他论文；
4. 如果目标论文中没有相关证据，系统会更倾向于回答“当前知识库中没有足够依据”。

当前 Agent 属于“可控任务流 Agent”，不是完全自主 Agent。
它通过固定任务模板来完成论文阅读：
- 一键总结论文
- 提取网络结构
- 提取方法创新点
- 提取数据集与评价指标
- 提取局限性与不足
- 生成论文阅读笔记
- 生成汇报PPT大纲
- 生成完整论文阅读报告
"""

from typing import Dict, List

from dehaze_rag.query_engine import QueryEngine


class PaperReadingAgent:
    """
    论文阅读 Agent。

    它复用 QueryEngine 完成 RAG 检索和大模型回答。
    本类主要负责任务拆解和论文级检索约束。
    """

    def __init__(self, query_engine: QueryEngine):
        """
        初始化论文阅读 Agent。

        Args:
            query_engine:
                已初始化的 RAG 问答引擎。
        """

        self.query_engine = query_engine

    def build_focus_keywords(self, paper_name: str) -> List[str]:
        """
        根据论文名或方法名构造论文级过滤关键词。

        Args:
            paper_name:
                用户输入的论文名或方法名。

        Returns:
            keywords:
                过滤关键词。
        """

        paper_name = paper_name.strip()

        if not paper_name:
            return []

        keywords = [paper_name]

        lower_name = paper_name.lower()

        if lower_name == "dcp" or "dark channel prior" in lower_name or "暗通道" in lower_name:
            for alias in ["DCP", "Dark Channel Prior"]:
                if alias not in keywords:
                    keywords.append(alias)

        if lower_name == "gridformer":
            if "GridFormer" not in keywords:
                keywords.append("GridFormer")

        if lower_name == "dehazeformer":
            if "DehazeFormer" not in keywords:
                keywords.append("DehazeFormer")

        if lower_name == "mulsormer":
            if "Mulsormer" not in keywords:
                keywords.append("Mulsormer")

        if "taylorformer" in lower_name:
            for alias in ["TaylorFormer", "MB-TaylorFormer"]:
                if alias not in keywords:
                    keywords.append(alias)

        return keywords

    def build_task_question(self, paper_name: str, task_type: str) -> str:
        """
        根据论文名和任务类型，生成适合 RAG 检索的问题。

        Args:
            paper_name:
                论文名或方法名。

            task_type:
                任务类型。

        Returns:
            question:
                具体问题。
        """

        paper_name = paper_name.strip()

        evidence_rule = (
            "请只基于目标论文自身的检索片段回答。"
            "如果检索片段没有足够依据，请明确说明“当前目标论文片段中没有足够依据”。"
            "不要引用其他论文内容补充。"
        )

        if task_type == "一键总结论文":
            return (
                f"请基于目标论文片段总结 {paper_name} 的核心内容，"
                f"包括研究问题、核心思想、主要方法、实验结果和结论。"
                f"{evidence_rule}"
            )

        if task_type == "提取网络结构":
            return (
                f"请说明 {paper_name} 的网络结构或方法架构，"
                f"包括主要模块、模块作用、数据流过程和关键设计。"
                f"如果 {paper_name} 不是深度网络模型，而是传统先验或传统算法，"
                f"请说明它没有神经网络结构，并转而解释其方法流程。"
                f"{evidence_rule}"
            )

        if task_type == "提取方法创新点":
            return (
                f"请提取 {paper_name} 的主要创新点和贡献，"
                f"包括相比已有方法的改进、提出的新模块、新先验或新机制。"
                f"{evidence_rule}"
            )

        if task_type == "提取数据集与评价指标":
            return (
                f"请整理 {paper_name} 使用的数据集、实验设置和评价指标。"
                f"如果目标论文片段没有明确给出数据集或 PSNR、SSIM 等指标，"
                f"请直接说明缺失，不要引用其他论文。"
                f"{evidence_rule}"
            )

        if task_type == "提取局限性与不足":
            return (
                f"请提取 {paper_name} 的局限性、不足、缺点或失败案例。"
                f"只有在目标论文片段明确提到 limitation、drawback、failure case、"
                f"complexity、overhead、invalid、cannot 或类似不足时才总结。"
                f"如果没有直接依据，请明确回答“当前目标论文片段中没有足够依据说明其缺点”。"
                f"{evidence_rule}"
            )

        if task_type == "生成论文阅读笔记":
            return (
                f"请为 {paper_name} 生成一份论文阅读笔记，"
                f"包括研究背景、核心问题、方法结构、创新点、实验结果、局限性和可借鉴之处。"
                f"{evidence_rule}"
            )

        if task_type == "生成汇报PPT大纲":
            return (
                f"请基于 {paper_name} 生成一份论文汇报 PPT 大纲，"
                f"包括封面、研究背景、方法原理、核心步骤、实验结果、局限性和总结。"
                f"{evidence_rule}"
            )

        return f"请总结 {paper_name} 的主要内容。{evidence_rule}"

    def run_task(
        self,
        paper_name: str,
        task_type: str,
        answer_language: str = "中文",
        top_k: int = 5,
    ) -> Dict:
        """
        执行单个论文阅读任务。

        Args:
            paper_name:
                论文名或方法名。

            task_type:
                任务类型。

            answer_language:
                回答语言。

            top_k:
                检索片段数量。

        Returns:
            result:
                Agent 任务结果。
        """

        paper_name = paper_name.strip()

        if not paper_name:
            return {
                "task_type": task_type,
                "question": "",
                "answer": "请先输入论文名或方法名，例如 GridFormer、DCP、DehazeFormer。",
                "retrieval_query": "",
                "contexts": [],
            }

        question = self.build_task_question(
            paper_name=paper_name,
            task_type=task_type,
        )

        focus_keywords = self.build_focus_keywords(paper_name)

        rag_result = self.query_engine.ask(
            question=question,
            top_k=top_k,
            answer_language=answer_language,
            use_query_rewrite=True,
            focus_keywords=focus_keywords,
            strict_source_filter=True,
        )

        return {
            "task_type": task_type,
            "question": question,
            "answer": rag_result.get("answer", ""),
            "retrieval_query": rag_result.get("retrieval_query", ""),
            "contexts": rag_result.get("contexts", []),
        }

    def merge_contexts(
        self,
        results: List[Dict],
        max_contexts: int = 8,
    ) -> List[Dict]:
        """
        合并多个任务的检索片段，并做简单去重。

        Args:
            results:
                多个任务结果。

            max_contexts:
                最多保留多少个片段。

        Returns:
            merged:
                合并后的片段列表。
        """

        merged: List[Dict] = []
        seen = set()

        for result in results:
            contexts = result.get("contexts", [])

            for item in contexts:
                key = (
                    item.get("source_file", ""),
                    item.get("page_number", ""),
                    item.get("text", "")[:200],
                )

                if key in seen:
                    continue

                seen.add(key)
                merged.append(item)

                if len(merged) >= max_contexts:
                    return merged

        return merged

    def run_full_report(
        self,
        paper_name: str,
        answer_language: str = "中文",
        top_k: int = 5,
    ) -> Dict:
        """
        生成完整论文阅读报告。

        Args:
            paper_name:
                论文名或方法名。

            answer_language:
                回答语言。

            top_k:
                每个子任务检索片段数量。

        Returns:
            result:
                完整报告结果。
        """

        paper_name = paper_name.strip()

        if not paper_name:
            return {
                "task_type": "生成完整论文阅读报告",
                "question": "",
                "answer": "请先输入论文名或方法名，例如 GridFormer、DCP、DehazeFormer。",
                "retrieval_query": "",
                "contexts": [],
            }

        task_list = [
            "一键总结论文",
            "提取网络结构",
            "提取方法创新点",
            "提取数据集与评价指标",
            "提取局限性与不足",
            "生成汇报PPT大纲",
        ]

        results = []

        for task in task_list:
            result = self.run_task(
                paper_name=paper_name,
                task_type=task,
                answer_language=answer_language,
                top_k=top_k,
            )
            results.append(result)

        report_parts = [
            f"# {paper_name} 论文阅读报告",
            "",
            "说明：本报告仅基于目标论文自身检索片段生成，不主动引用其他论文内容。",
            "",
        ]

        for result in results:
            report_parts.append(f"## {result['task_type']}")
            report_parts.append("")
            report_parts.append(result.get("answer", ""))
            report_parts.append("")

        retrieval_query_parts = []

        for result in results:
            retrieval_query_parts.append(f"## {result['task_type']}")
            retrieval_query_parts.append(result.get("retrieval_query", ""))

        merged_contexts = self.merge_contexts(results)

        return {
            "task_type": "生成完整论文阅读报告",
            "question": f"请生成 {paper_name} 的完整论文阅读报告。",
            "answer": "\n".join(report_parts),
            "retrieval_query": "\n\n".join(retrieval_query_parts),
            "contexts": merged_contexts,
        }