# -*- coding: utf-8 -*-
"""
query_engine.py

RAG 问答引擎。

本版本重点升级：
1. 支持多 Query 检索；
2. 支持中文问题改写为英文检索 Query；
3. 支持检索结果去重与重排；
4. 支持论文级来源过滤；
5. 支持 Agent 场景下严格限制目标论文来源。

为什么要做论文级来源过滤？
在论文阅读 Agent 场景中，用户输入 DCP、GridFormer、DehazeFormer 等论文名或方法名时，
系统应该优先从对应论文中检索证据，而不是从整个知识库中混合检索。

否则会出现：
用户想分析 DCP，但报告中混入 GridFormer、Mulsormer 或其他论文内容。

本版本新增两个参数：
- focus_keywords:
    目标论文或方法关键词，例如 ["DCP"]、["GridFormer"]。
- strict_source_filter:
    是否严格限制来源文件。
    Agent 任务建议设为 True。
    普通问答可以设为 False。
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

from dehaze_rag.config import (
    CHUNKS_PATH,
    EMBEDDINGS_PATH,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    TOP_K,
)
from dehaze_rag.embedding_model import EmbeddingModel
from dehaze_rag.llm_client import (
    generate_answer_with_api,
    rewrite_query_for_retrieval,
)
from dehaze_rag.vector_store import VectorStore, load_chunks


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


COMMON_ENGLISH_WORDS = {
    "what",
    "is",
    "are",
    "the",
    "a",
    "an",
    "of",
    "for",
    "to",
    "in",
    "on",
    "with",
    "and",
    "or",
    "why",
    "how",
    "which",
    "paper",
    "method",
    "model",
    "image",
    "dehazing",
    "restoration",
    "used",
    "use",
    "main",
    "idea",
    "architecture",
    "limitation",
    "limitations",
    "drawback",
    "drawbacks",
}


class QueryEngine:
    """
    RAG 问答引擎。

    主要功能：
    1. 加载文本块和向量库；
    2. 对用户问题进行多 Query 检索；
    3. 对检索结果做去重、重排；
    4. 调用大模型生成回答；
    5. 支持论文级来源过滤。
    """

    def __init__(self):
        """
        初始化 QueryEngine。

        初始化时会加载：
        1. Embedding 模型；
        2. chunks.json；
        3. FAISS / NumPy 向量库。
        """

        logger.info("加载 Embedding 模型：%s", EMBEDDING_MODEL_NAME)
        self.embedding_model = EmbeddingModel(EMBEDDING_MODEL_NAME)

        self.chunks = []
        self.vector_store = VectorStore()

        self.reload_index()

    def reload_index(self) -> None:
        """
        重新加载 chunks 和向量库。

        使用场景：
        1. 第一次启动系统；
        2. 上传 PDF 并重建知识库之后。
        """

        logger.info("重新加载文本块：%s", CHUNKS_PATH)
        self.chunks = load_chunks(CHUNKS_PATH)

        logger.info("重新加载向量库")
        self.vector_store = VectorStore()
        self.vector_store.load(FAISS_INDEX_PATH, EMBEDDINGS_PATH)

        logger.info("知识库加载完成，文本块数量：%d", len(self.chunks))

    # ========================================================
    # 1. 文本规范化与来源过滤
    # ========================================================

    def normalize_for_match(self, text: str) -> str:
        """
        将文本规范化，方便做模糊匹配。

        例如：
        - "GridFormer_2.pdf" -> "gridformer2pdf"
        - "Dark Channel Prior" -> "darkchannelprior"

        Args:
            text:
                输入文本。

        Returns:
            normalized:
                只保留小写字母和数字后的文本。
        """

        text = text.lower()
        text = re.sub(r"[^a-z0-9]+", "", text)
        return text

    def expand_focus_keywords(self, keywords: Optional[List[str]]) -> List[str]:
        """
        扩展目标论文或方法关键词。

        例如：
        用户输入 DCP，扩展为：
        - DCP
        - Dark Channel Prior

        用户输入 Dark Channel Prior，扩展为：
        - Dark Channel Prior
        - DCP

        Args:
            keywords:
                原始关键词列表。

        Returns:
            expanded:
                扩展后的关键词列表。
        """

        if not keywords:
            return []

        expanded: List[str] = []

        for keyword in keywords:
            keyword = str(keyword).strip()

            if not keyword:
                continue

            if keyword not in expanded:
                expanded.append(keyword)

            lower_keyword = keyword.lower()

            if lower_keyword == "dcp" or "dark channel prior" in lower_keyword or "暗通道" in lower_keyword:
                for alias in ["DCP", "Dark Channel Prior"]:
                    if alias not in expanded:
                        expanded.append(alias)

            if lower_keyword == "gridformer":
                if "GridFormer" not in expanded:
                    expanded.append("GridFormer")

            if lower_keyword == "dehazeformer":
                if "DehazeFormer" not in expanded:
                    expanded.append("DehazeFormer")

            if lower_keyword == "mulsormer":
                if "Mulsormer" not in expanded:
                    expanded.append("Mulsormer")

            if "taylorformer" in lower_keyword:
                for alias in ["TaylorFormer", "MB-TaylorFormer"]:
                    if alias not in expanded:
                        expanded.append(alias)

        return expanded

    def source_matches_focus(
        self,
        source_file: str,
        focus_keywords: Optional[List[str]],
    ) -> bool:
        """
        判断某个来源文件是否匹配目标论文关键词。

        Args:
            source_file:
                来源文件名，例如 DCP.pdf、GridFormer.pdf。

            focus_keywords:
                目标论文或方法关键词。

        Returns:
            bool:
                True 表示匹配；
                False 表示不匹配。
        """

        expanded_keywords = self.expand_focus_keywords(focus_keywords)

        if not expanded_keywords:
            return True

        normalized_source = self.normalize_for_match(source_file)

        for keyword in expanded_keywords:
            normalized_keyword = self.normalize_for_match(keyword)

            if normalized_keyword and normalized_keyword in normalized_source:
                return True

        return False

    # ========================================================
    # 2. 从问题中提取方法名
    # ========================================================

    def extract_paper_or_method_names(self, question: str) -> List[str]:
        """
        从用户问题中提取可能的论文名或方法名。

        Args:
            question:
                用户原始问题。

        Returns:
            names:
                可能的论文名或方法名。
        """

        names: List[str] = []

        tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]*", question)

        for token in tokens:
            lower_token = token.lower()

            if lower_token in COMMON_ENGLISH_WORDS:
                continue

            is_candidate = (
                any(char.isupper() for char in token)
                or any(char.isdigit() for char in token)
                or "-" in token
                or token.endswith(("Former", "Net", "GAN", "CNN", "Transformer"))
                or token.upper() in {"DCP", "MSCNN", "FFA", "AOD"}
            )

            if is_candidate and token not in names:
                names.append(token)

        if "dark channel prior" in question.lower() or "暗通道" in question:
            if "Dark Channel Prior" not in names:
                names.append("Dark Channel Prior")
            if "DCP" not in names:
                names.append("DCP")

        return names

    # ========================================================
    # 3. 根据问题意图生成英文关键词
    # ========================================================

    def build_aspect_keywords(self, question: str) -> List[str]:
        """
        根据用户问题识别检索意图，并生成英文关键词。

        Args:
            question:
                用户原始问题。

        Returns:
            aspects:
                英文关键词列表。
        """

        aspects: List[str] = []
        lower_question = question.lower()

        if any(word in question for word in ["架构", "结构", "模块", "网络"]) or any(
            word in lower_question
            for word in ["architecture", "structure", "framework", "module", "network"]
        ):
            aspects.append("architecture network structure framework modules components")

        if any(word in question for word in ["缺点", "不足", "局限", "限制", "问题"]) or any(
            word in lower_question
            for word in ["limitation", "limitations", "drawback", "drawbacks", "disadvantage"]
        ):
            aspects.append("limitations drawbacks disadvantages failure cases weaknesses")

        if any(word in question for word in ["创新", "贡献", "提出", "改进"]) or any(
            word in lower_question
            for word in ["contribution", "innovation", "propose", "novel"]
        ):
            aspects.append("contributions innovations proposed method main idea")

        if any(word in question for word in ["数据集", "实验数据"]) or any(
            word in lower_question
            for word in ["dataset", "datasets", "benchmark"]
        ):
            aspects.append("datasets benchmark training testing evaluation")

        if any(word in question for word in ["指标", "评价", "评估", "psnr", "ssim"]) or any(
            word in lower_question
            for word in ["psnr", "ssim", "metric", "metrics", "evaluation"]
        ):
            aspects.append("PSNR SSIM metrics evaluation quantitative comparison")

        if not aspects:
            aspects.append("method architecture experiment results limitations")

        return aspects

    # ========================================================
    # 4. 构造多条检索 Query
    # ========================================================

    def build_retrieval_queries(
        self,
        question: str,
        use_query_rewrite: bool = True,
        focus_keywords: Optional[List[str]] = None,
        max_queries: int = 8,
    ) -> List[str]:
        """
        根据用户问题构造多个检索 Query。

        Args:
            question:
                用户原始问题。

            use_query_rewrite:
                是否启用大模型改写。

            focus_keywords:
                目标论文关键词。

            max_queries:
                最多返回多少条 Query。

        Returns:
            queries:
                检索 Query 列表。
        """

        queries: List[str] = []

        question = question.strip()

        if not question:
            return queries

        if use_query_rewrite:
            rewritten_query = rewrite_query_for_retrieval(question)
            if rewritten_query and rewritten_query not in queries:
                queries.append(rewritten_query)

        if question not in queries:
            queries.append(question)

        method_names = self.extract_paper_or_method_names(question)

        if focus_keywords:
            for keyword in self.expand_focus_keywords(focus_keywords):
                if keyword not in method_names:
                    method_names.append(keyword)

        aspect_keywords = self.build_aspect_keywords(question)

        if method_names:
            for name in method_names:
                for aspect in aspect_keywords:
                    query = f"{name} {aspect}"
                    if query not in queries:
                        queries.append(query)

                extra_queries = [
                    f"{name} proposed method architecture",
                    f"{name} network architecture modules",
                    f"{name} limitations drawbacks",
                    f"{name} ablation study components",
                    f"{name} datasets metrics experiments",
                ]

                for query in extra_queries:
                    if query not in queries:
                        queries.append(query)
        else:
            for aspect in aspect_keywords:
                if aspect not in queries:
                    queries.append(aspect)

        cleaned_queries = []

        for query in queries:
            query = query.strip()
            if query and query not in cleaned_queries:
                cleaned_queries.append(query)

        return cleaned_queries[:max_queries]

    # ========================================================
    # 5. 单 Query 检索
    # ========================================================

    def retrieve_single_query(
        self,
        retrieval_query: str,
        top_k: int,
    ) -> List[Dict]:
        """
        使用单条 Query 检索相关论文片段。

        Args:
            retrieval_query:
                检索 Query。

            top_k:
                返回数量。

        Returns:
            retrieved_chunks:
                检索结果。
        """

        retrieval_query = retrieval_query.strip()

        if not retrieval_query:
            return []

        if not self.chunks:
            logger.warning("当前 chunks 为空，请先构建知识库。")
            return []

        query_embedding = self.embedding_model.encode_query(retrieval_query)
        results = self.vector_store.search(query_embedding, top_k=top_k)

        retrieved_chunks: List[Dict] = []

        for chunk_index, score in results:
            if chunk_index < 0 or chunk_index >= len(self.chunks):
                logger.warning("检索结果 chunk_index 越界：%s", chunk_index)
                continue

            chunk = self.chunks[chunk_index].copy()
            chunk["score"] = float(score)
            chunk["retrieval_query"] = retrieval_query
            retrieved_chunks.append(chunk)

        return retrieved_chunks

    # ========================================================
    # 6. 结果去重与重排
    # ========================================================

    def build_text_fingerprint(self, text: str, max_len: int = 400) -> str:
        """
        为文本片段构造简单指纹，用于去重。

        Args:
            text:
                片段文本。

            max_len:
                指纹最大长度。

        Returns:
            fingerprint:
                文本指纹。
        """

        text = text.lower()
        text = re.sub(r"[^a-z0-9]+", "", text)
        return text[:max_len]

    def compute_rank_score(
        self,
        chunk: Dict,
        method_names: List[str],
        focus_keywords: Optional[List[str]] = None,
    ) -> float:
        """
        计算重排分数。

        Args:
            chunk:
                检索片段。

            method_names:
                问题中的方法名。

            focus_keywords:
                目标论文关键词。

        Returns:
            rank_score:
                重排分数。
        """

        raw_score = float(chunk.get("score", 0.0))
        source_file = chunk.get("source_file", "")
        text = chunk.get("text", "").lower()

        rank_score = raw_score

        all_names = list(method_names)

        if focus_keywords:
            all_names.extend(self.expand_focus_keywords(focus_keywords))

        for name in all_names:
            name_lower = name.lower()

            if not name_lower:
                continue

            if self.source_matches_focus(source_file, [name]):
                rank_score += 0.12

            if name_lower in text:
                rank_score += 0.03

        important_words = [
            "architecture",
            "network",
            "module",
            "proposed",
            "method",
            "limitation",
            "drawback",
            "disadvantage",
            "dataset",
            "benchmark",
            "psnr",
            "ssim",
            "ablation",
            "experiment",
        ]

        for word in important_words:
            if word in text:
                rank_score += 0.01

        if "references" in text[:200]:
            rank_score -= 0.05

        chunk["rank_score"] = rank_score

        return rank_score

    def retrieve_enhanced(
        self,
        question: str,
        top_k: int = TOP_K,
        use_query_rewrite: bool = True,
        focus_keywords: Optional[List[str]] = None,
        strict_source_filter: bool = False,
    ) -> Tuple[List[Dict], List[str]]:
        """
        增强版检索：多 Query 检索 + 来源过滤 + 去重 + 重排。

        Args:
            question:
                用户原始问题。

            top_k:
                最终返回 Top-K 数量。

            use_query_rewrite:
                是否启用中文问题改写。

            focus_keywords:
                目标论文或方法关键词。

            strict_source_filter:
                是否严格限制来源文件。
                Agent 任务建议使用 True。

        Returns:
            final_contexts:
                最终检索片段。

            retrieval_queries:
                实际使用的检索 Query 列表。
        """

        retrieval_queries = self.build_retrieval_queries(
            question=question,
            use_query_rewrite=use_query_rewrite,
            focus_keywords=focus_keywords,
        )

        method_names = self.extract_paper_or_method_names(question)

        if focus_keywords:
            for keyword in self.expand_focus_keywords(focus_keywords):
                if keyword not in method_names:
                    method_names.append(keyword)

        logger.info("增强检索 Query 列表：%s", retrieval_queries)
        logger.info("目标论文过滤关键词：%s", focus_keywords)

        per_query_top_k = max(top_k, 8)

        all_candidates: List[Dict] = []

        for query in retrieval_queries:
            candidates = self.retrieve_single_query(
                retrieval_query=query,
                top_k=per_query_top_k,
            )
            all_candidates.extend(candidates)

        # 论文级来源过滤。
        if focus_keywords:
            filtered_candidates = [
                item
                for item in all_candidates
                if self.source_matches_focus(
                    item.get("source_file", ""),
                    focus_keywords,
                )
            ]

            if filtered_candidates:
                all_candidates = filtered_candidates
            elif strict_source_filter:
                logger.warning(
                    "严格来源过滤后没有检索结果，focus_keywords=%s",
                    focus_keywords,
                )
                return [], retrieval_queries

        best_by_fingerprint: Dict[str, Dict] = {}

        for chunk in all_candidates:
            text = chunk.get("text", "")
            fingerprint = self.build_text_fingerprint(text)

            if not fingerprint:
                continue

            rank_score = self.compute_rank_score(
                chunk=chunk,
                method_names=method_names,
                focus_keywords=focus_keywords,
            )

            if fingerprint not in best_by_fingerprint:
                best_by_fingerprint[fingerprint] = chunk
            else:
                old_chunk = best_by_fingerprint[fingerprint]
                old_score = float(old_chunk.get("rank_score", 0.0))

                if rank_score > old_score:
                    best_by_fingerprint[fingerprint] = chunk

        unique_candidates = list(best_by_fingerprint.values())

        unique_candidates.sort(
            key=lambda item: float(item.get("rank_score", 0.0)),
            reverse=True,
        )

        final_contexts = unique_candidates[:top_k]

        return final_contexts, retrieval_queries

    # ========================================================
    # 7. 对外问答接口
    # ========================================================

    def ask(
        self,
        question: str,
        top_k: int = TOP_K,
        answer_language: str = "中文",
        use_query_rewrite: bool = True,
        focus_keywords: Optional[List[str]] = None,
        strict_source_filter: bool = False,
    ) -> Dict:
        """
        对外提供问答接口。

        Args:
            question:
                用户原始问题。

            top_k:
                检索片段数量。

            answer_language:
                回答语言。

            use_query_rewrite:
                是否启用中文问题改写和增强检索。

            focus_keywords:
                目标论文或方法关键词。
                Agent 任务会传入，例如 ["DCP"]。

            strict_source_filter:
                是否严格限制来源文件。
                Agent 任务建议 True。

        Returns:
            result:
                {
                    "answer": 回答,
                    "contexts": 检索片段,
                    "retrieval_query": 实际检索 Query
                }
        """

        question = question.strip()

        if not question:
            return {
                "answer": "请输入问题。",
                "contexts": [],
                "retrieval_query": "",
            }

        logger.info("用户原始问题：%s", question)

        contexts, retrieval_queries = self.retrieve_enhanced(
            question=question,
            top_k=top_k,
            use_query_rewrite=use_query_rewrite,
            focus_keywords=focus_keywords,
            strict_source_filter=strict_source_filter,
        )

        retrieval_query_text = "\n".join(
            [f"{i}. {query}" for i, query in enumerate(retrieval_queries, start=1)]
        )

        answer = generate_answer_with_api(
            question=question,
            contexts=contexts,
            answer_language=answer_language,
        )

        return {
            "answer": answer,
            "contexts": contexts,
            "retrieval_query": retrieval_query_text,
        }


def main() -> None:
    """
    命令行问答入口。

    运行：
        python -m dehaze_rag.query_engine
    """

    engine = QueryEngine()

    print("图像去雾论文 RAG 问答系统启动。输入 exit 退出。")

    while True:
        question = input("\n请输入问题：").strip()

        if question.lower() in {"exit", "quit", "q"}:
            print("已退出。")
            break

        result = engine.ask(
            question=question,
            top_k=TOP_K,
            answer_language="中文",
            use_query_rewrite=True,
        )

        print("\n实际检索 Query：")
        print(result["retrieval_query"])

        print("\n回答：")
        print(result["answer"])


if __name__ == "__main__":
    main()