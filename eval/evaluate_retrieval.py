# -*- coding: utf-8 -*-
"""
evaluate_retrieval.py

RAG 检索效果评估脚本。

运行方式：
    python eval/evaluate_retrieval.py

这个脚本主要评估：
1. Hit@1
2. Hit@3
3. Hit@5
4. Source Purity
5. Keyword Hit Rate
6. Duplicate Rate
7. Average Similarity Score

为什么要做评估？
因为 RAG 系统不能只看页面是否能回答，
还需要评估检索到的片段是否真的来自正确论文、是否包含预期关键词、
是否存在大量重复片段。

本脚本不会调用大模型 API，
只评估检索模块，所以不会消耗大模型 token。
"""

import csv
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from dehaze_rag.query_engine import QueryEngine


# ============================================================
# 1. 路径配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVAL_QUESTIONS_PATH = PROJECT_ROOT / "eval" / "eval_questions.json"

EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results"


# ============================================================
# 2. 工具函数
# ============================================================

def normalize_text(text: str) -> str:
    """
    文本规范化，用于不区分大小写地匹配关键词。

    Args:
        text:
            输入文本。

    Returns:
        normalized:
            小写后的文本。
    """

    return str(text).lower()


def load_eval_questions(path: Path) -> List[Dict]:
    """
    加载评估问题集。

    Args:
        path:
            eval_questions.json 路径。

    Returns:
        questions:
            问题列表。
    """

    if not path.exists():
        raise FileNotFoundError(f"评估问题文件不存在：{path}")

    with path.open("r", encoding="utf-8") as f:
        questions = json.load(f)

    if not isinstance(questions, list):
        raise ValueError("eval_questions.json 顶层必须是列表。")

    return questions


def source_hit(context: Dict, expected_source_keywords: List[str]) -> bool:
    """
    判断单个片段的来源文件是否命中预期来源关键词。

    Args:
        context:
            检索片段。

        expected_source_keywords:
            预期来源关键词，例如 ["DCP"]。

    Returns:
        bool:
            True 表示命中。
    """

    if not expected_source_keywords:
        return True

    source_file = normalize_text(context.get("source_file", ""))

    for keyword in expected_source_keywords:
        keyword = normalize_text(keyword)
        if keyword and keyword in source_file:
            return True

    return False


def keyword_hit(contexts: List[Dict], expected_text_keywords: List[str]) -> bool:
    """
    判断 Top-K 片段中是否至少包含一个预期文本关键词。

    Args:
        contexts:
            检索片段列表。

        expected_text_keywords:
            预期关键词列表。

    Returns:
        bool:
            True 表示至少命中一个关键词。
    """

    if not expected_text_keywords:
        return True

    combined_text = " ".join(
        str(item.get("text", "")) for item in contexts
    )

    combined_text = normalize_text(combined_text)

    for keyword in expected_text_keywords:
        keyword = normalize_text(keyword)
        if keyword and keyword in combined_text:
            return True

    return False


def compute_hit_at_k(
    contexts: List[Dict],
    expected_source_keywords: List[str],
    k: int,
) -> bool:
    """
    计算 Hit@K。

    Hit@K 表示：
    Top-K 检索结果中，只要有一个片段来自预期论文，就算命中。

    Args:
        contexts:
            检索片段列表。

        expected_source_keywords:
            预期来源关键词。

        k:
            K 值。

    Returns:
        bool:
            True 表示命中。
    """

    top_contexts = contexts[:k]

    if not top_contexts:
        return False

    for context in top_contexts:
        if source_hit(context, expected_source_keywords):
            return True

    return False


def compute_source_purity(
    contexts: List[Dict],
    expected_source_keywords: List[str],
) -> float:
    """
    计算来源纯度 Source Purity。

    Source Purity = Top-K 中来自目标论文的片段数量 / Top-K 片段总数

    如果 expected_source_keywords 为空，说明该问题不限定来源，
    此时返回 1.0。

    Args:
        contexts:
            检索片段。

        expected_source_keywords:
            预期来源关键词。

    Returns:
        purity:
            来源纯度。
    """

    if not contexts:
        return 0.0

    if not expected_source_keywords:
        return 1.0

    hit_count = 0

    for context in contexts:
        if source_hit(context, expected_source_keywords):
            hit_count += 1

    return hit_count / len(contexts)


def build_text_fingerprint(text: str, max_len: int = 300) -> str:
    """
    构造文本指纹，用于判断重复片段。

    Args:
        text:
            片段文本。

        max_len:
            指纹长度。

    Returns:
        fingerprint:
            文本指纹。
    """

    text = normalize_text(text)

    # 只保留字母和数字，减少空格、标点差异带来的影响。
    cleaned_chars = []

    for char in text:
        if char.isalnum():
            cleaned_chars.append(char)

    fingerprint = "".join(cleaned_chars)

    return fingerprint[:max_len]


def compute_duplicate_rate(contexts: List[Dict]) -> float:
    """
    计算重复率。

    Duplicate Rate = 1 - 去重后数量 / 原始数量

    Args:
        contexts:
            检索片段。

    Returns:
        duplicate_rate:
            重复率。
    """

    if not contexts:
        return 0.0

    fingerprints = []

    for context in contexts:
        text = context.get("text", "")
        fingerprint = build_text_fingerprint(text)

        if fingerprint:
            fingerprints.append(fingerprint)

    if not fingerprints:
        return 0.0

    unique_count = len(set(fingerprints))
    total_count = len(fingerprints)

    return 1.0 - unique_count / total_count


def average_score(contexts: List[Dict]) -> float:
    """
    计算平均相似度分数。

    Args:
        contexts:
            检索片段。

    Returns:
        avg_score:
            平均 score。
    """

    if not contexts:
        return 0.0

    scores = []

    for context in contexts:
        try:
            scores.append(float(context.get("score", 0.0)))
        except Exception:
            continue

    if not scores:
        return 0.0

    return statistics.mean(scores)


# ============================================================
# 3. 单条问题评估
# ============================================================

def evaluate_one_question(
    engine: QueryEngine,
    item: Dict,
    top_k: int = 5,
) -> Dict:
    """
    评估单条问题。

    Args:
        engine:
            QueryEngine 实例。

        item:
            eval_questions.json 中的一条样本。

        top_k:
            检索返回数量。

    Returns:
        row:
            评估结果字典。
    """

    question_id = item.get("id", "")
    question = item.get("question", "")

    focus_keywords = item.get("focus_keywords", [])
    strict_source_filter = bool(item.get("strict_source_filter", False))

    expected_source_keywords = item.get("expected_source_keywords", [])
    expected_text_keywords = item.get("expected_text_keywords", [])

    contexts, retrieval_queries = engine.retrieve_enhanced(
        question=question,
        top_k=top_k,
        use_query_rewrite=True,
        focus_keywords=focus_keywords,
        strict_source_filter=strict_source_filter,
    )

    hit_at_1 = compute_hit_at_k(
        contexts=contexts,
        expected_source_keywords=expected_source_keywords,
        k=1,
    )

    hit_at_3 = compute_hit_at_k(
        contexts=contexts,
        expected_source_keywords=expected_source_keywords,
        k=3,
    )

    hit_at_5 = compute_hit_at_k(
        contexts=contexts,
        expected_source_keywords=expected_source_keywords,
        k=5,
    )

    source_purity = compute_source_purity(
        contexts=contexts,
        expected_source_keywords=expected_source_keywords,
    )

    keyword_hit_value = keyword_hit(
        contexts=contexts,
        expected_text_keywords=expected_text_keywords,
    )

    duplicate_rate = compute_duplicate_rate(contexts)

    avg_score = average_score(contexts)

    top_sources = [
        context.get("source_file", "")
        for context in contexts
    ]

    top_pages = [
        str(context.get("page_number", ""))
        for context in contexts
    ]

    row = {
        "id": question_id,
        "question": question,
        "hit_at_1": int(hit_at_1),
        "hit_at_3": int(hit_at_3),
        "hit_at_5": int(hit_at_5),
        "source_purity": round(source_purity, 4),
        "keyword_hit": int(keyword_hit_value),
        "duplicate_rate": round(duplicate_rate, 4),
        "avg_score": round(avg_score, 4),
        "num_contexts": len(contexts),
        "top_sources": " | ".join(top_sources),
        "top_pages": " | ".join(top_pages),
        "retrieval_queries": " || ".join(retrieval_queries),
    }

    return row


# ============================================================
# 4. 主评估流程
# ============================================================

def main() -> None:
    """
    主函数。

    运行：
        python eval/evaluate_retrieval.py
    """

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    questions = load_eval_questions(EVAL_QUESTIONS_PATH)

    print("正在初始化 QueryEngine，请稍等...")
    engine = QueryEngine()

    print(f"共加载 {len(questions)} 条评估问题。")

    rows = []

    for item in questions:
        print(f"正在评估：{item.get('id')} - {item.get('question')}")

        row = evaluate_one_question(
            engine=engine,
            item=item,
            top_k=5,
        )

        rows.append(row)

    # 汇总指标
    total = len(rows)

    mean_hit_at_1 = sum(row["hit_at_1"] for row in rows) / total
    mean_hit_at_3 = sum(row["hit_at_3"] for row in rows) / total
    mean_hit_at_5 = sum(row["hit_at_5"] for row in rows) / total
    mean_source_purity = sum(row["source_purity"] for row in rows) / total
    mean_keyword_hit = sum(row["keyword_hit"] for row in rows) / total
    mean_duplicate_rate = sum(row["duplicate_rate"] for row in rows) / total
    mean_avg_score = sum(row["avg_score"] for row in rows) / total

    print("\n========== 检索评估结果 ==========")
    print(f"Hit@1: {mean_hit_at_1:.4f}")
    print(f"Hit@3: {mean_hit_at_3:.4f}")
    print(f"Hit@5: {mean_hit_at_5:.4f}")
    print(f"Source Purity: {mean_source_purity:.4f}")
    print(f"Keyword Hit Rate: {mean_keyword_hit:.4f}")
    print(f"Duplicate Rate: {mean_duplicate_rate:.4f}")
    print(f"Average Similarity Score: {mean_avg_score:.4f}")

    # 保存 CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = EVAL_RESULTS_DIR / f"retrieval_eval_{timestamp}.csv"

    fieldnames = [
        "id",
        "question",
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "source_purity",
        "keyword_hit",
        "duplicate_rate",
        "avg_score",
        "num_contexts",
        "top_sources",
        "top_pages",
        "retrieval_queries",
    ]

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\n详细评估结果已保存到：{output_path}")


if __name__ == "__main__":
    main()