# -*- coding: utf-8 -*-
"""
embedding_model.py

这个文件负责加载 SentenceTransformers Embedding 模型，
并将论文文本、用户问题转换成向量。

本版本重点解决的问题：
1. 启动系统时，sentence-transformers 会尝试访问 HuggingFace；
2. 国内网络访问 HuggingFace 经常超时；
3. 项目已经下载过模型后，应该优先从本地缓存加载；
4. 因此本文件默认开启离线加载，避免每次启动都联网检查。

当前默认模型：
sentence-transformers/all-MiniLM-L6-v2

该模型适合英文论文检索：
- 模型小；
- 加载速度快；
- 向量维度适中；
- 适合当前图像去雾英文论文检索场景。

如果后续要增强中文问题直接检索英文论文的能力，
可以考虑换成多语言 Embedding 模型。
"""

import logging
import os
from typing import List

import numpy as np


# ============================================================
# 1. HuggingFace 离线模式设置
# ============================================================

# 这两个环境变量会告诉 transformers / huggingface_hub：
# 优先使用本地缓存，不要去 HuggingFace 联网检查。
#
# 注意：
# 这适合“模型已经下载过”的情况。
# 你之前已经成功 build_index，说明模型大概率已经在本地缓存里。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# 关闭 HuggingFace 额外遥测请求，减少无关联网行为。
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


# ============================================================
# 2. 导入 SentenceTransformers
# ============================================================

# 一定要先设置上面的环境变量，再导入 sentence_transformers。
# 否则库可能已经开始读取默认联网配置。
from sentence_transformers import SentenceTransformer  # noqa: E402


logger = logging.getLogger(__name__)


def str_to_bool(value: str) -> bool:
    """
    将字符串转换成布尔值。

    Args:
        value:
            字符串，例如：
            - "true"
            - "false"
            - "1"
            - "0"
            - "yes"
            - "no"

    Returns:
        bool:
            转换后的布尔值。
    """

    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


class EmbeddingModel:
    """
    Embedding 模型封装类。

    主要功能：
    1. 加载 SentenceTransformers 模型；
    2. 将文本列表转换成向量；
    3. 将用户问题转换成查询向量；
    4. 对向量进行归一化，方便使用余弦相似度检索。

    为什么要封装成类？
    因为后续如果你要替换 Embedding 模型，
    或者接入其他向量模型，只需要改这个文件。
    """

    def __init__(self, model_name: str):
        """
        初始化 Embedding 模型。

        Args:
            model_name:
                模型名称，例如：
                sentence-transformers/all-MiniLM-L6-v2

        Raises:
            RuntimeError:
                当本地没有模型缓存，或者模型缓存不完整时抛出。
        """

        self.model_name = model_name

        # 默认只从本地加载模型，避免联网访问 HuggingFace。
        #
        # 如果你以后真的想重新联网下载模型，可以在 PowerShell 临时设置：
        # $env:EMBEDDING_LOCAL_FILES_ONLY="false"
        #
        # 但平时运行项目，建议保持 true。
        local_files_only = str_to_bool(
            os.getenv("EMBEDDING_LOCAL_FILES_ONLY", "true")
        )

        logger.info("正在加载 Embedding 模型：%s", model_name)
        logger.info("Embedding 本地离线加载模式：%s", local_files_only)

        try:
            self.model = SentenceTransformer(
                model_name,
                local_files_only=local_files_only,
            )

        except Exception as exc:
            raise RuntimeError(
                "\nEmbedding 模型加载失败。\n\n"
                "当前项目默认使用本地离线模式加载模型，避免访问 HuggingFace 超时。\n\n"
                "可能原因：\n"
                "1. 本地没有缓存该模型；\n"
                "2. 模型缓存不完整；\n"
                "3. 模型名称配置错误；\n"
                "4. sentence-transformers 版本较旧。\n\n"
                "解决方法：\n"
                "方法一：如果你之前成功运行过 build_index，通常说明模型已经下载过，"
                "请重启 PyCharm 或重新运行程序。\n\n"
                "方法二：如果本地确实没有模型缓存，可以临时允许联网下载：\n"
                "PowerShell 中运行：\n"
                '$env:EMBEDDING_LOCAL_FILES_ONLY="false"\n'
                '$env:HF_HUB_OFFLINE="0"\n'
                '$env:TRANSFORMERS_OFFLINE="0"\n'
                "python -m dehaze_rag.app\n\n"
                "方法三：如果 HuggingFace 网络不通，可以打开代理或更换网络，"
                "先让模型下载完整；下载完成后再切回离线模式。\n\n"
                f"当前模型：{model_name}\n"
                f"原始错误：{exc}"
            ) from exc

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        将文本列表转换成向量。

        Args:
            texts:
                文本列表，例如论文切分后的多个 chunk。

        Returns:
            np.ndarray:
                shape 为 [文本数量, 向量维度] 的 float32 向量矩阵。

        Raises:
            ValueError:
                当 texts 为空时抛出。
        """

        if not texts:
            raise ValueError("texts 不能为空，无法生成 Embedding。")

        # show_progress_bar=True 可以在构建索引时显示进度条。
        # normalize_embeddings=True 会对向量做 L2 归一化，
        # 这样 FAISS 内积检索可以近似等价于余弦相似度。
        embeddings = self.model.encode(
            texts,
            batch_size=16,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.astype("float32")

    def encode_query(self, query: str) -> np.ndarray:
        """
        将用户问题转换成查询向量。

        Args:
            query:
                用户问题，或者经过改写后的英文检索 Query。

        Returns:
            np.ndarray:
                shape 为 [1, 向量维度] 的 float32 查询向量。

        Raises:
            ValueError:
                当 query 为空时抛出。
        """

        query = query.strip()

        if not query:
            raise ValueError("query 不能为空，无法生成查询向量。")

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding.astype("float32")