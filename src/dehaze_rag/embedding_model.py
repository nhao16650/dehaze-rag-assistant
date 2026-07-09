import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    文本向量化模型封装。

    使用 SentenceTransformers 将文本转成向量。
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        logger.info("正在加载 Embedding 模型：%s", model_name)
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        将文本列表转换为向量矩阵。

        Args:
            texts: 文本列表。

        Returns:
            shape = [文本数量, 向量维度] 的 numpy 数组。
        """
        if not texts:
            raise ValueError("texts 不能为空")

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
        将用户问题转换成向量。

        Args:
            query: 用户问题。

        Returns:
            shape = [1, 向量维度] 的向量。
        """
        query = query.strip()

        if not query:
            raise ValueError("问题不能为空")

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding.astype("float32")