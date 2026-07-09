import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError:
    faiss = None


class VectorStore:
    """
    向量检索封装。

    优先使用 FAISS。
    如果本地没有安装 FAISS，则自动退化为 NumPy 相似度检索。
    """

    def __init__(self):
        self.index = None
        self.embeddings: np.ndarray | None = None
        self.use_faiss = faiss is not None

    def build(self, embeddings: np.ndarray) -> None:
        """
        构建向量索引。

        Args:
            embeddings: shape = [N, dim] 的向量矩阵。
        """
        if embeddings.ndim != 2:
            raise ValueError("embeddings 必须是二维数组")

        self.embeddings = embeddings.astype("float32")
        dim = self.embeddings.shape[1]

        if self.use_faiss:
            logger.info("使用 FAISS 构建向量索引")
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings)
        else:
            logger.warning("未安装 FAISS，使用 NumPy 检索作为备用方案")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        检索与 query 最相似的文本块。

        Args:
            query_embedding: shape = [1, dim] 的查询向量。
            top_k: 返回前 k 个结果。

        Returns:
            [(chunk_index, score), ...]
        """
        if self.embeddings is None:
            raise RuntimeError("向量库尚未构建或加载")

        if query_embedding.ndim != 2:
            raise ValueError("query_embedding 必须是二维数组")

        top_k = min(top_k, len(self.embeddings))

        if self.use_faiss and self.index is not None:
            scores, indices = self.index.search(query_embedding.astype("float32"), top_k)
            return [
                (int(idx), float(score))
                for idx, score in zip(indices[0], scores[0])
                if idx != -1
            ]

        scores = np.dot(self.embeddings, query_embedding[0])
        indices = np.argsort(scores)[::-1][:top_k]

        return [(int(idx), float(scores[idx])) for idx in indices]

    def save(self, index_path: Path, embeddings_path: Path) -> None:
        """
        保存向量索引。

        Args:
            index_path: FAISS 索引保存路径。
            embeddings_path: NumPy 向量保存路径。
        """
        if self.embeddings is None:
            raise RuntimeError("没有可保存的 embeddings")

        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_path, self.embeddings)

        if self.use_faiss and self.index is not None:
            faiss.write_index(self.index, str(index_path))

    def load(self, index_path: Path, embeddings_path: Path) -> None:
        """
        加载向量索引。

        Args:
            index_path: FAISS 索引路径。
            embeddings_path: NumPy 向量路径。
        """
        if not embeddings_path.exists():
            raise FileNotFoundError(f"找不到向量文件：{embeddings_path}")

        self.embeddings = np.load(embeddings_path).astype("float32")

        if self.use_faiss and index_path.exists():
            self.index = faiss.read_index(str(index_path))
        else:
            self.use_faiss = False
            logger.warning("未加载 FAISS 索引，将使用 NumPy 检索")


def save_chunks(chunks: List[Dict[str, Any]], path: Path) -> None:
    """保存文本块元数据。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def load_chunks(path: Path) -> List[Dict[str, Any]]:
    """加载文本块元数据。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到 chunks 文件：{path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)