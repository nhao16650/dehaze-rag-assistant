import logging
from typing import Dict, List

from dehaze_rag.config import (
    FAISS_INDEX_PATH,
    EMBEDDINGS_PATH,
    CHUNKS_PATH,
    EMBEDDING_MODEL_NAME,
    TOP_K,
)
from dehaze_rag.embedding_model import EmbeddingModel
from dehaze_rag.llm_client import generate_answer_with_api
from dehaze_rag.vector_store import VectorStore, load_chunks


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


class QueryEngine:
    """
    RAG 问答引擎。

    负责：
    1. 加载向量库
    2. 接收用户问题
    3. 检索相关论文片段
    4. 生成回答
    """

    def __init__(self):
        logger.info("加载文本块")
        self.chunks = load_chunks(CHUNKS_PATH)

        logger.info("加载 Embedding 模型")
        self.embedding_model = EmbeddingModel(EMBEDDING_MODEL_NAME)

        logger.info("加载向量库")
        self.vector_store = VectorStore()
        self.vector_store.load(FAISS_INDEX_PATH, EMBEDDINGS_PATH)

    def retrieve(self, question: str, top_k: int = TOP_K) -> List[Dict]:
        """
        检索相关论文片段。
        """
        query_embedding = self.embedding_model.encode_query(question)
        results = self.vector_store.search(query_embedding, top_k=top_k)

        retrieved_chunks: List[Dict] = []

        for chunk_index, score in results:
            chunk = self.chunks[chunk_index].copy()
            chunk["score"] = score
            retrieved_chunks.append(chunk)

        return retrieved_chunks

    def ask(self, question: str, top_k: int = TOP_K) -> Dict:
        """
        对外提供问答接口。
        """
        question = question.strip()

        if not question:
            return {
                "answer": "请输入问题。",
                "contexts": [],
            }

        contexts = self.retrieve(question, top_k=top_k)
        answer = generate_answer_with_api(question, contexts)

        return {
            "answer": answer,
            "contexts": contexts,
        }


def main() -> None:
    engine = QueryEngine()

    print("图像去雾论文 RAG 问答系统启动。输入 exit 退出。")

    while True:
        question = input("\n请输入问题：").strip()

        if question.lower() in {"exit", "quit", "q"}:
            print("已退出。")
            break

        result = engine.ask(question)

        print("\n【回答】")
        print(result["answer"])


if __name__ == "__main__":
    main()