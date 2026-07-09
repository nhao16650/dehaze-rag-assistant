import logging
from dataclasses import asdict

from tqdm import tqdm

from dehaze_rag.config import (
    PAPERS_DIR,
    INDEX_DIR,
    FAISS_INDEX_PATH,
    EMBEDDINGS_PATH,
    CHUNKS_PATH,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from dehaze_rag.embedding_model import EmbeddingModel
from dehaze_rag.pdf_loader import load_pdfs_from_folder
from dehaze_rag.text_splitter import build_chunks_from_pages
from dehaze_rag.vector_store import VectorStore, save_chunks


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    """
    构建论文知识库：
    1. 读取 papers/ 中的 PDF
    2. 提取文本
    3. 切分文本
    4. 生成 embedding
    5. 保存向量库和文本块
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("开始读取 PDF")
    pages = load_pdfs_from_folder(PAPERS_DIR)
    logger.info("共读取到 %d 页有效文本", len(pages))

    logger.info("开始切分文本")
    chunks = build_chunks_from_pages(
        pages,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    if not chunks:
        raise RuntimeError("没有生成任何文本块，请检查 PDF 是否能正常提取文本")

    logger.info("共生成 %d 个文本块", len(chunks))

    texts = [chunk.text for chunk in chunks]

    logger.info("开始生成文本向量")
    embedding_model = EmbeddingModel(EMBEDDING_MODEL_NAME)
    embeddings = embedding_model.encode(texts)

    logger.info("开始构建向量库")
    vector_store = VectorStore()
    vector_store.build(embeddings)
    vector_store.save(FAISS_INDEX_PATH, EMBEDDINGS_PATH)

    logger.info("保存文本块元数据")
    chunk_dicts = [asdict(chunk) for chunk in chunks]
    save_chunks(chunk_dicts, CHUNKS_PATH)

    logger.info("知识库构建完成！")
    logger.info("向量文件：%s", EMBEDDINGS_PATH)
    logger.info("文本块文件：%s", CHUNKS_PATH)


if __name__ == "__main__":
    main()