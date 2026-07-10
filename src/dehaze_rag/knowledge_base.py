# -*- coding: utf-8 -*-
"""
knowledge_base.py

这个文件负责知识库构建与更新。

当前版本主要用于支持网页上传 PDF 后自动更新知识库。

核心功能：
1. 保存用户上传的 PDF 到 papers/ 文件夹；
2. 读取 papers/ 文件夹中的所有 PDF；
3. 解析 PDF 文本；
4. 清洗与切分文本；
5. 使用 Embedding 模型生成向量；
6. 使用 FAISS 构建向量索引；
7. 保存 chunks.json、embeddings.npy、dehaze_faiss.index；
8. 返回构建结果，供 Gradio 页面展示。

本版本修复的问题：
1. 修复 VectorStore.save() 参数名错误；
   你的 vector_store.py 中 save 方法定义为：
   save(index_path, embeddings_path)
   所以这里必须写：
   vector_store.save(FAISS_INDEX_PATH, EMBEDDINGS_PATH)

2. 修复 TextChunk dataclass 不能直接保存为 JSON 的问题；
   chunks 需要先通过 dataclasses.asdict() 转成字典列表，
   再交给 save_chunks() 保存。
"""

import logging
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Optional

from dehaze_rag.config import (
    CHUNKS_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDINGS_PATH,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    INDEX_DIR,
    PAPERS_DIR,
)
from dehaze_rag.embedding_model import EmbeddingModel
from dehaze_rag.pdf_loader import load_pdfs_from_folder
from dehaze_rag.text_splitter import build_chunks_from_pages
from dehaze_rag.vector_store import VectorStore, save_chunks


logger = logging.getLogger(__name__)


def ensure_project_dirs() -> None:
    """
    确保项目运行所需目录存在。

    包括：
    - papers/
    - data/index/

    如果目录不存在，就自动创建。
    """

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_pdf_filename(filename: str) -> str:
    """
    清理上传 PDF 的文件名，避免出现危险路径或异常字符。

    为什么要清理文件名？
    用户上传的文件名可能包含：
    - 路径符号；
    - 特殊字符；
    - 空格；
    - 中文符号。

    为了保存稳定，这里只保留相对安全的字符。

    Args:
        filename:
            原始文件名。

    Returns:
        cleaned_name:
            清理后的 PDF 文件名。
    """

    # 只取文件名，不保留目录，防止路径穿越。
    name = Path(filename).name

    # 如果文件名没有 .pdf 后缀，就补上。
    if not name.lower().endswith(".pdf"):
        name = name + ".pdf"

    safe_chars = []

    for char in name:
        # 保留：
        # 1. 中英文和数字；
        # 2. 空格；
        # 3. 点号、横线、下划线；
        # 4. 中英文括号。
        if char.isalnum() or char in {" ", ".", "-", "_", "(", ")", "（", "）"}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")

    cleaned_name = "".join(safe_chars).strip()

    # 避免空文件名。
    if not cleaned_name or cleaned_name.lower() == ".pdf":
        cleaned_name = "uploaded_paper.pdf"

    return cleaned_name


def save_uploaded_pdfs(uploaded_files: Optional[Iterable]) -> List[Path]:
    """
    保存 Gradio 上传的 PDF 文件到 papers/ 文件夹。

    Args:
        uploaded_files:
            Gradio File 组件传入的文件对象列表。
            不同 Gradio 版本可能返回：
            - 临时文件对象；
            - 文件路径字符串；
            - 带 name / orig_name 属性的对象。

    Returns:
        saved_paths:
            成功保存到 papers/ 文件夹中的 PDF 路径列表。
    """

    ensure_project_dirs()

    saved_paths: List[Path] = []

    if not uploaded_files:
        return saved_paths

    for uploaded_file in uploaded_files:
        # Gradio 常见返回形式：
        # 1. uploaded_file.name 是临时文件路径；
        # 2. uploaded_file 本身是字符串路径。
        if hasattr(uploaded_file, "name"):
            source_path = Path(uploaded_file.name)
        else:
            source_path = Path(str(uploaded_file))

        if not source_path.exists():
            logger.warning("上传文件不存在，已跳过：%s", source_path)
            continue

        # 优先使用原始文件名；
        # 如果没有 orig_name，就退回临时文件名。
        original_name = getattr(uploaded_file, "orig_name", source_path.name)
        safe_name = sanitize_pdf_filename(original_name)

        target_path = PAPERS_DIR / safe_name

        # 如果同名文件已经存在，自动追加编号，避免覆盖原文件。
        if target_path.exists():
            stem = target_path.stem
            suffix = target_path.suffix
            counter = 1

            while True:
                candidate = PAPERS_DIR / f"{stem}_{counter}{suffix}"
                if not candidate.exists():
                    target_path = candidate
                    break
                counter += 1

        shutil.copy2(source_path, target_path)
        saved_paths.append(target_path)

        logger.info("已保存上传 PDF：%s", target_path)

    return saved_paths


def build_knowledge_base() -> dict:
    """
    重新构建整个本地论文知识库。

    流程：
    1. 读取 papers/ 中的 PDF；
    2. 解析 PDF 页面文本；
    3. 文本清洗与分块；
    4. Embedding 向量化；
    5. FAISS 构建索引；
    6. 保存索引和 chunks 元数据。

    Returns:
        result:
            构建结果字典，例如：
            {
                "success": True,
                "message": "...",
                "num_pages": 58,
                "num_chunks": 455,
                "num_pdfs": 5,
                "elapsed_seconds": 12.34
            }
    """

    ensure_project_dirs()

    start_time = time.time()

    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))

    if not pdf_files:
        return {
            "success": False,
            "message": "papers/ 文件夹中没有 PDF 文件，请先上传或放入论文 PDF。",
            "num_pdfs": 0,
            "num_pages": 0,
            "num_chunks": 0,
            "elapsed_seconds": 0.0,
        }

    logger.info("开始重建知识库，PDF 数量：%d", len(pdf_files))

    # ========================================================
    # 1. 解析 PDF
    # ========================================================

    pages = load_pdfs_from_folder(PAPERS_DIR)

    if not pages:
        elapsed_seconds = time.time() - start_time

        return {
            "success": False,
            "message": "没有从 PDF 中解析到有效文本，请检查 PDF 是否为可复制文本。",
            "num_pdfs": len(pdf_files),
            "num_pages": 0,
            "num_chunks": 0,
            "elapsed_seconds": elapsed_seconds,
        }

    # ========================================================
    # 2. 文本清洗与分块
    # ========================================================

    chunks = build_chunks_from_pages(
        pages=pages,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    if not chunks:
        elapsed_seconds = time.time() - start_time

        return {
            "success": False,
            "message": "文本分块结果为空，请检查 text_splitter.py。",
            "num_pdfs": len(pdf_files),
            "num_pages": len(pages),
            "num_chunks": 0,
            "elapsed_seconds": elapsed_seconds,
        }

    # 提取每个 chunk 的正文，用于生成 Embedding。
    texts = [chunk.text for chunk in chunks]

    # ========================================================
    # 3. 生成 Embedding
    # ========================================================

    embedding_model = EmbeddingModel(EMBEDDING_MODEL_NAME)
    embeddings = embedding_model.encode(texts)

    # ========================================================
    # 4. 构建并保存向量库
    # ========================================================

    vector_store = VectorStore()
    vector_store.build(embeddings)

    # 注意：
    # 你的 VectorStore.save() 定义是：
    # save(index_path, embeddings_path)
    #
    # 所以这里不能写 faiss_index_path=...
    # 否则就会出现：
    # VectorStore.save() got an unexpected keyword argument 'faiss_index_path'
    vector_store.save(
        FAISS_INDEX_PATH,
        EMBEDDINGS_PATH,
    )

    # ========================================================
    # 5. 保存 chunks 元数据
    # ========================================================

    # chunks 是 TextChunk dataclass 对象列表。
    # JSON 不能直接保存 dataclass，所以要先 asdict。
    chunk_dicts = [asdict(chunk) for chunk in chunks]

    save_chunks(
        chunk_dicts,
        CHUNKS_PATH,
    )

    elapsed_seconds = time.time() - start_time

    message = (
        "知识库重建完成："
        f"共 {len(pdf_files)} 个 PDF，"
        f"{len(pages)} 页有效文本，"
        f"{len(chunks)} 个文本块，"
        f"耗时 {elapsed_seconds:.2f} 秒。"
    )

    logger.info(message)

    return {
        "success": True,
        "message": message,
        "num_pdfs": len(pdf_files),
        "num_pages": len(pages),
        "num_chunks": len(chunks),
        "elapsed_seconds": elapsed_seconds,
    }


def upload_and_rebuild_knowledge_base(uploaded_files: Optional[Iterable]) -> dict:
    """
    保存上传 PDF，并重建知识库。

    这个函数主要给 app.py 调用。

    Args:
        uploaded_files:
            Gradio 上传文件列表。

    Returns:
        result:
            包含保存文件和知识库构建状态的字典。
    """

    saved_paths = save_uploaded_pdfs(uploaded_files)

    if not saved_paths:
        return {
            "success": False,
            "message": "没有检测到上传文件，请先选择 PDF 文件。",
            "saved_files": [],
        }

    build_result = build_knowledge_base()

    saved_file_names = [path.name for path in saved_paths]

    build_result["saved_files"] = saved_file_names

    if build_result.get("success"):
        build_result["message"] = (
            "已保存上传文件："
            + "、".join(saved_file_names)
            + "\n"
            + build_result["message"]
        )

    return build_result