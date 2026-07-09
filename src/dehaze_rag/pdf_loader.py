import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pymupdf


logger = logging.getLogger(__name__)


@dataclass
class PageText:
    """保存单页PDF文本信息。"""
    source_file: str
    page_number: int
    text: str


def extract_text_from_pdf(pdf_path: Path) -> List[PageText]:
    """
    从单个 PDF 中提取每一页的文本。

    Args:
        pdf_path: PDF 文件路径。

    Returns:
        PageText 列表，每个元素对应 PDF 的一页。
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在：{pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"文件不是 PDF：{pdf_path}")

    pages: List[PageText] = []

    try:
        doc = pymupdf.open(pdf_path)

        for page_index in range(len(doc)):
            page = doc[page_index]
            text = page.get_text("text").strip()

            if text:
                pages.append(
                    PageText(
                        source_file=pdf_path.name,
                        page_number=page_index + 1,
                        text=text,
                    )
                )

        doc.close()

    except Exception as exc:
        logger.exception("解析 PDF 失败：%s", pdf_path)
        raise RuntimeError(f"解析 PDF 失败：{pdf_path}") from exc

    return pages


def load_pdfs_from_folder(folder: Path) -> List[PageText]:
    """
    从文件夹中批量读取 PDF。

    Args:
        folder: papers 文件夹路径。

    Returns:
        所有 PDF 的 PageText 列表。
    """
    if not folder.exists():
        raise FileNotFoundError(f"论文文件夹不存在：{folder}")

    pdf_files = sorted(folder.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"没有在 {folder} 中找到 PDF，请先把论文放进 papers/ 文件夹。"
        )

    all_pages: List[PageText] = []

    for pdf_path in pdf_files:
        logger.info("正在解析 PDF：%s", pdf_path.name)
        pages = extract_text_from_pdf(pdf_path)
        all_pages.extend(pages)

    return all_pages