"""
text_splitter.py

这个文件负责：
1. 清洗 PDF 提取出来的论文文本；
2. 修复 PDF 中常见的断词问题；
3. 去除 IEEE Xplore 等下载页脚噪声；
4. 将长文本切分成多个适合向量检索的 chunk；
5. 为每个 chunk 保留来源文件、页码等信息。

为什么要做文本清洗？
因为 PDF 直接提取出来的文本经常包含：
- 页眉页脚；
- 下载版权信息；
- 断词，比如 obser- vation；
- 多余换行；
- 奇怪空格。

这些噪声会影响 Embedding 向量质量，从而影响检索效果。
"""

import re
from dataclasses import dataclass
from typing import List

from dehaze_rag.pdf_loader import PageText


@dataclass
class TextChunk:
    """
    文本块数据结构。

    Attributes:
        chunk_id:
            文本块编号。

        source_file:
            文本块来自哪篇 PDF。

        page_number:
            文本块来自 PDF 第几页。

        text:
            文本块正文内容。
    """

    chunk_id: int
    source_file: str
    page_number: int
    text: str


def remove_noise_lines(text: str) -> str:
    """
    按行删除 PDF 噪声内容。

    这一步在合并空格之前做。
    因为一旦把换行全部合并，很多页脚噪声就不好识别了。

    Args:
        text:
            PDF 原始提取文本。

    Returns:
        删除噪声行后的文本。
    """

    noise_keywords = [
        "authorized licensed use limited",
        "downloaded on",
        "ieee xplore",
        "restrictions apply",
        "lanzhou university of technology",
    ]

    cleaned_lines = []

    for line in text.splitlines():
        raw_line = line.strip()
        lower_line = raw_line.lower()

        # 跳过空行
        if not raw_line:
            continue

        # 如果这一行包含典型页脚/版权/下载记录关键词，就删除
        if any(keyword in lower_line for keyword in noise_keywords):
            continue

        cleaned_lines.append(raw_line)

    return "\n".join(cleaned_lines)


def remove_inline_noise(text: str) -> str:
    """
    删除混入正文中的页脚噪声。

    有些 PDF 的页脚信息不是单独一行，而是混在正文中。
    所以这里再做一次正则清理。

    Args:
        text:
            初步清理后的文本。

    Returns:
        进一步清理后的文本。
    """

    # 删除 Authorized licensed use limited to ... 之类内容
    text = re.sub(
        r"Authorized licensed use limited to:.*?Restrictions apply\.?",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # 删除 Downloaded on ... from IEEE Xplore. Restrictions apply.
    text = re.sub(
        r"Downloaded on .*?IEEE Xplore\. Restrictions apply\.?",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # 删除残留的 from IEEE Xplore. Restrictions apply.
    text = re.sub(
        r"from IEEE Xplore\. Restrictions apply\.?",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # 删除残留的 Restrictions apply.
    text = re.sub(
        r"Restrictions apply\.?",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # 删除残留的 Lanzhou University of Technology 相关内容
    text = re.sub(
        r"Lanzhou University of Technology.*?(?=\s[A-Z][a-z]|\s\d+\.|\sFigure|\sTable|$)",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    return text


def fix_hyphenated_words(text: str) -> str:
    """
    修复 PDF 解析中的英文断词。

    例如：
        obser- vation  -> observation
        trans- mission -> transmission

    Args:
        text:
            输入文本。

    Returns:
        修复断词后的文本。
    """

    # 修复“字母- 空格 字母”的情况
    text = re.sub(r"([A-Za-z])-\s+([A-Za-z])", r"\1\2", text)

    return text


def clean_text(text: str) -> str:
    """
    统一文本清洗入口。

    Args:
        text:
            PDF 原始文本。

    Returns:
        清洗后的文本。
    """

    # 去掉空字符
    text = text.replace("\x00", " ")

    # 先按行删除明显噪声
    text = remove_noise_lines(text)

    # 修复英文断词
    text = fix_hyphenated_words(text)

    # 再删除混在正文里的噪声
    text = remove_inline_noise(text)

    # 合并多个空格、换行、制表符
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_text_by_window(
    text: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> List[str]:
    """
    使用滑动窗口切分文本。

    Args:
        text:
            输入文本。

        chunk_size:
            每个文本块最大字符数。

        overlap:
            相邻文本块之间的重叠字符数。

    Returns:
        文本块列表。
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")

    if overlap < 0:
        raise ValueError("overlap 不能小于 0")

    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")

    text = clean_text(text)

    if not text:
        return []

    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def build_chunks_from_pages(
    pages: List[PageText],
    chunk_size: int = 800,
    overlap: int = 150,
) -> List[TextChunk]:
    """
    将 PDF 页面文本切分成 TextChunk 列表。

    Args:
        pages:
            PDF 页面文本。

        chunk_size:
            每个 chunk 的最大字符数。

        overlap:
            相邻 chunk 的重叠字符数。

    Returns:
        TextChunk 列表。
    """

    chunks: List[TextChunk] = []
    chunk_id = 0

    for page in pages:
        page_chunks = split_text_by_window(
            page.text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk_text in page_chunks:
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    source_file=page.source_file,
                    page_number=page.page_number,
                    text=chunk_text,
                )
            )

            chunk_id += 1

    return chunks