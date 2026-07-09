"""
这个文件用于测试：
1. 程序能不能找到 papers 文件夹；
2. 程序能不能读取里面的 PDF；
3. PyMuPDF 是否能正常解析论文文本。

运行方式：
python test_pdf_loader.py
"""

from pathlib import Path

from dehaze_rag.pdf_loader import load_pdfs_from_folder


def main():
    # 当前项目根目录，也就是 dehaze-rag-assistant
    project_root = Path(__file__).resolve().parent

    # papers 文件夹路径
    papers_dir = project_root / "papers"

    print("正在检查 papers 文件夹：", papers_dir)

    # 读取 papers 文件夹里的所有 PDF
    pages = load_pdfs_from_folder(papers_dir)

    print(f"PDF 解析成功！共读取到 {len(pages)} 页有效文本。")

    # 打印第一页的部分内容，确认确实读到了论文正文
    first_page = pages[0]
    print("\n====== 第一页信息 ======")
    print("文件名：", first_page.source_file)
    print("页码：", first_page.page_number)
    print("文本预览：")
    print(first_page.text[:500])


if __name__ == "__main__":
    main()