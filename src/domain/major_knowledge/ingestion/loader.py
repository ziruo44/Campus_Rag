"""Document loader for various file formats."""

from pathlib import Path
from typing import List
from langchain_core.documents import Document


def load_document(file_path: str | Path) -> Document:
    """加载单个 Markdown 文档

    Args:
        file_path: 文件路径

    Returns:
        Document 对象
    """
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    return Document(
        page_content=content,
        metadata={
            "source": str(path.resolve()),
            "filename": path.name,
        }
    )


def load_documents(data_path: str | Path) -> List[Document]:
    """加载目录下所有 Markdown 文档

    Args:
        data_path: 目录路径或文件路径

    Returns:
        Document 列表
    """
    path = Path(data_path)

    # 单文件
    if path.is_file():
        return [load_document(path)]

    # 目录
    documents = []
    for md_file in path.rglob("*.md"):
        try:
            documents.append(load_document(md_file))
        except Exception as e:
            print(f"Warning: Failed to load {md_file}: {e}")

    return documents


if __name__ == "__main__":
    from utils.paths import get_raw_data_dir

    docs = load_documents(get_raw_data_dir())
    print(f"Loaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.metadata['source']}: {len(doc.page_content)} chars")
