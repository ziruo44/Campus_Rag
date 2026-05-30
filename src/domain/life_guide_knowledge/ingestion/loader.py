"""生活指南文档加载器"""

from pathlib import Path
from typing import List
from langchain_core.documents import Document


def load_life_guide(data_path: str | Path = None) -> List[Document]:
    """加载生活指南文档

    Args:
        data_path: 生活指南目录路径，默认使用 data/raw/life_guide

    Returns:
        Document 列表
    """
    if data_path is None:
        from utils.paths import get_life_guide_raw_data_dir
        data_path = get_life_guide_raw_data_dir()

    path = Path(data_path)
    life_guide_file = path / "生活指南.md"

    if not life_guide_file.exists():
        raise FileNotFoundError(f"生活指南文件不存在: {life_guide_file}")

    with open(life_guide_file, "r", encoding="utf-8") as f:
        content = f.read()

    return [Document(
        page_content=content,
        metadata={
            "source": str(life_guide_file.resolve()),
            "filename": life_guide_file.name,
        }
    )]


if __name__ == "__main__":
    docs = load_life_guide()
    print(f"Loaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.metadata['source']}: {len(doc.page_content)} chars")
