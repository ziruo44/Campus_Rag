"""路径工具。"""

from pathlib import Path


def get_project_root() -> Path:
    """返回项目根目录。"""
    return Path(__file__).resolve().parent.parent.parent


def get_data_dir() -> Path:
    """返回数据目录。"""
    return get_project_root() / "data"


def get_major_raw_data_dir() -> Path:
    """返回专业知识库原始数据目录。"""
    return get_data_dir() / "raw" / "majors"


def get_raw_data_dir() -> Path:
    """兼容旧调用，返回专业知识库原始数据目录。"""
    return get_major_raw_data_dir()


def get_major_chroma_db_dir() -> Path:
    """返回专业知识库向量库目录。"""
    return get_data_dir() / "vector_index" / "majors"


def get_chroma_db_dir() -> Path:
    """兼容旧调用，返回专业知识库向量库目录。"""
    return get_major_chroma_db_dir()


def get_life_guide_raw_data_dir() -> Path:
    """返回生活指南知识库原始数据目录。"""
    return get_data_dir() / "raw" / "life_guide"


def get_life_guide_chroma_db_dir() -> Path:
    """返回生活指南知识库向量库目录。"""
    return get_data_dir() / "vector_index" / "life_guide_knowledge"
