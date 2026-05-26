"""Path utilities."""

from pathlib import Path


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def get_data_dir() -> Path:
    """Get data directory."""
    return get_project_root() / "data"


def get_raw_data_dir() -> Path:
    """Get raw data directory."""
    return get_data_dir() / "raw"


def get_chroma_db_dir() -> Path:
    """Get ChromaDB persistence directory."""
    return get_data_dir() / "vector_index"
