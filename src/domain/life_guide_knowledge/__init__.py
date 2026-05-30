"""生活指南知识模块 - 扁平化分块结构"""

from .ingestion import load_life_guide, chunk_life_guide_all
from .indexing import LifeGuideIndexBuilder
from .retrieval import LifeGuideHybridRetriever
from .runtime import LifeGuideKnowledgeRuntime

__all__ = [
    "load_life_guide",
    "chunk_life_guide_all",
    "LifeGuideIndexBuilder",
    "LifeGuideHybridRetriever",
    "LifeGuideKnowledgeRuntime",
]
