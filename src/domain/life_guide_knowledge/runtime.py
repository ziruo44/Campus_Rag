"""生活指南知识库运行时。"""

from __future__ import annotations

import logging
from threading import Lock
from typing import Any

from domain.life_guide_knowledge.ingestion import chunk_life_guide_all, load_life_guide
from domain.life_guide_knowledge.indexing import LifeGuideIndexBuilder
from domain.life_guide_knowledge.retrieval import LifeGuideHybridRetriever
from shared.observability.performance import measure_stage
from utils.paths import get_life_guide_raw_data_dir

logger = logging.getLogger(__name__)


class RuntimeUnavailableError(RuntimeError):
    """生活指南运行时初始化失败时抛出。"""


class LifeGuideKnowledgeRuntime:
    """延迟初始化的生活指南检索运行时。"""

    def __init__(self) -> None:
        self._lock = Lock()
        self._initialized = False
        self._documents: list[Any] = []
        self._chunks: list[Any] = []
        self._index_builder: LifeGuideIndexBuilder | None = None
        self._retriever: LifeGuideHybridRetriever | None = None

    @property
    def is_initialized(self) -> bool:
        """是否已经完成初始化。"""
        return self._initialized

    @property
    def retriever(self) -> LifeGuideHybridRetriever:
        """返回共享检索器实例。"""
        if self._retriever is None:
            raise RuntimeUnavailableError("Life guide retriever is not initialized.")
        return self._retriever

    @property
    def documents(self) -> list[Any]:
        """返回原始文档。"""
        return list(self._documents)

    @property
    def chunks(self) -> list[Any]:
        """返回当前方案 A 的细粒度分块结果。"""
        return list(self._chunks)

    def ensure_initialized(self) -> None:
        """初始化生活指南文档、向量库与混合检索器。"""
        if self._initialized:
            return

        with measure_stage("life_guide_runtime.ensure_initialized"):
            with self._lock:
                if self._initialized:
                    return

                try:
                    with measure_stage("life_guide_runtime.load_documents"):
                        documents = load_life_guide(get_life_guide_raw_data_dir())
                    with measure_stage("life_guide_runtime.chunk_documents"):
                        chunks = chunk_life_guide_all(documents)
                    self._documents = documents
                    self._chunks = chunks

                    builder = LifeGuideIndexBuilder()
                    with measure_stage("life_guide_runtime.load_or_build_index"):
                        builder.load_or_build_index(self._chunks)
                    self._index_builder = builder

                    with measure_stage("life_guide_runtime.build_shared_retriever"):
                        self._retriever = LifeGuideHybridRetriever(builder, self._chunks)

                    self._initialized = True
                    logger.info(
                        "Life guide runtime initialized with %s documents and %s chunks",
                        len(self._documents),
                        len(self._chunks),
                    )
                except Exception as exc:
                    logger.exception("Failed to initialize life guide runtime")
                    raise RuntimeUnavailableError(
                        "Failed to initialize life guide runtime."
                    ) from exc
