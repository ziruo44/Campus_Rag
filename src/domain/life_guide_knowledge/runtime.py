"""生活指南知识库运行时。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from domain.life_guide_knowledge.ingestion import chunk_life_guide_all, load_life_guide
from domain.life_guide_knowledge.indexing import LifeGuideIndexBuilder
from domain.life_guide_knowledge.retrieval import LifeGuideHybridRetriever
from domain.runtime_base import LazyRuntimeBase
from shared.observability.performance import measure_stage
from utils.paths import get_life_guide_raw_data_dir

logger = logging.getLogger(__name__)


class RuntimeUnavailableError(RuntimeError):
    """生活指南运行时初始化失败时抛出。"""


class LifeGuideKnowledgeRuntime(LazyRuntimeBase):
    """延迟初始化的生活指南检索运行时。"""

    def __init__(
        self,
        *,
        raw_data_dir_factory: Callable[[], Path] | None = None,
        document_loader: Callable[[Path], list[Any]] | None = None,
        document_chunker: Callable[[list[Any]], list[Any]] | None = None,
        index_builder_factory: Callable[[], LifeGuideIndexBuilder] | None = None,
        retriever_factory: Callable[..., LifeGuideHybridRetriever] | None = None,
    ) -> None:
        super().__init__(
            stage_prefix="life_guide_runtime",
            failure_message="Failed to initialize life guide runtime.",
            log_name="life guide runtime",
        )
        self._raw_data_dir_factory = raw_data_dir_factory
        self._document_loader = document_loader
        self._document_chunker = document_chunker
        self._index_builder_factory = index_builder_factory
        self._retriever_factory = retriever_factory
        self._documents: list[Any] = []
        self._chunks: list[Any] = []
        self._index_builder: LifeGuideIndexBuilder | None = None
        self._retriever: LifeGuideHybridRetriever | None = None

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

    @property
    def runtime_error_class(self):
        return RuntimeUnavailableError

    def _initialize_once(self) -> None:
        raw_data_dir_factory = self._raw_data_dir_factory or get_life_guide_raw_data_dir
        document_loader = self._document_loader or load_life_guide
        document_chunker = self._document_chunker or chunk_life_guide_all
        index_builder_factory = self._index_builder_factory or LifeGuideIndexBuilder
        retriever_factory = self._retriever_factory or LifeGuideHybridRetriever

        with measure_stage("life_guide_runtime.load_documents"):
            documents = document_loader(raw_data_dir_factory())
        with measure_stage("life_guide_runtime.chunk_documents"):
            chunks = document_chunker(documents)
        self._documents = documents
        self._chunks = chunks

        builder = index_builder_factory()
        with measure_stage("life_guide_runtime.load_or_build_index"):
            builder.load_or_build_index(self._chunks)
        self._index_builder = builder

        with measure_stage("life_guide_runtime.build_shared_retriever"):
            self._retriever = retriever_factory(builder, self._chunks)

        logger.info(
            "Life guide runtime initialized with %s documents and %s chunks",
            len(self._documents),
            len(self._chunks),
        )
