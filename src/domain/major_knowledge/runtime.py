"""专业知识库运行时。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from domain.major_knowledge.ingestion import chunk_documents, load_documents
from domain.major_knowledge.indexing import IndexBuilder
from domain.major_knowledge.retrieval.hybrid_search import HybridRetriever
from domain.runtime_base import LazyRuntimeBase
from shared.observability.performance import measure_stage
from utils.paths import get_raw_data_dir

logger = logging.getLogger(__name__)


class RuntimeUnavailableError(RuntimeError):
    """Raised when the shared knowledge runtime cannot be initialized."""


class KnowledgeRuntime(LazyRuntimeBase):
    """延迟初始化的专业知识检索运行时。"""

    def __init__(
        self,
        *,
        raw_data_dir_factory: Callable[[], Path] | None = None,
        document_loader: Callable[[Path], list[Any]] | None = None,
        document_chunker: Callable[[list[Any]], tuple[list[Any], list[Any]]] | None = None,
        index_builder_factory: Callable[[], IndexBuilder] | None = None,
        retriever_factory: Callable[..., HybridRetriever] | None = None,
    ) -> None:
        super().__init__(
            stage_prefix="knowledge_runtime",
            failure_message="Failed to initialize knowledge runtime.",
            log_name="knowledge runtime",
        )
        self._raw_data_dir_factory = raw_data_dir_factory
        self._document_loader = document_loader
        self._document_chunker = document_chunker
        self._index_builder_factory = index_builder_factory
        self._retriever_factory = retriever_factory
        self._parent_documents: list[Any] = []
        self._child_documents: list[Any] = []
        self._index_builder: IndexBuilder | None = None
        self._retriever: HybridRetriever | None = None

    @property
    def retriever(self) -> HybridRetriever:
        """返回共享检索器实例。"""
        if self._retriever is None:
            raise RuntimeUnavailableError("Shared retriever is not initialized.")
        return self._retriever

    @property
    def parent_documents(self) -> list[Any]:
        """返回父文档。"""
        return list(self._parent_documents)

    @property
    def child_documents(self) -> list[Any]:
        """返回索引与检索使用的子块。"""
        return list(self._child_documents)

    @property
    def runtime_error_class(self):
        return RuntimeUnavailableError

    def _initialize_once(self) -> None:
        raw_data_dir_factory = self._raw_data_dir_factory or get_raw_data_dir
        document_loader = self._document_loader or load_documents
        document_chunker = self._document_chunker or chunk_documents
        index_builder_factory = self._index_builder_factory or IndexBuilder
        retriever_factory = self._retriever_factory or HybridRetriever

        with measure_stage("knowledge_runtime.load_documents"):
            docs = document_loader(raw_data_dir_factory())
        with measure_stage("knowledge_runtime.chunk_documents"):
            parents, children = document_chunker(docs)
        self._parent_documents = parents
        self._child_documents = children

        builder = index_builder_factory()
        with measure_stage("knowledge_runtime.load_or_build_index"):
            builder.load_or_build_index(self._child_documents)
        self._index_builder = builder

        with measure_stage("knowledge_runtime.build_shared_retriever"):
            self._retriever = retriever_factory(
                builder,
                self._child_documents,
                parent_documents=self._parent_documents,
            )

        logger.info(
            "Knowledge runtime initialized with %s parent docs and %s child chunks",
            len(self._parent_documents),
            len(self._child_documents),
        )
