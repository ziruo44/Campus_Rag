"""Shared knowledge runtime for documents, indexes, and retrievers."""

from __future__ import annotations

import logging
from threading import Lock
from typing import Any

from domain.major_knowledge.ingestion import chunk_documents, load_documents
from domain.major_knowledge.indexing import IndexBuilder
from domain.major_knowledge.retrieval.hybrid_search import HybridRetriever
from shared.observability.performance import measure_stage
from utils.paths import get_raw_data_dir

logger = logging.getLogger(__name__)


class RuntimeUnavailableError(RuntimeError):
    """Raised when the shared knowledge runtime cannot be initialized."""


class KnowledgeRuntime:
    """Lazy runtime container for shared retrieval assets."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._initialized = False
        self._parent_documents: list[Any] = []
        self._child_documents: list[Any] = []
        self._parent_document_map: dict[str, Any] = {}
        self._index_builder: IndexBuilder | None = None
        self._retriever: HybridRetriever | None = None

    @property
    def is_initialized(self) -> bool:
        """Whether the runtime has been initialized."""
        return self._initialized

    @property
    def retriever(self) -> HybridRetriever:
        """Return the shared retriever instance."""
        if self._retriever is None:
            raise RuntimeUnavailableError("Shared retriever is not initialized.")
        return self._retriever

    @property
    def parent_documents(self) -> list[Any]:
        """Return parent documents retained outside the vector index."""
        return list(self._parent_documents)

    @property
    def child_documents(self) -> list[Any]:
        """Return child chunks used for indexing and retrieval."""
        return list(self._child_documents)

    def ensure_initialized(self) -> None:
        """Initialize documents, vector index, and the shared retriever."""
        if self._initialized:
            return

        with measure_stage("knowledge_runtime.ensure_initialized"):
            with self._lock:
                if self._initialized:
                    return

                try:
                    with measure_stage("knowledge_runtime.load_documents"):
                        docs = load_documents(get_raw_data_dir())
                    with measure_stage("knowledge_runtime.chunk_documents"):
                        parents, children = chunk_documents(docs)
                    self._parent_documents = parents
                    self._child_documents = children
                    self._parent_document_map = {
                        doc.metadata.get("parent_id"): doc
                        for doc in parents
                        if doc.metadata.get("parent_id")
                    }

                    builder = IndexBuilder()
                    with measure_stage("knowledge_runtime.load_or_build_index"):
                        builder.load_or_build_index(self._child_documents)
                    self._index_builder = builder

                    with measure_stage("knowledge_runtime.build_shared_retriever"):
                        self._retriever = HybridRetriever(
                            builder,
                            self._child_documents,
                            parent_documents=self._parent_documents,
                        )

                    self._initialized = True
                    logger.info(
                        "Knowledge runtime initialized with %s parent docs and %s child chunks",
                        len(self._parent_documents),
                        len(self._child_documents),
                    )
                except Exception as exc:
                    logger.exception("Failed to initialize knowledge runtime")
                    raise RuntimeUnavailableError(
                        "Failed to initialize knowledge runtime."
                    ) from exc
