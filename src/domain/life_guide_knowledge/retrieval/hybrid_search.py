"""Hybrid retrieval with RRF reranking - combines vector and BM25 search for life guide."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document

from .bm25_index import LifeGuideBM25Indexer, prewarm_jieba
from .config import LifeGuideRetrievalSettings

logger = logging.getLogger(__name__)


class LifeGuideHybridRetriever:
    """Hybrid retriever combining vector search and BM25 with RRF reranking."""

    def __init__(
        self,
        index_builder,
        chunks: list[Document],
        *,
        settings: LifeGuideRetrievalSettings | None = None,
    ):
        """
        Initialize hybrid retriever

        Args:
            index_builder: IndexBuilder instance with vectorstore
            chunks: All document chunks
            settings: Retrieval settings (uses defaults if not provided)
        """
        self.index_builder = index_builder
        self.settings = settings or LifeGuideRetrievalSettings()
        self._bm25_indexer: LifeGuideBM25Indexer | None = None
        self._setup_bm25(chunks)
        self._categories: frozenset[str] = self._extract_categories()

    @property
    def categories(self) -> frozenset[str]:
        """Return known category names extracted from indexed chunks."""
        return self._categories

    def _setup_bm25(self, chunks: list[Document]) -> None:
        """Build BM25 index from chunks."""
        logger.info("Setting up BM25 index for life guide...")
        prewarm_jieba()
        self._bm25_indexer = LifeGuideBM25Indexer(chunks)
        logger.info("BM25 index setup complete")

    def _all_chunks(self) -> list[Document]:
        if self._bm25_indexer is None:
            raise RuntimeError("BM25 index not built")
        return self._bm25_indexer.chunks

    def _extract_categories(self) -> frozenset[str]:
        categories = {
            str(doc.metadata.get("category")).strip()
            for doc in self._all_chunks()
            if doc.metadata.get("category")
        }
        return frozenset(cat for cat in categories if cat)

    def vector_search(self, query: str, k: int | None = None) -> list[Document]:
        """Vector similarity search."""
        k = k or self.settings.vector_k
        return self.index_builder.similarity_search(query, k=k)

    def bm25_search(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
        """BM25 keyword search."""
        k = k or self.settings.bm25_k
        return self._bm25_indexer.search(query, k=k)

    def hybrid_search(
        self,
        query: str,
        top_k: int | None = None,
        vector_k: int | None = None,
        bm25_k: int | None = None,
    ) -> list[Document]:
        """Hybrid search combining vector and BM25 with RRF reranking."""
        top_k = top_k or self.settings.default_top_k
        vector_k = vector_k or self.settings.vector_k
        bm25_k = bm25_k or self.settings.bm25_k

        vector_docs = self.vector_search(query, k=vector_k)
        bm25_results = self.bm25_search(query, k=bm25_k)
        bm25_docs = [doc for doc, _ in bm25_results]

        reranked = self._rrf_rerank(vector_docs, bm25_docs)
        return reranked[:top_k]

    def filtered_search(
        self,
        query: str,
        filters: dict[str, Any],
        top_k: int | None = None,
    ) -> list[Document]:
        """Search with metadata filtering."""
        top_k = top_k or self.settings.default_top_k

        docs = self.hybrid_search(query, top_k=top_k * 3)

        filtered = []
        for doc in docs:
            match = True
            for key, value in filters.items():
                if key not in doc.metadata:
                    match = False
                    break
                if isinstance(value, list):
                    if doc.metadata[key] not in value:
                        match = False
                        break
                elif doc.metadata[key] != value:
                    match = False
                    break
            if match:
                filtered.append(doc)
                if len(filtered) >= top_k:
                    break

        return filtered

    def category_search(
        self,
        query: str,
        category: str | None = None,
        top_k: int | None = None,
    ) -> list[Document]:
        """Search with optional category filtering."""
        if category:
            return self.filtered_search(query, {"category": category}, top_k=top_k)
        return self.hybrid_search(query, top_k=top_k)

    def _rrf_rerank(
        self,
        vector_docs: list[Document],
        bm25_docs: list[Document],
        k: int | None = None,
    ) -> list[Document]:
        """RRF (Reciprocal Rank Fusion) reranking."""
        k = k or self.settings.rrf_k

        doc_scores: dict[str, float] = {}
        doc_objects: dict[str, Document] = {}

        for rank, doc in enumerate(vector_docs):
            doc_id = self._document_key(doc)
            doc_objects[doc_id] = doc
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

        for rank, doc in enumerate(bm25_docs):
            doc_id = self._document_key(doc)
            doc_objects[doc_id] = doc
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

        sorted_ids = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        reranked = []
        for doc_id, score in sorted_ids:
            if doc_id in doc_objects:
                doc = doc_objects[doc_id]
                doc.metadata["rrf_score"] = score
                reranked.append(doc)

        logger.debug(
            f"RRF rerank: {len(vector_docs)} vector docs, "
            f"{len(bm25_docs)} BM25 docs -> {len(reranked)} results"
        )
        return reranked

    def _document_key(self, doc: Document) -> str:
        metadata = doc.metadata
        category = str(metadata.get("category", "") or "").strip()
        service_name = str(metadata.get("service_name", "") or "").strip()
        sub_service_name = str(metadata.get("sub_service_name", "") or "").strip()
        source = str(metadata.get("source", "") or metadata.get("filename", "") or "").strip()
        if category or service_name or sub_service_name or source:
            return f"{source}|{category}|{service_name}|{sub_service_name}"
        return doc.page_content.strip()
