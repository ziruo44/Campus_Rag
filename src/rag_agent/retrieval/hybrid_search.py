"""Hybrid retrieval with RRF reranking - combines vector and BM25 search"""

import logging
from typing import Any

from langchain_core.documents import Document

from rag_agent.observability.performance import measure_stage
from rag_agent.retrieval.bm25_index import BM25Indexer
from rag_agent.retrieval.config import RetrievalSettings

logger = logging.getLogger(__name__)

# Metadata field aliases (English <-> Chinese)
COLLEGE_FIELDS = ["college", "二级学院", "学院"]
MAJOR_FIELDS = ["major", "专业名称"]
SECTION_FIELDS = ["section", "章节"]
ACADEMY_FIELD = "academy"


def _get_field(doc: Document, fields: list[str], default: Any = None) -> Any:
    """Get field value from document metadata, trying multiple field names"""
    for field in fields:
        if field in doc.metadata:
            return doc.metadata[field]
    return default


class HybridRetriever:
    """Hybrid retriever combining vector search and BM25 with RRF reranking"""

    def __init__(
        self,
        index_builder,
        chunks: list[Document],
        settings: RetrievalSettings | None = None,
    ):
        """
        Initialize hybrid retriever

        Args:
            index_builder: IndexBuilder instance with vectorstore
            chunks: All document chunks
            settings: Retrieval settings (uses defaults if not provided)
        """
        self.index_builder = index_builder
        self.chunks = chunks
        self.settings = settings or RetrievalSettings()
        self._bm25_indexer: BM25Indexer | None = None
        self._setup_bm25()

    def _setup_bm25(self):
        """Build BM25 index from chunks"""
        logger.info("Setting up BM25 index...")
        self._bm25_indexer = BM25Indexer(self.chunks)
        logger.info("BM25 index setup complete")

    def vector_search(self, query: str, k: int | None = None) -> list[Document]:
        """
        Vector similarity search

        Args:
            query: Query text
            k: Number of results

        Returns:
            List of documents
        """
        with measure_stage("retrieval.vector_search"):
            k = k or self.settings.vector_k
            return self.index_builder.similarity_search(query, k=k)

    def bm25_search(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
        """
        BM25 keyword search

        Args:
            query: Query text
            k: Number of results

        Returns:
            List of (Document, score) tuples
        """
        with measure_stage("retrieval.bm25_search"):
            k = k or self.settings.bm25_k
            return self._bm25_indexer.search(query, k=k)

    def hybrid_search(
        self,
        query: str,
        top_k: int | None = None,
        vector_k: int | None = None,
        bm25_k: int | None = None,
    ) -> list[Document]:
        """
        Hybrid search combining vector and BM25 with RRF reranking

        Args:
            query: Query text
            top_k: Number of final results to return
            vector_k: Number of results from vector search
            bm25_k: Number of results from BM25 search

        Returns:
            List of reranked documents
        """
        with measure_stage("retrieval.hybrid_search"):
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
        """
        Search with metadata filtering

        Args:
            query: Query text
            filters: Metadata filter conditions
            top_k: Number of results to return

        Returns:
            Filtered and reranked documents
        """
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

    def metadata_search(
        self,
        query: str = "",
        college: str | None = None,
        major: str | None = None,
        section: str | None = None,
        top_k: int | None = None,
    ) -> list[Document]:
        """
        Search with metadata filtering using field aliases.

        Supports both English and Chinese field names:
        - college: "college", "二级学院", "学院"
        - major: "major", "专业名称"
        - section: "section", "章节"

        Args:
            query: Query text (optional, uses hybrid search if provided)
            college: College/academy name (supports alias: 二级学院, 学院)
            major: Major name (supports alias: 专业名称)
            section: Section name (supports alias: 章节)
            top_k: Number of results to return

        Returns:
            Filtered documents matching the metadata criteria
        """
        with measure_stage("retrieval.metadata_search"):
            top_k = top_k or self.settings.default_top_k

            # Build filters with field alias support
            filters = {}
            if college:
                filters["college"] = college
            if major:
                filters["major"] = major
            if section:
                filters["section"] = section

            # If no query, search all chunks directly with metadata filter
            if not query:
                filtered = []
                for doc in self.chunks:
                    match = True
                    for key, value in filters.items():
                        doc_val = _get_field(doc, COLLEGE_FIELDS if key == "college" else MAJOR_FIELDS if key == "major" else SECTION_FIELDS)
                        if doc_val is None or value not in doc_val:
                            match = False
                            break
                    if match:
                        filtered.append(doc)
                        if len(filtered) >= top_k:
                            break
                return filtered

            # With query: use hybrid search then filter
            docs = self.hybrid_search(query, top_k=top_k * 3)

            filtered = []
            for doc in docs:
                match = True
                for key, value in filters.items():
                    doc_val = _get_field(doc, COLLEGE_FIELDS if key == "college" else MAJOR_FIELDS if key == "major" else SECTION_FIELDS)
                    if doc_val is None or value not in doc_val:
                        match = False
                        break
                if match:
                    filtered.append(doc)
                    if len(filtered) >= top_k:
                        break

            return filtered

    def _rrf_rerank(
        self,
        vector_docs: list[Document],
        bm25_docs: list[Document],
        k: int | None = None,
    ) -> list[Document]:
        """
        RRF (Reciprocal Rank Fusion) reranking

        Args:
            vector_docs: Documents from vector search
            bm25_docs: Documents from BM25 search
            k: RRF smoothing parameter

        Returns:
            Reranked documents with rrf_score in metadata
        """
        k = k or self.settings.rrf_k

        doc_scores: dict[int, float] = {}
        doc_objects: dict[int, Document] = {}

        for rank, doc in enumerate(vector_docs):
            doc_id = id(doc)
            doc_objects[doc_id] = doc
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

        for rank, doc in enumerate(bm25_docs):
            doc_id = id(doc)
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
