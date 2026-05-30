"""Hybrid retrieval with RRF reranking - combines vector and BM25 search."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document

from domain.major_knowledge.retrieval.bm25_index import BM25Indexer
from domain.major_knowledge.retrieval.config import RetrievalSettings
from domain.major_knowledge.retrieval.metadata import (
    COLLEGE_FIELDS,
    MAJOR_FIELDS,
    SECTION_FIELDS,
    get_metadata_field,
)
from shared.observability.performance import measure_stage

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retriever combining vector search and BM25 with RRF reranking."""

    def __init__(
        self,
        index_builder,
        chunks: list[Document],
        *,
        parent_documents: list[Document] | None = None,
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
        self.settings = settings or RetrievalSettings()
        self._bm25_indexer: BM25Indexer | None = None
        self._parent_documents = list(parent_documents or [])
        self._parent_document_map = {
            doc.metadata.get("parent_id"): doc
            for doc in self._parent_documents
            if doc.metadata.get("parent_id")
        }
        self._setup_bm25(chunks)
        self._colleges: frozenset[str] = self._extract_colleges()
        self._majors: frozenset[str] = self._extract_majors()

    @property
    def colleges(self) -> frozenset[str]:
        """Return known college names extracted from indexed chunks."""
        return self._colleges

    @property
    def majors(self) -> frozenset[str]:
        """Return known major names extracted from indexed chunks."""
        return self._majors

    def _setup_bm25(self, chunks: list[Document]) -> None:
        """Build BM25 index from chunks."""
        logger.info("Setting up BM25 index...")
        self._bm25_indexer = BM25Indexer(chunks)
        logger.info("BM25 index setup complete")

    def _all_chunks(self) -> list[Document]:
        if self._bm25_indexer is None:
            raise RuntimeError("BM25 index not built")
        return self._bm25_indexer.chunks

    def _extract_colleges(self) -> frozenset[str]:
        colleges = {
            str(doc.metadata.get("college")).strip()
            for doc in self._parent_documents
            if doc.metadata.get("college")
        }
        colleges.update(
            str(chunk.metadata.get("college")).strip()
            for chunk in self._all_chunks()
            if chunk.metadata.get("college")
        )
        return frozenset(college for college in colleges if college)

    def _extract_majors(self) -> frozenset[str]:
        majors = {
            str(doc.metadata.get("major")).strip()
            for doc in self._parent_documents
            if doc.metadata.get("major")
        }
        majors.update(
            str(chunk.metadata.get("major")).strip()
            for chunk in self._all_chunks()
            if chunk.metadata.get("major")
        )
        return frozenset(major for major in majors if major)

    def vector_search(self, query: str, k: int | None = None) -> list[Document]:
        """Vector similarity search."""
        with measure_stage("retrieval.vector_search"):
            k = k or self.settings.vector_k
            return self.index_builder.similarity_search(query, k=k)

    def bm25_search(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
        """BM25 keyword search."""
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
        """Hybrid search combining vector and BM25 with RRF reranking."""
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

    def metadata_search(
        self,
        query: str = "",
        college: str | None = None,
        major: str | None = None,
        section: str | None = None,
        top_k: int | None = None,
    ) -> list[Document]:
        """Search with metadata filtering using field aliases."""
        with measure_stage("retrieval.metadata_search"):
            top_k = top_k or self.settings.default_top_k

            filters = {}
            if college:
                filters["college"] = college
            if major:
                filters["major"] = major
            if section:
                filters["section"] = section

            if not query:
                filtered = []
                for doc in self._all_chunks():
                    match = True
                    for key, value in filters.items():
                        doc_val = get_metadata_field(
                            doc,
                            COLLEGE_FIELDS if key == "college" else MAJOR_FIELDS if key == "major" else SECTION_FIELDS,
                        )
                        if doc_val is None or value not in doc_val:
                            match = False
                            break
                    if match:
                        filtered.append(doc)
                        if len(filtered) >= top_k:
                            break
                return filtered

            docs = self.hybrid_search(query, top_k=top_k * 3)

            filtered = []
            for doc in docs:
                match = True
                for key, value in filters.items():
                    doc_val = get_metadata_field(
                        doc,
                        COLLEGE_FIELDS if key == "college" else MAJOR_FIELDS if key == "major" else SECTION_FIELDS,
                    )
                    if doc_val is None or value not in doc_val:
                        match = False
                        break
                if match:
                    filtered.append(doc)
                    if len(filtered) >= top_k:
                        break

            return filtered

    def group_child_results(
        self,
        child_docs: list[Document],
        *,
        top_k_groups: int | None = None,
        max_children_per_group: int = 3,
    ) -> list[Document]:
        """Group retrieved child chunks by parent and build one assembled result per parent."""
        top_k_groups = top_k_groups or self.settings.default_top_k
        parent_groups: dict[str, dict[str, Any]] = {}

        for rank, doc in enumerate(child_docs):
            parent_id = str(doc.metadata.get("parent_id", "") or "")
            if not parent_id:
                continue

            group = parent_groups.setdefault(
                parent_id,
                {
                    "parent": self._parent_document_map.get(parent_id),
                    "children": [],
                    "score": 0.0,
                    "first_rank": rank,
                },
            )
            score = float(doc.metadata.get("rrf_score", 0.0) or 0.0)
            group["score"] += score if score > 0 else 1.0 / (rank + 1)
            group["children"].append(doc)

        sorted_groups = sorted(
            parent_groups.values(),
            key=lambda item: (item["score"], -item["first_rank"]),
            reverse=True,
        )

        assembled: list[Document] = []
        for group in sorted_groups[:top_k_groups]:
            assembled.append(
                self._assemble_grouped_document(
                    group["parent"],
                    group["children"][:max_children_per_group],
                    group["score"],
                )
            )

        return assembled

    def _rrf_rerank(
        self,
        vector_docs: list[Document],
        bm25_docs: list[Document],
        k: int | None = None,
    ) -> list[Document]:
        """RRF (Reciprocal Rank Fusion) reranking."""
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

    def _assemble_grouped_document(
        self,
        parent_doc: Document | None,
        child_docs: list[Document],
        group_score: float,
    ) -> Document:
        anchor = parent_doc or child_docs[0]
        metadata = dict(anchor.metadata)
        metadata.update(
            {
                "doc_type": "grouped_parent",
                "matched_sections": [
                    doc.metadata.get("section")
                    for doc in child_docs
                    if doc.metadata.get("section")
                ],
                "matched_child_count": len(child_docs),
                "rrf_score": group_score,
            }
        )

        content_parts = [
            f"### {metadata.get('major') or metadata.get('college') or 'Result'}",
        ]
        if metadata.get("college"):
            content_parts.append(f"学院：{metadata['college']}")
        if metadata.get("major"):
            content_parts.append(f"专业：{metadata['major']}")

        section_texts: list[str] = []
        seen_sections: set[str] = set()
        for doc in child_docs:
            section = str(doc.metadata.get("section", "") or "").strip()
            label = section or "相关片段"
            dedupe_key = f"{label}:{doc.page_content.strip()}"
            if dedupe_key in seen_sections:
                continue
            seen_sections.add(dedupe_key)
            section_texts.append(f"#### {label}\n{doc.page_content.strip()}")

        if section_texts:
            content_parts.append("### Matched Sections")
            content_parts.extend(section_texts)

        return Document(
            page_content="\n".join(content_parts).strip(),
            metadata=metadata,
        )
