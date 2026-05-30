"""BM25 index for keyword-based retrieval - life guide knowledge."""

import logging
from typing import List

from langchain_core.documents import Document

try:
    import jieba
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("rank_bm25 and jieba are required. Install with: uv add rank_bm25 jieba")

logger = logging.getLogger(__name__)


def prewarm_jieba(*, silent: bool = True) -> None:
    """Initialize jieba ahead of first retrieval to avoid first-turn latency."""
    if silent:
        jieba.setLogLevel(logging.ERROR)
    jieba.initialize()


class LifeGuideBM25Indexer:
    """BM25 indexer for keyword-based retrieval"""

    def __init__(self, chunks: List[Document]):
        """
        Initialize BM25 indexer

        Args:
            chunks: Document chunks to index
        """
        self.chunks = chunks
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: BM25Okapi | None = None
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        """Chinese tokenization using jieba"""
        return list(jieba.cut(text))

    def _build_index(self):
        """Build BM25 index from chunks"""
        if not self.chunks:
            logger.warning("No chunks provided for BM25 indexing")
            return

        logger.info(f"Building BM25 index for {len(self.chunks)} chunks...")

        self.tokenized_corpus = [
            self._tokenize(doc.page_content) for doc in self.chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info("BM25 index built successfully")

    def search(self, query: str, k: int = 5) -> List[tuple[Document, float]]:
        """
        Search BM25 index

        Args:
            query: Query text
            k: Number of results

        Returns:
            List of (Document, score) tuples
        """
        if not self.bm25:
            raise RuntimeError("BM25 index not built")

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k document indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.chunks[idx], scores[idx]))

        return results