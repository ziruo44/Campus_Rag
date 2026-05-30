"""Index builder module - ChromaDB for life guide knowledge."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import LifeGuideIndexingSettings
from .embeddings import LifeGuideEmbeddings
from utils.paths import get_life_guide_chroma_db_dir

logger = logging.getLogger(__name__)
_settings = LifeGuideIndexingSettings()
class LifeGuideIndexBuilder:
    """Vector index builder for life guide - ChromaDB."""

    def __init__(
        self,
        index_save_path: str | None = None,
        embedding_model: str | None = None,
        dimension: int | None = None,
        collection_name: str = "life_guide_knowledge_collection",
    ) -> None:
        self.index_save_path = index_save_path or str(get_life_guide_chroma_db_dir())
        self.collection_name = collection_name
        resolved_dimension = dimension if dimension is not None else _settings.embedding_dimension
        self.embeddings = LifeGuideEmbeddings(model=embedding_model, dimension=resolved_dimension)
        self.vectorstore = None

    def _get_existing_chunk_keys(self) -> set[str]:
        """Get the set of existing chunk keys from the stored index."""
        if not Path(self.index_save_path).exists():
            return set()

        try:
            existing = Chroma(
                persist_directory=self.index_save_path,
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
            )
            results = existing.get(include=["metadatas"])
            chunk_keys = set()
            for meta in results.get("metadatas", []):
                if not meta:
                    continue
                chunk_key = self._chunk_key_from_metadata(meta)
                if chunk_key:
                    chunk_keys.add(chunk_key)
            return chunk_keys
        except Exception:
            return set()

    def _filter_new_chunks(self, chunks: list[Document]) -> list[Document]:
        """Filter out chunks that already exist."""
        existing_keys = self._get_existing_chunk_keys()
        new_chunks = [
            c for c in chunks
            if self._chunk_key_from_metadata(c.metadata) not in existing_keys
        ]

        skipped = len(chunks) - len(new_chunks)
        if skipped > 0:
            logger.info("Skipped %s existing chunks", skipped)
        return new_chunks

    def _chunk_key_from_metadata(self, metadata: dict) -> str | None:
        category = metadata.get("category")
        service_name = metadata.get("service_name")
        sub_service_name = metadata.get("sub_service_name")
        if category is None or service_name is None:
            return None
        return f"{category}:{service_name}:{sub_service_name or ''}"

    def build_index(self, chunks: list[Document], skip_duplicates: bool = True) -> Chroma:
        """Build the ChromaDB index from document chunks."""
        logger.info("Building ChromaDB vector index for life guide...")

        if not chunks:
            raise ValueError("Chunk list cannot be empty")

        if skip_duplicates:
            chunks = self._filter_new_chunks(chunks)
            if not chunks:
                logger.info("All chunks already exist, no import needed")
                return self.vectorstore

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.index_save_path,
            collection_name=self.collection_name,
        )

        logger.info("Vector index built, added %s vectors", len(chunks))
        return self.vectorstore

    def load_or_build_index(self, chunks: list[Document] | None = None) -> Chroma:
        """Load the index if it exists, otherwise build it from chunks."""
        if Path(self.index_save_path).exists():
            self.vectorstore = self.load_index()
            return self.vectorstore
        if chunks:
            index = self.build_index(chunks)
            self.save_index()
            self.vectorstore = index
            return self.vectorstore
        raise ValueError("Index does not exist and no chunks provided")

    def save_index(self) -> None:
        """Persist the vector index locally."""
        if not self.vectorstore:
            raise ValueError("Please build index first")

        Path(self.index_save_path).mkdir(parents=True, exist_ok=True)
        logger.info("Vector index saved to: %s", self.index_save_path)

    def load_index(self) -> Chroma:
        """Load the vector index from the configured path."""
        if not Path(self.index_save_path).exists():
            logger.info("Index path does not exist: %s", self.index_save_path)
            raise ValueError("Index path does not exist")

        self.vectorstore = Chroma(
            persist_directory=self.index_save_path,
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
        )

        logger.info("Vector index loaded from: %s", self.index_save_path)
        return self.vectorstore

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        """Run similarity search against the current vector store."""
        if not self.vectorstore:
            raise ValueError("Please build or load index first")

        return self.vectorstore.similarity_search(query, k=k)


if __name__ == "__main__":
    from domain.life_guide_knowledge.ingestion import load_life_guide, chunk_life_guide_all

    docs = load_life_guide()
    chunks = chunk_life_guide_all(docs)

    builder = LifeGuideIndexBuilder()
    builder.build_index(chunks)
    builder.save_index()

    results = builder.similarity_search("食堂外卖", k=3)
    print(f"\nRetrieved {len(results)} results for '食堂外卖':")
    for doc in results:
        print(f"  - [{doc.metadata.get('category')}] {doc.metadata.get('service_name')}")
