"""Index builder module - ChromaDB + caching"""

import os
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv
from .embeddings import DashScopeEmbeddings
from rag_agent.utils.path import get_data_dir

logger = logging.getLogger(__name__)

load_dotenv

embedding_model = os.getenv("EMBEDDING_MODEL")


class IndexBuilder:
    """Vector index builder - ChromaDB + caching + deduplication"""

    def __init__(
        self,
        index_save_path: str | None = None,
        embedding_model: str = embedding_model,
        dimension: int = 768,
        collection_name: str = "rag_collection",
    ):
        self.index_save_path = index_save_path or str(get_data_dir() / "vector_index")
        self.collection_name = collection_name
        self.embeddings = DashScopeEmbeddings(model=embedding_model, dimension=dimension)
        self.vectorstore = None

    def _get_existing_parent_ids(self) -> set:
        """Get set of existing parent_ids from stored index"""
        if not Path(self.index_save_path).exists():
            return set()

        try:
            existing = Chroma(
                persist_directory=self.index_save_path,
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
            )
            results = existing.get(include=["metadatas"])
            parent_ids = set()
            for meta in results.get("metadatas", []):
                if meta and meta.get("parent_id"):
                    parent_ids.add(meta["parent_id"])
            return parent_ids
        except Exception:
            return set()

    def _filter_new_chunks(self, chunks: List[Document]) -> List[Document]:
        """Filter out chunks that already exist"""
        existing_ids = self._get_existing_parent_ids()
        new_chunks = [c for c in chunks if c.metadata.get("parent_id") not in existing_ids]

        skipped = len(chunks) - len(new_chunks)
        if skipped > 0:
            logger.info(f"Skipped {skipped} existing chunks")
        return new_chunks

    def build_index(self, chunks: List[Document], skip_duplicates: bool = True) -> Chroma:
        """Build ChromaDB index

        Args:
            chunks: List of document chunks
            skip_duplicates: Whether to skip existing chunks
        """
        logger.info("Building ChromaDB vector index...")

        if not chunks:
            raise ValueError("Chunk list cannot be empty")

        # Deduplication
        if skip_duplicates:
            chunks = self._filter_new_chunks(chunks)
            if not chunks:
                logger.info("All chunks already exist, no import needed")
                return self.vectorstore

        embeddings = self.embeddings

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=self.index_save_path,
            collection_name=self.collection_name,
        )

        logger.info(f"Vector index built, added {len(chunks)} vectors")
        return self.vectorstore

    def load_or_build_index(self, chunks: List[Document] | None = None) -> Chroma:
        """Index caching: load if exists, otherwise build

        Args:
            chunks: Optional list of document chunks

        Returns:
            Chroma vector store object
        """
        if Path(self.index_save_path).exists():
            self.vectorstore = self.load_index()
            return self.vectorstore
        elif chunks:
            index = self.build_index(chunks)
            self.save_index()
            self.vectorstore = index
            return self.vectorstore
        else:
            raise ValueError("Index does not exist and no chunks provided")

    def save_index(self):
        """Save vector index to local (Chroma auto-persists)"""
        if not self.vectorstore:
            raise ValueError("Please build index first")

        Path(self.index_save_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Vector index saved to: {self.index_save_path}")

    def load_index(self) -> Chroma:
        """Load vector index from configured path

        Returns:
            Loaded vector store object
        """
        embeddings = self.embeddings

        if not Path(self.index_save_path).exists():
            logger.info(f"Index path does not exist: {self.index_save_path}")
            raise ValueError("Index path does not exist")

        self.vectorstore = Chroma(
            persist_directory=self.index_save_path,
            embedding_function=embeddings,
            collection_name=self.collection_name,
        )

        logger.info(f"Vector index loaded from: {self.index_save_path}")
        return self.vectorstore

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Similarity search

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            List of similar documents
        """
        if not self.vectorstore:
            raise ValueError("Please build or load index first")

        return self.vectorstore.similarity_search(query, k=k)

    def metadata_filtered_search(
        self,
        query: str,
        filters: dict,
        k: int = 5,
    ) -> List[Document]:
        """Metadata-filtered search

        Args:
            query: Query text
            filters: Metadata filter conditions, e.g., {"college": "信息工程学院", "major": "计算机"}
            k: Number of results to return

        Returns:
            Filtered list of similar documents
        """
        if not self.vectorstore:
            raise ValueError("Please build or load index first")

        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k * 3, "filter": filters}
        )
        results = retriever.invoke(query)
        return results[:k]

    def get_collection(self):
        """Get Collection for metadata filtering"""
        if not self.vectorstore:
            raise ValueError("Please build or load index first")
        return self.vectorstore._collection


if __name__ == "__main__":
    from rag_agent.data_processing import load_documents, chunk_documents
    from rag_agent.utils.path import get_raw_data_dir

    logging.basicConfig(level=logging.INFO)

    # Load and chunk documents
    docs = load_documents(get_raw_data_dir())
    print(f"Loaded {len(docs)} documents")

    parents, children = chunk_documents(docs)
    all_chunks = parents + children
    print(f"Total chunks: {len(all_chunks)}")

    # Build index
    builder = IndexBuilder()
    builder.load_or_build_index(all_chunks)

    # Test search
    results = builder.similarity_search("计算机科学与技术", k=3)
    print(f"\nSearch results ({len(results)}):")
    for i, doc in enumerate(results):
        print(f"  {i+1}. {doc.page_content}...")
