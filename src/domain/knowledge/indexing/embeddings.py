"""Cloud embedding module - DashScope tongyi-embedding."""

import logging
from typing import List

import dashscope
from langchain_core.embeddings import Embeddings

from domain.knowledge.indexing.config import IndexingSettings

logger = logging.getLogger(__name__)

_settings = IndexingSettings()


class DashScopeEmbeddings(Embeddings):
    """DashScope multimodal embedding model wrapper - text-only fusion vector."""

    def __init__(
        self,
        model: str | None = _settings.embedding_model,
        dimension: int = _settings.embedding_dimension,
        api_key: str | None = _settings.resolved_embedding_api_key,
    ) -> None:
        self.model = model
        self.dimension = dimension
        self.api_key = api_key

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Batch embed documents"""
        embeddings = []
        for text in texts:
            resp = dashscope.MultiModalEmbedding.call(
                api_key=self.api_key,
                model=self.model,
                input=[{"text": text}],
            )
            if resp.status_code == 200:
                embeddings.append(resp.output["embeddings"][0]["embedding"])
            else:
                logger.error(f"Embedding failed: {resp.message}")
                raise ValueError(f"Embedding failed: {resp.message}")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        resp = dashscope.MultiModalEmbedding.call(
            api_key=self.api_key,
            model=self.model,
            input=[{"text": text}],
        )
        if resp.status_code == 200:
            return resp.output["embeddings"][0]["embedding"]
        else:
            logger.error(f"Embedding failed: {resp.message}")
            raise ValueError(f"Embedding failed: {resp.message}")
