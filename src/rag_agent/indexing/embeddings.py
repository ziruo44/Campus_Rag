"""Cloud embedding module - DashScope tongyi-embedding"""

import os
import logging
from typing import List

import dashscope
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

load_dotenv()

logger = logging.getLogger(__name__)

model_name = os.getenv("EMBEDDING_MODEL")
api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")


class DashScopeEmbeddings(Embeddings):
    """DashScope multimodal embedding model wrapper - text-only fusion vector"""

    def __init__(
        self,
        model: str = model_name,
        dimension: int = 768,
        api_key: str = api_key,
    ):
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
