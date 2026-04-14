"""LangChain Embeddings adapter for an existing sentence-transformers model."""

from __future__ import annotations

from typing import List

from langchain_core.embeddings import Embeddings


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model) -> None:
        self._model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        emb = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.tolist()

    def embed_query(self, text: str) -> List[float]:
        emb = self._model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.tolist()[0]
