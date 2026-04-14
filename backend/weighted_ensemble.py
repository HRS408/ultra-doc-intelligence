"""
Weighted ensemble retriever: reciprocal rank fusion (RRF) over multiple LangChain retrievers.

LangChain 0.3 exposed ``EnsembleRetriever`` in ``langchain``; LangChain 1.x moved packages.
This mirrors the usual weighted RRF pattern with ``weights=[0.4, 0.6]`` (BM25, vector).
"""

from __future__ import annotations

from typing import Any, List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, model_validator


class WeightedEnsembleRetriever(BaseRetriever):
    retrievers: List[Any] = Field(default_factory=list, repr=False)
    weights: List[float] = Field(default_factory=list)
    rrf_k: int = 60

    @model_validator(mode="after")
    def _check(self) -> WeightedEnsembleRetriever:
        if len(self.retrievers) != len(self.weights):
            raise ValueError("retrievers and weights must have the same length")
        if not self.retrievers:
            raise ValueError("At least one retriever is required")
        return self

    def _invoke_retriever(self, r: Any, query: str) -> List[Document]:
        if hasattr(r, "invoke"):
            out = r.invoke(query)
            return list(out) if out is not None else []
        return r.get_relevant_documents(query)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        scores: dict[str, float] = {}
        by_key: dict[str, Document] = {}
        for r, w in zip(self.retrievers, self.weights):
            docs = self._invoke_retriever(r, query)
            for rank, doc in enumerate(docs, start=1):
                meta = doc.metadata or {}
                key = str(meta.get("chunk_index", ""))
                scores[key] = scores.get(key, 0.0) + float(w) / (self.rrf_k + rank)
                by_key[key] = doc
        if not scores:
            return []
        ordered = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return [by_key[k] for k in ordered]
