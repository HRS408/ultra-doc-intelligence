"""Ensemble retrieval (BM25 + Chroma vector) and Groq RAG completion."""

import os
from typing import Any, Callable, List, Tuple

from groq import Groq
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from backend.embeddings_lc import SentenceTransformerEmbeddings
from backend.guardrails import compute_confidence, confidence_label_from_score, low_confidence_answer
from backend.metadata_filter import doc_type_prefilter_where
from backend.schemas import AskResponse, SourceSnippet
from backend.weighted_ensemble import WeightedEnsembleRetriever

MODEL = "llama-3.3-70b-versatile"

_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=_api_key) if _api_key else Groq(api_key="dummy")

SYSTEM_RAG = """You are a logistics document assistant. Answer ONLY using the provided
document context. If the answer is not present in the context, respond exactly with:
"NOT_IN_DOCUMENT". Do not infer, hallucinate, or use outside knowledge.
Be concise and specific."""


def _distance_to_similarity(distance: float) -> float:
    """Chroma cosine space: distance ≈ 1 - cosine_similarity for normalized vectors."""
    sim = 1.0 - float(distance)
    return max(0.0, min(1.0, sim))


def _cosine_from_embeddings(a: List[float], b: List[float]) -> float:
    return max(0.0, min(1.0, sum(x * y for x, y in zip(a, b))))


def _int_chunk_index(meta: Any) -> int:
    m = meta or {}
    v = m.get("chunk_index", -1)
    try:
        return int(v)
    except (TypeError, ValueError):
        return -1


def retrieve_ensemble(
    collection: Any,
    chroma_client: Any,
    question: str,
    embed_fn: Callable[[List[str]], List[List[float]]],
    st_model: Any,
    min_similarity: float = 0.30,
) -> Tuple[List[Tuple[str, int, float]], List[float]]:
    """
    LangChain-style ensemble: BM25Retriever (k=5) + Chroma vector retriever (k=5),
    weights [0.4, 0.6], with optional doc_type metadata pre-filter from the question.
    Returns (top-3 context rows (text, chunk_index, sim), auxiliary sim list for scoring).
    """
    where = doc_type_prefilter_where(question)
    include = ["documents", "metadatas"]

    if where:
        data = collection.get(where=where, include=include)
    else:
        data = collection.get(include=include)

    ids = data.get("ids") or []
    active_where = where
    if where and len(ids) == 0:
        data = collection.get(include=include)
        active_where = None

    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    if not documents:
        return [], []

    lc_docs = [
        Document(page_content=(d or ""), metadata=dict(m or {}))
        for d, m in zip(documents, metadatas)
    ]

    bm25 = BM25Retriever.from_documents(lc_docs)
    bm25.k = 5

    emb = SentenceTransformerEmbeddings(st_model)
    vs = Chroma(
        client=chroma_client,
        collection_name=collection.name,
        embedding_function=emb,
    )
    search_kwargs: dict = {"k": 5}
    if active_where:
        search_kwargs["filter"] = active_where
    r_vec = vs.as_retriever(search_kwargs=search_kwargs)

    ensemble = WeightedEnsembleRetriever(
        retrievers=[bm25, r_vec],
        weights=[0.4, 0.6],
    )
    ranked = ensemble.invoke(question)

    seen: set[int] = set()
    unique_ranked: List[Document] = []
    for doc in ranked:
        ix = _int_chunk_index(doc.metadata)
        if ix < 0 or ix in seen:
            continue
        seen.add(ix)
        unique_ranked.append(doc)
        if len(unique_ranked) >= 5:
            break

    q_emb = embed_fn([question])[0]
    n_tot = len(documents)
    n_results = min(25, max(1, n_tot))
    q_kw: dict = {
        "query_embeddings": [q_emb],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if active_where:
        q_kw["where"] = active_where
    qres = collection.query(**q_kw)

    docs_q = (qres.get("documents") or [[]])[0]
    metas_q = (qres.get("metadatas") or [[]])[0]
    dists_q = (qres.get("distances") or [[]])[0]
    sim_map: dict[int, float] = {}
    for _d, meta, dist in zip(docs_q, metas_q, dists_q):
        ix = _int_chunk_index(meta)
        if ix < 0:
            continue
        sim_map[ix] = _distance_to_similarity(dist)

    passed: List[Tuple[str, int, float]] = []
    for doc in unique_ranked:
        ix = _int_chunk_index(doc.metadata)
        text = doc.page_content or ""
        sim = sim_map.get(ix)
        if sim is None:
            sim = _cosine_from_embeddings(embed_fn([text])[0], q_emb)
        if sim >= min_similarity:
            passed.append((text, ix, sim))

    if not passed:
        return [], []

    top3 = passed[:3]
    aux = [t[2] for t in passed[:5]]
    return top3, aux


def ask_llm(context: str, question: str) -> str:
    user_prompt = f"""Context:
{context}

Question: {question}

Answer:"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_RAG},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=1024,
    )
    msg = response.choices[0].message.content
    return (msg or "").strip()


def run_ask(
    collection: Any,
    chroma_client: Any,
    question: str,
    embed_fn: Callable[[List[str]], List[List[float]]],
    st_model: Any,
) -> AskResponse:
    top3, _aux_sims = retrieve_ensemble(
        collection,
        chroma_client,
        question,
        embed_fn,
        st_model,
    )

    if not top3:
        return AskResponse(
            answer="Not found in document",
            sources=[],
            confidence=0.0,
            confidence_label="low",
            guardrail_triggered=True,
            guardrail_reason="no_chunks_above_relevance_floor",
        )

    context_str = "\n\n".join(x[0] for x in top3)

    raw_answer = ask_llm(context_str, question)

    guardrail = False
    reason = None

    if raw_answer.strip() == "NOT_IN_DOCUMENT":
        guardrail = True
        reason = "not_in_document"
        raw_answer = "Not found in document"

    chunks_for_conf = [t[0] for t in top3]
    sims_for_conf = [t[2] for t in top3]
    confidence = compute_confidence(
        question,
        raw_answer,
        context_str,
        chunks_for_conf,
        sims_for_conf,
    )
    label = confidence_label_from_score(confidence)

    if label == "low":
        guardrail = True
        reason = reason or "low_confidence"
        final_answer = low_confidence_answer(confidence)
    elif guardrail:
        final_answer = raw_answer
    else:
        final_answer = raw_answer

    sources = [
        SourceSnippet(text=t[0][:2000], chunk_index=t[1], similarity=round(t[2], 4))
        for t in top3
    ]

    return AskResponse(
        answer=final_answer,
        sources=sources,
        confidence=round(confidence, 4),
        confidence_label=label,
        guardrail_triggered=guardrail,
        guardrail_reason=reason,
    )
