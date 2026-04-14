"""Confidence scoring and context checks for RAG answers."""

import os
from typing import Optional

from groq import Groq

MODEL = "llama-3.3-70b-versatile"

_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=_api_key) if _api_key else Groq(api_key="dummy")

VERIFY_PROMPT = """You are verifying an answer based on context.

Context:
{context}

Question:
{question}

Answer:
{answer}

Does the answer EXACTLY exist in the context?

Respond with only one:
HIGH
MEDIUM
LOW
"""


def confidence_label_from_score(confidence: float) -> str:
    if confidence >= 0.70:
        return "high"
    if confidence >= 0.45:
        return "medium"
    return "low"


def _llm_verification_score(question: str, answer: str, context: str) -> float:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": VERIFY_PROMPT.format(
                    context=context,
                    question=question,
                    answer=answer,
                ),
            }
        ],
        temperature=0.0,
        max_tokens=8,
    )
    verdict = ((response.choices[0].message.content or "").strip().upper())
    if verdict == "HIGH":
        return 0.2
    if verdict == "MEDIUM":
        return 0.1
    return 0.0


def compute_confidence(
    question: str,
    answer: str,
    context: str,
    chunks: list,
    similarities: list,
) -> float:
    """Score confidence from answer-evidence alignment rather than raw similarity alone."""
    answer_norm = (answer or "").strip().lower()
    context_norm = (context or "").strip().lower()
    chunk_norms = [(chunk or "").strip().lower() for chunk in chunks]

    if not answer_norm:
        return 0.0

    # Primary signal: exact answer text in retrieved context is direct grounding.
    base_score = 0.6 if answer_norm in context_norm else 0.2

    # Critical structured-field signal: exact answers in atomic [FIELD] chunks are highly reliable.
    field_bonus = 0.0
    for chunk in chunk_norms:
        if "[field]" in chunk and answer_norm in chunk:
            field_bonus = 0.3
            break

    # Secondary signal: retrieval similarity contributes, but cannot dominate confidence.
    sim_score = max(similarities) if similarities else 0.0
    sim_contrib = 0.2 * max(0.0, min(1.0, float(sim_score)))

    # Faithfulness check: avoid the extra LLM call when exact field grounding is already present.
    llm_score = 0.0
    if base_score < 0.6 or field_bonus == 0:
        llm_score = _llm_verification_score(question, answer, context)

    confidence = base_score + field_bonus + sim_contrib + llm_score
    return min(confidence, 0.95)


def confidence_gate(
    confidence: float,
    confidence_label: str,
    guardrail_already: bool,
) -> Tuple[bool, Optional[str]]:
    """
    Returns (guardrail_triggered, reason) when low confidence requires blocking.
    If guardrail_already True, preserve that state.
    """
    if guardrail_already:
        return True, "retrieval_or_llm_guardrail"
    if confidence_label == "low":
        return True, "low_confidence"
    return False, None


def context_coverage_check(question: str, context: str, min_context_chars: int = 80) -> bool:
    """Heuristic: enough retrieved text vs empty or trivial context."""
    if not context or len(context.strip()) < min_context_chars:
        return False
    if not question or len(question.strip()) < 2:
        return False
    return True


def low_confidence_answer(confidence: float) -> str:
    return (
        "I cannot confidently answer this from the document. "
        f"(Confidence: {confidence:.0%}). Try rephrasing or check the source."
    )
