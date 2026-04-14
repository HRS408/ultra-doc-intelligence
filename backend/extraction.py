"""Structured field extraction via Groq JSON completion."""

import json
import os
import re
from typing import Any

from groq import Groq

from backend.schemas import ExtractResponse

MODEL = "llama-3.3-70b-versatile"

_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=_api_key) if _api_key else Groq(api_key="dummy")

SYSTEM_EXTRACT = """Extract shipment fields from this logistics document. Return ONLY a
valid JSON object with exactly these keys. Use null for any missing field.
Do not add commentary, markdown, or extra keys."""


def _gather_context(collection: Any, max_chars: int = 8000) -> str:
    data = collection.get(include=["documents", "metadatas"])
    docs = data.get("documents") or []
    metas = data.get("metadatas") or []
    pairs = list(zip(docs, metas))
    def _chunk_key(item: tuple) -> int:
        m = item[1] or {}
        v = m.get("chunk_index", 0)
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0

    pairs.sort(key=_chunk_key)
    parts: list[str] = []
    total = 0
    for doc, _meta in pairs:
        piece = doc or ""
        if total >= max_chars:
            break
        remaining = max_chars - total
        chunk = piece[:remaining]
        parts.append(chunk)
        total += len(chunk)
    return "\n\n".join(parts)


def _call_extract(full_context: str, retry: bool) -> str:
    user_base = f"""Document:
{full_context}

Extract these fields as JSON:
shipment_id, shipper, consignee, pickup_datetime, delivery_datetime,
equipment_type, mode, rate (number), currency, weight (number), carrier_name

pickup_datetime and delivery_datetime in ISO 8601 if possible, else raw string."""
    if retry:
        user_prompt = 'Return ONLY raw JSON, no markdown fences.\n\n' + user_base
    else:
        user_prompt = user_base
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_EXTRACT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=1024,
    )
    msg = response.choices[0].message.content
    return (msg or "").strip()


def _parse_json_payload(text: str) -> dict:
    t = text.strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", t, re.IGNORECASE)
    if fence:
        t = fence.group(1).strip()
    return json.loads(t)


def extract_structured(collection: Any) -> ExtractResponse:
    full_context = _gather_context(collection, 8000)
    if not full_context.strip():
        return ExtractResponse()

    raw = _call_extract(full_context, retry=False)
    try:
        data = _parse_json_payload(raw)
    except json.JSONDecodeError:
        raw = _call_extract(full_context, retry=True)
        try:
            data = _parse_json_payload(raw)
        except json.JSONDecodeError as e:
            raise ValueError("Model returned invalid JSON after retry") from e

    return ExtractResponse(
        shipment_id=data.get("shipment_id"),
        shipper=data.get("shipper"),
        consignee=data.get("consignee"),
        pickup_datetime=data.get("pickup_datetime"),
        delivery_datetime=data.get("delivery_datetime"),
        equipment_type=data.get("equipment_type"),
        mode=data.get("mode"),
        rate=_coerce_float(data.get("rate")),
        currency=data.get("currency"),
        weight=_coerce_float(data.get("weight")),
        carrier_name=data.get("carrier_name"),
    )


def _coerce_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
