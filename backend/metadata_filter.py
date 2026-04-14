"""Question-based Chroma metadata pre-filter for doc_type."""

from __future__ import annotations

from typing import Dict, Optional

_CARRIER_KWS = ("carrier", "driver", "mc number", "truck")
_SHIPPER_KWS = ("customer rate", "shipper rate", "agreed")


def doc_type_prefilter_where(question: str) -> Optional[Dict[str, str]]:
    """
    If the question matches carrier vs shipper cues, restrict retrieval to that doc_type.
    Carrier keywords are checked first; shipper second.
    """
    q = question.lower()
    if any(kw in q for kw in _CARRIER_KWS):
        return {"doc_type": "carrier_rc"}
    if any(kw in q for kw in _SHIPPER_KWS):
        return {"doc_type": "shipper_rc"}
    return None
