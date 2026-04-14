"""Document parsing, section-aware chunking, embedding, and ChromaDB storage."""

import hashlib
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import chromadb
import fitz
from docx import Document as DocxDocument

from backend.section_chunking import section_chunks_from_page_text

STORAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage", "chroma")

# (prefixed_text, page_num, section_name, doc_type, load_id, field_name)
ChunkRow = Tuple[str, int, str, str, str, str]

KV_FIELDS = [
    "load_id",
    "ship_date",
    "delivery_date",
    "po_number",
    "freight_charges",
    "cod",
    "cod_value",
    "shipper",
    "consignee",
    "units",
    "weight",
    "description",
]

FIELD_LABELS = {
    "load_id": "Load ID",
    "ship_date": "Ship Date",
    "delivery_date": "Delivery Date",
    "po_number": "PO Number",
    "freight_charges": "Freight Charges",
    "cod": "COD",
    "cod_value": "COD Value",
    "shipper": "Shipper",
    "consignee": "Consignee",
    "units": "Units",
    "weight": "Weight",
    "description": "Description",
}


def stable_doc_id(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()[:16]


def _clean_text_value(value: str) -> str:
    value = re.sub(r"\s+", " ", value or "").strip()
    value = value.strip(" |:-,\t\r\n")
    return value


def _normalize_text(text: str) -> str:
    lines = [_clean_text_value(line) for line in (text or "").splitlines()]
    return "\n".join(line for line in lines if line)


def _clean_block_text(text: str) -> str:
    return _clean_text_value(" ".join(line for line in (text or "").splitlines() if line.strip()))


def _page_text_from_blocks(page: fitz.Page) -> str:
    blocks = []
    for block in page.get_text("blocks") or []:
        if len(block) < 5:
            continue
        x0, y0, _x1, _y1, text = block[:5]
        clean = _clean_block_text(text)
        if clean:
            blocks.append((float(y0), float(x0), clean))

    blocks.sort(key=lambda b: (b[0], b[1]))

    rows: List[List[Tuple[float, str]]] = []
    current: List[Tuple[float, str]] = []
    current_y: Optional[float] = None
    y_tolerance = 4.0

    for y0, x0, text in blocks:
        if current_y is None or abs(y0 - current_y) <= y_tolerance:
            current.append((x0, text))
            current_y = y0 if current_y is None else (current_y + y0) / 2
            continue
        rows.append(current)
        current = [(x0, text)]
        current_y = y0

    if current:
        rows.append(current)

    lines = []
    for row in rows:
        row.sort(key=lambda r: r[0])
        lines.append(" | ".join(text for _, text in row))
    return "\n".join(lines)


def _segments_from_pdf(file_bytes: bytes) -> List[Tuple[int, str]]:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        out: List[Tuple[int, str]] = []
        for i in range(len(doc)):
            text = _page_text_from_blocks(doc[i])
            if text.strip():
                out.append((i + 1, text))
        return out
    finally:
        doc.close()


def _segments_from_docx(file_bytes: bytes) -> List[Tuple[int, str]]:
    from io import BytesIO

    d = DocxDocument(BytesIO(file_bytes))
    parts = [p.text for p in d.paragraphs if p.text and p.text.strip()]
    text = "\n".join(parts)
    return [(1, text)] if text.strip() else []


def _segments_from_txt(file_bytes: bytes) -> List[Tuple[int, str]]:
    text = file_bytes.decode("utf-8", errors="replace")
    return [(0, text)] if text.strip() else []


def parse_document(file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
    """
    Returns list of (page_num, text) segments. page_num is 1-based for PDF/DOCX, 0 for TXT.
    """
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return _segments_from_pdf(file_bytes)
    if lower.endswith(".docx"):
        return _segments_from_docx(file_bytes)
    if lower.endswith(".txt"):
        return _segments_from_txt(file_bytes)
    raise ValueError(f"Unsupported file type: {filename}")


def _regex_value(text: str, pattern: str, flags: int = re.IGNORECASE) -> Optional[str]:
    match = re.search(pattern, text, flags)
    if not match:
        return None
    return _clean_text_value(match.group(1))


def _extract_party_names(text: str) -> Tuple[Optional[str], Optional[str]]:
    match = re.search(
        r"(?is)\bShipper\b\s*(?:\|\s*)?\bConsignee\b(?P<body>.*?)(?:\b3rd Party Billing\b|\bTransportation Company\b|# Of Units)",
        text,
    )
    if not match:
        return None, None

    body = match.group("body")
    numbered = [
        _clean_text_value(m.group(1))
        for m in re.finditer(r"\b1\.\s*([^,\n|]+)", body)
        if _clean_text_value(m.group(1))
    ]
    shipper = numbered[0] if len(numbered) >= 1 else None
    consignee = numbered[1] if len(numbered) >= 2 else None
    return shipper, consignee


def _extract_commodity_fields(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    match = re.search(
        r"(?is)#\s*Of\s*Units.*?Description\s*Of\s*The\s*Commodity.*?Weight.*?\n(?P<row>.+)",
        text,
    )
    if not match:
        return out

    row = _clean_text_value(match.group("row"))
    parts = [_clean_text_value(p) for p in row.split("|") if _clean_text_value(p)]
    if len(parts) >= 3:
        out["units"] = parts[0].lstrip("#")
        out["description"] = parts[1]
        out["weight"] = parts[2]
        return out

    fallback = re.search(r"#?\s*([0-9,]+)\s+(.+?)\s+([0-9,]+(?:\.\d+)?\s*(?:lbs?|pounds?|kg)?)\b", row, re.I)
    if fallback:
        out["units"] = _clean_text_value(fallback.group(1))
        out["description"] = _clean_text_value(fallback.group(2))
        out["weight"] = _clean_text_value(fallback.group(3))
    return out


def extract_kv_pairs(text: str) -> Dict[str, str]:
    """Regex-based field extraction for logistics/BOL documents."""
    normalized = _normalize_text(text)
    kv: Dict[str, str] = {}

    patterns = {
        "load_id": r"\bLoad\s*ID\s*(?:[:|])?\s*([A-Z0-9][A-Z0-9_-]*)",
        "ship_date": r"\bShip\s*Date\s*(?:[:|])?\s*([0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{2,4}(?:\s+[0-9]{1,2}:[0-9]{2})?)",
        "delivery_date": r"\bDelivery\s*Date\s*(?:[:|])?\s*([0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{2,4}(?:\s+[0-9]{1,2}:[0-9]{2})?)",
        "po_number": r"\bPO\s*Number(?:\s+Pickup)?\s*(?:[:|])?\s*([A-Z0-9][A-Z0-9_-]*)",
        "freight_charges": r"\bFreight\s*Charges\s*(?:[:|])?\s*([A-Za-z]+)",
        "cod": r"(?im)^\s*COD\s*(?:[:|])?\s*([A-Za-z]+)\s*$",
        "cod_value": r"\bCOD\s*Value\b(?:(?!\bConsignor\s+name\b)[\s\S])*?(\$?\s*[0-9,.]+(?:\s*[A-Z]{3})?)",
    }

    for field, pattern in patterns.items():
        value = _regex_value(normalized, pattern)
        if value:
            kv[field] = value

    shipper, consignee = _extract_party_names(normalized)
    if shipper:
        kv["shipper"] = shipper
    if consignee:
        kv["consignee"] = consignee

    kv.update(_extract_commodity_fields(normalized))

    for field in KV_FIELDS:
        if field in kv:
            kv[field] = _clean_text_value(kv[field])
    return kv


def build_chunk_rows(
    segments: List[Tuple[int, str]],
    doc_type: str,
    load_id: str,
) -> List[ChunkRow]:
    """Section-aware chunks with prefix injected for embedding."""
    dt = (doc_type or "general").strip() or "general"
    lid = (load_id or "").strip()
    rows: List[ChunkRow] = []
    all_text = "\n".join(seg_text for _, seg_text in segments)
    kv = extract_kv_pairs(all_text)
    resolved_lid = kv.get("load_id") or lid

    summary_lines = [
        f"{FIELD_LABELS[field]}: {kv[field]}"
        for field in KV_FIELDS
        if kv.get(field)
    ]
    if summary_lines:
        summary_text = "\n".join(summary_lines)
        prefixed = f"[{dt.upper()}] [Load: {resolved_lid}] [STRUCTURED SUMMARY]\n{summary_text}"
        rows.append((prefixed, 0, "STRUCTURED SUMMARY", dt, resolved_lid, ""))

    for field in KV_FIELDS:
        value = kv.get(field)
        if not value:
            continue
        label = FIELD_LABELS[field]
        atomic_text = f"[FIELD] {label}: {value}"
        prefixed = f"[{dt.upper()}] [Load: {resolved_lid}] [FIELD] [{label}]\n{atomic_text}"
        rows.append((prefixed, 0, label, dt, resolved_lid, field))

    for page_num, seg_text in segments:
        for section_name, raw in section_chunks_from_page_text(seg_text):
            prefixed = f"[{dt.upper()}] [Load: {resolved_lid}] [{section_name}]\n{raw}"
            sec_meta = section_name[:500] if len(section_name) > 500 else section_name
            rows.append((prefixed, page_num, sec_meta, dt, resolved_lid, ""))
    return rows


def get_chroma_client() -> chromadb.PersistentClient:
    os.makedirs(STORAGE_PATH, exist_ok=True)
    return chromadb.PersistentClient(path=STORAGE_PATH)


def embed_and_store(
    doc_id: str,
    source_filename: str,
    chunk_rows: List[ChunkRow],
    embed_fn: Callable[[List[str]], List[List[float]]],
    chroma_client: chromadb.PersistentClient,
) -> int:
    """
    Wipes collection doc_{doc_id} if present, then indexes all chunks.
    """
    collection_name = f"doc_{doc_id}"
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    if not chunk_rows:
        return 0

    texts = [c[0] for c in chunk_rows]
    embeddings = embed_fn(texts)
    ids = [f"{doc_id}_{i}" for i in range(len(chunk_rows))]
    metadatas: List[Any] = []
    for i, (_, page_num, section_name, doc_type, load_id, field_name) in enumerate(chunk_rows):
        metadatas.append(
            {
                "doc_id": doc_id,
                "chunk_index": i,
                "source_filename": source_filename,
                "page_num": int(page_num),
                "doc_type": str(doc_type),
                "load_id": str(load_id),
                "section_name": str(section_name),
                "field_name": str(field_name),
            }
        )

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    return len(chunk_rows)
