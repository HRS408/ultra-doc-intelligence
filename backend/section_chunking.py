"""Section-aware chunking: headers (ALL-CAPS or trailing ':'), min 50 / max 600, inner recursive split."""

from __future__ import annotations

from typing import List, Tuple

from backend.text_splitter import RecursiveCharacterTextSplitter

_INNER = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""],
)


def _is_colon_header(line: str) -> bool:
    s = line.strip()
    return len(s) >= 2 and s.endswith(":")


def _is_all_caps_header(line: str) -> bool:
    s = line.strip()
    if len(s) < 3:
        return False
    if not any(c.isalpha() for c in s):
        return False
    return s == s.upper()


def _is_section_header(line: str) -> bool:
    return _is_colon_header(line) or _is_all_caps_header(line)


def _header_name(line: str) -> str:
    s = line.strip()
    if s.endswith(":"):
        name = s[:-1].strip()
        return name or "SECTION"
    return s or "SECTION"


def _split_into_sections(text: str) -> List[Tuple[str, str]]:
    """Split text into (section_title, body) by header lines."""
    lines = text.splitlines()
    sections: List[Tuple[str, str]] = []
    current_name = "GENERAL"
    current_lines: List[str] = []

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        sections.append((current_name, body))

    for line in lines:
        if _is_section_header(line):
            flush()
            current_name = _header_name(line)
            current_lines = []
        else:
            current_lines.append(line)
    flush()

    out = [(n, b) for n, b in sections if b.strip()]
    if not out and text.strip():
        return [("GENERAL", text.strip())]
    return out


def _merge_min_length(pieces: List[Tuple[str, str]], min_len: int = 50) -> List[Tuple[str, str]]:
    if not pieces:
        return []
    merged: List[Tuple[str, str]] = []
    buf_n, buf_b = pieces[0]
    for name, body in pieces[1:]:
        if len(buf_b) < min_len:
            buf_b = (buf_b + "\n" + body).strip()
            buf_n = f"{buf_n}+{name}"
        else:
            merged.append((buf_n, buf_b))
            buf_n, buf_b = name, body
    merged.append((buf_n, buf_b))
    return merged


def section_chunks_from_page_text(page_text: str) -> List[Tuple[str, str]]:
    """
    Returns (section_name, raw_chunk_text) without the doc-type/load prefix.
    Rules: section by headers; each section → one chunk unless >600 chars (then recursive split);
    merge chunks under 50 characters with the following chunk.
    """
    pieces: List[Tuple[str, str]] = []
    for sec_name, body in _split_into_sections(page_text):
        b = body.strip()
        if not b:
            continue
        if len(b) > 600:
            for part in _INNER.split_text(b):
                p = part.strip()
                if p:
                    pieces.append((sec_name, p))
        else:
            pieces.append((sec_name, b))

    pieces = _merge_min_length(pieces, 50)

    final: List[Tuple[str, str]] = []
    for name, body in pieces:
        if len(body) > 600:
            for part in _INNER.split_text(body):
                p = part.strip()
                if p:
                    final.append((name, p))
        else:
            final.append((name, body))
    return final
