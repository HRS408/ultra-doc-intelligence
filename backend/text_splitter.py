"""
RecursiveCharacterTextSplitter — same parameters and strategy as LangChain's
(langchain-text-splitters): recursive separators + merge with overlap.

In-repo copy avoids langchain-core on Python 3.14 (pydantic v1 compatibility issues).
"""

from __future__ import annotations

import logging
import re
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


def _split_text_with_regex(text: str, separator: str) -> List[str]:
    if separator:
        splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class RecursiveCharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        length_function: Callable[[str], int] = len,
        is_separator_regex: bool = False,
        strip_whitespace: bool = True,
    ) -> None:
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be <= chunk_size ({chunk_size})"
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._separators = separators or ["\n\n", "\n", " ", ""]
        self._is_separator_regex = is_separator_regex
        self._strip_whitespace = strip_whitespace

    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        text = separator.join(docs)
        if self._strip_whitespace:
            text = text.strip()
        return None if text == "" else text

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        separator_len = self._length_function(separator)
        docs: List[str] = []
        current_doc: List[str] = []
        total = 0
        for d in splits:
            _len = self._length_function(d)
            if total + _len + (separator_len if len(current_doc) > 0 else 0) > self._chunk_size:
                if total > self._chunk_size:
                    logger.warning(
                        "Created a chunk of size %s, longer than chunk_size %s",
                        total,
                        self._chunk_size,
                    )
                if current_doc:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    while total > self._chunk_overlap or (
                        total + _len + (separator_len if len(current_doc) > 0 else 0)
                        > self._chunk_size
                        and total > 0
                    ):
                        total -= self._length_function(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators: List[str] = []
        for i, _s in enumerate(separators):
            _sep = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_sep, text):
                separator = _s
                new_separators = separators[i + 1 :]
                break

        _sep = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(text, _sep)

        good: List[str] = []
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                good.append(s)
            else:
                if good:
                    final_chunks.extend(self._merge_splits(good, separator))
                    good = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    final_chunks.extend(self._split_text(s, new_separators))
        if good:
            final_chunks.extend(self._merge_splits(good, separator))
        return final_chunks

    def split_text(self, text: str) -> List[str]:
        return self._split_text(text, self._separators)
