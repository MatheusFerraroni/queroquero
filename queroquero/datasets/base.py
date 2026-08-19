from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Callable, Dict, List, Optional, Protocol


@dataclass(frozen=True)
class Document:
    text: str
    source_ref: str
    source_position: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScanResult:
    documents: List[Document]
    source_fingerprint: Dict[str, Any]
    metrics: Dict[str, int]
    cursor: Dict[str, Any] = field(default_factory=dict)
    extra_reports: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    resume_cursor: Optional[Dict[str, Any]] = None


CheckpointCallback = Callable[[Dict[str, Any], List[Document]], None]


class Adapter(Protocol):
    def scan(
        self,
        config: Dict[str, Any],
        resume_cursor: Optional[Dict[str, Any]] = None,
        resume_documents: Optional[List[Document]] = None,
        checkpoint: Optional[CheckpointCallback] = None,
    ) -> ScanResult: ...


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self._ignored_depth = 0

    def handle_data(self, data: str) -> None:
        if self._ignored_depth == 0:
            self.parts.append(data)

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        del attrs
        if tag in {"script", "style"}:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag in {"br", "p", "div", "li", "tr", "blockquote"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1
            return
        if self._ignored_depth:
            return
        if tag in {"p", "div", "li", "tr", "blockquote"}:
            self.parts.append("\n")


_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>|<!--")
_HORIZONTAL_SPACE_RE = re.compile(r"[^\S\n]+")
_EXCESS_NEWLINES_RE = re.compile(r"\n{3,}")


def clean_text(value: str, strip_html: bool = True) -> str:
    if not isinstance(value, str):
        raise TypeError("text must be a string")
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL_RE.sub("", text)
    if strip_html and _HTML_TAG_RE.search(text):
        parser = _TextExtractor()
        try:
            parser.feed(text)
            parser.close()
            text = "".join(parser.parts)
        except Exception:
            text = html.unescape(text)
    else:
        text = html.unescape(text)
    # Entity decoding and HTML parsing can introduce decomposed characters or
    # non-ASCII control codes, so normalize and filter once more afterwards.
    text = unicodedata.normalize("NFC", text)
    text = "".join(
        character
        for character in text
        if character in {"\n", "\t"} or unicodedata.category(character) != "Cc"
    )
    text = _HORIZONTAL_SPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _EXCESS_NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def stable_hash(*parts: Any) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def safe_source_hash(source_ref: str) -> str:
    return hashlib.sha256(source_ref.encode("utf-8")).hexdigest()
