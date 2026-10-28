#!/usr/bin/env python3
"""Fetch arXiv paper text (ar5iv, then PDF) into .cache/papers/<id>.txt."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import io
import logging
import re

import requests

from io_utils import atomic_write_text
from paths import PAPER_CACHE

log = logging.getLogger(__name__)

_HTTP_TIMEOUT = 60
_MAX_HTTP_BYTES = 50 * 1024 * 1024
_UA = {"User-Agent": "quantport/0.2 (+https://github.com/chris320211/QuantPort)"}
_ARXIV_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _limited_response(url: str) -> tuple[int, bytes] | None:
    try:
        with requests.get(url, timeout=_HTTP_TIMEOUT, headers=_UA, stream=True) as response:
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > _MAX_HTTP_BYTES:
                    log.warning("response too large for %s", url)
                    return None
                chunks.append(chunk)
            return response.status_code, b"".join(chunks)
    except requests.RequestException as e:
        log.warning("fetch failed for %s: %s", url, e)
        return None


def cache_path(arxiv_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", arxiv_id)
    return PAPER_CACHE / f"{safe}.txt"


def _strip_html(html: str) -> str:
    """Coarse HTML -> text. ar5iv markup is clean enough that this reads well."""
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</(p|div|h[1-6]|li|section|tr)>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", "", text)
    for ent, ch in (("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&#39;", "'")):
        text = text.replace(ent, ch)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _fetch_ar5iv(arxiv_id: str) -> str | None:
    url = f"https://ar5iv.org/abs/{arxiv_id}"
    result = _limited_response(url)
    if result is None:
        return None
    status, body = result
    text = body.decode("utf-8", errors="replace")
    if status == 200 and "<html" in text.lower():
        return _strip_html(text) or None
    return None


def _fetch_pdf_text(arxiv_id: str) -> str | None:
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    result = _limited_response(url)
    if result is None:
        return None
    status, body = result
    if status != 200:
        log.warning("arXiv PDF %s: HTTP %s", arxiv_id, status)
        return None
    try:
        import pypdf

        reader = pypdf.PdfReader(io.BytesIO(body))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text.strip() or None
    except Exception as e:  # noqa: BLE001 — pypdf raises a zoo of errors on odd PDFs
        log.warning("arXiv PDF parse failed for %s: %s", arxiv_id, e)
        return None


def fetch_paper_text(arxiv_id: str, *, use_cache: bool = True) -> str | None:
    """Return full plain text of an arXiv paper (cached to disk). None if unavailable.

    Does not extract GitHub URLs from the PDF or HTML.
    """
    cache = cache_path(arxiv_id)
    if use_cache and cache.exists():
        return cache.read_text(errors="replace")
    text = _fetch_ar5iv(arxiv_id) or _fetch_pdf_text(arxiv_id)
    if text:
        atomic_write_text(cache, text)
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description="Download arXiv paper text into .cache/papers/")
    parser.add_argument("--arxiv-id", required=True)
    args = parser.parse_args()
    arxiv_id = args.arxiv_id.strip()
    if not arxiv_id or not _ARXIV_ID_RE.fullmatch(arxiv_id):
        print(f"invalid arxiv id: {args.arxiv_id!r}", file=sys.stderr)
        return 1
    text = fetch_paper_text(arxiv_id)
    if not text:
        print(f"could not fetch arXiv paper {arxiv_id}", file=sys.stderr)
        return 1
    path = cache_path(arxiv_id)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
