"""Shared helpers used across the transcription and summarization modules."""

from __future__ import annotations

import re
import time

from openai import OpenAIError

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def retry_openai_call(fn, tries: int = 3, base: float = 1.5):
    """Retry an OpenAI API call with exponential backoff on OpenAIError."""
    if tries < 1:
        raise ValueError("tries must be >= 1")
    for i in range(tries):
        try:
            return fn()
        except OpenAIError:
            if i == tries - 1:
                raise
            time.sleep(base**i)
    raise AssertionError("unreachable")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences at period/question/exclamation boundaries."""
    return [p for p in _SENTENCE_BOUNDARY.split(text) if p.strip()]
