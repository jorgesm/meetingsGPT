"""Transcription backend: OpenAI gpt-transcribe API."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from rich.progress import Progress

from meetingsgpt.utils import retry_openai_call, split_sentences

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenAI gpt-transcribe backend
# ---------------------------------------------------------------------------


class WhisperAPIBackend:
    """Transcription via the OpenAI gpt-transcribe API."""

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def transcribe(
        self,
        audio_path: Path,
        language: str,
        initial_prompt: str = "",
        keywords: list[str] | None = None,
    ) -> list[str]:
        # Per-chunk cache, valid only for the same request inputs
        cache_path = audio_path.with_suffix(".json")
        cache_key = {"language": language, "prompt": initial_prompt, "keywords": keywords or []}
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(cached, dict) and cached.get("key") == cache_key:
                    return cached["segments"]
            except Exception:
                logger.warning(f"Error reading cache {cache_path}; re-transcribing.")

        def _call():
            with open(audio_path, "rb") as f:
                return self.client.audio.transcriptions.create(
                    model="gpt-transcribe",
                    file=f,
                    languages=[language],
                    keywords=keywords or [],
                    response_format="json",
                    prompt=initial_prompt if initial_prompt else "",
                )

        result = retry_openai_call(_call)
        text = result.text.strip()
        segments = [text] if text else []

        try:
            cache = {"key": cache_key, "segments": segments}
            cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        except Exception:
            logger.warning("Error saving transcription cache.")

        return segments


# ---------------------------------------------------------------------------
# Orchestration: parallel chunk transcription
# ---------------------------------------------------------------------------


def _extract_last_sentences(text: str, n: int = 3) -> str:
    """Extract the last N complete sentences for context."""
    sentences = split_sentences(text.strip())
    return " ".join(sentences[-n:]) if sentences else text[-480:]


def transcribe_chunks(
    backend: WhisperAPIBackend,
    chunks: list[Path],
    language: str,
    vocabulary: list[str] | None = None,
    max_workers: int = 4,
) -> list[str]:
    """Transcribe a list of audio chunks in parallel and return unified segments."""
    vocabulary = vocabulary or []
    if max_workers > 1 and len(chunks) > 1:
        return _transcribe_parallel(backend, chunks, language, vocabulary, max_workers)

    # Sequential fallback (single worker)
    all_segments: list[str] = []
    context = ""
    with Progress() as progress:
        task = progress.add_task("Transcribing...", total=len(chunks))
        for i, chunk_path in enumerate(chunks, start=1):
            try:
                segments = backend.transcribe(
                    chunk_path, language, initial_prompt=context, keywords=vocabulary
                )
            except Exception:
                logger.exception(f"Error transcribing chunk {i}")
                raise
            all_segments.extend(segments)

            if segments:
                context = _extract_last_sentences(" ".join(segments))

            progress.update(task, advance=1)

    return all_segments


def _transcribe_parallel(
    backend: WhisperAPIBackend,
    chunks: list[Path],
    language: str,
    vocabulary: list[str],
    max_workers: int,
) -> list[str]:
    """Parallel transcription for API backends."""
    results: list[list[str]] = [[] for _ in chunks]
    failed: list[int] = []

    with Progress() as progress:
        task = progress.add_task("Transcribing in parallel...", total=len(chunks))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(backend.transcribe, chunk, language, "", vocabulary): i
                for i, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    logger.exception(f"Error transcribing chunk {idx + 1}")
                    failed.append(idx + 1)
                progress.update(task, advance=1)

    if failed:
        raise RuntimeError(f"Transcription failed for chunks {sorted(failed)}")

    all_segments: list[str] = []
    for chunk_segments in results:
        all_segments.extend(chunk_segments)

    return all_segments
