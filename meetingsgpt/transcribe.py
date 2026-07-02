"""Transcription backend: OpenAI Whisper API."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Protocol

from openai import OpenAI, OpenAIError
from rich.progress import Progress

from meetingsgpt.utils import retry_openai_call, split_sentences

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class TranscriptionBackend(Protocol):
    def transcribe(
        self,
        audio_path: Path,
        language: str,
        initial_prompt: str = "",
    ) -> list[str]: ...


# ---------------------------------------------------------------------------
# OpenAI Whisper API backend
# ---------------------------------------------------------------------------


class WhisperAPIBackend:
    """Transcription via the OpenAI Whisper API."""

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def transcribe(
        self,
        audio_path: Path,
        language: str,
        initial_prompt: str = "",
    ) -> list[str]:
        # Check per-chunk cache
        cache_path = audio_path.with_suffix(".json")
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                logger.warning(f"Error reading cache {cache_path}; re-transcribing.")

        def _call():
            with open(audio_path, "rb") as f:
                return self.client.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe",
                    file=f,
                    language=language,
                    response_format="text",
                    prompt=initial_prompt if initial_prompt else "",
                )

        try:
            result = retry_openai_call(_call)
            text = result.strip() if isinstance(result, str) else str(result).strip()
            segments = [text] if text else []

            try:
                cache_path.write_text(
                    json.dumps(segments, ensure_ascii=False), encoding="utf-8"
                )
            except Exception:
                logger.warning("Error saving transcription cache.")

            return segments
        except OpenAIError:
            logger.exception(f"API error transcribing {audio_path}")
            return []


# ---------------------------------------------------------------------------
# Orchestration: parallel chunk transcription
# ---------------------------------------------------------------------------


def _extract_last_sentences(text: str, n: int = 3) -> str:
    """Extract the last N complete sentences for context."""
    sentences = split_sentences(text.strip())
    return " ".join(sentences[-n:]) if sentences else text[-480:]


def transcribe_chunks(
    backend: TranscriptionBackend,
    chunks: list[Path],
    language: str,
    vocabulary: str = "",
    max_workers: int = 4,
) -> list[str]:
    """Transcribe a list of audio chunks in parallel and return unified segments."""
    if max_workers > 1 and len(chunks) > 1:
        return _transcribe_parallel(backend, chunks, language, vocabulary, max_workers)

    # Sequential fallback (single worker)
    all_segments: list[str] = []
    context = vocabulary
    with Progress() as progress:
        task = progress.add_task("Transcribing...", total=len(chunks))
        for chunk_path in chunks:
            segments = backend.transcribe(chunk_path, language, initial_prompt=context)
            all_segments.extend(segments)

            if segments:
                chunk_text = " ".join(segments)
                context = _extract_last_sentences(chunk_text)
                if vocabulary:
                    context = f"{vocabulary}. {context}"

            progress.update(task, advance=1)

    return all_segments


def _transcribe_parallel(
    backend: TranscriptionBackend,
    chunks: list[Path],
    language: str,
    vocabulary: str,
    max_workers: int,
) -> list[str]:
    """Parallel transcription for API backends."""
    results: list[list[str]] = [[] for _ in chunks]

    with Progress() as progress:
        task = progress.add_task("Transcribing in parallel...", total=len(chunks))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(backend.transcribe, chunk, language, vocabulary): i
                for i, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    logger.exception(f"Error transcribing chunk {idx + 1}")
                progress.update(task, advance=1)

    all_segments: list[str] = []
    for chunk_segments in results:
        all_segments.extend(chunk_segments)

    return all_segments
