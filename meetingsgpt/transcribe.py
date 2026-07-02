"""Transcription backend: OpenAI Whisper API."""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Protocol

from openai import OpenAI, OpenAIError
from rich.progress import Progress

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transcription result
# ---------------------------------------------------------------------------


class TranscriptionSegment:
    """A segment of transcription with optional timestamp."""

    def __init__(self, text: str, start: float | None = None, end: float | None = None):
        self.text = text
        self.start = start
        self.end = end

    def formatted(self, with_timestamps: bool = True) -> str:
        if with_timestamps and self.start is not None:
            ts = _format_timestamp(self.start)
            return f"[{ts}] {self.text}"
        return self.text


def _format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class TranscriptionBackend(Protocol):
    def transcribe(
        self,
        audio_path: Path,
        language: str,
        initial_prompt: str = "",
    ) -> list[TranscriptionSegment]: ...


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------


def _retry(fn, tries: int = 3, base: float = 1.5):
    for i in range(tries):
        try:
            return fn()
        except OpenAIError:
            if i == tries - 1:
                raise
            time.sleep(base**i)


# ---------------------------------------------------------------------------
# OpenAI Whisper API backend
# ---------------------------------------------------------------------------


class WhisperAPIBackend:
    """Transcription via OpenAI Whisper API with timestamps."""

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def transcribe(
        self,
        audio_path: Path,
        language: str,
        initial_prompt: str = "",
    ) -> list[TranscriptionSegment]:
        # Check per-chunk cache
        cache_path = audio_path.with_suffix(".json")
        if cache_path.exists():
            try:
                import json

                data = json.loads(cache_path.read_text(encoding="utf-8"))
                return [
                    TranscriptionSegment(s["text"], s.get("start"), s.get("end"))
                    for s in data
                ]
            except Exception:
                logger.warning(f"Error leyendo caché {cache_path}; re-transcribiendo.")

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
            result = _retry(_call)
            # gpt-4o-mini-transcribe returns plain text (no per-segment timestamps)
            # Timestamps are added at chunk level by the orchestrator
            text = result.strip() if isinstance(result, str) else str(result).strip()
            segments = [TranscriptionSegment(text=text)] if text else []

            # Save cache
            try:
                import json

                cache_data = [{"text": s.text, "start": s.start, "end": s.end} for s in segments]
                cache_path.write_text(
                    json.dumps(cache_data, ensure_ascii=False), encoding="utf-8"
                )
            except Exception:
                logger.warning("Error guardando caché de transcripción.")

            return segments
        except OpenAIError:
            logger.exception(f"Error de API transcribiendo {audio_path}")
            return []


# ---------------------------------------------------------------------------
# Orchestration: parallel chunk transcription
# ---------------------------------------------------------------------------


def _extract_last_sentences(text: str, n: int = 3) -> str:
    """Extract the last N complete sentences for context."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentences[-n:]) if sentences else text[-480:]


def transcribe_chunks(
    backend: TranscriptionBackend,
    chunks: list[Path],
    language: str,
    vocabulary: str = "",
    max_workers: int = 4,
    chunk_time_offsets: list[float] | None = None,
) -> list[TranscriptionSegment]:
    """Transcribe a list of audio chunks in parallel and return unified segments."""
    if max_workers > 1 and len(chunks) > 1:
        return _transcribe_parallel(
            backend, chunks, language, vocabulary, max_workers, chunk_time_offsets
        )

    # Sequential fallback (single worker)
    all_segments: list[TranscriptionSegment] = []
    context = vocabulary
    with Progress() as progress:
        task = progress.add_task("Transcribiendo...", total=len(chunks))
        for i, chunk_path in enumerate(chunks):
            offset = (chunk_time_offsets[i] if chunk_time_offsets else 0.0) or 0.0
            segments = backend.transcribe(chunk_path, language, initial_prompt=context)

            for seg in segments:
                if seg.start is not None:
                    seg.start += offset
                if seg.end is not None:
                    seg.end += offset

            all_segments.extend(segments)

            if segments:
                chunk_text = " ".join(s.text for s in segments)
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
    chunk_time_offsets: list[float] | None,
) -> list[TranscriptionSegment]:
    """Parallel transcription for API backends."""
    results: list[list[TranscriptionSegment]] = [[] for _ in chunks]

    with Progress() as progress:
        task = progress.add_task("Transcribiendo en paralelo...", total=len(chunks))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(backend.transcribe, chunk, language, vocabulary): i
                for i, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    segments = future.result()
                    offset = (chunk_time_offsets[idx] if chunk_time_offsets else 0.0) or 0.0
                    for seg in segments:
                        if seg.start is not None:
                            seg.start += offset
                        if seg.end is not None:
                            seg.end += offset
                    results[idx] = segments
                except Exception:
                    logger.exception(f"Error transcribiendo chunk {idx + 1}")
                progress.update(task, advance=1)

    all_segments: list[TranscriptionSegment] = []
    for chunk_segments in results:
        all_segments.extend(chunk_segments)

    return all_segments
