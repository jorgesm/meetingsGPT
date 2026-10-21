"""Audio extraction and chunking."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import split_on_silence
from rich.progress import Progress

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_MS = 10 * 60 * 1000

# Bump when the chunking algorithm changes, so cached chunks from the old one are discarded.
_CHUNKING_VERSION = 2


# ---------------------------------------------------------------------------
# Audio extraction (ffmpeg, no moviepy)
# ---------------------------------------------------------------------------


def extract_audio(input_path: Path, output_path: Path) -> Path:
    """Extract the audio track of a video or audio file to mp3 using ffmpeg.

    Returns the output audio path.
    """
    if output_path.exists():
        logger.info(f"Audio already exists at {output_path}. Skipping extraction.")
        return output_path

    logger.info(f"Extracting audio from {input_path} -> {output_path}")
    # Write to a temp file so an interrupted extraction never leaves a reusable partial mp3
    tmp_path = output_path.with_name(f"{output_path.stem}.partial.mp3")
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(input_path),
                "-vn",
                "-acodec",
                "libmp3lame",
                "-q:a",
                "2",
                "-y",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr[:500]}")
        tmp_path.replace(output_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    logger.info(f"Audio extracted to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Chunk manifest (cache validation)
# ---------------------------------------------------------------------------


def _get_audio_fingerprint(audio_path: Path) -> dict:
    stat = audio_path.stat()
    return {"mtime": stat.st_mtime, "size": stat.st_size}


def _load_manifest(chunk_folder: Path) -> dict | None:
    manifest_path = chunk_folder / "chunks.json"
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Error reading chunk manifest; chunks will be regenerated.")
        return None


def _save_manifest(
    chunk_folder: Path,
    audio_path: Path,
    chunk_count: int,
    chunk_ms: int,
    overlap_ms: int,
    use_silence: bool,
) -> None:
    manifest = {
        "version": _CHUNKING_VERSION,
        "audio": _get_audio_fingerprint(audio_path),
        "params": {
            "chunk_ms": chunk_ms,
            "overlap_ms": overlap_ms,
            "use_silence": use_silence,
        },
        "chunk_count": chunk_count,
    }
    try:
        (chunk_folder / "chunks.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    except Exception:
        logger.warning("Error saving chunk manifest.")


def _chunks_cache_valid(
    chunk_folder: Path,
    audio_path: Path,
    chunk_ms: int,
    overlap_ms: int,
    use_silence: bool,
) -> bool:
    manifest = _load_manifest(chunk_folder)
    if manifest is None:
        return False
    try:
        if manifest.get("version") != _CHUNKING_VERSION:
            logger.info("Chunking algorithm changed; regenerating chunks.")
            return False
        if _get_audio_fingerprint(audio_path) != manifest.get("audio", {}):
            logger.info("Audio changed; regenerating chunks.")
            return False
        params = manifest.get("params", {})
        if (
            params.get("chunk_ms") != chunk_ms
            or params.get("overlap_ms") != overlap_ms
            or params.get("use_silence") != use_silence
        ):
            logger.info("Chunking parameters changed; regenerating chunks.")
            return False
        expected = manifest.get("chunk_count", 0)
        actual = len(list(chunk_folder.glob("chunk_*.mp3")))
        if actual != expected:
            logger.info("Chunk count mismatch; regenerating.")
            return False
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Chunk splitting
# ---------------------------------------------------------------------------


def _fixed_windows(audio: AudioSegment, window_ms: int, overlap_ms: int) -> list[AudioSegment]:
    """Cut audio into consecutive windows of ``window_ms`` that overlap by ``overlap_ms``."""
    step = max(window_ms - overlap_ms, 1)
    windows: list[AudioSegment] = []
    start = 0
    while start < len(audio):
        end = min(start + window_ms, len(audio))
        windows.append(audio[start:end])
        if end == len(audio):
            break
        start += step
    return windows


def _merge_parts(
    parts: list[AudioSegment], max_ms: int, overlap_ms: int
) -> list[AudioSegment]:
    """Greedily join consecutive silence-split parts into chunks of at most ``max_ms``.

    Split points stay on silences, but chunks are long enough to give the transcription
    model context. A part longer than ``max_ms`` (no usable pause) is cut into
    overlapping fixed-duration windows instead.
    """
    pieces = [window for part in parts for window in _fixed_windows(part, max_ms, overlap_ms)]
    groups: list[list[AudioSegment]] = []
    group_ms = 0
    for piece in pieces:
        if groups and group_ms + len(piece) <= max_ms:
            groups[-1].append(piece)
            group_ms += len(piece)
        else:
            groups.append([piece])
            group_ms = len(piece)
    return [_concat(group) for group in groups]


def _concat(segments: list[AudioSegment]) -> AudioSegment:
    """Join same-format segments with one copy (``+`` copies the accumulated audio each time)."""
    first = segments[0]
    return AudioSegment(
        data=b"".join(segment.raw_data for segment in segments),
        sample_width=first.sample_width,
        frame_rate=first.frame_rate,
        channels=first.channels,
    )


def split_audio(
    audio_path: Path,
    chunk_folder: Path,
    *,
    chunk_ms: int = DEFAULT_CHUNK_MS,
    overlap_ms: int = 5000,
    use_silence: bool = True,
) -> list[Path]:
    """Split audio into chunks of at most ``chunk_ms``, cut on silences or at fixed duration.

    Audio no longer than ``chunk_ms`` becomes a single chunk, so the transcription model
    sees it whole.

    Returns list of chunk file paths.
    """
    chunk_folder.mkdir(parents=True, exist_ok=True)
    existing = sorted(chunk_folder.glob("chunk_*.mp3"))

    cache_valid = _chunks_cache_valid(chunk_folder, audio_path, chunk_ms, overlap_ms, use_silence)
    if existing and cache_valid:
        logger.info(f"Reusing {len(existing)} cached chunks.")
        return existing

    # Clear stale chunks along with their transcription caches, which are keyed by chunk name
    for old in [*existing, *chunk_folder.glob("chunk_*.json")]:
        old.unlink()

    logger.info(f"Splitting audio into chunks from {audio_path}")
    # Downmix before splitting: chunks are sent at this format anyway, and every copy
    # made while splitting and merging is ~5x smaller.
    audio = AudioSegment.from_file(str(audio_path)).set_channels(1).set_frame_rate(16000)

    if len(audio) <= chunk_ms:
        segments = [audio] if len(audio) else []
    elif use_silence:
        parts = split_on_silence(
            audio,
            min_silence_len=600,
            silence_thresh=audio.dBFS - 16,
            keep_silence=250,
        )
        segments = _merge_parts(parts, chunk_ms, overlap_ms)
    else:
        segments = _fixed_windows(audio, chunk_ms, overlap_ms)

    paths: list[Path] = []
    with Progress() as progress:
        task = progress.add_task("Exporting chunks...", total=len(segments))
        for i, segment in enumerate(segments, start=1):
            path = chunk_folder / f"chunk_{i:03d}.mp3"
            segment.export(str(path), format="mp3")
            paths.append(path)
            progress.update(task, advance=1)

    if paths:
        _save_manifest(chunk_folder, audio_path, len(paths), chunk_ms, overlap_ms, use_silence)

    logger.info(f"Audio split into {len(paths)} chunks.")
    return paths
