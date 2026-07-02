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


# ---------------------------------------------------------------------------
# Audio extraction (ffmpeg, no moviepy)
# ---------------------------------------------------------------------------


def extract_audio(video_path: Path, output_path: Path) -> Path:
    """Extract audio from video using ffmpeg directly.

    Returns the output audio path.
    """
    if output_path.exists():
        logger.info(f"Audio already exists at {output_path}. Skipping extraction.")
        return output_path

    logger.info(f"Extracting audio from {video_path} -> {output_path}")
    result = subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "libmp3lame",
            "-q:a",
            "2",
            "-y",
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[:500]}")

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


def split_audio(
    audio_path: Path,
    chunk_folder: Path,
    *,
    chunk_ms: int = 5 * 60 * 1000,
    overlap_ms: int = 5000,
    use_silence: bool = True,
) -> list[Path]:
    """Split audio into chunks by silence or fixed duration.

    Returns list of chunk file paths.
    """
    chunk_folder.mkdir(parents=True, exist_ok=True)
    existing = sorted(chunk_folder.glob("chunk_*.mp3"))

    cache_valid = _chunks_cache_valid(chunk_folder, audio_path, chunk_ms, overlap_ms, use_silence)
    if existing and cache_valid:
        logger.info(f"Reusing {len(existing)} cached chunks.")
        return existing

    # Clear stale chunks
    for old in existing:
        old.unlink()

    logger.info(f"Splitting audio into chunks from {audio_path}")
    audio = AudioSegment.from_file(str(audio_path))
    paths: list[Path] = []

    with Progress() as progress:
        if use_silence:
            task = progress.add_task("Splitting by silence...", total=None)
            parts = split_on_silence(
                audio,
                min_silence_len=600,
                silence_thresh=audio.dBFS - 16,
                keep_silence=250,
            )
            progress.update(task, total=len(parts))
            for i, part in enumerate(parts, start=1):
                part = part.set_channels(1).set_frame_rate(16000)
                path = chunk_folder / f"chunk_{i:03d}.mp3"
                part.export(str(path), format="mp3")
                paths.append(path)
                progress.update(task, advance=1)
        else:
            step = max(chunk_ms - overlap_ms, 1)
            total = len(audio)
            num_chunks = (total + step - 1) // step
            task = progress.add_task("Splitting by fixed duration...", total=num_chunks)
            i = 0
            start = 0
            while start < total:
                end = min(start + chunk_ms, total)
                chunk = audio[start:end].set_channels(1).set_frame_rate(16000)
                path = chunk_folder / f"chunk_{i + 1:03d}.mp3"
                chunk.export(str(path), format="mp3")
                paths.append(path)
                i += 1
                progress.update(task, advance=1)
                if end == total:
                    break
                start += step

    if paths:
        _save_manifest(chunk_folder, audio_path, len(paths), chunk_ms, overlap_ms, use_silence)

    logger.info(f"Audio split into {len(paths)} chunks.")
    return paths
