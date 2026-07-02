"""Pipeline orchestrator: extract → chunk → transcribe → summarize → save."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path

from meetingsgpt.audio import extract_audio, split_audio
from meetingsgpt.config import (
    TemplatesRegistry,
    ensure_ffmpeg,
    get_openai_key,
    load_vocabulary,
)
from meetingsgpt.summarize import DEFAULT_MODEL, post_process_transcription, summarize_text
from meetingsgpt.transcribe import (
    TranscriptionSegment,
    WhisperAPIBackend,
    transcribe_chunks,
)

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.parent


def _get_recording_date(path: Path) -> str:
    """Return recording file date for output folder naming."""
    stat = path.stat()
    timestamp = getattr(stat, "st_birthtime", stat.st_mtime)
    return datetime.fromtimestamp(timestamp).strftime("%Y%m%d")


def _segments_to_text(segments: list[TranscriptionSegment], with_timestamps: bool) -> str:
    return "\n".join(seg.formatted(with_timestamps=with_timestamps) for seg in segments)


def _segments_to_plain_text(segments: list[TranscriptionSegment]) -> str:
    return " ".join(seg.text for seg in segments)


def process(
    video_path: str | Path,
    *,
    language: str = "es",
    keep_intermediate: bool = False,
    chunk_ms: int = 5 * 60 * 1000,
    overlap_ms: int = 5000,
    use_silence: bool = True,
    transcribe_only: bool = False,
    summary_type: str | None = None,
    templates: TemplatesRegistry | None = None,
    model: str = DEFAULT_MODEL,
    max_workers: int = 4,
    post_process: bool = True,
    timestamps: bool = True,
    vocabulary_path: Path | None = None,
) -> None:
    """Run the full meeting processing pipeline."""
    ensure_ffmpeg()

    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video no encontrado: {video_path}")

    # Output directory
    recording_date = _get_recording_date(video_path)
    output_stem = f"{recording_date}_{video_path.stem}"
    out_dir = SCRIPT_DIR / "outputs" / output_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_path = out_dir / f"{video_path.stem}.mp3"
    chunk_folder = out_dir / "chunks"

    try:
        # Step 1: Extract audio
        extract_audio(video_path, audio_path)

        # Step 2: Load vocabulary hints
        vocabulary = load_vocabulary(vocabulary_path)
        if vocabulary:
            logger.info(f"Vocabulario cargado: {vocabulary[:100]}...")

        # Step 3: Chunk and transcribe (parallel)
        api_key = get_openai_key()
        chunks = split_audio(
            audio_path,
            chunk_folder,
            chunk_ms=chunk_ms,
            overlap_ms=overlap_ms,
            use_silence=use_silence,
        )
        backend = WhisperAPIBackend(api_key=api_key)

        # Calculate time offsets for each chunk (approximate)
        chunk_time_offsets: list[float] | None = None
        if not use_silence:
            step_ms = max(chunk_ms - overlap_ms, 1)
            chunk_time_offsets = [i * step_ms / 1000.0 for i in range(len(chunks))]

        segments = transcribe_chunks(
            backend,
            chunks,
            language,
            vocabulary=vocabulary,
            max_workers=max_workers,
            chunk_time_offsets=chunk_time_offsets,
        )

        if not segments:
            logger.error("No se obtuvo transcripción.")
            return

        # Build text outputs
        transcription_with_ts = _segments_to_text(segments, with_timestamps=timestamps)
        transcription_plain = _segments_to_plain_text(segments)

        # Step 4: Post-process transcription
        if post_process and not transcribe_only:
            logger.info("Post-procesando transcripción con GPT...")
            api_key = get_openai_key()
            transcription_clean = post_process_transcription(
                transcription_plain,
                api_key=api_key,
                language=language,
                model=model,
            )
        else:
            transcription_clean = transcription_plain

        # Save transcriptions
        _save(out_dir, output_stem, "transcription", transcription_with_ts)
        if transcription_clean != transcription_plain:
            _save(out_dir, output_stem, "transcription_clean", transcription_clean)

        if transcribe_only:
            logger.info("Solo transcripción. Fin.")
            return

        # Step 5: Summarize
        if templates is None:
            from meetingsgpt.config import load_templates

            templates = load_templates()

        type_key, template = templates.get_template(summary_type, language=language)
        logger.info(f"Resumiendo con template '{template.name}' (modelo: {model})")

        api_key = get_openai_key()
        summary = summarize_text(
            transcription_clean,
            template.prompt,
            api_key=api_key,
            model=model,
        )

        if summary:
            _save(out_dir, output_stem, f"summary_{type_key}", summary)
            logger.info("Transcripción y resumen guardados correctamente.")
        else:
            logger.error("No se pudo generar el resumen.")

    except Exception:
        logger.exception("Error en el pipeline.")
    finally:
        _cleanup(out_dir, audio_path, chunk_folder, keep_intermediate)


def _save(out_dir: Path, stem: str, suffix: str, content: str) -> None:
    path = out_dir / f"{stem}_{suffix}.txt"
    try:
        path.write_text(content, encoding="utf-8")
        logger.info(f"{suffix} guardado en {path}")
    except Exception:
        logger.exception(f"Error guardando {suffix}")


def _cleanup(
    out_dir: Path,
    audio_path: Path,
    chunk_folder: Path,
    keep_intermediate: bool,
) -> None:
    if keep_intermediate:
        logger.info("Manteniendo archivos intermedios.")
        return
    try:
        if audio_path.exists():
            audio_path.unlink()
        if chunk_folder.exists():
            shutil.rmtree(chunk_folder)
    except Exception:
        logger.exception("Error durante limpieza.")
