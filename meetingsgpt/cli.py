"""CLI interface using Typer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from meetingsgpt.audio import DEFAULT_CHUNK_MS
from meetingsgpt.config import load_templates
from meetingsgpt.pipeline import process
from meetingsgpt.summarize import DEFAULT_MODEL

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
for noisy in ("openai", "httpx"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger(noisy).propagate = False

app = typer.Typer(
    name="meetingsgpt",
    help="Transcribe and summarize meeting recordings (video or audio).",
    add_completion=False,
)
console = Console()


Language = Annotated[
    str, typer.Option("--language", "-l", help="Language code (es, en, ...).")
]
KeepIntermediate = Annotated[
    bool, typer.Option("--keep-intermediate", help="Keep intermediate files.")
]
NoSilence = Annotated[
    bool, typer.Option("--no-silence", help="Use fixed duration instead of silence detection.")
]
ChunkMs = Annotated[int, typer.Option("--chunk-ms", help="Max duration of each chunk in ms.")]
OverlapMs = Annotated[
    int, typer.Option("--overlap-ms", help="Overlap between fixed-duration chunks in ms.")
]
Workers = Annotated[
    int, typer.Option("--workers", "-w", help="Parallel threads for API transcription.")
]
NoPostProcess = Annotated[
    bool,
    typer.Option("--no-post-process", help="Disable GPT post-processing of the transcription."),
]
Vocabulary = Annotated[
    Path | None,
    typer.Option("--vocabulary", help="File with domain vocabulary (one term per line)."),
]


@app.command()
def run(
    input_path: Annotated[
        Path, typer.Argument(help="Path to the recording to process (video or audio).")
    ],
    language: Language = "es",
    keep_intermediate: KeepIntermediate = False,
    no_silence: NoSilence = False,
    chunk_ms: ChunkMs = DEFAULT_CHUNK_MS,
    overlap_ms: OverlapMs = 5000,
    summary_type: Annotated[
        str | None, typer.Option("--type", "-t", help="Summary type (see list-types).")
    ] = None,
    transcribe_only: Annotated[
        bool, typer.Option("--transcribe-only", help="Only transcribe, skip summarization.")
    ] = False,
    model: Annotated[
        str, typer.Option("--model", "-m", help="GPT model for summarization.")
    ] = DEFAULT_MODEL,
    max_workers: Workers = 4,
    no_post_process: NoPostProcess = False,
    vocabulary: Vocabulary = None,
) -> None:
    """Process a meeting recording: extract audio, transcribe, and summarize."""
    process(
        input_path,
        language=language,
        keep_intermediate=keep_intermediate,
        chunk_ms=chunk_ms,
        overlap_ms=overlap_ms,
        use_silence=not no_silence,
        transcribe_only=transcribe_only,
        summary_type=summary_type,
        model=model,
        max_workers=max_workers,
        post_process=not no_post_process and not transcribe_only,
        vocabulary_path=vocabulary,
    )


@app.command()
def transcribe(
    input_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the audio or video file (any ffmpeg-readable format: opus, m4a, ...)."
        ),
    ],
    language: Language = "es",
    model: Annotated[
        str, typer.Option("--model", "-m", help="GPT model for post-processing.")
    ] = DEFAULT_MODEL,
    max_workers: Workers = 4,
    no_post_process: NoPostProcess = False,
    vocabulary: Vocabulary = None,
    keep_intermediate: KeepIntermediate = False,
    no_silence: NoSilence = False,
    chunk_ms: ChunkMs = DEFAULT_CHUNK_MS,
    overlap_ms: OverlapMs = 5000,
) -> None:
    """Transcribe a recording (e.g. a voice note) with GPT cleanup, without summarizing it."""
    process(
        input_path,
        language=language,
        keep_intermediate=keep_intermediate,
        chunk_ms=chunk_ms,
        overlap_ms=overlap_ms,
        use_silence=not no_silence,
        transcribe_only=True,
        model=model,
        max_workers=max_workers,
        post_process=not no_post_process,
        vocabulary_path=vocabulary,
    )


@app.command("list-types")
def list_types() -> None:
    """Show the available summary types."""
    try:
        templates = load_templates()
        table = Table(title="Available summary types")
        table.add_column("Key", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Description")
        table.add_column("Default", justify="center")

        for key, config in templates.types.items():
            is_default = "✓" if key == templates.default else ""
            table.add_row(key, config.name, config.description, is_default)

        console.print(table)
    except Exception as e:
        console.print(f"[red]Error loading templates:[/red] {e}")
