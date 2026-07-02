"""CLI interface using Typer."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

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
    help="Transcribe and summarize meeting videos.",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    video_path: Path = typer.Argument(..., help="Path to the video file to process."),
    language: str = typer.Option("es", "--language", "-l", help="Language code (es, en, ...)."),
    keep_intermediate: bool = typer.Option(
        False, "--keep-intermediate", help="Keep intermediate files."
    ),
    no_silence: bool = typer.Option(
        False, "--no-silence", help="Use fixed duration instead of silence detection."
    ),
    chunk_ms: int = typer.Option(
        5 * 60 * 1000, "--chunk-ms", help="Duration of each chunk in ms."
    ),
    overlap_ms: int = typer.Option(5000, "--overlap-ms", help="Overlap between chunks in ms."),
    summary_type: str | None = typer.Option(
        None, "--type", "-t", help="Summary type (see list-types)."
    ),
    transcribe_only: bool = typer.Option(
        False, "--transcribe-only", help="Only transcribe, skip summarization."
    ),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="GPT model for summarization."),
    max_workers: int = typer.Option(
        4, "--workers", "-w", help="Parallel threads for API transcription."
    ),
    no_post_process: bool = typer.Option(
        False, "--no-post-process", help="Disable GPT post-processing of the transcription."
    ),
    vocabulary: Path | None = typer.Option(
        None, "--vocabulary", help="File with domain vocabulary (one term per line)."
    ),
) -> None:
    """Process a meeting video: extract audio, transcribe, and summarize."""
    process(
        video_path,
        language=language,
        keep_intermediate=keep_intermediate,
        chunk_ms=chunk_ms,
        overlap_ms=overlap_ms,
        use_silence=not no_silence,
        transcribe_only=transcribe_only,
        summary_type=summary_type,
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
