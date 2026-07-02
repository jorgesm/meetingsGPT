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
    help="Transcribe y resume vídeos de reuniones.",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    video_path: Path = typer.Argument(..., help="Ruta al fichero de vídeo a procesar."),
    language: str = typer.Option("es", "--language", "-l", help="Código de idioma (es, en, ...)."),
    keep_intermediate: bool = typer.Option(
        False, "--keep-intermediate", help="Mantener archivos intermedios."
    ),
    no_silence: bool = typer.Option(
        False, "--no-silence", help="Usar duración fija en vez de silencios."
    ),
    chunk_ms: int = typer.Option(
        5 * 60 * 1000, "--chunk-ms", help="Duración de cada chunk en ms."
    ),
    overlap_ms: int = typer.Option(5000, "--overlap-ms", help="Solapamiento entre chunks en ms."),
    summary_type: str | None = typer.Option(
        None, "--type", "-t", help="Tipo de resumen (ver list-types)."
    ),
    transcribe_only: bool = typer.Option(
        False, "--transcribe-only", help="Solo transcribir, sin resumir."
    ),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="Modelo GPT para resumen."),
    max_workers: int = typer.Option(
        4, "--workers", "-w", help="Hilos paralelos para transcripción API."
    ),
    no_post_process: bool = typer.Option(
        False, "--no-post-process", help="Desactivar post-procesamiento GPT de transcripción."
    ),
    no_timestamps: bool = typer.Option(
        False, "--no-timestamps", help="No incluir timestamps en la transcripción."
    ),
    vocabulary: Path | None = typer.Option(
        None, "--vocabulary", help="Fichero con vocabulario del dominio (uno por línea)."
    ),
) -> None:
    """Procesa un vídeo de reunión: extrae audio, transcribe y resume."""
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
        timestamps=not no_timestamps,
        vocabulary_path=vocabulary,
    )


@app.command("list-types")
def list_types() -> None:
    """Muestra los tipos de resumen disponibles."""
    try:
        templates = load_templates()
        table = Table(title="Tipos de resumen disponibles")
        table.add_column("Clave", style="cyan")
        table.add_column("Nombre", style="green")
        table.add_column("Descripción")
        table.add_column("Default", justify="center")

        for key, config in templates.types.items():
            is_default = "✓" if key == templates.default else ""
            table.add_row(key, config.name, config.description, is_default)

        console.print(table)
    except Exception as e:
        console.print(f"[red]Error cargando templates:[/red] {e}")
