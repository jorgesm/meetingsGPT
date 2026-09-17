# meetingsGPT

![Python - 3.13+](https://img.shields.io/badge/Python-3.13%2B-blue?logo=python&logoColor=white)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A CLI tool that turns meeting recordings into clean, ready-to-share summaries: extract audio, transcribe it, and generate a structured write-up — all through a single command.

## Features

- **One command, full pipeline**: video in, transcription + summary out
- **Fast, parallel transcription** via OpenAI's `gpt-transcribe`, chunked and processed concurrently
- **Customizable summary templates** (`client`, `internal`, `detailed`, or your own) defined in a single YAML file — no code changes needed to add a new one
- **Smart caching**: safe to re-run — already-processed audio chunks and transcriptions aren't redone (or re-billed)
- **Domain vocabulary hints** to improve accuracy on proper nouns, acronyms, and jargon

## Requirements

- Python 3.13+ and [uv](https://docs.astral.sh/uv/) for dependency management
- `ffmpeg` available on `PATH`
- An OpenAI API key — note that running this tool calls the OpenAI API for both transcription and summarization, which incurs usage costs on your account

## Installation

```bash
git clone https://github.com/jorgesm/meetingsGPT.git
cd meetingsGPT
uv sync
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Quick usage

Options go right after `run`, with the video path last — handy when you're swapping only the path in a command you keep reusing:

```bash
# Summary with the default type (client)
uv run meetingsgpt run /path/to/video.mov

# Specify a summary type (path stays as the last, easy-to-swap token)
uv run meetingsgpt run --type internal /path/to/video.mov

# Transcribe only (no summary)
uv run meetingsgpt run --transcribe-only /path/to/video.mov

# List available summary types
uv run meetingsgpt list-types
```

## Summary types

Configurable in `templates.yaml`:

| Type       | Description                              |
|------------|-------------------------------------------|
| `client`   | Executive follow-up minutes (default)     |
| `internal` | Concise notes for internal use            |
| `detailed` | Full formal minutes with all the detail   |

You can add new types by editing `templates.yaml`.

## Parameters

| Parameter              | Description                                            |
|------------------------|---------------------------------------------------------|
| `--type`, `-t`         | Summary type to generate                                |
| `--language`, `-l`     | Transcription language (default: `es`)                  |
| `--model`, `-m`        | GPT model for summarization (default: `gpt-5.6-luna`)   |
| `--workers`, `-w`      | Parallel threads for transcription (default: 4)         |
| `--no-post-process`    | Disable GPT cleanup of the transcription                |
| `--vocabulary`         | File with domain vocabulary (one term per line)         |
| `--keep-intermediate`  | Keep intermediate files                                 |
| `--no-silence`         | Use fixed duration instead of silence detection         |
| `--chunk-ms`           | Chunk duration in ms (default: 300000)                  |
| `--overlap-ms`         | Overlap between chunks in ms (default: 5000)            |
| `--transcribe-only`    | Only transcribe, skip summarization                     |

Run `uv run meetingsgpt run --help` for the full, always-up-to-date list.

## Domain vocabulary

Create a `vocabulary.txt` file (or pass `--vocabulary path/to/file.txt`) with proper nouns, acronyms, and technical terms to improve transcription accuracy:

```text
OpenAI
Whisper
```

## Output

```text
outputs/20260326_video/
├── video_transcription.txt        # Raw transcription
├── video_transcription_clean.txt  # GPT post-processed
└── video_summary_client.txt       # Summary per template
```

The folder is named after the video file's own creation date, not the date you run the tool.

## Architecture

```
meetingsgpt/
├── cli.py          # CLI (Typer)
├── config.py       # Configuration and templates (Pydantic)
├── pipeline.py     # Pipeline orchestrator
├── audio.py        # ffmpeg extraction + chunking
├── transcribe.py   # Transcription via OpenAI gpt-transcribe API
├── summarize.py    # Map-reduce summarization + post-processing
└── utils.py        # Shared retry/sentence-splitting helpers
```

## License

[MIT](LICENSE)
