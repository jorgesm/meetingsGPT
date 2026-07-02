# CLAUDE.md

## Project Overview
meetingsGPT transcribes and summarizes meeting videos using OpenAI Whisper API + GPT. CLI tool built with Typer, modular Python package.

## Quick Commands
```bash
# Run on a video
python -m meetingsgpt run /path/to/video.mov

# List summary types
python -m meetingsgpt list-types

# Lint
uv run ruff check meetingsgpt/
```

## Architecture
```
meetingsgpt/
├── cli.py          # Typer CLI entry point, logging setup, load_dotenv()
├── config.py       # Pydantic models, template loading (base_rules + per-type prompt), vocabulary
├── pipeline.py     # Orchestrator: extract → chunk → transcribe → post-process → summarize → save
├── audio.py        # ffmpeg extraction, pydub chunking (silence/fixed), chunk cache manifest
├── transcribe.py   # WhisperAPIBackend with parallel chunk transcription
└── summarize.py    # Token-aware splitting (tiktoken), map-reduce summarization, GPT post-processing
```

## Key Design Decisions
- **Templates**: `templates.yaml` has `base_rules` (injected into all templates with `{language}` placeholder) + per-type prompts. `config.py:get_template()` composes the final prompt.
- **Transcription**: OpenAI `gpt-4o-mini-transcribe` with parallel chunk processing (ThreadPoolExecutor).
- **No moviepy**: Audio extraction via direct ffmpeg subprocess call.
- **API key validated lazily**: Only when actually needed, so `list-types` works without credentials.
- **Caching**: Chunk manifest with audio fingerprint (mtime+size) + chunking params. Per-chunk transcription cache as .json files.

## Models
- Summarization/post-processing: `gpt-5.4-mini` (configurable via `--model`)
- Transcription: `gpt-4o-mini-transcribe`

## Dependencies
Managed with `uv`. Key deps: `typer`, `rich`, `pydantic`, `openai`, `tiktoken`, `pydub`, `pyyaml`.

## Style
- Ruff with line-length 100, select E/F/I/W
- Spanish for user-facing messages, English for code/docstrings
- Type hints on public functions
- PEP 8, 4-space indent; modules in snake_case, classes in PascalCase
- Run `ruff check --fix` before opening a PR

## Testing
- Tests live in `tests/`, named `test_<feature>.py`
- Use small fixtures (1-2s audio clips) under `tests/data/`
- Cover: chunking, token-aware splitting, config/templates, pipeline with mocked API calls

## Commits & PRs
- Short, imperative commit summaries; body focused on motivation
- PRs should include testing evidence, affected configs, and output snippets if behavior changed
