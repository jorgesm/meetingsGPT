# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

meetingsGPT is a CLI tool that transcribes and summarizes meeting videos: extract audio (ffmpeg) → chunk it → transcribe via OpenAI's `gpt-4o-mini-transcribe` → optionally clean up the transcription with GPT → summarize with a template-driven prompt. Built with Typer, dependency-managed with `uv`.

## Commands

```bash
# Install
uv sync
cp .env.example .env   # add OPENAI_API_KEY

# Run the pipeline
uv run python -m meetingsgpt run /path/to/video.mov
uv run python -m meetingsgpt run /path/to/video.mov --type internal --language en
uv run python -m meetingsgpt run /path/to/video.mov --transcribe-only

# List available summary types (works without an API key)
uv run python -m meetingsgpt list-types

# Lint
uv run ruff check meetingsgpt/
uv run ruff check --fix meetingsgpt/
```

There is no test suite yet — there's no `tests/` directory despite the package having enough surface area (chunking, token splitting, template composition) to warrant one. When adding tests, add pytest as a dev dependency and place fixtures under `tests/data/` as small (1-2s) audio clips.

## Architecture

```
meetingsgpt/
├── cli.py          # Typer entry point: `run` and `list-types` commands, logging setup, load_dotenv()
├── config.py       # Pydantic models for templates.yaml, env/ffmpeg checks, vocabulary loading
├── pipeline.py     # Orchestrator: extract → chunk → transcribe → post-process → summarize → save
├── audio.py        # ffmpeg extraction (subprocess, no moviepy), pydub chunking + chunk cache manifest
├── transcribe.py   # WhisperAPIBackend, parallel chunk transcription (ThreadPoolExecutor)
├── summarize.py    # tiktoken-based block splitting, map-reduce summarization, GPT post-processing
└── utils.py        # Shared `retry_openai_call` (exponential backoff) and `split_sentences`
```

Everything flows through `pipeline.process()`. Each output goes to `outputs/<YYYYMMDD>_<video-stem>/`, where the date is the recording file's birth/mtime, not the run date.

### Templates (`templates.yaml`)

- `base_rules` is a shared preamble (with a `{language}` placeholder) injected before every per-type prompt by `TemplatesRegistry.get_template()` in `config.py`.
- Each type (`client`, `internal`, `detailed`) has `name`, `description`, `prompt`.
- Prompts are written in Spanish by design — they instruct GPT to produce Spanish-language business documents. Keep this convention when adding types unless the user explicitly wants another output language.
- Adding a new summary type only requires editing `templates.yaml`; no code changes needed.

### Two-level caching

- **Chunk cache**: `audio.py` writes `chunks.json` in the chunk folder with the audio fingerprint (mtime+size) and chunking params. If unchanged, `split_audio()` reuses existing chunk files instead of re-splitting.
- **Transcription cache**: `transcribe.py`'s `WhisperAPIBackend` caches each chunk's transcription as `<chunk>.json` next to the chunk, keyed only by the chunk file itself — reruns skip the API call if the cache file exists.
- Both caches are wiped by default at the end of a run (`pipeline._cleanup`) unless `--keep-intermediate` is passed.

### Transcription context continuity

Whisper has no memory across chunks. `transcribe.py` carries context forward between chunks: `_extract_last_sentences()` pulls the tail of the previous chunk's output and feeds it (prefixed with domain vocabulary) as `initial_prompt` to the next chunk — but only in the sequential path (single worker). The parallel path (`_transcribe_parallel`, used whenever `max_workers > 1`) has no cross-chunk context since chunks are transcribed concurrently; it passes only the static vocabulary as `initial_prompt`.

### Post-processing vs. summarization are separate GPT calls

`post_process_transcription()` cleans the raw transcription (punctuation, dedup from chunk overlap) while explicitly preserving all content. `summarize_text()` is a distinct map-reduce pass over the (cleaned) transcription using the template's prompt as the system prompt. Both use the same `_split_by_tokens()` helper from `summarize.py` to stay under `DEFAULT_MAX_TOKENS_PER_BLOCK` (3000), splitting on paragraph/sentence boundaries via `utils.split_sentences()`.

## Conventions

- Ruff, line-length 100, rules `E`, `F`, `I`, `W` (see `pyproject.toml`).
- English everywhere: code, identifiers, comments, CLI help/log/error strings, commit messages. Exception: `templates.yaml`'s prompt text (see above) is intentionally Spanish.
- Type hints on public functions; PEP 8, 4-space indent; snake_case modules, PascalCase classes.
- API keys and ffmpeg presence are validated lazily (`get_openai_key()`, `ensure_ffmpeg()`), only when actually needed — this is why `list-types` works with no `.env` configured.
