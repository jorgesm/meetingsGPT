# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

meetingsGPT is a CLI tool that transcribes and summarizes meeting videos: extract audio (ffmpeg) → chunk it → transcribe via OpenAI's `gpt-transcribe` → optionally clean up the transcription with GPT → summarize with a template-driven prompt. Built with Typer, dependency-managed with `uv`.

## Commands

```bash
# Install
uv sync
cp .env.example .env   # add OPENAI_API_KEY

# Run the pipeline (options before the path, so the path is the last, easy-to-swap token)
uv run meetingsgpt run /path/to/video.mov
uv run meetingsgpt run --type internal --language en /path/to/video.mov
uv run meetingsgpt run --transcribe-only /path/to/video.mov
uv run meetingsgpt transcribe /path/to/audio.opus   # audio/voice notes: transcribe + GPT cleanup, no summary

# List available summary types (works without an API key)
uv run meetingsgpt list-types

# Tests
uv run pytest

# Lint
uv run ruff check meetingsgpt/
uv run ruff check --fix meetingsgpt/
```

Tests live in `tests/` (pytest, dev dependency) and currently cover only chunk merging; token splitting and template composition are still untested. Build audio inputs synthetically with `pydub` (e.g. `pydub.generators.Sine`); if a real recording is needed, keep it under `tests/data/` as a small (1-2s) clip.

## Architecture

```
meetingsgpt/
├── cli.py          # Typer entry point: `run`, `transcribe` and `list-types` commands, logging setup, load_dotenv()
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

- **Chunk cache**: `audio.py` writes `chunks.json` in the chunk folder with the audio fingerprint (mtime+size), chunking params and `_CHUNKING_VERSION`. If unchanged, `split_audio()` reuses existing chunk files instead of re-splitting; otherwise it deletes the old chunks and their transcription caches.
- **Transcription cache**: `transcribe.py`'s `WhisperAPIBackend` caches each chunk's transcription as `<chunk>.json` next to the chunk, keyed only by the chunk file itself — reruns skip the API call if the cache file exists.
- Both caches are wiped by default at the end of a successful run (`pipeline._cleanup`) unless `--keep-intermediate` is passed. A failed run keeps them, so a rerun only re-transcribes the chunks that failed. A chunk whose API call fails after retries fails the whole run rather than leaving a silent gap in the transcription.

### Chunking

Chunks are as long as `--chunk-ms` allows (`DEFAULT_CHUNK_MS`, 10 min): fewer cuts means more context for `gpt-transcribe`, which noticeably improves consistency of names and terms. Audio no longer than that is sent whole (voice notes). Longer audio is split on silences and the resulting parts are merged back up to `chunk_ms` (`_merge_parts()`), so every cut lands on a pause; a stretch with no usable pause longer than `chunk_ms` is cut at fixed duration.

### Transcription context continuity

`gpt-transcribe` has no memory across chunks. `transcribe.py` carries context forward between chunks: `_extract_last_sentences()` pulls the tail of the previous chunk's output and feeds it as the `prompt` param to the next chunk — but only in the sequential path (single worker). The parallel path (`_transcribe_parallel`, used whenever `max_workers > 1`) has no cross-chunk context since chunks are transcribed concurrently; it passes an empty `prompt`. Domain vocabulary (`config.load_vocabulary()`) is independent of this continuity context — it's always sent via the `keywords` param on every chunk, in both paths.

### Post-processing vs. summarization are separate GPT calls

`post_process_transcription()` cleans the raw transcription (punctuation, dedup from chunk overlap) while explicitly preserving all content. `summarize_text()` is a distinct map-reduce pass over the (cleaned) transcription using the template's prompt as the system prompt. Both use the same `_split_by_tokens()` helper from `summarize.py` to stay under `DEFAULT_MAX_TOKENS_PER_BLOCK` (3000), splitting on paragraph/sentence boundaries via `utils.split_sentences()`, and share the `_chat_call()` / `_map_blocks_with_progress()` helpers for the actual chat-completion calls and progress reporting.

## Conventions

- Ruff, line-length 100, rules `E`, `F`, `I`, `W` (see `pyproject.toml`).
- English everywhere: code, identifiers, comments, CLI help/log/error strings, commit messages. Exception: `templates.yaml`'s prompt text (see above) is intentionally Spanish.
- Type hints on public functions; PEP 8, 4-space indent; snake_case modules, PascalCase classes.
- API keys and ffmpeg presence are validated lazily (`get_openai_key()`, `ensure_ffmpeg()`), only when actually needed — this is why `list-types` works with no `.env` configured.
