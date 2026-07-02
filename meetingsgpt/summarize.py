"""Text summarization with token-aware splitting."""

from __future__ import annotations

import logging

import tiktoken
from openai import OpenAI, OpenAIError
from rich.progress import Progress

from meetingsgpt.config import LANGUAGE_NAMES
from meetingsgpt.utils import retry_openai_call, split_sentences

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_MAX_TOKENS_PER_BLOCK = 3000


# ---------------------------------------------------------------------------
# Token-aware text splitting
# ---------------------------------------------------------------------------


def _split_by_tokens(
    text: str,
    max_tokens: int,
    model: str,
) -> list[str]:
    """Split text into blocks respecting sentence boundaries and token limits."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    # Split into paragraphs first, then sentences within paragraphs
    paragraphs = text.split("\n\n")
    blocks: list[str] = []
    current_block: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = len(enc.encode(para))

        if current_tokens + para_tokens <= max_tokens:
            current_block.append(para)
            current_tokens += para_tokens
        else:
            # If current block has content, save it
            if current_block:
                blocks.append("\n\n".join(current_block))
                current_block = []
                current_tokens = 0

            # If a single paragraph exceeds max_tokens, split by sentences
            if para_tokens > max_tokens:
                sentences = split_sentences(para)
                for sentence in sentences:
                    s_tokens = len(enc.encode(sentence))
                    if current_tokens + s_tokens <= max_tokens:
                        current_block.append(sentence)
                        current_tokens += s_tokens
                    else:
                        if current_block:
                            blocks.append(" ".join(current_block))
                        current_block = [sentence]
                        current_tokens = s_tokens
            else:
                current_block.append(para)
                current_tokens = para_tokens

    if current_block:
        blocks.append("\n\n".join(current_block))

    return blocks if blocks else [text]


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------


def summarize_text(
    text: str,
    system_prompt: str,
    *,
    api_key: str,
    model: str = DEFAULT_MODEL,
    max_tokens_per_block: int = DEFAULT_MAX_TOKENS_PER_BLOCK,
) -> str:
    """Summarize text using map-reduce with token-aware splitting.

    Args:
        text: Full transcription text to summarize.
        system_prompt: System prompt from the selected template.
        api_key: OpenAI API key.
        model: Model name for chat completions.
        max_tokens_per_block: Maximum tokens per block for splitting.
    """
    client = OpenAI(api_key=api_key)

    def _summarize_block(block: str) -> str:
        # Prompt kept in Spanish: the templates it composes with produce
        # Spanish-language business documents by design.
        user_prompt = f"Las notas de la última reunión son:\n{block}"

        def _call():
            return client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

        completion = retry_openai_call(_call)
        return completion.choices[0].message.content.strip()

    try:
        blocks = _split_by_tokens(text, max_tokens_per_block, model)

        if len(blocks) == 1:
            logger.info("Text fits in a single block. Summarizing directly.")
            return _summarize_block(blocks[0])

        # Map phase
        logger.info(f"Map-reduce: {len(blocks)} blocks.")
        partials: list[str] = []
        with Progress() as progress:
            task = progress.add_task("Summarizing blocks...", total=len(blocks))
            for block in blocks:
                partials.append(_summarize_block(block))
                progress.update(task, advance=1)

        # Reduce phase
        logger.info("Reduce phase: merging partial summaries.")
        final_input = "\n\n".join(partials)
        return _summarize_block(final_input)

    except OpenAIError:
        logger.exception("API error during summarization.")
        return ""
    except Exception:
        logger.exception("Unexpected error during summarization.")
        return ""


# ---------------------------------------------------------------------------
# Post-processing: clean up transcription with GPT
# ---------------------------------------------------------------------------


def post_process_transcription(
    text: str,
    *,
    api_key: str,
    language: str = "es",
    model: str = DEFAULT_MODEL,
) -> str:
    """Clean up the raw transcription using GPT.

    Fixes common issues: punctuation, capitalization, proper nouns,
    removes duplicate text from chunk overlaps.
    """
    client = OpenAI(api_key=api_key)

    lang_name = LANGUAGE_NAMES.get(language, language)

    # Prompt kept in Spanish: it edits Spanish-language meeting transcriptions.
    system_prompt = f"""Eres un editor de transcripciones de reuniones en {lang_name}.

Tu tarea es limpiar y mejorar la transcripción manteniendo el contenido EXACTO.

REGLAS ESTRICTAS:
- NO cambies el significado ni añadas información
- Corrige puntuación y capitalización
- Corrige errores obvios de transcripción automática (nombres propios, términos técnicos)
- Elimina texto duplicado (producto del solapamiento entre fragmentos de audio)
- Separa en párrafos lógicos por cambio de tema o hablante
- Si detectas cambios de hablante, márcalos con líneas separadas
- Mantén TODO el contenido original, solo mejora la forma"""

    blocks = _split_by_tokens(text, max_tokens=3000, model=model)

    def _clean_block(block: str) -> str:
        def _call():
            return client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": block},
                ],
            )

        return retry_openai_call(_call).choices[0].message.content.strip()

    if len(blocks) == 1:
        try:
            return _clean_block(text)
        except Exception:
            logger.warning("Error during post-processing; returning original transcription.")
            return text

    # Process blocks preserving order
    cleaned_parts: list[str] = []
    with Progress() as progress:
        task = progress.add_task("Post-processing transcription...", total=len(blocks))
        for block in blocks:
            try:
                cleaned_parts.append(_clean_block(block))
            except Exception:
                logger.warning("Error post-processing block; using original.")
                cleaned_parts.append(block)
            progress.update(task, advance=1)

    return "\n\n".join(cleaned_parts)
