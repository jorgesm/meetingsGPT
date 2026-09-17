"""Configuration and template loading."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.parent
TEMPLATES_PATH = SCRIPT_DIR / "templates.yaml"


# ---------------------------------------------------------------------------
# Template models
# ---------------------------------------------------------------------------


class TemplateConfig(BaseModel):
    name: str
    description: str = ""
    prompt: str


LANGUAGE_NAMES = {
    "es": "español",
    "en": "inglés",
    "fr": "francés",
    "de": "alemán",
    "pt": "portugués",
    "it": "italiano",
    "ca": "catalán",
    "gl": "gallego",
    "eu": "euskera",
}


class TemplatesRegistry(BaseModel):
    default: str
    base_rules: str = ""
    types: dict[str, TemplateConfig]

    def get_template(
        self, summary_type: str | None = None, language: str = "es"
    ) -> tuple[str, TemplateConfig]:
        """Return (type_key, template) with base_rules injected and language resolved."""
        key = summary_type or self.default
        if key not in self.types:
            available = list(self.types.keys())
            raise ValueError(f"Summary type '{key}' does not exist. Available: {available}")

        template = self.types[key]

        # Compose final prompt: base_rules + template-specific prompt
        lang_name = LANGUAGE_NAMES.get(language, language)
        base = self.base_rules.replace("{language}", lang_name) if self.base_rules else ""

        if base:
            final_prompt = f"{base.strip()}\n\n{template.prompt.strip()}"
        else:
            final_prompt = template.prompt

        composed = TemplateConfig(
            name=template.name,
            description=template.description,
            prompt=final_prompt,
        )
        return key, composed


def load_templates(path: Path = TEMPLATES_PATH) -> TemplatesRegistry:
    """Load and validate summary templates from YAML."""
    if not path.exists():
        raise FileNotFoundError(f"templates.yaml not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not raw:
        raise ValueError("templates.yaml is empty.")

    default_type = raw.pop("default", None)
    base_rules = raw.pop("base_rules", "")
    template_types: dict[str, TemplateConfig] = {}

    for name, cfg in raw.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"Type '{name}' must be a dictionary.")
        if "prompt" not in cfg or not cfg["prompt"].strip():
            raise ValueError(f"Type '{name}' must have a non-empty 'prompt' field.")
        template_types[name] = TemplateConfig(
            name=cfg.get("name", name),
            description=cfg.get("description", ""),
            prompt=cfg["prompt"],
        )

    if not template_types:
        raise ValueError("templates.yaml must define at least one summary type.")

    if default_type is None:
        default_type = next(iter(template_types))
        logger.warning(f"No 'default' defined in templates.yaml. Using '{default_type}'.")
    elif default_type not in template_types:
        raise ValueError(
            f"Default type '{default_type}' does not exist. "
            f"Available types: {list(template_types.keys())}"
        )

    return TemplatesRegistry(default=default_type, base_rules=base_rules, types=template_types)


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------


def ensure_ffmpeg() -> None:
    """Raise if ffmpeg is not installed."""
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError(
            "ffmpeg is not installed or not in PATH. "
            "Install it and make sure 'ffmpeg' is accessible."
        )


def get_openai_key() -> str:
    """Return the OpenAI API key, raising only when actually needed."""
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise EnvironmentError("OPENAI_API_KEY is not set in the environment variables.")
    return key


def load_vocabulary(path: Path | None = None) -> list[str]:
    """Load domain vocabulary hints from a file (one term per line).

    Returns a list of terms suitable for gpt-transcribe's `keywords` parameter.
    """
    if path is None:
        path = SCRIPT_DIR / "vocabulary.txt"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
