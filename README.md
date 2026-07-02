# meetingsGPT

![Python - 3.13](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Transcribe y resume vídeos de reuniones con Whisper + GPT.

Proyecto de [jorgesm](https://github.com/jorgesm).

## Requisitos

- Python 3.13+ y [uv](https://docs.astral.sh/uv/) para gestión de dependencias
- `ffmpeg` disponible en el `PATH`
- Una API key de OpenAI

## Instalación

```bash
uv sync
cp .env.example .env
# Edita .env y añade tu OPENAI_API_KEY
```

## Uso rápido

```bash
# Resumen con tipo por defecto (client)
python -m meetingsgpt run /ruta/al/video.mov

# Especificar tipo de resumen
python -m meetingsgpt run /ruta/al/video.mov --type internal

# Solo transcribir (sin resumen)
python -m meetingsgpt run /ruta/al/video.mov --transcribe-only

# Ver tipos de resumen disponibles
python -m meetingsgpt list-types
```

## Tipos de resumen

Configurables en `templates.yaml`:

| Tipo       | Descripción                                  |
|------------|----------------------------------------------|
| `client`   | Acta ejecutiva de seguimiento (por defecto)  |
| `internal` | Notas concisas para uso interno              |
| `detailed` | Acta formal completa con todo el detalle     |

Puedes agregar nuevos tipos editando `templates.yaml`.

## Parámetros

| Parámetro                  | Descripción                                       |
|----------------------------|---------------------------------------------------|
| `--type`, `-t`             | Tipo de resumen a generar                         |
| `--language`, `-l`         | Idioma de transcripción (defecto: `es`)           |
| `--model`, `-m`            | Modelo GPT para resumen (defecto: `gpt-5.4-mini`)|
| `--workers`, `-w`          | Hilos paralelos para transcripción (defecto: 4)  |
| `--no-post-process`        | Desactivar limpieza GPT de la transcripción       |
| `--no-timestamps`          | No incluir timestamps en la transcripción         |
| `--vocabulary`             | Fichero con vocabulario del dominio (uno por línea)|
| `--keep-intermediate`      | Mantener archivos intermedios                     |
| `--no-silence`             | Usar duración fija en vez de silencios            |
| `--chunk-ms`               | Duración de chunks en ms (defecto: 300000)        |
| `--overlap-ms`             | Solapamiento entre chunks (defecto: 5000)         |
| `--transcribe-only`        | Solo transcribir, sin resumir                     |

## Vocabulario del dominio

Crea un fichero `vocabulary.txt` (o pasa `--vocabulary ruta/al/fichero.txt`) con nombres propios, acrónimos y términos técnicos frecuentes para mejorar la transcripción:

```text
GPTadvisor
CNMV
```

## Salida

```text
outputs/20260326_video/
├── video_transcription.txt        # Con timestamps
├── video_transcription_clean.txt  # Post-procesada con GPT
└── video_summary_client.txt       # Resumen según template
```

## Arquitectura

```
meetingsgpt/
├── cli.py          # CLI (Typer)
├── config.py       # Configuración y templates (Pydantic)
├── pipeline.py     # Orquestador del pipeline
├── audio.py        # Extracción ffmpeg + chunking
├── transcribe.py   # Transcripción con OpenAI Whisper API
└── summarize.py    # Resumen map-reduce + post-procesamiento
```
