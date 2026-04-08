# Windows Setup

## Prerequisites

1. Install Python 3.11.
2. Install uv.
3. Install PostgreSQL and ensure pgvector is available.
4. Install Ollama and verify it can run locally.
5. For audio/model stages, install GPU drivers and CUDA runtime.

## Project Initialization

1. Copy environment template.

```powershell
copy .env.example .env
```

1. Install baseline dependencies.

```bash
uv sync --group dev
```

1. Install runtime extras for the full pipeline.

```bash
uv sync --group dev --extra data --extra services --extra gpu
```

## Runtime Services

1. Start PostgreSQL and create database.

```bash
createdb meeting_pipeline
```

1. Run DB migrations.

```bash
uv run python scripts/run_migrations.py
```

1. Start Ollama and pull required models.

```bash
ollama serve
ollama pull nomic-embed-text-v2-moe
ollama pull qwen3:4b
```

## Retrieval Chunking Settings

Set these in `.env` (or keep defaults):

```bash
RETRIEVAL_CHUNK_WINDOW_SECONDS=45
RETRIEVAL_CHUNK_OVERLAP_SECONDS=15
```

If you change these values for existing meetings, rerun ingestion with replacement so stored
embeddings are rebuilt with the new retrieval chunk windows.

## Verification Checklist

```bash
uv run python scripts/check_gpu.py
uv run ruff check .
uv run black --check .
uv run mypy src
uv run pytest
```

## Common Pitfalls

- Wrong Python interpreter: always run commands through `uv run`.
- DB auth/host mismatch: ensure `.env` values match PostgreSQL instance.
- Missing Ollama model pulls: retrieval/answer steps fail if models are absent.
- Missing FFmpeg on PATH: install FFmpeg (for example `winget install --id Gyan.FFmpeg -e`) and open a new shell before running transcription.
- CUDA/CuDNN DLL errors on Windows GPU runs: if you hit `cudnn_ops_infer64_8.dll` errors, install matching NVIDIA CUDA/CuDNN runtimes or run `scripts/run_transcription.py` with `--device cpu`.
- GPU OOM on 8 GB VRAM: run transcription and diarization sequentially, not in parallel.
- Empty app meetings list: ingestion step was skipped or failed.
