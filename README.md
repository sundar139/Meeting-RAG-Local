# Meeting RAG Local

Local-first multimodal meeting analysis pipeline for turning meeting audio into searchable transcript evidence and grounded answers, with a demo-ready Streamlit experience.

## Value Proposition

- Runs fully on local infrastructure (PostgreSQL + pgvector + Ollama + local model stack).
- Produces traceable outputs from raw data through grounded RAG answers.
- Keeps retrieval and answer logic separate from the UI for maintainability.

## Architecture Overview

The system is organized as phase-aligned modules:

- Audio wrappers: transcription, alignment, diarization, GPU lifecycle.
- Data shaping: AMI parsing, attribution, turn building.
- Persistence + retrieval: PostgreSQL repository layer, pgvector search.
- RAG services: query rewrite, retrieval orchestration, grounded answer generation.
- App layer: Streamlit meeting browser + evidence-aware chat.
- Evaluation layer: transcript diagnostics + retrieval benchmark metrics.

See docs/architecture.md for full flow and 8 GB VRAM sequencing notes.

## Key Capabilities

- Parse AMI word-level XML into typed interim artifacts.
- Produce aligned transcript and diarization artifacts from meeting audio.
- Build canonical speaker turns and ingest embeddings into PostgreSQL.
- Retrieve semantically similar evidence chunks per meeting.
- Generate grounded structured answers with explicit insufficient-context handling.
- Browse meetings and run evidence-aware chat in Streamlit.
- Run lightweight, honest transcript/retrieval evaluation from CLI.

## Tech Stack

- Python 3.11
- uv for environment/dependency management
- PostgreSQL + pgvector
- WhisperX + pyannote for audio processing
- Ollama for embeddings and chat models
- Streamlit for local UI
- pytest, ruff, black, mypy for quality gates

## Repository Structure

- src/meeting_pipeline/: application package
- scripts/: operational CLIs for each pipeline stage
- tests/: unit and script-level tests
- migrations/: SQL schema migrations
- docs/: architecture/setup/dataset/evaluation docs
- data/: raw/interim/processed/eval artifacts

## Prerequisites

- Python 3.11
- uv
- PostgreSQL instance with pgvector available
- Ollama running locally
- For audio phases: CUDA-capable environment (recommended) and required model packages

## Environment Setup

1. Copy env template.

```powershell
copy .env.example .env
```

or

```bash
cp .env.example .env
```

2. Install base development dependencies.

```bash
uv sync --group dev
```

3. Install runtime extras as needed.

```bash
uv sync --group dev --extra data --extra services
uv sync --group dev --extra data --extra services --extra gpu
```

## Database Setup

1. Create database.

```bash
createdb meeting_pipeline
```

2. Run migrations.

```bash
uv run python scripts/run_migrations.py
```

## Dataset Setup

- Place AMI assets under data/raw/ami/.
- Expected XML pattern: <meeting_id>.<speaker_id>.words.xml.
- Example audio file used in commands below: data/raw/ami/ES2002a.Mix-Headset.wav.

## End-to-End Pipeline Workflow

Canonical order:

1. Migrate DB

```bash
uv run python scripts/run_migrations.py
```

2. Parse dataset (AMI XML)

```bash
uv run python scripts/parse_ami_xml.py --meeting-id ES2002a --input-dir data/raw/ami --output-dir data/interim
```

3. Transcribe + align

```bash
uv run python scripts/run_transcription.py --audio-path data/raw/ami/ES2002a.Mix-Headset.wav --meeting-id ES2002a --output-dir data/interim
```

4. Diarize

```bash
uv run python scripts/run_diarization.py --audio-path data/raw/ami/ES2002a.Mix-Headset.wav --meeting-id ES2002a --output-dir data/interim
```

5. Build canonical turns

```bash
uv run python scripts/build_turns.py --meeting-id ES2002a --aligned-path data/interim/ES2002a_aligned.json --diarization-path data/interim/ES2002a_diarization.json --output-dir data/processed
```

6. Ingest embeddings

```bash
uv run python scripts/ingest_embeddings.py --meeting-id ES2002a --turns-path data/processed/ES2002a_turns.json --replace-existing --batch-size 16
```

7. Optional smoke RAG run

```bash
uv run python scripts/smoke_rag.py --meeting-id ES2002a --question "What decisions were made?" --top-k 5
```

8. Launch app

```bash
uv run streamlit run src/meeting_pipeline/app/app.py
```

## Evaluation Workflow

Transcript diagnostics only:

```bash
uv run python run_eval.py --transcript-reference-path data/interim/ES2002a_ground_truth_words.json --transcript-prediction-path data/processed/ES2002a_turns.json
```

Retrieval benchmark with precomputed predictions:

```bash
uv run python run_eval.py --retrieval-benchmark-path data/eval/retrieval_benchmark.json --retrieval-predictions-path data/eval/retrieval_predictions.json --retrieval-top-k 5
```

Retrieval benchmark with live retrieval generation:

```bash
uv run python run_eval.py --retrieval-benchmark-path data/eval/retrieval_benchmark.json --retrieval-top-k 5 --live-retrieval
```

Write evaluation output:

```bash
uv run python run_eval.py --transcript-reference-path data/interim/ES2002a_ground_truth_words.json --transcript-prediction-path data/processed/ES2002a_turns.json --output-path data/eval/eval_summary.json
```

## Launching the Streamlit App

```bash
uv run streamlit run src/meeting_pipeline/app/app.py
```

App behavior:

- Sidebar meeting selection from ingested PostgreSQL data.
- Transcript browser with speaker filter and text filter.
- Grounded chat panel with rewritten query, structured answer, and evidence expansion.
- Meeting insights (chunk count, speakers, transcript span, recent questions).

## Demo Script (Recommended)

1. Start PostgreSQL and Ollama.
2. Confirm models:
   - ollama pull nomic-embed-text-v2-moe
   - ollama pull llama3.2:3b-instruct
3. Run steps 1-6 from End-to-End Workflow on ES2002a.
4. Run smoke_rag command once to validate retrieval/answer path.
5. Launch Streamlit and demo:
   - meeting selection
   - transcript filtering
   - grounded Q&A with evidence expansion

## Troubleshooting

- DB connection errors:
  - verify POSTGRES_* values in .env
  - verify server reachable and migrations applied
- Ollama unavailable/model-not-found:
  - start ollama serve
  - pull OLLAMA_MODEL and OLLAMA_CHAT_MODEL
- Empty app state:
  - ensure at least one meeting is ingested via ingest_embeddings
- GPU phase failures:
  - validate CUDA/driver environment
  - run uv run python scripts/check_gpu.py

## Limitations and Future Improvements

- Retrieval evaluation quality depends on benchmark fixture quality and coverage.
- Transcript diagnostics are lightweight and not a full ASR benchmark suite.
- Live retrieval evaluation depends on local DB/Ollama availability.
- UI is intentionally minimal and optimized for local demo reliability rather than multi-user deployment.
