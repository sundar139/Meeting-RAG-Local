# Meeting RAG Local
Local-first meeting intelligence pipeline that turns meeting audio into searchable transcript evidence and grounded Q&A.

## Hero Summary
- Privacy-preserving by design: runs with local PostgreSQL, local Ollama models, and local artifact storage.
- End-to-end pipeline from AMI assets and audio to canonical turns, retrieval chunks, and evidence-backed answers.
- Practical stack for local RAG systems: Python + PostgreSQL/pgvector + Ollama + Streamlit.
- Evaluation-oriented workflow with explicit artifacts, reproducible scripts, and measurable retrieval/transcript diagnostics.

## Feature Highlights
- AMI word-level XML parsing into normalized JSON artifacts.
- Transcription/alignment and diarization wrappers for timed lexical + speaker signals.
- Canonical speaker turn construction for stable downstream retrieval behavior.
- Retrieval chunk generation with configurable window/overlap and deterministic chunk keys.
- Embedding ingestion into PostgreSQL + pgvector (single meeting and batch workflows).
- Adaptive retrieval modes plus grounded answer generation with confidence-tier outputs.
- Streamlit interface for transcript browsing and evidence-aware Q&A.
- Operational scripts for readiness checks, smoke tests, benchmarks, and evaluation.

## Architecture Overview
The system separates artifact generation from retrieval so each stage is inspectable and testable.

```text
AMI XML + audio (.wav)
        |
        | scripts/parse_ami_xml.py
        v
data/interim/<meeting_id>_ground_truth_words.json

audio (.wav) --------------------------------------+
        |                                           |
        | scripts/run_transcription.py              | scripts/run_diarization.py
        v                                           v
data/interim/<meeting_id>_aligned.json     data/interim/<meeting_id>_diarization.json
                       \                     /
                        \                   /
                         +-- scripts/build_turns.py --+
                                                   |
                                                   v
                               data/processed/<meeting_id>_turns.json
                                                   |
                                                   | scripts/ingest_embeddings.py
                                                   | scripts/ingest_many_meetings.py
                                                   v
                                     PostgreSQL + pgvector (meeting_transcripts)
                                                   |
                                                   | query rewrite + embedding + adaptive retrieval
                                                   v
                                     grounded answer generation with evidence
                                                   |
                                                   v
                                   Streamlit app (transcript + grounded chat)
```

## Tech Stack
- Language/runtime: Python 3.11
- Dependency/tooling: uv, Typer, Pydantic
- Data/storage: PostgreSQL, pgvector
- Audio/ML pipeline: WhisperX, pyannote.audio
- LLM/RAG: Ollama, local embedding/chat models, retrieval orchestration modules
- Interface: Streamlit
- Quality/tooling: pytest, ruff, black, mypy

## Repository Structure
```text
.
├── src/meeting_pipeline/
│   ├── app/                  # Streamlit app orchestration and UI components
│   ├── audio/                # Parsing, alignment, diarization, attribution, turn/chunk builders
│   ├── db/                   # DB connection, migrations runner, repository, pgvector search
│   ├── embeddings/           # Ollama client and embedding service
│   ├── eval/                 # Transcript/retrieval evaluation logic and metrics
│   ├── rag/                  # Query rewrite, retriever, answer generator, models
│   └── schemas/              # Typed artifact/data contracts
├── scripts/                  # CLI entrypoints for pipeline and operations
├── migrations/               # SQL migrations (schema + retrieval chunk key)
├── tests/                    # Unit and script-level regression tests
├── docs/                     # Setup, architecture, dataset, and evaluation notes
├── data/                     # raw/interim/processed/eval artifacts
├── run_eval.py               # Root wrapper for evaluation CLI
├── .env.example              # Environment variable template
└── pyproject.toml            # Project metadata and dependencies
```

## Why This Project Is Technically Interesting
- Local-first architecture: avoids external hosted APIs for core retrieval and answer generation paths.
- Evidence-grounded answers: retrieval outputs are first-class inputs to answer synthesis, with explicit insufficient-evidence behavior.
- Adaptive retrieval routing: different question intents can trigger different retrieval policies.
- Multi-stage meeting pipeline: parsing, alignment, diarization, attribution, turns, chunking, embedding, retrieval.
- Practical evaluation hooks: transcript and retrieval diagnostics are scriptable and reproducible.

## Setup / Prerequisites
Required software:
- Python 3.11
- uv
- PostgreSQL instance reachable from this machine
- Ollama installed locally

For the full audio pipeline (transcription + diarization), also prepare:
- CUDA-capable environment (recommended)
- GPU dependencies via the `gpu` extra
- Hugging Face token(s) for pyannote model access

Install and initialize:

```bash
# 1) Install dependencies (full local pipeline)
uv sync --group dev --extra data --extra services --extra gpu

# 2) Create database (must match POSTGRES_DB in .env)
createdb meeting_rag

# 3) Apply DB migrations
uv run python scripts/run_migrations.py

# 4) Pull required local models
ollama pull nomic-embed-text-v2-moe
ollama pull qwen3:4b
```

Start Ollama if it is not already running as a background service:

```bash
ollama serve
```

## Environment Configuration
Create a local environment file from `.env.example`:

- Windows PowerShell:

```powershell
copy .env.example .env
```

- Bash:

```bash
cp .env.example .env
```

Key variables most users need to verify first:
- Database: `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`
- Ollama: `OLLAMA_HOST`, `OLLAMA_MODEL`, `OLLAMA_CHAT_MODEL`
- Retrieval chunking: `RETRIEVAL_CHUNK_WINDOW_SECONDS`, `RETRIEVAL_CHUNK_OVERLAP_SECONDS`
- Audio pipeline defaults: `WHISPERX_MODEL`, `WHISPERX_DEVICE`, `WHISPERX_COMPUTE_TYPE`
- Diarization auth: `PYANNOTE_AUTH_TOKEN`, `HUGGINGFACE_TOKEN`

For additional setup and tuning details, see `docs/setup-windows.md`, `docs/architecture.md`, and `docs/evaluation.md`.

## End-to-End Workflow
Example meeting ID: `ES2002a`

1. Migrate DB schema.

```bash
uv run python scripts/run_migrations.py
```

2. Parse AMI word-level XML.

```bash
uv run python scripts/parse_ami_xml.py --meeting-id ES2002a --input-dir data/raw/ami --output-dir data/interim
```

3. Transcribe and align audio.

```bash
uv run python scripts/run_transcription.py --audio-path data/raw/ami/ES2002a.Mix-Headset.wav --meeting-id ES2002a --output-dir data/interim
```

4. Run diarization.

```bash
uv run python scripts/run_diarization.py --audio-path data/raw/ami/ES2002a.Mix-Headset.wav --meeting-id ES2002a --output-dir data/interim
```

5. Build canonical speaker turns.

```bash
uv run python scripts/build_turns.py --meeting-id ES2002a --aligned-path data/interim/ES2002a_aligned.json --diarization-path data/interim/ES2002a_diarization.json --output-dir data/processed
```

6. Ingest embeddings into PostgreSQL/pgvector.

```bash
uv run python scripts/ingest_embeddings.py --meeting-id ES2002a --turns-path data/processed/ES2002a_turns.json --replace-existing --batch-size 16
```

For many meetings:

```bash
uv run python scripts/ingest_many_meetings.py discover --raw-ami-dir data/raw/ami --turns-dir data/processed --discovery-source both
uv run python scripts/ingest_many_meetings.py main --raw-ami-dir data/raw/ami --turns-dir data/processed --skip-existing --batch-size 16 --discovery-source both
```

7. Optional smoke test of retrieval + answer generation.

```bash
uv run python scripts/smoke_rag.py --meeting-id ES2002a --question "What decisions were made?" --top-k 5 --debug --preview-evidence
```

8. Launch the Streamlit app.

```bash
uv run streamlit run src/meeting_pipeline/app/app.py
```

9. Run evaluation.

```bash
uv run python run_eval.py --transcript-reference-path data/interim/ES2002a_ground_truth_words.json --transcript-prediction-path data/processed/ES2002a_turns.json --output-path data/eval/eval_summary.json
```

## Running the App
Launch:

```bash
uv run streamlit run src/meeting_pipeline/app/app.py
```

Typical usage flow:
- Select an ingested meeting from the sidebar.
- Browse transcript chunks and filter by speaker/content.
- Ask a grounded question in chat.
- Inspect evidence and debug metadata to validate answer grounding.

If no meetings appear, run the readiness and ingestion commands in the workflow above.

## Evaluation
The evaluation pipeline supports two practical tracks:
- Transcript diagnostics (reference vs generated artifacts)
- Retrieval benchmark diagnostics (fixture-based benchmark + live or precomputed retrieval)

Transcript diagnostics command:

```bash
uv run python run_eval.py --transcript-reference-path data/interim/ES2002a_ground_truth_words.json --transcript-prediction-path data/processed/ES2002a_turns.json
```

Retrieval diagnostics with live retrieval (requires benchmark fixture):

```bash
uv run python run_eval.py --retrieval-benchmark-path data/eval/retrieval_benchmark.json --retrieval-top-k 5 --live-retrieval --output-path data/eval/retrieval_eval.json
```

Evaluation details and expected fixtures are documented in `docs/evaluation.md`.

## Demo: Recommended Flow
For a clean live demo:

1. Verify local services are running: PostgreSQL and Ollama.
2. Ensure at least one meeting is ingested (for example `ES2002a`).
3. Start Streamlit and open the app.
4. Show transcript browsing first, then ask 2 to 3 grounded questions.
5. Use the smoke script afterward to show reproducible CLI-level validation.

## Limitations
- Local-first single-operator posture; this is not a multi-user production deployment.
- End-to-end quality depends on local model availability, model choice, and artifact preparation quality.
- Retrieval quality is sensitive to chunk window/overlap settings and evidence coverage.
- Confidence tiers are heuristic policy outputs, not learned probabilistic calibration.
- Full audio pipeline requires heavier dependencies and suitable hardware/runtime setup.

## Future Improvements
- Broaden retrieval benchmark fixtures and scenario coverage.
- Add richer end-to-end integration tests for artifact generation pipelines.
- Improve telemetry exports for latency/cache trends over time.
- Expand incremental multi-meeting ingestion observability and reporting.
- Add optional containerized local deployment profiles.

## Contributing / License
Contributions are welcome via issues and pull requests.

Before opening a PR, run local checks:

```bash
uv run ruff check .
uv run black --check .
uv run mypy src
uv run pytest
```

This project is licensed under the MIT License. See the `LICENSE` file for details.
