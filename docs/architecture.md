# Architecture Overview

## Objective

Meeting RAG Local transforms AMI meeting assets into searchable, attributable evidence chunks and grounded Q&A output, while staying fully local-first.

## System Layers

1. Audio and model wrappers (`meeting_pipeline.audio`)
2. Typed schemas and transformation logic (`meeting_pipeline.schemas`)
3. Persistence and indexing (`meeting_pipeline.db`)
4. Retrieval and answer services (`meeting_pipeline.rag`)
5. Evaluation tooling (`meeting_pipeline.eval`, `scripts/run_eval.py`)
6. Demo UX (`meeting_pipeline.app`)

## End-to-End Data Flow

1. Parse AMI XML into normalized ground-truth words.
2. Run transcription + alignment on meeting audio.
3. Run diarization.
4. Build canonical speaker turns from aligned + diarized artifacts.
5. Embed turns and ingest into PostgreSQL + pgvector.
6. Retrieve evidence by semantic similarity (meeting-scoped).
7. Generate grounded answer sections from retrieved evidence only.
8. Evaluate transcript diagnostics and retrieval benchmark metrics.

## 8 GB VRAM Sequencing Guidance

For 8 GB VRAM machines, run heavy stages sequentially and avoid overlapping GPU tasks.

- Recommended order: transcription -> diarization -> release GPU -> ingestion/retrieval/eval.
- Do not run Streamlit, diarization, and transcription concurrently.
- If memory pressure appears, process one meeting at a time and close other GPU consumers.
- Keep batch sizes conservative during embedding ingestion.

## Design Principles

- Strong typing at boundaries and artifact contracts.
- Config-driven runtime behavior through environment settings.
- Clear service interfaces for retrieval and answer generation.
- Deterministic testability for non-GPU paths.
