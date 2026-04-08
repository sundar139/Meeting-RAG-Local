from pathlib import Path

EXPECTED_ENTRYPOINTS = [
    "run_eval.py",
    "scripts/run_migrations.py",
    "scripts/parse_ami_xml.py",
    "scripts/run_transcription.py",
    "scripts/run_diarization.py",
    "scripts/build_turns.py",
    "scripts/ingest_embeddings.py",
    "scripts/ingest_many_meetings.py",
    "scripts/report_ami_meeting_readiness.py",
    "scripts/smoke_rag.py",
    "scripts/benchmark_rag.py",
    "src/meeting_pipeline/app/app.py",
]


def test_documented_entrypoints_exist() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    missing = [path for path in EXPECTED_ENTRYPOINTS if not (repo_root / path).is_file()]
    assert missing == []
