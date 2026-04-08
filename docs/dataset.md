# Dataset Notes

## Source Layout

Place AMI assets under `data/raw/ami/`.

Expected files:

- Word XML: `<meeting_id>.<speaker_id>.words.xml`
- Meeting audio (example): `<meeting_id>.Mix-Headset.wav`

Example:

- `data/raw/ami/ES2002a.A.words.xml`
- `data/raw/ami/ES2002a.B.words.xml`
- `data/raw/ami/ES2002a.Mix-Headset.wav`

## Parse Command

```bash
uv run python scripts/parse_ami_xml.py --meeting-id ES2002a --input-dir data/raw/ami --output-dir data/interim
```

## Generated Artifact Map

- Parser output:
  - `data/interim/<meeting_id>_ground_truth_words.json`
- Transcription/alignment output:
  - `data/interim/<meeting_id>_aligned.json`
- Diarization output:
  - `data/interim/<meeting_id>_diarization.json`
- Canonical turns output:
  - `data/processed/<meeting_id>_turns.json`
- Optional evaluation outputs:
  - `data/eval/*.json`

## Artifact Roles

- `ground_truth_words.json`: lexical/timing reference for transcript diagnostics.
- `aligned.json`: ASR-aligned segment output from transcription stage.
- `diarization.json`: speaker/time attribution output.
- `turns.json`: canonical RAG-ready meeting turns used for embedding ingestion.

## Validation Guidance

- Verify meeting IDs match across all generated files before ingestion.
- Keep one meeting per pipeline run while iterating on setup.
- Treat interim files as regenerable artifacts; treat benchmark fixtures as source-of-truth for evaluation.
