# Evaluation

## Scope

Evaluation currently supports two practical tracks:

1. Transcript diagnostics
2. Retrieval benchmark diagnostics

The entrypoint is `run_eval.py` (root wrapper for `scripts/run_eval.py`).

## Transcript Diagnostics

Inputs:

- Reference words JSON (`*_ground_truth_words.json`)
- Prediction turns JSON (`*_turns.json`)

Command:

```bash
uv run python run_eval.py --transcript-reference-path data/interim/ES2002a_ground_truth_words.json --transcript-prediction-path data/processed/ES2002a_turns.json
```

Reported metrics include lexical error signals and structural/timing checks.

## Retrieval Benchmark Diagnostics

Inputs:

- Retrieval benchmark fixture JSON
- Either precomputed retrieval predictions JSON, or live retrieval mode

Precomputed predictions mode:

```bash
uv run python run_eval.py --retrieval-benchmark-path data/eval/retrieval_benchmark.json --retrieval-predictions-path data/eval/retrieval_predictions.json --retrieval-top-k 5
```

Live retrieval mode:

```bash
uv run python run_eval.py --retrieval-benchmark-path data/eval/retrieval_benchmark.json --retrieval-top-k 5 --live-retrieval
```

Typical metrics:

- recall@k
- evidence_hit_rate
- empty_retrieval_rate
- chunk/speaker/hint hit rates

## Output Handling

To persist a summary:

```bash
uv run python run_eval.py --transcript-reference-path data/interim/ES2002a_ground_truth_words.json --transcript-prediction-path data/processed/ES2002a_turns.json --output-path data/eval/eval_summary.json
```

## Exclusions and Non-Goals

- Not a full ASR benchmark suite.
- Not a full factuality benchmark for generated answers.
- Not intended to replace human review of evidence quality.
- No claim of domain-general model benchmarking.
