from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import typer

from meeting_pipeline.schemas.transcript import AlignedTranscript, WordToken

LOGGER = logging.getLogger(__name__)
WORD_FILE_PATTERN = re.compile(r"^(?P<meeting>[^.]+)\.(?P<speaker>[^.]+)\.words\.xml$")

app = typer.Typer(help="Parse AMI word-level XML into normalized ground-truth JSON.")


@dataclass(frozen=True)
class ParseStats:
    files_found: int
    tokens_parsed: int
    tokens_skipped: int


def _parse_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_word_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.split())
    if not normalized:
        return None
    return normalized


def _infer_speaker_id(file_path: Path, meeting_id: str, attributes: dict[str, str]) -> str | None:
    match = WORD_FILE_PATTERN.match(file_path.name)
    if match and match.group("meeting") == meeting_id:
        speaker = match.group("speaker").strip()
        if speaker:
            return speaker

    for key in ("speaker", "speaker_id", "agent", "who"):
        value = attributes.get(key)
        if value is not None and value.strip():
            return value.strip()

    return None


def discover_word_files(meeting_id: str, input_dir: Path) -> list[Path]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory does not exist or is not a directory: {input_dir}"
        )

    files = sorted(path for path in input_dir.glob(f"{meeting_id}.*.words.xml") if path.is_file())
    if not files:
        raise FileNotFoundError(
            f"No word-level XML files found for meeting '{meeting_id}' in {input_dir}"
        )

    return files


def parse_word_file(meeting_id: str, xml_path: Path) -> tuple[list[WordToken], int]:
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as exc:
        LOGGER.warning("Skipping malformed XML file %s: %s", xml_path, exc)
        return [], 0

    words: list[WordToken] = []
    skipped = 0

    for node in tree.getroot().iter():
        start_raw = node.attrib.get("starttime")
        end_raw = node.attrib.get("endtime")

        # AMI files include many structural nodes; only timed nodes can represent words.
        if start_raw is None and end_raw is None:
            continue

        if start_raw is None or end_raw is None:
            skipped += 1
            continue

        start_time = _parse_float(start_raw)
        end_time = _parse_float(end_raw)
        if start_time is None or end_time is None:
            skipped += 1
            continue

        text = _normalize_word_text(node.text)
        if text is None:
            skipped += 1
            continue

        speaker_id = _infer_speaker_id(xml_path, meeting_id, node.attrib)
        if speaker_id is None:
            skipped += 1
            continue

        try:
            words.append(
                WordToken(
                    speaker_id=speaker_id,
                    start_time=start_time,
                    end_time=end_time,
                    text=text,
                )
            )
        except ValueError:
            skipped += 1

    return words, skipped


def parse_ami_words(meeting_id: str, input_dir: Path) -> tuple[AlignedTranscript, ParseStats]:
    normalized_meeting_id = meeting_id.strip()
    if not normalized_meeting_id:
        raise ValueError("meeting_id must be a non-empty string")

    files = discover_word_files(normalized_meeting_id, input_dir)

    merged_words: list[WordToken] = []
    skipped_total = 0

    for word_file in files:
        parsed_words, skipped = parse_word_file(normalized_meeting_id, word_file)
        merged_words.extend(parsed_words)
        skipped_total += skipped

    merged_words.sort(key=lambda item: (item.start_time, item.end_time, item.speaker_id, item.text))

    transcript = AlignedTranscript(meeting_id=normalized_meeting_id, words=merged_words)
    stats = ParseStats(
        files_found=len(files),
        tokens_parsed=len(merged_words),
        tokens_skipped=skipped_total,
    )
    return transcript, stats


def write_aligned_transcript(transcript: AlignedTranscript, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{transcript.meeting_id}_ground_truth_words.json"
    payload = transcript.model_dump(mode="json")
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def parse_and_write_ami_words(
    meeting_id: str,
    input_dir: Path,
    output_dir: Path | None = None,
) -> tuple[Path, ParseStats]:
    transcript, stats = parse_ami_words(meeting_id=meeting_id, input_dir=input_dir)
    target_output_dir = output_dir if output_dir is not None else Path("data/interim")
    output_path = write_aligned_transcript(transcript=transcript, output_dir=target_output_dir)

    LOGGER.info("AMI parse summary for %s", transcript.meeting_id)
    LOGGER.info("  files_found=%d", stats.files_found)
    LOGGER.info("  tokens_parsed=%d", stats.tokens_parsed)
    LOGGER.info("  tokens_skipped=%d", stats.tokens_skipped)
    LOGGER.info("  output=%s", output_path)

    return output_path, stats


@app.command()
def main(
    meeting_id: str = typer.Option(
        ..., "--meeting-id", help="AMI meeting identifier (e.g. ES2002a)."
    ),
    input_dir: Path = typer.Option(
        ...,
        "--input-dir",
        help="Directory containing AMI word-level XML files.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Directory for normalized JSON output. Defaults to data/interim.",
    ),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        output_path, _ = parse_and_write_ami_words(
            meeting_id=meeting_id,
            input_dir=input_dir,
            output_dir=output_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("AMI parse failed: %s", exc)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        LOGGER.exception("Unexpected AMI parse failure: %s", exc)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Wrote {output_path}")


if __name__ == "__main__":
    app()
