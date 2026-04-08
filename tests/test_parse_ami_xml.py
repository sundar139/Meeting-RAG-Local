from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.parse_ami_xml import parse_ami_words, parse_and_write_ami_words


def _write_words_xml(path: Path, body: str) -> None:
    xml = f"<root>{body}</root>"
    path.write_text(xml, encoding="utf-8")


def _write_xml(path: Path, xml: str) -> None:
    path.write_text(xml, encoding="utf-8")


def test_parse_ami_words_merges_multiple_speakers_in_time_order(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    input_dir.mkdir(parents=True)

    _write_words_xml(
        input_dir / "ES2002a.A.words.xml",
        """
    <w starttime="1.20" endtime="1.50"> there </w>
    <w starttime="0.20" endtime="0.40"> hello </w>
    """,
    )
    _write_words_xml(
        input_dir / "ES2002a.B.words.xml",
        """
    <w starttime="0.60" endtime="0.80"> team </w>
    """,
    )

    transcript, stats = parse_ami_words(meeting_id="ES2002a", input_dir=input_dir)

    assert stats.files_found == 2
    assert stats.tokens_parsed == 3
    assert stats.tokens_skipped == 0
    assert [token.speaker_id for token in transcript.words] == ["A", "B", "A"]
    assert [token.text for token in transcript.words] == ["hello", "team", "there"]


def test_parse_ami_words_skips_malformed_tokens(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    input_dir.mkdir(parents=True)

    _write_words_xml(
        input_dir / "ES2002a.A.words.xml",
        """
    <w starttime="0.0" endtime="0.2"> ok </w>
    <w starttime="bad" endtime="0.4"> nope </w>
    <w endtime="0.6"> missing_start </w>
    <w starttime="0.8" endtime="0.7"> reversed </w>
    <w starttime="1.0" endtime="1.1">    </w>
    """,
    )

    transcript, stats = parse_ami_words(meeting_id="ES2002a", input_dir=input_dir)

    assert len(transcript.words) == 1
    assert transcript.words[0].text == "ok"
    assert stats.tokens_skipped == 4


def test_parse_ami_words_handles_ami_nite_style_word_nodes(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    input_dir.mkdir(parents=True)

    _write_xml(
        input_dir / "ES2002a.A.words.xml",
        """<?xml version="1.0" encoding="UTF-8"?>
<nite:root xmlns:nite="http://nite.sourceforge.net/">
    <w nite:id="ES2002a.A.words1" starttime="0.10" endtime="0.30"> Hello </w>
    <w nite:id="ES2002a.A.words2" starttime="0.31" endtime="0.50"> world </w>
    <vocalsound nite:id="ES2002a.A.words3" starttime="0.51" endtime="0.60" />
    <w nite:id="ES2002a.A.words4" starttime="0.61" endtime="0.90"> again </w>
</nite:root>
""",
    )

    transcript, stats = parse_ami_words(meeting_id="ES2002a", input_dir=input_dir)

    assert [word.text for word in transcript.words] == ["Hello", "world", "again"]
    assert all(word.speaker_id == "A" for word in transcript.words)
    assert stats.tokens_parsed == 3
    assert stats.tokens_skipped == 1


def test_parse_ami_words_fails_when_meeting_files_missing(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    input_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        parse_ami_words(meeting_id="ES4040z", input_dir=input_dir)


def test_parse_and_write_ami_words_writes_expected_json_contract(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "interim"
    input_dir.mkdir(parents=True)

    _write_words_xml(
        input_dir / "ES2002a.A.words.xml",
        """
    <w starttime="12.34" endtime="12.56"> hello </w>
    """,
    )

    output_path, _ = parse_and_write_ami_words(
        meeting_id="ES2002a",
        input_dir=input_dir,
        output_dir=output_dir,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert set(payload.keys()) == {"meeting_id", "words"}
    assert set(payload["words"][0].keys()) == {
        "speaker_id",
        "start_time",
        "end_time",
        "text",
    }
    assert payload["meeting_id"] == "ES2002a"
    assert payload["words"][0] == {
        "speaker_id": "A",
        "start_time": 12.34,
        "end_time": 12.56,
        "text": "hello",
    }
