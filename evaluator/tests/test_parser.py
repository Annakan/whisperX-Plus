from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from evaluator.models import Segment
from evaluator.parser import (
    _extract_transcript_section,
    _parse_segments,
    normalize_speakers,
    parse_source,
)


class TestExtractTranscriptSection:
    def test_with_separator(self):
        lines = [
            "Some metadata",
            "More metadata",
            "-----",
            "=====",
            "+A:",
            "Hello world",
        ]
        result = _extract_transcript_section(lines)
        assert result == ["+A:", "Hello world"]

    def test_with_long_separator(self):
        lines = [
            "Metadata",
            "----------",
            "==========",
            "+A:",
            "Content here",
        ]
        result = _extract_transcript_section(lines)
        assert result == ["+A:", "Content here"]

    def test_without_separator(self):
        lines = [
            "+A:",
            "Direct content",
            "+B:",
            "More content",
        ]
        result = _extract_transcript_section(lines)
        assert result == lines

    def test_separator_not_consecutive(self):
        lines = [
            "-----",
            "",
            "=====",
            "+A:",
            "Content",
        ]
        result = _extract_transcript_section(lines)
        assert result == lines


class TestParseSegments:
    def test_simple_segments(self):
        lines = [
            "+A:",
            "Hello world",
            "+B:",
            "Goodbye world",
        ]
        segments = _parse_segments(lines)
        assert len(segments) == 2
        assert segments[0].locuteur == "A"
        assert segments[0].text == "Hello world"
        assert segments[1].locuteur == "B"
        assert segments[1].text == "Goodbye world"

    def test_with_timestamps(self):
        lines = [
            "[00:01] | 1234 ms",
            "+A:",
            "First segment",
            "[00:05] | 5678 ms",
            "+B:",
            "Second segment",
        ]
        segments = _parse_segments(lines)
        assert len(segments) == 2
        assert segments[0].text == "First segment"
        assert segments[1].text == "Second segment"

    def test_empty_lines_ignored(self):
        lines = [
            "",
            "+A:",
            "Content",
            "",
            "",
            "+B:",
            "More content",
        ]
        segments = _parse_segments(lines)
        assert len(segments) == 2

    def test_three_speakers(self):
        lines = [
            "+A:",
            "Speaker A text",
            "+B:",
            "Speaker B text",
            "+C:",
            "Speaker C text",
        ]
        segments = _parse_segments(lines)
        assert len(segments) == 3
        assert segments[2].locuteur == "C"


class TestParseSource:
    def test_full_file_with_header(self):
        content = """Some metadata here
Another line
-----
=====
[00:01] | 100 ms
+A:
Ensuite, nous retrouverons notre présidente.
+B:
Merci, madame la présidente.
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            segments = parse_source(path)
            assert len(segments) == 2
            assert segments[0].locuteur == "A"
            assert "présidente" in segments[0].text
            assert segments[1].locuteur == "B"
        finally:
            path.unlink()

    def test_file_without_header(self):
        content = """+A:
Direct content here
+B:
More content
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            segments = parse_source(path)
            assert len(segments) == 2
        finally:
            path.unlink()


class TestNormalizeSpeakers:
    def test_simple_normalization(self):
        segments = [
            Segment(locuteur="SPEAKER_00", text="Hello"),
            Segment(locuteur="SPEAKER_01", text="Hi"),
            Segment(locuteur="SPEAKER_00", text="How are you"),
        ]
        normalized = normalize_speakers(segments)
        assert normalized[0].locuteur == "A"
        assert normalized[1].locuteur == "B"
        assert normalized[2].locuteur == "A"

    def test_order_of_appearance(self):
        segments = [
            Segment(locuteur="SPEAKER_02", text="First"),
            Segment(locuteur="SPEAKER_00", text="Second"),
            Segment(locuteur="SPEAKER_02", text="Third"),
            Segment(locuteur="SPEAKER_01", text="Fourth"),
        ]
        normalized = normalize_speakers(segments)
        assert normalized[0].locuteur == "A"
        assert normalized[1].locuteur == "B"
        assert normalized[2].locuteur == "A"
        assert normalized[3].locuteur == "C"

    def test_empty_list(self):
        segments: list[Segment] = []
        normalized = normalize_speakers(segments)
        assert normalized == []

    def test_single_speaker(self):
        segments = [
            Segment(locuteur="SPEAKER_00", text="Only one"),
            Segment(locuteur="SPEAKER_00", text="Still one"),
        ]
        normalized = normalize_speakers(segments)
        assert all(s.locuteur == "A" for s in normalized)
