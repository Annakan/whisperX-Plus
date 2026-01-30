from __future__ import annotations

import re
from pathlib import Path

from evaluator.models import Segment


def parse_source(path: Path) -> list[Segment]:
    """
    Parse a reference transcript file and extract segments.

    The file can have two formats:
    - Format A: Metadata header followed by ----- and ===== separator, then transcript
    - Format B: No separator, entire file is transcript content

    Segments are in the format:
        [optional timestamp line]
        +A:
        Speaker A's text here

    Args:
        path: Path to the reference transcript file

    Returns:
        List of Segment objects with locuteur and text
    """
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()

    transcript_lines = _extract_transcript_section(lines)
    segments = _parse_segments(transcript_lines)

    return segments


def _extract_transcript_section(lines: list[str]) -> list[str]:
    """
    Extract the transcript section from the file.

    Looks for pattern: line of 5+ dashes immediately followed by line of 5+ equals.
    If found, returns content after the equals line.
    If not found, returns all lines (entire file is transcript).
    """
    dash_pattern = re.compile(r"^-{5,}\s*$")
    equals_pattern = re.compile(r"^={5,}\s*$")

    for i in range(len(lines) - 1):
        if dash_pattern.match(lines[i]) and equals_pattern.match(lines[i + 1]):
            return lines[i + 2 :]

    return lines


def _parse_segments(lines: list[str]) -> list[Segment]:
    """
    Parse transcript lines into segments.

    Expected format:
        [optional timestamp] [<chars>] | (digits:) (digit+ ms)
        +A:
        Speaker text here

    Timestamp lines are skipped.
    """
    segments: list[Segment] = []
    locuteur_pattern = re.compile(r"^\+([A-Z]):?\s*$")
    timestamp_pattern = re.compile(r"^\[.*\]\s*\|.*\d+.*ms", re.IGNORECASE)

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        if timestamp_pattern.match(line):
            i += 1
            continue

        match = locuteur_pattern.match(line)
        if match:
            locuteur = match.group(1)
            i += 1
            if i < len(lines):
                text = lines[i].strip()
                if text and not locuteur_pattern.match(text):
                    segments.append(Segment(locuteur=locuteur, text=text))
                    i += 1
                else:
                    segments.append(Segment(locuteur=locuteur, text=""))
        else:
            i += 1

    return segments


def parse_whisperx_json(path: Path) -> list[Segment]:
    """
    Parse a WhisperX JSON output file and extract segments.

    WhisperX JSON format has segments with 'text' and optionally 'speaker' fields.

    Args:
        path: Path to the WhisperX JSON output file

    Returns:
        List of Segment objects with normalized speaker labels
    """
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    raw_segments: list[Segment] = []

    for seg in data.get("segments", []):
        text = seg.get("text", "").strip()
        speaker = seg.get("speaker", "UNKNOWN")
        if text:
            raw_segments.append(Segment(locuteur=speaker, text=text))

    return normalize_speakers(raw_segments)


def normalize_speakers(segments: list[Segment]) -> list[Segment]:
    """
    Replace SPEAKER_XX labels with A, B, C... based on order of first appearance.

    Args:
        segments: List of segments with original speaker labels

    Returns:
        List of segments with normalized speaker labels (A, B, C, ...)
    """
    speaker_map: dict[str, str] = {}
    next_letter = ord("A")

    normalized: list[Segment] = []
    for segment in segments:
        if segment.locuteur not in speaker_map:
            speaker_map[segment.locuteur] = chr(next_letter)
            next_letter += 1
        normalized.append(
            Segment(locuteur=speaker_map[segment.locuteur], text=segment.text)
        )

    return normalized
