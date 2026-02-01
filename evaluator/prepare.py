from __future__ import annotations

import re
import unicodedata

from evaluator.models import Segment


def merge_consecutive_speakers(segments: list[Segment]) -> list[Segment]:
    """
    Merge consecutive segments with the same speaker.

    Args:
        segments: List of Segment objects

    Returns:
        List of Segment objects with consecutive same-speaker segments merged
    """
    if not segments:
        return []

    merged: list[Segment] = []
    current = segments[0]

    for segment in segments[1:]:
        if segment.locuteur == current.locuteur:
            current = Segment(
                locuteur=current.locuteur,
                text=f"{current.text} {segment.text}",
            )
        else:
            merged.append(current)
            current = segment

    merged.append(current)
    return merged


def prepare(
    segments: list[Segment],
    with_locuteur: bool = True,
    with_paragraphs: bool = True,
    lower_no_punct: bool = True,
) -> str:
    """
    Process segments and produce a continuous text stream.

    Args:
        segments: List of Segment objects from parse_source
        with_locuteur: Include speaker labels (+A:, +B:, etc.)
        with_paragraphs: Add newlines between segments
        lower_no_punct: Remove punctuation and lowercase text

    Returns:
        Formatted text string
    """
    result_parts: list[str] = []

    for segment in segments:
        text = segment.text

        if lower_no_punct:
            text = normalize_text(text)

        if with_locuteur:
            line = f"+{segment.locuteur}: {text}"
        else:
            line = text

        result_parts.append(line)

    separator = "\n" if with_paragraphs else ""
    return separator.join(result_parts)


def normalize_text(text: str) -> str:
    """
    Normalize text by lowercasing and replacing punctuation with spaces.

    Replaces punctuation with spaces for readability, then collapses
    multiple spaces into one.

    Args:
        text: Input text string

    Returns:
        Normalized text (lowercase, punctuation replaced with spaces)
    """
    text = text.lower()

    normalized = unicodedata.normalize("NFD", text)
    text = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    text = unicodedata.normalize("NFC", text)

    text = re.sub(r"[^\w\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def prepare_va(segments: list[Segment]) -> str:
    """VA: with_locuteur=True, with_paragraphs=True, lower_no_punct=False"""
    return prepare(segments, with_locuteur=True, with_paragraphs=True, lower_no_punct=False)


def prepare_vb(segments: list[Segment]) -> str:
    """VB: with_locuteur=False, with_paragraphs=False, lower_no_punct=False"""
    return prepare(segments, with_locuteur=False, with_paragraphs=False, lower_no_punct=False)


def prepare_vc(segments: list[Segment]) -> str:
    """VC: with_locuteur=True, with_paragraphs=True, lower_no_punct=True"""
    return prepare(segments, with_locuteur=True, with_paragraphs=True, lower_no_punct=True)


def prepare_vd(segments: list[Segment]) -> str:
    """VD: with_locuteur=False, with_paragraphs=False, lower_no_punct=True"""
    return prepare(segments, with_locuteur=False, with_paragraphs=False, lower_no_punct=True)


VARIANTS = {
    "VA": prepare_va,
    "VB": prepare_vb,
    "VC": prepare_vc,
    "VD": prepare_vd,
}


def prepare_all_variants(segments: list[Segment]) -> dict[str, str]:
    """
    Generate all four text variants from segments.

    Merges consecutive segments with the same speaker before generating variants.

    Returns:
        Dict mapping variant name (VA, VB, VC, VD) to prepared text
    """
    merged = merge_consecutive_speakers(segments)
    return {name: func(merged) for name, func in VARIANTS.items()}
