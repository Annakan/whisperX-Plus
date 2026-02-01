from __future__ import annotations

import pytest

from evaluator.models import Segment
from evaluator.prepare import (
    merge_consecutive_speakers,
    normalize_text,
    prepare,
    prepare_all_variants,
    prepare_va,
    prepare_vb,
    prepare_vc,
    prepare_vd,
)


class TestMergeConsecutiveSpeakers:
    def test_empty_list(self):
        assert merge_consecutive_speakers([]) == []

    def test_single_segment(self):
        segments = [Segment(locuteur="A", text="Hello")]
        result = merge_consecutive_speakers(segments)
        assert len(result) == 1
        assert result[0].text == "Hello"

    def test_different_speakers_not_merged(self):
        segments = [
            Segment(locuteur="A", text="Hello"),
            Segment(locuteur="B", text="World"),
        ]
        result = merge_consecutive_speakers(segments)
        assert len(result) == 2

    def test_same_speaker_merged(self):
        segments = [
            Segment(locuteur="A", text="Hello"),
            Segment(locuteur="A", text="World"),
        ]
        result = merge_consecutive_speakers(segments)
        assert len(result) == 1
        assert result[0].locuteur == "A"
        assert result[0].text == "Hello World"

    def test_multiple_consecutive_merged(self):
        segments = [
            Segment(locuteur="A", text="One"),
            Segment(locuteur="A", text="Two"),
            Segment(locuteur="A", text="Three"),
            Segment(locuteur="B", text="Four"),
        ]
        result = merge_consecutive_speakers(segments)
        assert len(result) == 2
        assert result[0].text == "One Two Three"
        assert result[1].text == "Four"

    def test_alternating_speakers(self):
        segments = [
            Segment(locuteur="A", text="One"),
            Segment(locuteur="B", text="Two"),
            Segment(locuteur="A", text="Three"),
        ]
        result = merge_consecutive_speakers(segments)
        assert len(result) == 3


class TestNormalizeText:
    def test_lowercase(self):
        assert normalize_text("HELLO WORLD") == "hello world"

    def test_replace_punctuation_with_space(self):
        assert normalize_text("Hello, world!") == "hello world"
        assert normalize_text("don't") == "don t"
        assert normalize_text("well...okay") == "well okay"

    def test_remove_accents(self):
        assert normalize_text("présidente") == "presidente"
        assert normalize_text("café") == "cafe"

    def test_preserve_spaces(self):
        assert normalize_text("hello   world") == "hello world"

    def test_complex_french(self):
        result = normalize_text("Ensuite, nous retrouverons notre présidente.")
        assert result == "ensuite nous retrouverons notre presidente"


class TestPrepare:
    @pytest.fixture
    def sample_segments(self) -> list[Segment]:
        return [
            Segment(
                locuteur="A",
                text="Ensuite, nous retrouverons notre présidente.",
            ),
            Segment(
                locuteur="B",
                text="Merci, madame la présidente.",
            ),
        ]

    def test_with_locuteur_and_paragraphs(self, sample_segments):
        result = prepare(
            sample_segments,
            with_locuteur=True,
            with_paragraphs=True,
            lower_no_punct=False,
        )
        lines = result.split("\n")
        assert len(lines) == 2
        assert lines[0].startswith("+A:")
        assert lines[1].startswith("+B:")

    def test_without_locuteur(self, sample_segments):
        result = prepare(
            sample_segments,
            with_locuteur=False,
            with_paragraphs=True,
            lower_no_punct=False,
        )
        assert "+A:" not in result
        assert "+B:" not in result
        assert "présidente" in result

    def test_without_paragraphs(self, sample_segments):
        result = prepare(
            sample_segments,
            with_locuteur=False,
            with_paragraphs=False,
            lower_no_punct=False,
        )
        assert "\n" not in result

    def test_with_normalization(self, sample_segments):
        result = prepare(
            sample_segments,
            with_locuteur=False,
            with_paragraphs=False,
            lower_no_punct=True,
        )
        assert "présidente" not in result
        assert "presidente" in result
        assert result == result.lower()


class TestVariants:
    @pytest.fixture
    def sample_segments(self) -> list[Segment]:
        return [
            Segment(locuteur="A", text="Hello, world!"),
            Segment(locuteur="B", text="Goodbye, world!"),
        ]

    def test_va_with_locuteur_paragraphs_no_normalize(self, sample_segments):
        result = prepare_va(sample_segments)
        assert "+A:" in result
        assert "\n" in result
        assert "Hello, world!" in result

    def test_vb_no_locuteur_no_paragraphs_no_normalize(self, sample_segments):
        result = prepare_vb(sample_segments)
        assert "+A:" not in result
        assert "\n" not in result
        assert "Hello, world!" in result

    def test_vc_with_locuteur_paragraphs_normalized(self, sample_segments):
        result = prepare_vc(sample_segments)
        assert "+A:" in result
        assert "\n" in result
        assert "hello world" in result

    def test_vd_no_locuteur_no_paragraphs_normalized(self, sample_segments):
        result = prepare_vd(sample_segments)
        assert "+A:" not in result
        assert "\n" not in result
        assert "hello world" in result


class TestPrepareAllVariants:
    def test_returns_all_four_variants(self):
        segments = [Segment(locuteur="A", text="Test text.")]
        variants = prepare_all_variants(segments)
        assert set(variants.keys()) == {"VA", "VB", "VC", "VD"}

    def test_variants_differ(self):
        segments = [
            Segment(locuteur="A", text="Hello, World!"),
            Segment(locuteur="B", text="Goodbye!"),
        ]
        variants = prepare_all_variants(segments)
        values = list(variants.values())
        assert len(set(values)) == len(values)
