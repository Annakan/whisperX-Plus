from __future__ import annotations

import pytest

from evaluator.models import Segment
from evaluator.prepare import (
    normalize_text,
    prepare,
    prepare_all_variants,
    prepare_va,
    prepare_vb,
    prepare_vc,
    prepare_vd,
)


class TestNormalizeText:
    def test_lowercase(self):
        assert normalize_text("HELLO WORLD") == "hello world"

    def test_remove_punctuation(self):
        assert normalize_text("Hello, world!") == "hello world"

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
